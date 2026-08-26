package ipc

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/tamnd/kuma/dtype"
)

// The metadata blob is a count of pairs followed by the pairs, each one a
// length and then that many bytes, four times over. Every number is a 32 bit
// signed integer in the byte order of the machine, because the two libraries
// exchanging it are in the same process, and the strings are not terminated.
const (
	metadataHeader = 4 // the pair count
	metadataLength = 4 // one key or value length
)

// EncodeMetadata packs metadata into the blob the C data interface expects in
// the metadata member of a schema.
//
// Empty metadata encodes as nil, not as a count of zero. The C data interface
// says the member is a null pointer when there is nothing to say, and a blob
// holding only a zero would be a pointer the consumer has to read to learn
// nothing.
//
// The keys and values are bytes rather than text. Nothing checks that they are
// valid UTF-8, since Arrow does not require it and a producer that puts a
// serialized message in a value is doing something the format allows.
func EncodeMetadata(m dtype.Metadata) ([]byte, error) {
	if len(m) == 0 {
		return nil, nil
	}
	if len(m) > math.MaxInt32 {
		return nil, fmt.Errorf("ipc: %w: %d pairs is more than the count can hold", ErrMetadata, len(m))
	}

	size := metadataHeader
	for _, kv := range m {
		if len(kv.Key) > math.MaxInt32 || len(kv.Value) > math.MaxInt32 {
			return nil, fmt.Errorf("ipc: %w: the pair %q is too large to encode", ErrMetadata, kv.Key)
		}
		size += 2*metadataLength + len(kv.Key) + len(kv.Value)
	}

	b := make([]byte, 0, size)
	b = appendInt32(b, int32(len(m)))
	for _, kv := range m {
		b = appendInt32(b, int32(len(kv.Key)))
		b = append(b, kv.Key...)
		b = appendInt32(b, int32(len(kv.Value)))
		b = append(b, kv.Value...)
	}
	return b, nil
}

// DecodeMetadata reads the blob EncodeMetadata writes.
//
// An empty or nil blob is no metadata and not an error, which is what a null
// metadata member arrives as. Anything else that does not describe a whole set
// of pairs is an error rather than as much as could be read, since a truncated
// blob means the two sides disagree about the layout and the rest of the schema
// is not to be trusted either.
//
// The keys and values are copied. The blob usually points into memory the
// producer owns and will free, and metadata that turns to nonsense after a
// release callback runs would be a bug nobody could reproduce.
func DecodeMetadata(b []byte) (dtype.Metadata, error) {
	if len(b) == 0 {
		return nil, nil
	}
	if len(b) < metadataHeader {
		return nil, fmt.Errorf("ipc: %w: %d bytes is too few to hold the pair count", ErrMetadata, len(b))
	}
	n := int32(binary.NativeEndian.Uint32(b))
	if n < 0 {
		return nil, fmt.Errorf("ipc: %w: negative pair count %d", ErrMetadata, n)
	}
	rest := b[metadataHeader:]

	// A pair is at least eight bytes, so a count that could not fit in what is
	// left is wrong. This is checked before allocating rather than after, so
	// that a blob claiming two billion pairs does not ask for the memory first.
	if int64(n)*2*metadataLength > int64(len(rest)) {
		return nil, fmt.Errorf("ipc: %w: %d pairs do not fit in %d bytes", ErrMetadata, n, len(rest))
	}

	m := make(dtype.Metadata, 0, n)
	for i := int32(0); i < n; i++ {
		var kv dtype.KeyValue
		var err error
		if kv.Key, rest, err = takeString(rest); err != nil {
			return nil, err
		}
		if kv.Value, rest, err = takeString(rest); err != nil {
			return nil, err
		}
		m = append(m, kv)
	}
	if len(rest) != 0 {
		return nil, fmt.Errorf("ipc: %w: %d bytes left after %d pairs", ErrMetadata, len(rest), n)
	}
	return m, nil
}

// takeString reads one length prefixed string and returns it with what is left.
func takeString(b []byte) (string, []byte, error) {
	if len(b) < metadataLength {
		return "", nil, fmt.Errorf("ipc: %w: %d bytes is too few to hold a length", ErrMetadata, len(b))
	}
	n := int32(binary.NativeEndian.Uint32(b))
	b = b[metadataLength:]
	if n < 0 || int64(n) > int64(len(b)) {
		return "", nil, fmt.Errorf("ipc: %w: length %d with %d bytes left", ErrMetadata, n, len(b))
	}
	return string(b[:n]), b[n:], nil
}

func appendInt32(b []byte, n int32) []byte {
	return binary.NativeEndian.AppendUint32(b, uint32(n))
}
