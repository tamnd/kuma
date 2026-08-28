package parquet

import (
	"encoding/binary"
	"fmt"
	"io"
	"strings"
)

// The filter that answers for the values bounds cannot.
//
// Bounds skip a row group by its range, which works when the column is sorted
// or clustered and does nothing when it is not. An identifier is the case: a
// file of a million orders scattered across a hundred row groups holds one
// customer's identifier in one of them, and every group covers a range with
// that identifier somewhere in the middle of it, so the bounds keep every group
// and the scan reads the lot.
//
// A bloom filter is what the format offers for that. The writer hashes every
// value it wrote and sets eight bits per value in a bitset it puts at the end
// of the file. A reader hashes the value it is looking for, looks at the same
// eight bits, and skips the chunk when any of them is clear. That answer is
// certain: a value that was written set those bits, so a clear one means the
// value was never there. The other answer is not certain, because other values
// set bits too, and a filter that keeps a chunk holding nothing wanted has cost
// a read and not an answer.
//
// The format names one algorithm and one hash and no way to negotiate either.
// The bitset is split into blocks of thirty two bytes, a value picks its block
// from the top half of its hash and eight bits from the bottom half, and the
// hash is XXH64 of the value as a page would write it. Nothing here is
// configurable, and a file that says it used something else is refused rather
// than read the only way this reader knows.

// The shape of a split block bloom filter, which is what the format calls the
// one algorithm it defines.
const (
	// bloomBlock is the bytes in a block, which is eight words of four. A value
	// touches one block and no other, which is what makes a lookup one cache
	// line rather than eight scattered bits.
	bloomBlock = 32

	// bloomWords is the words in a block, and so the bits a value sets.
	bloomWords = 8

	// bloomHeaderMax is how much of the file is read to find a header that has
	// no length in front of it. The header is fifteen bytes in every file
	// anybody has written, being a number and three empty structures, so this
	// is room for a writer that adds to it.
	bloomHeaderMax = 128
)

// The eight odd constants the format multiplies a hash by, one per word of a
// block. They are what turns one hash into eight bits that are not the same bit
// eight times.
var bloomSalt = [bloomWords]uint32{
	0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d,
	0x705495c7, 0x2df1424b, 0x9efc4947, 0x5c6bfb31,
}

// BloomFilter is the filter a writer wrote for one column chunk.
//
// It says no for certain and yes with a chance of being wrong, so a scan uses
// it to skip and never to answer: a chunk the filter kept still has to be read
// and looked at. How often it is wrong is what the writer chose when it wrote
// the thing, and nothing in the file says what that was.
type BloomFilter struct {
	bits []byte
}

// Bytes is the size of the bitset, which is the only thing a filter says about
// how much it can be trusted. A writer sizes it for the values it expected and
// the error rate it wanted, so a bigger one over the same chunk is wrong less
// often, and neither number is written down anywhere.
func (f *BloomFilter) Bytes() int {
	if f == nil {
		return 0
	}
	return len(f.bits)
}

// Has says whether the chunk may hold the value.
//
// False is certain and true is not. A value that was written set the bits this
// looks at, so a clear bit means the chunk never held it and the whole chunk
// can be skipped. A set one means the value is there or something else set the
// same bits, which is what a scan reads the chunk to find out.
//
// The value is one value written the way a page writes it, which is the same
// bytes a bound in Statistics holds: a number little endian in its four or
// eight or twelve bytes, and a byte array as itself with no length in front of
// it. A caller with a Go value turns it into those bytes the way the column's
// type says, since the filter was built on what the file holds rather than on
// what a reader would rather have.
//
// A filter that is nil rules nothing out and says yes to everything. That is
// what a chunk without a filter should do, and it is what ReadBloomFilter hands
// back for one, so a scan can ask without looking first.
func (f *BloomFilter) Has(value []byte) bool { return f.has(xxh64(value)) }

// HasString is Has for a value already in hand as a string, which is what a
// filter on a column of names carries. It is the same lookup and copies
// nothing.
func (f *BloomFilter) HasString(value string) bool { return f.has(xxh64(value)) }

// has looks the eight bits of one hash up.
//
// The block comes from the top half of the hash and the bits from the bottom,
// which is why the hash has to be the one the writer used: a different hash of
// the same value looks in a different block and finds it empty, and answering
// no to a value the chunk holds is the one mistake this must never make.
func (f *BloomFilter) has(h uint64) bool {
	if f == nil || len(f.bits) == 0 {
		return true
	}

	// The block is the top half of the hash scaled to however many blocks there
	// are, which is a multiply and a shift where a modulo would divide.
	blocks := uint64(len(f.bits) / bloomBlock)
	at := int((h>>32)*blocks>>32) * bloomBlock

	low := uint32(h)
	for i, salt := range bloomSalt {
		word := binary.LittleEndian.Uint32(f.bits[at+i*4:])
		if word&(1<<((low*salt)>>27)) == 0 {
			return false
		}
	}
	return true
}

// ReadBloomFilter reads the filter a writer wrote for one column chunk.
//
// The size is the size of the file, the same one ReadMetadata was given, since
// where the filter is and how long it is are numbers out of a footer.
//
// It comes back nil when the chunk has no filter, which is most chunks: a
// writer writes them for the columns it is told to and no writer is told to by
// default, the bitset costing bytes in the file for a saving that only some
// scans take. That is not an error, and a caller that gets nil is left with the
// bounds.
//
// A filter written with an algorithm, a hash or a compression this package does
// not know is refused rather than read anyway. The format has one of each so
// far, so this is a file from a future nobody has written yet.
func ReadBloomFilter(r io.ReaderAt, size int64, c *ColumnChunk) (*BloomFilter, error) {
	at := c.Meta.BloomFilterOffset
	if at <= 0 {
		return nil, nil
	}
	name := strings.Join(c.Meta.Path, ".")

	// The length covers the header and the bitset together and is what a modern
	// writer puts in the footer. A writer that left it out is read by taking
	// enough for the header and going back for the rest.
	span := int64(c.Meta.BloomFilterLength)
	if span <= 0 {
		span = min(bloomHeaderMax, size-at)
	}
	buf, err := chunkBytes(r, size, at, span, "bloom filter", name)
	if err != nil {
		return nil, err
	}

	var h bloomHeader
	head := &reader{buf: buf}
	if err = h.read(head); err != nil {
		return nil, fmt.Errorf("parquet: the bloom filter of %s: %w", name, err)
	}
	if err = h.usable(name); err != nil {
		return nil, err
	}

	n, used := int64(h.numBytes), int64(head.pos)
	if n <= 0 || n%bloomBlock != 0 {
		return nil, fmt.Errorf("parquet: %w: the bloom filter of %s is %d bytes, which is not whole blocks of %d",
			ErrFormat, name, n, bloomBlock)
	}
	if c.Meta.BloomFilterLength > 0 && used+n > span {
		return nil, fmt.Errorf("parquet: %w: the bloom filter of %s is %d bytes at the end of %d",
			ErrFormat, name, n, span)
	}

	// The bitset is usually in what has been read already, the length in the
	// footer covering both parts. A writer that wrote no length leaves it to be
	// read on its own.
	if used+n <= int64(len(buf)) {
		return &BloomFilter{bits: buf[used : used+n]}, nil
	}
	bits, err := chunkBytes(r, size, at+used, n, "bloom filter", name)
	if err != nil {
		return nil, err
	}
	return &BloomFilter{bits: bits}, nil
}

// BloomFilter returns the filter the writer wrote for one projected column of
// one row group.
//
// The column is its place in the projection rather than in the file, the same
// as Bounds and PageBounds. It comes back nil when that chunk has no filter,
// which is what a scan falls back to the bounds on.
func (r *FileReader) BloomFilter(group, column int) (*BloomFilter, error) {
	g, err := r.group(group)
	if err != nil {
		return nil, err
	}
	if column < 0 || column >= len(r.take) {
		return nil, fmt.Errorf("parquet: column %d of a projection of %d", column, len(r.take))
	}

	ch, _, err := r.chunkOf(g, group, column)
	if err != nil {
		return nil, err
	}
	return ReadBloomFilter(r.src, r.size, ch)
}

// bloomHeader is what sits in front of the bitset.
//
// Three of its four fields are unions of one member each, which is how thrift
// writes a choice that has never had anything to choose from. They are read as
// what they are, a struct holding a struct, and what matters about each is
// whether the member written is the one this reader knows.
type bloomHeader struct {
	numBytes     int32
	block        bool
	xxhash       bool
	uncompressed bool
}

// read fills in the header from the file.
func (h *bloomHeader) read(r *reader) error {
	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			h.numBytes, err = r.int32(t)
		case 2:
			h.block, err = member(r, t, 1)
		case 3:
			h.xxhash, err = member(r, t, 1)
		case 4:
			h.uncompressed, err = member(r, t, 1)
		default:
			err = r.skip(t)
		}
		return err
	})
}

// usable says whether the filter is one this package can look a value up in.
func (h *bloomHeader) usable(name string) error {
	for _, c := range []struct {
		ok   bool
		what string
	}{
		{h.block, "an algorithm"},
		{h.xxhash, "a hash"},
		{h.uncompressed, "a compression"},
	} {
		if !c.ok {
			return fmt.Errorf("parquet: %w: the bloom filter of %s uses %s this reader does not know",
				ErrUnsupported, name, c.what)
		}
	}
	return nil
}

// member reads a thrift union and says whether the member written is the one
// wanted.
//
// A union is a struct with one field set, and which field it is is the whole of
// what it says. Reading past a member this package does not know is what lets
// the header be read at all, since refusing is the caller's to do once it knows
// which of the three it was.
func member(r *reader, t thriftType, want int16) (bool, error) {
	if t != thriftStruct {
		return false, fmt.Errorf("parquet: %w: a union written as a %s", ErrFormat, t)
	}

	found := false
	err := r.fields(func(id int16, t thriftType) error {
		if id == want {
			found = true
		}
		return r.skip(t)
	})
	return found, err
}
