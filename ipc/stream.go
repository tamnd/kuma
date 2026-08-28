package ipc

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"iter"

	"github.com/tamnd/kuma/dtype"
)

// The Arrow IPC stream format, which is what the messages are for.
//
// A stream is a schema message, then a record batch message for every batch,
// then a length of zero to say there are no more. That is the whole format.
// Everything hard about it is in the messages themselves, which is why this
// file is short and batch.go is not.
//
// The reader hands out arrays that point into the bytes it read, the same way
// DecodeBatch does, so every batch gets a buffer of its own rather than one
// buffer being reused. A reader that recycled its buffer would hand out the
// same memory twice and quietly change the batch somebody was still holding.
//
// Nothing here closes what it was given. A stream is a thing to write into a
// file or a socket, and whoever opened that is the one who knows when it should
// be closed.

// streamEnd is the end of stream marker, which is a message of no bytes: the
// continuation, then a length of zero.
var streamEnd = []byte{0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0}

// maxHint is as much room as a reader takes before it has seen the bytes to put
// in it. A real message is nowhere near this and a message that claims more than
// this is either enormous or lying, and either way the memory should follow the
// bytes rather than the claim.
const maxHint = 1 << 20

// Writer writes the Arrow IPC stream format.
//
// The schema goes out when the writer is made and every batch has to match it.
// Close writes the end of stream marker, and a stream without one is a stream
// that was cut off, so it is worth deferring.
//
// An error from the underlying writer sticks: the stream is broken at that
// point and nothing after it would be readable. An error about a batch does
// not, since the stream is untouched and a caller can fix the batch and try
// again.
type Writer struct {
	dst    io.Writer
	schema dtype.Schema
	err    error
	closed bool
}

// NewWriter starts a stream over w and writes the schema, which is the first
// message of one. The error is either a schema Arrow cannot name or whatever w
// said about the write.
func NewWriter(w io.Writer, s dtype.Schema) (*Writer, error) {
	msg, err := EncodeSchema(s)
	if err != nil {
		return nil, err
	}
	if _, err := w.Write(msg); err != nil {
		return nil, err
	}
	return &Writer{dst: w, schema: s}, nil
}

// Schema is the schema every batch in the stream belongs to.
func (w *Writer) Schema() dtype.Schema { return w.schema }

// Write appends one record batch. The columns have to be the types the schema
// says they are and there have to be as many of them as it has fields.
func (w *Writer) Write(b Batch) error {
	if w.err != nil {
		return w.err
	}
	if w.closed {
		return fmt.Errorf("ipc: %w", ErrClosed)
	}

	msg, err := EncodeBatch(w.schema, b)
	if err != nil {
		return err
	}
	if _, err := w.dst.Write(msg); err != nil {
		w.err = err
		return err
	}
	return nil
}

// Close writes the end of stream marker. It does not close the writer
// underneath, which belongs to whoever opened it.
//
// Closing twice is not an error, so a deferred Close after an explicit one
// reports what the explicit one did rather than something new.
func (w *Writer) Close() error {
	if w.closed {
		return w.err
	}
	w.closed = true
	if w.err != nil {
		return w.err
	}
	if _, err := w.dst.Write(streamEnd); err != nil {
		w.err = err
		return err
	}
	return nil
}

// Reader reads the Arrow IPC stream format.
//
// The schema is read when the reader is made, so a stream whose first message
// is not one is refused before any values are looked at. After that Next reads
// a batch at a time until the stream ends or something in it does not hold
// together.
//
// The arrays in a batch point into the bytes that batch was read from, which is
// a buffer of its own rather than one the reader keeps reusing. Holding on to a
// batch after reading the next one is fine and costs the memory of the batch
// being held.
type Reader struct {
	src    io.Reader
	schema dtype.Schema
	batch  Batch
	err    error
	done   bool
}

// NewReader reads the schema off the front of a stream and returns a reader for
// the batches after it.
func NewReader(r io.Reader) (*Reader, error) {
	msg, _, err := readMessage(r)
	if err != nil {
		return nil, err
	}
	if msg == nil {
		return nil, fmt.Errorf("ipc: %w: a stream that ends where its schema should be", ErrMessage)
	}
	s, err := DecodeSchema(msg)
	if err != nil {
		return nil, err
	}
	return &Reader{src: r, schema: s}, nil
}

// Schema is the schema every batch in the stream belongs to.
func (r *Reader) Schema() dtype.Schema { return r.schema }

// Next reads the next batch and reports whether there is one. It returns false
// at the end of the stream and at the first error, which Err then holds.
func (r *Reader) Next() bool {
	if r.done {
		return false
	}

	msg, kind, err := readMessage(r.src)
	if err != nil {
		r.done, r.err = true, err
		return false
	}
	if msg == nil {
		r.done = true
		return false
	}
	if kind == fbHeaderDictionaryBatch {
		r.done = true
		r.err = fmt.Errorf("ipc: %w: the stream has a dictionary batch in it, which this cannot read yet",
			ErrUnsupported)
		return false
	}

	// The rest after a batch is the next message, which the next call reads for
	// itself. Everything this reader hands back came out of one message and one
	// body, so there is nothing left over to keep.
	b, _, err := DecodeBatch(r.schema, msg)
	if err != nil {
		r.done, r.err = true, err
		return false
	}
	r.batch = b
	return true
}

// Batch is the batch the last call to Next read.
func (r *Reader) Batch() Batch { return r.batch }

// Err is the error that stopped the reader, or nil if the stream simply ended.
func (r *Reader) Err() error { return r.err }

// All is the batches of the stream, for a range loop. It stops at the end of
// the stream or at the first error, which is then in Err, so a loop over this
// still has to check Err afterwards.
func (r *Reader) All() iter.Seq[Batch] {
	return func(yield func(Batch) bool) {
		for r.Next() {
			if !yield(r.batch) {
				return
			}
		}
	}
}

// readMessage reads one encapsulated message and the body after it, and returns
// the pair as the single slice the decoders take, along with the kind of message
// it turned out to be.
//
// A stream that ends where a message would start comes back as no bytes and no
// error, which is the end of it. That covers both ways a stream ends: the
// marker, and a writer that stopped without one, since the marker was optional
// for long enough that files without it are still around.
//
// The bytes are read as they arrive rather than into a buffer sized by what the
// message claims. A message saying it carries four gigabytes gets four
// gigabytes of allocation out of a reader that believes it, and out of one that
// does not it gets an unexpected end of file after however much was really
// there.
func readMessage(r io.Reader) ([]byte, uint8, error) {
	meta, err := readPrefix(r)
	if err != nil || meta == 0 {
		return nil, fbHeaderNone, err
	}
	return readRest(r, meta)
}

// readPrefix reads the length in front of a message, in whichever of the two
// framings it arrives in. A length of zero and the end of the file where a
// prefix would start are both the end of the stream, and both come back as zero
// and no error.
func readPrefix(r io.Reader) (int32, error) {
	var prefix [fbPrefix]byte
	n, err := io.ReadFull(r, prefix[:4])
	if n == 0 && (errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF)) {
		return 0, nil
	}
	if err != nil {
		return 0, fmt.Errorf("ipc: %w: reading a message prefix: %w", ErrMessage, unexpected(err))
	}

	// The old format with no continuation is still readable, so the first four
	// bytes are either the marker or the length itself.
	size := prefix[:4]
	if binary.LittleEndian.Uint32(size) == fbContinuation {
		if _, err := io.ReadFull(r, prefix[4:]); err != nil {
			return 0, fmt.Errorf("ipc: %w: reading a message length: %w", ErrMessage, unexpected(err))
		}
		size = prefix[4:]
	}

	meta := int32(binary.LittleEndian.Uint32(size))
	if meta < 0 {
		return 0, fmt.Errorf("ipc: %w: a message of %d bytes", ErrMessage, meta)
	}
	return meta, nil
}

// readRest reads the metadata of a message and the body after it, and writes the
// framing back in front of the pair so that everything downstream sees one
// framing whichever one arrived.
//
// The room is taken in front only up to a point. A length is four bytes of
// somebody else's file and can say two gigabytes, and a reader that made room
// for that before reading a byte would be one message away from being out of
// memory. Past the cap the buffer grows as the bytes turn up, which is what it
// would do anyway.
func readRest(r io.Reader, meta int32) ([]byte, uint8, error) {
	var size [4]byte
	binary.LittleEndian.PutUint32(size[:], uint32(meta))

	var out bytes.Buffer
	out.Grow(fbPrefix + min(int(meta), maxHint))
	out.Write(streamEnd[:4])
	out.Write(size[:])
	if err := copyN(&out, r, int64(meta), "the metadata of a message"); err != nil {
		return nil, fbHeaderNone, err
	}

	kind, body, err := messageHeader(out.Bytes()[fbPrefix:])
	if err != nil {
		return nil, fbHeaderNone, err
	}
	if err := copyN(&out, r, body, "the body of a message"); err != nil {
		return nil, fbHeaderNone, err
	}
	return out.Bytes(), kind, nil
}

// copyN reads n bytes into dst, and says which part of a message came up short
// when there were not that many.
func copyN(dst *bytes.Buffer, src io.Reader, n int64, what string) error {
	got, err := io.CopyN(dst, src, n)
	if err != nil {
		return fmt.Errorf("ipc: %w: %s says it is %d bytes and %d arrived: %w",
			ErrMessage, what, n, got, unexpected(err))
	}
	return nil
}

// unexpected turns the end of the file in the middle of something into the
// error that says so, since a message that stops half way through is a
// truncated file rather than the end of a stream.
func unexpected(err error) error {
	if errors.Is(err, io.EOF) {
		return io.ErrUnexpectedEOF
	}
	return err
}
