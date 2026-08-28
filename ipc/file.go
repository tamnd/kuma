package ipc

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/tamnd/kuma/dtype"
)

// The Arrow IPC file format, which is the stream format made seekable.
//
// A file is the magic ARROW1 and two bytes of padding, then exactly the bytes
// of a stream, then a footer, the length of the footer as a little endian
// int32, and the magic once more. The footer carries the schema a second time
// and a block per record batch saying where in the file that batch starts and
// how long it is, so a reader can open a file of a thousand batches and read the
// last one without touching the other nine hundred and ninety nine.
//
// The middle of a file being a stream is the point of the design rather than a
// coincidence. The same message goes down either way, so a writer of one is a
// writer of the other with a header and an index around it, and a reader that
// only knows the stream format reads a file by skipping the first eight bytes.
//
// The padding after the magic is what keeps the messages aligned. Every buffer
// in a body sits on eight bytes from the start of the file, which is only true
// if the first message starts on eight, and ARROW1 is six.
//
// Nothing here closes what it was given, the same as the stream.

const (
	// What a file starts and ends with.
	fileMagic = "ARROW1"

	// The magic and the padding after it, which is where the stream inside a
	// file begins.
	filePad = fileMagic + "\x00\x00"

	// The footer length and the magic after it, which is what a reader looking
	// for the footer reads first.
	fileTail = 4 + len(fileMagic)
)

// The field numbers of the Footer table, in declaration order.
const (
	fbFooterVersion = iota
	fbFooterSchema
	fbFooterDictionaries
	fbFooterBatches
	fbFooterMetadata
)

// fbBlockSize is the size of a Block, which is how a footer says where a message
// is: an offset, the length of the metadata and the length of the body. It is
// twenty bytes of numbers in twenty four bytes of struct, since the int in the
// middle is followed by four bytes of padding that the int64 after it needs.
const fbBlockSize = 24

// block is one of those. The metadata length counts the framing in front of the
// message as well as the metadata itself, so a message runs from offset for
// meta plus body bytes and the body starts at offset plus meta.
type block struct {
	offset int64
	meta   int32
	body   int64
}

// size is the whole message, which is what reading one block reads.
func (b block) size() int64 { return int64(b.meta) + b.body }

// inside says whether a block really is in a file with limit bytes of messages
// in it.
//
// Every number in a block was written by somebody else, so a reader that took
// one at its word would read from wherever it pointed, or ask for a slice of the
// size it claimed rather than the size the file has.
func (b block) inside(limit int64) bool {
	if b.offset < int64(len(filePad)) || b.offset > limit || b.meta < 0 || b.body < 0 {
		return false
	}
	// The subtraction is what keeps this from overflowing, since a length in a
	// file is free to be the largest number there is.
	room := limit - b.offset
	if int64(b.meta) > room || b.body > room-int64(b.meta) {
		return false
	}
	// The message is read into memory in one piece, so a block that does not fit
	// in an int is one this machine cannot read even if the file really is that
	// large.
	return int64(int(b.size())) == b.size()
}

// blockOf is where a message that is about to be written lands: the offset it
// starts at and the split between its metadata and its body. The message came
// out of this package, so the length in front of it is the padded one the
// framing wrote and the body is everything after that.
func blockOf(at int64, msg []byte) (block, error) {
	meta := int64(fbPrefix) + int64(binary.LittleEndian.Uint32(msg[4:]))
	if meta > math.MaxInt32 {
		return block{}, fmt.Errorf("ipc: %w: a message header of %d bytes, which does not fit in the int32 a block holds it in",
			ErrMessage, meta)
	}
	return block{offset: at, meta: int32(meta), body: int64(len(msg)) - meta}, nil
}

// FileWriter writes the Arrow IPC file format.
//
// The schema goes out when the writer is made and every batch has to match it,
// the same as a stream. Close writes the footer, which is the index of
// everything written before it, and a file without one is not a file: the
// batches are all on the disk and there is nothing saying where. So Close
// matters more here than it does on a stream, and its error more still.
//
// An error from the underlying writer sticks, since the file is broken at that
// point. An error about a batch does not, since nothing went out.
type FileWriter struct {
	dst     io.Writer
	schema  dtype.Schema
	at      int64 // how many bytes have gone out, which is where the next message starts
	batches []block
	err     error
	closed  bool
}

// NewFileWriter starts a file on w and writes the magic and the schema. The
// error is either a schema Arrow cannot name or whatever w said about the write.
func NewFileWriter(w io.Writer, s dtype.Schema) (*FileWriter, error) {
	msg, err := EncodeSchema(s)
	if err != nil {
		return nil, err
	}
	f := &FileWriter{dst: w, schema: s}
	if err := f.write([]byte(filePad)); err != nil {
		return nil, err
	}
	if err := f.write(msg); err != nil {
		return nil, err
	}
	return f, nil
}

// Schema is the schema every batch in the file belongs to.
func (w *FileWriter) Schema() dtype.Schema { return w.schema }

// Write appends one record batch and remembers where it went. The columns have
// to be the types the schema says they are and there have to be as many of them
// as it has fields.
func (w *FileWriter) Write(b Batch) error {
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
	at, err := blockOf(w.at, msg)
	if err != nil {
		return err
	}
	if err := w.write(msg); err != nil {
		return err
	}
	w.batches = append(w.batches, at)
	return nil
}

// Close writes the end of stream marker and the footer after it. It does not
// close the writer underneath, which belongs to whoever opened it.
//
// Closing twice is not an error, so a deferred Close after an explicit one
// reports what the explicit one did rather than writing a second footer.
func (w *FileWriter) Close() error {
	if w.closed {
		return w.err
	}
	w.closed = true
	if w.err != nil {
		return w.err
	}

	// The marker goes in before the footer so that everything between the magic
	// and here is a stream and nothing else, which is what lets a reader of the
	// stream format read a file.
	if err := w.write(streamEnd); err != nil {
		return err
	}

	footer, err := encodeFooter(w.schema, w.batches)
	if err != nil {
		w.err = err
		return err
	}
	if err := checkLength(len(footer), "a footer"); err != nil {
		w.err = err
		return err
	}
	end := make([]byte, 0, len(footer)+fileTail)
	end = append(end, footer...)
	end = binary.LittleEndian.AppendUint32(end, uint32(len(footer)))
	end = append(end, fileMagic...)
	return w.write(end)
}

// write sends the bytes and counts them, since a footer is a list of where each
// message started.
func (w *FileWriter) write(p []byte) error {
	n, err := w.dst.Write(p)
	w.at += int64(n)
	if err != nil {
		w.err = err
	}
	return err
}

// encodeFooter writes the footer of a file, which is the schema and where every
// batch is.
func encodeFooter(s dtype.Schema, batches []block) ([]byte, error) {
	sw := schemaWriter{types: make(map[string]typeRef)}
	schema, err := sw.schema(s)
	if err != nil {
		return nil, err
	}

	// The vectors go down before the table that points at them. The dictionaries
	// are written as a vector of nothing rather than left out, because that is
	// what every other writer produces and a reader of somebody else's file is
	// the last place to find out which readers check.
	w := &sw.w
	dicts := w.blocks(nil)
	index := w.blocks(batches)

	w.startTable()
	w.slotInt(fbFooterVersion, fbVersionV5, 0)
	w.slotOffset(fbFooterSchema, schema)
	w.slotOffset(fbFooterDictionaries, dicts)
	w.slotOffset(fbFooterBatches, index)
	return w.finish(w.endTable()), nil
}

// FileReader reads the Arrow IPC file format.
//
// The footer is read when the reader is made, which is the schema and where
// every batch in the file is, and after that a batch is read by number. Nothing
// else is touched: reading the last batch of a file reads the footer and that
// batch. That is what the format is for, and it is the difference between a file
// and a stream.
//
// The schema is the one in the footer rather than the one in front of the
// batches. Both are written by the same writer and say the same thing, and the
// footer is the one a reader already has in hand.
//
// Each batch is read into a buffer of its own and its arrays point into that
// buffer, the same bargain the rest of the package makes. A batch costs the
// memory of the batch for as long as it is held and nothing once it is dropped.
type FileReader struct {
	src     io.ReaderAt
	schema  dtype.Schema
	batches []block
}

// NewFileReader reads the footer of a file and returns a reader for the batches
// it indexes. The size is how many bytes the file has, which the caller knows
// and an io.ReaderAt does not, the same way a zip reader is opened.
func NewFileReader(r io.ReaderAt, size int64) (*FileReader, error) {
	footer, limit, err := readFooter(r, size)
	if err != nil {
		return nil, err
	}
	s, batches, err := decodeFooter(footer, limit)
	if err != nil {
		return nil, err
	}
	return &FileReader{src: r, schema: s, batches: batches}, nil
}

// Schema is the schema every batch in the file belongs to.
func (r *FileReader) Schema() dtype.Schema { return r.schema }

// NumBatches is how many record batches the file holds.
func (r *FileReader) NumBatches() int { return len(r.batches) }

// Batch reads batch i, which is a read of that batch and nothing else. The
// batches of a file can be read in any order and more than once.
func (r *FileReader) Batch(i int) (Batch, error) {
	if i < 0 || i >= len(r.batches) {
		return Batch{}, fmt.Errorf("ipc: batch %d of a file of %d batches", i, len(r.batches))
	}

	at := r.batches[i]
	buf := make([]byte, int(at.size()))
	if err := readAt(r.src, buf, at.offset); err != nil {
		return Batch{}, fmt.Errorf("ipc: %w: reading batch %d: %w", ErrMessage, i, err)
	}
	b, _, err := DecodeBatch(r.schema, buf)
	return b, err
}

// readFooter reads the footer off the end of a file and returns it along with
// where it starts, which is as far into the file as any message can reach.
func readFooter(r io.ReaderAt, size int64) ([]byte, int64, error) {
	if least := int64(len(filePad) + fileTail); size < least {
		return nil, 0, fmt.Errorf("ipc: %w: a file of %d bytes, fewer than the %d an empty one has",
			ErrMessage, size, least)
	}

	var head [len(filePad)]byte
	if err := readAt(r, head[:], 0); err != nil {
		return nil, 0, fmt.Errorf("ipc: %w: reading the head of a file: %w", ErrMessage, err)
	}
	if string(head[:len(fileMagic)]) != fileMagic {
		return nil, 0, fmt.Errorf("ipc: %w: a file starting with %q rather than %q",
			ErrMessage, head[:len(fileMagic)], fileMagic)
	}

	var tail [fileTail]byte
	if err := readAt(r, tail[:], size-int64(fileTail)); err != nil {
		return nil, 0, fmt.Errorf("ipc: %w: reading the end of a file: %w", ErrMessage, err)
	}
	if string(tail[4:]) != fileMagic {
		return nil, 0, fmt.Errorf("ipc: %w: a file ending with %q rather than %q",
			ErrMessage, tail[4:], fileMagic)
	}

	// The length is the last thing in the file to be trusted, so it is checked
	// against the file rather than believed: a footer that starts before the
	// magic at the front is one that overlaps the messages it indexes.
	n := int32(binary.LittleEndian.Uint32(tail[:]))
	limit := size - int64(fileTail) - int64(n)
	if n <= 0 || limit < int64(len(filePad)) {
		return nil, 0, fmt.Errorf("ipc: %w: a footer of %d bytes at the end of a file of %d",
			ErrMessage, n, size)
	}

	footer := make([]byte, n)
	if err := readAt(r, footer, limit); err != nil {
		return nil, 0, fmt.Errorf("ipc: %w: reading the footer: %w", ErrMessage, err)
	}
	return footer, limit, nil
}

// decodeFooter reads the schema and the block index out of a footer.
func decodeFooter(footer []byte, limit int64) (dtype.Schema, []block, error) {
	root, err := fbRoot(footer)
	if err != nil {
		return dtype.Schema{}, nil, err
	}
	version, err := root.integer(fbFooterVersion, int16(0))
	if err != nil {
		return dtype.Schema{}, nil, err
	}
	if version > fbVersionV5 {
		return dtype.Schema{}, nil, fmt.Errorf("ipc: %w: metadata version %d, this reads up to V5",
			ErrUnsupported, version)
	}

	// A file with dictionaries in it is refused here rather than when a batch
	// turns out to need one, since the reader knows from the footer alone and
	// the caller would rather hear it on the way in.
	dicts, ok, err := root.vector(fbFooterDictionaries)
	if err != nil {
		return dtype.Schema{}, nil, err
	}
	if ok && dicts.len() > 0 {
		return dtype.Schema{}, nil, fmt.Errorf("ipc: %w: the file has %d dictionary batches in it, which this cannot read yet",
			ErrUnsupported, dicts.len())
	}

	table, ok, err := root.table(fbFooterSchema)
	if err != nil {
		return dtype.Schema{}, nil, err
	}
	if !ok {
		return dtype.Schema{}, nil, fmt.Errorf("ipc: %w: a footer with no schema in it", ErrMessage)
	}
	s, err := newDecoder(footer).schema(table)
	if err != nil {
		return dtype.Schema{}, nil, err
	}

	index, ok, err := root.vector(fbFooterBatches)
	if err != nil {
		return dtype.Schema{}, nil, err
	}
	if !ok {
		return s, nil, nil
	}
	batches, err := footerBlocks(index, limit)
	if err != nil {
		return dtype.Schema{}, nil, err
	}
	return s, batches, nil
}

// footerBlocks reads the block index and checks every block against the file it
// came out of.
func footerBlocks(index fbVector, limit int64) ([]block, error) {
	batches := make([]block, index.len())
	for i := range batches {
		b, err := index.block(i)
		if err != nil {
			return nil, err
		}
		if !b.inside(limit) {
			return nil, fmt.Errorf("ipc: %w: batch %d is %d bytes at %d, in %d bytes of messages",
				ErrMessage, i, b.size(), b.offset, limit)
		}
		batches[i] = b
	}
	return batches, nil
}

// readAt fills p from off and says so when there were not that many bytes.
//
// ReadAt is allowed to report the end of the file along with a read that filled
// the slice, so the count is what decides rather than the error.
func readAt(src io.ReaderAt, p []byte, off int64) error {
	n, err := src.ReadAt(p, off)
	if n == len(p) {
		return nil
	}
	if err == nil {
		err = io.ErrUnexpectedEOF
	}
	return unexpected(err)
}
