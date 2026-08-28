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
//
// The values of a dictionary encoded column go out once, in front of the first
// batch that reads from them, and the footer says where they are. A file has one
// dictionary per column and no way to replace it, so a batch arriving with
// different values is an error rather than a second dictionary.
type FileWriter struct {
	dst     io.Writer
	schema  dtype.Schema
	dicts   *dictWriter
	at      int64 // how many bytes have gone out, which is where the next message starts
	values  []block
	batches []block
	err     error
	closed  bool
}

// NewFileWriter starts a file on w and writes the magic and the schema. The
// error is either a schema Arrow cannot name or whatever w said about the write.
func NewFileWriter(w io.Writer, s dtype.Schema) (*FileWriter, error) {
	msg, ids, err := encodeSchemaMessage(s)
	if err != nil {
		return nil, err
	}
	f := &FileWriter{dst: w, schema: s, dicts: newDictWriter(s, ids)}
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
//
// A dictionary encoded column has to arrive as the column and not as its
// indices, since the values are what this has to write, and every batch of one
// column has to arrive with the same values.
func (w *FileWriter) Write(b Batch) error {
	if w.err != nil {
		return w.err
	}
	if w.closed {
		return fmt.Errorf("ipc: %w", ErrClosed)
	}

	// Everything is encoded before anything is written, so that a batch that
	// turns out to be wrong leaves the file where it was.
	msg, err := EncodeBatch(w.schema, b)
	if err != nil {
		return err
	}
	values, err := w.dicts.once(w.schema, b)
	if err != nil {
		return err
	}

	if err = w.writeValues(values); err != nil {
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

// writeValues writes the dictionary messages that go in front of a batch and
// remembers where each of them went. They are blocks in the footer of their own
// rather than batches, since a reader needs all of them before it reads
// anything and none of them are rows.
func (w *FileWriter) writeValues(values []dictMessage) error {
	for _, d := range values {
		at, err := blockOf(w.at, d.msg)
		if err != nil {
			return err
		}
		if err := w.write(d.msg); err != nil {
			return err
		}
		w.values = append(w.values, at)
		w.dicts.done(d)
	}
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

	footer, err := encodeFooter(w.schema, w.values, w.batches)
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
// message a reader has to find again is.
func encodeFooter(s dtype.Schema, values, batches []block) ([]byte, error) {
	sw := schemaWriter{types: make(map[string]typeRef)}
	schema, err := sw.schema(s)
	if err != nil {
		return nil, err
	}

	// The vectors go down before the table that points at them. A file with no
	// dictionaries in it writes a vector of nothing rather than leaving the
	// field out, because that is what every other writer produces and a reader
	// of somebody else's file is the last place to find out which readers check.
	w := &sw.w
	dicts := w.blocks(values)
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
	dicts   *dicts
	batches []block
}

// NewFileReader reads the footer of a file and returns a reader for the batches
// it indexes. The size is how many bytes the file has, which the caller knows
// and an io.ReaderAt does not, the same way a zip reader is opened.
//
// The dictionaries are read here as well. They are needed by any batch that
// reads from them, so a reader that put it off would be doing it on whichever
// batch happened to be asked for first, and there is one of them per column
// rather than one per batch.
func NewFileReader(r io.ReaderAt, size int64) (*FileReader, error) {
	footer, limit, err := readFooter(r, size)
	if err != nil {
		return nil, err
	}
	f, err := decodeFooter(footer, limit)
	if err != nil {
		return nil, err
	}
	d, err := newDicts(f.schema, f.ids)
	if err != nil {
		return nil, err
	}

	out := &FileReader{src: r, schema: f.schema, dicts: d, batches: f.batches}
	for i, at := range f.values {
		buf := make([]byte, int(at.size()))
		if err := readAt(r, buf, at.offset); err != nil {
			return nil, fmt.Errorf("ipc: %w: reading dictionary %d: %w", ErrMessage, i, err)
		}
		if err := d.read(buf); err != nil {
			return nil, err
		}
	}
	// A file writes one dictionary per identifier and the footer says how many
	// it wrote, so a file that wrote two of them under one identifier is one
	// whose dictionaries do not add up. Replacing a dictionary is a thing a
	// stream may do and a file may not, since the batches of a file are read in
	// whatever order the reader wants them.
	if d.count() != len(f.values) {
		return nil, fmt.Errorf("ipc: %w: the file has %d dictionary batches under %d identifiers, and a file cannot replace a dictionary",
			ErrUnsupported, len(f.values), d.count())
	}
	return out, nil
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
	if err != nil {
		return Batch{}, err
	}
	if err := r.dicts.bind(r.schema, &b); err != nil {
		return Batch{}, err
	}
	return b, nil
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

// footer is what the end of a file says: the schema its batches belong to, the
// dictionary identifier of each of its columns, and where every message a
// reader has to find again is.
type footer struct {
	schema  dtype.Schema
	ids     []int64
	values  []block
	batches []block
}

// decodeFooter reads the schema and the block index out of a footer.
func decodeFooter(b []byte, limit int64) (footer, error) {
	root, err := fbRoot(b)
	if err != nil {
		return footer{}, err
	}
	version, err := root.integer(fbFooterVersion, int16(0))
	if err != nil {
		return footer{}, err
	}
	if version > fbVersionV5 {
		return footer{}, fmt.Errorf("ipc: %w: metadata version %d, this reads up to V5",
			ErrUnsupported, version)
	}

	table, ok, err := root.table(fbFooterSchema)
	if err != nil {
		return footer{}, err
	}
	if !ok {
		return footer{}, fmt.Errorf("ipc: %w: a footer with no schema in it", ErrMessage)
	}

	var f footer
	d := newDecoder(b)
	if f.schema, err = d.schema(table); err != nil {
		return footer{}, err
	}
	f.ids = d.ids

	if f.values, err = footerBlocks(root, fbFooterDictionaries, limit); err != nil {
		return footer{}, err
	}
	if f.batches, err = footerBlocks(root, fbFooterBatches, limit); err != nil {
		return footer{}, err
	}
	return f, nil
}

// footerBlocks reads one of the two block indexes of a footer and checks every
// block in it against the file it came out of.
func footerBlocks(root fbTable, id int, limit int64) ([]block, error) {
	index, ok, err := root.vector(id)
	if err != nil || !ok {
		return nil, err
	}

	blocks := make([]block, index.len())
	for i := range blocks {
		b, err := index.block(i)
		if err != nil {
			return nil, err
		}
		if !b.inside(limit) {
			return nil, fmt.Errorf("ipc: %w: message %d is %d bytes at %d, in %d bytes of messages",
				ErrMessage, i, b.size(), b.offset, limit)
		}
		blocks[i] = b
	}
	return blocks, nil
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
