package ipc

import (
	"errors"
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/strview"
)

// The Arrow IPC record batch message.
//
// A record batch is the values of a schema that went past earlier. The message
// itself holds no types and no names, only a row count, one node per column
// saying how many values it has and how many of them are missing, and one
// description per buffer saying where in the body it starts and how long it is.
// Everything else comes from the schema, which is why both halves of this file
// take one.
//
// The body follows the metadata and is the buffers themselves, in the order the
// descriptions name them, each starting on an eight byte boundary. Nothing in
// it is copied on the way in: an array that comes out of DecodeBatch points
// into the bytes it was decoded from.
//
// The buffers of a column are the ones its layout has, which is what Export and
// Import already agree on: no buffers for a null column, a validity bitmap and
// the values for a fixed width one, and a validity bitmap, the views and one
// buffer per data block for text and bytes. The last of those is why the
// metadata carries variadicBufferCounts, since a reader counting buffers has no
// other way to know how many blocks a column of strings brought with it.

// The field numbers of the RecordBatch table, in declaration order.
const (
	fbBatchLength = iota
	fbBatchNodes
	fbBatchBuffers
	fbBatchCompression
	fbBatchVariadic
)

// fbNodeSize is the size of the two structs the metadata is made of. A
// FieldNode is a length and a null count and a Buffer is an offset and a
// length, so both of them are sixteen bytes of two int64s.
const fbNodeSize = 16

// Batch is one record batch: some rows of every column of a schema.
//
// Length is on the struct rather than taken from the columns because a batch of
// no columns still has a number of rows, which is what a count of a table with
// no projection reads. Every column has to be that long.
type Batch struct {
	// Length is the number of rows.
	Length int

	// Columns are the values, one array per field of the schema, in the
	// order the schema has them.
	Columns []*array.Array
}

// EncodeBatch returns the Arrow IPC record batch message for b, with the values
// after it.
//
// The result is an encapsulated message, framed the way a schema is, followed by
// the body it describes. Every buffer in the body starts on an eight byte
// boundary and the body is padded out to one, so a reader can use the buffers
// where they lie and whatever comes next starts on the alignment the format
// promises.
//
// The columns have to be the types the schema says they are, and there have to
// be as many of them as the schema has fields. The nested types cannot be
// written yet, and the error names the column that could not be.
//
// A dictionary encoded column is written as its indices. The values travel in a
// dictionary batch of their own, which is a message this does not write, so a
// caller doing its own framing has to write those as well and the Writer and the
// FileWriter are the ones that do. Either form of the column is taken: the
// dictionary encoded column itself, which is what array.NewDictionary builds, or
// the indices on their own, which is what DecodeBatch hands back.
//
// A column carrying an offset, which is what slicing a batch out of a longer one
// gives, is trimmed on the way out. The format has no per column offset, so the
// values in front of the first one are not written, and a validity bitmap that
// begins part way through a byte is shifted rather than sliced.
func EncodeBatch(s dtype.Schema, b Batch) ([]byte, error) {
	nodes, body, variadic, err := batchBody(s, b)
	if err != nil {
		return nil, err
	}

	msg := encodeBatchMessage(b.Length, nodes, body.descs, variadic, len(body.buf))
	if err := checkLength(len(msg), "a record batch"); err != nil {
		return nil, err
	}
	return append(frame(msg), body.buf...), nil
}

// batchBody lays the buffers of every column of a batch out in a body and
// returns the field nodes describing the columns along with it. A record batch
// message and a dictionary batch message are the same thing at this level and
// differ only in what is written around it.
func batchBody(s dtype.Schema, b Batch) ([]span, bodyWriter, []int64, error) {
	var body bodyWriter
	if len(b.Columns) != len(s.Fields) {
		return nil, body, nil, fmt.Errorf("ipc: %w: a batch of %d columns for a schema of %d fields",
			ErrBuffers, len(b.Columns), len(s.Fields))
	}
	if b.Length < 0 {
		return nil, body, nil, fmt.Errorf("ipc: %w: a batch of %d rows", ErrBuffers, b.Length)
	}

	nodes := make([]span, len(b.Columns))
	var variadic []int64

	for i, col := range b.Columns {
		f := s.Fields[i]
		if col == nil {
			return nil, body, nil, fmt.Errorf("ipc: %w: column %q is nil", ErrBuffers, f.Name)
		}
		stored := storedType(f.Type)
		if !dtype.Equal(col.DType(), f.Type) && !dtype.Equal(col.DType(), stored) {
			return nil, body, nil, fmt.Errorf("ipc: %w: column %q is a %s, the schema says %s",
				ErrBuffers, f.Name, col.DType(), f.Type)
		}
		if col.Len() != b.Length {
			return nil, body, nil, fmt.Errorf("ipc: %w: column %q has %d values in a batch of %d rows",
				ErrBuffers, f.Name, col.Len(), b.Length)
		}
		if d := col.Indices(); d != nil {
			// What goes on the wire is the indices, and the values went out in a
			// message of their own or are about to.
			col = d
		}

		l, err := Export(col)
		if err != nil {
			return nil, body, nil, fmt.Errorf("%w: column %q", err, f.Name)
		}
		bufs, err := trim(stored, l)
		if err != nil {
			return nil, body, nil, fmt.Errorf("%w: column %q", err, f.Name)
		}
		if isText(stored) {
			variadic = append(variadic, int64(len(bufs)-2))
		}

		nodes[i] = span{first: int64(l.Length), second: int64(l.NullCount)}
		for _, p := range bufs {
			body.add(p)
		}
	}
	body.pad()
	return nodes, body, variadic, nil
}

// encodeBatchMessage writes the metadata on its own. The stream and the file
// both write this and then the body, and the file writes down where the pair of
// them started.
func encodeBatchMessage(length int, nodes, buffers []span, variadic []int64, bodyLength int) []byte {
	var w fbBuilder
	batch := recordBatch(&w, length, nodes, buffers, variadic)

	w.startTable()
	w.slotInt(fbMessageVersion, fbVersionV5, 0)
	w.slotUint8(fbMessageHeaderType, fbHeaderRecordBatch)
	w.slotOffset(fbMessageHeader, batch)
	w.slotInt(fbMessageBodyLength, int64(bodyLength), 0)
	return w.finish(w.endTable())
}

// recordBatch writes the RecordBatch table and returns where it starts. It is
// the header of a record batch message and the middle of a dictionary batch
// message, which is a record batch of one column with an identifier on it.
func recordBatch(w *fbBuilder, length int, nodes, buffers []span, variadic []int64) fbOffset {
	// The vectors go down before the table that points at them, since nothing
	// can be pointed at before it exists.
	nodesVec := w.spans(nodes)
	buffersVec := w.spans(buffers)
	variadicVec := fbOffset(0)
	if len(variadic) > 0 {
		variadicVec = w.int64s(variadic)
	}

	w.startTable()
	w.slotInt(fbBatchLength, int64(length), 0)
	w.slotOffset(fbBatchNodes, nodesVec)
	w.slotOffset(fbBatchBuffers, buffersVec)
	w.slotOffset(fbBatchVariadic, variadicVec)
	return w.endTable()
}

// storedType is the type whose buffers a column of this type travels as. It is
// the type itself for everything but a dictionary encoded column, which travels
// as its indices and leaves its values to a message of their own.
// A dictionary indexed by something that is not an integer is left alone, so
// that it is refused as a type nothing can name rather than read as whatever
// the index happens to be. Nothing that has been through dtype.Validate is one
// of those, and a schema handed straight to DecodeBatch has not been.
func storedType(t dtype.DataType) dtype.DataType {
	if d, ok := t.(dtype.Dictionary); ok && dtype.IsInteger(d.Index) {
		return d.Index
	}
	return t
}

// bodyWriter lays the buffers of a batch out one after another and remembers
// where each of them went.
type bodyWriter struct {
	buf   []byte
	descs []span
}

// add appends one buffer on the alignment every buffer starts on, and writes
// down where it landed. A buffer of no bytes still gets a description, since a
// reader finds the buffers of a column by counting rather than by name, and a
// column with no nulls is a validity buffer that is not there.
func (b *bodyWriter) add(p []byte) {
	b.pad()
	b.descs = append(b.descs, span{first: int64(len(b.buf)), second: int64(len(p))})
	b.buf = append(b.buf, p...)
}

// pad rounds the body up to the alignment.
func (b *bodyWriter) pad() {
	for len(b.buf)%fbPad != 0 {
		b.buf = append(b.buf, 0)
	}
}

// trim cuts the offset off the front of a layout, which is what a batch sliced
// out of a longer one carries and what the format has nowhere to put.
//
// Everything but the bitmaps is a subslice, so a column that starts at zero,
// which is nearly all of them, is not copied at all. A bitmap whose first bit is
// not the first bit of a byte has to be shifted, and that is one byte of copying
// for every eight rows.
func trim(t dtype.DataType, l Layout) ([][]byte, error) {
	if l.Offset == 0 || len(l.Buffers) == 0 {
		return l.Buffers, nil
	}

	out := make([][]byte, len(l.Buffers))
	valid, err := bitRange(l.Buffers[0], l.Offset, l.Length)
	if err != nil {
		return nil, err
	}
	out[0] = valid

	if isText(t) {
		// The views are cut down and the data blocks are not. A view names its
		// block by number, so dropping a block would mean rewriting every view
		// that points past it, and the blocks a trimmed column no longer reaches
		// are dead weight rather than a wrong answer.
		if out[1], err = byteRange(l.Buffers[1], l.Offset, l.Length, strview.Size); err != nil {
			return nil, err
		}
		copy(out[2:], l.Buffers[2:])
		return out, nil
	}

	bits, ok := dtype.Bits(t)
	if !ok {
		return nil, fmt.Errorf("ipc: %w: writing a %s column", ErrUnsupported, t)
	}
	if bits == 1 {
		if out[1], err = bitRange(l.Buffers[1], l.Offset, l.Length); err != nil {
			return nil, err
		}
		return out, nil
	}
	if out[1], err = byteRange(l.Buffers[1], l.Offset, l.Length, bits/8); err != nil {
		return nil, err
	}
	return out, nil
}

// bitRange is bits off through off+n-1 of a bitmap, renumbered from zero. A
// buffer of no bytes stays one, since that is how a column with no nulls
// travels.
func bitRange(b []byte, off, n int) ([]byte, error) {
	if len(b) == 0 {
		return nil, nil
	}
	if len(b)*8 < off+n {
		return nil, fmt.Errorf("ipc: %w: a bitmap of %d bytes holds %d bits, need %d",
			ErrBuffers, len(b), len(b)*8, off+n)
	}
	if off%8 == 0 {
		return b[off/8 : (off+n+7)/8], nil
	}
	return bitmap.FromBytes(b, off+n).Slice(off, off+n).Bytes(), nil
}

// byteRange is values off through off+n-1 of a buffer of fixed width values.
func byteRange(b []byte, off, n, width int) ([]byte, error) {
	if len(b) < (off+n)*width {
		return nil, fmt.Errorf("ipc: %w: %d values of %d bytes need %d, the buffer has %d",
			ErrBuffers, off+n, width, (off+n)*width, len(b))
	}
	return b[off*width : (off+n)*width], nil
}

// DecodeBatch reads one encapsulated Arrow IPC record batch message and the body
// after it, and returns the batch along with whatever follows the pair of them.
//
// The schema is the one the batch belongs to, which the format expects a reader
// to have already: a batch on the wire says how many values each column has and
// where its buffers are, and nothing at all about what they mean.
//
// Nothing is copied. The arrays point into b, so they are alive for as long as
// those bytes are unmodified, the same bargain Import makes.
//
// A dictionary encoded column comes back as its indices, of the index type the
// schema names, since the values are in a dictionary batch and one message on
// its own has no way to know them. The Reader and the FileReader put the two
// together, which is what a caller who wants the column rather than the message
// should be using.
//
// The bytes are somebody else's, so everything in them is checked. Every buffer
// has to lie inside the body, every column has to have as many values as the
// batch says it has rows, and the buffers have to add up to what the schema
// needs. A compressed body and the nested types are refused rather than half
// read.
//
// Text and bytes are the one place a schema does not say enough. Arrow has four
// layouts for each and kuma collapses them into one type, so this reads the
// layout out of the batch instead: a column with a variadic buffer count is the
// view layout kuma writes, and one without is the offset layout everybody else
// does. A batch that mixes the two is refused, since there is nothing in either
// message saying which column is which.
func DecodeBatch(s dtype.Schema, b []byte) (Batch, []byte, error) {
	table, body, after, err := messageBody(b, fbHeaderRecordBatch)
	if err != nil {
		return Batch{}, nil, err
	}
	batch, err := decodeBatch(s, table, body)
	if err != nil {
		return Batch{}, nil, err
	}
	return batch, after, nil
}

// messageBody takes an encapsulated message apart into the header table, the
// body the header describes and whatever follows the pair of them, and refuses
// a message that is not the kind the caller is reading.
//
// It is the part of every message that is the same whatever the message holds:
// the framing, the version, which of the headers it is, and how many of the
// bytes after the metadata belong to it.
func messageBody(b []byte, want uint8) (fbTable, []byte, []byte, error) {
	msg, rest, err := unframe(b)
	if err != nil {
		return fbTable{}, nil, nil, err
	}
	if msg == nil {
		return fbTable{}, nil, nil, fmt.Errorf("ipc: %w: a %s of no bytes", ErrMessage, headerName(want))
	}

	root, err := fbRoot(msg)
	if err != nil {
		return fbTable{}, nil, nil, err
	}
	version, err := root.integer(fbMessageVersion, int16(0))
	if err != nil {
		return fbTable{}, nil, nil, err
	}
	if version > fbVersionV5 {
		return fbTable{}, nil, nil, fmt.Errorf("ipc: %w: metadata version %d, this reads up to V5",
			ErrUnsupported, version)
	}
	header, err := root.uint8(fbMessageHeaderType, fbHeaderNone)
	if err != nil {
		return fbTable{}, nil, nil, err
	}
	if header != want {
		return fbTable{}, nil, nil, fmt.Errorf("ipc: %w: the message is a %s, want a %s",
			ErrMessage, headerName(header), headerName(want))
	}

	size, err := root.integer(fbMessageBodyLength, int64(0))
	if err != nil {
		return fbTable{}, nil, nil, err
	}
	if size < 0 || size > int64(len(rest)) {
		return fbTable{}, nil, nil, fmt.Errorf("ipc: %w: a body of %d bytes with %d left to read it from",
			ErrMessage, size, len(rest))
	}

	table, ok, err := root.table(fbMessageHeader)
	if err != nil {
		return fbTable{}, nil, nil, err
	}
	if !ok {
		return fbTable{}, nil, nil, fmt.Errorf("ipc: %w: the message says it holds a %s and holds nothing",
			ErrMessage, headerName(want))
	}
	return table, rest[:size], rest[size:], nil
}

// decodeBatch reads the RecordBatch table and builds the columns out of the
// body it describes.
func decodeBatch(s dtype.Schema, t fbTable, body []byte) (Batch, error) {
	rows, err := t.integer(fbBatchLength, int64(0))
	if err != nil {
		return Batch{}, err
	}
	length, err := asInt(rows, "a batch length")
	if err != nil {
		return Batch{}, err
	}
	_, compressed, err := t.table(fbBatchCompression)
	if err != nil {
		return Batch{}, err
	}
	if compressed {
		return Batch{}, fmt.Errorf("ipc: %w: the body is compressed", ErrUnsupported)
	}

	nodes, _, err := t.vector(fbBatchNodes)
	if err != nil {
		return Batch{}, err
	}
	buffers, _, err := t.vector(fbBatchBuffers)
	if err != nil {
		return Batch{}, err
	}
	variadic, _, err := t.vector(fbBatchVariadic)
	if err != nil {
		return Batch{}, err
	}
	if nodes.len() != len(s.Fields) {
		return Batch{}, fmt.Errorf("ipc: %w: %d field nodes for a schema of %d fields, which is a batch of nested columns or one of another schema",
			ErrMessage, nodes.len(), len(s.Fields))
	}
	views, err := viewLayout(s, variadic.len())
	if err != nil {
		return Batch{}, err
	}

	b := Batch{Length: length, Columns: make([]*array.Array, len(s.Fields))}
	at, text := 0, 0
	for i, f := range s.Fields {
		node, err := nodes.span(i)
		if err != nil {
			return Batch{}, err
		}
		if node.first != rows {
			return Batch{}, fmt.Errorf("ipc: %w: column %q has %d values in a batch of %d rows",
				ErrMessage, f.Name, node.first, rows)
		}

		count, format, err := columnBuffers(f, views, variadic, text)
		if err != nil {
			return Batch{}, err
		}
		if isText(f.Type) {
			text++
		}
		if at+count > buffers.len() {
			return Batch{}, fmt.Errorf("ipc: %w: column %q needs %d buffers, the batch has %d left of %d",
				ErrMessage, f.Name, count, buffers.len()-at, buffers.len())
		}

		l := Layout{Length: length, NullCount: int(node.second), Buffers: make([][]byte, count)}
		for k := range count {
			if l.Buffers[k], err = bodyBuffer(buffers, at+k, body, f.Name); err != nil {
				return Batch{}, err
			}
		}
		at += count

		if format == "" {
			// A text column in the offset layout, whose two widths are told
			// apart by how long the offsets are. Only a column of no rows can
			// be either, and then both readings are the same empty column.
			format = offsetFormat(f.Type, l.Buffers[1], length)
		}
		if b.Columns[i], err = importColumn(f, format, l); err != nil {
			return Batch{}, err
		}
	}
	if at != buffers.len() {
		return Batch{}, fmt.Errorf("ipc: %w: the schema accounts for %d buffers, the batch has %d",
			ErrMessage, at, buffers.len())
	}
	return b, nil
}

// importColumn builds one column and names it in whatever goes wrong.
//
// The kind of the error matters as much as the words. The array package refuses
// buffers that do not hold what they say they do, in its own words and with no
// error to compare against, and a caller holding a file that has been truncated
// or written by something with a bug in it wants one thing to test for. What a
// batch could not be built out of is bad buffers.
func importColumn(f dtype.Field, format string, l Layout) (*array.Array, error) {
	a, err := Import(format, l)
	switch {
	case err == nil:
		return a, nil
	case errors.Is(err, ErrBuffers), errors.Is(err, ErrFormat), errors.Is(err, ErrUnsupported):
		return nil, fmt.Errorf("%w: column %q", err, f.Name)
	default:
		return nil, fmt.Errorf("ipc: %w: column %q: %w", ErrBuffers, f.Name, err)
	}
}

// columnBuffers is how many buffers a column takes out of the batch and the
// format string its layout is named by. An empty format string means a text
// column in one of the offset layouts, which is decided from the buffer itself.
func columnBuffers(f dtype.Field, views bool, variadic fbVector, text int) (int, string, error) {
	if f.Type == nil {
		return 0, "", fmt.Errorf("ipc: %w: field %q has no type", ErrType, f.Name)
	}
	if childFields(f.Type) != nil {
		return 0, "", fmt.Errorf("ipc: %w: column %q is a %s, and there are no nested arrays to read it into",
			ErrUnsupported, f.Name, f.Type)
	}

	// A dictionary encoded column is its indices here, which are two buffers of
	// an integer type whatever the values turn out to be. The values are a
	// column of their own in a message of their own and come through this the
	// same way when they get there.
	t := storedType(f.Type)
	format, err := Format(t)
	if err != nil {
		return 0, "", err
	}
	switch {
	case t.Kind() == dtype.NullKind:
		// A null column is its own description: every value is missing and
		// there is nothing to point at.
		return 0, format, nil

	case isText(t) && views:
		n, err := variadic.int64at(text)
		if err != nil {
			return 0, "", err
		}
		blocks, err := asInt(n, "a variadic buffer count")
		if err != nil {
			return 0, "", err
		}
		return 2 + blocks, format, nil

	case isText(t):
		return 3, "", nil

	case t.Kind() == dtype.LargeStringKind, t.Kind() == dtype.LargeBinaryKind:
		// The wide offset layouts are the one text shape a schema does name,
		// since kuma keeps them as their own types.
		return 3, format, nil

	default:
		return 2, format, nil
	}
}

// viewLayout says whether the text columns of a batch are in the view layout
// kuma writes or the offset layout most other implementations do.
//
// The count is the only thing on the wire that tells them apart. A view column
// has one, an offset column does not, and a batch whose counts match neither all
// of its text columns nor none of them is one this cannot take apart.
func viewLayout(s dtype.Schema, counts int) (bool, error) {
	text := 0
	for _, f := range s.Fields {
		if isText(f.Type) {
			text++
		}
	}
	switch counts {
	case 0:
		return false, nil
	case text:
		return true, nil
	default:
		return false, fmt.Errorf("ipc: %w: %d variadic buffer counts for %d text columns, which is a batch that lays them out two different ways",
			ErrUnsupported, counts, text)
	}
}

// bodyBuffer is the bytes buffer i of a batch describes, checked against the
// body they are supposed to be inside.
func bodyBuffer(buffers fbVector, i int, body []byte, name string) ([]byte, error) {
	sp, err := buffers.span(i)
	if err != nil {
		return nil, err
	}
	// The two halves are compared with the body separately rather than added
	// up, since a length near the top of an int64 added to an offset is a
	// number that comes out small.
	if sp.first < 0 || sp.second < 0 || sp.first > int64(len(body)) || sp.second > int64(len(body))-sp.first {
		return nil, fmt.Errorf("ipc: %w: column %q has a buffer of %d bytes at %d, in a body of %d",
			ErrMessage, name, sp.second, sp.first, len(body))
	}
	return body[sp.first : sp.first+sp.second], nil
}

// offsetFormat picks between the two offset layouts by the size of the offsets
// buffer, which holds one more offset than there are values.
func offsetFormat(t dtype.DataType, offsets []byte, length int) string {
	wide := len(offsets) >= (length+1)*8
	if t.Kind() == dtype.BinaryKind {
		if wide {
			return "Z"
		}
		return "z"
	}
	if wide {
		return "U"
	}
	return "u"
}

// isText reports whether a column of this type is one of the two kuma holds as
// views, which are the two whose buffer count is not fixed.
func isText(t dtype.DataType) bool {
	k := t.Kind()
	return k == dtype.StringKind || k == dtype.BinaryKind
}

// asInt narrows one of the int64 counts in a message to the int the rest of the
// package counts in. On a 64 bit machine the only thing it refuses is a
// negative number.
func asInt(v int64, what string) (int, error) {
	if v < 0 || int64(int(v)) != v {
		return 0, fmt.Errorf("ipc: %w: %s of %d", ErrMessage, what, v)
	}
	return int(v), nil
}
