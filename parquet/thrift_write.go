package parquet

import "encoding/binary"

// Writing the Thrift compact protocol.
//
// This is thrift.go the other way round and it is the same protocol for the
// same shapes, rather than a general implementation of Thrift. A file that
// describes itself in Thrift is a file that cannot be written without writing
// Thrift, so the footer of every file this package produces comes out of here.
//
// The writing side is the easier of the two and for one reason: the bytes are
// this package's own. A reader is handed somebody else's claims and has to
// check every one of them before it acts, and a writer is handed a structure it
// built itself. So there is nothing here that validates and nothing that can
// fail, and what would have been an error is a caller passing a value the format
// has no room for, which is checked where the value comes from.
//
// The one thing worth care is the field header. It carries how far the id has
// moved since the last field rather than the id, so writing a struct means
// saving where the reader was and putting it back, exactly as reading one does.
// Get that wrong and the bytes are still a well formed Thrift structure, just
// one describing different fields, which is the kind of mistake that reads back
// as a file with its columns in the wrong order rather than as a file that
// refuses to open.

// writer is a Thrift structure being built.
//
// The bytes go into a buffer rather than to an io.Writer because a footer is
// written behind its own length, so nothing about it can go out until all of it
// is in hand. That is the same reason the page encoders hold a buffer.
type writer struct {
	buf []byte

	// last is the id of the field written most recently in the struct being
	// written, since a field header holds the distance from it rather than the
	// id itself. It goes back to nought at the start of every struct.
	last int16
}

// put appends one byte.
func (w *writer) put(b byte) { w.buf = append(w.buf, b) }

// uvarint writes an unsigned varint, seven bits to a byte, least significant
// first, with the top bit saying that another byte follows.
func (w *writer) uvarint(v uint64) { w.buf = binary.AppendUvarint(w.buf, v) }

// varint writes a signed varint, zigzagged so that a small negative number
// takes a small number of bytes.
func (w *writer) varint(v int64) { w.uvarint(uint64(v<<1 ^ v>>63)) }

// header writes the header of a field, which is how far the id has moved since
// the last one and the type of the value behind it.
func (w *writer) header(id int16, t thriftType) {
	if delta := id - w.last; delta > 0 && delta <= 15 {
		w.put(byte(delta)<<4 | byte(t))
	} else {
		// A distance of more than four bits, or one that goes backwards, is
		// written as the type on its own with the id behind it. Nothing in a
		// parquet footer needs it, since the fields of every struct in it are
		// written in order and none of them is fifteen apart, but a writer that
		// only handled the short form would be one field number away from
		// writing a file nothing can read.
		w.put(byte(t))
		w.varint(int64(id))
	}
	w.last = id
}

// structure writes a field holding a struct, calling f to write what is in it.
//
// The id the header moved to is put back afterwards, since the fields inside
// the struct are counted from nought and the field after it is counted from
// this one.
func (w *writer) structure(id int16, f func()) {
	w.header(id, thriftStruct)

	last := w.last
	w.last = 0
	f()
	w.put(byte(thriftStop))
	w.last = last
}

// int32Field writes an integer field. Every integer is a varint on the wire and
// the width in the header only says what the writer called it, so this and the
// two below differ in that and in nothing else.
func (w *writer) int32Field(id int16, v int32) {
	w.header(id, thriftInt32)
	w.varint(int64(v))
}

// int64Field writes a long field.
func (w *writer) int64Field(id int16, v int64) {
	w.header(id, thriftInt64)
	w.varint(v)
}

// int16Field writes a short field.
func (w *writer) int16Field(id, v int16) {
	w.header(id, thriftInt16)
	w.varint(int64(v))
}

// int8Field writes a byte field, which is the one integer that is not a varint.
// It is one byte whatever it holds, which is the same one byte a varint of a
// value that small would have been.
func (w *writer) int8Field(id int16, v int8) {
	w.header(id, thriftInt8)
	w.put(byte(v))
}

// boolField writes a bool field, which has no value of its own: which of the
// two types the header carries is the value.
func (w *writer) boolField(id int16, v bool) {
	if v {
		w.header(id, thriftTrue)
	} else {
		w.header(id, thriftFalse)
	}
}

// binary writes the header of a field holding n bytes, which the caller appends
// behind it.
func (w *writer) binary(id int16, n int) {
	w.header(id, thriftBinary)
	w.uvarint(uint64(n))
}

// bytesField writes a binary field.
func (w *writer) bytesField(id int16, v []byte) {
	w.binary(id, len(v))
	w.buf = append(w.buf, v...)
}

// textField writes a string field, which is a binary field holding the bytes of
// the string.
func (w *writer) textField(id int16, v string) {
	w.binary(id, len(v))
	w.buf = append(w.buf, v...)
}

// listHeader writes a field holding a list of n elements, all of one type.
//
// A list of fourteen or fewer says how many it has in the same byte as the type
// of its elements, and a longer one puts the count behind that byte. Fifteen is
// the value that means the count did not fit, which is why fourteen rather than
// fifteen is the largest that does.
func (w *writer) listHeader(id int16, t thriftType, n int) {
	w.header(id, thriftList)
	if n < 15 {
		w.put(byte(n)<<4 | byte(t))
	} else {
		w.put(0xf0 | byte(t))
		w.uvarint(uint64(n))
	}
}

// writeStructs writes a list of structs, calling write for each of them.
//
// Every element starts a struct of its own, so the field id goes back to nought
// in front of each one and the field after the list carries on from the list.
func writeStructs[T any](w *writer, id int16, vals []T, write func(*T, *writer)) {
	w.listHeader(id, thriftStruct, len(vals))

	last := w.last
	for i := range vals {
		w.last = 0
		write(&vals[i], w)
		w.put(byte(thriftStop))
	}
	w.last = last
}

// writeTexts writes a list of strings. A list has one header for all of its
// elements, so what follows it is the values with nothing between them.
func writeTexts(w *writer, id int16, vals []string) {
	w.listHeader(id, thriftBinary, len(vals))
	for _, v := range vals {
		w.uvarint(uint64(len(v)))
		w.buf = append(w.buf, v...)
	}
}

// writeEnums writes a list of integers as whatever the format calls them. Every
// enumeration in parquet is an int32 on the wire, so the one function covers the
// encodings a column chunk used and anything else that turns up.
func writeEnums[T ~int32](w *writer, id int16, vals []T) {
	w.listHeader(id, thriftInt32, len(vals))
	for _, v := range vals {
		w.varint(int64(v))
	}
}
