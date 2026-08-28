package ipc

import (
	"encoding/binary"
	"fmt"
)

// Reading the FlatBuffers that Arrow writes its metadata in.
//
// Arrow IPC describes a schema and a batch in FlatBuffers, so reading the
// format at all means reading FlatBuffers. This is the whole of it, in the one
// direction and for the handful of shapes Arrow uses, rather than a general
// implementation of the format.
//
// A buffer starts with an offset to the root table. A table starts with a
// signed offset backwards to its vtable, and the vtable says where each field
// sits, or that it is absent and the default stands. Everything else is a
// scalar, a string, a vector or another table, and every one of them is
// reached by an offset from where the offset itself is.
//
// The point of doing it by hand is that every read here is bounds checked. A
// message arriving over a socket is somebody else's bytes, and the generated
// readers for this format famously are not checked, so a file with one number
// changed reads whatever is next in memory. Nothing in this file indexes
// without asking first, and everything it refuses says why.

// fbTable is a table inside a buffer: where it starts, and where its vtable
// starts and ends. The buffer is the whole message, since every offset in a
// FlatBuffer is relative to something inside it.
type fbTable struct {
	buf  []byte
	pos  int // where the table starts
	vt   int // where the vtable starts
	vlen int // how many bytes of vtable there are
}

// fbVector is a vector inside a buffer: where the elements start and how many
// of them there are. What one element is depends on the field, which is why
// there is a read per shape rather than a type parameter.
type fbVector struct {
	buf []byte
	pos int // where the first element starts
	n   int
}

// fbRoot returns the root table of a buffer.
func fbRoot(buf []byte) (fbTable, error) {
	off, err := fbUint32(buf, 0)
	if err != nil {
		return fbTable{}, err
	}
	return fbTableAt(buf, int(off))
}

// fbTableAt reads the table that starts at pos.
//
// The first thing in a table is a signed offset backwards to its vtable, which
// is the one offset in the format that is subtracted rather than added.
func fbTableAt(buf []byte, pos int) (fbTable, error) {
	back, err := fbInt32(buf, pos)
	if err != nil {
		return fbTable{}, err
	}
	vt := pos - int(back)
	if vt < 0 || vt+4 > len(buf) {
		return fbTable{}, fmt.Errorf("ipc: %w: a table at %d has its vtable at %d, outside a buffer of %d bytes",
			ErrMessage, pos, vt, len(buf))
	}
	vlen := int(binary.LittleEndian.Uint16(buf[vt:]))
	if vlen < 4 || vt+vlen > len(buf) {
		return fbTable{}, fmt.Errorf("ipc: %w: a vtable at %d is %d bytes, which does not fit in %d",
			ErrMessage, vt, vlen, len(buf))
	}
	return fbTable{buf: buf, pos: pos, vt: vt, vlen: vlen}, nil
}

// slot is where field id of this table sits, if it is there at all.
//
// A vtable that stops before a field means the writer did not have one, and a
// zero in the slot means the same thing. Both are how a default is written,
// and neither is an error: it is what lets a reader of an old message read a
// new one.
func (t fbTable) slot(id int) (int, bool, error) {
	off := 4 + id*2
	if off+2 > t.vlen {
		return 0, false, nil
	}
	v := int(binary.LittleEndian.Uint16(t.buf[t.vt+off:]))
	if v == 0 {
		return 0, false, nil
	}
	pos := t.pos + v
	if pos < 0 || pos >= len(t.buf) {
		return 0, false, fmt.Errorf("ipc: %w: field %d of a table at %d is at %d, outside a buffer of %d bytes",
			ErrMessage, id, t.pos, pos, len(t.buf))
	}
	return pos, true, nil
}

// The scalar reads. Each returns the default when the field is not there,
// which is what a FlatBuffer means by leaving one out.
func (t fbTable) uint8(id int, def uint8) (uint8, error) {
	pos, ok, err := t.slot(id)
	if err != nil || !ok {
		return def, err
	}
	return t.buf[pos], nil
}

func (t fbTable) boolean(id int, def bool) (bool, error) {
	pos, ok, err := t.slot(id)
	if err != nil || !ok {
		return def, err
	}
	return t.buf[pos] != 0, nil
}

// integer reads an integer field of whatever width the default's type has,
// which is the width the schema declares the field with. The three widths are
// one method for the same reason the three writes are.
func (t fbTable) integer[T int16 | int32 | int64](id int, def T) (T, error) {
	pos, ok, err := t.slot(id)
	if err != nil || !ok {
		return def, err
	}
	switch any(def).(type) {
	case int16:
		v, err := fbUint16(t.buf, pos)
		return T(int16(v)), err
	case int32:
		v, err := fbInt32(t.buf, pos)
		return T(v), err
	default:
		v, err := fbInt64(t.buf, pos)
		return T(v), err
	}
}

// indirect follows the offset stored at pos and returns where it points.
//
// This is the one rule the whole format is built on: an offset is counted from
// where the offset itself is, so that a piece of a buffer can be written once
// and pointed at from anywhere later.
func fbIndirect(buf []byte, pos int) (int, error) {
	off, err := fbUint32(buf, pos)
	if err != nil {
		return 0, err
	}
	to := pos + int(off)
	if to < 0 || to >= len(buf) {
		return 0, fmt.Errorf("ipc: %w: an offset at %d points to %d, outside a buffer of %d bytes",
			ErrMessage, pos, to, len(buf))
	}
	return to, nil
}

// table reads a child table. The second result says whether the field was
// there at all, since a table that is not there is not an error.
func (t fbTable) table(id int) (fbTable, bool, error) {
	pos, ok, err := t.slot(id)
	if err != nil || !ok {
		return fbTable{}, false, err
	}
	to, err := fbIndirect(t.buf, pos)
	if err != nil {
		return fbTable{}, false, err
	}
	child, err := fbTableAt(t.buf, to)
	return child, err == nil, err
}

// str reads a string field. An absent string is the empty string, which is
// what every reader of this format does and what Arrow's own defaults assume.
func (t fbTable) str(id int) (string, error) {
	pos, ok, err := t.slot(id)
	if err != nil || !ok {
		return "", err
	}
	b, err := fbBytesAt(t.buf, pos)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// vector reads a vector field and returns where its elements start.
func (t fbTable) vector(id int) (fbVector, bool, error) {
	pos, ok, err := t.slot(id)
	if err != nil || !ok {
		return fbVector{}, false, err
	}
	to, err := fbIndirect(t.buf, pos)
	if err != nil {
		return fbVector{}, false, err
	}
	n, err := fbUint32(t.buf, to)
	if err != nil {
		return fbVector{}, false, err
	}

	// The length is checked against the buffer here rather than at each read,
	// because a caller allocates for it before reading anything. No element is
	// shorter than a byte, so a vector of more elements than there are bytes
	// left is a number somebody changed, and believing it is how a message of
	// twelve bytes asks for a slice of two billion.
	if int64(n) > int64(len(t.buf)-(to+4)) {
		return fbVector{}, false, fmt.Errorf("ipc: %w: a vector at %d says it has %d elements, with %d bytes left to hold them",
			ErrMessage, to, n, len(t.buf)-(to+4))
	}
	return fbVector{buf: t.buf, pos: to + 4, n: int(n)}, true, nil
}

// fbBytesAt reads the length prefixed bytes an offset at pos points to. A
// string and a vector of bytes are the same thing here, and the null byte a
// writer puts after a string is not part of it.
func fbBytesAt(buf []byte, pos int) ([]byte, error) {
	to, err := fbIndirect(buf, pos)
	if err != nil {
		return nil, err
	}
	n, err := fbUint32(buf, to)
	if err != nil {
		return nil, err
	}
	if int64(to)+4+int64(n) > int64(len(buf)) {
		return nil, fmt.Errorf("ipc: %w: %d bytes at %d run past the end of a buffer of %d",
			ErrMessage, n, to+4, len(buf))
	}
	return buf[to+4 : to+4+int(n)], nil
}

// len is how many elements a vector has.
func (v fbVector) len() int { return v.n }

// table reads element i of a vector of tables.
func (v fbVector) table(i int) (fbTable, error) {
	pos, err := v.elem(i, 4)
	if err != nil {
		return fbTable{}, err
	}
	to, err := fbIndirect(v.buf, pos)
	if err != nil {
		return fbTable{}, err
	}
	return fbTableAt(v.buf, to)
}

// span reads element i of a vector of the two int64 structs the record batch
// metadata is made of.
func (v fbVector) span(i int) (span, error) {
	pos, err := v.elem(i, fbNodeSize)
	if err != nil {
		return span{}, err
	}
	first, err := fbInt64(v.buf, pos)
	if err != nil {
		return span{}, err
	}
	second, err := fbInt64(v.buf, pos+8)
	return span{first: first, second: second}, err
}

// int64at reads element i of a vector of int64.
func (v fbVector) int64at(i int) (int64, error) {
	pos, err := v.elem(i, 8)
	if err != nil {
		return 0, err
	}
	return fbInt64(v.buf, pos)
}

// elem is where element i of a vector of fixed size elements starts. A vector
// of structs is read with this and the size of the struct, since a struct in
// FlatBuffers is bytes in place rather than something pointed at.
func (v fbVector) elem(i, size int) (int, error) {
	if i < 0 || i >= v.n {
		return 0, fmt.Errorf("ipc: %w: element %d of a vector of %d", ErrMessage, i, v.n)
	}
	pos := v.pos + i*size
	if pos < 0 || pos+size > len(v.buf) {
		return 0, fmt.Errorf("ipc: %w: element %d of a vector at %d ends at %d, outside a buffer of %d bytes",
			ErrMessage, i, v.pos, pos+size, len(v.buf))
	}
	return pos, nil
}

// The four reads that touch the buffer. Everything above goes through one of
// them, so the bounds check is in four places rather than forty.
func fbUint16(buf []byte, pos int) (uint16, error) {
	if pos < 0 || pos+2 > len(buf) {
		return 0, fbOutside(pos, 2, len(buf))
	}
	return binary.LittleEndian.Uint16(buf[pos:]), nil
}

func fbUint32(buf []byte, pos int) (uint32, error) {
	if pos < 0 || pos+4 > len(buf) {
		return 0, fbOutside(pos, 4, len(buf))
	}
	return binary.LittleEndian.Uint32(buf[pos:]), nil
}

func fbInt32(buf []byte, pos int) (int32, error) {
	v, err := fbUint32(buf, pos)
	return int32(v), err
}

func fbInt64(buf []byte, pos int) (int64, error) {
	if pos < 0 || pos+8 > len(buf) {
		return 0, fbOutside(pos, 8, len(buf))
	}
	return int64(binary.LittleEndian.Uint64(buf[pos:])), nil
}

func fbOutside(pos, n, size int) error {
	return fmt.Errorf("ipc: %w: %d bytes at %d, outside a buffer of %d bytes", ErrMessage, n, pos, size)
}
