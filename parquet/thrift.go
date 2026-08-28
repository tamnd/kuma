package parquet

import "fmt"

// Reading the Thrift compact protocol.
//
// A parquet file describes itself in a Thrift structure at the end of it, so
// reading the format at all means reading Thrift. This is the compact protocol
// in the one direction and for the shapes the parquet metadata is made of,
// rather than a general implementation of Thrift.
//
// Everything is a field header followed by a value. The header is one byte
// holding how far the field id has moved since the last field of this struct
// and what type the value is, which is why reading a struct means saving the id
// the reader was last at and putting it back afterwards. A value is a varint, or
// eight bytes, or a length and that many bytes, or a list, or another struct.
//
// A field the reader does not know about is skipped rather than refused, which
// is the whole reason the format carries types on the wire. Files written by a
// newer parquet than this one read fine and lose only what they added.
//
// Every read here is bounds checked, and every length is checked against what is
// left of the buffer before anything is allocated for it. A footer is somebody
// else's bytes: a list header claiming four billion elements is a claim, and a
// reader that believes it has been talked into allocating four billion elements
// by eight bytes of input.

// thriftType is the wire type of a field or of a list element. A bool is two
// types rather than one, since a field that is a bool carries its value in the
// header and has no bytes of its own.
type thriftType byte

const (
	thriftStop   thriftType = 0
	thriftTrue   thriftType = 1
	thriftFalse  thriftType = 2
	thriftInt8   thriftType = 3
	thriftInt16  thriftType = 4
	thriftInt32  thriftType = 5
	thriftInt64  thriftType = 6
	thriftDouble thriftType = 7
	thriftBinary thriftType = 8
	thriftList   thriftType = 9
	thriftSet    thriftType = 10
	thriftMap    thriftType = 11
	thriftStruct thriftType = 12
)

var thriftNames = [...]string{
	thriftStop:   "stop",
	thriftTrue:   "true",
	thriftFalse:  "false",
	thriftInt8:   "int8",
	thriftInt16:  "int16",
	thriftInt32:  "int32",
	thriftInt64:  "int64",
	thriftDouble: "double",
	thriftBinary: "binary",
	thriftList:   listName,
	thriftSet:    "set",
	thriftMap:    mapName,
	thriftStruct: "struct",
}

// String returns the name the Thrift specification gives the type.
func (t thriftType) String() string {
	if int(t) >= len(thriftNames) {
		return fmt.Sprintf("type %d", byte(t))
	}
	return thriftNames[t]
}

// maxDepth is how far structs and lists may nest inside each other. The parquet
// metadata is five deep at its worst, so anything near this is a file built to
// run a reader out of stack rather than a file with a lot in it.
const maxDepth = 64

// reader is a position in the bytes of a Thrift structure.
//
// It holds the whole buffer rather than reading from an io.Reader, because a
// footer is read in one go and because every string in the result points into
// these bytes rather than copying out of them.
type reader struct {
	buf []byte
	pos int

	// last is the field id the previous header gave, which the next one is
	// written as a distance from. It belongs to the struct being read, so
	// fields saves it and puts it back.
	last int16

	// depth is how many structs and lists are open, which is the only bound on
	// how far a malformed file can push the stack.
	depth int
}

// left is how many bytes have not been read.
func (r *reader) left() int { return len(r.buf) - r.pos }

// next returns the next byte.
func (r *reader) next() (byte, error) {
	if r.pos >= len(r.buf) {
		return 0, fmt.Errorf("parquet: %w: the metadata ends in the middle of a value", ErrFormat)
	}
	b := r.buf[r.pos]
	r.pos++
	return b, nil
}

// uvarint reads an unsigned varint, seven bits to a byte, least significant
// first, with the top bit saying that another byte follows.
func (r *reader) uvarint() (uint64, error) {
	var v uint64
	for shift := 0; ; shift += 7 {
		if shift >= 64 {
			return 0, fmt.Errorf("parquet: %w: a varint of more than ten bytes", ErrFormat)
		}
		b, err := r.next()
		if err != nil {
			return 0, err
		}
		v |= uint64(b&0x7f) << shift
		if b&0x80 == 0 {
			return v, nil
		}
	}
}

// varint reads a signed varint, which is written zigzagged so that a small
// negative number is a small number of bytes.
func (r *reader) varint() (int64, error) {
	v, err := r.uvarint()
	if err != nil {
		return 0, err
	}
	return int64(v>>1) ^ -int64(v&1), nil
}

// header reads the header of the next field of a struct, returning the field id
// and the type of its value. The stop type ends the struct and has no id.
func (r *reader) header() (int16, thriftType, error) {
	b, err := r.next()
	if err != nil {
		return 0, thriftStop, err
	}
	t := thriftType(b & 0x0f)
	if t == thriftStop {
		return 0, thriftStop, nil
	}

	// The high nibble is how far the id has moved since the last field, and
	// zero means it did not fit in four bits and follows as a varint.
	if delta := int16(b >> 4); delta != 0 {
		r.last += delta
		return r.last, t, nil
	}

	id, err := r.varint()
	if err != nil {
		return 0, thriftStop, err
	}
	if id < 0 || id > 0x7fff {
		return 0, thriftStop, fmt.Errorf("parquet: %w: a field numbered %d", ErrFormat, id)
	}
	r.last = int16(id)
	return r.last, t, nil
}

// fields reads a struct, calling f with the id and type of every field in it.
//
// Every field f is given has to be consumed by it, which for a field it does
// not know about means calling skip. That is what lets a file written by a
// newer parquet than this one be read at all.
func (r *reader) fields(f func(id int16, t thriftType) error) error {
	r.depth++
	if r.depth > maxDepth {
		return fmt.Errorf("parquet: %w: the metadata nests more than %d deep", ErrFormat, maxDepth)
	}
	last := r.last
	r.last = 0

	for {
		id, t, err := r.header()
		if err != nil {
			return err
		}
		if t == thriftStop {
			break
		}
		if err := f(id, t); err != nil {
			return err
		}
	}

	r.last = last
	r.depth--
	return nil
}

// listHeader reads the header of a list, returning how many elements it has and
// what type they are.
//
// The count is checked against what is left of the buffer, since the shortest
// element of any type is one byte and a list claiming more elements than there
// are bytes cannot be read whatever it holds.
func (r *reader) listHeader(t thriftType) (int, thriftType, error) {
	if t != thriftList && t != thriftSet {
		return 0, thriftStop, fmt.Errorf("parquet: %w: a list written as a %s", ErrFormat, t)
	}
	b, err := r.next()
	if err != nil {
		return 0, thriftStop, err
	}

	elem := thriftType(b & 0x0f)
	n := int(b >> 4)
	if n == 0x0f {
		size, err := r.uvarint()
		if err != nil {
			return 0, thriftStop, err
		}
		if size > uint64(r.left()) {
			return 0, thriftStop, fmt.Errorf("parquet: %w: a list of %d elements with %d bytes left",
				ErrFormat, size, r.left())
		}
		n = int(size)
	}
	return n, elem, nil
}

// integer reads a field written as any of the four integer widths. They are all
// varints on the wire and the width only says what the writer called it, so a
// reader that wants an int32 and is handed an int16 has been handed an int32.
func (r *reader) integer(t thriftType) (int64, error) {
	switch t {
	case thriftInt8:
		b, err := r.next()
		return int64(int8(b)), err
	case thriftInt16, thriftInt32, thriftInt64:
		return r.varint()
	default:
		return 0, fmt.Errorf("parquet: %w: an integer written as a %s", ErrFormat, t)
	}
}

// int32 reads an integer field and checks that it fits in thirty two bits.
func (r *reader) int32(t thriftType) (int32, error) {
	v, err := r.integer(t)
	if err != nil {
		return 0, err
	}
	if v < -1<<31 || v > 1<<31-1 {
		return 0, fmt.Errorf("parquet: %w: %d does not fit in an int32", ErrFormat, v)
	}
	return int32(v), nil
}

// int16 reads an integer field and checks that it fits in sixteen bits.
func (r *reader) int16(t thriftType) (int16, error) {
	v, err := r.integer(t)
	if err != nil {
		return 0, err
	}
	if v < -1<<15 || v > 1<<15-1 {
		return 0, fmt.Errorf("parquet: %w: %d does not fit in an int16", ErrFormat, v)
	}
	return int16(v), nil
}

// int8 reads an integer field and checks that it fits in eight bits.
func (r *reader) int8(t thriftType) (int8, error) {
	v, err := r.integer(t)
	if err != nil {
		return 0, err
	}
	if v < -1<<7 || v > 1<<7-1 {
		return 0, fmt.Errorf("parquet: %w: %d does not fit in an int8", ErrFormat, v)
	}
	return int8(v), nil
}

// boolean reads a bool field, whose value is the type in its own header.
func (r *reader) boolean(t thriftType) (bool, error) {
	switch t {
	case thriftTrue:
		return true, nil
	case thriftFalse:
		return false, nil
	default:
		return false, fmt.Errorf("parquet: %w: a bool written as a %s", ErrFormat, t)
	}
}

// bytes reads a binary field, which is a length and that many bytes.
//
// What comes back points into the buffer rather than copying out of it, so the
// statistics of a thousand column chunks cost the bytes they were read from and
// nothing more.
func (r *reader) bytes(t thriftType) ([]byte, error) {
	if t != thriftBinary {
		return nil, fmt.Errorf("parquet: %w: a string written as a %s", ErrFormat, t)
	}
	n, err := r.uvarint()
	if err != nil {
		return nil, err
	}
	if n > uint64(r.left()) {
		return nil, fmt.Errorf("parquet: %w: a value of %d bytes with %d left", ErrFormat, n, r.left())
	}
	at := r.pos
	r.pos += int(n)
	return r.buf[at:r.pos], nil
}

// text reads a binary field as a string. This is the one place a value is
// copied, because a string a caller keeps should not hold the whole footer
// alive behind it.
func (r *reader) text(t thriftType) (string, error) {
	b, err := r.bytes(t)
	return string(b), err
}

// skip reads past a value of any type without looking at it.
func (r *reader) skip(t thriftType) error {
	r.depth++
	if r.depth > maxDepth {
		return fmt.Errorf("parquet: %w: the metadata nests more than %d deep", ErrFormat, maxDepth)
	}
	defer func() { r.depth-- }()

	switch t {
	case thriftTrue, thriftFalse:
		return nil
	case thriftInt8:
		_, err := r.next()
		return err
	case thriftInt16, thriftInt32, thriftInt64:
		_, err := r.uvarint()
		return err
	case thriftDouble:
		if r.left() < 8 {
			return fmt.Errorf("parquet: %w: a double with %d bytes left", ErrFormat, r.left())
		}
		r.pos += 8
		return nil
	case thriftBinary:
		_, err := r.bytes(t)
		return err
	case thriftList, thriftSet:
		return r.skipList(t)
	case thriftMap:
		return r.skipMap()
	case thriftStruct:
		return r.fields(func(_ int16, t thriftType) error { return r.skip(t) })
	default:
		return fmt.Errorf("parquet: %w: a value of %s", ErrFormat, t)
	}
}

// skipList reads past every element of a list.
func (r *reader) skipList(t thriftType) error {
	n, elem, err := r.listHeader(t)
	if err != nil {
		return err
	}
	for range n {
		if err := r.skip(elem); err != nil {
			return err
		}
	}
	return nil
}

// skipMap reads past every key and value of a map. An empty map is a size and
// nothing else, not even the types, which is the one place the protocol leaves
// something out.
func (r *reader) skipMap() error {
	n, err := r.uvarint()
	if err != nil {
		return err
	}
	if n == 0 {
		return nil
	}

	b, err := r.next()
	if err != nil {
		return err
	}
	if n > uint64(r.left()) {
		return fmt.Errorf("parquet: %w: a map of %d entries with %d bytes left", ErrFormat, n, r.left())
	}
	key, value := thriftType(b>>4), thriftType(b&0x0f)
	for range n {
		if err := r.skip(key); err != nil {
			return err
		}
		if err := r.skip(value); err != nil {
			return err
		}
	}
	return nil
}

// structs reads a list of structs, reading each of them with read.
//
// The slice is grown rather than allocated up front. The count has already been
// checked against the bytes that are left, which bounds it, but a struct is
// bigger than the one byte that bound gives it and a file should not be able to
// ask for a hundred megabytes with two bytes of input.
func structs[T any](r *reader, t thriftType, read func(*T, *reader) error) ([]T, error) {
	n, elem, err := r.listHeader(t)
	if err != nil {
		return nil, err
	}
	if n > 0 && elem != thriftStruct {
		return nil, fmt.Errorf("parquet: %w: a list of %s where a list of structs was written",
			ErrFormat, elem)
	}

	out := make([]T, 0, min(n, 64))
	for range n {
		var v T
		if err := read(&v, r); err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// texts reads a list of strings.
func texts(r *reader, t thriftType) ([]string, error) {
	n, elem, err := r.listHeader(t)
	if err != nil {
		return nil, err
	}
	if n > 0 && elem != thriftBinary {
		return nil, fmt.Errorf("parquet: %w: a list of %s where a list of strings was written",
			ErrFormat, elem)
	}

	out := make([]string, 0, min(n, 64))
	for range n {
		s, err := r.text(thriftBinary)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, nil
}

// enums reads a list of integers as whatever the format calls them. Every
// enumeration in parquet is an int32 on the wire, so the one function covers
// the encodings a column chunk used and anything else that turns up.
func enums[T ~int32](r *reader, t thriftType) ([]T, error) {
	n, elem, err := r.listHeader(t)
	if err != nil {
		return nil, err
	}

	out := make([]T, 0, min(n, 64))
	for range n {
		v, err := r.int32(elem)
		if err != nil {
			return nil, err
		}
		out = append(out, T(v))
	}
	return out, nil
}
