package parquet

import (
	"encoding/binary"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// builder writes the compact protocol.
//
// The tests hand the reader bytes rather than cutting them out of a file, since
// the interesting inputs are the ones no writer produces: a field id that does
// not fit in the header, a list of four billion elements, a struct nested a
// hundred deep. This is the smallest thing that writes the ones that do.
type builder struct {
	b    []byte
	last int16
}

func (w *builder) raw(b ...byte) *builder {
	w.b = append(w.b, b...)
	return w
}

func (w *builder) uvarint(v uint64) *builder {
	for v >= 0x80 {
		w.b = append(w.b, byte(v)|0x80)
		v >>= 7
	}
	return w.raw(byte(v))
}

func (w *builder) varint(v int64) *builder {
	return w.uvarint(uint64(v<<1) ^ uint64(v>>63))
}

// field writes a field header, in the short form when the id is close enough to
// the last one and the long form when it is not, which is what a writer does.
func (w *builder) field(id int16, t thriftType) *builder {
	if delta := id - w.last; delta > 0 && delta <= 15 {
		w.raw(byte(delta)<<4 | byte(t))
	} else {
		w.raw(byte(t)).varint(int64(id))
	}
	w.last = id
	return w
}

// structure writes whatever f writes and the stop byte behind it, keeping the
// field ids of the enclosing struct the way the reader keeps them.
func (w *builder) structure(f func()) *builder {
	last := w.last
	w.last = 0
	f()
	w.raw(0)
	w.last = last
	return w
}

func (w *builder) binary(s string) *builder {
	return w.uvarint(uint64(len(s))).raw([]byte(s)...)
}

func (w *builder) double(f float64) *builder {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], math.Float64bits(f))
	return w.raw(b[:]...)
}

// list writes a list header, in the short form when the count fits in the four
// bits it has and the long form when it does not.
func (w *builder) list(n int, t thriftType) *builder {
	if n < 0x0f {
		return w.raw(byte(n)<<4 | byte(t))
	}
	return w.raw(0xf0 | byte(t)).uvarint(uint64(n))
}

func (w *builder) reader() *reader { return &reader{buf: w.b} }

func TestUvarint(t *testing.T) {
	tests := []struct {
		bytes []byte
		want  uint64
	}{
		{[]byte{0x00}, 0},
		{[]byte{0x01}, 1},
		{[]byte{0x7f}, 127},
		{[]byte{0x80, 0x01}, 128},
		{[]byte{0xff, 0x01}, 255},
		{[]byte{0xac, 0x02}, 300},
		{[]byte{0xff, 0xff, 0xff, 0xff, 0x0f}, math.MaxUint32},
		{[]byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01}, math.MaxUint64},
		// A value written in more bytes than it needs, which is legal and reads
		// the same way, since what ends it is the first byte without the top bit
		// set rather than the length.
		{[]byte{0x80, 0x80, 0x80, 0x00}, 0},
	}

	for _, tt := range tests {
		r := &reader{buf: tt.bytes}
		got, err := r.uvarint()
		if err != nil {
			t.Errorf("uvarint(% x) = %v", tt.bytes, err)
			continue
		}
		if got != tt.want {
			t.Errorf("uvarint(% x) = %d, want %d", tt.bytes, got, tt.want)
		}
		if r.left() != 0 {
			t.Errorf("uvarint(% x) left %d bytes unread", tt.bytes, r.left())
		}
	}
}

func TestUvarintRefused(t *testing.T) {
	tests := []struct {
		name  string
		bytes []byte
	}{
		{"nothing at all", nil},
		{"a value that never ends", []byte{0x80}},
		{"a value that stops partway", []byte{0x80, 0x80, 0x80}},
		{"more than ten bytes", []byte{
			0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01,
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &reader{buf: tt.bytes}
			if _, err := r.uvarint(); !errors.Is(err, ErrFormat) {
				t.Fatalf("uvarint(% x) = %v, want a format error", tt.bytes, err)
			}
		})
	}
}

// TestVarint checks the zigzag, which is what keeps a small negative number
// small: the sign goes in the low bit rather than the high one, so minus one is
// one byte rather than ten.
func TestVarint(t *testing.T) {
	tests := []struct {
		want  int64
		bytes []byte
	}{
		{0, []byte{0x00}},
		{-1, []byte{0x01}},
		{1, []byte{0x02}},
		{-2, []byte{0x03}},
		{2, []byte{0x04}},
		{-64, []byte{0x7f}},
		{64, []byte{0x80, 0x01}},
	}

	for _, tt := range tests {
		r := &reader{buf: tt.bytes}
		got, err := r.varint()
		if err != nil || got != tt.want {
			t.Errorf("varint(% x) = %d, %v, want %d", tt.bytes, got, err, tt.want)
		}
	}

	// The extremes and everything near zero, written and read back.
	for _, v := range []int64{
		math.MinInt64, math.MinInt32, -1 << 20, -1, 0, 1, 1 << 20, math.MaxInt32, math.MaxInt64,
	} {
		w := (&builder{}).varint(v)
		got, err := w.reader().varint()
		if err != nil || got != v {
			t.Errorf("%d written as % x reads back as %d, %v", v, w.b, got, err)
		}
	}
}

// TestHeader checks both spellings of a field header. The short one holds the
// distance from the previous field, and the long one turns up when that
// distance will not fit in the four bits it has, which is what a struct with a
// gap in its field numbers produces.
func TestHeader(t *testing.T) {
	want := []struct {
		id int16
		t  thriftType
	}{
		{1, thriftInt32},
		{2, thriftBinary},
		{100, thriftInt64},
		{101, thriftTrue},
		{4, thriftList},
		{32767, thriftStruct},
	}

	w := &builder{}
	for _, f := range want {
		w.field(f.id, f.t)
	}
	w.raw(0)

	r := w.reader()
	for _, f := range want {
		id, typ, err := r.header()
		if err != nil {
			t.Fatalf("header: %v", err)
		}
		if id != f.id || typ != f.t {
			t.Fatalf("header = field %d of type %s, want field %d of type %s", id, typ, f.id, f.t)
		}
	}
	if _, typ, err := r.header(); err != nil || typ != thriftStop {
		t.Fatalf("the last header = %s, %v, want the stop type", typ, err)
	}
}

func TestHeaderRefused(t *testing.T) {
	tests := []struct {
		name  string
		bytes []byte
	}{
		{"nothing at all", nil},
		{"a long form with no id behind it", []byte{0x05}},
		{"a negative field id", (&builder{}).raw(0x05).varint(-1).b},
		{"a field id past what an int16 holds", (&builder{}).raw(0x05).varint(40000).b},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &reader{buf: tt.bytes}
			if _, _, err := r.header(); !errors.Is(err, ErrFormat) {
				t.Fatalf("header(% x) = %v, want a format error", tt.bytes, err)
			}
		})
	}
}

// TestFieldsNested checks that reading a struct inside a struct puts the field
// ids back the way it found them.
//
// They are written as distances rather than as numbers, so a reader that
// forgets where it was reads the next field of the outer struct as the wrong
// field rather than as a bad one. That is the sort of bug that gives a file
// which reads without complaint and comes out holding somebody else's values.
func TestFieldsNested(t *testing.T) {
	w := &builder{}
	w.structure(func() {
		w.field(3, thriftStruct)
		w.structure(func() {
			w.field(9, thriftInt32).varint(7)
			w.field(10, thriftInt32).varint(8)
		})
		w.field(4, thriftInt32).varint(9)
	})

	var outer, inner []int16
	r := w.reader()
	err := r.fields(func(id int16, t thriftType) error {
		outer = append(outer, id)
		if t != thriftStruct {
			_, err := r.int32(t)
			return err
		}
		return r.fields(func(id int16, t thriftType) error {
			inner = append(inner, id)
			_, err := r.int32(t)
			return err
		})
	})
	if err != nil {
		t.Fatalf("fields: %v", err)
	}

	if len(outer) != 2 || outer[0] != 3 || outer[1] != 4 {
		t.Errorf("the outer struct held fields %v, want 3 and 4", outer)
	}
	if len(inner) != 2 || inner[0] != 9 || inner[1] != 10 {
		t.Errorf("the inner struct held fields %v, want 9 and 10", inner)
	}
	if r.left() != 0 {
		t.Errorf("%d bytes left after the struct, want none", r.left())
	}
	if r.depth != 0 {
		t.Errorf("the reader is %d structs deep afterwards, want none", r.depth)
	}
}

// TestFieldsDepth is a struct nested further than any real one, which is what a
// file built to run a reader out of stack looks like. Reading it has to stop
// rather than recurse as far as it is told to.
func TestFieldsDepth(t *testing.T) {
	w := &builder{}
	for range 4 * maxDepth {
		w.raw(0x1c) // one field along, of type struct.
	}
	for range 4 * maxDepth {
		w.raw(0)
	}

	r := w.reader()
	var read func(id int16, t thriftType) error
	read = func(_ int16, t thriftType) error {
		if t == thriftStruct {
			return r.fields(read)
		}
		return r.skip(t)
	}
	if err := r.fields(read); !errors.Is(err, ErrFormat) {
		t.Fatalf("reading a struct %d deep = %v, want a format error", 4*maxDepth, err)
	}

	// skip is the other way in, and it is the one a real file takes, since a
	// field this package does not know about is skipped whatever is inside it.
	if err := w.reader().skip(thriftStruct); !errors.Is(err, ErrFormat) {
		t.Fatalf("skipping a struct %d deep = %v, want a format error", 4*maxDepth, err)
	}
}

// TestFieldsStops checks that a struct which never ends is refused rather than
// read to the end of the buffer and past it.
func TestFieldsStops(t *testing.T) {
	w := &builder{}
	w.field(1, thriftInt32).varint(1)
	w.field(2, thriftInt32).varint(2)

	r := w.reader()
	err := r.fields(func(_ int16, t thriftType) error {
		_, err := r.int32(t)
		return err
	})
	if !errors.Is(err, ErrFormat) {
		t.Fatalf("a struct with no stop byte = %v, want a format error", err)
	}
}

func TestListHeader(t *testing.T) {
	tests := []struct {
		name string
		n    int
		elem thriftType
	}{
		{"empty", 0, thriftInt32},
		{"the largest short count", 14, thriftInt8},
		{"the smallest long count", 15, thriftInt8},
		{"a long count", 200, thriftInt8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// The elements themselves, since the count is checked against what
			// is left and a header with nothing behind it is a lie.
			w := (&builder{}).list(tt.n, tt.elem)
			for range tt.n {
				w.raw(0)
			}

			n, elem, err := w.reader().listHeader(thriftList)
			if err != nil {
				t.Fatalf("listHeader: %v", err)
			}
			if n != tt.n || elem != tt.elem {
				t.Fatalf("listHeader = %d elements of type %s, want %d of type %s",
					n, elem, tt.n, tt.elem)
			}
		})
	}

	// A set is a list with a different name on it and is read the same way.
	n, elem, err := (&builder{}).list(0, thriftBinary).reader().listHeader(thriftSet)
	if err != nil || n != 0 || elem != thriftBinary {
		t.Errorf("listHeader of a set = %d, %s, %v, want an empty list of binary", n, elem, err)
	}
}

// TestListHeaderRefused covers the count that is a claim rather than a fact. A
// list saying it holds four billion elements is eight bytes of input, and a
// reader that believes it has been talked into allocating four billion
// elements' worth of memory by them.
func TestListHeaderRefused(t *testing.T) {
	tests := []struct {
		name  string
		bytes []byte
		as    thriftType
	}{
		{"nothing at all", nil, thriftList},
		{"a long count with no count behind it", []byte{0xf3}, thriftList},
		{"more elements than there are bytes", (&builder{}).list(1<<20, thriftInt8).b, thriftList},
		{"more elements than there are bytes anywhere", (&builder{}).list(1<<40, thriftInt8).b, thriftList},
		{"a list written as a struct", (&builder{}).list(0, thriftInt8).b, thriftStruct},
		{"a list written as a map", (&builder{}).list(0, thriftInt8).b, thriftMap},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &reader{buf: tt.bytes}
			if _, _, err := r.listHeader(tt.as); !errors.Is(err, ErrFormat) {
				t.Fatalf("listHeader(% x) = %v, want a format error", tt.bytes, err)
			}
		})
	}
}

// TestSkip reads past a value of every type there is, which is what happens to
// every field of a file written by a newer parquet than this one. What is being
// checked is that skipping a value lands on the byte behind it, since a skip
// that stops one byte short reads the rest of the footer as nonsense.
func TestSkip(t *testing.T) {
	tests := []struct {
		name  string
		typ   thriftType
		write func(*builder)
	}{
		{"a true", thriftTrue, func(*builder) {}},
		{"a false", thriftFalse, func(*builder) {}},
		{"an int8", thriftInt8, func(w *builder) { w.raw(0xff) }},
		{"an int16", thriftInt16, func(w *builder) { w.varint(-1000) }},
		{"an int32", thriftInt32, func(w *builder) { w.varint(math.MaxInt32) }},
		{"an int64", thriftInt64, func(w *builder) { w.varint(math.MinInt64) }},
		{"a double", thriftDouble, func(w *builder) { w.double(1.5) }},
		{"an empty string", thriftBinary, func(w *builder) { w.binary("") }},
		{"a string", thriftBinary, func(w *builder) { w.binary("parquet") }},
		{"an empty list", thriftList, func(w *builder) { w.list(0, thriftInt32) }},
		{"a list of ints", thriftList, func(w *builder) {
			w.list(3, thriftInt32).varint(1).varint(2).varint(3)
		}},
		{"a list of strings", thriftList, func(w *builder) {
			w.list(2, thriftBinary).binary("one").binary("two")
		}},
		{"a set", thriftSet, func(w *builder) { w.list(2, thriftInt8).raw(1, 2) }},
		{"an empty map", thriftMap, func(w *builder) { w.uvarint(0) }},
		{"a map", thriftMap, func(w *builder) {
			w.uvarint(2).raw(byte(thriftBinary)<<4 | byte(thriftInt32))
			w.binary("a").varint(1).binary("b").varint(2)
		}},
		{"an empty struct", thriftStruct, func(w *builder) { w.structure(func() {}) }},
		{"a struct", thriftStruct, func(w *builder) {
			w.structure(func() {
				w.field(1, thriftBinary).binary("name")
				w.field(4, thriftList).list(2, thriftStruct)
				w.structure(func() { w.field(1, thriftTrue) })
				w.structure(func() { w.field(1, thriftFalse) })
			})
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := &builder{}
			tt.write(w)
			n := len(w.b)
			w.binary("behind it")

			r := w.reader()
			if err := r.skip(tt.typ); err != nil {
				t.Fatalf("skip(%s): %v", tt.typ, err)
			}
			if r.pos != n {
				t.Fatalf("skip(%s) stopped at %d of %d bytes", tt.typ, r.pos, n)
			}
			// The value behind it reads, which is the thing a skip is for.
			if s, err := r.text(thriftBinary); err != nil || s != "behind it" {
				t.Fatalf("after skip(%s) the next value is %q, %v", tt.typ, s, err)
			}
			if r.depth != 0 {
				t.Fatalf("the reader is %d values deep afterwards, want none", r.depth)
			}
		})
	}
}

func TestSkipRefused(t *testing.T) {
	tests := []struct {
		name  string
		typ   thriftType
		bytes []byte
	}{
		{"a type this protocol does not have", thriftType(13), nil},
		{"a type past the four bits it has", thriftType(200), nil},
		{"the stop type on its own", thriftStop, nil},
		{"an int8 with no byte behind it", thriftInt8, nil},
		{"a double with seven bytes behind it", thriftDouble, make([]byte, 7)},
		{"a string longer than what is left", thriftBinary, (&builder{}).uvarint(1 << 30).b},
		{"a list that stops partway", thriftList, (&builder{}).list(3, thriftBinary).binary("a").b},
		{"a map whose size stops partway", thriftMap, []byte{0x80}},
		{"a map with no types behind its size", thriftMap, (&builder{}).uvarint(2).b},
		{"a map of more entries than bytes", thriftMap, (&builder{}).
			uvarint(1 << 30).raw(byte(thriftBinary)<<4 | byte(thriftInt32)).b},
		{"a map whose value stops partway", thriftMap, (&builder{}).
			uvarint(1).raw(byte(thriftBinary)<<4 | byte(thriftInt32)).binary("a").raw(0x80).b},
		{"a map that stops partway", thriftMap, (&builder{}).
			uvarint(2).raw(byte(thriftBinary)<<4 | byte(thriftInt32)).binary("a").varint(1).b},
		{"a struct that never stops", thriftStruct, (&builder{}).raw(0x15, 0x02).b},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &reader{buf: tt.bytes}
			if err := r.skip(tt.typ); !errors.Is(err, ErrFormat) {
				t.Fatalf("skip(%s) of % x = %v, want a format error", tt.typ, tt.bytes, err)
			}
		})
	}
}

// TestInteger checks the widths. They are all varints on the wire and the width
// only says what the writer called it, so what is left to check is that a value
// too large for the width a field is read at is refused rather than truncated
// into something that looks like a number.
func TestInteger(t *testing.T) {
	widths := []thriftType{thriftInt16, thriftInt32, thriftInt64}

	for _, width := range widths {
		for _, v := range []int64{0, 1, -1, 127, -128, 32767, -32768} {
			w := (&builder{}).varint(v)
			got, err := w.reader().integer(width)
			if err != nil || got != v {
				t.Errorf("integer(%s) of %d = %d, %v", width, v, got, err)
			}
		}
	}

	// An int8 is the one width that is a byte rather than a varint.
	for _, v := range []int8{0, 1, -1, 127, -128} {
		r := &reader{buf: []byte{byte(v)}}
		got, err := r.int8(thriftInt8)
		if err != nil || got != v {
			t.Errorf("int8 of % x = %d, %v, want %d", byte(v), got, err, v)
		}
	}

	narrowing := []struct {
		name string
		v    int64
		read func(*reader) error
	}{
		{"an int8 given 128", 128, func(r *reader) error { _, err := r.int8(thriftInt32); return err }},
		{"an int8 given -129", -129, func(r *reader) error { _, err := r.int8(thriftInt32); return err }},
		{"an int16 given 32768", 32768, func(r *reader) error { _, err := r.int16(thriftInt32); return err }},
		{"an int16 given -32769", -32769, func(r *reader) error { _, err := r.int16(thriftInt32); return err }},
		{"an int32 given 1<<31", 1 << 31, func(r *reader) error { _, err := r.int32(thriftInt64); return err }},
		{"an int32 given -1<<31 - 1", -1<<31 - 1, func(r *reader) error { _, err := r.int32(thriftInt64); return err }},
	}
	for _, tt := range narrowing {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.read((&builder{}).varint(tt.v).reader()); !errors.Is(err, ErrFormat) {
				t.Fatalf("reading %d = %v, want a format error", tt.v, err)
			}
		})
	}

	// The widths that hold their bound read it rather than refusing it.
	edges := []struct {
		v    int64
		read func(*reader) error
	}{
		{math.MaxInt32, func(r *reader) error { _, err := r.int32(thriftInt64); return err }},
		{math.MinInt32, func(r *reader) error { _, err := r.int32(thriftInt64); return err }},
		{math.MaxInt16, func(r *reader) error { _, err := r.int16(thriftInt32); return err }},
		{math.MinInt16, func(r *reader) error { _, err := r.int16(thriftInt32); return err }},
	}
	for _, tt := range edges {
		if err := tt.read((&builder{}).varint(tt.v).reader()); err != nil {
			t.Errorf("reading %d = %v, want it read", tt.v, err)
		}
	}

	for _, typ := range []thriftType{thriftBinary, thriftDouble, thriftList, thriftTrue} {
		r := &reader{buf: []byte{0, 0, 0, 0, 0, 0, 0, 0}}
		if _, err := r.integer(typ); !errors.Is(err, ErrFormat) {
			t.Errorf("integer(%s) = %v, want a format error", typ, err)
		}
	}
}

// TestBoolean checks the one type whose value is its own header. A bool field
// has no bytes behind it, which is why the protocol has two types for it.
func TestBoolean(t *testing.T) {
	r := &reader{}
	if v, err := r.boolean(thriftTrue); err != nil || !v {
		t.Errorf("boolean(true) = %v, %v, want true", v, err)
	}
	if v, err := r.boolean(thriftFalse); err != nil || v {
		t.Errorf("boolean(false) = %v, %v, want false", v, err)
	}
	if _, err := r.boolean(thriftInt32); !errors.Is(err, ErrFormat) {
		t.Errorf("boolean(int32) = %v, want a format error", err)
	}
	if r.pos != 0 {
		t.Errorf("reading three bools moved %d bytes, want none", r.pos)
	}
}

// TestBytes checks that a binary value points into the footer rather than
// copying out of it, and that a string does the opposite. The statistics of a
// thousand column chunks cost the bytes they were read from and nothing more,
// and a column name a caller keeps does not hold the whole footer alive.
func TestBytes(t *testing.T) {
	w := (&builder{}).binary("GB").binary("JP")
	r := w.reader()

	b, err := r.bytes(thriftBinary)
	if err != nil || string(b) != "GB" {
		t.Fatalf("bytes = %q, %v, want GB", b, err)
	}
	if len(b) > 0 && &b[0] != &r.buf[r.pos-len(b)] {
		t.Error("bytes copied the value, want a view of the buffer it was read from")
	}

	s, err := r.text(thriftBinary)
	if err != nil || s != "JP" {
		t.Fatalf("text = %q, %v, want JP", s, err)
	}
	// The buffer is damaged where the string was read from. A string that
	// pointed into it would change with it, which is what unsafe.String would
	// give and what a caller keeping a column name would then be holding.
	r.buf[r.pos-1] = 'X'
	if s != "JP" {
		t.Errorf("the string changed to %q when the buffer behind it did", s)
	}
}

func TestBytesRefused(t *testing.T) {
	tests := []struct {
		name  string
		typ   thriftType
		bytes []byte
	}{
		{"a string written as an int", thriftInt32, []byte{0x02}},
		{"a string written as a list", thriftList, []byte{0x02}},
		{"a length with nothing behind it", thriftBinary, nil},
		{"a length longer than what is left", thriftBinary, []byte{0x08, 'a', 'b'}},
		{"a length no file could hold", thriftBinary, (&builder{}).uvarint(math.MaxUint64).b},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &reader{buf: tt.bytes}
			if _, err := r.bytes(tt.typ); !errors.Is(err, ErrFormat) {
				t.Fatalf("bytes(%s) of % x = %v, want a format error", tt.typ, tt.bytes, err)
			}
		})
	}
}

// pair is a struct of one field, which is enough to read a list of structs
// with. The parquet structures are read the same way and are longer.
type pair struct {
	id   int32
	name string
}

func (p *pair) read(r *reader) error {
	return r.fields(func(id int16, t thriftType) (err error) {
		switch id {
		case 1:
			p.id, err = r.int32(t)
		case 2:
			p.name, err = r.text(t)
		default:
			err = r.skip(t)
		}
		return err
	})
}

func TestStructs(t *testing.T) {
	w := (&builder{}).list(2, thriftStruct)
	w.structure(func() {
		w.field(1, thriftInt32).varint(7)
		w.field(2, thriftBinary).binary("seven")
		// A field this reader does not know about, which is what a file written
		// by a newer parquet holds and which has to be skipped rather than
		// refused.
		w.field(9, thriftList).list(2, thriftBinary).binary("a").binary("b")
	})
	w.structure(func() {
		w.field(2, thriftBinary).binary("eight")
	})

	got, err := structs(w.reader(), thriftList, (*pair).read)
	if err != nil {
		t.Fatalf("structs: %v", err)
	}
	want := []pair{{7, "seven"}, {0, "eight"}}
	if len(got) != len(want) {
		t.Fatalf("structs read %d of them, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("struct %d is %+v, want %+v", i, got[i], want[i])
		}
	}

	empty, err := structs((&builder{}).list(0, thriftStop).reader(), thriftList, (*pair).read)
	if err != nil || len(empty) != 0 {
		t.Errorf("an empty list read as %v, %v, want nothing", empty, err)
	}

	// A list of the wrong thing, which is a footer that disagrees with itself
	// rather than one that stops early.
	w = (&builder{}).list(2, thriftBinary).binary("a").binary("b")
	if _, err := structs(w.reader(), thriftList, (*pair).read); !errors.Is(err, ErrFormat) {
		t.Errorf("a list of strings read as structs = %v, want a format error", err)
	}
}

func TestTexts(t *testing.T) {
	w := (&builder{}).list(3, thriftBinary).binary("one").binary("").binary("three")
	got, err := texts(w.reader(), thriftList)
	if err != nil {
		t.Fatalf("texts: %v", err)
	}
	want := []string{"one", "", "three"}
	if len(got) != len(want) {
		t.Fatalf("texts read %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("string %d is %q, want %q", i, got[i], want[i])
		}
	}

	w = (&builder{}).list(2, thriftInt32).varint(1).varint(2)
	if _, err := texts(w.reader(), thriftList); !errors.Is(err, ErrFormat) {
		t.Errorf("a list of ints read as strings = %v, want a format error", err)
	}
	w = (&builder{}).list(3, thriftBinary).binary("one")
	if _, err := texts(w.reader(), thriftList); !errors.Is(err, ErrFormat) {
		t.Errorf("a list that stops partway = %v, want a format error", err)
	}
}

func TestEnums(t *testing.T) {
	w := (&builder{}).list(3, thriftInt32).varint(int64(Plain)).
		varint(int64(RLEDictionary)).varint(int64(ByteStreamSplit))

	got, err := enums[Encoding](w.reader(), thriftList)
	if err != nil {
		t.Fatalf("enums: %v", err)
	}
	want := []Encoding{Plain, RLEDictionary, ByteStreamSplit}
	if len(got) != len(want) {
		t.Fatalf("enums read %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("encoding %d is %s, want %s", i, got[i], want[i])
		}
	}

	// An enumeration this package has never heard of is read as the number it
	// is, since the format grows and a file outlives the reader that reads it.
	w = (&builder{}).list(1, thriftInt32).varint(4242)
	unknown, err := enums[Encoding](w.reader(), thriftList)
	if err != nil || len(unknown) != 1 || unknown[0] != 4242 {
		t.Fatalf("an unknown encoding read as %v, %v, want 4242", unknown, err)
	}
	if got := unknown[0].String(); got != "encoding 4242" {
		t.Errorf("an unknown encoding prints as %q, want its number", got)
	}

	w = (&builder{}).list(1, thriftBinary).binary("plain")
	if _, err := enums[Encoding](w.reader(), thriftList); !errors.Is(err, ErrFormat) {
		t.Errorf("a list of strings read as enums = %v, want a format error", err)
	}
}

// TestMetadataPrefixes reads every prefix of a real footer.
//
// A footer arrives as somebody else's bytes and half of one is the shape they
// come in when a copy went wrong or a write was interrupted. Every prefix but
// the whole thing has to be refused, since the stop byte of the outermost
// struct is the last byte there is, and none of them may read past the end of
// what they were given.
func TestMetadataPrefixes(t *testing.T) {
	footer := footerOf(t, "alltypes.parquet")

	for n := range len(footer) {
		var m Metadata
		if err := m.read(&reader{buf: footer[:n]}); err == nil {
			t.Fatalf("the first %d bytes of a %d byte footer read as a whole one", n, len(footer))
		}
	}

	var m Metadata
	if err := m.read(&reader{buf: footer}); err != nil {
		t.Fatalf("the whole footer: %v", err)
	}
	if len(m.Nodes) == 0 {
		t.Error("the footer read with no schema in it")
	}
}

// TestMetadataDamaged changes one byte of a real footer at a time. Most of them
// give a footer that still reads and holds something else, which is fine and is
// not what this is looking for. What it is looking for is a read past the end
// of the buffer or an allocation of what a length claims rather than what is
// there, and either of those is a panic rather than an error.
func TestMetadataDamaged(t *testing.T) {
	footer := footerOf(t, "chunks.parquet")
	damaged := make([]byte, len(footer))

	for i := range footer {
		for _, b := range []byte{0x00, 0x01, 0x7f, 0x80, 0xff} {
			copy(damaged, footer)
			damaged[i] = b

			var m Metadata
			// The error is the ordinary outcome and is not checked. A panic is
			// the failure, and it fails the test by being one.
			_ = m.read(&reader{buf: damaged})
		}
	}
}

// footerOf reads the footer bytes out of one of the files in testdata, without
// going through ReadMetadata, so that the Thrift reader can be handed them on
// their own.
func footerOf(t *testing.T, name string) []byte {
	t.Helper()

	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(b) < len(magic)+trailer {
		t.Fatalf("%s is %d bytes, too small to hold a footer", name, len(b))
	}

	n := int(binary.LittleEndian.Uint32(b[len(b)-trailer:]))
	at := len(b) - trailer - n
	if at < len(magic) {
		t.Fatalf("%s says its footer is %d bytes, which does not fit", name, n)
	}
	return b[at : len(b)-trailer]
}
