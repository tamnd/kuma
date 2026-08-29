package parquet

import (
	"math"
	"testing"
)

// The tests in here are the corners of the protocol that a parquet footer never
// reaches: a field id too far from the one before it to fit in a header, a field
// id that goes backwards, a list too long to say how long it is in the four bits
// it has. The writer handles all three because a writer that handled only what
// today's footer needs is one field number away from producing a file that
// nothing can read.
//
// The reader has a builder of its own in thrift_test.go and this does not
// replace it. A reader checked with the writer next door and a writer checked
// with the reader next door agree with each other and with nothing else, and
// what the format is worth depends on both of them agreeing with pyarrow. So the
// reader keeps an encoder written by hand for the inputs no writer produces, and
// the writer is checked against real files elsewhere.

// TestWriterHeaderLongForm writes fields whose ids do not step forward by one to
// fifteen, which is the only way a field header is two bytes rather than one.
func TestWriterHeaderLongForm(t *testing.T) {
	var w writer
	w.int32Field(1, 11)
	w.int32Field(30, 22)
	w.int32Field(2, 33)
	w.put(byte(thriftStop))

	got := map[int16]int32{}
	r := &reader{buf: w.buf}
	err := r.fields(func(id int16, kind thriftType) error {
		v, err := r.int32(kind)
		got[id] = v
		return err
	})
	if err != nil {
		t.Fatalf("reading back: %v", err)
	}

	want := map[int16]int32{1: 11, 30: 22, 2: 33}
	if len(got) != len(want) {
		t.Fatalf("read %d fields, want %d", len(got), len(want))
	}
	for id, v := range want {
		if got[id] != v {
			t.Errorf("field %d came back as %d, want %d", id, got[id], v)
		}
	}
}

// TestWriterListLongForm writes a list of more than fourteen elements, which is
// the point where the count stops fitting beside the element type.
func TestWriterListLongForm(t *testing.T) {
	for _, n := range []int{0, 1, 14, 15, 16, 300} {
		vals := make([]string, n)
		for i := range vals {
			vals[i] = "a value"
		}

		var w writer
		writeTexts(&w, 1, vals)
		w.put(byte(thriftStop))

		r := &reader{buf: w.buf}
		var got []string
		err := r.fields(func(_ int16, kind thriftType) error {
			var err error
			got, err = texts(r, kind)
			return err
		})
		if err != nil {
			t.Fatalf("reading back %d strings: %v", n, err)
		}
		if len(got) != n {
			t.Errorf("a list of %d came back with %d in it", n, len(got))
		}
	}
}

// TestWriterIntegers writes the four widths and reads them back, which is where
// the zigzag is checked: a negative number that came back positive would be a
// row count near the largest int64 rather than a field the file left out.
func TestWriterIntegers(t *testing.T) {
	var w writer
	w.int8Field(1, -128)
	w.int16Field(2, math.MinInt16)
	w.int32Field(3, math.MinInt32)
	w.int64Field(4, math.MinInt64)
	w.int64Field(5, math.MaxInt64)
	w.put(byte(thriftStop))

	want := map[int16]int64{
		1: -128, 2: math.MinInt16, 3: math.MinInt32, 4: math.MinInt64, 5: math.MaxInt64,
	}

	r := &reader{buf: w.buf}
	err := r.fields(func(id int16, kind thriftType) error {
		v, err := r.integer(kind)
		if err != nil {
			return err
		}
		if v != want[id] {
			t.Errorf("field %d came back as %d, want %d", id, v, want[id])
		}
		return nil
	})
	if err != nil {
		t.Fatalf("reading back: %v", err)
	}
}

// TestFooterTrailerTooLong checks that a footer with no room to say how long it
// is gets refused rather than written with a length that wrapped.
func TestFooterTrailerTooLong(t *testing.T) {
	// The largest footer that can say how long it is, kept as an int64 and
	// narrowed rather than written as a constant, since it does not fit in an
	// int on a thirty two bit platform and there is nothing to check there.
	most := int64(math.MaxUint32)
	if int64(int(most)) != most {
		t.Skip("an int on this platform cannot hold a footer that large")
	}

	if _, err := footerTrailer(int(most)); err != nil {
		t.Errorf("a footer of %d bytes: %v", most, err)
	}
	if _, err := footerTrailer(int(most) + 1); err == nil {
		t.Error("a footer of more than four gigabytes was accepted")
	}
}
