package parquet_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"slices"
	"testing"

	"github.com/tamnd/kuma/parquet"
)

// int96 writes a timestamp the way the deprecated type writes it, which is a
// count of nanoseconds into a day and then the Julian day it falls in.
func int96(nanos int64, day uint32) []byte {
	b := binary.LittleEndian.AppendUint64(nil, uint64(nanos))
	return binary.LittleEndian.AppendUint32(b, day)
}

// byteArrays writes values the way the plain encoding writes a byte array,
// which is four bytes of length in front of each of them.
func byteArrays(values ...string) []byte {
	var out []byte
	for _, v := range values {
		out = binary.LittleEndian.AppendUint32(out, uint32(len(v)))
		out = append(out, v...)
	}
	return out
}

// strings turns decoded byte arrays into something a test can compare, since
// what the decoder hands back points into the page and what a test knows is
// what the values say.
func strs(values [][]byte) []string {
	out := make([]string, len(values))
	for i, v := range values {
		out[i] = string(v)
	}
	return out
}

// valuesOf is the part of a page that is the values, with the levels in front
// of them taken off.
//
// Where the levels are depends on which version of the data page it is, which
// is the one thing about a page a decoder of values does not want to know. The
// second version puts them in bytes of their own at the front of the body and
// says in the header how long they are. The first version puts each run inside
// the body behind four bytes of its length. A dictionary page has no levels at
// all.
func valuesOf(t *testing.T, p parquet.Page, c parquet.Column) []byte {
	t.Helper()

	if p.Kind == parquet.DictionaryPage {
		return p.Data
	}
	if p.Kind == parquet.DataPageV2 {
		return p.Data[p.RepetitionLength+p.DefinitionLength:]
	}

	b := p.Data
	for _, level := range []int{c.MaxRepetition, c.MaxDefinition} {
		if level == 0 {
			continue
		}
		n := binary.LittleEndian.Uint32(b)
		b = b[4+n:]
	}
	return b
}

// dictionaryOf is the values of the dictionary page of a column, which is the
// one page in the format that is always plain.
func dictionaryOf(t *testing.T, name, column string) []byte {
	t.Helper()

	pages, c := pagesOf(t, name, column)
	if pages[0].Kind != parquet.DictionaryPage {
		t.Fatalf("the first page of %s is a %s, want a %s", column, pages[0].Kind, parquet.DictionaryPage)
	}
	return valuesOf(t, pages[0], c)
}

// TestPlainNumbers decodes each of the fixed width types out of bytes written
// by hand, so that the widths and the byte order are checked against the
// format's own description of them rather than against another reader.
func TestPlainNumbers(t *testing.T) {
	t.Run("int32", func(t *testing.T) {
		data := []byte{0x01, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x80}
		want := []int32{1, -1, math.MinInt32}

		got := make([]int32, len(want))
		n, err := parquet.NewPlainDecoder(data).Int32(got)
		if err != nil {
			t.Fatalf("Int32: %v", err)
		}
		if !slices.Equal(got[:n], want) {
			t.Errorf("got %v, want %v", got[:n], want)
		}
	})

	t.Run("int64", func(t *testing.T) {
		data := binary.LittleEndian.AppendUint64(nil, 1)
		data = binary.LittleEndian.AppendUint64(data, math.MaxUint64)
		data = binary.LittleEndian.AppendUint64(data, uint64(math.MaxInt64))
		want := []int64{1, -1, math.MaxInt64}

		got := make([]int64, len(want))
		n, err := parquet.NewPlainDecoder(data).Int64(got)
		if err != nil {
			t.Fatalf("Int64: %v", err)
		}
		if !slices.Equal(got[:n], want) {
			t.Errorf("got %v, want %v", got[:n], want)
		}
	})

	t.Run("float", func(t *testing.T) {
		want := []float32{1.5, -2.5, float32(math.Inf(1))}

		var data []byte
		for _, v := range want {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
		}

		got := make([]float32, len(want))
		n, err := parquet.NewPlainDecoder(data).Float(got)
		if err != nil {
			t.Fatalf("Float: %v", err)
		}
		if !slices.Equal(got[:n], want) {
			t.Errorf("got %v, want %v", got[:n], want)
		}
	})

	t.Run("double", func(t *testing.T) {
		want := []float64{1.25, -2.5, math.Inf(-1)}

		var data []byte
		for _, v := range want {
			data = binary.LittleEndian.AppendUint64(data, math.Float64bits(v))
		}

		got := make([]float64, len(want))
		n, err := parquet.NewPlainDecoder(data).Double(got)
		if err != nil {
			t.Fatalf("Double: %v", err)
		}
		if !slices.Equal(got[:n], want) {
			t.Errorf("got %v, want %v", got[:n], want)
		}
	})
}

// TestPlainBatches reads a run of values a few at a time, which is what a scan
// filling a fixed buffer does and is the case where the decoder has to pick up
// where it left off.
func TestPlainBatches(t *testing.T) {
	const values = 1000

	want := make([]int32, values)
	var data []byte
	for i := range want {
		want[i] = int32(i * 3)
		data = binary.LittleEndian.AppendUint32(data, uint32(want[i]))
	}

	for _, batch := range []int{1, 7, 64, values, values + 1} {
		d := parquet.NewPlainDecoder(data)
		buf := make([]int32, batch)

		var got []int32
		for {
			n, err := d.Int32(buf)
			got = append(got, buf[:n]...)
			if err != nil {
				if !errors.Is(err, io.EOF) {
					t.Fatalf("Int32: %v", err)
				}
				break
			}
		}
		if !slices.Equal(got, want) {
			t.Errorf("read %d at a time: got %d values, want %d", batch, len(got), len(want))
		}
	}
}

// TestPlainBoolean reads booleans, which are the one type the encoding writes
// smaller than a byte.
func TestPlainBoolean(t *testing.T) {
	// Ten values so that the run carries on into a second byte, and so that the
	// second byte has six bits in it that are not values.
	want := []bool{true, false, true, true, false, false, false, true, true, false}
	data := []byte{0b10001101, 0b00000001}

	for _, batch := range []int{1, 3, 8, 16} {
		d := parquet.NewPlainDecoder(data)
		buf := make([]bool, batch)

		var got []bool
		for len(got) < len(want) {
			n, err := d.Boolean(buf)
			if err != nil {
				t.Fatalf("Boolean: %v", err)
			}
			got = append(got, buf[:n]...)
		}
		if !slices.Equal(got[:len(want)], want) {
			t.Errorf("read %d at a time: got %v, want %v", batch, got[:len(want)], want)
		}
	}
}

// TestPlainByteArray reads byte arrays, which are the one type the encoding
// writes a length in front of.
func TestPlainByteArray(t *testing.T) {
	want := []string{"one", "", "three", "a rather longer one than the others"}
	d := parquet.NewPlainDecoder(byteArrays(want...))

	got := make([][]byte, len(want))
	n, err := d.ByteArray(got)
	if err != nil {
		t.Fatalf("ByteArray: %v", err)
	}
	if !slices.Equal(strs(got[:n]), want) {
		t.Errorf("got %q, want %q", strs(got[:n]), want)
	}
	if left := d.Left(); left != 0 {
		t.Errorf("after reading everything there are %d bytes left", left)
	}
	if _, err := d.ByteArray(got); !errors.Is(err, io.EOF) {
		t.Errorf("reading past the end: got %v, want %v", err, io.EOF)
	}
}

// TestPlainFixed reads fixed length byte arrays, which have their width in the
// schema rather than in front of each value.
func TestPlainFixed(t *testing.T) {
	want := []string{"abcd", "efgh", "ijkl"}
	d := parquet.NewPlainDecoder([]byte("abcdefghijkl"))

	got := make([][]byte, len(want))
	n, err := d.Fixed(got, 4)
	if err != nil {
		t.Fatalf("Fixed: %v", err)
	}
	if !slices.Equal(strs(got[:n]), want) {
		t.Errorf("got %q, want %q", strs(got[:n]), want)
	}

	if _, err := d.Fixed(got, 0); !errors.Is(err, parquet.ErrFormat) {
		t.Errorf("a width of nought: got %v, want %v", err, parquet.ErrFormat)
	}
}

// TestPlainInt96 reads the deprecated timestamp and checks that it comes back
// as nanoseconds since the epoch.
//
// Both of the parts have to be right for the answer to be, and the day is
// counted from a morning in 4713 BC, so a decoder that got the epoch wrong
// would be wrong by thousands of years rather than by a little.
func TestPlainInt96(t *testing.T) {
	cases := []struct {
		name  string
		nanos int64
		day   uint32
		want  int64
	}{
		{name: "the epoch itself", nanos: 0, day: 2440588, want: 0},
		{name: "noon on the first day", nanos: 12 * 3600 * 1e9, day: 2440588, want: 12 * 3600 * 1e9},
		{name: "the day before the epoch", nanos: 0, day: 2440587, want: -24 * 3600 * 1e9},
		{
			name:  "a moment in 2026",
			nanos: 43200123456000,
			day:   2461281,
			want:  1787918400123456000,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := make([]int64, 1)
			if _, err := parquet.NewPlainDecoder(int96(c.nanos, c.day)).Int96(got); err != nil {
				t.Fatalf("Int96: %v", err)
			}
			if got[0] != c.want {
				t.Errorf("got %d, want %d", got[0], c.want)
			}
		})
	}
}

// TestPlainInt96Refused is the timestamps that do not fit in the nanoseconds
// kuma counts in, which is a little under two hundred and ninety two years
// either side of 1970 and is a long way short of what twelve bytes can hold.
func TestPlainInt96Refused(t *testing.T) {
	cases := []struct {
		name  string
		nanos int64
		day   uint32
	}{
		{name: "a day too far ahead", nanos: 0, day: math.MaxUint32},
		{name: "a day too far back", nanos: 0, day: 0},
		{name: "the last day and too many nanoseconds into it", nanos: math.MaxInt64, day: 2547339},
		{name: "the first day and too few", nanos: math.MinInt64, day: 2333837},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := make([]int64, 1)
			_, err := parquet.NewPlainDecoder(int96(c.nanos, c.day)).Int96(got)
			if !errors.Is(err, parquet.ErrFormat) {
				t.Errorf("got %v, want %v", err, parquet.ErrFormat)
			}
		})
	}
}

// TestPlainShort is data that ends in the middle of a value.
//
// A page says how many values it holds and holds no part of one, so a tail too
// short to be a value is a page somebody wrote wrong, and a reader that stopped
// quietly on it would hand back a column with a hole in it and say nothing.
func TestPlainShort(t *testing.T) {
	t.Run("a value that runs off the end", func(t *testing.T) {
		d := parquet.NewPlainDecoder([]byte{1, 0, 0, 0, 2, 0})

		got := make([]int32, 2)
		n, err := d.Int32(got)
		if err != nil {
			t.Fatalf("the whole value: %v", err)
		}
		if n != 1 || got[0] != 1 {
			t.Fatalf("got %d values, %v", n, got[:n])
		}
		if _, err := d.Int32(got); !errors.Is(err, parquet.ErrFormat) {
			t.Errorf("the part of one: got %v, want %v", err, parquet.ErrFormat)
		}
	})

	t.Run("a byte array with no length", func(t *testing.T) {
		d := parquet.NewPlainDecoder([]byte{3, 0, 0, 0, 'o', 'n', 'e', 0, 0})

		got := make([][]byte, 2)
		n, err := d.ByteArray(got)
		if !errors.Is(err, parquet.ErrFormat) {
			t.Fatalf("got %v, want %v", err, parquet.ErrFormat)
		}
		if n != 1 || string(got[0]) != "one" {
			t.Errorf("got %d values, %q", n, strs(got[:n]))
		}
	})

	t.Run("a byte array longer than what is left", func(t *testing.T) {
		got := make([][]byte, 1)
		d := parquet.NewPlainDecoder([]byte{9, 0, 0, 0, 'o', 'n', 'e'})
		if _, err := d.ByteArray(got); !errors.Is(err, parquet.ErrFormat) {
			t.Errorf("got %v, want %v", err, parquet.ErrFormat)
		}
	})

	t.Run("a fixed value that runs off the end", func(t *testing.T) {
		got := make([][]byte, 1)
		d := parquet.NewPlainDecoder([]byte("abc"))
		if _, err := d.Fixed(got, 4); !errors.Is(err, parquet.ErrFormat) {
			t.Errorf("got %v, want %v", err, parquet.ErrFormat)
		}
	})
}

// TestPlainEnd is what the decoder does at the end of the data and when it is
// asked for nothing, both of which a loop over pages runs into.
func TestPlainEnd(t *testing.T) {
	d := parquet.NewPlainDecoder(nil)

	if n, err := d.Int32(nil); n != 0 || err != nil {
		t.Errorf("asked for no values: got %d, %v", n, err)
	}
	if n, err := d.Boolean(nil); n != 0 || err != nil {
		t.Errorf("asked for no booleans: got %d, %v", n, err)
	}
	if n, err := d.ByteArray(nil); n != 0 || err != nil {
		t.Errorf("asked for no byte arrays: got %d, %v", n, err)
	}

	for _, c := range []struct {
		name string
		read func() error
	}{
		{name: "int32", read: func() error { _, err := d.Int32(make([]int32, 1)); return err }},
		{name: "int64", read: func() error { _, err := d.Int64(make([]int64, 1)); return err }},
		{name: "int96", read: func() error { _, err := d.Int96(make([]int64, 1)); return err }},
		{name: "float", read: func() error { _, err := d.Float(make([]float32, 1)); return err }},
		{name: "double", read: func() error { _, err := d.Double(make([]float64, 1)); return err }},
		{name: "boolean", read: func() error { _, err := d.Boolean(make([]bool, 1)); return err }},
		{name: "byte array", read: func() error { _, err := d.ByteArray(make([][]byte, 1)); return err }},
		{name: "fixed", read: func() error { _, err := d.Fixed(make([][]byte, 1), 4); return err }},
	} {
		if err := c.read(); !errors.Is(err, io.EOF) {
			t.Errorf("%s at the end: got %v, want %v", c.name, err, io.EOF)
		}
	}
}

// TestPlainReset points a decoder at other bytes, which is what a scan reading
// a thousand pages of one column does instead of allocating a decoder for each
// of them.
//
// The reset happens in the middle of a run of booleans, since that is the one
// place the decoder is holding something other than a byte position.
func TestPlainReset(t *testing.T) {
	d := parquet.NewPlainDecoder([]byte{0b00000011, 0xff})

	got := make([]bool, 2)
	if _, err := d.Boolean(got); err != nil {
		t.Fatalf("Boolean: %v", err)
	}
	if want := []bool{true, true}; !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	if left := d.Left(); left != 1 {
		t.Errorf("in the middle of a byte there are %d bytes left, want 1", left)
	}

	d.Reset([]byte{7, 0, 0, 0})
	if left := d.Left(); left != 4 {
		t.Errorf("after the reset there are %d bytes left, want 4", left)
	}

	values := make([]int32, 1)
	if _, err := d.Int32(values); err != nil {
		t.Fatalf("Int32: %v", err)
	}
	if values[0] != 7 {
		t.Errorf("got %d, want 7", values[0])
	}
}

// TestPlainAligned reads a byte aligned value after a run of booleans that
// stopped in the middle of a byte.
//
// No real page does this, because a page is all one type. It is here because
// the decoder is told what to read by its caller rather than by the data, and a
// caller that reads the wrong thing should get bytes from where it asked rather
// than bits from the middle of a byte.
func TestPlainAligned(t *testing.T) {
	data := append([]byte{0b00000001}, 9, 0, 0, 0)
	d := parquet.NewPlainDecoder(data)

	flags := make([]bool, 1)
	if _, err := d.Boolean(flags); err != nil {
		t.Fatalf("Boolean: %v", err)
	}

	got := make([]int32, 1)
	if _, err := d.Int32(got); err != nil {
		t.Fatalf("Int32: %v", err)
	}
	if got[0] != 9 {
		t.Errorf("got %d, want 9", got[0])
	}
}

// TestPlainRealDictionaries decodes the dictionary pages of a real file.
//
// A dictionary page is plain whatever the data pages that lean on it are, so
// this is a column of every type the format writes, decoded out of bytes
// pyarrow wrote, and the values are the ones the script next to the file put
// in.
func TestPlainRealDictionaries(t *testing.T) {
	const file = "alltypes.parquet"

	t.Run("small", func(t *testing.T) {
		got := make([]int32, 2)
		n, err := parquet.NewPlainDecoder(dictionaryOf(t, file, "small")).Int32(got)
		if err != nil {
			t.Fatalf("Int32: %v", err)
		}
		if want := []int32{1, -3}; !slices.Equal(got[:n], want) {
			t.Errorf("got %v, want %v", got[:n], want)
		}
	})

	t.Run("total", func(t *testing.T) {
		got := make([]int64, 3)
		n, err := parquet.NewPlainDecoder(dictionaryOf(t, file, "total")).Int64(got)
		if err != nil {
			t.Fatalf("Int64: %v", err)
		}
		if want := []int64{100, 200, 300}; !slices.Equal(got[:n], want) {
			t.Errorf("got %v, want %v", got[:n], want)
		}
	})

	t.Run("ratio", func(t *testing.T) {
		got := make([]float32, 2)
		n, err := parquet.NewPlainDecoder(dictionaryOf(t, file, "ratio")).Float(got)
		if err != nil {
			t.Fatalf("Float: %v", err)
		}
		if want := []float32{1.5, 2.5}; !slices.Equal(got[:n], want) {
			t.Errorf("got %v, want %v", got[:n], want)
		}
	})

	t.Run("weight", func(t *testing.T) {
		got := make([]float64, 3)
		n, err := parquet.NewPlainDecoder(dictionaryOf(t, file, "weight")).Double(got)
		if err != nil {
			t.Fatalf("Double: %v", err)
		}
		if want := []float64{1.25, 2.5, 3.75}; !slices.Equal(got[:n], want) {
			t.Errorf("got %v, want %v", got[:n], want)
		}
	})

	t.Run("blob", func(t *testing.T) {
		got := make([][]byte, 3)
		n, err := parquet.NewPlainDecoder(dictionaryOf(t, file, "blob")).ByteArray(got)
		if err != nil {
			t.Fatalf("ByteArray: %v", err)
		}
		if want := []string{"one", "two", "three"}; !slices.Equal(strs(got[:n]), want) {
			t.Errorf("got %q, want %q", strs(got[:n]), want)
		}
	})

	t.Run("fixed", func(t *testing.T) {
		got := make([][]byte, 3)
		n, err := parquet.NewPlainDecoder(dictionaryOf(t, file, "fixed")).Fixed(got, 4)
		if err != nil {
			t.Fatalf("Fixed: %v", err)
		}
		if want := []string{"abcd", "efgh", "ijkl"}; !slices.Equal(strs(got[:n]), want) {
			t.Errorf("got %q, want %q", strs(got[:n]), want)
		}
	})

	// The decimal column is written as four fixed bytes holding a big endian
	// two's complement number, which is the format's own layout and not
	// anything the plain decoder undoes. The scale of two is in the schema, so
	// 1.25 is written as 125.
	t.Run("price", func(t *testing.T) {
		got := make([][]byte, 2)
		n, err := parquet.NewPlainDecoder(dictionaryOf(t, file, "price")).Fixed(got, 4)
		if err != nil {
			t.Fatalf("Fixed: %v", err)
		}
		want := []int32{125, 250}
		for i := range n {
			if v := int32(binary.BigEndian.Uint32(got[i])); v != want[i] {
				t.Errorf("value %d: got %d, want %d", i, v, want[i])
			}
		}
	})

	// The date is a count of days and the time of day is a count of
	// microseconds, both of them written as the integer they fit in, which is
	// the whole of what makes a date different from a number in this encoding.
	t.Run("day", func(t *testing.T) {
		got := make([]int32, 1)
		if _, err := parquet.NewPlainDecoder(dictionaryOf(t, file, "day")).Int32(got); err != nil {
			t.Fatalf("Int32: %v", err)
		}
		if got[0] != 20693 {
			t.Errorf("got %d, want 20693", got[0])
		}
	})

	t.Run("clock", func(t *testing.T) {
		got := make([]int64, 1)
		if _, err := parquet.NewPlainDecoder(dictionaryOf(t, file, "clock")).Int64(got); err != nil {
			t.Fatalf("Int64: %v", err)
		}
		if got[0] != 45015000000 {
			t.Errorf("got %d, want 45015000000", got[0])
		}
	})
}

// TestPlainRealBooleans decodes the one column of a real file that is written
// as plain booleans.
//
// The column is required, so the page has no levels in front of its values and
// its whole body is three bits.
func TestPlainRealBooleans(t *testing.T) {
	pages, column := pagesOf(t, "alltypes.parquet", "flag")

	page := pages[0]
	if page.Encoding != parquet.Plain {
		t.Fatalf("the page is %s, want %s", page.Encoding, parquet.Plain)
	}

	got := make([]bool, page.NumValues)
	n, err := parquet.NewPlainDecoder(valuesOf(t, page, column)).Boolean(got)
	if err != nil {
		t.Fatalf("Boolean: %v", err)
	}
	if want := []bool{true, false, true}; !slices.Equal(got[:n], want) {
		t.Errorf("got %v, want %v", got[:n], want)
	}
}

// TestPlainRealPages decodes a column that runs over more than one page, which
// is what a decoder does all day and is where a cursor left in the wrong place
// at the end of one page turns into wrong values in the next.
//
// The column counts from nought to four hundred and ninety nine, in two pages
// written the second way.
func TestPlainRealPages(t *testing.T) {
	pages, column := pagesOf(t, "pages.parquet", "n")
	if len(pages) < 2 {
		t.Fatalf("the column has %d pages, want more than one", len(pages))
	}

	var d parquet.PlainDecoder
	var got []int32
	for _, page := range pages {
		d.Reset(valuesOf(t, page, column))

		buf := make([]int32, page.NumValues)
		n, err := d.Int32(buf)
		if err != nil {
			t.Fatalf("Int32: %v", err)
		}
		if int32(n) != page.NumValues {
			t.Fatalf("a page of %d values decoded %d", page.NumValues, n)
		}
		got = append(got, buf...)
	}

	if len(got) != 500 {
		t.Fatalf("got %d values, want 500", len(got))
	}
	for i, v := range got {
		if v != int32(i) {
			t.Fatalf("value %d is %d", i, v)
		}
	}
}

// TestPlainRealLegacy decodes the file written the old way all round, which is
// the only place the deprecated timestamp turns up and is the reason the
// decoder has to read it.
//
// One of the two rows is before 1970, so the day comes out negative once it is
// moved to the epoch, and a decoder that treated the Julian day as unsigned all
// the way through would have it thousands of years out.
func TestPlainRealLegacy(t *testing.T) {
	t.Run("moment", func(t *testing.T) {
		pages, column := pagesOf(t, "legacy.parquet", "moment")

		got := make([]int64, pages[0].NumValues)
		n, err := parquet.NewPlainDecoder(valuesOf(t, pages[0], column)).Int96(got)
		if err != nil {
			t.Fatalf("Int96: %v", err)
		}

		// The 28th of August 2026 at noon and a little, and the moment Apollo
		// 11 landed.
		want := []int64{1787918400123456000, -14182940000000000}
		if !slices.Equal(got[:n], want) {
			t.Errorf("got %v, want %v", got[:n], want)
		}
	})

	t.Run("label", func(t *testing.T) {
		pages, column := pagesOf(t, "legacy.parquet", "label")

		got := make([][]byte, pages[0].NumValues)
		n, err := parquet.NewPlainDecoder(valuesOf(t, pages[0], column)).ByteArray(got)
		if err != nil {
			t.Fatalf("ByteArray: %v", err)
		}
		if want := []string{"now", "then"}; !slices.Equal(strs(got[:n]), want) {
			t.Errorf("got %q, want %q", strs(got[:n]), want)
		}
	})
}

// TestPlainNoCopy checks that a byte array points into the page rather than
// copying out of it, which is what a column of a million strings is counting on
// and is a promise the doc comment makes.
func TestPlainNoCopy(t *testing.T) {
	data := byteArrays("one", "two")

	got := make([][]byte, 2)
	if _, err := parquet.NewPlainDecoder(data).ByteArray(got); err != nil {
		t.Fatalf("ByteArray: %v", err)
	}

	data[4] = 'O'
	if !bytes.Equal(got[0], []byte("One")) {
		t.Errorf("the value is %q, so it was copied out of the page", got[0])
	}
}

// BenchmarkPlain decodes a page of each type.
//
// This is the bottom of the reader and every other encoding ends up here, so
// what it costs per value is close to the floor of what reading a column costs.
// A fixed width type should come out at a memory copy and a byte array should
// not, since every one of them has a length to read first.
func BenchmarkPlain(b *testing.B) {
	const values = 100000

	var numbers []byte
	for i := range values {
		numbers = binary.LittleEndian.AppendUint64(numbers, uint64(i))
	}

	var arrays []byte
	for i := range values {
		arrays = binary.LittleEndian.AppendUint32(arrays, 8)
		arrays = binary.LittleEndian.AppendUint64(arrays, uint64(i))
	}

	int32s := make([]int32, values)
	int64s := make([]int64, values)
	doubles := make([]float64, values)
	booleans := make([]bool, values)
	blobs := make([][]byte, values)

	cases := []struct {
		name string
		read func(d *parquet.PlainDecoder) (int, error)
	}{
		{name: "int32", read: func(d *parquet.PlainDecoder) (int, error) { return d.Int32(int32s) }},
		{name: "int64", read: func(d *parquet.PlainDecoder) (int, error) { return d.Int64(int64s) }},
		{name: "double", read: func(d *parquet.PlainDecoder) (int, error) { return d.Double(doubles) }},
		{name: "int96", read: func(d *parquet.PlainDecoder) (int, error) { return d.Int96(int64s) }},
		{name: "boolean", read: func(d *parquet.PlainDecoder) (int, error) { return d.Boolean(booleans) }},
		{name: "fixed", read: func(d *parquet.PlainDecoder) (int, error) { return d.Fixed(blobs, 8) }},
	}

	var d parquet.PlainDecoder
	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			b.SetBytes(int64(len(numbers)))
			for b.Loop() {
				d.Reset(numbers)
				if _, err := c.read(&d); err != nil {
					b.Fatalf("read: %v", err)
				}
			}
		})
	}

	b.Run("byte array", func(b *testing.B) {
		b.SetBytes(int64(len(arrays)))
		for b.Loop() {
			d.Reset(arrays)
			if _, err := d.ByteArray(blobs); err != nil {
				b.Fatalf("ByteArray: %v", err)
			}
		}
	})
}
