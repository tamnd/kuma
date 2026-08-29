package parquet_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"slices"
	"testing"

	"github.com/tamnd/kuma/parquet"
)

// TestPlainEncoderBytes checks the bytes each type is written as.
//
// The values here are written out by hand rather than worked out, because what
// the encoder has to agree with is the format and not this package's idea of it.
// Every one of them is little endian and the width of the physical type, which
// is the whole of the plain encoding apart from the three types that are not.
func TestPlainEncoderBytes(t *testing.T) {
	for _, c := range []struct {
		name string
		put  func(*parquet.PlainEncoder)
		want []byte
	}{
		{
			"int32",
			func(e *parquet.PlainEncoder) { e.Int32([]int32{1, -2}) },
			[]byte{0x01, 0x00, 0x00, 0x00, 0xfe, 0xff, 0xff, 0xff},
		},
		{
			"int64",
			func(e *parquet.PlainEncoder) { e.Int64([]int64{1, -2}) },
			[]byte{
				0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
				0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
			},
		},
		{
			"float",
			func(e *parquet.PlainEncoder) { e.Float([]float32{1.5}) },
			[]byte{0x00, 0x00, 0xc0, 0x3f},
		},
		{
			"double",
			func(e *parquet.PlainEncoder) { e.Double([]float64{1.5}) },
			[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x3f},
		},
		{
			"booleans packed eight to a byte from the bottom up",
			func(e *parquet.PlainEncoder) { e.Boolean([]bool{true, false, true}) },
			[]byte{0x05},
		},
		{
			"nine booleans, which is a byte and one bit of the next",
			func(e *parquet.PlainEncoder) {
				e.Boolean([]bool{true, true, true, true, true, true, true, true, true})
			},
			[]byte{0xff, 0x01},
		},
		{
			"a byte array, which is four bytes of length and then the bytes",
			func(e *parquet.PlainEncoder) { e.ByteArray([][]byte{[]byte("GB"), nil}) },
			[]byte{0x02, 0x00, 0x00, 0x00, 'G', 'B', 0x00, 0x00, 0x00, 0x00},
		},
		{
			"a string, which is a byte array without the copy",
			func(e *parquet.PlainEncoder) { e.ByteArrayString([]string{"GB", ""}) },
			[]byte{0x02, 0x00, 0x00, 0x00, 'G', 'B', 0x00, 0x00, 0x00, 0x00},
		},
		{
			"a fixed width value, which has no length in front of it",
			func(e *parquet.PlainEncoder) { e.Fixed([][]byte{{0x0a, 0x0b}, {0x0c, 0x0d}}) },
			[]byte{0x0a, 0x0b, 0x0c, 0x0d},
		},
		{
			"nothing at all",
			func(e *parquet.PlainEncoder) { e.Int32(nil) },
			nil,
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			var e parquet.PlainEncoder
			c.put(&e)

			if got := e.Bytes(); !bytes.Equal(got, c.want) {
				t.Errorf("wrote %x, want %x", got, c.want)
			}
			if got, want := e.Len(), len(c.want); got != want {
				t.Errorf("the page is %d bytes, want %d", got, want)
			}
		})
	}
}

// TestPlainEncoderRoundTrip reads back what the encoder wrote.
//
// Bytes written by hand say the encoder agrees with the format. Reading them
// back says the two halves of this package agree with each other, which is the
// thing a file written here and read here depends on and the thing that would
// quietly stop being true.
func TestPlainEncoderRoundTrip(t *testing.T) {
	var e parquet.PlainEncoder

	i32 := []int32{0, 1, -1, math.MaxInt32, math.MinInt32}
	e.Int32(i32)
	got32 := make([]int32, len(i32))
	d := parquet.NewPlainDecoder(e.Bytes())
	if n, err := d.Int32(got32); err != nil || n != len(i32) {
		t.Fatalf("read %d values, %v", n, err)
	}
	if !slices.Equal(got32, i32) {
		t.Errorf("read %v, want %v", got32, i32)
	}

	e.Reset()
	i64 := []int64{0, 1, -1, math.MaxInt64, math.MinInt64}
	e.Int64(i64)
	got64 := make([]int64, len(i64))
	d.Reset(e.Bytes())
	if n, err := d.Int64(got64); err != nil || n != len(i64) {
		t.Fatalf("read %d values, %v", n, err)
	}
	if !slices.Equal(got64, i64) {
		t.Errorf("read %v, want %v", got64, i64)
	}

	// A NaN is written as the bits it is rather than compared, since a NaN is
	// unequal to itself and a round trip that lost one would say nothing.
	e.Reset()
	f64 := []float64{0, -0.5, math.Inf(-1), math.NaN()}
	e.Double(f64)
	gotf := make([]float64, len(f64))
	d.Reset(e.Bytes())
	if n, err := d.Double(gotf); err != nil || n != len(f64) {
		t.Fatalf("read %d values, %v", n, err)
	}
	for i := range f64 {
		if math.Float64bits(gotf[i]) != math.Float64bits(f64[i]) {
			t.Errorf("read %v at %d, want %v", gotf[i], i, f64[i])
		}
	}

	e.Reset()
	f32 := []float32{0, -0.5, float32(math.Inf(1))}
	e.Float(f32)
	gots := make([]float32, len(f32))
	d.Reset(e.Bytes())
	if n, err := d.Float(gots); err != nil || n != len(f32) {
		t.Fatalf("read %d values, %v", n, err)
	}
	if !slices.Equal(gots, f32) {
		t.Errorf("read %v, want %v", gots, f32)
	}

	e.Reset()
	flags := []bool{true, false, false, true, true, false, true, true, false, true}
	e.Boolean(flags)
	gotb := make([]bool, len(flags))
	d.Reset(e.Bytes())
	if n, err := d.Boolean(gotb); err != nil || n != len(flags) {
		t.Fatalf("read %d values, %v", n, err)
	}
	if !slices.Equal(gotb, flags) {
		t.Errorf("read %v, want %v", gotb, flags)
	}

	e.Reset()
	words := []string{"", "alpha", "a value long enough to be worth the length in front of it"}
	e.ByteArrayString(words)
	gotw := make([][]byte, len(words))
	d.Reset(e.Bytes())
	if n, err := d.ByteArray(gotw); err != nil || n != len(words) {
		t.Fatalf("read %d values, %v", n, err)
	}
	if !slices.Equal(strs(gotw), words) {
		t.Errorf("read %v, want %v", strs(gotw), words)
	}

	e.Reset()
	fixed := [][]byte{{0x00, 0x01, 0x02}, {0x03, 0x04, 0x05}}
	e.Fixed(fixed)
	gotx := make([][]byte, len(fixed))
	d.Reset(e.Bytes())
	if n, err := d.Fixed(gotx, 3); err != nil || n != len(fixed) {
		t.Fatalf("read %d values, %v", n, err)
	}
	if !slices.EqualFunc(gotx, fixed, bytes.Equal) {
		t.Errorf("read %x, want %x", gotx, fixed)
	}
}

// TestPlainEncoderRuns checks that a page written in several calls is the page
// written in one.
//
// A writer fills a page from a column that arrives in chunks, so the values of
// one page are put down in as many calls as the column had chunks. Nothing in
// the encoding is stateful except the run of booleans, and that is the one this
// is about: a byte half full of them has to be finished by the next call rather
// than started again.
func TestPlainEncoderRuns(t *testing.T) {
	var one, many parquet.PlainEncoder

	one.Boolean([]bool{true, false, true, true, false, false, true, false, true})
	many.Boolean([]bool{true, false, true})
	many.Boolean(nil)
	many.Boolean([]bool{true, false, false, true, false})
	many.Boolean([]bool{true})

	if !bytes.Equal(one.Bytes(), many.Bytes()) {
		t.Errorf("wrote %x in pieces, want %x", many.Bytes(), one.Bytes())
	}
}

// TestPlainEncoderAlign checks that a value after a run of booleans starts on a
// byte of its own.
//
// A page is all one type, so mixing the two is not something a writer here does.
// The decoder moves past the half written byte whatever put it there, so the
// encoder has to leave one, and a test is the only place the two ever meet.
func TestPlainEncoderAlign(t *testing.T) {
	var e parquet.PlainEncoder
	e.Boolean([]bool{true})
	e.Int32([]int32{7})

	want := append([]byte{0x01}, 0x07, 0x00, 0x00, 0x00)
	if got := e.Bytes(); !bytes.Equal(got, want) {
		t.Fatalf("wrote %x, want %x", got, want)
	}

	d := parquet.NewPlainDecoder(e.Bytes())
	flags := make([]bool, 1)
	if _, err := d.Boolean(flags); err != nil {
		t.Fatal(err)
	}
	got := make([]int32, 1)
	if _, err := d.Int32(got); err != nil {
		t.Fatal(err)
	}
	if got[0] != 7 {
		t.Errorf("read %d, want 7", got[0])
	}
}

// TestPlainEncoderReset checks that an encoder put back to the start writes the
// page a new one would.
//
// The buffer is kept, which is the point of it: a writer putting down a thousand
// pages of one column allocates for the largest of them rather than for each. So
// the bytes of the last page are still there and the next page has to start at
// the front of them rather than after them.
func TestPlainEncoderReset(t *testing.T) {
	var e parquet.PlainEncoder
	e.Boolean([]bool{true, true, true})
	e.Reset()

	if got := e.Len(); got != 0 {
		t.Errorf("a page of %d bytes after a reset, want none", got)
	}
	e.Int32([]int32{1})

	want := []byte{0x01, 0x00, 0x00, 0x00}
	if got := e.Bytes(); !bytes.Equal(got, want) {
		t.Errorf("wrote %x, want %x", got, want)
	}
}

// TestPlainEncoderPage checks the encoder against pages pyarrow wrote.
//
// These bytes came out of a real file rather than out of this package, so what
// this says is that a page written here is the page another writer wrote for the
// same values. The file has no dictionary, no compression and one page per
// column, which is why it is the one to read the values back out of.
func TestPlainEncoderPage(t *testing.T) {
	for _, c := range []struct {
		column string
		put    func(*parquet.PlainEncoder, []byte, int) error
	}{
		{"count", func(e *parquet.PlainEncoder, data []byte, n int) error {
			vals := make([]int32, n)
			n, err := parquet.NewPlainDecoder(data).Int32(vals)
			if err != nil {
				return err
			}
			e.Int32(vals[:n])
			return nil
		}},
		{"total", func(e *parquet.PlainEncoder, data []byte, n int) error {
			vals := make([]int64, n)
			n, err := parquet.NewPlainDecoder(data).Int64(vals)
			if err != nil {
				return err
			}
			e.Int64(vals[:n])
			return nil
		}},
		{"weight", func(e *parquet.PlainEncoder, data []byte, n int) error {
			vals := make([]float64, n)
			n, err := parquet.NewPlainDecoder(data).Double(vals)
			if err != nil {
				return err
			}
			e.Double(vals[:n])
			return nil
		}},
		{"flag", func(e *parquet.PlainEncoder, data []byte, n int) error {
			vals := make([]bool, n)
			n, err := parquet.NewPlainDecoder(data).Boolean(vals)
			if err != nil {
				return err
			}
			e.Boolean(vals[:n])
			return nil
		}},
		{"name", func(e *parquet.PlainEncoder, data []byte, n int) error {
			vals := make([][]byte, n)
			n, err := parquet.NewPlainDecoder(data).ByteArray(vals)
			if err != nil {
				return err
			}
			e.ByteArray(vals[:n])
			return nil
		}},
	} {
		t.Run(c.column, func(t *testing.T) {
			pages, col := pagesOf(t, "plain.parquet", c.column)
			if len(pages) != 1 {
				t.Fatalf("the column is in %d pages, want one", len(pages))
			}
			want := valuesOf(t, pages[0], col)

			var e parquet.PlainEncoder
			if err := c.put(&e, want, int(pages[0].NumValues)); err != nil {
				t.Fatal(err)
			}
			if got := e.Bytes(); !bytes.Equal(got, want) {
				t.Errorf("wrote %x, want the %x pyarrow wrote", got, want)
			}
		})
	}
}

// TestPlainEncoderLength checks the four bytes of length in front of a value.
//
// A length is written as an unsigned four byte number, so a reader that read it
// as a signed one would take a value of two gigabytes for a value of a negative
// length. Nothing here writes one that big, and what this checks is that the
// bytes say what the format says they say.
func TestPlainEncoderLength(t *testing.T) {
	var e parquet.PlainEncoder
	e.ByteArrayString([]string{"abc"})

	if got := binary.LittleEndian.Uint32(e.Bytes()); got != 3 {
		t.Errorf("the length is %d, want 3", got)
	}
}

// BenchmarkPlainWrite encodes a page of each type.
//
// This is the bottom of the writer the way BenchmarkPlain is the bottom of the
// reader, and a page of a hundred thousand values is what a real one holds. A
// fixed width type should come out at a memory copy. A byte array should not,
// since every value has a length written in front of it, and a boolean should
// not either, since eight of them go into a byte one bit at a time.
func BenchmarkPlainWrite(b *testing.B) {
	const values = 100000

	int32s := make([]int32, values)
	int64s := make([]int64, values)
	doubles := make([]float64, values)
	booleans := make([]bool, values)
	blobs := make([][]byte, values)
	words := make([]string, values)
	for i := range values {
		int32s[i] = int32(i)
		int64s[i] = int64(i)
		doubles[i] = float64(i)
		booleans[i] = i%3 == 0
		blobs[i] = []byte("a value")
		words[i] = "a value"
	}

	var e parquet.PlainEncoder
	for _, c := range []struct {
		name string
		put  func(*parquet.PlainEncoder)
	}{
		{"int32", func(e *parquet.PlainEncoder) { e.Int32(int32s) }},
		{"int64", func(e *parquet.PlainEncoder) { e.Int64(int64s) }},
		{"double", func(e *parquet.PlainEncoder) { e.Double(doubles) }},
		{"boolean", func(e *parquet.PlainEncoder) { e.Boolean(booleans) }},
		{"fixed", func(e *parquet.PlainEncoder) { e.Fixed(blobs) }},
		{"byte array", func(e *parquet.PlainEncoder) { e.ByteArray(blobs) }},
		{"string", func(e *parquet.PlainEncoder) { e.ByteArrayString(words) }},
	} {
		b.Run(c.name, func(b *testing.B) {
			e.Reset()
			c.put(&e)
			b.SetBytes(int64(e.Len()))

			for b.Loop() {
				e.Reset()
				c.put(&e)
			}
		})
	}
}
