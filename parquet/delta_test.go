package parquet_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"math/bits"
	"slices"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/parquet"
)

// deltaOf writes values the way a writer of the encoding does, which is a
// header, then the smallest difference of each block, then the width of each of
// its miniblocks, then the differences packed at those widths.
//
// The tests need this because no file here can be made to hold the shapes worth
// reading. A file written by pyarrow has blocks of a hundred and twenty eight
// values in four miniblocks and nothing else, so a block of another size, a
// miniblock wider than an int32, or a page of one value are all cases a real
// file will not produce and a reader still has to read.
func deltaOf(block, minis int, values ...int64) []byte {
	var first int64
	if len(values) > 0 {
		first = values[0]
	}

	b := binary.AppendUvarint(nil, uint64(block))
	b = binary.AppendUvarint(b, uint64(minis))
	b = binary.AppendUvarint(b, uint64(len(values)))
	b = binary.AppendVarint(b, first)

	deltas := make([]int64, 0, len(values))
	for i := 1; i < len(values); i++ {
		deltas = append(deltas, values[i]-values[i-1])
	}

	perMini := block / minis
	for len(deltas) > 0 {
		n := min(block, len(deltas))
		group, rest := deltas[:n], deltas[n:]
		deltas = rest

		// The smallest difference comes off every value of the block, so what
		// is packed is always positive and the width of a miniblock is the
		// width of the largest number left in it.
		minDelta := slices.Min(group)
		widths := make([]byte, minis)
		for i := range minis {
			for _, d := range group[min(i*perMini, len(group)):min((i+1)*perMini, len(group))] {
				widths[i] = max(widths[i], byte(bits.Len64(uint64(d-minDelta))))
			}
		}

		b = binary.AppendVarint(b, minDelta)
		b = append(b, widths...)
		for i := range minis {
			at := min(i*perMini, len(group))
			b = packDelta(b, group[at:min(at+perMini, len(group))], perMini, uint(widths[i]), minDelta)
		}
	}
	return b
}

// packDelta packs one miniblock, which is as many differences as it holds with
// the smallest one taken off them, at width bits each from the bottom of a byte
// up. A miniblock is written whole, so the ones past the end of the values are
// nought.
func packDelta(b []byte, deltas []int64, perMini int, width uint, minDelta int64) []byte {
	buf := make([]byte, perMini*int(width)/8)
	bit := 0
	for _, d := range deltas {
		v := uint64(d - minDelta)
		for i := range width {
			if v>>i&1 == 1 {
				buf[(bit+int(i))/8] |= 1 << ((bit + int(i)) % 8)
			}
		}
		bit += int(width)
	}
	return append(b, buf...)
}

// TestDeltaDecoder reads pages written by hand, which is where the shapes a
// file will not produce are.
func TestDeltaDecoder(t *testing.T) {
	cases := []struct {
		name   string
		block  int
		minis  int
		values []int64
	}{
		{
			// The one the encoding is for. Every difference is the same, so
			// nothing is left once the smallest one comes off and the values
			// are packed at no bits at all.
			name: "a constant difference", block: 128, minis: 4,
			values: counting(0, 1, 300),
		},
		{
			name: "going down", block: 128, minis: 4,
			values: counting(1000, -3, 300),
		},
		{
			// Differences as wide as a value can be, starting at every offset
			// into a byte, which is where a value is read out of two words.
			name: "the widest differences there are", block: 128, minis: 4,
			values: []int64{math.MinInt64, math.MaxInt64, math.MinInt64, 0, math.MaxInt64},
		},
		{
			name: "one value", block: 128, minis: 4,
			values: []int64{-7},
		},
		{
			name: "no values at all", block: 128, minis: 4,
			values: nil,
		},
		{
			// A page that stops in the middle of a block, so the widths of the
			// miniblocks it never wrote are still there and are never used.
			name: "a block half written", block: 256, minis: 8,
			values: counting(0, 5, 40),
		},
		{
			name: "a miniblock of one value", block: 128, minis: 4,
			values: counting(0, 9, 33),
		},
		{
			name: "more blocks than one", block: 128, minis: 2,
			values: wobbling(1000),
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, err := parquet.NewDeltaDecoder(deltaOf(c.block, c.minis, c.values...))
			if err != nil {
				t.Fatalf("NewDeltaDecoder: %v", err)
			}

			got := make([]int64, len(c.values))
			for n := 0; n < len(got); {
				m, err := d.Read(got[n:])
				if err != nil {
					t.Fatalf("Read: %v", err)
				}
				n += m
			}
			if !slices.Equal(got, c.values) {
				t.Fatalf("got %v, want %v", got, c.values)
			}

			// The page holds what its header says and nothing after it, so
			// whatever is left in the last miniblock is padding rather than a
			// value somebody could read.
			if _, err := d.Read(make([]int64, 1)); !errors.Is(err, io.EOF) {
				t.Errorf("reading past the end: got %v, want %v", err, io.EOF)
			}
		})
	}
}

// counting is n values from first, each step further on than the one before it.
func counting(first, step int64, n int) []int64 {
	values := make([]int64, n)
	for i := range values {
		values[i] = first + int64(i)*step
	}
	return values
}

// wobbling is n values that go up and down by a different amount each time, so
// that the widths of the miniblocks are all different and none of them is the
// width of the block.
func wobbling(n int) []int64 {
	values := make([]int64, n)
	for i := 1; i < n; i++ {
		step := int64(i % 37 * (i % 11))
		if i%3 == 0 {
			step = -step
		}
		values[i] = values[i-1] + step
	}
	return values
}

// TestDeltaDecoderInt32 reads a page into the narrower of the two types the
// encoding is written for.
//
// The values are the ones a writer of int32 produces at the ends of the type:
// the difference between the largest and the smallest of them is wider than an
// int32, and the writer wrote it wrapped, so a reader that did the arithmetic
// any other way would come back with something else.
func TestDeltaDecoderInt32(t *testing.T) {
	want := []int32{0, math.MaxInt32, math.MinInt32, -1, 0, 7}

	wide := make([]int64, len(want))
	for i, v := range want {
		wide[i] = int64(v)
	}

	d, err := parquet.NewDeltaDecoder(deltaOf(128, 4, wide...))
	if err != nil {
		t.Fatalf("NewDeltaDecoder: %v", err)
	}

	got := make([]int32, len(want))
	if n, err := d.Read(got); err != nil || n != len(want) {
		t.Fatalf("Read: %d values, %v", n, err)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

// TestDeltaDecoderRead reads a page a few values at a time, which is what a
// column does when a page holds more values than the buffer it is reading into.
func TestDeltaDecoderRead(t *testing.T) {
	want := counting(5, 3, 100)

	d := &parquet.DeltaDecoder{}
	if err := d.Reset(deltaOf(128, 4, want...)); err != nil {
		t.Fatalf("Reset: %v", err)
	}

	var got []int64
	buf := make([]int64, 7)
	for {
		n, err := d.Read(buf)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		got = append(got, buf[:n]...)
	}
	if !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}

	// A decoder reads the page it was last given, so a column reading a
	// thousand pages hands them to the same one.
	if err := d.Reset(deltaOf(128, 4, 1, 2, 3)); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if n, err := d.Read(buf); err != nil || n != 3 {
		t.Fatalf("Read: %d values, %v", n, err)
	}
	if !slices.Equal(buf[:3], []int64{1, 2, 3}) {
		t.Fatalf("got %v after resetting", buf[:3])
	}
}

// TestDeltaDecoderRefused is the pages a decoder has to refuse.
//
// Every one of them is bytes that could be read into values rather than into an
// error. A width or a count taken from the page decides how the bytes behind it
// are cut up, so believing one that the page cannot hold is how a reader ends up
// handing back numbers nobody wrote.
func TestDeltaDecoderRefused(t *testing.T) {
	// A number of more than ten bytes, which is more than a varint of sixty
	// four bits can be.
	long := bytes.Repeat([]byte{0xff}, 11)

	header := func(block, minis, total int) []byte {
		b := binary.AppendUvarint(nil, uint64(block))
		b = binary.AppendUvarint(b, uint64(minis))
		return binary.AppendUvarint(b, uint64(total))
	}

	cases := []struct {
		name string
		data []byte
	}{
		{name: "a page of no bytes", data: nil},
		{name: "a block size that stops half way", data: header(128, 4, 0)[:1]},
		{name: "a header that stops after the block size", data: binary.AppendUvarint(nil, 128)},
		{name: "a header that stops before the first value", data: header(128, 4, 2)},
		{name: "a count of more than ten bytes", data: long},
		{name: "a first value of more than ten bytes", data: append(header(128, 4, 2), long...)},
		{name: "more values than a page can hold", data: header(128, 4, 1<<40)},
		{name: "a block of no values", data: header(0, 4, 2)},
		{name: "a block that is not a multiple of a hundred and twenty eight", data: header(200, 4, 2)},
		{name: "a block of more values than a page has", data: header(1<<21, 4, 2)},
		{name: "a block of no miniblocks", data: header(128, 0, 2)},
		{name: "miniblocks that do not divide the block", data: header(128, 5, 2)},
		{name: "a miniblock that is not a multiple of thirty two", data: header(128, 8, 2)},
		{
			// A header saying two values follow and then nothing where the
			// smallest difference of the first block should be.
			name: "a block header that is not there",
			data: append(header(128, 4, 2), 0),
		},
		{
			name: "a block that stops before its widths",
			data: append(header(128, 4, 2), 0, 0, 1),
		},
		{
			name: "a miniblock wider than a value",
			data: append(header(128, 4, 2), 0, 0, 65, 0, 0, 0),
		},
		{
			// One value of the second block written where thirty two of them
			// were promised, at four bits each.
			name: "a miniblock shorter than its values",
			data: append(header(128, 4, 2), 0, 0, 4, 0, 0, 0),
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, err := parquet.NewDeltaDecoder(c.data)
			if err == nil {
				_, err = d.Read(make([]int64, 2))
			}
			if !errors.Is(err, parquet.ErrFormat) {
				t.Errorf("got %v, want %v", err, parquet.ErrFormat)
			}
		})
	}
}

// TestReadColumnDelta reads the columns of a real file that pyarrow wrote as
// differences.
//
// They are the shapes the encoding is written for: a column that climbs by one,
// one that falls, one that jumps about, differences that need most of an int64,
// and a column with holes in it whose differences are between the values that
// are there rather than between the rows.
func TestReadColumnDelta(t *testing.T) {
	t.Run("a constant difference", func(t *testing.T) {
		deltaColumn(t, "n", dtype.Int32, func(i int) int32 { return int32(i) })
	})

	t.Run("going down", func(t *testing.T) {
		deltaColumn(t, "down", dtype.Int32, func(i int) int32 { return int32(1000 - i) })
	})

	t.Run("up and down", func(t *testing.T) {
		deltaColumn(t, "wobble", dtype.Int32, func(i int) int32 {
			if i%2 == 0 {
				return int32(i % 17 * -1000)
			}
			return int32(i % 17 * 1000)
		})
	})

	t.Run("wide differences", func(t *testing.T) {
		deltaColumn(t, "big", dtype.Int64, func(i int) int64 {
			v := uint64(i)*6364136223846793005 + 1442695040888963407
			return int64(v % (1 << 62))
		})
	})

	// A column parquet has no type for, so it is written as int32 and narrowed
	// on the way in like the values of a plain page are.
	t.Run("narrowed", func(t *testing.T) {
		deltaColumn(t, "small", dtype.Int8, func(i int) int8 { return int8(i%251 - 125) })
	})

	// The same again where the narrowing is what makes the value: the column
	// runs past where an int32 turns negative, and the writer wrapped the
	// arithmetic rather than widening it.
	t.Run("wrapping", func(t *testing.T) {
		deltaColumn(t, "unsigned", dtype.Uint32, func(i int) uint32 {
			return uint32(uint64(i) * 4000000 % (1 << 32))
		})
	})

	// A page only holds a difference for a row that has a value, so the levels
	// are the only thing that says which rows those are and the differences run
	// from one present value to the next.
	t.Run("with nulls", func(t *testing.T) {
		a := readColumn(t, "delta.parquet", "maybe")
		if a.Len() != 1000 || a.NullCount() != 200 {
			t.Fatalf("%d values, %d of them null", a.Len(), a.NullCount())
		}

		values := a.Values[int64]()
		for i := range a.Len() {
			if i%5 == 0 {
				if !a.IsNull(i) {
					t.Fatalf("value %d is there and should be missing", i)
				}
				continue
			}
			if values[i] != int64(i*7) {
				t.Fatalf("value %d: got %d, want %d", i, values[i], i*7)
			}
		}
	})
}

// deltaColumn reads a column of delta.parquet and checks every value in it
// against what the script that wrote the file put there.
func deltaColumn[T array.Numeric](t *testing.T, column string, want dtype.DataType, at func(i int) T) {
	t.Helper()

	a := readColumn(t, "delta.parquet", column)
	if a.DType() != want {
		t.Fatalf("the column is a %s, want a %s", a.DType(), want)
	}
	if a.Len() != 1000 || a.NullCount() != 0 {
		t.Fatalf("%d values, %d of them null", a.Len(), a.NullCount())
	}

	got := a.Values[T]()
	for i := range a.Len() {
		if got[i] != at(i) {
			t.Fatalf("value %d: got %v, want %v", i, got[i], at(i))
		}
	}
}

// deltaPage builds a data page of delta encoded values, which is levels behind
// four bytes of their length and then the header and the blocks.
func deltaPage(levels []byte, values int32, encoded []byte) parquet.Page {
	body := binary.LittleEndian.AppendUint32(nil, uint32(len(levels)))
	body = append(body, levels...)
	body = append(body, encoded...)
	return parquet.Page{
		PageHeader: parquet.PageHeader{
			Kind: parquet.DataPage, Encoding: parquet.DeltaBinaryPacked,
			DefinitionEncoding: parquet.RLE, NumValues: values,
		},
		Data: body,
	}
}

// TestColumnReaderDeltaPages reads delta encoded pages built by hand.
func TestColumnReaderDeltaPages(t *testing.T) {
	// Two rows present and one missing, so the two differences in the page are
	// between the values that are there and the levels put the hole back.
	t.Run("with a hole in it", func(t *testing.T) {
		r := readerOf(t, optional())
		if err := r.Page(deltaPage([]byte{0x03, 0x05}, 3, deltaOf(128, 4, 10, 40))); err != nil {
			t.Fatalf("Page: %v", err)
		}

		a := finish(t, r)
		if a.Len() != 3 || a.NullCount() != 1 || !a.IsNull(1) {
			t.Fatalf("%d values, %d of them null", a.Len(), a.NullCount())
		}
		if a.Value[int32](0) != 10 || a.Value[int32](2) != 40 {
			t.Errorf("got %d and %d, want 10 and 40", a.Value[int32](0), a.Value[int32](2))
		}
	})

	// A page in which every row is missing. There are no values under the
	// levels, and pyarrow writes nothing at all rather than a header saying so.
	t.Run("a page of nothing but nulls", func(t *testing.T) {
		r := readerOf(t, optional())
		if err := r.Page(deltaPage([]byte{0x06, 0x00}, 3, nil)); err != nil {
			t.Fatalf("Page: %v", err)
		}

		a := finish(t, r)
		if a.Len() != 3 || a.NullCount() != 3 {
			t.Fatalf("%d values, %d of them null", a.Len(), a.NullCount())
		}
	})

	// A required column, whose pages hold no levels at all, so the header
	// starts at the first byte of the body.
	t.Run("a required column", func(t *testing.T) {
		required := optional()
		required.Element.Repetition = parquet.Required
		required.MaxDefinition = 0

		r := readerOf(t, required)
		page := deltaPage(nil, 2, nil)
		page.Data = deltaOf(128, 4, -5, 5)
		if err := r.Page(page); err != nil {
			t.Fatalf("Page: %v", err)
		}

		a := finish(t, r)
		if a.Len() != 2 || a.Value[int32](0) != -5 || a.Value[int32](1) != 5 {
			t.Fatalf("%d values, %d and %d", a.Len(), a.Value[int32](0), a.Value[int32](1))
		}
	})

	// A writer that fills its dictionary gives up on it and writes the rest of
	// the chunk in whatever encoding it would have used all along, which here is
	// differences rather than plain values. The rows read as indices are put
	// back as values when the first of those turns up.
	t.Run("falling back from a dictionary", func(t *testing.T) {
		r := readerOf(t, optional())
		if err := r.Page(dictPage(10, 20)); err != nil {
			t.Fatalf("Page: %v", err)
		}
		if err := r.Page(indexPage([]byte{0x04, 0x01}, 2, 1, 0)); err != nil {
			t.Fatalf("Page: %v", err)
		}
		if err := r.Page(deltaPage([]byte{0x04, 0x01}, 2, deltaOf(128, 4, 1, 2))); err != nil {
			t.Fatalf("Page: %v", err)
		}

		a := finish(t, r)
		if a.DType() != dtype.Int32 || a.Len() != 4 {
			t.Fatalf("%d values of %s", a.Len(), a.DType())
		}
		if got := a.Values[int32](); !slices.Equal(got, []int32{20, 10, 1, 2}) {
			t.Errorf("got %v, want [20 10 1 2]", got)
		}
	})
}

// TestColumnReaderRefusedDelta is the delta encoded pages a column has to
// refuse.
func TestColumnReaderRefusedDelta(t *testing.T) {
	doubles := optional()
	doubles.Element.Type = parquet.Double
	doubles.Type = dtype.Float64

	cases := []struct {
		name   string
		column parquet.Column
		pages  []parquet.Page
		want   error
	}{
		{
			// The encoding is written for the two integer widths, so a page of
			// differences of anything else is a file that contradicts the
			// schema in front of it.
			name:   "a column the encoding is not written for",
			column: doubles,
			pages:  []parquet.Page{deltaPage([]byte{0x04, 0x01}, 2, deltaOf(128, 4, 1, 2))},
			want:   parquet.ErrFormat,
		},
		{
			name:   "a page whose header is not there",
			column: optional(),
			pages:  []parquet.Page{deltaPage([]byte{0x04, 0x01}, 2, []byte{0x80})},
			want:   parquet.ErrFormat,
		},
		{
			name:   "a page holding fewer values than its levels want",
			column: optional(),
			pages:  []parquet.Page{deltaPage([]byte{0x08, 0x01}, 4, deltaOf(128, 4, 1, 2))},
			want:   parquet.ErrFormat,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := readerOf(t, c.column)

			var err error
			for _, p := range c.pages {
				if err = r.Page(p); err != nil {
					break
				}
			}
			if !errors.Is(err, c.want) {
				t.Errorf("got %v, want %v", err, c.want)
			}
		})
	}
}

// BenchmarkReadColumnDelta reads a delta encoded column out of a file.
//
// The two of them are the ends of what the encoding costs. A column that climbs
// by one is a bit a value and the reader spends its time adding, and a column of
// wide differences is most of an int64 a value and it spends its time unpacking
// them out of two words each.
func BenchmarkReadColumnDelta(b *testing.B) {
	for _, column := range []string{"n", "big"} {
		b.Run(column, func(b *testing.B) {
			file, chunk, c := chunkOf(&testing.T{}, "delta.parquet", column)
			r := bytes.NewReader(file)

			b.SetBytes(chunk.Meta.TotalCompressedSize)
			b.ReportAllocs()
			for b.Loop() {
				if _, err := parquet.ReadColumn(r, int64(len(file)), chunk, c); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
