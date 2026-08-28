package parquet_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/parquet"
)

// chunkOf finds the chunk holding a column in a file, along with the column
// itself and the bytes of the file.
func chunkOf(t *testing.T, name, column string) ([]byte, *parquet.ColumnChunk, parquet.Column) {
	t.Helper()

	b := bytesOf(t, name)
	m, err := parquet.ReadMetadata(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatalf("ReadMetadata(%s): %v", name, err)
	}

	columns, err := m.Columns()
	if err != nil {
		t.Fatalf("Columns: %v", err)
	}
	i := slices.IndexFunc(columns, func(c parquet.Column) bool { return c.Name() == column })
	if i < 0 {
		t.Fatalf("%s has no column called %s", name, column)
	}

	for g := range m.RowGroups {
		for c := range m.RowGroups[g].Columns {
			chunk := &m.RowGroups[g].Columns[c]
			if columnName(chunk) == column {
				return b, chunk, columns[i]
			}
		}
	}
	t.Fatalf("%s has no chunk for %s", name, column)
	return nil, nil, parquet.Column{}
}

// columnName is what a chunk calls the column it holds.
func columnName(c *parquet.ColumnChunk) string {
	return strings.Join(c.Meta.Path, ".")
}

// readColumn reads one column of a file into an array.
func readColumn(t *testing.T, name, column string) *array.Array {
	t.Helper()

	b, chunk, c := chunkOf(t, name, column)
	a, err := parquet.ReadColumn(bytes.NewReader(b), int64(len(b)), chunk, c)
	if err != nil {
		t.Fatalf("ReadColumn(%s.%s): %v", name, column, err)
	}
	return a
}

// readerOf makes a reader for a column and fails the test if it cannot.
func readerOf(t *testing.T, c parquet.Column) *parquet.ColumnReader {
	t.Helper()

	r, err := parquet.NewColumnReader(c)
	if err != nil {
		t.Fatalf("NewColumnReader: %v", err)
	}
	return r
}

// finish takes the array off a reader and fails the test if it cannot.
func finish(t *testing.T, r *parquet.ColumnReader) *array.Array {
	t.Helper()

	a, err := r.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return a
}

// numbersOf checks a column of fixed width values against what the script that
// wrote the file put in it. The third row of plain.parquet is missing in every
// column, which is what makes it worth reading with the levels.
func numbersOf[T array.Numeric](t *testing.T, a *array.Array, want []T) {
	t.Helper()

	if a.Len() != 4 || a.NullCount() != 1 || !a.IsNull(2) {
		t.Fatalf("%d values, %d of them null, and the third is null: %v",
			a.Len(), a.NullCount(), !a.IsNull(2))
	}

	got := a.Values[T]()
	for i, at := range []int{0, 1, 3} {
		if got[at] != want[i] {
			t.Errorf("value %d: got %v, want %v", at, got[at], want[i])
		}
	}
}

// TestReadColumnTypes reads a column of every type kuma can assemble out of a
// file that has no dictionary in it, so that every one of them arrives as plain
// values and is put back together from its own levels.
//
// Three of the four rows have a value and the third is missing, so each of these
// exercises the two runs the assembly walks in.
func TestReadColumnTypes(t *testing.T) {
	const file = "plain.parquet"

	t.Run("flag", func(t *testing.T) {
		a := readColumn(t, file, "flag")
		if a.Len() != 4 || a.NullCount() != 1 || !a.IsNull(2) {
			t.Fatalf("%d values, %d null", a.Len(), a.NullCount())
		}
		for i, want := range map[int]bool{0: true, 1: false, 3: true} {
			if a.Bool(i) != want {
				t.Errorf("value %d: got %v, want %v", i, a.Bool(i), want)
			}
		}
	})

	// The narrow integers are the interesting ones. Parquet writes all six of
	// them as an int32, so the file says nothing about their width and the
	// schema is the only thing that does.
	t.Run("small", func(t *testing.T) {
		numbersOf(t, readColumn(t, file, "small"), []int8{1, -2, 127})
	})
	t.Run("short", func(t *testing.T) {
		numbersOf(t, readColumn(t, file, "short"), []int16{1000, -2000, 32767})
	})
	t.Run("byte", func(t *testing.T) {
		numbersOf(t, readColumn(t, file, "byte"), []uint8{1, 2, 255})
	})
	t.Run("ushort", func(t *testing.T) {
		numbersOf(t, readColumn(t, file, "ushort"), []uint16{1, 2, 65535})
	})
	t.Run("unsigned", func(t *testing.T) {
		numbersOf(t, readColumn(t, file, "unsigned"), []uint32{1, 2, 4294967295})
	})
	t.Run("big", func(t *testing.T) {
		numbersOf(t, readColumn(t, file, "big"), []uint64{1, 2, 18446744073709551615})
	})

	t.Run("count", func(t *testing.T) {
		numbersOf(t, readColumn(t, file, "count"), []int32{10, -20, 30})
	})
	t.Run("total", func(t *testing.T) {
		numbersOf(t, readColumn(t, file, "total"), []int64{100, -200, 300})
	})
	t.Run("ratio", func(t *testing.T) {
		numbersOf(t, readColumn(t, file, "ratio"), []float32{1.5, -2.5, 3.5})
	})
	t.Run("weight", func(t *testing.T) {
		numbersOf(t, readColumn(t, file, "weight"), []float64{1.25, -2.5, 3.75})
	})

	// A date is a count of days, a time of day is a count of microseconds and a
	// timestamp is a count of milliseconds. All three are stored as the integer
	// they fit in, and what makes them a date is the type rather than the bytes.
	t.Run("day", func(t *testing.T) {
		a := readColumn(t, file, "day")
		if a.DType() != dtype.Date32 {
			t.Fatalf("the column is a %s", a.DType())
		}
		numbersOf(t, a, []int32{20693, -165, 0})
	})
	t.Run("clock", func(t *testing.T) {
		a := readColumn(t, file, "clock")
		if want := (dtype.Time64{Unit: dtype.Microsecond}); a.DType() != want {
			t.Fatalf("the column is a %s, want a %s", a.DType(), want)
		}
		numbersOf(t, a, []int64{45015000000, 0, 86399000000})
	})
	t.Run("moment", func(t *testing.T) {
		a := readColumn(t, file, "moment")
		if want := (dtype.Timestamp{Unit: dtype.Millisecond, Zone: "UTC"}); a.DType() != want {
			t.Fatalf("the column is a %s, want a %s", a.DType(), want)
		}
		numbersOf(t, a, []int64{1787918400000, -14182940000, 0})
	})

	t.Run("name", func(t *testing.T) {
		bytesOfColumn(t, readColumn(t, file, "name"), []string{"one", "", "four"})
	})
	t.Run("blob", func(t *testing.T) {
		bytesOfColumn(t, readColumn(t, file, "blob"), []string{"one", "", "four"})
	})
	t.Run("fixed", func(t *testing.T) {
		bytesOfColumn(t, readColumn(t, file, "fixed"), []string{"abcd", "efgh", "mnop"})
	})
}

// bytesOfColumn checks a column of byte valued values the way numbersOf checks
// the fixed width ones.
func bytesOfColumn(t *testing.T, a *array.Array, want []string) {
	t.Helper()

	if a.Len() != 4 || a.NullCount() != 1 || !a.IsNull(2) {
		t.Fatalf("%d values, %d of them null, and the third is null: %v",
			a.Len(), a.NullCount(), !a.IsNull(2))
	}
	for i, at := range []int{0, 1, 3} {
		if got := string(a.Bytes(at)); got != want[i] {
			t.Errorf("value %d: got %q, want %q", at, got, want[i])
		}
	}
}

// TestReadColumnNulls reads a column that is a third missing over two pages.
//
// This is the case the assembly is for. The column is null wherever the row
// number divides by three and holds the row number everywhere else, which the
// script that wrote the file says and the levels have to reproduce. Getting one
// level wrong would move every value after it.
func TestReadColumnNulls(t *testing.T) {
	a := readColumn(t, "pages.parquet", "maybe")

	if a.Len() != 500 {
		t.Fatalf("got %d values, want 500", a.Len())
	}
	if a.NullCount() != 167 {
		t.Fatalf("got %d nulls, want 167", a.NullCount())
	}

	values := a.Values[int32]()
	for i := range a.Len() {
		if i%3 == 0 {
			if !a.IsNull(i) {
				t.Fatalf("value %d is %d and should be missing", i, values[i])
			}
			continue
		}
		if a.IsNull(i) {
			t.Fatalf("value %d is missing and should be %d", i, i)
		}
		if values[i] != int32(i) {
			t.Fatalf("value %d is %d", i, values[i])
		}
	}
}

// TestReadColumnPages reads a column with nothing missing that runs over more
// than one page, which is where a page boundary handled wrong turns into
// values in the wrong order.
//
// A column with no nulls in it keeps no validity bitmap at all, which is what
// the builder is for and is worth checking arrives that way from a file.
func TestReadColumnPages(t *testing.T) {
	a := readColumn(t, "pages.parquet", "n")

	if a.Len() != 500 || a.NullCount() != 0 {
		t.Fatalf("%d values, %d null", a.Len(), a.NullCount())
	}
	if a.Validity() != nil {
		t.Error("a column with nothing missing kept a validity bitmap")
	}

	values := a.Values[int32]()
	for i := range a.Len() {
		if values[i] != int32(i) {
			t.Fatalf("value %d is %d", i, values[i])
		}
	}
}

// TestReadColumnLegacy reads the file written the old way all round, where the
// pages are the first version and the timestamp is the deprecated one.
func TestReadColumnLegacy(t *testing.T) {
	t.Run("moment", func(t *testing.T) {
		a := readColumn(t, "legacy.parquet", "moment")
		if want := (dtype.Timestamp{Unit: dtype.Nanosecond}); a.DType() != want {
			t.Fatalf("the column is a %s, want a %s", a.DType(), want)
		}
		want := []int64{1787918400123456000, -14182940000000000}
		if got := a.Values[int64](); !slices.Equal(got, want) {
			t.Errorf("got %v, want %v", got, want)
		}
	})

	t.Run("label", func(t *testing.T) {
		a := readColumn(t, "legacy.parquet", "label")
		for i, want := range []string{"now", "then"} {
			if got := string(a.Bytes(i)); got != want {
				t.Errorf("value %d: got %q, want %q", i, got, want)
			}
		}
	})
}

// TestReadColumnRequired reads a column nothing is missing from.
//
// A required column writes no levels at all, not even a run of them saying that
// every value is there, so neither version of the page leaves room for any and
// the values start at the first byte of the body. Reading four bytes of level
// length off the front of one would take the first value apart.
func TestReadColumnRequired(t *testing.T) {
	a := readColumn(t, "alltypes.parquet", "flag")

	if a.Len() != 3 || a.NullCount() != 0 {
		t.Fatalf("%d values, %d null", a.Len(), a.NullCount())
	}
	if a.Validity() != nil {
		t.Error("a required column kept a validity bitmap")
	}
	for i, want := range []bool{true, false, true} {
		if a.Bool(i) != want {
			t.Errorf("value %d: got %v, want %v", i, a.Bool(i), want)
		}
	}
}

// TestReadColumnStruct reads a column two levels deep, which is a field of a
// struct that is itself optional.
//
// It is the flattest column that needs more than one bit of level. A level of
// two is the value, one is a struct that is there with the field missing, and
// nought is no struct at all, and the last two both come out as a null since
// there is nothing else a flat column can say.
func TestReadColumnStruct(t *testing.T) {
	t.Run("y", func(t *testing.T) {
		a := readColumn(t, "plain.parquet", "point.y")
		if a.Len() != 4 || a.NullCount() != 2 {
			t.Fatalf("%d values, %d null", a.Len(), a.NullCount())
		}
		if !a.IsNull(1) || !a.IsNull(2) {
			t.Error("the missing y and the missing point should both be null")
		}
		for i, want := range map[int]float64{0: 2, 3: 6} {
			if got := a.Value[float64](i); got != want {
				t.Errorf("value %d: got %v, want %v", i, got, want)
			}
		}
	})

	// x is a field that cannot be missing in a struct that can, which is one
	// level rather than two and a null wherever the struct is not there.
	t.Run("x", func(t *testing.T) {
		a := readColumn(t, "plain.parquet", "point.x")
		if a.Len() != 4 || a.NullCount() != 1 || !a.IsNull(2) {
			t.Fatalf("%d values, %d null", a.Len(), a.NullCount())
		}
		numbersOf(t, a, []float64{1, 3, 5})
	})
}

// TestReadColumnEmptyChunk reads a chunk with no bytes in it, which is a column
// of a row group with no rows.
//
// It comes back as a column of nothing rather than as an error. A file can hold
// one, and a scan that refused it would refuse a file that is not wrong.
func TestReadColumnEmptyChunk(t *testing.T) {
	b, chunk, column := chunkOf(t, "empty.parquet", "id")

	nothing := *chunk
	nothing.Meta.TotalCompressedSize = 0

	a, err := parquet.ReadColumn(bytes.NewReader(b), int64(len(b)), &nothing, column)
	if err != nil {
		t.Fatalf("ReadColumn: %v", err)
	}
	if a.Len() != 0 || a.NullCount() != 0 {
		t.Fatalf("%d values, %d null", a.Len(), a.NullCount())
	}
	if a.DType() != dtype.Int64 {
		t.Errorf("the column is a %s", a.DType())
	}
}

// TestReadColumnRefused is the columns of the real files that this reader
// cannot read yet.
//
// Each of them is a piece that is still to come rather than a file that is
// wrong, so all three say so with the same error and none of them is read half
// way and handed back.
func TestReadColumnRefused(t *testing.T) {
	cases := []struct {
		name   string
		file   string
		column string
	}{
		{name: "a compressed chunk", file: "chunks.parquet", column: "code"},
		{name: "a repeated column", file: "nested.parquet", column: "tags.list.element"},
		{name: "a decimal", file: "plain.parquet", column: "price"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			b, chunk, column := chunkOf(t, c.file, c.column)
			_, err := parquet.ReadColumn(bytes.NewReader(b), int64(len(b)), chunk, column)
			if !errors.Is(err, parquet.ErrUnsupported) {
				t.Errorf("got %v, want %v", err, parquet.ErrUnsupported)
			}
		})
	}
}

// optional is a column of one nullable int32, which is the shape most of the
// pages built by hand below are for.
func optional() parquet.Column {
	return parquet.Column{
		Path:          []string{"n"},
		Element:       parquet.SchemaElement{Name: "n", Type: parquet.Int32, Repetition: parquet.Optional},
		Type:          dtype.Int32,
		MaxDefinition: 1,
	}
}

// pageV1 builds a page of the first version, which is levels behind four bytes
// of their length and then the values.
func pageV1(levels []byte, values ...int32) parquet.Page {
	body := binary.LittleEndian.AppendUint32(nil, uint32(len(levels)))
	body = append(body, levels...)
	for _, v := range values {
		body = binary.LittleEndian.AppendUint32(body, uint32(v))
	}
	return parquet.Page{
		PageHeader: parquet.PageHeader{
			Kind:               parquet.DataPage,
			Encoding:           parquet.Plain,
			DefinitionEncoding: parquet.RLE,
			NumValues:          int32(len(values)),
		},
		Data: body,
	}
}

// TestNewColumnReaderRefused is the columns a reader cannot be made for.
//
// A Column is a struct rather than something only this package can hand out, so
// the reader is made from what it is given rather than from what Columns would
// have produced.
func TestNewColumnReaderRefused(t *testing.T) {
	deep := optional()
	deep.MaxDefinition = -1

	cases := []struct {
		name   string
		column parquet.Column
		want   error
	}{
		{name: "a column of a depth that is not one", column: deep, want: parquet.ErrFormat},
		{name: "a decimal", column: decimal(), want: parquet.ErrUnsupported},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := parquet.NewColumnReader(c.column); !errors.Is(err, c.want) {
				t.Errorf("got %v, want %v", err, c.want)
			}
		})
	}

	// A column with no type at all, which the builder is the one that knows is
	// not a column. The error is its rather than this package's, so it is the
	// column being named that says it came from here.
	t.Run("a column of no type", func(t *testing.T) {
		blank := optional()
		blank.Type = nil

		_, err := parquet.NewColumnReader(blank)
		if err == nil {
			t.Fatal("a column of no type was read")
		}
		if !strings.Contains(err.Error(), "n") {
			t.Errorf("%v does not say which column it was", err)
		}
	})
}

// decimal is a column of the one type in a real file that cannot be assembled
// yet, which is the same column plain.parquet holds.
func decimal() parquet.Column {
	return parquet.Column{
		Path: []string{"price"},
		Element: parquet.SchemaElement{
			Name: "price", Type: parquet.FixedLenByteArray,
			Repetition: parquet.Optional, TypeLength: 4,
		},
		Type:          dtype.Decimal128{Precision: 9, Scale: 2},
		MaxDefinition: 1,
	}
}

// missing is a column of nothing, which is what a writer that knew no type for
// a column writes when every value of it is missing.
func missing() parquet.Column {
	return parquet.Column{
		Path:          []string{"nothing"},
		Element:       parquet.SchemaElement{Name: "nothing", Type: parquet.Int32, Repetition: parquet.Optional},
		Type:          dtype.Null,
		MaxDefinition: 1,
	}
}

// TestColumnReaderMissing reads a column with no type and no values.
//
// A column of nothing holds nothing, and the physical type a writer chose for
// it means nothing either, so the bytes of the page are never looked at. Both
// levels come out as a null: one says the row is missing and the other says
// there is a value of a type that has none.
func TestColumnReaderMissing(t *testing.T) {
	r := readerOf(t, missing())

	// Two levels of one and then two of nought.
	if err := r.Page(withValues(pageV1([]byte{0x04, 0x01, 0x04, 0x00}), 4)); err != nil {
		t.Fatalf("Page: %v", err)
	}

	a := finish(t, r)
	if a.Len() != 4 || a.NullCount() != 4 {
		t.Fatalf("%d values, %d null", a.Len(), a.NullCount())
	}
	if a.DType() != dtype.Null {
		t.Errorf("the column is a %s", a.DType())
	}
}

// withValues says a page holds a different number of values than it does, which
// is how a page that lost count is built.
func withValues(p parquet.Page, n int32) parquet.Page {
	p.NumValues = n
	return p
}

// TestColumnReaderPackedLevels reads a page whose levels are written in the
// encoding parquet deprecated.
//
// No file here has one, because nothing has written one for years, and a reader
// still has to read them. They are the only levels in the format with no length
// in front of them: how many bytes they take follows from how many values the
// page holds, so a reader that guessed would take the first value as a level.
func TestColumnReaderPackedLevels(t *testing.T) {
	r := readerOf(t, optional())

	// Five values, the middle one missing, at one bit each packed from the top
	// of the byte down. That is one byte with three bits of padding.
	body := []byte{0b11011000}
	for _, v := range []int32{10, 20, 40, 50} {
		body = binary.LittleEndian.AppendUint32(body, uint32(v))
	}

	page := parquet.Page{
		PageHeader: parquet.PageHeader{
			Kind:               parquet.DataPage,
			Encoding:           parquet.Plain,
			DefinitionEncoding: parquet.BitPacked,
			NumValues:          5,
		},
		Data: body,
	}
	if err := r.Page(page); err != nil {
		t.Fatalf("Page: %v", err)
	}

	a := finish(t, r)
	if a.Len() != 5 || a.NullCount() != 1 || !a.IsNull(2) {
		t.Fatalf("%d values, %d null", a.Len(), a.NullCount())
	}
	want := []int32{10, 20, 0, 40, 50}
	if got := a.Values[int32](); !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// TestColumnReaderReuse reads two chunks of the same column with one reader,
// which is what a scan over a file of many row groups does.
func TestColumnReaderReuse(t *testing.T) {
	r := readerOf(t, optional())

	// A repeat of two ones, so both values are present.
	for _, want := range [][]int32{{1, 2}, {3, 4}} {
		if err := r.Page(pageV1([]byte{0x04, 0x01}, want...)); err != nil {
			t.Fatalf("Page: %v", err)
		}
		if r.Len() != 2 {
			t.Fatalf("got %d values, want 2", r.Len())
		}

		a := finish(t, r)
		if got := a.Values[int32](); !slices.Equal(got, want) {
			t.Errorf("got %v, want %v", got, want)
		}
		if r.Len() != 0 {
			t.Errorf("after finishing there are %d values left", r.Len())
		}
	}
}

// TestColumnReaderRefusedPages is the pages a reader has to refuse.
//
// Every one of them is a page that could be read into something rather than
// into an error, which is the reason to refuse it. A page shorter than it says
// would take the next page's bytes as its own values, and a page whose levels
// go deeper than the column does is a page whose levels were read at the wrong
// width.
func TestColumnReaderRefusedPages(t *testing.T) {
	cases := []struct {
		name string
		page parquet.Page
		want error
	}{
		{
			name: "a page with no room for a level length",
			page: parquet.Page{
				PageHeader: parquet.PageHeader{
					Kind: parquet.DataPage, Encoding: parquet.Plain,
					DefinitionEncoding: parquet.RLE, NumValues: 1,
				},
				Data: []byte{0, 0},
			},
			want: parquet.ErrFormat,
		},
		{
			name: "levels longer than the page",
			page: parquet.Page{
				PageHeader: parquet.PageHeader{
					Kind: parquet.DataPage, Encoding: parquet.Plain,
					DefinitionEncoding: parquet.RLE, NumValues: 1,
				},
				Data: []byte{0xff, 0, 0, 0, 0x04, 0x01},
			},
			want: parquet.ErrFormat,
		},
		{
			name: "packed levels longer than the page",
			page: parquet.Page{
				PageHeader: parquet.PageHeader{
					Kind: parquet.DataPage, Encoding: parquet.Plain,
					DefinitionEncoding: parquet.BitPacked, NumValues: 500,
				},
				Data: []byte{0x01},
			},
			want: parquet.ErrFormat,
		},
		{
			// A level of three written at the one bit a column one deep is
			// read at, which the level decoder is the one to refuse.
			name: "a level that does not fit in its width",
			page: pageV1([]byte{0x04, 0x03}, 1, 2),
			want: parquet.ErrFormat,
		},
		{
			// Four levels, all of them present, and two values behind them.
			name: "fewer values than the page says",
			page: withValues(pageV1([]byte{0x08, 0x01}, 1, 2), 4),
			want: parquet.ErrFormat,
		},
		{
			name: "levels in an encoding that is not one",
			page: parquet.Page{
				PageHeader: parquet.PageHeader{
					Kind: parquet.DataPage, Encoding: parquet.Plain,
					DefinitionEncoding: parquet.DeltaBinaryPacked, NumValues: 1,
				},
				Data: []byte{0x04, 0x01},
			},
			want: parquet.ErrUnsupported,
		},
		{
			name: "values in an encoding that is not read yet",
			page: parquet.Page{
				PageHeader: parquet.PageHeader{
					Kind: parquet.DataPage, Encoding: parquet.DeltaBinaryPacked,
					DefinitionEncoding: parquet.RLE, NumValues: 1,
				},
				Data: []byte{0x04, 0x01},
			},
			want: parquet.ErrUnsupported,
		},
		{
			name: "a page holding fewer values than nothing",
			page: parquet.Page{
				PageHeader: parquet.PageHeader{
					Kind: parquet.DataPage, Encoding: parquet.Plain,
					DefinitionEncoding: parquet.RLE, NumValues: -1,
				},
			},
			want: parquet.ErrFormat,
		},
		{
			// The second version of the page says how long its levels are, so
			// levels that do not fit in it are a header that disagrees with the
			// page behind it.
			name: "levels longer than a page of the second version",
			page: parquet.Page{
				PageHeader: parquet.PageHeader{
					Kind: parquet.DataPageV2, Encoding: parquet.Plain,
					DefinitionEncoding: parquet.RLE,
					NumValues:          1, DefinitionLength: 40,
				},
				Data: []byte{0x02, 0x01, 1, 0, 0, 0},
			},
			want: parquet.ErrFormat,
		},
		{
			name: "levels of a length that is not one",
			page: parquet.Page{
				PageHeader: parquet.PageHeader{
					Kind: parquet.DataPageV2, Encoding: parquet.Plain,
					DefinitionEncoding: parquet.RLE,
					NumValues:          1, DefinitionLength: -2,
				},
				Data: []byte{0x02, 0x01, 1, 0, 0, 0},
			},
			want: parquet.ErrFormat,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := readerOf(t, optional())
			if err := r.Page(c.page); !errors.Is(err, c.want) {
				t.Errorf("got %v, want %v", err, c.want)
			}
		})
	}
}

// TestColumnReaderShortValues refuses a page whose values run out before its
// levels say they should.
//
// The three of these are the three ways the values of a page are read. Each one
// says it holds two values and holds one, and the decoder that reads them stops
// where the page does, so the count is the only thing that catches it. A blob
// gets a fourth way to be wrong, which is a length longer than the page it sits
// in, and that comes back from the decoder rather than from the count.
func TestColumnReaderShortValues(t *testing.T) {
	narrow := optional()
	narrow.Type = dtype.Int8

	blob := optional()
	blob.Element.Type = parquet.ByteArray
	blob.Type = dtype.String

	cases := []struct {
		name   string
		column parquet.Column
		body   []byte
	}{
		{name: "a number", column: optional(), body: []byte{1, 0, 0, 0}},
		{name: "a narrowed number", column: narrow, body: []byte{1, 0, 0, 0}},
		{name: "a blob", column: blob, body: []byte{1, 0, 0, 0, 'a'}},
		{name: "a blob longer than the page", column: blob, body: []byte{9, 0, 0, 0, 'a'}},
	}

	// Two levels, both of them present, so both pages want two values.
	levels := []byte{0x02, 0, 0, 0, 0x04, 0x01}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := readerOf(t, c.column)

			page := parquet.Page{
				PageHeader: parquet.PageHeader{
					Kind: parquet.DataPage, Encoding: parquet.Plain,
					DefinitionEncoding: parquet.RLE, NumValues: 2,
				},
				Data: append(slices.Clone(levels), c.body...),
			}
			if err := r.Page(page); !errors.Is(err, parquet.ErrFormat) {
				t.Errorf("got %v, want %v", err, parquet.ErrFormat)
			}
		})
	}
}

// TestColumnReaderDeepLevel refuses a page with a level deeper than the column
// it belongs to.
//
// A level says how many of the groups a column sits in are there, so one deeper
// than the column is a level that means nothing. It is what levels read at the
// wrong width or out of the wrong bytes look like, and taking it as a value
// present would shift every value after it.
func TestColumnReaderDeepLevel(t *testing.T) {
	// A column two deep, which is a field of an optional group, read at the two
	// bits that needs. A level of three fits in the two bits and is still one
	// group deeper than there is.
	deep := optional()
	deep.MaxDefinition = 2

	r := readerOf(t, deep)
	if err := r.Page(pageV1([]byte{0x04, 0x03}, 1, 2)); !errors.Is(err, parquet.ErrFormat) {
		t.Errorf("got %v, want %v", err, parquet.ErrFormat)
	}
}

// TestColumnReaderNullCount refuses a page of the second version whose levels
// do not come to the number of nulls it says it holds.
//
// That count is the one thing in a page that says what its levels should add up
// to, so a page that disagrees with itself is a page that was read wrong, and
// there is nothing else in the format that would catch it.
func TestColumnReaderNullCount(t *testing.T) {
	r := readerOf(t, optional())

	levels := []byte{0x04, 0x01}
	body := append(slices.Clone(levels), 1, 0, 0, 0, 2, 0, 0, 0)
	page := parquet.Page{
		PageHeader: parquet.PageHeader{
			Kind: parquet.DataPageV2, Encoding: parquet.Plain,
			DefinitionEncoding: parquet.RLE,
			NumValues:          2,
			NumNulls:           1,
			DefinitionLength:   int32(len(levels)),
		},
		Data: body,
	}
	if err := r.Page(page); !errors.Is(err, parquet.ErrFormat) {
		t.Errorf("got %v, want %v", err, parquet.ErrFormat)
	}
}

// TestColumnReaderIndexPage steps over a page type that is not a data page.
//
// The format has one and no writer has ever produced it. A column has nothing
// to take from it, and refusing a chunk for holding one would refuse a file
// that is not wrong.
func TestColumnReaderIndexPage(t *testing.T) {
	r := readerOf(t, optional())

	page := parquet.Page{PageHeader: parquet.PageHeader{Kind: parquet.IndexPage}}
	if err := r.Page(page); err != nil {
		t.Fatalf("Page: %v", err)
	}
	if r.Len() != 0 {
		t.Errorf("an index page added %d values", r.Len())
	}
	if r.DType() != dtype.Int32 {
		t.Errorf("the column is a %s", r.DType())
	}
}

// TestReadColumnShort refuses a chunk whose pages do not hold as many values as
// the footer said the chunk does.
//
// The count in the footer is what a scan works from, since it is what says how
// long the column is before a page of it is read. A chunk that comes up short
// would leave a frame with columns of different lengths in it.
func TestReadColumnShort(t *testing.T) {
	b, chunk, column := chunkOf(t, "pages.parquet", "n")

	short := *chunk
	short.Meta.NumValues = 499
	_, err := parquet.ReadColumn(bytes.NewReader(b), int64(len(b)), &short, column)
	if !errors.Is(err, parquet.ErrFormat) {
		t.Errorf("got %v, want %v", err, parquet.ErrFormat)
	}
}

// TestReadColumnBadChunk refuses a chunk that does not sit in the file it says
// it does.
//
// The offsets and the size come out of the footer, which is one part of a file
// making a claim about another, so a chunk that runs off the end and a chunk
// whose pages start in the middle of something else are both a footer that
// cannot be trusted rather than a column to read.
func TestReadColumnBadChunk(t *testing.T) {
	b, chunk, column := chunkOf(t, "pages.parquet", "n")

	t.Run("longer than the file", func(t *testing.T) {
		long := *chunk
		long.Meta.TotalCompressedSize = int64(len(b)) + 1

		_, err := parquet.ReadColumn(bytes.NewReader(b), int64(len(b)), &long, column)
		if !errors.Is(err, parquet.ErrFormat) {
			t.Errorf("got %v, want %v", err, parquet.ErrFormat)
		}
	})

	t.Run("starting in the middle of a page", func(t *testing.T) {
		off := *chunk
		off.Meta.DataPageOffset++

		_, err := parquet.ReadColumn(bytes.NewReader(b), int64(len(b)), &off, column)
		if !errors.Is(err, parquet.ErrFormat) {
			t.Errorf("got %v, want %v", err, parquet.ErrFormat)
		}
	})
}

// BenchmarkReadColumn reads a column out of a file, levels and all.
//
// This is the first thing in the package that measures the whole path rather
// than one decoder, which is the figure worth having: a page comes out of the
// chunk, its levels are decoded next to its values, and both are put into an
// array. The two columns are the two cases, one with nothing missing and one a
// third missing, and the difference between them is what the nulls cost.
func BenchmarkReadColumn(b *testing.B) {
	for _, column := range []string{"n", "maybe"} {
		b.Run(column, func(b *testing.B) {
			file, chunk, c := chunkOf(&testing.T{}, "pages.parquet", column)
			r := bytes.NewReader(file)

			b.SetBytes(chunk.Meta.TotalCompressedSize)
			for b.Loop() {
				if _, err := parquet.ReadColumn(r, int64(len(file)), chunk, c); err != nil {
					b.Fatalf("ReadColumn: %v", err)
				}
			}
		})
	}
}
