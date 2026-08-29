package parquet_test

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/parquet"
)

// tableOf builds a table from one filler per field, each of which appends that
// column's values to a builder of the field's type.
func tableOf(tb testing.TB, fields []dtype.Field, fill ...func(*array.Builder)) *array.Table {
	tb.Helper()

	t := &array.Table{Schema: dtype.Schema{Fields: fields}}
	for i, f := range fields {
		b, err := array.NewBuilder(f.Type)
		if err != nil {
			tb.Fatalf("NewBuilder(%s): %v", f.Type, err)
		}
		fill[i](b)

		c, err := array.NewChunked(f.Type, b.Finish())
		if err != nil {
			tb.Fatalf("NewChunked(%s): %v", f.Type, err)
		}
		t.Columns = append(t.Columns, c)
	}
	return t
}

// written is a table written and read back, which is what nearly every test
// here is asking about.
func written(tb testing.TB, t *array.Table) *array.Table {
	tb.Helper()

	got, _ := writtenBytes(tb, t, nil)
	return got
}

// writtenBytes is written with the file itself as well, for the tests that look
// at the footer rather than at the values.
func writtenBytes(tb testing.TB, t *array.Table, opts *parquet.WriteOptions) (*array.Table, []byte) {
	tb.Helper()

	var buf bytes.Buffer
	n, err := parquet.Write(&buf, t, opts)
	if err != nil {
		tb.Fatalf("Write: %v", err)
	}
	if n != int64(buf.Len()) {
		tb.Fatalf("Write says it wrote %d bytes and wrote %d", n, buf.Len())
	}

	got, err := parquet.Read(bytes.NewReader(buf.Bytes()), int64(buf.Len()), nil)
	if err != nil {
		tb.Fatalf("Read: %v", err)
	}
	return got, buf.Bytes()
}

// cells is a column read out as text, one entry a row, so two columns of any
// type can be compared and a failure names the row it is in.
func cells(c *array.Chunked) []string {
	out := make([]string, 0, c.Len())
	for _, a := range c.Chunks() {
		for i := range a.Len() {
			if a.IsNull(i) {
				out = append(out, "null")
				continue
			}
			out = append(out, cell(a, i))
		}
	}
	return out
}

// cell is one value as text. The types are the ones this writer writes, and
// anything else is a test asking about a column it cannot have got back.
func cell(a *array.Array, i int) string {
	// A column that came out of a file is usually indices into a dictionary, and
	// what it holds is what those indices point at.
	if d := a.Dictionary(); d != nil {
		return cell(d, a.Index(i))
	}

	switch a.DType().Kind() {
	case dtype.BoolKind:
		return strconv.FormatBool(a.Bool(i))
	case dtype.Int8Kind:
		return strconv.Itoa(int(a.Value[int8](i)))
	case dtype.Int16Kind:
		return strconv.Itoa(int(a.Value[int16](i)))
	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		return strconv.Itoa(int(a.Value[int32](i)))
	case dtype.Int64Kind, dtype.Time64Kind, dtype.TimestampKind:
		return strconv.FormatInt(a.Value[int64](i), 10)
	case dtype.Uint8Kind:
		return strconv.Itoa(int(a.Value[uint8](i)))
	case dtype.Uint16Kind:
		return strconv.Itoa(int(a.Value[uint16](i)))
	case dtype.Uint32Kind:
		return strconv.FormatUint(uint64(a.Value[uint32](i)), 10)
	case dtype.Uint64Kind:
		return strconv.FormatUint(a.Value[uint64](i), 10)
	case dtype.Float32Kind:
		return fmt.Sprint(a.Value[float32](i))
	case dtype.Float64Kind:
		return fmt.Sprint(a.Value[float64](i))
	default:
		return fmt.Sprintf("%q", a.Bytes(i))
	}
}

// same fails the test if the two tables do not hold the same schema and the
// same values.
func same(t *testing.T, got, want *array.Table) {
	t.Helper()

	if !got.Schema.Equal(want.Schema) {
		t.Fatalf("the schema came back different\n got %s\nwant %s", got.Schema, want.Schema)
	}
	if got.NumRows() != want.NumRows() {
		t.Fatalf("%d rows came back, want %d", got.NumRows(), want.NumRows())
	}

	for i := range want.Columns {
		g, w := cells(got.Columns[i]), cells(want.Columns[i])
		for row := range w {
			if g[row] != w[row] {
				t.Errorf("%s row %d: got %s, want %s",
					want.Schema.Fields[i].Name, row, g[row], w[row])
			}
		}
	}
}

// everyType is a table with one column of every type this writer writes, half
// of them nullable and holding a null, which is the table most of these tests
// are about.
func everyType(tb testing.TB) *array.Table {
	tb.Helper()

	return tableOf(tb,
		[]dtype.Field{
			{Name: "flag", Type: dtype.Bool},
			{Name: "small", Type: dtype.Int8, Nullable: true},
			{Name: "short", Type: dtype.Int16},
			{Name: "count", Type: dtype.Int32, Nullable: true},
			{Name: "total", Type: dtype.Int64},
			{Name: "byte", Type: dtype.Uint8, Nullable: true},
			{Name: "word", Type: dtype.Uint16},
			{Name: "unsigned", Type: dtype.Uint32, Nullable: true},
			{Name: "big", Type: dtype.Uint64},
			{Name: "ratio", Type: dtype.Float32, Nullable: true},
			{Name: "weight", Type: dtype.Float64},
			{Name: "name", Type: dtype.String, Nullable: true},
			{Name: "blob", Type: dtype.Binary},
			{Name: "hash", Type: dtype.FixedSizeBinary{ByteWidth: 4}, Nullable: true},
			{Name: "day", Type: dtype.Date32},
			{Name: "clock", Type: dtype.Time32{Unit: dtype.Millisecond}, Nullable: true},
			{Name: "precise", Type: dtype.Time64{Unit: dtype.Nanosecond}},
			{Name: "moment", Type: dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}, Nullable: true},
			{Name: "local", Type: dtype.Timestamp{Unit: dtype.Millisecond}},
		},
		func(b *array.Builder) { b.AppendBools([]bool{true, false, true, true}) },
		func(b *array.Builder) { b.AppendValues([]int8{-128, 127, 0}); b.AppendNull() },
		func(b *array.Builder) { b.AppendValues([]int16{-32768, 32767, 0, 1}) },
		func(b *array.Builder) { b.AppendNull(); b.AppendValues([]int32{-1, 0, 2147483647}) },
		func(b *array.Builder) { b.AppendValues([]int64{-9223372036854775808, 0, 1, 9223372036854775807}) },
		func(b *array.Builder) { b.AppendValues([]uint8{0, 255}); b.AppendNull(); b.Append[uint8](7) },
		func(b *array.Builder) { b.AppendValues([]uint16{0, 65535, 1, 2}) },
		func(b *array.Builder) { b.AppendValues([]uint32{0, 4294967295}); b.AppendNull(); b.Append[uint32](3) },
		func(b *array.Builder) { b.AppendValues([]uint64{0, 18446744073709551615, 1, 2}) },
		func(b *array.Builder) { b.AppendValues([]float32{-1.5, 0, 3.25}); b.AppendNull() },
		func(b *array.Builder) { b.AppendValues([]float64{-1.5, 0, 3.25, 1e300}) },
		func(b *array.Builder) {
			b.AppendString("one")
			b.AppendNull()
			b.AppendString("")
			b.AppendString("three")
		},
		func(b *array.Builder) {
			for _, v := range [][]byte{{0}, {1, 2}, {}, {3, 4, 5}} {
				b.AppendBytes(v)
			}
		},
		func(b *array.Builder) {
			b.AppendBytes([]byte("abcd"))
			b.AppendNull()
			b.AppendBytes([]byte{0, 0, 0, 0})
			b.AppendBytes([]byte("wxyz"))
		},
		func(b *array.Builder) { b.AppendValues([]int32{0, 19000, -1, 1}) },
		func(b *array.Builder) { b.AppendValues([]int32{0, 86399999}); b.AppendNull(); b.Append[int32](1) },
		func(b *array.Builder) { b.AppendValues([]int64{0, 86399999999999, 1, 2}) },
		func(b *array.Builder) { b.AppendNull(); b.AppendValues([]int64{0, 1700000000000000, -1}) },
		func(b *array.Builder) { b.AppendValues([]int64{0, 1700000000000, -1, 1}) },
	)
}

// TestWriteRoundTrip writes a column of every type this writer writes and reads
// it back.
//
// This is the test the rest of them lean on. Every column here is one the schema
// writer maps, one the plain encoder writes and one the reader assembles, and
// the values are the ends of each type's range rather than small numbers,
// because a value written at the wrong width or with the wrong sign comes back
// looking fine until it is the largest one.
func TestWriteRoundTrip(t *testing.T) {
	want := everyType(t)
	same(t, written(t, want), want)
}

// TestWriteEmpty writes a table with no rows in it.
//
// A file of no rows is still a file: it has a schema, a footer and no row
// groups, and reading it back gives the columns with nothing in them. A writer
// that skipped the footer because there was nothing to describe would produce
// four bytes that no reader would open.
func TestWriteEmpty(t *testing.T) {
	want := tableOf(t,
		[]dtype.Field{
			{Name: "id", Type: dtype.Int64},
			{Name: "name", Type: dtype.String, Nullable: true},
		},
		func(*array.Builder) {},
		func(*array.Builder) {},
	)

	got, raw := writtenBytes(t, want, nil)
	same(t, got, want)

	m, err := parquet.ReadMetadata(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	if len(m.RowGroups) != 0 {
		t.Errorf("a table of no rows wrote %d row groups", len(m.RowGroups))
	}
	if m.NumRows != 0 {
		t.Errorf("the footer says %d rows", m.NumRows)
	}
}

// TestWriteNulls writes columns that are entirely missing and columns that are
// entirely there.
//
// Those are the two ends of the level encoding and both of them come out as a
// single run, which is what makes the levels of a real column cost nothing. A
// column of nothing but nulls also writes no values at all, so the page is its
// levels and the reader has to put a row back for each of them without anything
// to read.
func TestWriteNulls(t *testing.T) {
	want := tableOf(t,
		[]dtype.Field{
			{Name: "missing", Type: dtype.Int64, Nullable: true},
			{Name: "present", Type: dtype.Int64, Nullable: true},
			{Name: "nothing", Type: dtype.Null, Nullable: true},
			{Name: "words", Type: dtype.String, Nullable: true},
		},
		func(b *array.Builder) { b.AppendNulls(5) },
		func(b *array.Builder) { b.AppendValues([]int64{1, 2, 3, 4, 5}) },
		func(b *array.Builder) { b.AppendNulls(5) },
		func(b *array.Builder) { b.AppendNulls(5) },
	)

	same(t, written(t, want), want)
}

// TestWriteRowGroups writes a table in more than one row group.
//
// The row groups are what a reader skips whole, so a writer that got the row
// counts or the offsets of the second one wrong would produce a file that reads
// the first group and then reads nonsense. The chunking is checked as well: a
// column comes back with one chunk per row group, which is what the reader does
// with a file any other writer produced.
func TestWriteRowGroups(t *testing.T) {
	want := everyType(t)
	got, raw := writtenBytes(t, want, &parquet.WriteOptions{RowGroupSize: 3})
	same(t, got, want)

	m, err := parquet.ReadMetadata(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	if len(m.RowGroups) != 2 {
		t.Fatalf("%d row groups, want 2", len(m.RowGroups))
	}
	for i, n := range []int64{3, 1} {
		if m.RowGroups[i].NumRows != n {
			t.Errorf("group %d holds %d rows, want %d", i, m.RowGroups[i].NumRows, n)
		}
		if got := m.RowGroups[i].Ordinal; got != int16(i) {
			t.Errorf("group %d says it is number %d", i, got)
		}
	}

	if got := got.Columns[0].NumChunks(); got != 2 {
		t.Errorf("a column came back in %d chunks, want one per row group", got)
	}
}

// TestWritePages writes a column across more than one page.
//
// A page has no length in front of it, so a reader finds the second one by
// reading the first header and adding up. That makes the second page of a chunk
// the thing a writer gets wrong, and a test that only ever wrote one page would
// never find out.
func TestWritePages(t *testing.T) {
	want := tableOf(t,
		[]dtype.Field{
			{Name: "total", Type: dtype.Int64, Nullable: true},
			{Name: "name", Type: dtype.String},
		},
		func(b *array.Builder) {
			for i := range 400 {
				if i%7 == 0 {
					b.AppendNull()
					continue
				}
				b.Append(int64(i))
			}
		},
		func(b *array.Builder) {
			for i := range 400 {
				b.AppendString(strings.Repeat("x", i%17))
			}
		},
	)

	// Small enough that both columns take several pages, and not so small that
	// a page holds one value, which would be a different test.
	got, raw := writtenBytes(t, want, &parquet.WriteOptions{PageSize: 64})
	same(t, got, want)

	m, err := parquet.ReadMetadata(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	for i, chunk := range m.RowGroups[0].Columns {
		pages, err := parquet.ReadPages(bytes.NewReader(raw), int64(len(raw)), &chunk)
		if err != nil {
			t.Fatalf("ReadPages: %v", err)
		}
		if n := len(allPages(t, pages)); n < 2 {
			t.Errorf("column %d was written in %d pages", i, n)
		}
	}
}

// TestWriteOptions checks that the defaults are the ones documented and that a
// caller who says something gets what they said.
func TestWriteOptions(t *testing.T) {
	want := everyType(t)

	_, raw := writtenBytes(t, want, nil)
	m, err := parquet.ReadMetadata(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	if len(m.RowGroups) != 1 {
		t.Errorf("four rows went in %d row groups, want one", len(m.RowGroups))
	}
	if m.CreatedBy != "kuma" {
		t.Errorf("the footer says it was written by %q", m.CreatedBy)
	}
	if m.Version != 1 {
		t.Errorf("the footer says version %d", m.Version)
	}

	_, raw = writtenBytes(t, want, &parquet.WriteOptions{CreatedBy: "a test"})
	m, err = parquet.ReadMetadata(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	if m.CreatedBy != "a test" {
		t.Errorf("the footer says it was written by %q", m.CreatedBy)
	}
}

// TestWriteChunk checks what the footer says about a column chunk against what
// is actually at the offset it gives.
//
// The offsets are the only thing in the footer a reader cannot check, since a
// page has nothing in front of it saying it is a page. An offset that is one
// byte out gives a file that opens and reads garbage, so this reads a chunk
// through its own metadata and compares it against reading the file whole.
func TestWriteChunk(t *testing.T) {
	want := everyType(t)
	_, raw := writtenBytes(t, want, &parquet.WriteOptions{RowGroupSize: 2})

	r := bytes.NewReader(raw)
	m, err := parquet.ReadMetadata(r, int64(len(raw)))
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	cols, err := m.Columns()
	if err != nil {
		t.Fatalf("Columns: %v", err)
	}

	for g, group := range m.RowGroups {
		for i := range group.Columns {
			chunk := &group.Columns[i]
			if chunk.Meta.Codec != parquet.Uncompressed {
				t.Errorf("%s is written with the %s codec", cols[i].Name(), chunk.Meta.Codec)
			}
			if chunk.Meta.NumValues != group.NumRows {
				t.Errorf("%s holds %d values in a group of %d rows",
					cols[i].Name(), chunk.Meta.NumValues, group.NumRows)
			}

			a, err := parquet.ReadColumn(r, int64(len(raw)), chunk, cols[i])
			if err != nil {
				t.Fatalf("ReadColumn(%s): %v", cols[i].Name(), err)
			}
			if a.Len() != int(group.NumRows) {
				t.Errorf("group %d of %s came back with %d rows, want %d",
					g, cols[i].Name(), a.Len(), group.NumRows)
			}
		}
	}
}

// TestWriteEncodings checks the encodings a plainly written chunk says it holds.
//
// A required column writes no levels, so saying it uses the run length encoding
// would have a reader believe there is something in the page that is not there.
func TestWriteEncodings(t *testing.T) {
	want := tableOf(t,
		[]dtype.Field{
			{Name: "id", Type: dtype.Int64},
			{Name: "maybe", Type: dtype.Int64, Nullable: true},
		},
		func(b *array.Builder) { b.AppendValues([]int64{1, 2}) },
		func(b *array.Builder) { b.AppendValues([]int64{1}); b.AppendNull() },
	)

	_, raw := writtenBytes(t, want, &parquet.WriteOptions{Plain: true})
	m, err := parquet.ReadMetadata(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}

	for i, encodings := range [][]parquet.Encoding{
		{parquet.Plain},
		{parquet.Plain, parquet.RLE},
	} {
		got := m.RowGroups[0].Columns[i].Meta.Encodings
		if len(got) != len(encodings) {
			t.Errorf("column %d says it uses %v, want %v", i, got, encodings)
			continue
		}
		for j := range encodings {
			if got[j] != encodings[j] {
				t.Errorf("column %d says it uses %v, want %v", i, got, encodings)
				break
			}
		}
	}
}

// empty is a table of the given types with no rows in it, which is the only way
// to hold some of them: nothing in kuma builds a large string, and a column of
// one is something that arrived from somewhere else.
func empty(tb testing.TB, fields ...dtype.Field) *array.Table {
	tb.Helper()

	t := &array.Table{Schema: dtype.Schema{Fields: fields}}
	for _, f := range fields {
		c, err := array.NewChunked(f.Type)
		if err != nil {
			tb.Fatalf("NewChunked(%s): %v", f.Type, err)
		}
		t.Columns = append(t.Columns, c)
	}
	return t
}

// TestWriteCollapsed writes the types kuma has two of and parquet has one of.
//
// Each of them comes back as the one parquet has, which is what the schema
// writer says will happen and what a caller reading their own file back needs to
// know before they wonder where their type went. Nothing is lost by it, since
// the two differ in how long a column of them may be rather than in what a value
// is.
func TestWriteCollapsed(t *testing.T) {
	got := written(t, empty(t,
		dtype.Field{Name: "name", Type: dtype.LargeString, Nullable: true},
		dtype.Field{Name: "blob", Type: dtype.LargeBinary},
		dtype.Field{Name: "code", Type: dtype.Dictionary{Index: dtype.Int32, Value: dtype.String}, Nullable: true},
	))

	for i, want := range []dtype.DataType{dtype.String, dtype.Binary, dtype.String} {
		if !dtype.Equal(got.Schema.Fields[i].Type, want) {
			t.Errorf("column %d came back as %s, want %s", i, got.Schema.Fields[i].Type, want)
		}
	}
}

// TestWriteDictionary writes back a file that was read dictionary encoded.
//
// This is the round trip a caller doing anything at all to a parquet file makes:
// most of a real file is dictionary encoded, the reader hands those columns back
// encoded because that is the shape the kernels want them in, and writing one
// out again expands it and builds a dictionary of its own, since in parquet a
// dictionary is a decision about a chunk rather than a type. The values are what
// is checked rather than the encoding, because the point is that nothing was
// lost on the way through.
func TestWriteDictionary(t *testing.T) {
	want := readTable(t, "dictionary.parquet", &parquet.Options{Dictionary: true})
	if want.Columns[0].Chunks()[0].Dictionary() == nil {
		t.Fatal("the file came back with nothing dictionary encoded in it")
	}

	got := written(t, want)
	if !dtype.Equal(got.Schema.Fields[0].Type, dtype.String) {
		t.Errorf("the codes came back as %s, want a string", got.Schema.Fields[0].Type)
	}

	for i := range want.Columns {
		g, w := cells(got.Columns[i]), cells(want.Columns[i])
		for row := range w {
			if g[row] != w[row] {
				t.Fatalf("%s row %d: got %s, want %s",
					want.Schema.Fields[i].Name, row, g[row], w[row])
			}
		}
	}
}

// TestWriteFile writes a file to disk and reads it back through ReadFile, which
// is the pair of calls a caller uses and the one nothing else here covers.
func TestWriteFile(t *testing.T) {
	want := everyType(t)
	path := filepath.Join(t.TempDir(), "out.parquet")

	if err := parquet.WriteFile(path, want, nil); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	got, err := parquet.ReadFile(path, nil)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	same(t, got, want)
}

// TestWriteFileRefused checks the two ways WriteFile fails: the file cannot be
// made, and the table cannot be written once it has been.
//
// The second is the one worth having. The file is open by then, so the error the
// caller sees has to be the one from the write rather than the one from closing
// a file that nothing went wrong with.
func TestWriteFileRefused(t *testing.T) {
	dir := t.TempDir()

	if err := parquet.WriteFile(filepath.Join(dir, "no", "such", "dir", "out.parquet"), everyType(t), nil); err == nil {
		t.Error("writing into a directory that is not there worked")
	}

	path := filepath.Join(dir, "out.parquet")
	bad := tableOf(t,
		[]dtype.Field{{Name: "took", Type: dtype.Duration{Unit: dtype.Nanosecond}}},
		func(b *array.Builder) { b.AppendValues([]int64{1}) },
	)

	err := parquet.WriteFile(path, bad, nil)
	if !errors.Is(err, parquet.ErrUnsupported) {
		t.Errorf("writing a duration gave %v, want an unsupported error", err)
	}
	if err != nil && !strings.Contains(err.Error(), path) {
		t.Errorf("the error does not name the file: %v", err)
	}
}

// TestWriteRefused is every table this writer will not write.
//
// The types parquet cannot hold are refused by the schema writer and the ones it
// can hold that nothing here encodes yet are refused by this one, and either way
// the caller gets a name and a reason rather than a file that reads back as
// something else.
func TestWriteRefused(t *testing.T) {
	for _, c := range []struct {
		name  string
		table func(testing.TB) *array.Table
	}{
		{
			name: "a type parquet has no way of writing",
			table: func(tb testing.TB) *array.Table {
				return tableOf(tb,
					[]dtype.Field{{Name: "took", Type: dtype.Duration{Unit: dtype.Nanosecond}}},
					func(b *array.Builder) { b.AppendValues([]int64{1}) })
			},
		},
		{
			name: "a decimal, which parquet holds and nothing here encodes yet",
			table: func(tb testing.TB) *array.Table {
				return tableOf(tb,
					[]dtype.Field{{Name: "price", Type: dtype.Decimal128{Precision: 9, Scale: 2}}},
					func(b *array.Builder) { b.AppendBytes(make([]byte, 16)) })
			},
		},
		{
			name: "a list, which needs repetition levels",
			table: func(tb testing.TB) *array.Table {
				return empty(tb, dtype.Field{
					Name: "tags", Type: dtype.List{Elem: dtype.String}, Nullable: true,
				})
			},
		},
		{
			name: "a map, which is a list of pairs",
			table: func(tb testing.TB) *array.Table {
				return empty(tb, dtype.Field{
					Name: "labels", Type: dtype.Map{Key: dtype.String, Value: dtype.String}, Nullable: true,
				})
			},
		},
		{
			name: "a struct, whose fields are columns of their own",
			table: func(tb testing.TB) *array.Table {
				return empty(tb, dtype.Field{
					Name: "point", Type: dtype.Struct{Fields: []dtype.Field{
						{Name: "x", Type: dtype.Float64},
					}},
				})
			},
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := parquet.Write(&bytes.Buffer{}, c.table(t), nil)
			if !errors.Is(err, parquet.ErrUnsupported) {
				t.Fatalf("that gave %v, want an unsupported error", err)
			}
		})
	}
}

// TestWriteMismatched is a table whose columns do not match the schema it
// carries.
//
// The schema is what the file says the columns are and the columns are what goes
// in the pages, so a table where the two disagree is a file that says one thing
// and holds another. Refusing it here is the only place that can happen, since
// nothing downstream looks at the two together again.
func TestWriteMismatched(t *testing.T) {
	fields := []dtype.Field{
		{Name: "id", Type: dtype.Int64},
		{Name: "name", Type: dtype.String},
	}

	short := tableOf(t, fields,
		func(b *array.Builder) { b.AppendValues([]int64{1, 2}) },
		func(b *array.Builder) { b.AppendString("one"); b.AppendString("two") })
	short.Columns = short.Columns[:1]
	if _, err := parquet.Write(&bytes.Buffer{}, short, nil); err == nil {
		t.Error("a table with a column missing was written")
	}

	ragged := tableOf(t, fields,
		func(b *array.Builder) { b.AppendValues([]int64{1, 2}) },
		func(b *array.Builder) { b.AppendString("one") })
	if _, err := parquet.Write(&bytes.Buffer{}, ragged, nil); err == nil {
		t.Error("a table whose columns are different lengths was written")
	}
}

// TestWriteTooManyGroups asks for row groups small enough that the file could
// not say which group is which.
//
// The ordinal of a row group is two bytes wide. Nothing sensible gets near it,
// and a caller who asked for a group per row on a table of any size would
// otherwise get a file whose group numbers wrapped round.
func TestWriteTooManyGroups(t *testing.T) {
	b, err := array.NewBuilder(dtype.Int32)
	if err != nil {
		t.Fatal(err)
	}
	b.AppendValues(make([]int32, 40000))

	c, err := array.NewChunked(dtype.Int32, b.Finish())
	if err != nil {
		t.Fatal(err)
	}

	tbl := &array.Table{
		Schema:  dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int32}}},
		Columns: []*array.Chunked{c},
	}
	if _, err := parquet.Write(&bytes.Buffer{}, tbl, &parquet.WriteOptions{RowGroupSize: 1}); err == nil {
		t.Error("a file of forty thousand row groups was written")
	}
}

// TestWriteBroken checks that a writer that stops working is reported rather
// than written past.
//
// The three places it can happen are the magic, a page and the footer, and each
// of them is a different call. What comes back is how much got out before it
// stopped, which is what a caller cleaning up after a half written file wants.
func TestWriteBroken(t *testing.T) {
	want := everyType(t)

	for _, after := range []int{0, 4, 200} {
		w := &stopper{after: after}
		n, err := parquet.Write(w, want, &parquet.WriteOptions{PageSize: 16})
		if err == nil {
			t.Fatalf("writing to a writer that stops after %d bytes worked", after)
		}
		if n < int64(after) {
			t.Errorf("the write says %d bytes went out and %d did", n, after)
		}
	}
}

// stopper writes the first few bytes it is given and refuses everything after
// that, which is a disk filling up part way through a file.
type stopper struct {
	after int
	n     int
}

func (s *stopper) Write(p []byte) (int, error) {
	if s.n >= s.after {
		return 0, os.ErrClosed
	}
	n := min(len(p), s.after-s.n)
	s.n += n
	if n < len(p) {
		return n, os.ErrClosed
	}
	return n, nil
}

// BenchmarkWrite writes a million rows of the shapes a real file is made of: a
// required number, one that is a tenth missing, and a string.
func BenchmarkWrite(b *testing.B) {
	t := tableOf(b,
		[]dtype.Field{
			{Name: "id", Type: dtype.Int64},
			{Name: "total", Type: dtype.Int64, Nullable: true},
			{Name: "name", Type: dtype.String},
		},
		func(bl *array.Builder) {
			for i := range 1 << 20 {
				bl.Append(int64(i))
			}
		},
		func(bl *array.Builder) {
			for i := range 1 << 20 {
				if i%10 == 0 {
					bl.AppendNull()
					continue
				}
				bl.Append(int64(i))
			}
		},
		func(bl *array.Builder) {
			for i := range 1 << 20 {
				bl.AppendString(names[i%len(names)])
			}
		},
	)

	var buf bytes.Buffer
	b.ResetTimer()
	for b.Loop() {
		buf.Reset()
		if _, err := parquet.Write(&buf, t, nil); err != nil {
			b.Fatal(err)
		}
	}
	b.SetBytes(int64(buf.Len()))
}

// names is what the string column of the benchmark holds, which is a handful of
// values repeated, since that is what a real column of names looks like.
var names = []string{"alice", "bob", "carol", "dave", "erin", "frank", "grace"}
