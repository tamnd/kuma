package parquet_test

import (
	"bytes"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/parquet"
)

// readTable reads one of the files in testdata and fails the test if it cannot.
func readTable(tb testing.TB, name string, opts *parquet.Options) *array.Table {
	tb.Helper()

	t, err := parquet.ReadFile(filepath.Join("testdata", name), opts)
	if err != nil {
		tb.Fatalf("ReadFile(%s): %v", name, err)
	}
	return t
}

// text is a whole column of strings as strings, chunks and encoding and all.
func text(c *array.Chunked) []string {
	out := make([]string, 0, c.Len())
	for _, a := range c.Chunks() {
		out = append(out, textColumn(a)...)
	}
	return out
}

// numbers is the same for a column of numbers.
func numbers[T array.Numeric](c *array.Chunked) []T {
	out := make([]T, 0, c.Len())
	for _, a := range c.Chunks() {
		out = append(out, numberColumn[T](a)...)
	}
	return out
}

// TestRead reads a whole file of two row groups.
//
// The columns come back as the types the file says they are, one chunk per row
// group, and the rows are in the order the file holds them. The chunking is
// worth checking rather than assuming: a reader that joined the row groups into
// one array per column would be copying every value of every file it read to
// gain nothing.
func TestRead(t *testing.T) {
	tab := readTable(t, "chunks.parquet", nil)

	if got, want := tab.NumRows(), 6; got != want {
		t.Errorf("the table holds %d rows, want %d", got, want)
	}
	if got, want := tab.NumCols(), 2; got != want {
		t.Fatalf("the table holds %d columns, want %d", got, want)
	}

	want := dtype.Schema{Fields: []dtype.Field{
		{Name: "code", Type: dtype.String, Nullable: true},
		{Name: "n", Type: dtype.Int64, Nullable: true},
	}, Metadata: dtype.Metadata{}}
	if got := tab.Schema; !got.Equal(want) {
		t.Errorf("the schema is %v, want %v", got, want)
	}

	for i, c := range tab.Columns {
		if got, want := c.NumChunks(), 2; got != want {
			t.Errorf("column %d came back in %d chunks, want %d", i, got, want)
		}
	}

	if got, want := text(tab.Columns[0]), []string{"GB", "JP", "US", "FR", "DE", "GB"}; !slices.Equal(got, want) {
		t.Errorf("the codes are %q, want %q", got, want)
	}
	if got, want := numbers[int64](tab.Columns[1]), []int64{0, 1, 2, 3, 4, 5}; !slices.Equal(got, want) {
		t.Errorf("the numbers are %v, want %v", got, want)
	}
}

// TestReadDecodes checks that the encoding a file used does not reach the
// caller.
//
// Both columns of this file were written as indices into a dictionary, which
// pyarrow does to nearly everything and does here to a column of six integers.
// A caller who asked for the file wants the column the schema names, so what
// comes back is a column of strings and a column of int64.
func TestReadDecodes(t *testing.T) {
	tab := readTable(t, "chunks.parquet", nil)

	for i, c := range tab.Columns {
		if _, ok := c.DType().(dtype.Dictionary); ok {
			t.Errorf("column %d came back as %s, want it decoded", i, c.DType())
		}
		for j, a := range c.Chunks() {
			if a.Dictionary() != nil {
				t.Errorf("column %d chunk %d is still encoded", i, j)
			}
		}
	}
}

// TestReadDictionary keeps the encoding instead.
//
// The type says so and the values are the same either way, which is the whole
// point: the encoding is a way of storing the column and not a different
// column.
func TestReadDictionary(t *testing.T) {
	tab := readTable(t, "chunks.parquet", &parquet.Options{Dictionary: true})

	want := dtype.Dictionary{Index: dtype.Int32, Value: dtype.String}
	if got := tab.Columns[0].DType(); !dtype.Equal(got, want) {
		t.Errorf("the codes came back as %s, want %s", got, want)
	}
	if got := tab.Schema.Fields[0].Type; !dtype.Equal(got, want) {
		t.Errorf("the schema says %s, want %s", got, want)
	}

	if got, want := text(tab.Columns[0]), []string{"GB", "JP", "US", "FR", "DE", "GB"}; !slices.Equal(got, want) {
		t.Errorf("the codes are %q, want %q", got, want)
	}
	if got, want := numbers[int64](tab.Columns[1]), []int64{0, 1, 2, 3, 4, 5}; !slices.Equal(got, want) {
		t.Errorf("the numbers are %v, want %v", got, want)
	}
}

// TestReadNulls reads a file with a column of missing values in it.
//
// The nulls survive the read and are counted, which is what a caller filtering
// them out is going to ask the column for rather than walking it.
func TestReadNulls(t *testing.T) {
	tab := readTable(t, "dictionary.parquet", nil)

	if got, want := tab.NumRows(), 1000; got != want {
		t.Fatalf("the table holds %d rows, want %d", got, want)
	}
	if got, want := tab.Columns[1].NullCount(), 143; got != want {
		t.Errorf("the sizes hold %d nulls, want %d", got, want)
	}

	// Every seventh row, counting the first, is the missing one.
	sizes := tab.Columns[1]
	for _, i := range []int{0, 7, 994} {
		if !sizes.IsNull(i) {
			t.Errorf("row %d is not missing", i)
		}
	}
	if sizes.IsNull(1) {
		t.Error("row 1 is missing")
	}
}

// TestReadProjection reads two columns of a file and nothing else.
//
// The file holds a decimal column, which is not assembled yet, so reading the
// whole of it fails. That the projection works is what says the columns that
// were not named were not read: a reader that read them all and threw the rest
// away would fail here for the same reason reading the file does.
func TestReadProjection(t *testing.T) {
	tab := readTable(t, "alltypes.parquet", &parquet.Options{Columns: []string{"total", "name"}})

	if got, want := fieldNames(tab.Schema), []string{"total", "name"}; !slices.Equal(got, want) {
		t.Errorf("the table holds %q, want %q", got, want)
	}
	if got, want := numbers[int64](tab.Columns[0]), []int64{100, 200, 300}; !slices.Equal(got, want) {
		t.Errorf("the totals are %v, want %v", got, want)
	}
	if got, want := text(tab.Columns[1]), []string{"GB", "JP", ""}; !slices.Equal(got, want) {
		t.Errorf("the names are %q, want %q", got, want)
	}
	if !tab.Columns[1].IsNull(2) {
		t.Error("the missing name is not missing")
	}

	if _, err := parquet.ReadFile(filepath.Join("testdata", "alltypes.parquet"), nil); err == nil {
		t.Error("reading the whole file worked, so the projection proves nothing")
	}
}

// TestReadProjectionOrder checks that a table holds its columns in the order
// they were asked for rather than the order the file holds them.
//
// Naming a column twice reads it twice, which is a waste and not a mistake, and
// the second copy is a column of its own rather than the same one.
func TestReadProjectionOrder(t *testing.T) {
	tab := readTable(t, "chunks.parquet", &parquet.Options{Columns: []string{"n", "code", "n"}})

	if got, want := fieldNames(tab.Schema), []string{"n", "code", "n"}; !slices.Equal(got, want) {
		t.Errorf("the table holds %q, want %q", got, want)
	}
	if tab.Columns[0] == tab.Columns[2] {
		t.Error("the two copies of n are the same column")
	}
	if got, want := numbers[int64](tab.Columns[2]), []int64{0, 1, 2, 3, 4, 5}; !slices.Equal(got, want) {
		t.Errorf("the second copy of n is %v, want %v", got, want)
	}
}

// TestReadNoColumns projects a file down to nothing, which is what counting the
// rows of one costs nothing with.
//
// A table of no columns has no rows to speak of either, since the rows are what
// the columns hold. The count is on the metadata for anyone who wants it.
func TestReadNoColumns(t *testing.T) {
	tab := readTable(t, "chunks.parquet", &parquet.Options{Columns: []string{}})

	if got, want := tab.NumCols(), 0; got != want {
		t.Errorf("the table holds %d columns, want %d", got, want)
	}
	if got, want := tab.NumRows(), 0; got != want {
		t.Errorf("the table holds %d rows, want %d", got, want)
	}
}

// TestReadEmpty reads a file with a schema and no rows in it.
//
// The columns are there and are the types the file says, and each of them holds
// nothing. A row group of no rows contributes no chunk, so the columns come back
// with none rather than with an empty one nobody would want to step over.
func TestReadEmpty(t *testing.T) {
	tab := readTable(t, "empty.parquet", nil)

	if got, want := tab.NumCols(), 2; got != want {
		t.Fatalf("the table holds %d columns, want %d", got, want)
	}
	if got, want := tab.NumRows(), 0; got != want {
		t.Errorf("the table holds %d rows, want %d", got, want)
	}

	want := []dtype.DataType{dtype.Int64, dtype.String}
	for i, c := range tab.Columns {
		if got := c.DType(); !dtype.Equal(got, want[i]) {
			t.Errorf("column %d is a %s column, want %s", i, got, want[i])
		}
		if got := c.NumChunks(); got != 0 {
			t.Errorf("column %d came back in %d chunks, want none", i, got)
		}
	}
}

// TestReadReader reads through a reader rather than through a path, which is
// what a file that arrived over the network is read with.
func TestReadReader(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("testdata", "chunks.parquet"))
	if err != nil {
		t.Fatal(err)
	}

	tab, err := parquet.Read(bytes.NewReader(b), int64(len(b)), nil)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if got, want := tab.NumRows(), 6; got != want {
		t.Errorf("the table holds %d rows, want %d", got, want)
	}
}

// TestReadErrors covers the ways a read gives up.
func TestReadErrors(t *testing.T) {
	if _, err := parquet.ReadFile(filepath.Join(t.TempDir(), "gone.parquet"), nil); err == nil {
		t.Error("reading a file that is not there worked")
	}

	// A footer that is not a footer stops before any column is read.
	junk := []byte("this is not a parquet file at all")
	if _, err := parquet.Read(bytes.NewReader(junk), int64(len(junk)), nil); err == nil {
		t.Error("reading a file of junk worked")
	}

	// A name the file does not have, which is caught before anything is read.
	_, err := parquet.ReadFile(filepath.Join("testdata", "chunks.parquet"), &parquet.Options{
		Columns: []string{"code", "nope"},
	})
	if err == nil || !strings.Contains(err.Error(), `"nope"`) {
		t.Errorf("got %v, want an error naming nope", err)
	}

	// A column this package cannot assemble yet, which is caught while the
	// first row group is read.
	if _, err := parquet.ReadFile(filepath.Join("testdata", "nested.parquet"), nil); err == nil {
		t.Error("reading a file of lists worked")
	}
}

// TestReadTwice reads the same file twice through the same path and gets the
// same table, which is what says nothing is left over between reads.
func TestReadTwice(t *testing.T) {
	first := readTable(t, "pages.parquet", nil)
	second := readTable(t, "pages.parquet", nil)

	if !first.Schema.Equal(second.Schema) {
		t.Fatalf("the schemas differ: %v and %v", first.Schema, second.Schema)
	}
	for _, i := range []int{0, 2} {
		if !slices.Equal(numbers[int32](first.Columns[i]), numbers[int32](second.Columns[i])) {
			t.Errorf("column %d differs between the two reads", i)
		}
	}
	if !slices.Equal(text(first.Columns[1]), text(second.Columns[1])) {
		t.Error("the words differ between the two reads")
	}
}

func BenchmarkRead(b *testing.B) {
	buf, err := os.ReadFile(filepath.Join("testdata", "dictionary.parquet"))
	if err != nil {
		b.Fatal(err)
	}
	r := bytes.NewReader(buf)

	b.Run("decode", func(b *testing.B) {
		for b.Loop() {
			if _, err := parquet.Read(r, int64(len(buf)), nil); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("keep", func(b *testing.B) {
		opts := &parquet.Options{Dictionary: true}
		for b.Loop() {
			if _, err := parquet.Read(r, int64(len(buf)), opts); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("project", func(b *testing.B) {
		opts := &parquet.Options{Columns: []string{"size"}}
		for b.Loop() {
			if _, err := parquet.Read(r, int64(len(buf)), opts); err != nil {
				b.Fatal(err)
			}
		}
	})

	// The file writes no statistics, so nothing here is skipped and what this
	// measures is the filtering itself: a thousand rows compared and the half of
	// them that pass gathered out of the column.
	b.Run("filter", func(b *testing.B) {
		opts := &parquet.Options{
			Columns: []string{"size"},
			Filter:  []parquet.Predicate{parquet.Where("size", kernel.OpGt, int64(150))},
		}
		for b.Loop() {
			if _, err := parquet.Read(r, int64(len(buf)), opts); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// TestReadFilter reads the rows a filter keeps rather than the row groups that
// might hold them.
//
// The file holds twelve rows in three row groups with n running from nought to
// eleven, so a filter for eight and over leaves two of the groups unread and the
// third comes back whole. One chunk rather than three is what says the other two
// were never opened.
func TestReadFilter(t *testing.T) {
	tab := readTable(t, "stats.parquet", &parquet.Options{
		Columns: []string{"n", "word"},
		Filter:  []parquet.Predicate{parquet.Where("n", kernel.OpGe, int64(8))},
	})

	if got, want := numbers[int64](tab.Columns[0]), []int64{8, 9, 10, 11}; !slices.Equal(got, want) {
		t.Errorf("read %v, want %v", got, want)
	}
	if got, want := text(tab.Columns[1]), []string{"zulu", "yankee", "victor", "sierra"}; !slices.Equal(got, want) {
		t.Errorf("read %v, want %v", got, want)
	}
	if got := tab.Columns[0].NumChunks(); got != 1 {
		t.Errorf("the rows came back in %d chunks, want the one row group that was read", got)
	}
}

// TestReadFilterRows cuts rows out of the middle of the row groups it reads.
//
// Two predicates are an and, and these two want a range that starts inside the
// first group and ends inside the last, so every group is read and none of them
// comes back whole. That is the case a row group filter cannot do on its own and
// the reason this is not just a list of row groups.
func TestReadFilterRows(t *testing.T) {
	tab := readTable(t, "stats.parquet", &parquet.Options{
		Columns: []string{"n"},
		Filter: []parquet.Predicate{
			parquet.Where("n", kernel.OpGe, int64(3)),
			parquet.Where("n", kernel.OpLe, int64(9)),
		},
	})

	if got, want := numbers[int64](tab.Columns[0]), []int64{3, 4, 5, 6, 7, 8, 9}; !slices.Equal(got, want) {
		t.Errorf("read %v, want %v", got, want)
	}
	if got := tab.Columns[0].NumChunks(); got != 3 {
		t.Errorf("the rows came back in %d chunks, want one per row group", got)
	}
}

// TestReadFilterUnprojected filters on a column the caller did not ask for.
//
// Filtering on a column and reading it are different questions, and filtering a
// year of orders on a timestamp that is not wanted in the result is the ordinary
// case. The column is read to compare the rows against and is not in the table.
func TestReadFilterUnprojected(t *testing.T) {
	tab := readTable(t, "stats.parquet", &parquet.Options{
		Columns: []string{"word"},
		Filter: []parquet.Predicate{
			parquet.Where("n", kernel.OpGe, int64(3)),
			parquet.Where("n", kernel.OpLe, int64(9)),
		},
	})

	if got, want := tab.NumCols(), 1; got != want {
		t.Fatalf("the table holds %d columns, want %d", got, want)
	}
	if got, want := tab.Schema.Fields[0].Name, "word"; got != want {
		t.Errorf("the column is %s, want %s", got, want)
	}

	want := []string{"echo", "mike", "november", "oscar", "papa", "zulu", "yankee"}
	if got := text(tab.Columns[0]); !slices.Equal(got, want) {
		t.Errorf("read %v, want %v", got, want)
	}
}

// TestReadFilterNull checks that a row whose value is missing does not pass.
//
// Nothing compares to a value that is not there, so a missing value is not a
// value that failed the test, it is a row nobody can say belongs in the result.
// That is what SQL does and what kernel.Filter does with the mask.
//
// The file writes no statistics at all, so nothing is skipped and every row of
// it is compared, which is the path a filter takes on a file whose writer was in
// a hurry.
func TestReadFilterNull(t *testing.T) {
	opts := &parquet.Options{Columns: []string{"size"}}
	all := readTable(t, "dictionary.parquet", opts).Columns[0]

	want := 0
	for i := range all.Len() {
		if !all.IsNull(i) && all.Value[int64](i) > 150 {
			want++
		}
	}
	if want == 0 || want == all.Len() {
		t.Fatalf("%d rows of %d pass, which tests nothing", want, all.Len())
	}

	opts.Filter = []parquet.Predicate{parquet.Where("size", kernel.OpGt, int64(150))}
	got := readTable(t, "dictionary.parquet", opts).Columns[0]

	if got.Len() != want {
		t.Errorf("read %d rows, want %d", got.Len(), want)
	}
	if got.NullCount() != 0 {
		t.Errorf("%d of the rows are missing a value, want none of them", got.NullCount())
	}
}

// TestReadFilterNaN keeps a NaN out of a not equal.
//
// A NaN is unequal to everything, so it passes, and a writer leaves it out of
// the bounds it writes, so the row group holding it is read rather than skipped.
// Getting the second of those wrong would drop the row and look like the file
// was short.
func TestReadFilterNaN(t *testing.T) {
	tab := readTable(t, "stats.parquet", &parquet.Options{
		Columns: []string{"n", "ratio"},
		Filter:  []parquet.Predicate{parquet.Where("ratio", kernel.OpNe, 2.0)},
	})

	// The twelve rows hold 2.0 twice, at n of six and eleven, and a NaN at n of
	// five.
	if got, want := numbers[int64](tab.Columns[0]), []int64{0, 1, 2, 3, 4, 5, 7, 8, 9, 10}; !slices.Equal(got, want) {
		t.Errorf("read rows %v, want %v", got, want)
	}
	if got := numbers[float64](tab.Columns[1])[5]; !math.IsNaN(got) {
		t.Errorf("the sixth row read %v, want the NaN", got)
	}
}

// TestReadFilterNone reads a filter nothing passes.
//
// What comes back is a table of no rows rather than an error or a table of no
// columns, since the columns are what the file says they are whether or not any
// row survived and a caller appending this to another table wants to be able to.
func TestReadFilterNone(t *testing.T) {
	tab := readTable(t, "stats.parquet", &parquet.Options{
		Columns: []string{"n", "word"},
		Filter:  []parquet.Predicate{parquet.Where("n", kernel.OpGt, int64(99))},
	})

	if got, want := tab.NumRows(), 0; got != want {
		t.Errorf("the table holds %d rows, want %d", got, want)
	}
	want := []dtype.DataType{dtype.Int64, dtype.String}
	for i, c := range tab.Columns {
		if got := c.DType(); !dtype.Equal(got, want[i]) {
			t.Errorf("column %d is a %s column, want %s", i, got, want[i])
		}
		if got := c.NumChunks(); got != 0 {
			t.Errorf("column %d came back in %d chunks, want none", i, got)
		}
	}
}

// TestReadFilterDictionary filters a column the file wrote as indices into a
// dictionary, with the encoding kept and with it dropped.
//
// A comparison reads through the encoding and a gather keeps it, so the two
// answers are the same rows and the kept one is still encoded afterwards. That
// is the whole reason to keep it: a filter on a column of country codes is
// answered on the indices and the strings are never touched.
func TestReadFilterDictionary(t *testing.T) {
	opts := &parquet.Options{
		Columns: []string{"code", "n"},
		Filter:  []parquet.Predicate{parquet.WhereString("code", kernel.OpEq, "GB")},
	}
	plain := readTable(t, "chunks.parquet", opts)

	if got, want := text(plain.Columns[0]), []string{"GB", "GB"}; !slices.Equal(got, want) {
		t.Fatalf("read %v, want %v", got, want)
	}
	if got, want := numbers[int64](plain.Columns[1]), []int64{0, 5}; !slices.Equal(got, want) {
		t.Errorf("read rows %v, want %v", got, want)
	}

	opts.Dictionary = true
	kept := readTable(t, "chunks.parquet", opts)

	if got, want := text(kept.Columns[0]), text(plain.Columns[0]); !slices.Equal(got, want) {
		t.Errorf("keeping the encoding read %v, want %v", got, want)
	}
	if _, ok := kept.Columns[0].DType().(dtype.Dictionary); !ok {
		t.Errorf("the codes came back as %s, want the encoding kept", kept.Columns[0].DType())
	}
}

// TestReadFilterBloom reads a file whose writer wrote bloom filters, which is
// what answers for a value the bounds keep every group for.
//
// The identifiers go up in sevens, so 1004 is inside the range of the first
// group and is not in the file. The bounds keep the group, the filter throws it
// out, and no page of the file is read.
func TestReadFilterBloom(t *testing.T) {
	opts := &parquet.Options{
		Columns: []string{"name"},
		Filter:  []parquet.Predicate{parquet.Where("id", kernel.OpEq, int64(1004))},
	}
	if got := readTable(t, "bloom.parquet", opts).NumRows(); got != 0 {
		t.Errorf("a value that is not in the file read %d rows", got)
	}

	opts.Filter = []parquet.Predicate{parquet.Where("id", kernel.OpEq, int64(1007))}
	tab := readTable(t, "bloom.parquet", opts)

	if got, want := text(tab.Columns[0]), []string{"user-0007"}; !slices.Equal(got, want) {
		t.Errorf("read %v, want %v", got, want)
	}
}

// TestReadFilterBytesRead checks a filter reads less of the file.
//
// This is the point of the whole thing and it is worth counting rather than
// trusting. A read that skipped nothing would give the same rows at the cost of
// the whole file, which is the answer a caller cannot tell apart from the fast
// one by looking at the table.
func TestReadFilterBytesRead(t *testing.T) {
	buf, err := os.ReadFile(filepath.Join("testdata", "stats.parquet"))
	if err != nil {
		t.Fatal(err)
	}

	opts := &parquet.Options{Columns: []string{"n", "word"}}
	whole := &countingReader{src: bytes.NewReader(buf)}
	if _, err := parquet.Read(whole, int64(len(buf)), opts); err != nil {
		t.Fatalf("Read: %v", err)
	}

	opts.Filter = []parquet.Predicate{parquet.Where("n", kernel.OpGe, int64(8))}
	part := &countingReader{src: bytes.NewReader(buf)}
	if _, err := parquet.Read(part, int64(len(buf)), opts); err != nil {
		t.Fatalf("Read: %v", err)
	}

	if part.n >= whole.n {
		t.Errorf("filtering read %d bytes and reading it all read %d, want less", part.n, whole.n)
	}
}

// countingReader adds up what has been read through it.
type countingReader struct {
	src io.ReaderAt
	n   int64
}

// ReadAt reads from the file underneath and counts what it asked for.
func (c *countingReader) ReadAt(p []byte, off int64) (int, error) {
	c.n += int64(len(p))
	return c.src.ReadAt(p, off)
}

// TestReadFilterErrors covers the ways a filter is refused.
//
// All of them are refused before any of the file is read, since a filter that
// was quietly dropped would read the whole file and look like it worked.
func TestReadFilterErrors(t *testing.T) {
	for _, c := range []struct {
		name string
		pred parquet.Predicate
		want string
	}{
		{
			"a column the file does not have",
			parquet.Where("nope", kernel.OpEq, int64(1)),
			`"nope"`,
		},
		{
			"a value of a type the column cannot be compared with",
			parquet.WhereString("n", kernel.OpEq, "eight"),
			"cannot compare n",
		},
		{
			"a predicate with no value at all",
			parquet.Predicate{Column: "n", Op: kernel.OpEq},
			"compares against no value",
		},
		{
			"a value that is not there, which no row passes",
			parquet.Predicate{Column: "n", Op: kernel.OpEq, Value: array.NewNull(1)},
			"is not there",
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := parquet.ReadFile(filepath.Join("testdata", "stats.parquet"), &parquet.Options{
				Filter: []parquet.Predicate{c.pred},
			})
			if err == nil {
				t.Fatal("that filter worked")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("got %q, want it to mention %s", err, c.want)
			}
		})
	}
}
