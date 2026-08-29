package parquet

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// stringChunk builds a plain chunk of strings.
func stringChunk(t *testing.T, vs ...string) *array.Array {
	t.Helper()

	b, err := array.NewBuilder(dtype.String)
	if err != nil {
		t.Fatal(err)
	}
	for _, v := range vs {
		b.AppendString(v)
	}
	return b.Finish()
}

// dictChunk builds a chunk of the same strings held as indices into a
// dictionary, which is the other shape a row group can hand a column back in.
func dictChunk[T array.Numeric](t *testing.T, index dtype.DataType, vs ...string) *array.Array {
	t.Helper()

	values := stringChunk(t, vs...)
	b, err := array.NewBuilder(index)
	if err != nil {
		t.Fatal(err)
	}
	for i := range vs {
		b.Append(T(i))
	}
	a, err := array.NewDictionary(b.Finish(), values)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

// values reads a whole column as strings, whatever its chunks are.
func values(t *testing.T, c *array.Chunked) []string {
	t.Helper()

	out := make([]string, 0, c.Len())
	for _, a := range c.Chunks() {
		d := a.Dictionary()
		for i := range a.Len() {
			if d != nil {
				out = append(out, string(d.Bytes(a.Index(i))))
				continue
			}
			out = append(out, string(a.Bytes(i)))
		}
	}
	return out
}

// TestColumnMixed puts a chunk that came back encoded and a chunk that came
// back plain into one column.
//
// A writer decides per row group whether to write indices into a dictionary,
// and is allowed to stop once the values stop repeating, so a file can hold both
// shapes of the same column. None of the files in testdata does, because
// pyarrow writes a dictionary page for every row group of a column it encodes at
// all, so this is built by hand rather than read.
//
// A column is one type, so the answer is the type the file's schema names and
// the encoded chunk is decoded into it. Which is the same answer as asking for
// the decoded column in the first place, and the point is that keeping the
// encoding does not turn a file like this into an error.
func TestColumnMixed(t *testing.T) {
	f := dtype.Field{Name: "code", Type: dtype.String}
	chunks := []*array.Array{
		dictChunk[int32](t, dtype.Int32, "GB", "JP"),
		stringChunk(t, "US", "FR"),
	}

	for _, keep := range []bool{false, true} {
		c := column(f, chunks, keep)
		if got := c.DType(); !dtype.Equal(got, dtype.String) {
			t.Errorf("keep %v gave a %s column, want string", keep, got)
		}
		if got, want := values(t, c), []string{"GB", "JP", "US", "FR"}; !slices.Equal(got, want) {
			t.Errorf("keep %v gave %q, want %q", keep, got, want)
		}
	}
}

// TestColumnMixedIndex is the same when the chunks are both encoded and
// disagree about how wide the index is.
//
// The format has one index type and this package reads every dictionary into
// int32, so nothing here produces this today. What decides the type of a column
// is that the chunks agree rather than that one of them is encoded, and this is
// that decision being made on something other than the encoding.
func TestColumnMixedIndex(t *testing.T) {
	f := dtype.Field{Name: "code", Type: dtype.String}
	chunks := []*array.Array{
		dictChunk[int32](t, dtype.Int32, "GB", "JP"),
		dictChunk[int8](t, dtype.Int8, "US", "FR"),
	}

	c := column(f, chunks, true)
	if got := c.DType(); !dtype.Equal(got, dtype.String) {
		t.Errorf("got a %s column, want string", got)
	}
	if got, want := values(t, c), []string{"GB", "JP", "US", "FR"}; !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

// TestColumnKept is the case that is worth keeping, which is every chunk saying
// the same thing.
func TestColumnKept(t *testing.T) {
	f := dtype.Field{Name: "code", Type: dtype.String}
	chunks := []*array.Array{
		dictChunk[int32](t, dtype.Int32, "GB", "JP"),
		dictChunk[int32](t, dtype.Int32, "US", "FR"),
	}

	c := column(f, chunks, true)
	want := dtype.Dictionary{Index: dtype.Int32, Value: dtype.String}
	if got := c.DType(); !dtype.Equal(got, want) {
		t.Errorf("got a %s column, want %s", got, want)
	}
	if c.Chunk(0) != chunks[0] {
		t.Error("the chunk was rebuilt rather than kept")
	}
	if got, want := values(t, c), []string{"GB", "JP", "US", "FR"}; !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

// TestColumnNoChunks is a column of a file with no rows in it, which has
// nothing to say about its own type and takes the one the schema names.
func TestColumnNoChunks(t *testing.T) {
	f := dtype.Field{Name: "code", Type: dtype.String}

	for _, keep := range []bool{false, true} {
		c := column(f, nil, keep)
		if got := c.DType(); !dtype.Equal(got, dtype.String) {
			t.Errorf("keep %v gave a %s column, want string", keep, got)
		}
		if c.Len() != 0 {
			t.Errorf("keep %v gave %d rows, want none", keep, c.Len())
		}
	}
}

// TestColumnImpossible checks that the two things this file says cannot happen
// say so if they ever do.
//
// A chunk of a column is either the column's own type or a dictionary of it,
// which is what a page is read into and not something a file gets a say in, so
// neither of these is a bad file and neither is worth an error return that every
// caller would have to carry. What they are worth is a stack rather than a wrong
// column, which is what an assembler that grew a third shape would get.
func TestColumnImpossible(t *testing.T) {
	t.Run("a chunk that is not the type it is put under", func(t *testing.T) {
		defer wantPanic(t)
		chunked(dtype.Int64, stringChunk(t, "GB"))
	})
	t.Run("a cast that is not a decode", func(t *testing.T) {
		defer wantPanic(t)
		decode(stringChunk(t, "not a number"), dtype.Int64)
	})
}

// TestReadOpenClosed reads a file that was closed underneath the reader.
//
// Opening it worked and everything after that is a read of a file that is not
// there any more, which turns up at the size on the way to turning up at the
// footer. What the operating system calls it is its own business and the
// platforms do not agree, so what is checked is that it is an error at all
// rather than a panic or an empty table.
func TestReadOpenClosed(t *testing.T) {
	f, err := os.Open(filepath.Join("testdata", "chunks.parquet"))
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err := readOpen(f, nil); err == nil {
		t.Error("reading a closed file worked")
	}
}

// wantPanic fails the test if what it is deferred in did not panic.
func wantPanic(t *testing.T) {
	t.Helper()

	e := recover()
	if e == nil {
		t.Fatal("that worked")
	}
	if s, ok := e.(string); !ok || !strings.HasPrefix(s, "parquet: ") {
		t.Errorf("panicked with %v, want a parquet message", e)
	}
}

// TestReadAlso checks the projection a filter needs on top of the caller's.
//
// A predicate on a column the caller did not ask for adds it to the end, where
// it is read and then left out of the table. A predicate on a column that is
// already being read is compared against that one, and two predicates on the
// same column share it, so a filter of any size adds at most one read per column
// it names.
func TestReadAlso(t *testing.T) {
	r := openStats(t)
	if err := r.Project("word"); err != nil {
		t.Fatal(err)
	}

	// The file holds n, word, size, ratio, absent and flag in that order, so
	// two predicates on n and one on word is two that have to be added and one
	// that is already there.
	tests := []test{{column: 0}, {column: 0}, {column: 1}}
	r.readAlso(tests)

	if want := []int{1, 0}; !slices.Equal(r.take, want) {
		t.Errorf("the projection reads columns %v, want %v", r.take, want)
	}
	for i, want := range []int{1, 1, 0} {
		if got := tests[i].slot; got != want {
			t.Errorf("predicate %d compares slot %d, want %d", i, got, want)
		}
	}
	if got, want := len(r.schema.Fields), 2; got != want {
		t.Errorf("a batch holds %d columns, want %d", got, want)
	}
}

// TestReadAlsoNothing checks that a read with no filter is projected the way the
// caller left it, which is what says an unfiltered read costs what it did.
func TestReadAlsoNothing(t *testing.T) {
	r := openStats(t)
	if err := r.Project("word", "n"); err != nil {
		t.Fatal(err)
	}

	before := slices.Clone(r.take)
	r.readAlso(nil)

	if !slices.Equal(r.take, before) {
		t.Errorf("the projection became %v, want %v", r.take, before)
	}
}

// TestFilterFileError checks that a footer that contradicts itself stops a
// filtered read before any of the file is opened.
//
// The row groups are chosen from the statistics, so a chunk claiming more
// missing values than it holds rows is found there rather than on the way
// through a page. What matters is that it is an error and not a table missing
// the rows of that group.
func TestFilterFileError(t *testing.T) {
	r := openStats(t)
	r.meta.RowGroups[0].Columns[0].Meta.Stats.NullCount = 99

	opts := &Options{Filter: []Predicate{Where("n", kernel.OpEq, int64(1))}}
	if _, err := r.table(opts); !errors.Is(err, ErrFormat) {
		t.Errorf("got %v, want it to be an %v", err, ErrFormat)
	}
}

// TestPickUnchecked checks that a comparison that cannot be made is an error out
// of the read rather than a panic partway through it.
//
// Nothing arrives here in this state. A filter is looked over before any of the
// file is read, exactly so that a query written against the wrong type is
// refused before it costs anything, and this is the same predicate reaching the
// rows with that step skipped. It is here because the alternative to returning
// the error is not returning it.
func TestPickUnchecked(t *testing.T) {
	r := openStats(t)
	bad := []test{{pred: WhereString("n", kernel.OpEq, "eight"), column: 0, slot: 0}}

	_, err := r.chunksOf([]int{0}, bad, 1)
	if err == nil {
		t.Fatal("comparing a column of numbers against a word worked")
	}
	if !strings.Contains(err.Error(), "filtering on n") {
		t.Errorf("got %q, want it to name the column", err)
	}
}

// TestAndImpossible checks the mask joiner refuses what is not a mask.
//
// Both sides come out of kernel.Compare, which returns conditions and nothing
// else, so this cannot happen from a read. What it is worth is a stack rather
// than a wrong answer if the two ever stop agreeing.
func TestAndImpossible(t *testing.T) {
	defer wantPanic(t)
	c := chunked(dtype.String, stringChunk(t, "GB"))
	and(c, c)
}
