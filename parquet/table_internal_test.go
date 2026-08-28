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
// there any more, which is the same error whether it turns up at the size or at
// the footer. What it is not is a panic or an empty table.
func TestReadOpenClosed(t *testing.T) {
	f, err := os.Open(filepath.Join("testdata", "chunks.parquet"))
	if err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err := readOpen(f, nil); !errors.Is(err, os.ErrClosed) {
		t.Errorf("got %v, want a closed file", err)
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
