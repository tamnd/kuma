package kernel_test

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// int64List is the list type these use, a list of numbers being the shape a
// repeated column in a file usually has.
var int64List = dtype.List{Elem: dtype.Int64}

// listChunk builds one chunk of a list column out of the rows given, where a nil
// row is a null and an empty one is a row that is present and holds nothing.
func listChunk(t *testing.T, rows ...[]int64) *array.Array {
	t.Helper()

	b, err := array.NewListBuilder(int64List)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}
	for _, r := range rows {
		if r == nil {
			b.AppendNull()
			continue
		}
		b.Elem().AppendValues(r)
		b.Append()
	}
	return b.Finish()
}

// listCol builds a list column of one chunk.
func listCol(t *testing.T, rows ...[]int64) *array.Chunked {
	t.Helper()

	c, err := array.NewChunked(int64List, listChunk(t, rows...))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	return c
}

// listRows reads a list column back as the rows it holds, with a null row coming
// back nil so that a test can tell it from an empty one.
func listRows(t *testing.T, c *array.Chunked) [][]int64 {
	t.Helper()

	var out [][]int64
	for _, a := range c.Chunks() {
		for i := range a.Len() {
			if a.IsNull(i) {
				out = append(out, nil)
				continue
			}
			out = append(out, slices.Clone(a.List(i).Values[int64]()))
		}
	}
	return out
}

// wantRows fails unless c holds exactly the rows given.
func wantRows(t *testing.T, c *array.Chunked, want [][]int64) {
	t.Helper()

	got := listRows(t, c)
	if !slices.EqualFunc(got, want, slices.Equal) {
		t.Errorf("the rows are %v, want %v", got, want)
	}
}

func TestTakeList(t *testing.T) {
	c := listCol(t, []int64{1, 2}, []int64{3}, []int64{}, []int64{4, 5, 6})

	got := kernel.Take(c, []int{3, 0, 2})
	if !dtype.Equal(got.DType(), int64List) {
		t.Errorf("DType = %s, want %s", got.DType(), int64List)
	}
	wantRows(t, got, [][]int64{{4, 5, 6}, {1, 2}, {}})
}

// TestTakeListRepeats is the shape an explode leans on, where one row is asked
// for several times and the elements are gathered rather than the rows copied.
func TestTakeListRepeats(t *testing.T) {
	c := listCol(t, []int64{1, 2}, []int64{3})

	wantRows(t, kernel.Take(c, []int{0, 0, 1, 0}),
		[][]int64{{1, 2}, {1, 2}, {3}, {1, 2}})
}

// TestTakeListNulls covers the two ways a gathered row can be missing, which are
// a null in the column and a position below zero from the caller.
func TestTakeListNulls(t *testing.T) {
	c := listCol(t, []int64{1}, nil, []int64{})

	got := kernel.Take(c, []int{1, -1, 0, 2})
	wantRows(t, got, [][]int64{nil, nil, {1}, {}})

	chunk := got.Chunks()[0]
	if chunk.NullCount() != 2 {
		t.Errorf("NullCount = %d, want 2", chunk.NullCount())
	}
	if chunk.IsNull(3) {
		t.Error("the empty row came back null")
	}
}

// TestTakeListKeepsNoBitmapWhenNothingIsMissing is the case worth not paying
// for, since a column read out of a file with nothing missing is the common one.
func TestTakeListKeepsNoBitmapWhenNothingIsMissing(t *testing.T) {
	c := listCol(t, []int64{1}, []int64{2})

	chunk := kernel.Take(c, []int{1, 0}).Chunks()[0]
	if chunk.NullCount() != 0 {
		t.Errorf("NullCount = %d, want 0", chunk.NullCount())
	}
	if chunk.Validity() != nil {
		t.Error("the gather built a bitmap for a column with no nulls")
	}
}

// TestTakeListAcrossChunks is the case the children have to be laid end to end
// for, since a row in the second chunk points into that chunk's own child.
func TestTakeListAcrossChunks(t *testing.T) {
	c, err := array.NewChunked(int64List,
		listChunk(t, []int64{1, 2}, []int64{3}),
		listChunk(t, []int64{4, 5}, nil, []int64{6}),
	)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	wantRows(t, kernel.Take(c, []int{4, 0, 3, 2, 1}),
		[][]int64{{6}, {1, 2}, nil, {4, 5}, {3}})
}

// TestTakeListOfSlicedChunks is the case the offsets being absolute has to
// survive, where a chunk reaches part of a child it does not start at.
func TestTakeListOfSlicedChunks(t *testing.T) {
	c, err := array.NewChunked(int64List,
		listChunk(t, []int64{1, 2}, []int64{3}, []int64{4}).Slice(1, 3),
	)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	wantRows(t, kernel.Take(c, []int{1, 0}), [][]int64{{4}, {3}})
}

func TestTakeListEmpty(t *testing.T) {
	c := listCol(t, []int64{1, 2})

	got := kernel.Take(c, nil)
	if got.Len() != 0 {
		t.Errorf("Len = %d, want 0", got.Len())
	}
	if !dtype.Equal(got.DType(), int64List) {
		t.Errorf("DType = %s, want %s", got.DType(), int64List)
	}
}

// TestTakeListOfLists is the recursion, which works because the elements go
// through Take rather than through the switch inside it. The column here is
// built with NewListFrom because ListBuilder's element builder is an ordinary
// Builder and an ordinary Builder does not build lists.
func TestTakeListOfLists(t *testing.T) {
	outer := dtype.List{Elem: int64List}

	nested, err := array.NewListFrom(outer, []int32{0, 2, 3},
		listChunk(t, []int64{1, 2}, []int64{3}, []int64{4, 5}), nil)
	if err != nil {
		t.Fatalf("NewListFrom: %v", err)
	}
	c, err := array.NewChunked(outer, nested)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	got := kernel.Take(c, []int{1, 0})
	if got.Len() != 2 {
		t.Fatalf("Len = %d, want 2", got.Len())
	}

	chunk := got.Chunks()[0]
	if row := chunk.List(0); row.Len() != 1 ||
		!slices.Equal(row.List(0).Values[int64](), []int64{4, 5}) {
		t.Errorf("row 0 is %v, want one list of [4 5]", row)
	}
	if row := chunk.List(1); row.Len() != 2 ||
		!slices.Equal(row.List(1).Values[int64](), []int64{3}) {
		t.Errorf("row 1 is %v, want two lists ending in [3]", row)
	}
}

// TestTakeListOfStrings is the element type whose values do not live in the
// value buffer, so the gather one level down is the bytes case rather than the
// fixed width one.
func TestTakeListOfStrings(t *testing.T) {
	dt := dtype.List{Elem: dtype.String}

	b, err := array.NewListBuilder(dt)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}
	b.Elem().AppendString("a")
	b.Elem().AppendString("bb")
	b.Append()
	b.AppendNull()
	b.Elem().AppendString("ccc")
	b.Append()

	c, err := array.NewChunked(dt, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	got := kernel.Take(c, []int{2, 1, 0})
	chunk := got.Chunks()[0]
	if chunk.NullCount() != 1 {
		t.Errorf("NullCount = %d, want 1", chunk.NullCount())
	}
	if s := string(chunk.List(0).Bytes(0)); s != "ccc" {
		t.Errorf("row 0 is %q, want ccc", s)
	}
	if row := chunk.List(2); row.Len() != 2 ||
		string(row.Bytes(0)) != "a" || string(row.Bytes(1)) != "bb" {
		t.Errorf("row 2 is %v, want [a bb]", row)
	}
}

// TestFilterList is the gather reached the other way, since a filter is a take
// at the positions the mask selects.
func TestFilterList(t *testing.T) {
	c := listCol(t, []int64{1}, []int64{2, 3}, nil, []int64{4})

	mask, err := array.NewChunked(dtype.Bool, array.OfBools(true, false, true, true))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	wantRows(t, kernel.Filter(c, mask), [][]int64{{1}, nil, {4}})
}

func TestTakeListOutOfRange(t *testing.T) {
	c := listCol(t, []int64{1})

	defer func() {
		if recover() == nil {
			t.Error("a position past the end did not panic")
		}
	}()
	kernel.Take(c, []int{1})
}

// listWidth is how many elements a row of the benchmark column holds. It is
// eight because a repeated column in a real file is usually short rows rather
// than a few long ones, and because it keeps the element count at benchLen,
// which is what the gather benchmarks over flat columns move.
const listWidth = 8

// benchLists returns a list column of benchLen elements, in the given number of
// chunks, with no nulls.
func benchLists(b *testing.B, chunks int) *array.Chunked {
	b.Helper()

	bd, err := array.NewListBuilder(int64List)
	if err != nil {
		b.Fatalf("NewListBuilder: %v", err)
	}

	rows := benchLen / listWidth / chunks
	arrays := make([]*array.Array, chunks)
	for c := range chunks {
		for i := range rows {
			for e := range listWidth {
				bd.Elem().Append(int64((c*rows+i)*listWidth + e))
			}
			bd.Append()
		}
		arrays[c] = bd.Finish()
	}

	out, err := array.NewChunked(int64List, arrays...)
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	return out
}

// listOrder returns a position for every row of the benchmark column.
func listOrder(shuffled bool) []int {
	idx := make([]int, benchLen/listWidth)
	for i := range idx {
		idx[i] = i
	}
	if shuffled {
		r := rand.New(rand.NewPCG(1, 2))
		r.Shuffle(len(idx), func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })
	}
	return idx
}

// The bytes are the elements rather than the rows, so that these sit next to
// BenchmarkTake and say what a list column costs against the flat column holding
// the same numbers.
func BenchmarkTakeList(b *testing.B) {
	src := benchLists(b, 1)
	idx := listOrder(false)

	b.SetBytes(benchLen * 8)
	for b.Loop() {
		chunkedSink = kernel.Take(src, idx)
	}
}

func BenchmarkTakeListShuffled(b *testing.B) {
	src := benchLists(b, 1)
	idx := listOrder(true)

	b.SetBytes(benchLen * 8)
	for b.Loop() {
		chunkedSink = kernel.Take(src, idx)
	}
}

func BenchmarkTakeListChunked(b *testing.B) {
	src := benchLists(b, 16)
	idx := listOrder(true)

	b.SetBytes(benchLen * 8)
	for b.Loop() {
		chunkedSink = kernel.Take(src, idx)
	}
}
