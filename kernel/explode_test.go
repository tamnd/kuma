package kernel_test

import (
	"slices"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// exploded reads a flattened column back as the values it holds, with a missing
// value coming back as the zero and a second slice saying which of them were
// missing, since a column of numbers has no value that means nothing.
func exploded(t *testing.T, c *array.Chunked) (vals []int64, null []bool) {
	t.Helper()

	for _, a := range c.Chunks() {
		for i := range a.Len() {
			if a.IsNull(i) {
				vals = append(vals, 0)
				null = append(null, true)
				continue
			}
			vals = append(vals, a.Values[int64]()[i])
			null = append(null, false)
		}
	}
	return vals, null
}

// wantExplode checks the whole answer: the values, which of them are missing,
// and the row each came from.
func wantExplode(t *testing.T, c *array.Chunked, wantVals []int64, wantNull []bool, wantRows []int) {
	t.Helper()

	got, rows := kernel.Explode(c)
	if !dtype.Equal(got.DType(), dtype.Int64) {
		t.Errorf("DType = %s, want int64", got.DType())
	}
	if got.Len() != len(rows) {
		t.Errorf("%d values against %d rows, which have to be the same", got.Len(), len(rows))
	}

	vals, null := exploded(t, got)
	if !slices.Equal(vals, wantVals) {
		t.Errorf("values are %v, want %v", vals, wantVals)
	}
	if !slices.Equal(null, wantNull) {
		t.Errorf("the missing ones are %v, want %v", null, wantNull)
	}
	if !slices.Equal(rows, wantRows) {
		t.Errorf("rows are %v, want %v", rows, wantRows)
	}
}

func TestExplode(t *testing.T) {
	c := listCol(t, []int64{1, 2, 3}, []int64{4}, []int64{5, 6})

	wantExplode(t, c,
		[]int64{1, 2, 3, 4, 5, 6},
		[]bool{false, false, false, false, false, false},
		[]int{0, 0, 0, 1, 2, 2})
}

// TestExplodeEmptyAndNull is the rule that keeps the row count from depending on
// which column was taken apart: a row with nothing in it is still a row.
func TestExplodeEmptyAndNull(t *testing.T) {
	c := listCol(t, []int64{1, 2}, []int64{}, nil, []int64{3})

	wantExplode(t, c,
		[]int64{1, 2, 0, 0, 3},
		[]bool{false, false, true, true, false},
		[]int{0, 0, 1, 2, 3})
}

func TestExplodeAllEmpty(t *testing.T) {
	c := listCol(t, []int64{}, nil, []int64{})

	wantExplode(t, c,
		[]int64{0, 0, 0},
		[]bool{true, true, true},
		[]int{0, 1, 2})
}

func TestExplodeNoRows(t *testing.T) {
	c := listCol(t)

	got, rows := kernel.Explode(c)
	if got.Len() != 0 || len(rows) != 0 {
		t.Errorf("exploding no rows gave %d values and %d rows, want none of either", got.Len(), len(rows))
	}
	if !dtype.Equal(got.DType(), dtype.Int64) {
		t.Errorf("DType = %s, want int64", got.DType())
	}
}

// TestExplodeAcrossChunks is the case the element positions are laid out for,
// where the same element number means a different element depending on which
// chunk the row was in.
func TestExplodeAcrossChunks(t *testing.T) {
	c, err := array.NewChunked(int64List,
		listChunk(t, []int64{1, 2}, nil),
		listChunk(t, []int64{3}),
		listChunk(t, []int64{4, 5}, []int64{}))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	wantExplode(t, c,
		[]int64{1, 2, 0, 3, 4, 5, 0},
		[]bool{false, false, true, false, false, false, true},
		[]int{0, 0, 1, 2, 3, 3, 4})
}

// TestExplodeSlicedChunk reaches the offsets of a chunk that does not begin at
// the start of its own child, which is what a limit over a list column leaves
// behind.
func TestExplodeSlicedChunk(t *testing.T) {
	whole := listChunk(t, []int64{1}, []int64{2, 3}, []int64{4})

	c, err := array.NewChunked(int64List, whole.Slice(1, 3))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	wantExplode(t, c,
		[]int64{2, 3, 4},
		[]bool{false, false, false},
		[]int{0, 0, 1})
}

// TestExplodeNullsInside is a row whose elements are missing rather than a row
// that is missing. The nulls come out as values of the flattened column and the
// row is as long as it says it is.
func TestExplodeNullsInside(t *testing.T) {
	b, err := array.NewListBuilder(int64List)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}
	b.Elem().Append(int64(1))
	b.Elem().AppendNull()
	b.Append()
	b.Elem().Append(int64(2))
	b.Append()

	c, err := array.NewChunked(int64List, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	wantExplode(t, c,
		[]int64{1, 0, 2},
		[]bool{false, true, false},
		[]int{0, 0, 1})
}

// TestExplodeOfLists is one level of a nested column coming apart, which works
// because the elements go through Take rather than through a builder.
func TestExplodeOfLists(t *testing.T) {
	outer := dtype.List{Elem: int64List}

	inner := listChunk(t, []int64{1, 2}, []int64{3}, []int64{4})
	nested, err := array.NewListFrom(outer, []int32{0, 2, 3}, inner, nil)
	if err != nil {
		t.Fatalf("NewListFrom: %v", err)
	}
	c, err := array.NewChunked(outer, nested)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	got, rows := kernel.Explode(c)
	if !dtype.Equal(got.DType(), int64List) {
		t.Errorf("DType = %s, want %s", got.DType(), int64List)
	}
	if !slices.Equal(rows, []int{0, 0, 1}) {
		t.Errorf("rows are %v, want [0 0 1]", rows)
	}
	wantRows(t, got, [][]int64{{1, 2}, {3}, {4}})
}

func TestExplodeOfStrings(t *testing.T) {
	strList := dtype.List{Elem: dtype.String}

	b, err := array.NewListBuilder(strList)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}
	b.Elem().AppendString("a")
	b.Elem().AppendString("bb")
	b.Append()
	b.Elem().AppendString("ccc")
	b.Append()

	c, err := array.NewChunked(strList, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	got, rows := kernel.Explode(c)
	if !slices.Equal(rows, []int{0, 0, 1}) {
		t.Errorf("rows are %v, want [0 0 1]", rows)
	}

	var vals []string
	for _, a := range got.Chunks() {
		for i := range a.Len() {
			vals = append(vals, string(a.Bytes(i)))
		}
	}
	if !slices.Equal(vals, []string{"a", "bb", "ccc"}) {
		t.Errorf("values are %v, want [a bb ccc]", vals)
	}
}

// TestExplodeRowsGatherTheRest is what the row numbers are for: the columns
// beside the one coming apart are gathered by them, so a row that held three
// elements has its other values repeated three times.
func TestExplodeRowsGatherTheRest(t *testing.T) {
	c := listCol(t, []int64{1, 2, 3}, nil, []int64{4})

	other, err := array.NewChunked(dtype.Int64, array.Of[int64](10, 20, 30))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	_, rows := kernel.Explode(c)
	got := kernel.Take(other, rows)

	var vals []int64
	for _, a := range got.Chunks() {
		vals = append(vals, a.Values[int64]()...)
	}
	if !slices.Equal(vals, []int64{10, 10, 10, 20, 30}) {
		t.Errorf("the other column is %v, want [10 10 10 20 30]", vals)
	}
}

func TestExplodeMistakes(t *testing.T) {
	flat, err := array.NewChunked(dtype.Int64, array.Of[int64](1, 2))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	for _, tt := range []struct {
		name string
		col  *array.Chunked
	}{
		{"a nil column", nil},
		{"a column that is not a list", flat},
	} {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("exploding %s did not panic", tt.name)
				}
			}()
			kernel.Explode(tt.col)
		})
	}
}

func BenchmarkExplode(b *testing.B) {
	src := benchLists(b, 1)

	b.SetBytes(benchLen * 8)
	for b.Loop() {
		chunkedSink, indexSink = kernel.Explode(src)
	}
}

func BenchmarkExplodeChunked(b *testing.B) {
	src := benchLists(b, 16)

	b.SetBytes(benchLen * 8)
	for b.Loop() {
		chunkedSink, indexSink = kernel.Explode(src)
	}
}
