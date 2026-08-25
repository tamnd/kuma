package kuma_test

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// half returns a two row frame of the same shape trades has, so that two of
// them stack into something that can be checked against it.
func half(t *testing.T, symbols []string, prices []float64, qty []int64) *kuma.Frame[kuma.Dynamic] {
	t.Helper()

	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", symbols...).Column(),
		kuma.NewSeries("price", prices...).Column(),
		kuma.NewSeries("qty", qty...).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	return f
}

func TestConcat(t *testing.T) {
	top := half(t, []string{"AAPL", "MSFT"}, []float64{189.5, 411.2}, []int64{100, 50})
	bottom := half(t, []string{"AAPL", "NVDA"}, []float64{190.1, 121.0}, []int64{25, 400})

	got, err := kuma.Concat(top, bottom)
	if err != nil {
		t.Fatalf("Concat: %v", err)
	}
	if got.NumRows() != 4 || got.NumCols() != 3 {
		t.Fatalf("the result is %d by %d, want 4 by 3", got.NumRows(), got.NumCols())
	}

	want := rows(t, trades(t))
	if lines := rows(t, got); !equalLines(lines, want) {
		t.Errorf("the rows are\n%s\nwant\n%s",
			strings.Join(lines, "\n"), strings.Join(want, "\n"))
	}

	// Nothing was copied, so the result holds the chunks the two frames hold.
	if n := got.ColumnAt(0).Data().NumChunks(); n != 2 {
		t.Errorf("the stacked column has %d chunks, want the two that went in", n)
	}
}

// TestConcatColumnOrder is the rule that a frame written with its columns in
// another order is still the same table.
func TestConcatColumnOrder(t *testing.T) {
	top := half(t, []string{"AAPL"}, []float64{189.5}, []int64{100})
	bottom, err := half(t, []string{"MSFT"}, []float64{411.2}, []int64{50}).
		Select("qty", "price", "symbol")
	if err != nil {
		t.Fatalf("Select: %v", err)
	}

	got, err := kuma.Concat(top, bottom)
	if err != nil {
		t.Fatalf("Concat: %v", err)
	}
	if names := strings.Join(got.Names(), ","); names != "symbol,price,qty" {
		t.Errorf("the columns are %q, want the first frame's order", names)
	}
	want := []string{"AAPL 189.5 100", "MSFT 411.2 50"}
	if lines := rows(t, got); !equalLines(lines, want) {
		t.Errorf("the rows are\n%s", strings.Join(lines, "\n"))
	}
}

func TestConcatOne(t *testing.T) {
	f := trades(t)
	got, err := kuma.Concat(f)
	if err != nil {
		t.Fatalf("Concat: %v", err)
	}
	if got != f {
		t.Error("concatenating one frame built a new one, which there is no reason to do")
	}
}

func TestConcatEmptyFrames(t *testing.T) {
	f := trades(t)
	empty := f.Head(0)

	got, err := kuma.Concat(empty, f, empty)
	if err != nil {
		t.Fatalf("Concat: %v", err)
	}
	if got.NumRows() != f.NumRows() {
		t.Errorf("there are %d rows, want the %d that went in", got.NumRows(), f.NumRows())
	}
}

func TestConcatErrors(t *testing.T) {
	f := trades(t)

	if _, err := kuma.Concat[kuma.Dynamic](); err == nil {
		t.Error("concatenating nothing succeeded")
	}
	if _, err := kuma.Concat(f, nil); err == nil {
		t.Error("concatenating a nil frame succeeded")
	}

	// A column one frame has and the other does not.
	fewer, err := f.Drop("qty")
	if err != nil {
		t.Fatalf("Drop: %v", err)
	}
	if _, short := kuma.Concat(f, fewer); short == nil {
		t.Error("concatenating frames with different columns succeeded")
	}
	if _, long := kuma.Concat(fewer, f); long == nil {
		t.Error("concatenating frames with different columns succeeded the other way round")
	}

	// The same number of columns and not the same columns, which is the case
	// counting them does not catch.
	renamed, err := f.Rename("qty", "quantity")
	if err != nil {
		t.Fatalf("Rename: %v", err)
	}
	if _, wrongName := kuma.Concat(f, renamed); wrongName == nil {
		t.Error("concatenating frames with a renamed column succeeded")
	}

	// The same column name holding a different type.
	recast, err := f.Cast("qty", dtype.Int32)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	if _, wrongType := kuma.Concat(f, recast); wrongType == nil {
		t.Error("concatenating an int64 column onto an int32 one succeeded")
	}
}

func TestConcatUnion(t *testing.T) {
	left, err := kuma.NewFrame(
		kuma.NewSeries("a", int64(1), 2).Column(),
		kuma.NewSeries("b", "x", "y").Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	right, err := kuma.NewFrame(
		kuma.NewSeries("b", "z").Column(),
		kuma.NewSeries("c", 0.5).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	got, err := kuma.ConcatUnion(left, right)
	if err != nil {
		t.Fatalf("ConcatUnion: %v", err)
	}
	if names := strings.Join(got.Names(), ","); names != "a,b,c" {
		t.Errorf("the columns are %q, want them in the order they first appear", names)
	}
	want := []string{"1 x .", "2 y .", ". z 0.5"}
	if lines := rows(t, got); !equalLines(lines, want) {
		t.Errorf("the rows are\n%s\nwant\n%s",
			strings.Join(lines, "\n"), strings.Join(want, "\n"))
	}
}

// TestConcatUnionNullColumn is the column no frame has any values for, where
// the filled column is of the type that has no values.
func TestConcatUnionNullColumn(t *testing.T) {
	data, err := array.NewChunked(dtype.Null, array.NewNull(2))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	empty, err := kuma.NewColumn("n", data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	left, err := kuma.NewFrame(empty)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	right, err := kuma.NewFrame(kuma.NewSeries("v", int64(7)).Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	got, err := kuma.ConcatUnion(left, right)
	if err != nil {
		t.Fatalf("ConcatUnion: %v", err)
	}
	if got.NumRows() != 3 {
		t.Fatalf("there are %d rows, want 3", got.NumRows())
	}
	for i := range 3 {
		if !got.ColumnAt(0).IsNull(i) {
			t.Errorf("row %d of the null column has a value", i)
		}
	}
	if got.ColumnAt(1).Data().Value[int64](2) != 7 {
		t.Error("the value the second frame brought is not there")
	}
}

func TestConcatUnionErrors(t *testing.T) {
	f := trades(t)

	if _, err := kuma.ConcatUnion(); err == nil {
		t.Error("union of nothing succeeded")
	}
	if _, err := kuma.ConcatUnion(f, nil); err == nil {
		t.Error("union with a nil frame succeeded")
	}

	recast, err := f.Cast("qty", dtype.Int32)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	if _, wrongType := kuma.ConcatUnion(f, recast); wrongType == nil {
		t.Error("union of an int64 column and an int32 one succeeded")
	}
}

func TestHStack(t *testing.T) {
	left, err := kuma.NewFrame(kuma.NewSeries("a", int64(1), 2).Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	right, err := kuma.NewFrame(
		kuma.NewSeries("b", "x", "y").Column(),
		kuma.NewSeries("c", 0.5, 1.5).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	got, err := kuma.HStack(left, right)
	if err != nil {
		t.Fatalf("HStack: %v", err)
	}
	if names := strings.Join(got.Names(), ","); names != "a,b,c" {
		t.Errorf("the columns are %q", names)
	}
	want := []string{"1 x 0.5", "2 y 1.5"}
	if lines := rows(t, got); !equalLines(lines, want) {
		t.Errorf("the rows are\n%s", strings.Join(lines, "\n"))
	}
}

func TestHStackOne(t *testing.T) {
	f := trades(t)
	got, err := kuma.HStack(f)
	if err != nil {
		t.Fatalf("HStack: %v", err)
	}
	if got.NumCols() != f.NumCols() {
		t.Errorf("there are %d columns, want %d", got.NumCols(), f.NumCols())
	}
}

func TestHStackErrors(t *testing.T) {
	f := trades(t)

	if _, err := kuma.HStack(); err == nil {
		t.Error("stacking nothing succeeded")
	}
	if _, err := kuma.HStack(f, nil); err == nil {
		t.Error("stacking a nil frame succeeded")
	}
	if _, err := kuma.HStack(f, f.Head(2)); err == nil {
		t.Error("stacking frames of different heights succeeded")
	}
	if _, err := kuma.HStack(f, f); err == nil {
		t.Error("stacking a frame onto itself succeeded, and every name is in there twice")
	}
}

// TestConcatUnionUnfillable is the column a frame does not have and nothing can
// stand in for. A list column has no builder yet, so there is no way to make
// the nulls the other frame needs, and that is an error rather than a panic.
func TestConcatUnionUnfillable(t *testing.T) {
	data, err := array.NewChunked(dtype.List{Elem: dtype.Int64})
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	lists, err := kuma.NewColumn("l", data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	left, err := kuma.NewFrame(lists)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	right, err := kuma.NewFrame(kuma.NewSeries("v", int64(7)).Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	if _, err := kuma.ConcatUnion(left, right); err == nil {
		t.Error("filling a list column with nulls succeeded")
	}
}
