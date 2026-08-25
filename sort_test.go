package kuma_test

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

func TestFrameSort(t *testing.T) {
	f := mustFrame(t,
		kuma.NewSeries("symbol", "b", "a", "b", "a").Column(),
		kuma.NewSeries("qty", int64(2), 9, 7, 1).Column(),
	)

	tests := []struct {
		name   string
		sort   func() (*kuma.Frame[kuma.Dynamic], error)
		symbol []string
		qty    []int64
	}{
		{
			"one key",
			func() (*kuma.Frame[kuma.Dynamic], error) { return f.SortBy("symbol") },
			[]string{"a", "a", "b", "b"},
			[]int64{9, 1, 2, 7},
		},
		{
			"one key descending",
			func() (*kuma.Frame[kuma.Dynamic], error) { return f.SortDesc("qty") },
			[]string{"a", "b", "b", "a"},
			[]int64{9, 7, 2, 1},
		},
		{
			"two keys",
			func() (*kuma.Frame[kuma.Dynamic], error) {
				return f.Sort(kuma.Asc("symbol"), kuma.Desc("qty"))
			},
			[]string{"a", "a", "b", "b"},
			[]int64{9, 1, 7, 2},
		},
		{
			"two keys both ascending",
			func() (*kuma.Frame[kuma.Dynamic], error) { return f.SortBy("symbol", "qty") },
			[]string{"a", "a", "b", "b"},
			[]int64{1, 9, 2, 7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.sort()
			if err != nil {
				t.Fatalf("sorting: %v", err)
			}
			if got.NumRows() != f.NumRows() {
				t.Fatalf("the result has %d rows, want %d", got.NumRows(), f.NumRows())
			}

			symbol, err := got.Series[string]("symbol")
			if err != nil {
				t.Fatalf("Series: %v", err)
			}
			if !slices.Equal(symbol.Values(), tt.symbol) {
				t.Errorf("symbol is %v, want %v", symbol.Values(), tt.symbol)
			}

			qty, err := got.Series[int64]("qty")
			if err != nil {
				t.Fatalf("Series: %v", err)
			}
			if !slices.Equal(qty.Values(), tt.qty) {
				t.Errorf("qty is %v, want %v", qty.Values(), tt.qty)
			}
		})
	}
}

// TestFrameSortLeavesTheOriginal is the promise every operation here makes. A
// frame is immutable and sorting one gives back another.
func TestFrameSortLeavesTheOriginal(t *testing.T) {
	f := mustFrame(t, kuma.NewSeries("qty", int64(3), 1, 2).Column())

	if _, err := f.SortBy("qty"); err != nil {
		t.Fatalf("SortBy: %v", err)
	}

	qty, err := f.Series[int64]("qty")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if want := []int64{3, 1, 2}; !slices.Equal(qty.Values(), want) {
		t.Errorf("the original is %v, want %v", qty.Values(), want)
	}
}

func TestFrameSortIndex(t *testing.T) {
	f := mustFrame(t, kuma.NewSeries("qty", int64(3), 1, 2).Column())

	got, err := f.SortIndex(kuma.Asc("qty"))
	if err != nil {
		t.Fatalf("SortIndex: %v", err)
	}
	if want := []int{1, 2, 0}; !slices.Equal(got, want) {
		t.Errorf("the order is %v, want %v", got, want)
	}
}

func TestFrameSortErrors(t *testing.T) {
	f := mustFrame(t, kuma.NewSeries("qty", int64(3), 1, 2).Column())

	t.Run("a column that is not there", func(t *testing.T) {
		_, err := f.SortBy("quantity")
		if err == nil {
			t.Fatal("sorting by a column that is not there succeeded")
		}
		// The suggestion comes from the column error, which is what makes a
		// typo in a name findable.
		if !strings.Contains(err.Error(), "qty") {
			t.Errorf("the message is %q, want it to suggest the near miss", err.Error())
		}
	})

	t.Run("no columns", func(t *testing.T) {
		_, err := f.Sort()
		if err == nil {
			t.Fatal("sorting by nothing succeeded")
		}
		if !strings.Contains(err.Error(), "no columns") {
			t.Errorf("the message is %q, want it to say there is nothing to sort by", err.Error())
		}
	})
}

func TestSeriesSort(t *testing.T) {
	s := kuma.NewSeries("qty", int64(3), 1, 2)

	got, err := s.Sort(kuma.Order{})
	if err != nil {
		t.Fatalf("Sort: %v", err)
	}
	if want := []int64{1, 2, 3}; !slices.Equal(got.Values(), want) {
		t.Errorf("ascending is %v, want %v", got.Values(), want)
	}
	if got.Name() != "qty" {
		t.Errorf("the name is %q, want it kept", got.Name())
	}

	got, err = s.Sort(kuma.Order{Descending: true})
	if err != nil {
		t.Fatalf("Sort: %v", err)
	}
	if want := []int64{3, 2, 1}; !slices.Equal(got.Values(), want) {
		t.Errorf("descending is %v, want %v", got.Values(), want)
	}
}

// TestSeriesSortNulls checks the placement through the root API, since the
// whole point of the Order type is that a caller can say where they go.
func TestSeriesSortNulls(t *testing.T) {
	s := intsWithANull(t)

	last, err := s.Sort(kuma.Order{})
	if err != nil {
		t.Fatalf("Sort: %v", err)
	}
	if last.IsNull(0) || last.IsNull(1) || !last.IsNull(2) {
		t.Error("the null is not at the end")
	}

	first, err := s.Sort(kuma.Order{NullsFirst: true})
	if err != nil {
		t.Fatalf("Sort: %v", err)
	}
	if !first.IsNull(0) {
		t.Error("the null is not at the start")
	}
}

// intsWithANull returns three values, the middle one missing.
func intsWithANull(t *testing.T) kuma.Series[int64] {
	t.Helper()

	b, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	b.Append(int64(3))
	b.AppendNull()
	b.Append(int64(1))

	c, err := array.NewChunked(dtype.Int64, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	s, err := kuma.SeriesFrom[int64]("qty", c)
	if err != nil {
		t.Fatalf("SeriesFrom: %v", err)
	}
	return s
}

func TestSeriesSortIndex(t *testing.T) {
	s := kuma.NewSeries("qty", int64(3), 1, 2)

	got, err := s.SortIndex(kuma.Order{Descending: true})
	if err != nil {
		t.Fatalf("SortIndex: %v", err)
	}
	if want := []int{0, 2, 1}; !slices.Equal(got, want) {
		t.Errorf("the order is %v, want %v", got, want)
	}
}

// TestSeriesSortNaN is the same rule the kernel has, checked through the API
// people will actually hit it with. A NaN is a value and sorts after the
// numbers, so ascending order ends with the rows worth looking at.
func TestSeriesSortNaN(t *testing.T) {
	s := kuma.NewSeries("price", 2.0, math.NaN(), 1.0)

	got, err := s.Sort(kuma.Order{})
	if err != nil {
		t.Fatalf("Sort: %v", err)
	}
	values := got.Values()
	if values[0] != 1 || values[1] != 2 || !math.IsNaN(values[2]) {
		t.Errorf("the order is %v, want 1 then 2 then NaN", values)
	}
}

func TestColumnSort(t *testing.T) {
	c := kuma.NewSeries("symbol", "b", "a", "c").Column()

	got, err := c.Sort(kuma.Order{})
	if err != nil {
		t.Fatalf("Sort: %v", err)
	}

	s, err := got.As[string]()
	if err != nil {
		t.Fatalf("As: %v", err)
	}
	if want := []string{"a", "b", "c"}; !slices.Equal(s.Values(), want) {
		t.Errorf("the order is %v, want %v", s.Values(), want)
	}
	if got.Name() != "symbol" {
		t.Errorf("the name is %q, want it kept", got.Name())
	}
}

// TestSortByHelpers checks that the two shorthands build what they say they do,
// since everything else here goes through them.
func TestSortByHelpers(t *testing.T) {
	if got := kuma.Asc("qty"); got.Name != "qty" || got.Descending || got.NullsFirst {
		t.Errorf("Asc built %+v, want an ascending key with the nulls last", got)
	}
	if got := kuma.Desc("qty"); got.Name != "qty" || !got.Descending || got.NullsFirst {
		t.Errorf("Desc built %+v, want a descending key with the nulls last", got)
	}
}
