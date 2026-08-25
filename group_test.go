package kuma_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

func TestGroupBy(t *testing.T) {
	f := trades(t)

	g, err := f.GroupBy("symbol")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if got := g.NumGroups(); got != 3 {
		t.Errorf("there are %d groups, want 3", got)
	}
	if got := strings.Join(g.Names(), ","); got != "symbol" {
		t.Errorf("the keys are %q, want symbol", got)
	}
	if g.Frame() != f {
		t.Error("the grouping does not point back at the frame it came from")
	}
	if g.Groups().Len() != f.NumRows() {
		t.Errorf("the grouping covers %d rows, the frame has %d",
			g.Groups().Len(), f.NumRows())
	}

	keys := g.Keys()
	if len(keys) != 1 {
		t.Fatalf("there are %d key columns, want 1", len(keys))
	}
	if got := keys[0].Name(); got != "symbol" {
		t.Errorf("the key column is called %q, want symbol", got)
	}
	// First appearance order, so AAPL comes before MSFT because it was first.
	wantKeys := []string{"AAPL", "MSFT", "NVDA"}
	for i, want := range wantKeys {
		if got := string(keys[0].Data().Bytes(i)); got != want {
			t.Errorf("group %d has key %q, want %q", i, got, want)
		}
	}
}

func TestGroupByAgg(t *testing.T) {
	f := trades(t)

	g, err := f.GroupBy("symbol")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	got, err := g.Agg(
		kuma.Sum("qty").As("total"),
		kuma.Mean("price").As("avg"),
		kuma.Size(),
	)
	if err != nil {
		t.Fatalf("Agg: %v", err)
	}

	if want := "symbol,total,avg,size"; strings.Join(got.Names(), ",") != want {
		t.Errorf("the columns are %q, want %q", strings.Join(got.Names(), ","), want)
	}
	if got.NumRows() != 3 {
		t.Errorf("there are %d rows, want one per group", got.NumRows())
	}

	total, err := kuma.SeriesFrom[int64]("total", got.Columns()[1].Data())
	if err != nil {
		t.Fatalf("SeriesFrom: %v", err)
	}
	if v := total.Value(0); v != 125 {
		t.Errorf("AAPL adds up to %d, want 125", v)
	}

	avg, err := kuma.SeriesFrom[float64]("avg", got.Columns()[2].Data())
	if err != nil {
		t.Fatalf("SeriesFrom: %v", err)
	}
	if v := avg.Value(0); v != 189.8 {
		t.Errorf("AAPL averages %v, want 189.8", v)
	}
}

// TestGroupByEveryAggregation runs the whole list over one grouping, which is
// the check that each of them is wired to the kernel that shares its name.
func TestGroupByEveryAggregation(t *testing.T) {
	f, err := kuma.NewFrame(
		kuma.NewSeries("k", "a", "a", "a", "b").Column(),
		kuma.NewSeries("v", 1.0, 3.0, 5.0, 9.0).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	g, err := f.GroupBy("k")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}

	tests := []struct {
		agg  kuma.Aggregation
		want float64
	}{
		{kuma.Sum("v"), 9},
		{kuma.Mean("v"), 3},
		{kuma.Min("v"), 1},
		{kuma.Max("v"), 5},
		{kuma.First("v"), 1},
		{kuma.Last("v"), 5},
		{kuma.Var("v", 1), 4},
		{kuma.Std("v", 1), 2},
		{kuma.Median("v"), 3},
		{kuma.Quantile("v", 0.25, kuma.Linear), 2},
		{kuma.Quantile("v", 0.25, kuma.Lower), 1},
		{kuma.Quantile("v", 0.25, kuma.Higher), 3},
	}

	for _, tt := range tests {
		t.Run(tt.agg.String(), func(t *testing.T) {
			out, err := g.Agg(tt.agg)
			if err != nil {
				t.Fatalf("Agg: %v", err)
			}
			s, err := kuma.SeriesFrom[float64]("v", out.Columns()[1].Data())
			if err != nil {
				t.Fatalf("SeriesFrom: %v", err)
			}
			if v := s.Value(0); v != tt.want {
				t.Errorf("group a gives %v, want %v", v, tt.want)
			}
		})
	}

	// The three that count rather than measure, which come back as int64.
	counts := []struct {
		agg  kuma.Aggregation
		want int64
	}{
		{kuma.Count("v"), 3},
		{kuma.Size(), 3},
		{kuma.NUnique("v"), 3},
	}
	for _, tt := range counts {
		t.Run(tt.agg.String(), func(t *testing.T) {
			out, err := g.Agg(tt.agg)
			if err != nil {
				t.Fatalf("Agg: %v", err)
			}
			s, err := kuma.SeriesFrom[int64]("n", out.Columns()[1].Data())
			if err != nil {
				t.Fatalf("SeriesFrom: %v", err)
			}
			if v := s.Value(0); v != tt.want {
				t.Errorf("group a gives %d, want %d", v, tt.want)
			}
		})
	}
}

func TestGroupBySeveralKeys(t *testing.T) {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "AAPL", "MSFT", "AAPL").Column(),
		kuma.NewSeries("side", "BUY", "SELL", "BUY", "BUY").Column(),
		kuma.NewSeries("qty", int64(10), 20, 30, 40).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	g, err := f.GroupBy("symbol", "side")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if got := g.NumGroups(); got != 3 {
		t.Errorf("there are %d groups, want 3", got)
	}

	out, err := g.Agg(kuma.Sum("qty"))
	if err != nil {
		t.Fatalf("Agg: %v", err)
	}
	if want := "symbol,side,qty"; strings.Join(out.Names(), ",") != want {
		t.Errorf("the columns are %q, want %q", strings.Join(out.Names(), ","), want)
	}

	qty, err := kuma.SeriesFrom[int64]("qty", out.Columns()[2].Data())
	if err != nil {
		t.Fatalf("SeriesFrom: %v", err)
	}
	if v := qty.Value(0); v != 50 {
		t.Errorf("AAPL BUY adds up to %d, want 50", v)
	}
}

// TestGroupByNullKey is the rule that a missing key is a group rather than a
// reason to drop the row, which is what SQL does and what pandas does not.
func TestGroupByNullKey(t *testing.T) {
	b, err := array.NewBuilder(dtype.String)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	b.AppendString("BUY")
	b.AppendNull()
	b.AppendString("BUY")
	data, err := array.NewChunked(dtype.String, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	side, err := kuma.NewColumn("side", data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}

	f, err := kuma.NewFrame(side, kuma.NewSeries("qty", int64(1), 2, 3).Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	g, err := f.GroupBy("side")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if got := g.NumGroups(); got != 2 {
		t.Fatalf("there are %d groups, want 2 with the missing one of them", got)
	}

	out, err := g.Agg(kuma.Sum("qty"))
	if err != nil {
		t.Fatalf("Agg: %v", err)
	}
	if !out.Columns()[0].IsNull(1) {
		t.Error("the second group has a key, want the missing one")
	}
	qty, err := kuma.SeriesFrom[int64]("qty", out.Columns()[1].Data())
	if err != nil {
		t.Fatalf("SeriesFrom: %v", err)
	}
	if v := qty.Value(1); v != 2 {
		t.Errorf("the missing group adds up to %d, want 2", v)
	}
}

func TestGroupByCount(t *testing.T) {
	f := trades(t)

	g, err := f.GroupBy("symbol")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	out, err := g.Count()
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if want := "symbol,size"; strings.Join(out.Names(), ",") != want {
		t.Errorf("the columns are %q, want %q", strings.Join(out.Names(), ","), want)
	}

	n, err := kuma.SeriesFrom[int64]("size", out.Columns()[1].Data())
	if err != nil {
		t.Fatalf("SeriesFrom: %v", err)
	}
	if v := n.Value(0); v != 2 {
		t.Errorf("AAPL has %d rows, want 2", v)
	}
}

func TestGroupByErrors(t *testing.T) {
	f := trades(t)

	if _, err := f.GroupBy(); err == nil {
		t.Error("grouping by nothing succeeded")
	}
	if _, err := f.GroupBy("nope"); err == nil {
		t.Error("grouping by a column that is not there succeeded")
	} else if !strings.Contains(err.Error(), "nope") {
		t.Errorf("the message is %q, want it to name the column asked for", err)
	}

	g, err := f.GroupBy("symbol")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}

	if _, err := g.Agg(); err == nil {
		t.Error("aggregating nothing succeeded")
	}
	if _, err := g.Agg(kuma.Sum("nope")); err == nil {
		t.Error("summing a column that is not there succeeded")
	}
	if _, err := g.Agg(kuma.Sum("symbol")); err == nil {
		t.Error("summing a string column succeeded")
	}
	if _, err := g.Agg(kuma.Quantile("price", 2, kuma.Linear)); err == nil {
		t.Error("a quantile of two succeeded")
	}
	if _, err := g.Agg(kuma.Std("price", -1)); err == nil {
		t.Error("a negative ddof succeeded")
	}

	// Two aggregations of one column, neither of them named, which is a frame
	// with two columns called price.
	if _, err := g.Agg(kuma.Min("price"), kuma.Max("price")); err == nil {
		t.Error("two aggregations of one column with no names succeeded")
	}
	if _, err := g.Agg(kuma.Min("price"), kuma.Max("price").As("high")); err != nil {
		t.Errorf("naming one of them was not enough: %v", err)
	}
}

func TestAggregationName(t *testing.T) {
	tests := []struct {
		agg  kuma.Aggregation
		name string
		str  string
	}{
		{kuma.Sum("qty"), "qty", "Sum(qty)"},
		{kuma.Sum("qty").As("total"), "total", "Sum(qty) as total"},
		{kuma.Size(), "size", "Size() as size"},
		{kuma.Size().As("n"), "n", "Size() as n"},
		{kuma.Var("p", 1), "p", "Var(p, 1)"},
		{kuma.Std("p", 0), "p", "Std(p, 0)"},
		{kuma.Quantile("p", 0.95, kuma.Nearest), "p", "Quantile(p, 0.95, nearest)"},
		{kuma.Mean("p"), "p", "Mean(p)"},
		{kuma.Min("p"), "p", "Min(p)"},
		{kuma.Max("p"), "p", "Max(p)"},
		{kuma.Count("p"), "p", "Count(p)"},
		{kuma.First("p"), "p", "First(p)"},
		{kuma.Last("p"), "p", "Last(p)"},
		{kuma.Median("p"), "p", "Median(p)"},
		{kuma.NUnique("p"), "p", "NUnique(p)"},
	}

	for _, tt := range tests {
		t.Run(tt.str, func(t *testing.T) {
			if got := tt.agg.Name(); got != tt.name {
				t.Errorf("the result column is called %q, want %q", got, tt.name)
			}
			if got := tt.agg.String(); got != tt.str {
				t.Errorf("it reads as %q, want %q", got, tt.str)
			}
		})
	}
}

// TestGroupByNaN is the rule that NaN is a value and not a missing one, so it
// wins a maximum and loses a minimum.
func TestGroupByNaN(t *testing.T) {
	f, err := kuma.NewFrame(
		kuma.NewSeries("k", "a", "a").Column(),
		kuma.NewSeries("v", 1.0, math.NaN()).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	g, err := f.GroupBy("k")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}

	out, err := g.Agg(kuma.Min("v").As("low"), kuma.Max("v").As("high"))
	if err != nil {
		t.Fatalf("Agg: %v", err)
	}
	low, err := kuma.SeriesFrom[float64]("low", out.Columns()[1].Data())
	if err != nil {
		t.Fatalf("SeriesFrom: %v", err)
	}
	high, err := kuma.SeriesFrom[float64]("high", out.Columns()[2].Data())
	if err != nil {
		t.Fatalf("SeriesFrom: %v", err)
	}
	if low.Value(0) != 1 {
		t.Errorf("the smallest is %v, want 1", low.Value(0))
	}
	if !math.IsNaN(high.Value(0)) {
		t.Errorf("the largest is %v, want NaN", high.Value(0))
	}
}

// TestGroupByRefused is the error from the kernel coming through, which is a
// column of a type there is no key encoding for. A float is fine and a list is
// not, and the kernel tests cover the rest of that list.
func TestGroupByRefused(t *testing.T) {
	f := trades(t)
	if _, err := f.GroupBy("price"); err != nil {
		t.Errorf("grouping by a float column failed: %v", err)
	}
	if f.Schema().Fields[1].Type != dtype.Float64 {
		t.Error("the price column stopped being a float64")
	}

	data, err := array.NewChunked(dtype.List{Elem: dtype.Int64})
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	lists, err := kuma.NewColumn("l", data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	empty, err := kuma.NewFrame(lists)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	if _, err := empty.GroupBy("l"); err == nil {
		t.Error("grouping by a list column succeeded")
	}
}

// TestGroupByCountClash is the one way Count can fail, which is a key column
// that is already called size.
func TestGroupByCountClash(t *testing.T) {
	f, err := kuma.NewFrame(kuma.NewSeries("size", "a", "b", "a").Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	g, err := f.GroupBy("size")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if _, err := g.Count(); err == nil {
		t.Error("counting into a column called size succeeded")
	}
}
