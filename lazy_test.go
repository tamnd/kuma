package kuma_test

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/plan"
)

// The tests here are written the way the API is meant to be used: a query is
// built in one expression and collected at the end, and what is checked is the
// frame that came back. A lazy query that gives a different answer from the
// same query written eagerly is the mistake worth catching, so most of these
// have both halves written out.

// TestLazyIsTheSameAsEager is the rule the whole lazy API rests on. The plan is
// there to be optimized, not to answer differently.
func TestLazyIsTheSameAsEager(t *testing.T) {
	f := trades(t)
	price := kuma.F64("price")

	lazy, err := f.Lazy().Filter(price.Gt(150)).SortDesc("price").Head(2).Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	kept, err := f.Filter(price.Gt(150))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	sorted, err := kept.Sort(kuma.Desc("price"))
	if err != nil {
		t.Fatalf("Sort: %v", err)
	}
	eager := sorted.Head(2)

	if lazy.String() != eager.String() {
		t.Errorf("the lazy query gave\n%s\nand the eager one gave\n%s", lazy, eager)
	}
	if got := symbolsOf(t, lazy); !slices.Equal(got, []string{"MSFT", "AAPL"}) {
		t.Errorf("symbols = %v, want MSFT, AAPL", got)
	}
}

// TestLazyCollectIsTyped checks the half of Collect that Bind does, which is
// that a query over a typed frame gives a typed frame back.
func TestLazyCollectIsTyped(t *testing.T) {
	f := typedTrades(t)

	out, err := f.Lazy().Filter(tradeCols.Qty.Ge(100)).Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	// The result is a Frame[Trade], so the handles still work on it.
	prices, err := tradeCols.Price.Series(out)
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := prices.Values(); !slices.Equal(got, []float64{189.5, 121.0}) {
		t.Errorf("prices = %v, want 189.5, 121", got)
	}
}

// TestLazyOnATypedFrameKeepsTheType is the other half of the type parameter: a
// step that cannot change the columns gives a query of the same schema back, so
// the handles keep working along the chain and the frame at the end needs no
// Bind of its own.
func TestLazyOnATypedFrameKeepsTheType(t *testing.T) {
	out, err := typedTrades(t).Lazy().
		Filter(tradeCols.Price.Gt(150)).
		SortBy("price").
		Head(1).
		Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	// The frame is a Frame[Trade] rather than a Dynamic one, which is what the
	// line below says: a handle written for another schema does not compile
	// against it.
	prices, err := tradeCols.Price.Series(out)
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := prices.Values(); !slices.Equal(got, []float64{189.5}) {
		t.Errorf("prices = %v, want 189.5", got)
	}
}

func TestLazySelect(t *testing.T) {
	out, err := trades(t).Lazy().Select("qty", "symbol").Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if got := out.Names(); !slices.Equal(got, []string{"qty", "symbol"}) {
		t.Errorf("Names() = %v, want qty, symbol", got)
	}
}

// TestLazyWith covers both things With does, since which one happens is decided
// by a name that is either already there or not.
func TestLazyWith(t *testing.T) {
	f := trades(t)
	price, qty := kuma.F64("price"), kuma.I64("qty")

	out, err := f.Lazy().
		With("notional", price.MulExpr(qty.AsF64())).
		With("price", price.Mul(2)).
		Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	// The replaced column stays where it was and the new one goes on the end,
	// which is what the eager WithExpr does.
	want := []string{"symbol", "price", "qty", "notional"}
	if got := out.Names(); !slices.Equal(got, want) {
		t.Fatalf("Names() = %v, want %v", got, want)
	}

	prices, err := out.Series[float64]("price")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := prices.Values()[0]; got != 379 {
		t.Errorf("the first price is %v, want 379", got)
	}

	notional, err := out.Series[float64]("notional")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := notional.Values()[0]; got != 18950 {
		t.Errorf("the first notional is %v, want 18950", got)
	}
}

func TestLazyDrop(t *testing.T) {
	out, err := trades(t).Lazy().Drop("price").Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if got := out.Names(); !slices.Equal(got, []string{"symbol", "qty"}) {
		t.Errorf("Names() = %v, want symbol, qty", got)
	}
}

// TestLazyDropOfAColumnThatIsNotThere is the case where doing nothing would be
// the easy answer and the wrong one, since the name was typed for a reason.
func TestLazyDropOfAColumnThatIsNotThere(t *testing.T) {
	_, err := trades(t).Lazy().Drop("pric").Collect(t.Context())
	if !errors.Is(err, kuma.ErrNoColumn) {
		t.Fatalf("Drop of a name that is not there = %v, want ErrNoColumn", err)
	}

	var ce *kuma.ColumnError
	if !errors.As(err, &ce) {
		t.Fatalf("the error is %T, want a *kuma.ColumnError", err)
	}
	if ce.Op != "Drop" {
		t.Errorf("the error says %q, want it to name Drop", ce.Op)
	}
	if !strings.Contains(err.Error(), "did you mean: price?") {
		t.Errorf("the message is %q, want it to suggest price", err.Error())
	}
}

func TestLazySortByExpr(t *testing.T) {
	f := trades(t)
	price, qty := kuma.F64("price"), kuma.I64("qty")

	out, err := f.Lazy().SortByExpr(price.MulExpr(qty.AsF64()), kuma.Order{Descending: true}).Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	// The notionals are 18950, 20560, 4752.5 and 48400, so the largest first is
	// NVDA, MSFT, AAPL, AAPL.
	if got := symbolsOf(t, out); !slices.Equal(got, []string{"NVDA", "MSFT", "AAPL", "AAPL"}) {
		t.Errorf("symbols = %v, want NVDA, MSFT, AAPL, AAPL", got)
	}

	// Sorting by something worked out does not leave it in the result.
	if got := out.Names(); !slices.Equal(got, []string{"symbol", "price", "qty"}) {
		t.Errorf("Names() = %v, want the columns that went in", got)
	}
}

func TestLazySortByTwoKeys(t *testing.T) {
	out, err := trades(t).Lazy().Sort(kuma.Asc("symbol"), kuma.Desc("price")).Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	prices, err := out.Series[float64]("price")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := prices.Values(); !slices.Equal(got, []float64{190.1, 189.5, 411.2, 121.0}) {
		t.Errorf("prices = %v, want the two AAPL rows first and the larger of them first", got)
	}
}

// TestLazySlice is the arithmetic of the limit operator, which is where an off
// by one would live.
func TestLazySlice(t *testing.T) {
	cases := []struct {
		name   string
		off, n int
		want   []string
	}{
		{name: "the whole frame", off: 0, n: 4, want: []string{"AAPL", "MSFT", "AAPL", "NVDA"}},
		{name: "the middle", off: 1, n: 2, want: []string{"MSFT", "AAPL"}},
		{name: "more rows than there are", off: 0, n: 100, want: []string{"AAPL", "MSFT", "AAPL", "NVDA"}},
		{name: "past the end", off: 10, n: 2, want: nil},
		{name: "no rows", off: 0, n: 0, want: nil},
		{name: "the last row", off: 3, n: 9, want: []string{"NVDA"}},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out, err := trades(t).Lazy().Slice(c.off, c.n).Collect(t.Context())
			if err != nil {
				t.Fatalf("Collect: %v", err)
			}
			if out.NumRows() != len(c.want) {
				t.Fatalf("%s, want %d rows", out, len(c.want))
			}
			if len(c.want) == 0 {
				return
			}
			if got := symbolsOf(t, out); !slices.Equal(got, c.want) {
				t.Errorf("symbols = %v, want %v", got, c.want)
			}
		})
	}
}

func TestLazyHeadPastTheEnd(t *testing.T) {
	out, err := trades(t).Lazy().Head(100).Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if out.NumRows() != 4 {
		t.Errorf("%s, want the four rows there are", out)
	}
}

// TestLazyColumnThatIsNotThere checks that the check happens before anything
// runs and that the error says which operator the name was written in.
func TestLazyColumnThatIsNotThere(t *testing.T) {
	_, err := trades(t).Lazy().Filter(kuma.F64("prcie").Gt(150)).Collect(t.Context())
	if !errors.Is(err, kuma.ErrNoColumn) {
		t.Fatalf("Collect of a query naming a column that is not there = %v, want ErrNoColumn", err)
	}

	var ce *kuma.ColumnError
	if !errors.As(err, &ce) {
		t.Fatalf("the error is %T, want a *kuma.ColumnError", err)
	}
	if ce.Op != "Filter" {
		t.Errorf("the error says %q, want it to name Filter", ce.Op)
	}
}

// TestLazyKeepsTheFirstMistake is the rule that lets a query be one expression:
// a step that went wrong is remembered rather than reported, and the steps
// after it build nothing.
func TestLazyKeepsTheFirstMistake(t *testing.T) {
	q := trades(t).Lazy().Drop("nope").Select("symbol").Drop("symbol")

	err := q.Validate()
	if !errors.Is(err, kuma.ErrNoColumn) {
		t.Fatalf("Validate = %v, want ErrNoColumn", err)
	}

	var ce *kuma.ColumnError
	if !errors.As(err, &ce) {
		t.Fatalf("the error is %T, want a *kuma.ColumnError", err)
	}
	if ce.Name != "nope" {
		t.Errorf("the error is about %q, want the name of the first step that went wrong", ce.Name)
	}

	if _, err := q.Schema(); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("Schema = %v, want the same error Validate gave", err)
	}
	if _, err := q.Collect(t.Context()); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("Collect = %v, want the same error Validate gave", err)
	}
	if got := q.String(); !strings.HasPrefix(got, "invalid query: ") {
		t.Errorf("String() = %q, want it to say the query is not one", got)
	}
}

// TestLazyExpressionOfLiteralsOnly is the length check the executor makes. An
// expression that reads no column produces one value, and one value is not a
// column of a frame of four rows.
func TestLazyExpressionOfLiteralsOnly(t *testing.T) {
	_, err := trades(t).Lazy().With("one", kuma.Lit(int64(1))).Collect(t.Context())
	if !errors.Is(err, kuma.ErrLength) {
		t.Fatalf("Collect of an expression reading no column = %v, want ErrLength", err)
	}
}

// TestLazyDuplicateColumn is the frame nobody can read a column out of by name,
// which the plan check catches before anything runs.
func TestLazyDuplicateColumn(t *testing.T) {
	_, err := trades(t).Lazy().Select("symbol", "symbol").Collect(t.Context())
	if !errors.Is(err, kuma.ErrDuplicateColumn) {
		t.Fatalf("Collect of a query naming a column twice = %v, want ErrDuplicateColumn", err)
	}
}

// TestLazyCollectStopsWhenGivenUpOn is the context check between operators.
func TestLazyCollectStopsWhenGivenUpOn(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	_, err := trades(t).Lazy().Head(1).Collect(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Collect with a canceled context = %v, want context.Canceled", err)
	}
}

func TestLazySchema(t *testing.T) {
	q := trades(t).Lazy().Select("price", "symbol")

	if err := q.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}

	s, err := q.Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}
	if got := s.Names(); !slices.Equal(got, []string{"price", "symbol"}) {
		t.Errorf("Schema() = %s, want price and symbol in that order", s)
	}
}

// TestLazyPlan checks that the query built the operators it reads as, since the
// plan is what the optimizer passes and an explain will be given.
func TestLazyPlan(t *testing.T) {
	q := trades(t).Lazy().Filter(kuma.F64("price").Gt(150)).SortDesc("price").Head(2)

	want := []plan.Op{plan.OpLimit, plan.OpSort, plan.OpFilter, plan.OpScan}
	var got []plan.Op
	for n := q.Plan(); n != nil; n = n.Input() {
		got = append(got, n.Op())
	}
	if !slices.Equal(got, want) {
		t.Errorf("the plan is %v, want %v", got, want)
	}
}

// TestLazyString is the tree as it prints, which is what a test failure and a
// prompt show.
func TestLazyString(t *testing.T) {
	q := trades(t).Lazy().Filter(kuma.F64("price").Gt(150)).SortDesc("price").Head(2)

	want := strings.Join([]string{
		"Limit 2",
		"  Sort by price desc",
		"    Filter (price > 150)",
		"      Scan frame",
	}, "\n")
	if got := q.String(); got != want {
		t.Errorf("String() =\n%s\nwant\n%s", got, want)
	}
}

// TestLazyStepsDoNotDisturbEachOther is what makes a query safe to keep and
// branch from, which is how a program builds two reports off one filter.
func TestLazyStepsDoNotDisturbEachOther(t *testing.T) {
	base := trades(t).Lazy().Filter(kuma.F64("price").Gt(150))

	head, err := base.Head(1).Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	all, err := base.Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	if head.NumRows() != 1 || all.NumRows() != 3 {
		t.Errorf("the head has %d rows and the query it came from has %d, want 1 and 3", head.NumRows(), all.NumRows())
	}
}

// TestLazyOverAnEmptyFrame is the shape every operator has to survive, since a
// filter that keeps nothing is what the next step reads.
func TestLazyOverAnEmptyFrame(t *testing.T) {
	f := trades(t)

	out, err := f.Lazy().
		Filter(kuma.Str("symbol").Eq("TSLA")).
		With("double", kuma.F64("price").Mul(2)).
		SortBy("price").
		Head(3).
		Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if out.NumRows() != 0 {
		t.Fatalf("%s, want no rows", out)
	}
	if got := out.Names(); !slices.Equal(got, []string{"symbol", "price", "qty", "double"}) {
		t.Errorf("Names() = %v, want the columns the query asked for", got)
	}
}

// TestLazyGroupBy is the group by written both ways, which is the rule that
// matters: the plan is there to be optimized and not to answer differently.
func TestLazyGroupBy(t *testing.T) {
	f := trades(t)

	lazy, err := f.Lazy().GroupBy("symbol").Agg(
		kuma.Sum("qty").As("total"),
		kuma.Mean("price").As("avg"),
		kuma.Size(),
	).Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	g, err := f.GroupBy("symbol")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	eager, err := g.Agg(
		kuma.Sum("qty").As("total"),
		kuma.Mean("price").As("avg"),
		kuma.Size(),
	)
	if err != nil {
		t.Fatalf("Agg: %v", err)
	}

	if lazy.String() != eager.String() {
		t.Errorf("the lazy group by gave\n%s\nand the eager one gave\n%s", lazy, eager)
	}
	if got := lazy.Names(); !slices.Equal(got, []string{"symbol", "total", "avg", "size"}) {
		t.Errorf("Names() = %v, want the key and then the aggregations", got)
	}
	if got := symbolsOf(t, lazy); !slices.Equal(got, []string{"AAPL", "MSFT", "NVDA"}) {
		t.Errorf("symbols = %v, want the groups in the order they first appear", got)
	}
}

// TestLazyEveryAggregation runs the lot of them against the eager answer, since
// the two now share one switch and the way to know it is wired up right is to
// ask for every arm.
func TestLazyEveryAggregation(t *testing.T) {
	aggs := []kuma.Aggregation{
		kuma.Sum("qty"),
		kuma.Mean("qty"),
		kuma.Min("qty"),
		kuma.Max("qty"),
		kuma.Count("qty"),
		kuma.Size(),
		kuma.First("qty"),
		kuma.Last("qty"),
		kuma.Var("qty", 1),
		kuma.Std("qty", 1),
		kuma.Median("qty"),
		kuma.Quantile("qty", 0.9, kuma.Linear),
		kuma.NUnique("qty"),
	}

	f := trades(t)
	for _, a := range aggs {
		t.Run(a.String(), func(t *testing.T) {
			named := a.As("answer")

			lazy, err := f.Lazy().GroupBy("symbol").Agg(named).Collect(t.Context())
			if err != nil {
				t.Fatalf("Collect: %v", err)
			}

			g, err := f.GroupBy("symbol")
			if err != nil {
				t.Fatalf("GroupBy: %v", err)
			}
			eager, err := g.Agg(named)
			if err != nil {
				t.Fatalf("Agg: %v", err)
			}

			if lazy.String() != eager.String() {
				t.Errorf("lazy gave\n%s\nand eager gave\n%s", lazy, eager)
			}
		})
	}
}

// TestLazyGroupByCount is the group by anybody writes first.
func TestLazyGroupByCount(t *testing.T) {
	out, err := trades(t).Lazy().GroupBy("symbol").Count().Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	s, err := out.Series[int64]("size")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := s.Values(); !slices.Equal(got, []int64{2, 1, 1}) {
		t.Errorf("sizes = %v, want 2, 1, 1", got)
	}
}

// TestLazyGroupByTwoKeys checks that a row is in a group only when every key
// agrees, which here splits the two AAPL rows apart.
func TestLazyGroupByTwoKeys(t *testing.T) {
	out, err := trades(t).Lazy().GroupBy("symbol", "qty").Count().Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if out.NumRows() != 4 {
		t.Fatalf("%s, want one group per row", out)
	}
	if got := out.Names(); !slices.Equal(got, []string{"symbol", "qty", "size"}) {
		t.Errorf("Names() = %v, want both keys and then the count", got)
	}
}

// TestLazyGroupByIsAStepLikeAnyOther is what the lazy API buys over the eager
// one here, which is that the frame a group by produces is a query and can be
// gone on with.
func TestLazyGroupByIsAStepLikeAnyOther(t *testing.T) {
	out, err := trades(t).Lazy().
		Filter(kuma.F64("price").Gt(150)).
		GroupBy("symbol").
		Agg(kuma.Sum("qty").As("total")).
		SortDesc("total").
		Head(1).
		Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	if got := symbolsOf(t, out); !slices.Equal(got, []string{"AAPL"}) {
		t.Errorf("symbols = %v, want AAPL, whose two dear rows total 125", got)
	}
}

// TestLazyGroupBySchema is the schema of a group by worked out without reading
// anything, which is what a program checking a report before it runs asks for.
func TestLazyGroupBySchema(t *testing.T) {
	s, err := trades(t).Lazy().GroupBy("symbol").Agg(
		kuma.Sum("qty").As("total"),
		kuma.Size(),
	).Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}
	if got := s.Names(); !slices.Equal(got, []string{"symbol", "total", "size"}) {
		t.Errorf("Names() = %v, want the key and then the aggregations", got)
	}
}

// TestLazyGroupByString is the operator as an explain shows it, with the keys
// before the colon and the aggregations after.
func TestLazyGroupByString(t *testing.T) {
	q := trades(t).Lazy().GroupBy("symbol").Agg(kuma.Sum("qty").As("total"), kuma.Size())

	want := strings.Join([]string{
		"Aggregate by symbol: Sum(qty) as total, Size() as size",
		"  Scan frame",
	}, "\n")
	if got := q.String(); got != want {
		t.Errorf("String() =\n%s\nwant\n%s", got, want)
	}
}

func TestLazyGroupByMistakes(t *testing.T) {
	f := trades(t)

	tests := []struct {
		name string
		q    *kuma.LazyFrame[kuma.Dynamic]
		want error
	}{
		{
			name: "no keys",
			q:    f.Lazy().GroupBy().Agg(kuma.Size()),
			want: kuma.ErrLength,
		},
		{
			name: "no aggregations",
			q:    f.Lazy().GroupBy("symbol").Agg(),
			want: kuma.ErrLength,
		},
		{
			name: "a key that is not there",
			q:    f.Lazy().GroupBy("sybmol").Agg(kuma.Size()),
			want: kuma.ErrNoColumn,
		},
		{
			name: "a column that is not there",
			q:    f.Lazy().GroupBy("symbol").Agg(kuma.Sum("qtty")),
			want: kuma.ErrNoColumn,
		},
		{
			name: "two aggregations of one name",
			q:    f.Lazy().GroupBy("symbol").Agg(kuma.Sum("qty"), kuma.Mean("qty")),
			want: kuma.ErrDuplicateColumn,
		},
		{
			name: "a mistake written before the group by",
			q:    f.Lazy().Drop("prices").GroupBy("symbol").Agg(kuma.Size()),
			want: kuma.ErrNoColumn,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.q.Validate(); !errors.Is(err, tt.want) {
				t.Fatalf("Validate() = %v, want %v", err, tt.want)
			}
			if _, err := tt.q.Collect(t.Context()); !errors.Is(err, tt.want) {
				t.Fatalf("Collect() = %v, want %v", err, tt.want)
			}
		})
	}
}

// TestLazyAnAggregationWithNoAnswer is the check that costs nothing, since the
// sum of a column of strings is a mistake that can be found before a row is
// read rather than after the grouping has been paid for.
func TestLazyAnAggregationWithNoAnswer(t *testing.T) {
	err := trades(t).Lazy().GroupBy("qty").Agg(kuma.Sum("symbol")).Validate()
	if err == nil {
		t.Fatal("Validate() = nil, want the sum of a string turned away")
	}
	if !strings.Contains(err.Error(), "no sum of a string") {
		t.Errorf("the message is %q, want it to say there is no sum of a string", err)
	}
}

// BenchmarkLazyCollect is the whole path: the plan is checked, then a filter, a
// sort and a limit run one after another. What it is here to watch is the cost
// of the plan on top of the work, which is a schema check per query and a
// closure per operator and has to stay lost in the noise of the gather.
func BenchmarkLazyCollect(b *testing.B) {
	f := benchFrame(b)
	qty := kuma.I64("c00")
	ctx := b.Context()

	b.ReportAllocs()
	for b.Loop() {
		out, err := f.Lazy().Filter(qty.Gt(benchLen / 2)).SortDesc("c01").Head(64).Collect(ctx)
		if err != nil {
			b.Fatalf("Collect: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkLazyPlan is the query being written and not run, which is what a
// program does once per report and an optimizer pass does per rewrite.
func BenchmarkLazyPlan(b *testing.B) {
	f := benchFrame(b)
	qty := kuma.I64("c00")

	b.ReportAllocs()
	for b.Loop() {
		planSink = f.Lazy().Filter(qty.Gt(benchLen / 2)).SortDesc("c01").Head(64).Plan()
	}
}

// BenchmarkLazyGroupByAndAgg is BenchmarkFrameGroupByAndAgg written as a query,
// over the same frame and asking for the same two things. The gap between the
// two is the whole cost of going through the plan, since the grouping and the
// aggregations underneath are the same kernels on the same data.
func BenchmarkLazyGroupByAndAgg(b *testing.B) {
	f := benchGrouped(b).Frame()
	ctx := b.Context()

	b.ReportAllocs()
	for b.Loop() {
		out, err := f.Lazy().GroupBy("k").Agg(
			kuma.Sum("qty").As("total"),
			kuma.Mean("price").As("avg"),
		).Collect(ctx)
		if err != nil {
			b.Fatalf("Collect: %v", err)
		}
		frameSink = out
	}
}

// planSink keeps the plan the benchmark built from being optimized away.
var planSink *plan.Node
