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

// TestLazyJoin runs all seven joins both ways, since a join is where a wrong
// answer is easiest to write and hardest to notice.
func TestLazyJoin(t *testing.T) {
	left, right := trades(t), sectors(t)

	for _, how := range []kuma.JoinType{
		kuma.InnerJoin, kuma.LeftJoin, kuma.RightJoin, kuma.OuterJoin,
		kuma.SemiJoin, kuma.AntiJoin,
	} {
		t.Run(how.String(), func(t *testing.T) {
			lazy, err := left.Lazy().Join(right.Lazy(), kuma.Using("symbol"), how).Collect(t.Context())
			if err != nil {
				t.Fatalf("Collect: %v", err)
			}
			eager, err := left.Join(right, kuma.Using("symbol"), how)
			if err != nil {
				t.Fatalf("Join: %v", err)
			}
			if lazy.String() != eager.String() {
				t.Errorf("the lazy join gave\n%s\nand the eager one gave\n%s", lazy, eager)
			}
		})
	}
}

// TestLazyFilterOverAJoin is the case a wrong predicate pushdown gets quietly
// wrong. The optimizer is free to move a filter under the join, and moving one
// under the side of an outer join that gets filled with nulls changes the
// answer without changing the shape of it, so the check is against the eager
// join filtered afterwards, which cannot move anything anywhere.
func TestLazyFilterOverAJoin(t *testing.T) {
	left, right := trades(t), sectors(t)

	tests := []struct {
		name string
		how  kuma.JoinType
		cond kuma.BoolValue[kuma.Dynamic]
	}{
		{"the left side of an inner join", kuma.InnerJoin, kuma.F64("price").Gt(150)},
		{"the right side of an inner join", kuma.InnerJoin, kuma.Str("sector").Eq("hardware")},
		{"the left side of a left join", kuma.LeftJoin, kuma.F64("price").Gt(150)},
		{"the right side of a left join", kuma.LeftJoin, kuma.Str("sector").Eq("hardware")},
		{"the left side of a right join", kuma.RightJoin, kuma.F64("price").Gt(150)},
		{"the right side of a right join", kuma.RightJoin, kuma.Str("sector").Eq("hardware")},
		{"the left side of an outer join", kuma.OuterJoin, kuma.F64("price").Gt(150)},
		{"the right side of an outer join", kuma.OuterJoin, kuma.Str("sector").Eq("hardware")},
		{"both sides at once", kuma.LeftJoin, kuma.F64("price").Gt(150).And(kuma.Str("sector").Eq("hardware"))},
		{"a semi join, which keeps no column of its right side", kuma.SemiJoin, kuma.I64("qty").Lt(100)},
		{"an anti join, which keeps none of them either", kuma.AntiJoin, kuma.I64("qty").Lt(100)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lazy, err := left.Lazy().
				Join(right.Lazy(), kuma.Using("symbol"), tt.how).
				Filter(tt.cond).
				Collect(t.Context())
			if err != nil {
				t.Fatalf("Collect: %v", err)
			}

			joined, err := left.Join(right, kuma.Using("symbol"), tt.how)
			if err != nil {
				t.Fatalf("Join: %v", err)
			}
			eager, err := joined.Filter(tt.cond)
			if err != nil {
				t.Fatalf("Filter: %v", err)
			}

			if lazy.String() != eager.String() {
				t.Errorf("the lazy query gave\n%s\nand the eager one gave\n%s", lazy, eager)
			}
		})
	}
}

// TestLazyFilterOverTheRestOfThePlan is the same check for the operators a
// filter can be written above that are not joins, since each of them decides
// for itself whether what is above it may go below it.
func TestLazyFilterOverTheRestOfThePlan(t *testing.T) {
	f := trades(t)
	dear := kuma.F64("price").Gt(150)

	tests := []struct {
		name  string
		lazy  func() *kuma.LazyFrame[kuma.Dynamic]
		eager func() (*kuma.Frame[kuma.Dynamic], error)
	}{
		{
			name: "over a sort, which the filter goes under",
			lazy: func() *kuma.LazyFrame[kuma.Dynamic] {
				return f.Lazy().Sort(kuma.Asc("qty")).Filter(dear)
			},
			eager: func() (*kuma.Frame[kuma.Dynamic], error) {
				sorted, err := f.Sort(kuma.Asc("qty"))
				if err != nil {
					return nil, err
				}
				return sorted.Filter(dear)
			},
		},
		{
			name: "over a limit, which it stays above",
			lazy: func() *kuma.LazyFrame[kuma.Dynamic] {
				return f.Lazy().Sort(kuma.Asc("qty")).Head(3).Filter(dear)
			},
			eager: func() (*kuma.Frame[kuma.Dynamic], error) {
				sorted, err := f.Sort(kuma.Asc("qty"))
				if err != nil {
					return nil, err
				}
				return sorted.Head(3).Filter(dear)
			},
		},
		{
			name: "over a distinct by one column, which it stays above",
			lazy: func() *kuma.LazyFrame[kuma.Dynamic] {
				return f.Lazy().Distinct("symbol").Filter(dear)
			},
			eager: func() (*kuma.Frame[kuma.Dynamic], error) {
				one, err := f.Distinct("symbol")
				if err != nil {
					return nil, err
				}
				return one.Filter(dear)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lazy, err := tt.lazy().Collect(t.Context())
			if err != nil {
				t.Fatalf("Collect: %v", err)
			}
			eager, err := tt.eager()
			if err != nil {
				t.Fatalf("the eager query: %v", err)
			}
			if lazy.String() != eager.String() {
				t.Errorf("the lazy query gave\n%s\nand the eager one gave\n%s", lazy, eager)
			}
		})
	}
}

// TestLazySliceOverTheRestOfThePlan is the check on the slice pushdown, which
// is the pass most able to give a wrong answer that looks right, since almost
// every operator it meets is one it must not move under and a plan that moved
// under one of them still comes back with the right number of rows.
func TestLazySliceOverTheRestOfThePlan(t *testing.T) {
	f := trades(t)
	dear := kuma.F64("price").Gt(150)

	tests := []struct {
		name  string
		lazy  func() *kuma.LazyFrame[kuma.Dynamic]
		eager func() (*kuma.Frame[kuma.Dynamic], error)
	}{
		{
			name: "over a scan, which is where it ends up",
			lazy: func() *kuma.LazyFrame[kuma.Dynamic] { return f.Lazy().Slice(1, 2) },
			eager: func() (*kuma.Frame[kuma.Dynamic], error) {
				return f.Slice(1, 3), nil
			},
		},
		{
			name: "over a projection, which it goes under",
			lazy: func() *kuma.LazyFrame[kuma.Dynamic] {
				return f.Lazy().Select("qty", "symbol").Head(2)
			},
			eager: func() (*kuma.Frame[kuma.Dynamic], error) {
				some, err := f.Select("qty", "symbol")
				if err != nil {
					return nil, err
				}
				return some.Slice(0, 2), nil
			},
		},
		{
			name: "over a filter, which it stays above",
			lazy: func() *kuma.LazyFrame[kuma.Dynamic] { return f.Lazy().Filter(dear).Head(2) },
			eager: func() (*kuma.Frame[kuma.Dynamic], error) {
				kept, err := f.Filter(dear)
				if err != nil {
					return nil, err
				}
				return kept.Slice(0, 2), nil
			},
		},
		{
			name: "over a sort, which it stays above",
			lazy: func() *kuma.LazyFrame[kuma.Dynamic] { return f.Lazy().Sort(kuma.Asc("qty")).Head(2) },
			eager: func() (*kuma.Frame[kuma.Dynamic], error) {
				sorted, err := f.Sort(kuma.Asc("qty"))
				if err != nil {
					return nil, err
				}
				return sorted.Slice(0, 2), nil
			},
		},
		{
			name: "over a distinct, which it stays above",
			lazy: func() *kuma.LazyFrame[kuma.Dynamic] { return f.Lazy().Distinct("symbol").Head(1) },
			eager: func() (*kuma.Frame[kuma.Dynamic], error) {
				one, err := f.Distinct("symbol")
				if err != nil {
					return nil, err
				}
				return one.Slice(0, 1), nil
			},
		},
		{
			name: "two of them, which become one",
			lazy: func() *kuma.LazyFrame[kuma.Dynamic] { return f.Lazy().Slice(1, 3).Slice(1, 1) },
			eager: func() (*kuma.Frame[kuma.Dynamic], error) {
				return f.Slice(2, 3), nil
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lazy, err := tt.lazy().Collect(t.Context())
			if err != nil {
				t.Fatalf("Collect: %v", err)
			}
			eager, err := tt.eager()
			if err != nil {
				t.Fatalf("the eager query: %v", err)
			}
			if lazy.String() != eager.String() {
				t.Errorf("the lazy query gave\n%s\nand the eager one gave\n%s", lazy, eager)
			}
		})
	}
}

// TestLazyCrossJoin is the seventh, which takes no keys and is the one a
// forgotten key must not fall into.
func TestLazyCrossJoin(t *testing.T) {
	left := trades(t)
	right, err := kuma.NewFrame(kuma.NewSeries("venue", "NYSE", "NASDAQ").Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	out, err := left.Lazy().CrossJoin(right.Lazy()).Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if out.NumRows() != left.NumRows()*right.NumRows() {
		t.Fatalf("%s, want every pair of the four and the two", out)
	}

	eager, err := left.CrossJoin(right)
	if err != nil {
		t.Fatalf("CrossJoin: %v", err)
	}
	if out.String() != eager.String() {
		t.Errorf("the lazy cross join gave\n%s\nand the eager one gave\n%s", out, eager)
	}
}

// TestLazyJoinOnDifferentNames is the case most real data is in, where the two
// sides call the same thing by two names and both columns are kept.
func TestLazyJoinOnDifferentNames(t *testing.T) {
	left := trades(t)

	renamed, err := kuma.NewFrame(
		kuma.NewSeries("ticker", "MSFT", "AAPL").Column(),
		kuma.NewSeries("sector", "software", "hardware").Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	out, err := left.Lazy().
		Join(renamed.Lazy(), []kuma.On{{Left: "symbol", Right: "ticker"}}, kuma.InnerJoin).
		Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if got := out.Names(); !slices.Equal(got, []string{"symbol", "price", "qty", "ticker", "sector"}) {
		t.Errorf("Names() = %v, want both key columns kept", got)
	}
}

// TestLazyJoinRunsTheOtherQueryFirst is what the lazy join is for. The right
// side is a query rather than a frame, so the rows it contributes are the ones
// that survived it.
func TestLazyJoinRunsTheOtherQueryFirst(t *testing.T) {
	left, right := trades(t), sectors(t)

	out, err := left.Lazy().
		InnerJoin(right.Lazy().Filter(kuma.Str("sector").Eq("software")), "symbol").
		Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if got := symbolsOf(t, out); !slices.Equal(got, []string{"MSFT"}) {
		t.Errorf("symbols = %v, want MSFT, the only one left on the right", got)
	}
}

// TestLazyJoinString is the tree of a query with two inputs, which is the first
// one where the printed plan has a branch in it.
func TestLazyJoinString(t *testing.T) {
	q := trades(t).Lazy().LeftJoin(sectors(t).Lazy(), "symbol").Head(2)

	want := strings.Join([]string{
		"Limit 2",
		"  Join left on symbol",
		"    Scan frame",
		"    Scan frame",
	}, "\n")
	if got := q.String(); got != want {
		t.Errorf("String() =\n%s\nwant\n%s", got, want)
	}
}

func TestLazyJoinMistakes(t *testing.T) {
	f := trades(t)

	tests := []struct {
		name string
		q    *kuma.LazyFrame[kuma.Dynamic]
		want error
	}{
		{
			// A nil right side has no schema for the compiler to work the
			// type parameter out from, so this is the one call that has to
			// write it. It is also the one call nobody makes on purpose.
			name: "no query to join to",
			q:    f.Lazy().InnerJoin[kuma.Dynamic](nil, "symbol"),
			want: kuma.ErrNoValues,
		},
		{
			name: "a key that is not there on the left",
			q:    f.Lazy().InnerJoin(sectors(t).Lazy(), "ticker"),
			want: kuma.ErrNoColumn,
		},
		{
			name: "a key that is not there on the right",
			q:    f.Lazy().Join(sectors(t).Lazy(), []kuma.On{{Left: "symbol", Right: "ticker"}}, kuma.InnerJoin),
			want: kuma.ErrNoColumn,
		},
		{
			name: "a column that both sides have",
			q:    f.Lazy().InnerJoin(f.Lazy(), "symbol"),
			want: kuma.ErrDuplicateColumn,
		},
		{
			name: "a mistake written on the right",
			q:    f.Lazy().InnerJoin(sectors(t).Lazy().Drop("sctor"), "symbol"),
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

// TestLazyJoinWithoutKeys is the pair of mistakes the plan catches on the
// arguments alone: a join of a kind that needs keys and was given none, and a
// cross join that was given some.
func TestLazyJoinWithoutKeys(t *testing.T) {
	left, right := trades(t), sectors(t)

	err := left.Lazy().InnerJoin(right.Lazy()).Validate()
	if err == nil || !strings.Contains(err.Error(), "no keys") {
		t.Errorf("an inner join with no keys = %v, want it turned away", err)
	}

	err = left.Lazy().Join(right.Lazy(), kuma.Using("symbol"), kuma.CrossJoin).Validate()
	if err == nil || !strings.Contains(err.Error(), "cross join") {
		t.Errorf("a cross join with keys = %v, want it turned away", err)
	}
}

// TestLazyDistinct asks for the distinct rows both ways, since the two share a
// kernel and the point of the test is that they share a rule as well.
func TestLazyDistinct(t *testing.T) {
	f := repeats(t)

	for _, names := range [][]string{nil, {"symbol"}, {"symbol", "day"}, {"day"}} {
		t.Run(strings.Join(append([]string{"by"}, names...), " "), func(t *testing.T) {
			lazy, err := f.Lazy().Distinct(names...).Collect(t.Context())
			if err != nil {
				t.Fatalf("Collect: %v", err)
			}
			eager, err := f.Distinct(names...)
			if err != nil {
				t.Fatalf("Distinct: %v", err)
			}
			if lazy.String() != eager.String() {
				t.Errorf("the lazy distinct gave\n%s\nand the eager one gave\n%s", lazy, eager)
			}
		})
	}
}

// TestLazyDistinctIsAStepLikeAnyOther is a distinct in the middle of a query
// rather than at the end of one, which is where the rows it takes out are worth
// the most: everything after it works on fewer of them.
func TestLazyDistinctIsAStepLikeAnyOther(t *testing.T) {
	out, err := repeats(t).Lazy().
		Distinct("symbol", "day").
		Filter(kuma.Str("symbol").Eq("AAPL")).
		SortDesc("day").
		Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if want := []string{"AAPL 2 100", "AAPL 1 100"}; !slices.Equal(rows(t, out), want) {
		t.Errorf("the query gave %v, want %v", rows(t, out), want)
	}
}

// TestLazyDistinctKeepsTheSchemaType is what makes a distinct a step a typed
// query can take without giving up its handles, since it changes no columns.
func TestLazyDistinctKeepsTheSchemaType(t *testing.T) {
	out, err := typedTrades(t).Lazy().Distinct("symbol").Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if got := symbolsOf(t, out); !slices.Equal(got, []string{"AAPL", "MSFT", "NVDA"}) {
		t.Errorf("symbols = %v, want each of them once", got)
	}
}

// TestLazyDistinctSchema is the columns of the result worked out without
// reading anything, which a distinct leaves exactly as they arrived.
func TestLazyDistinctSchema(t *testing.T) {
	q := repeats(t).Lazy().Distinct("symbol")

	s, err := q.Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}
	if want := []string{"symbol", "day", "qty"}; !slices.Equal(s.Names(), want) {
		t.Errorf("Schema() = %v, want %v", s.Names(), want)
	}
}

// TestLazyDistinctString is the operator as an explain shows it, with the keys
// after a by and nothing at all when every column is compared.
func TestLazyDistinctString(t *testing.T) {
	f := repeats(t)

	want := strings.Join([]string{"Distinct by symbol, day", "  Scan frame"}, "\n")
	if got := f.Lazy().Distinct("symbol", "day").String(); got != want {
		t.Errorf("String() =\n%s\nwant\n%s", got, want)
	}

	want = strings.Join([]string{"Distinct", "  Scan frame"}, "\n")
	if got := f.Lazy().Distinct().String(); got != want {
		t.Errorf("String() =\n%s\nwant\n%s", got, want)
	}
}

// TestLazyDistinctOfNoColumns is the query that compares nothing, which is what
// selecting no columns and then asking for the distinct rows gives. A frame
// with no columns has no rows either, so the answer is the empty frame rather
// than a division by nothing.
func TestLazyDistinctOfNoColumns(t *testing.T) {
	out, err := repeats(t).Lazy().Select().Distinct().Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if out.NumRows() != 0 || out.NumCols() != 0 {
		t.Errorf("the query gave %d rows and %d columns, want none of either", out.NumRows(), out.NumCols())
	}
}

func TestLazyDistinctMistakes(t *testing.T) {
	f := repeats(t)

	tests := []struct {
		name string
		q    *kuma.LazyFrame[kuma.Dynamic]
		want error
	}{
		{
			name: "a column that is not there",
			q:    f.Lazy().Distinct("smybol"),
			want: kuma.ErrNoColumn,
		},
		{
			name: "a mistake written before it",
			q:    f.Lazy().Drop("smybol").Distinct(),
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

// BenchmarkLazyInnerJoin is BenchmarkFrameInnerJoin written as a query, over
// the same two frames and the same key, so the gap between the two is the whole
// cost of going through the plan.
func BenchmarkLazyInnerJoin(b *testing.B) {
	left, right := benchJoinSides(b)
	ctx := b.Context()

	b.ReportAllocs()
	for b.Loop() {
		out, err := left.Lazy().InnerJoin(right.Lazy(), "k").Collect(ctx)
		if err != nil {
			b.Fatalf("Collect: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkLazyJoinAs is BenchmarkFrameJoinAs written as a query, over the same
// two frames and the same struct, so the gap between the two is the whole cost
// of going through the plan.
func BenchmarkLazyJoinAs(b *testing.B) {
	left, right := benchJoinSides(b)
	ctx := b.Context()

	b.ReportAllocs()
	for b.Loop() {
		out, err := left.Lazy().
			JoinAs[benchJoined](right.Lazy(), kuma.Using("k"), kuma.InnerJoin).
			Collect(ctx)
		if err != nil {
			b.Fatalf("Collect: %v", err)
		}
		joinedSink = out
	}
}

// BenchmarkLazyDistinct is BenchmarkFrameDistinct written as a query, over the
// same frame and the same column, so the gap between the two is the whole cost
// of going through the plan.
func BenchmarkLazyDistinct(b *testing.B) {
	f := benchGrouped(b).Frame()
	ctx := b.Context()

	b.ReportAllocs()
	for b.Loop() {
		out, err := f.Lazy().Distinct("k").Collect(ctx)
		if err != nil {
			b.Fatalf("Collect: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkLazySelectAs is BenchmarkFrameSelectAs written as a query, over the
// same frame and the same struct, so the gap between the two is the whole cost
// of going through the plan. Both of them are per column rather than per row,
// so this is the one query where that cost is most of the number.
func BenchmarkLazySelectAs(b *testing.B) {
	f := benchFrame(b)
	ctx := b.Context()

	b.ReportAllocs()
	for b.Loop() {
		out, err := f.Lazy().SelectAs[benchRow]().Collect(ctx)
		if err != nil {
			b.Fatalf("Collect: %v", err)
		}
		boundSink = out
	}
}

// BenchmarkLazyAggAs is BenchmarkFrameAggAs written as a query, over the same
// frame and asking for the same three columns, so the gap between the two is the
// whole cost of going through the plan.
func BenchmarkLazyAggAs(b *testing.B) {
	f := benchGrouped(b).Frame()
	ctx := b.Context()

	b.ReportAllocs()
	for b.Loop() {
		out, err := f.Lazy().GroupBy("k").
			AggAs[benchTotal](kuma.Sum("qty").As("total"), kuma.Mean("price").As("avg")).
			Collect(ctx)
		if err != nil {
			b.Fatalf("Collect: %v", err)
		}
		totalSink = out
	}
}

// TestLazyRepeatsAValueAndGetsTheSameAnswer is the end to end check on the
// common subexpression pass. The pass is the one most able to give a wrong
// answer that looks right, since a hoisted value comes back under a name it
// made up for itself and a name that came out wrong would rename a column or
// hide one. The eager path runs no optimizer, so it is the answer to compare
// against.
func TestLazyRepeatsAValueAndGetsTheSameAnswer(t *testing.T) {
	f := trades(t)
	notional := kuma.F64("price").MulExpr(kuma.I64("qty").AsF64())

	tests := []struct {
		name string
		expr kuma.F64Expr[kuma.Dynamic]
	}{
		{
			name: "a value written twice in one expression",
			expr: notional.AddExpr(notional),
		},
		{
			name: "and three times",
			expr: notional.AddExpr(notional.MulExpr(notional)),
		},
		{
			name: "with the repeat inside a repeat",
			expr: notional.MulExpr(notional).AddExpr(notional.MulExpr(notional)),
		},
		{
			name: "and one written once, which nothing moves",
			expr: notional.Add(1),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lazy, err := f.Lazy().With("score", tt.expr).Collect(t.Context())
			if err != nil {
				t.Fatalf("Collect: %v", err)
			}
			eager, err := f.WithExpr("score", tt.expr)
			if err != nil {
				t.Fatalf("the eager query: %v", err)
			}
			if lazy.String() != eager.String() {
				t.Errorf("the lazy query gave\n%s\nand the eager one gave\n%s", lazy, eager)
			}
		})
	}
}

// TestLazyRepeatsAValueOverTheRestOfThePlan puts the repeat where the other
// passes have something to say about it, since a hoisted projection is a new
// operator in the middle of a plan the others have already moved things
// through.
func TestLazyRepeatsAValueOverTheRestOfThePlan(t *testing.T) {
	f := trades(t)
	notional := kuma.F64("price").MulExpr(kuma.I64("qty").AsF64())
	score := notional.AddExpr(notional.MulExpr(notional))
	dear := kuma.F64("price").Gt(150)

	tests := []struct {
		name  string
		lazy  func() *kuma.LazyFrame[kuma.Dynamic]
		eager func() (*kuma.Frame[kuma.Dynamic], error)
	}{
		{
			name: "under a filter that sinks past it",
			lazy: func() *kuma.LazyFrame[kuma.Dynamic] {
				return f.Lazy().With("score", score).Filter(dear)
			},
			eager: func() (*kuma.Frame[kuma.Dynamic], error) {
				with, err := f.WithExpr("score", score)
				if err != nil {
					return nil, err
				}
				return with.Filter(dear)
			},
		},
		{
			name: "under a head that sinks past it too",
			lazy: func() *kuma.LazyFrame[kuma.Dynamic] {
				return f.Lazy().With("score", score).Head(2)
			},
			eager: func() (*kuma.Frame[kuma.Dynamic], error) {
				with, err := f.WithExpr("score", score)
				if err != nil {
					return nil, err
				}
				return with.Slice(0, 2), nil
			},
		},
		{
			name: "under a select, which is what leaves the scan reading two columns",
			lazy: func() *kuma.LazyFrame[kuma.Dynamic] {
				return f.Lazy().With("score", score).Select("score")
			},
			eager: func() (*kuma.Frame[kuma.Dynamic], error) {
				with, err := f.WithExpr("score", score)
				if err != nil {
					return nil, err
				}
				return with.Select("score")
			},
		},
		{
			name: "under a sort by the value that was hoisted",
			lazy: func() *kuma.LazyFrame[kuma.Dynamic] {
				return f.Lazy().With("score", score).Sort(kuma.Asc("score"))
			},
			eager: func() (*kuma.Frame[kuma.Dynamic], error) {
				with, err := f.WithExpr("score", score)
				if err != nil {
					return nil, err
				}
				return with.Sort(kuma.Asc("score"))
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lazy, err := tt.lazy().Collect(t.Context())
			if err != nil {
				t.Fatalf("Collect: %v", err)
			}
			eager, err := tt.eager()
			if err != nil {
				t.Fatalf("the eager query: %v", err)
			}
			if lazy.String() != eager.String() {
				t.Errorf("the lazy query gave\n%s\nand the eager one gave\n%s", lazy, eager)
			}
		})
	}
}

// TestLazyWorksOutAValueTheDataDoesNotDecide is the constant folding pass from
// the outside. The lazy query is the one the optimizer shortened and the eager
// one is the query exactly as it was written, so the two agreeing is the pass
// having changed how the answer is reached and not what it is.
func TestLazyWorksOutAValueTheDataDoesNotDecide(t *testing.T) {
	f := trades(t)

	// A hundred and a tenth of it, which is a limit written the way it was
	// meant rather than as the 110 it comes to.
	limit := kuma.Lit(100.0).Mul(1.1)

	tests := []struct {
		name string
		cond kuma.BoolExpr[kuma.Dynamic]
	}{
		{
			name: "a limit written as a sum",
			cond: kuma.Dyn("price").GtExpr(limit),
		},
		{
			name: "a condition that holds whatever the data says",
			cond: kuma.F64("price").Gt(150).And(kuma.Lit(1).Lt(2)),
		},
		{
			name: "a condition that never holds, which is the case the pass leaves alone",
			cond: kuma.F64("price").Gt(150).And(kuma.Lit(1).Gt(2)),
		},
		{
			name: "a negation of a negation",
			cond: kuma.F64("price").Gt(150).Not().Not(),
		},
		{
			name: "a value inside a value",
			cond: kuma.Dyn("price").GtExpr(limit.Mul(2.0)),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lazy, err := f.Lazy().Filter(tt.cond).Collect(t.Context())
			if err != nil {
				t.Fatalf("Collect: %v", err)
			}
			eager, err := f.Filter(tt.cond)
			if err != nil {
				t.Fatalf("the eager query: %v", err)
			}
			if lazy.String() != eager.String() {
				t.Errorf("the lazy query gave\n%s\nand the eager one gave\n%s", lazy, eager)
			}
		})
	}
}

// TestLazyDropsACastAColumnDoesNotNeed is the one folding rule that is about a
// column rather than about a value written down, and the one that pays for
// itself on a big frame, since the cast it takes away would have read the
// column and written another one the same size.
func TestLazyDropsACastAColumnDoesNotNeed(t *testing.T) {
	f := trades(t)

	// The handle says the column is an int64 and the column is a float64, which
	// is what a query written against a frame whose types the caller was not
	// sure of looks like. The cast reads the whole column and writes another one
	// the same size to arrive back where it started.
	same := kuma.I64("price").AsF64()

	q := f.Lazy().With("notional", same)
	ran, err := plan.Optimize(q.Plan(), plan.Passes()...)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}
	if strings.Contains(ran.String(), "as float64") {
		t.Errorf("the cast is still in the plan that runs: %s", ran)
	}

	out, err := q.Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	eager, err := f.WithExpr("notional", same)
	if err != nil {
		t.Fatalf("the eager query: %v", err)
	}
	if out.String() != eager.String() {
		t.Errorf("the lazy query gave\n%s\nand the eager one gave\n%s", out, eager)
	}
}

// planSink keeps the plan the benchmark built from being optimized away.
var planSink *plan.Node
