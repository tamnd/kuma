package plan_test

// The tests for the predicate pushdown. Where a condition ends up is read off
// the printed plan, and what must not change is checked by running the pass
// over plans it has to leave alone and over its own output.

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// sunk moves the conditions of a plan and fails the test if the pass reports
// anything, since none of the plans here are wrong.
func sunk(t *testing.T, n *plan.Node) *plan.Node {
	t.Helper()
	out, err := plan.PushPredicate(n)
	if err != nil {
		t.Fatalf("PushPredicate: %v", err)
	}
	return out
}

// The conditions these tests move. They are built once because two
// expressions that say the same thing are the same expression, so a test that
// checks where one ended up is checking the one it wrote.
var (
	dear  = plan.Compare(kernel.OpGt, price, plan.Lit(100.0))
	big   = plan.Compare(kernel.OpGt, qty, plan.Lit(int64(10)))
	named = plan.Compare(kernel.OpEq, symbol, plan.Lit("ES"))
	wide  = plan.Compare(kernel.OpLt, bid, plan.Lit(1.0))
)

func TestPushPredicateSinksAFilter(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
		want string
	}{
		{
			name: "a filter over a scan is where it already was",
			plan: plan.Filter(scan, dear),
			want: "Filter (price > 100)\n  Scan trades/*.parquet",
		},
		{
			name: "a filter sinks under a sort, which then has less to order",
			plan: plan.Filter(plan.Sort(scan, []plan.SortKey{{Expr: qty}}), dear),
			want: "Sort by qty\n  Filter (price > 100)\n    Scan trades/*.parquet",
		},
		{
			name: "a filter stays above a limit",
			plan: plan.Filter(plan.Limit(scan, 0, 10), dear),
			want: "Filter (price > 100)\n  Limit 10\n    Scan trades/*.parquet",
		},
		{
			name: "a filter sinks under a projection that passes the column through",
			plan: plan.Filter(plan.Project(scan, []plan.Projection{{Expr: price}, {Expr: qty}}), dear),
			want: "Project price, qty\n  Filter (price > 100)\n    Scan trades/*.parquet",
		},
		{
			name: "a filter on a renamed column is rewritten to the name underneath",
			plan: plan.Filter(plan.Project(quoted, []plan.Projection{{Expr: bid, As: "best"}}),
				plan.Compare(kernel.OpLt, plan.Col("best"), plan.Lit(1.0))),
			want: "Project bid as best\n  Filter (bid < 1)\n    Scan quotes/*.parquet",
		},
		{
			name: "a filter on a worked out column stays above it",
			plan: plan.Filter(plan.Project(scan, []plan.Projection{{Expr: amount, As: "amount"}}),
				plan.Compare(kernel.OpGt, plan.Col("amount"), plan.Lit(1.0))),
			want: "Filter (amount > 1)\n  Project (price * (qty as float64)) as amount\n    Scan trades/*.parquet",
		},
		{
			name: "a filter on a group key sinks under the aggregation",
			plan: plan.Filter(plan.Aggregate(scan, []*plan.Expr{symbol},
				[]plan.Agg{{Func: plan.AggSum, Expr: qty, As: "volume"}}), named),
			want: "Aggregate by symbol: Sum(qty) as volume\n  Filter (symbol == \"ES\")\n    Scan trades/*.parquet",
		},
		{
			name: "a filter on what was worked out per group stays above it",
			plan: plan.Filter(plan.Aggregate(scan, []*plan.Expr{symbol},
				[]plan.Agg{{Func: plan.AggSum, Expr: qty, As: "volume"}}),
				plan.Compare(kernel.OpGt, plan.Col("volume"), plan.Lit(int64(1)))),
			want: "Filter (volume > 1)\n  Aggregate by symbol: Sum(qty) as volume\n    Scan trades/*.parquet",
		},
		{
			name: "a filter sinks under a distinct over whole rows",
			plan: plan.Filter(plan.Distinct(scan, nil), dear),
			want: "Distinct\n  Filter (price > 100)\n    Scan trades/*.parquet",
		},
		{
			name: "a filter on the column a distinct is taken by sinks under it",
			plan: plan.Filter(plan.Distinct(scan, []*plan.Expr{symbol}), named),
			want: "Distinct by symbol\n  Filter (symbol == \"ES\")\n    Scan trades/*.parquet",
		},
		{
			name: "a filter on any other column stays above a distinct by names",
			plan: plan.Filter(plan.Distinct(scan, []*plan.Expr{symbol}), dear),
			want: "Filter (price > 100)\n  Distinct by symbol\n    Scan trades/*.parquet",
		},
		{
			name: "two filters that meet become one",
			plan: plan.Filter(plan.Filter(scan, dear), big),
			want: "Filter ((price > 100) and (qty > 10))\n  Scan trades/*.parquet",
		},
		{
			name: "a condition is taken apart and only the part that is stopped stays",
			plan: plan.Filter(plan.Limit(plan.Sort(scan, []plan.SortKey{{Expr: qty}}), 0, 10),
				plan.And(dear, big)),
			want: "Filter ((price > 100) and (qty > 10))\n  Limit 10\n    Sort by qty\n      Scan trades/*.parquet",
		},
		{
			name: "a filter sinks past everything that will have it",
			plan: plan.Filter(plan.Sort(plan.Distinct(plan.Project(scan,
				[]plan.Projection{{Expr: symbol}, {Expr: price}}), nil),
				[]plan.SortKey{{Expr: price}}), dear),
			want: "Sort by price\n  Distinct\n    Project symbol, price" +
				"\n      Filter (price > 100)\n        Scan trades/*.parquet",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := tree(sunk(t, c.plan)); got != c.want {
				t.Errorf("the pass gave\n%s\nwant\n%s", got, c.want)
			}
		})
	}
}

func TestPushPredicateSplitsAcrossAJoin(t *testing.T) {
	both := plan.Compare(kernel.OpGt, price, bid)

	cases := []struct {
		name string
		plan *plan.Node
		want string
	}{
		{
			name: "each side gets the conditions that read it",
			plan: plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin), plan.And(dear, wide)),
			want: "Join inner on symbol\n  Filter (price > 100)\n    Scan trades/*.parquet" +
				"\n  Filter (bid < 1)\n    Scan quotes/*.parquet",
		},
		{
			name: "a condition that reads both sides stays above the join",
			plan: plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin), both),
			want: "Filter (price > bid)\n  Join inner on symbol" +
				"\n    Scan trades/*.parquet\n    Scan quotes/*.parquet",
		},
		{
			name: "a left join takes the left side and keeps the right one above",
			plan: plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.LeftJoin), plan.And(dear, wide)),
			want: "Filter (bid < 1)\n  Join left on symbol\n    Filter (price > 100)" +
				"\n      Scan trades/*.parquet\n    Scan quotes/*.parquet",
		},
		{
			name: "a right join takes the right side and keeps the left one above",
			plan: plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.RightJoin), plan.And(dear, wide)),
			want: "Filter (price > 100)\n  Join right on symbol\n    Scan trades/*.parquet" +
				"\n    Filter (bid < 1)\n      Scan quotes/*.parquet",
		},
		{
			name: "an outer join takes neither side",
			plan: plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.OuterJoin), dear),
			want: "Filter (price > 100)\n  Join outer on symbol" +
				"\n    Scan trades/*.parquet\n    Scan quotes/*.parquet",
		},
		{
			name: "a semi join answers about its left side, so a condition goes there",
			plan: plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.SemiJoin), dear),
			want: "Join semi on symbol\n  Filter (price > 100)\n    Scan trades/*.parquet" +
				"\n  Scan quotes/*.parquet",
		},
		{
			name: "an anti join is the same",
			plan: plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.AntiJoin), dear),
			want: "Join anti on symbol\n  Filter (price > 100)\n    Scan trades/*.parquet" +
				"\n  Scan quotes/*.parquet",
		},
		{
			name: "a cross join takes both sides, which is what makes it worth writing",
			plan: plan.Filter(plan.Join(scan, quoted, nil, kernel.CrossJoin), plan.And(dear, wide)),
			want: "Join cross\n  Filter (price > 100)\n    Scan trades/*.parquet" +
				"\n  Filter (bid < 1)\n    Scan quotes/*.parquet",
		},
		{
			name: "a condition on a key both sides call the same thing goes to the left",
			plan: plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin), named),
			want: "Join inner on symbol\n  Filter (symbol == \"ES\")\n    Scan trades/*.parquet" +
				"\n  Scan quotes/*.parquet",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := tree(sunk(t, c.plan)); got != c.want {
				t.Errorf("the pass gave\n%s\nwant\n%s", got, c.want)
			}
		})
	}
}

// TestPushPredicateAndAnExplode is the operator that changes what a column
// holds, so a condition on the column it takes apart cannot go under it and
// one on any other column can.
func TestPushPredicateAndAnExplode(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
		want string
	}{
		{
			name: "a condition on the column being taken apart stays above",
			plan: plan.Filter(plan.Explode(listed, []string{"sizes"}),
				plan.Compare(kernel.OpGt, plan.Col("sizes"), plan.Lit(int64(10)))),
			want: "Filter (sizes > 10)\n  Explode sizes\n    Scan orders/*.parquet",
		},
		{
			name: "a condition on any other column sinks under it",
			plan: plan.Filter(plan.Explode(listed, []string{"sizes"}), named),
			want: "Explode sizes\n  Filter (symbol == \"ES\")\n    Scan orders/*.parquet",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := tree(sunk(t, c.plan)); got != c.want {
				t.Errorf("the pass gave\n%s\nwant\n%s", got, c.want)
			}
		})
	}
}

// TestPushPredicateLeavesAPlanItFoundNothingIn is the contract every pass is
// written to, and the reason the optimizer can run them to fixpoint.
func TestPushPredicateLeavesAPlanItFoundNothingIn(t *testing.T) {
	plans := []*plan.Node{
		scan,
		plan.Filter(scan, dear),
		plan.Filter(scan, plan.And(dear, big)),
		plan.Filter(plan.Limit(scan, 0, 10), dear),
		plan.Project(scan, []plan.Projection{{Expr: symbol}}),
		plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.OuterJoin), dear),
	}

	for _, q := range plans {
		if got := sunk(t, q); got != q {
			t.Errorf("%s came back as a different plan with nothing to move", q)
		}
	}
}

// TestPushPredicateSettles runs the pass over its own output, which is what the
// optimizer does on the round after the one that found something.
func TestPushPredicateSettles(t *testing.T) {
	plans := []*plan.Node{
		plan.Filter(plan.Sort(scan, []plan.SortKey{{Expr: qty}}), dear),
		plan.Filter(plan.Filter(scan, dear), big),
		plan.Filter(scan, plan.And(dear, plan.And(big, named))),
		plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin), plan.And(dear, wide)),
		plan.Filter(plan.Project(quoted, []plan.Projection{{Expr: bid, As: "best"}}),
			plan.Compare(kernel.OpLt, plan.Col("best"), plan.Lit(1.0))),
	}

	for _, q := range plans {
		once := sunk(t, q)
		if twice := sunk(t, once); twice != once {
			t.Errorf("%s was moved twice, and the second time found something", q)
		}
	}
}

// TestPushPredicateKeepsWhatThePlanProduces is what makes the pass safe to run
// without being asked. Moving a filter must not change a column or its type,
// and the schema is what says so.
func TestPushPredicateKeepsWhatThePlanProduces(t *testing.T) {
	plans := []*plan.Node{
		plan.Filter(plan.Sort(scan, []plan.SortKey{{Expr: qty}}), dear),
		plan.Filter(plan.Project(scan, []plan.Projection{{Expr: price}, {Expr: qty}}), dear),
		plan.Filter(plan.Aggregate(scan, []*plan.Expr{symbol},
			[]plan.Agg{{Func: plan.AggSum, Expr: qty, As: "volume"}}), named),
		plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.LeftJoin), plan.And(dear, wide)),
		plan.Filter(plan.Distinct(scan, nil), dear),
	}

	for _, q := range plans {
		want, err := q.Schema()
		if err != nil {
			t.Fatalf("Schema: %v", err)
		}
		got, err := sunk(t, q).Schema()
		if err != nil {
			t.Fatalf("Schema after the pass: %v", err)
		}
		if !got.Equal(want) {
			t.Errorf("%s produces %v after the pass, want %v", q, got.Names(), want.Names())
		}
	}
}

// TestPushPredicateOfAPlanThatIsWrong is the pass meeting a query that was
// never going to run. Working out which side of a join a condition belongs to
// means asking both sides what they hold.
func TestPushPredicateOfAPlanThatIsWrong(t *testing.T) {
	q := plan.Filter(plan.Join(plan.Scan(missing{}), quoted, onSymbol, kernel.InnerJoin), dear)

	_, err := plan.PushPredicate(q)
	if err == nil || !strings.Contains(err.Error(), "no such file") {
		t.Fatalf("PushPredicate = %v, want what the source said", err)
	}
}

// TestTheTwoPushdownsTogether is what the optimizer runs. The filter sinks and
// then the scan under it is narrowed to the columns that are left, which is one
// pass giving the other something to find.
func TestTheTwoPushdownsTogether(t *testing.T) {
	q := plan.Project(plan.Sort(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin),
		[]plan.SortKey{{Expr: price}}), []plan.Projection{{Expr: symbol}, {Expr: price}})
	q = plan.Filter(q, dear)

	out, err := plan.Optimize(q, plan.Passes()...)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	want := "Project symbol, price\n  Sort by price\n    Join inner on symbol" +
		"\n      Filter (price > 100)\n        Scan trades/*.parquet [symbol, price]" +
		"\n      Scan quotes/*.parquet [symbol]"
	if got := tree(out); got != want {
		t.Errorf("the optimizer gave\n%s\nwant\n%s", got, want)
	}
}

// BenchmarkPushPredicate is the pass on its own over a query with a join in it,
// which is the shape that costs the most to walk.
func BenchmarkPushPredicate(b *testing.B) {
	q := plan.Filter(plan.Sort(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin),
		[]plan.SortKey{{Expr: price}}), plan.And(dear, wide))

	for b.Loop() {
		out, err := plan.PushPredicate(q)
		if err != nil {
			b.Fatalf("PushPredicate: %v", err)
		}
		sink = out
	}
}
