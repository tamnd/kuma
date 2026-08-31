package plan_test

// The tests for the projection pushdown. What the pass produces is checked by
// printing the plan, since the whole point of it is a scan that says which
// columns it reads, and what it must not change is checked by comparing the
// schema of the plan before and after.

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// tree is the plan as text, which is what these tests compare. A pass is
// easiest to read a failure of when the whole plan is in front of you.
func tree(n *plan.Node) string {
	return n.Tree()
}

// pushed optimizes a plan with the one pass under test and fails the test if it
// reports anything, since none of the plans here are wrong.
func pushed(t *testing.T, n *plan.Node) *plan.Node {
	t.Helper()
	out, err := plan.PushProjection(n)
	if err != nil {
		t.Fatalf("PushProjection: %v", err)
	}
	return out
}

func TestPushProjectionNarrowsTheScan(t *testing.T) {
	over100 := plan.Compare(kernel.OpGt, price, plan.Lit(100.0))
	volume := []plan.Agg{{Func: plan.AggSum, Expr: qty, As: "volume"}}

	cases := []struct {
		name string
		plan *plan.Node
		want string
	}{
		{
			name: "a projection reads the columns of its expressions",
			plan: plan.Project(scan, []plan.Projection{{Expr: symbol}}),
			want: "Project symbol\n  Scan trades/*.parquet [symbol]",
		},
		{
			name: "a projection of a worked out column reads what it is worked out from",
			plan: plan.Project(scan, []plan.Projection{{Expr: amount, As: "amount"}}),
			want: "Project (price * (qty as float64)) as amount\n  Scan trades/*.parquet [price, qty]",
		},
		{
			name: "a filter under a projection adds the column it reads",
			plan: plan.Project(plan.Filter(scan, over100), []plan.Projection{{Expr: symbol}}),
			want: "Project symbol\n  Filter (price > 100)\n    Scan trades/*.parquet [symbol, price]",
		},
		{
			name: "a filter on its own is asked for every column above it",
			plan: plan.Filter(scan, over100),
			want: "Filter (price > 100)\n  Scan trades/*.parquet",
		},
		{
			name: "an aggregate reads its keys and what it aggregates",
			plan: plan.Aggregate(scan, []*plan.Expr{symbol}, volume),
			want: "Aggregate by symbol: Sum(qty) as volume\n  Scan trades/*.parquet [symbol, qty]",
		},
		{
			name: "a size counts rows and reads nothing, so one column is kept to count",
			plan: plan.Aggregate(scan, nil, []plan.Agg{{Func: plan.AggSize}}),
			want: "Aggregate: Size()\n  Scan trades/*.parquet [symbol]",
		},
		{
			name: "a sort under a projection adds the column it orders by",
			plan: plan.Project(plan.Sort(scan, []plan.SortKey{{Expr: qty}}), []plan.Projection{{Expr: symbol}}),
			want: "Project symbol\n  Sort by qty\n    Scan trades/*.parquet [symbol, qty]",
		},
		{
			name: "a limit adds nothing of its own",
			plan: plan.Project(plan.Limit(scan, 0, 10), []plan.Projection{{Expr: price}}),
			want: "Project price\n  Limit 10\n    Scan trades/*.parquet [price]",
		},
		{
			name: "a distinct by named columns adds them",
			plan: plan.Project(plan.Distinct(scan, []*plan.Expr{symbol}), []plan.Projection{{Expr: price}}),
			want: "Project price\n  Distinct by symbol\n    Scan trades/*.parquet [symbol, price]",
		},
		{
			name: "a distinct over whole rows compares every column and keeps every column",
			plan: plan.Project(plan.Distinct(scan, nil), []plan.Projection{{Expr: price}}),
			want: "Project price\n  Distinct\n    Scan trades/*.parquet",
		},
		{
			name: "the columns come out in the order the source has them",
			plan: plan.Project(scan, []plan.Projection{{Expr: qty}, {Expr: symbol}}),
			want: "Project qty, symbol\n  Scan trades/*.parquet [symbol, qty]",
		},
		{
			name: "a projection of everything leaves the scan alone",
			plan: plan.Project(scan, []plan.Projection{{Expr: symbol}, {Expr: price}, {Expr: qty}}),
			want: "Project symbol, price, qty\n  Scan trades/*.parquet",
		},
		{
			name: "a projection of nothing but a literal keeps a column to count the rows with",
			plan: plan.Project(scan, []plan.Projection{{Expr: plan.Lit(int64(1)), As: "one"}}),
			want: "Project 1 as one\n  Scan trades/*.parquet [symbol]",
		},
		{
			name: "the whole of a query is narrowed at once",
			plan: plan.Limit(plan.Sort(plan.Aggregate(plan.Filter(scan, over100), []*plan.Expr{symbol},
				[]plan.Agg{{Func: plan.AggSum, Expr: price, As: "volume"}}),
				[]plan.SortKey{{Expr: plan.Col("volume"), Descending: true}}), 0, 20),
			want: "Limit 20\n  Sort by volume desc\n    Aggregate by symbol: Sum(price) as volume" +
				"\n      Filter (price > 100)\n        Scan trades/*.parquet [symbol, price]",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := tree(pushed(t, c.plan)); got != c.want {
				t.Errorf("the pass gave\n%s\nwant\n%s", got, c.want)
			}
		})
	}
}

func TestPushProjectionSplitsAJoin(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
		want string
	}{
		{
			name: "each side is read for what is wanted of it and for its key",
			plan: plan.Project(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin),
				[]plan.Projection{{Expr: bid}, {Expr: qty}}),
			want: "Project bid, qty\n  Join inner on symbol" +
				"\n    Scan trades/*.parquet [symbol, qty]\n    Scan quotes/*.parquet",
		},
		{
			name: "a side nothing is wanted of is read for its key alone",
			plan: plan.Project(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin),
				[]plan.Projection{{Expr: price}}),
			want: "Project price\n  Join inner on symbol" +
				"\n    Scan trades/*.parquet [symbol, price]\n    Scan quotes/*.parquet [symbol]",
		},
		{
			name: "a semi join keeps no column of the right side, so it is read for its key",
			plan: plan.Project(plan.Join(scan, quoted, onSymbol, kernel.SemiJoin),
				[]plan.Projection{{Expr: symbol}}),
			want: "Project symbol\n  Join semi on symbol" +
				"\n    Scan trades/*.parquet [symbol]\n    Scan quotes/*.parquet [symbol]",
		},
		{
			name: "a cross join has no keys and is read for what is wanted of it",
			plan: plan.Project(plan.Join(scan, quoted, nil, kernel.CrossJoin),
				[]plan.Projection{{Expr: bid}}),
			want: "Project bid\n  Join cross" +
				"\n    Scan trades/*.parquet [symbol]\n    Scan quotes/*.parquet [bid]",
		},
		{
			name: "a join nothing is above is read in full",
			plan: plan.Join(scan, quoted, onSymbol, kernel.InnerJoin),
			want: "Join inner on symbol\n  Scan trades/*.parquet\n  Scan quotes/*.parquet",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := tree(pushed(t, c.plan)); got != c.want {
				t.Errorf("the pass gave\n%s\nwant\n%s", got, c.want)
			}
		})
	}
}

// TestPushProjectionKeepsTheColumnsAnExplodeTakesApart is the operator that
// reads a column and hands it back changed, so the name has to survive the
// pass whether or not anything above reads the column afterwards.
func TestPushProjectionKeepsTheColumnsAnExplodeTakesApart(t *testing.T) {
	q := plan.Project(plan.Explode(listed, []string{"sizes"}), []plan.Projection{{Expr: plan.Col("symbol")}})

	want := "Project symbol\n  Explode sizes\n    Scan orders/*.parquet [symbol, sizes]"
	if got := tree(pushed(t, q)); got != want {
		t.Errorf("the pass gave\n%s\nwant\n%s", got, want)
	}
}

// TestPushProjectionLeavesAPlanItFoundNothingIn is the contract every pass is
// written to, and the reason the optimizer can run them to fixpoint.
func TestPushProjectionLeavesAPlanItFoundNothingIn(t *testing.T) {
	plans := []*plan.Node{
		scan,
		plan.Filter(scan, plan.Compare(kernel.OpGt, price, plan.Lit(100.0))),
		plan.Distinct(scan, nil),
		plan.Project(scan, []plan.Projection{{Expr: symbol}, {Expr: price}, {Expr: qty}}),
	}

	for _, q := range plans {
		if got := pushed(t, q); got != q {
			t.Errorf("%s came back as a different plan with nothing to narrow", q)
		}
	}
}

// TestPushProjectionSettles runs the pass over its own output, which is what
// the optimizer does on the round after the one that found something.
func TestPushProjectionSettles(t *testing.T) {
	plans := []*plan.Node{
		plan.Project(scan, []plan.Projection{{Expr: symbol}}),
		plan.Project(scan, []plan.Projection{{Expr: plan.Lit(int64(1)), As: "one"}}),
		plan.Project(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin), []plan.Projection{{Expr: bid}}),
	}

	for _, q := range plans {
		once := pushed(t, q)
		if twice := pushed(t, once); twice != once {
			t.Errorf("%s was narrowed twice, and the second time found something", q)
		}
	}
}

// TestPushProjectionKeepsWhatThePlanProduces is what makes the pass safe to run
// without being asked. A query that reads fewer columns and answers a different
// question is not an optimization.
func TestPushProjectionKeepsWhatThePlanProduces(t *testing.T) {
	plans := []*plan.Node{
		plan.Project(scan, []plan.Projection{{Expr: symbol}, {Expr: amount, As: "amount"}}),
		plan.Aggregate(scan, []*plan.Expr{symbol}, []plan.Agg{{Func: plan.AggSum, Expr: qty, As: "volume"}}),
		plan.Join(scan, quoted, onSymbol, kernel.InnerJoin),
		plan.Project(plan.Join(scan, quoted, onSymbol, kernel.LeftJoin), []plan.Projection{{Expr: bid}}),
		plan.Project(plan.Distinct(scan, []*plan.Expr{symbol}), []plan.Projection{{Expr: price}}),
		plan.Project(scan, []plan.Projection{{Expr: plan.Lit(int64(1)), As: "one"}}),
	}

	for _, q := range plans {
		want, err := q.Schema()
		if err != nil {
			t.Fatalf("Schema: %v", err)
		}
		got, err := pushed(t, q).Schema()
		if err != nil {
			t.Fatalf("Schema after the pass: %v", err)
		}
		if !got.Equal(want) {
			t.Errorf("%s produces %v after the pass, want %v", q, got.Names(), want.Names())
		}
	}
}

// TestPushProjectionOfAPlanThatIsWrong is the pass meeting a query that was
// never going to run. It reports what checking the plan reports, since working
// out which columns a step reads means asking what its input holds.
func TestPushProjectionOfAPlanThatIsWrong(t *testing.T) {
	q := plan.Project(plan.Join(plan.Scan(missing{}), quoted, onSymbol, kernel.InnerJoin),
		[]plan.Projection{{Expr: bid}})

	_, err := plan.PushProjection(q)
	if err == nil || !strings.Contains(err.Error(), "no such file") {
		t.Fatalf("PushProjection = %v, want what the source said", err)
	}
}

// BenchmarkPushProjection is the pass on its own over a query with a join in
// it, which is the shape that costs the most to walk.
func BenchmarkPushProjection(b *testing.B) {
	q := plan.Project(plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin),
		plan.Compare(kernel.OpGt, price, plan.Lit(100.0))),
		[]plan.Projection{{Expr: symbol}, {Expr: bid}})

	for b.Loop() {
		out, err := plan.PushProjection(q)
		if err != nil {
			b.Fatalf("PushProjection: %v", err)
		}
		sink = out
	}
}
