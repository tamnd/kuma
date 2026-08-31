package plan_test

import (
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// trades is the source these tests scan, since a plan has to start somewhere
// and what it starts from is not what is being tested here.
type trades struct{}

func (trades) Name() string { return "trades/*.parquet" }

func (trades) Schema() (dtype.Schema, error) {
	return dtype.Schema{Fields: []dtype.Field{
		{Name: "symbol", Type: dtype.String},
		{Name: "price", Type: dtype.Float64},
		{Name: "qty", Type: dtype.Int64},
	}}, nil
}

var (
	symbol = plan.Col("symbol")
	price  = plan.Col("price")
	qty    = plan.Col("qty")
)

func TestOperatorsPrintAsTheyWereWritten(t *testing.T) {
	scan := plan.Scan(trades{})
	notional := plan.Arith(kernel.OpMul, price, qty)

	cases := []struct {
		node *plan.Node
		want string
	}{
		{scan, "Scan trades/*.parquet"},
		{plan.Filter(scan, plan.Compare(kernel.OpGt, price, plan.Lit(100.0))), "Filter (price > 100)"},
		{
			plan.Project(scan, []plan.Projection{{Expr: symbol}, {Expr: notional, As: "notional"}}),
			"Project symbol, (price * qty) as notional",
		},
		{
			plan.Aggregate(scan, []*plan.Expr{symbol}, []plan.Agg{
				{Func: plan.AggSum, Expr: notional, As: "volume"},
				{Func: plan.AggSize},
			}),
			"Aggregate by symbol: Sum((price * qty)) as volume, Size()",
		},
		{
			plan.Aggregate(scan, nil, []plan.Agg{{Func: plan.AggMean, Expr: price}}),
			"Aggregate: Mean(price)",
		},
		{
			plan.Join(scan, scan, []plan.JoinKey{{Left: symbol, Right: plan.Col("sym")}}, kernel.LeftJoin),
			"Join left on symbol = sym",
		},
		{plan.Join(scan, scan, nil, kernel.CrossJoin), "Join cross"},
		{
			plan.Sort(scan, []plan.SortKey{{Expr: price, Descending: true}, {Expr: symbol, NullsFirst: true}}),
			"Sort by price desc, symbol nulls first",
		},
		{plan.Limit(scan, 0, 20), "Limit 20"},
		{plan.Limit(scan, 5, 20), "Limit 20 offset 5"},
		{plan.Distinct(scan, nil), "Distinct"},
		{plan.Distinct(scan, []*plan.Expr{symbol}), "Distinct by symbol"},
		{plan.Explode(scan, []string{"tags"}), "Explode tags"},
		{plan.Explode(scan, []string{"tags", "sizes"}), "Explode tags, sizes"},
	}

	for _, c := range cases {
		if got := c.node.String(); got != c.want {
			t.Errorf("the plan reads %q, want %q", got, c.want)
		}
	}
}

// TestAJoinKeyOfOneColumnPrintsOnce is the case where the two sides are called
// the same thing, which is most joins and would read badly written out twice.
func TestAJoinKeyOfOneColumnPrintsOnce(t *testing.T) {
	scan := plan.Scan(trades{})
	n := plan.Join(scan, scan, []plan.JoinKey{{Left: symbol, Right: symbol}}, kernel.InnerJoin)

	if got, want := n.String(), "Join inner on symbol"; got != want {
		t.Errorf("the join reads %q, want %q", got, want)
	}
}

func TestTheInputsAreWhatTheOperatorWasBuiltOver(t *testing.T) {
	scan := plan.Scan(trades{})
	right := plan.Scan(trades{})
	filter := plan.Filter(scan, plan.Compare(kernel.OpGt, price, plan.Lit(100.0)))
	join := plan.Join(filter, right, nil, kernel.CrossJoin)

	if scan.Input() != nil || scan.Right() != nil {
		t.Error("a scan reads something, and it is a leaf")
	}
	if filter.Input() != scan {
		t.Error("the filter does not read the scan it was built over")
	}
	if filter.Right() != nil {
		t.Error("the filter has a second input, and it takes one")
	}
	if join.Input() != filter || join.Right() != right {
		t.Error("the join reads something other than the two plans it was built over")
	}
}

// TestBuildingAPlanCopiesWhatItWasGiven is the promise that a plan does not
// change after it is built. Everything the passes do depends on it, since a
// pass keeps the tree it started from and builds the difference.
func TestBuildingAPlanCopiesWhatItWasGiven(t *testing.T) {
	scan := plan.Scan(trades{})

	cols := []plan.Projection{{Expr: symbol}, {Expr: price}}
	project := plan.Project(scan, cols)
	cols[1] = plan.Projection{Expr: qty}

	if got, want := project.String(), "Project symbol, price"; got != want {
		t.Errorf("after writing to the slice the projection reads %q, want %q", got, want)
	}

	keys := []plan.SortKey{{Expr: price}}
	sort := plan.Sort(scan, keys)
	keys[0].Descending = true

	if got, want := sort.String(), "Sort by price"; got != want {
		t.Errorf("after writing to the slice the sort reads %q, want %q", got, want)
	}

	names := []string{"tags"}
	explode := plan.Explode(scan, names)
	names[0] = "sizes"

	if got, want := explode.String(), "Explode tags"; got != want {
		t.Errorf("after writing to the slice the explode reads %q, want %q", got, want)
	}
}

// TestAnOperatorOnlyAnswersForItself checks that a field belonging to another
// operator reads as nothing rather than as a stale value, since a pass switches
// on the operator and asks for what that one has.
func TestAnOperatorOnlyAnswersForItself(t *testing.T) {
	scan := plan.Scan(trades{})
	n := plan.Filter(scan, plan.Compare(kernel.OpGt, price, plan.Lit(100.0)))

	if n.Op() != plan.OpFilter {
		t.Errorf("the filter says it is a %s", n.Op())
	}
	if n.Source() != nil {
		t.Error("the filter has a source, and only a scan does")
	}
	if n.Columns() != nil || n.By() != nil || n.Aggs() != nil {
		t.Error("the filter has the parts of a projection or an aggregate")
	}
	if n.SortKeys() != nil || n.JoinKeys() != nil {
		t.Error("the filter has the parts of a sort or a join")
	}
	if n.Offset() != 0 || n.Limit() != 0 {
		t.Error("the filter has the bounds of a limit")
	}
	if n.ExplodeNames() != nil {
		t.Error("the filter has the columns of an explode")
	}
	if scan.Source().Name() != "trades/*.parquet" {
		t.Error("the scan reads a source other than the one it was built over")
	}
	if scan.Cond() != nil {
		t.Error("the scan has a condition, and only a filter does")
	}
}

// BenchmarkBuildPlan is what it costs to say what a query wants, before any
// data has been read or any pass has run. The expressions are built outside the
// loop, the way a program that keeps its column handles around builds them, so
// what is measured here is the plan and not the table in intern.go.
func BenchmarkBuildPlan(b *testing.B) {
	src := trades{}
	cond := plan.Compare(kernel.OpGt, price, plan.Lit(100.0))
	by := []*plan.Expr{symbol}
	aggs := []plan.Agg{{Func: plan.AggSum, Expr: plan.Arith(kernel.OpMul, price, qty), As: "volume"}}
	keys := []plan.SortKey{{Expr: plan.Col("volume"), Descending: true}}

	for b.Loop() {
		sink = plan.Limit(plan.Sort(plan.Aggregate(plan.Filter(plan.Scan(src), cond), by, aggs), keys), 0, 20)
	}
}

// sink keeps the benchmark above from being optimized away.
var sink *plan.Node

func TestTheOperatorsAreNamed(t *testing.T) {
	ops := []plan.Op{
		plan.OpScan, plan.OpFilter, plan.OpProject, plan.OpAggregate,
		plan.OpJoin, plan.OpSort, plan.OpLimit, plan.OpDistinct, plan.OpExplode,
	}
	want := []string{"Scan", "Filter", "Project", "Aggregate", "Join", "Sort", "Limit", "Distinct", "Explode"}

	for i, op := range ops {
		if got := op.String(); got != want[i] {
			t.Errorf("operator %d is called %q, want %q", i, got, want[i])
		}
	}
}
