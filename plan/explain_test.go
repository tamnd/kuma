package plan_test

// The tests for printing a plan. The tree is checked against plans whose shape
// is the point, a join and a plan with nothing in it. The explain is checked
// against its documented format, since the format is the promise: a caller
// reads it to find out whether the query they wrote is the query that happens,
// and a change to it is a change they notice.

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

func TestTreePrintsThePlanTopDown(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
		want string
	}{
		{
			name: "one operator is one line",
			plan: scan,
			want: "Scan trades/*.parquet",
		},
		{
			name: "an input is indented under the operator that reads it",
			plan: plan.Filter(scan, dear),
			want: "Filter (price > 100)\n" +
				"  Scan trades/*.parquet",
		},
		{
			name: "both sides of a join are under it, the left one first",
			plan: plan.Join(plan.Filter(scan, dear), quoted, onSymbol, kernel.InnerJoin),
			want: "Join inner on symbol\n" +
				"  Filter (price > 100)\n" +
				"    Scan trades/*.parquet\n" +
				"  Scan quotes/*.parquet",
		},
		{
			name: "a plan with nothing in it is nothing",
			plan: nil,
			want: "",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.plan.Tree(); got != c.want {
				t.Errorf("Tree gave\n%s\nwant\n%s", got, c.want)
			}
		})
	}
}

// TestTreeDoesNotEndInANewline is what lets a caller put a tree inside
// something else, which is what the explain itself does.
func TestTreeDoesNotEndInANewline(t *testing.T) {
	got := plan.Filter(scan, dear).Tree()
	if strings.HasSuffix(got, "\n") {
		t.Errorf("Tree gave %q, which ends in a newline", got)
	}
}

// TestExplainShowsWhatTheOptimizerDid is the format itself, written out, since
// the format is the part of this that callers depend on.
func TestExplainShowsWhatTheOptimizerDid(t *testing.T) {
	q := plan.Limit(plan.Project(
		plan.Filter(plan.Sort(scan, []plan.SortKey{{Expr: qty, Descending: true}}), dear),
		[]plan.Projection{{Expr: symbol}, {Expr: price}}), 0, 20)

	want := "the query as written\n" +
		"  Limit 20\n" +
		"    Project symbol, price\n" +
		"      Filter (price > 100)\n" +
		"        Sort by qty desc\n" +
		"          Scan trades/*.parquet\n" +
		"\n" +
		"the query that runs\n" +
		"  Project symbol, price\n" +
		"    Limit 20\n" +
		"      Sort by qty desc\n" +
		"        Filter (price > 100)\n" +
		"          Scan trades/*.parquet\n" +
		"\n" +
		"changed by predicate pushdown and slice pushdown\n"

	got, err := plan.Explain(q, plan.Passes()...)
	if err != nil {
		t.Fatalf("Explain: %v", err)
	}
	if got != want {
		t.Errorf("Explain gave\n%s\nwant\n%s", got, want)
	}
}

// TestExplainOfAQueryNoPassChanges is the other shape the format has, and the
// reason it has one: a plan printed twice with a line between saying nothing
// happened is worse than saying so once.
func TestExplainOfAQueryNoPassChanges(t *testing.T) {
	want := "the query as written\n" +
		"  Filter (price > 100)\n" +
		"    Scan trades/*.parquet\n" +
		"\n" +
		"nothing the optimizer does changes it\n"

	got, err := plan.Explain(plan.Filter(scan, dear), plan.Passes()...)
	if err != nil {
		t.Fatalf("Explain: %v", err)
	}
	if got != want {
		t.Errorf("Explain gave\n%s\nwant\n%s", got, want)
	}
}

// TestExplainNamesEveryPassThatChangedSomething is the last line, which is the
// one a reader looks at first.
func TestExplainNamesEveryPassThatChangedSomething(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
		want string
	}{
		{
			name: "one pass reads as one name",
			plan: plan.Limit(scan, 0, 20),
			want: "changed by slice pushdown",
		},
		{
			name: "two read as a name and a name",
			plan: plan.Limit(plan.Project(plan.Filter(scan, dear),
				[]plan.Projection{{Expr: symbol}, {Expr: price}}), 0, 20),
			want: "changed by slice pushdown and projection pushdown",
		},
		{
			name: "three read as a list with an and at the end",
			plan: plan.Limit(plan.Project(plan.Filter(scan, dear), []plan.Projection{
				{Expr: symbol},
				{Expr: plan.Arith(kernel.OpMul, price,
					plan.Arith(kernel.OpAdd, plan.Lit(1.0), plan.Lit(0.1))), As: "up"},
			}), 0, 20),
			want: "changed by constant folding, slice pushdown and projection pushdown",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := plan.Explain(c.plan, plan.Passes()...)
			if err != nil {
				t.Fatalf("Explain: %v", err)
			}
			last := got[strings.LastIndex(strings.TrimRight(got, "\n"), "\n")+1:]
			if strings.TrimRight(last, "\n") != c.want {
				t.Errorf("the last line is %q, want %q", strings.TrimRight(last, "\n"), c.want)
			}
		})
	}
}

// TestExplainNamesAPassThatOnlyFoundSomethingOnce is the part of the contract
// that is easy to get wrong: the loop runs the passes until they all find
// nothing, so a pass is asked more times than it answers.
func TestExplainNamesAPassThatOnlyFoundSomethingOnce(t *testing.T) {
	got, err := plan.Explain(scan, wrap)
	if err != nil {
		t.Fatalf("Explain: %v", err)
	}
	if !strings.HasSuffix(got, "changed by wrap\n") {
		t.Errorf("Explain gave\n%s\nwant it to end by naming the pass that found something", got)
	}
	if strings.Count(got, "wrap") != 1 {
		t.Errorf("the pass is named %d times in\n%s\nwant once", strings.Count(got, "wrap"), got)
	}
}

// TestExplainOfAQueryThatIsWrong is why the check comes before the passes: the
// mistake reported has to be the one the caller made.
func TestExplainOfAQueryThatIsWrong(t *testing.T) {
	_, err := plan.Explain(plan.Filter(scan, plan.Compare(kernel.OpGt,
		plan.Col("volume"), plan.Lit(100.0))), plan.Passes()...)
	if err == nil {
		t.Fatal("Explain of a query over a column that is not there gave no error")
	}
	if !strings.Contains(err.Error(), "volume") {
		t.Errorf("Explain = %v, want the error to name the column", err)
	}
	for _, p := range plan.Passes() {
		if strings.Contains(err.Error(), p.Name) {
			t.Errorf("Explain = %v, want the query as written rather than the %s pass", err, p.Name)
		}
	}
}

func TestExplainOfNoPlan(t *testing.T) {
	_, err := plan.Explain(nil, plan.Passes()...)
	if err == nil || !strings.Contains(err.Error(), "no operator") {
		t.Fatalf("Explain = %v, want a plan with no operator in it", err)
	}
}

// TestExplainReportsAPassThatFails is the other error, which is a bug in a pass
// rather than in the query and says so by naming the pass.
func TestExplainReportsAPassThatFails(t *testing.T) {
	_, err := plan.Explain(scan, wrap, unwrap)
	if err == nil || !strings.Contains(err.Error(), "undoing each other") {
		t.Fatalf("Explain = %v, want the optimizer to give up", err)
	}
}

// TestExplainDoesNotChangeThePlan is the same promise Optimize makes, and it
// matters more here, since the whole point is printing the plan as written next
// to the one that runs.
func TestExplainDoesNotChangeThePlan(t *testing.T) {
	q := plan.Limit(plan.Filter(plan.Sort(scan, []plan.SortKey{{Expr: qty}}), dear), 0, 20)
	before := q.Tree()

	if _, err := plan.Explain(q, plan.Passes()...); err != nil {
		t.Fatalf("Explain: %v", err)
	}
	if after := q.Tree(); after != before {
		t.Errorf("explaining left the plan reading\n%s\nwant\n%s", after, before)
	}
}

// BenchmarkExplain is what printing a plan costs, which is a plan built once
// per call and a string built out of it, and is worth watching because an
// explain that is slower than the query is one nobody runs.
func BenchmarkExplain(b *testing.B) {
	q := plan.Limit(plan.Sort(
		plan.Aggregate(
			plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin),
				plan.Compare(kernel.OpGt, price, plan.Lit(100.0))),
			[]*plan.Expr{symbol},
			[]plan.Agg{{Func: plan.AggSum, Expr: qty, As: "volume"}}),
		[]plan.SortKey{{Expr: plan.Col("volume"), Descending: true}}), 0, 20)

	passes := plan.Passes()
	for b.Loop() {
		out, err := plan.Explain(q, passes...)
		if err != nil {
			b.Fatalf("Explain: %v", err)
		}
		textSink = out
	}
}

// BenchmarkTree is the printing on its own, without the passes, since that is
// what [github.com/tamnd/kuma.LazyFrame.String] costs.
func BenchmarkTree(b *testing.B) {
	q := plan.Limit(plan.Sort(
		plan.Aggregate(
			plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin),
				plan.Compare(kernel.OpGt, price, plan.Lit(100.0))),
			[]*plan.Expr{symbol},
			[]plan.Agg{{Func: plan.AggSum, Expr: qty, As: "volume"}}),
		[]plan.SortKey{{Expr: plan.Col("volume"), Descending: true}}), 0, 20)

	for b.Loop() {
		textSink = q.Tree()
	}
}

// textSink is where the printed plans go so that the compiler cannot decide the
// printing was not needed.
var textSink string
