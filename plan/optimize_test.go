package plan_test

// The tests for the loop the passes run in. What each pass does is tested with
// the pass, so what is left here is the contract every pass is written to: a
// plan in and a plan out, the same pointer back when nothing was found, and an
// answer that does not depend on how many times the loop went round.

import (
	"errors"
	"strings"
	"testing"

	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// nothing is a pass that finds nothing, which is what every pass eventually
// does and the reason the loop ends.
var nothing = plan.Pass{Name: "nothing", Rewrite: func(n *plan.Node) (*plan.Node, error) {
	return n, nil
}}

// wrap puts a limit on the plan, and unwrap takes one off. Neither is a real
// pass. They are the smallest pair that undo each other, which is the bug the
// round limit is there to catch.
var wrap = plan.Pass{Name: "wrap", Rewrite: func(n *plan.Node) (*plan.Node, error) {
	if n.Op() == plan.OpLimit {
		return n, nil
	}
	return plan.Limit(n, 0, 10), nil
}}

var unwrap = plan.Pass{Name: "unwrap", Rewrite: func(n *plan.Node) (*plan.Node, error) {
	if n.Op() != plan.OpLimit {
		return n, nil
	}
	return n.Input(), nil
}}

func TestOptimizeGivesBackAPlanNoPassChanged(t *testing.T) {
	got, err := plan.Optimize(scan, nothing)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}
	if got != scan {
		t.Error("a plan no pass changed came back as a different plan")
	}
}

func TestOptimizeWithNoPasses(t *testing.T) {
	got, err := plan.Optimize(scan)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}
	if got != scan {
		t.Error("a plan came back changed from an optimizer with nothing to run")
	}
}

func TestOptimizeOfNoPlan(t *testing.T) {
	_, err := plan.Optimize(nil, plan.Passes()...)
	if err == nil || !strings.Contains(err.Error(), "no operator") {
		t.Fatalf("Optimize = %v, want a plan with no operator in it", err)
	}
}

// TestOptimizeRunsThePassesUntilTheyFindNothing is the fixpoint: one pass gives
// the next one something to find, and the loop is what lets it.
func TestOptimizeRunsThePassesUntilTheyFindNothing(t *testing.T) {
	got, err := plan.Optimize(scan, wrap)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}
	if want := "Limit 10"; got.String() != want || got.Input() != scan {
		t.Errorf("Optimize gave %q over %q, want %q over the scan", got, got.Input(), want)
	}
}

// TestOptimizeStopsWhenTwoPassesUndoEachOther is the guard that turns a hang
// into an error, since a pair of passes that disagree would otherwise be a
// query that never comes back.
func TestOptimizeStopsWhenTwoPassesUndoEachOther(t *testing.T) {
	_, err := plan.Optimize(scan, wrap, unwrap)
	if err == nil || !strings.Contains(err.Error(), "undoing each other") {
		t.Fatalf("Optimize = %v, want the optimizer to give up", err)
	}
}

// TestOptimizeSaysWhichPassFailed is what a caller needs when a pass reports
// something, since the plan they wrote does not name the passes it goes
// through.
func TestOptimizeSaysWhichPassFailed(t *testing.T) {
	boom := errors.New("kuma: the pass could not work out what a step produces")
	failing := plan.Pass{Name: "made up", Rewrite: func(*plan.Node) (*plan.Node, error) {
		return nil, boom
	}}

	_, err := plan.Optimize(scan, failing)
	if !errors.Is(err, boom) {
		t.Fatalf("Optimize = %v, want the error the pass gave", err)
	}
	if !strings.Contains(err.Error(), "made up") {
		t.Errorf("the error is %q, and does not say which pass it came from", err)
	}
}

// TestThePassesThatRunOverEveryQuery is the list itself, so that a pass added
// to it is a line of a test rather than something noticed later.
func TestThePassesThatRunOverEveryQuery(t *testing.T) {
	var names []string
	for _, p := range plan.Passes() {
		if p.Rewrite == nil {
			t.Errorf("the %s pass has nothing to run", p.Name)
		}
		names = append(names, p.Name)
	}

	want := "predicate pushdown, slice pushdown, projection pushdown"
	if got := strings.Join(names, ", "); got != want {
		t.Errorf("the passes are %q, want %q", got, want)
	}
}

// TestOptimizeDoesNotWriteToThePlanItWasGiven is the promise the whole design
// rests on: a caller can print the query they wrote beside the one that ran.
func TestOptimizeDoesNotWriteToThePlanItWasGiven(t *testing.T) {
	q := plan.Project(plan.Scan(trades{}), []plan.Projection{{Expr: symbol}})
	before := q.String() + "\n" + q.Input().String()

	if _, err := plan.Optimize(q, plan.Passes()...); err != nil {
		t.Fatalf("Optimize: %v", err)
	}
	if after := q.String() + "\n" + q.Input().String(); after != before {
		t.Errorf("optimizing left the plan reading\n%s\nwant\n%s", after, before)
	}
}

// BenchmarkOptimize is what a query pays to be made better, which is paid once
// per query and has to stay small next to reading anything at all.
func BenchmarkOptimize(b *testing.B) {
	q := plan.Limit(plan.Sort(
		plan.Aggregate(
			plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin),
				plan.Compare(kernel.OpGt, price, plan.Lit(100.0))),
			[]*plan.Expr{symbol},
			[]plan.Agg{{Func: plan.AggSum, Expr: qty, As: "volume"}}),
		[]plan.SortKey{{Expr: plan.Col("volume"), Descending: true}}), 0, 20)

	passes := plan.Passes()
	for b.Loop() {
		out, err := plan.Optimize(q, passes...)
		if err != nil {
			b.Fatalf("Optimize: %v", err)
		}
		sink = out
	}
}
