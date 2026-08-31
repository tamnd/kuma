package plan_test

// The tests for the expression fusion pass. What was fused is read off the
// printed plan, since the whole of what the pass does is bring the expressions
// of two projections into one and leave one operator where there were two.

import (
	"testing"

	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// The values these tests share. A step written on top of a column that a
// projection underneath works out is the shape the pass exists for, so the
// pieces of that shape are built once.
var (
	bumped   = plan.Arith(kernel.OpAdd, price, plan.Lit(1.0))
	notional = plan.Col("notional")
	scaled   = plan.Arith(kernel.OpMul, notional, plan.Lit(2.0))
	raised   = plan.Arith(kernel.OpAdd, notional, plan.Lit(1.0))
)

// fusedPlan runs the pass and fails the test if it reports anything, since none
// of the plans it is given here are wrong.
func fusedPlan(t *testing.T, n *plan.Node) *plan.Node {
	t.Helper()
	out, err := plan.Fuse(n)
	if err != nil {
		t.Fatalf("Fuse: %v", err)
	}
	return out
}

func TestFuseBringsTwoProjectionsIntoOne(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
		want string
	}{
		{
			name: "a value worked out and then used once",
			plan: plan.Project(plan.Project(scan, []plan.Projection{
				{Expr: bumped, As: "notional"},
				{Expr: symbol},
			}), []plan.Projection{
				{Expr: scaled, As: "doubled"},
				{Expr: symbol},
			}),
			want: "Project ((price + 1) * 2) as doubled, symbol\n" +
				"  Scan trades/*.parquet",
		},
		{
			name: "one set of columns chosen out of another",
			plan: plan.Project(plan.Project(scan, []plan.Projection{
				{Expr: symbol},
				{Expr: price},
			}), []plan.Projection{
				{Expr: symbol},
			}),
			want: "Project symbol\n  Scan trades/*.parquet",
		},
		{
			name: "a column renamed twice",
			plan: plan.Project(plan.Project(scan, []plan.Projection{
				{Expr: price, As: "px"},
			}), []plan.Projection{
				{Expr: plan.Col("px"), As: "p"},
			}),
			want: "Project price as p\n  Scan trades/*.parquet",
		},
		{
			name: "a value worked out and then not used at all",
			plan: plan.Project(plan.Project(scan, []plan.Projection{
				{Expr: bumped, As: "notional"},
				{Expr: symbol},
			}), []plan.Projection{
				{Expr: symbol},
			}),
			want: "Project symbol\n  Scan trades/*.parquet",
		},
		{
			name: "three projections at once",
			plan: plan.Project(plan.Project(plan.Project(scan, []plan.Projection{
				{Expr: bumped, As: "notional"},
			}), []plan.Projection{
				{Expr: scaled, As: "notional"},
			}), []plan.Projection{
				{Expr: scaled, As: "doubled"},
			}),
			want: "Project (((price + 1) * 2) * 2) as doubled\n" +
				"  Scan trades/*.parquet",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := tree(fusedPlan(t, c.plan)); got != c.want {
				t.Errorf("Fuse gave\n%s\nwant\n%s", got, c.want)
			}
		})
	}
}

// TestFuseLeavesAValueThatWouldBeWorkedOutTwice is the rule the pass turns on.
// Inlining a value into two places means working it out in both of them, which
// is the thing [plan.HoistCommon] exists to undo, so the pass that would cause
// it does not.
func TestFuseLeavesAValueThatWouldBeWorkedOutTwice(t *testing.T) {
	n := plan.Project(plan.Project(scan, []plan.Projection{
		{Expr: bumped, As: "notional"},
	}), []plan.Projection{
		{Expr: scaled, As: "doubled"},
		{Expr: raised, As: "bumped"},
	})

	if out := fusedPlan(t, n); out != n {
		t.Errorf("Fuse worked a value out twice:\n%s", tree(out))
	}
}

// TestFuseInlinesAColumnAndAValueHoweverOftenTheyAreRead is the other half of
// that rule. Neither is work, so repeating them costs nothing, and these are
// the two [plan.HoistCommon] refuses to hoist for the same reason.
func TestFuseInlinesAColumnAndAValueHoweverOftenTheyAreRead(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
		want string
	}{
		{
			name: "a column read twice",
			plan: plan.Project(plan.Project(scan, []plan.Projection{
				{Expr: price, As: "px"},
			}), []plan.Projection{
				{Expr: plan.Arith(kernel.OpMul, plan.Col("px"), plan.Col("px")), As: "sq"},
			}),
			want: "Project (price * price) as sq\n  Scan trades/*.parquet",
		},
		{
			name: "a value read twice",
			plan: plan.Project(plan.Project(scan, []plan.Projection{
				{Expr: plan.Lit(2.0), As: "two"},
			}), []plan.Projection{
				{Expr: plan.Arith(kernel.OpMul, plan.Col("two"), plan.Col("two")), As: "four"},
			}),
			want: "Project (2 * 2) as four\n  Scan trades/*.parquet",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := tree(fusedPlan(t, c.plan)); got != c.want {
				t.Errorf("Fuse gave\n%s\nwant\n%s", got, c.want)
			}
		})
	}
}

func TestFuseHandsBackThePlanWhenThereIsNothingToBringTogether(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
	}{
		{"a scan on its own", scan},
		{"one projection", plan.Project(scan, []plan.Projection{{Expr: symbol}})},
		{"a filter between two projections", plan.Project(plan.Filter(
			plan.Project(scan, []plan.Projection{{Expr: symbol}, {Expr: price}}),
			dear), []plan.Projection{{Expr: symbol}})},
		{"an aggregation over a projection", plan.Aggregate(
			plan.Project(scan, []plan.Projection{{Expr: symbol}}),
			[]*plan.Expr{symbol}, nil)},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if out := fusedPlan(t, c.plan); out != c.plan {
				t.Errorf("Fuse rebuilt a plan it had nothing to do to:\n%s", tree(out))
			}
		})
	}
}

// TestFuseKeepsTheNameOfAProjection is the trap in the whole pass. A projection
// with no name of its own is called after the expression it holds, so inlining
// into one would quietly rename the column the query produces.
func TestFuseKeepsTheNameOfAProjection(t *testing.T) {
	n := plan.Project(plan.Project(scan, []plan.Projection{
		{Expr: bumped, As: "notional"},
	}), []plan.Projection{
		{Expr: scaled},
	})

	want := "Project ((price + 1) * 2) as (notional * 2)\n  Scan trades/*.parquet"
	if got := tree(fusedPlan(t, n)); got != want {
		t.Errorf("Fuse gave\n%s\nwant\n%s", got, want)
	}
}

// TestFuseKeepsWhatThePlanProduces is the property the naming rule is there to
// hold up. A plan the pass has been over has to produce the same columns under
// the same names and types as the plan it was given, or the pass has changed
// the query rather than the way it runs.
func TestFuseKeepsWhatThePlanProduces(t *testing.T) {
	plans := []*plan.Node{
		plan.Project(plan.Project(scan, []plan.Projection{
			{Expr: bumped, As: "notional"},
			{Expr: symbol},
		}), []plan.Projection{
			{Expr: scaled, As: "doubled"},
			{Expr: symbol},
		}),
		plan.Project(plan.Project(scan, []plan.Projection{
			{Expr: bumped, As: "notional"},
		}), []plan.Projection{
			{Expr: scaled},
		}),
		plan.Project(plan.Project(scan, []plan.Projection{
			{Expr: price, As: "px"},
		}), []plan.Projection{
			{Expr: plan.Col("px"), As: "p"},
		}),
	}

	for _, n := range plans {
		want, err := n.Schema()
		if err != nil {
			t.Fatalf("Schema of the plan as written: %v", err)
		}
		got, err := fusedPlan(t, n).Schema()
		if err != nil {
			t.Fatalf("Schema of the fused plan: %v", err)
		}
		if got.String() != want.String() {
			t.Errorf("Fuse of\n%s\nproduces %s, want %s", tree(n), got, want)
		}
	}
}

func TestFuseSettles(t *testing.T) {
	n := plan.Project(plan.Project(scan, []plan.Projection{
		{Expr: bumped, As: "notional"},
		{Expr: symbol},
	}), []plan.Projection{
		{Expr: scaled, As: "doubled"},
		{Expr: symbol},
	})

	out := fusedPlan(t, n)
	if out == n {
		t.Fatal("Fuse found nothing in a plan with a projection over a projection")
	}
	if again := fusedPlan(t, out); again != out {
		t.Errorf("Fuse of a plan it had already done gave\n%s\nwant the plan it was given", tree(again))
	}
}

// TestFuseOfAPlanThatIsWrong leaves the error to the check. A projection over
// one that does not produce the columns it reads is a plan with something wrong
// with it, and the message about that should name the query as it was written.
func TestFuseOfAPlanThatIsWrong(t *testing.T) {
	n := plan.Project(plan.Project(scan, []plan.Projection{
		{Expr: symbol},
	}), []plan.Projection{
		{Expr: price},
	})

	if out := fusedPlan(t, n); out != n {
		t.Errorf("Fuse rewrote a plan that is wrong:\n%s", tree(out))
	}
	if _, err := n.Schema(); err == nil {
		t.Error("the plan the pass left alone was checked without an error")
	}
}

// TestFuseWithTheOtherPasses is the one that would catch this pass and the
// common subexpression pass undoing each other. The query works one value out
// and reads it once, which fusion takes up, and works another out and reads it
// twice, which hoisting puts down, and the two of them have to settle.
func TestFuseWithTheOtherPasses(t *testing.T) {
	n := plan.Project(plan.Filter(plan.Project(scan, []plan.Projection{
		{Expr: bumped, As: "notional"},
		{Expr: symbol},
	}), plan.Compare(kernel.OpGt, notional, plan.Lit(100.0))), []plan.Projection{
		{Expr: scaled, As: "doubled"},
		{Expr: symbol},
	})

	out, err := plan.Optimize(n, plan.Passes()...)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	want, err := n.Schema()
	if err != nil {
		t.Fatalf("Schema of the plan as written: %v", err)
	}
	got, err := out.Schema()
	if err != nil {
		t.Fatalf("Schema of the optimized plan: %v", err)
	}
	if got.String() != want.String() {
		t.Errorf("the optimized plan produces %s, want %s\n%s", got, want, tree(out))
	}
}

// TestFuseAndHoistSettleOnAValueFusionRepeats is the case the two passes have
// to agree about. Taking a value up into the one place that reads it can put it
// next to a copy of itself that was written there already, which is then a value
// written twice and the common subexpression pass puts it back down. It has to
// stop there. The pass that took it up will not take it up again, because by
// then it is read twice and that is the one thing it refuses to do, and this is
// the test that says so rather than the argument.
func TestFuseAndHoistSettleOnAValueFusionRepeats(t *testing.T) {
	n := plan.Project(plan.Project(scan, []plan.Projection{
		{Expr: bumped, As: "k"},
		{Expr: price},
	}), []plan.Projection{
		{Expr: plan.Arith(kernel.OpMul, plan.Col("k"), bumped), As: "r"},
	})

	out, err := plan.Optimize(n, plan.Passes()...)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	want := "Project ((price + 1) * (price + 1)) as r\n" +
		"  Project (price + 1)\n" +
		"    Scan trades/*.parquet [price]"
	if got := tree(out); got != want {
		t.Errorf("Optimize gave\n%s\nwant\n%s", got, want)
	}
}

func BenchmarkFuse(b *testing.B) {
	n := plan.Project(plan.Project(scan, []plan.Projection{
		{Expr: bumped, As: "notional"},
		{Expr: symbol},
	}), []plan.Projection{
		{Expr: scaled, As: "doubled"},
		{Expr: symbol},
	})

	b.ReportAllocs()
	for b.Loop() {
		out, err := plan.Fuse(n)
		if err != nil {
			b.Fatal(err)
		}
		sink = out
	}
}
