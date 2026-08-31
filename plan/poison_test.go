package plan_test

// The tests for the operator that stands in for a step that could not be built.
// What the operator has to say is whatever it was given, so what is tested here
// is the standing in: that the steps after it are still in the plan, that the
// first of them is the one reported, and that a plan built on one twice does
// not have two queries writing over each other's errors.

import (
	"errors"
	"strings"
	"testing"

	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// bad is a step that could not be built, holding the error a check of its input
// would have given, which is the shape the query builder makes.
func bad() *plan.Node {
	under := plan.Project(scan, []plan.Projection{{Expr: plan.Col("nope")}})
	return plan.Poison(under, "With x", under.Validate())
}

func TestAPoisonedStepIsWrittenAsItWasAsked(t *testing.T) {
	n := bad()
	if got, want := n.String(), "With x"; got != want {
		t.Errorf("the step prints as %q, want %q", got, want)
	}
	if got, want := n.Step(), "With x"; got != want {
		t.Errorf("Step is %q, want %q", got, want)
	}
	if got := n.Op(); got != plan.OpPoison {
		t.Errorf("the operator is %s, want a poisoned one", got)
	}
	if got := scan.Step(); got != "" {
		t.Errorf("a scan says its step is %q, want nothing", got)
	}
}

// TestTheStepsAfterAPoisonedOneAreStillInThePlan is the whole point of the
// operator. A query is written over several calls and has no line numbers, so
// the steps that were written after the mistake are most of what says which
// call it is about.
func TestTheStepsAfterAPoisonedOneAreStillInThePlan(t *testing.T) {
	n := plan.Limit(plan.Sort(bad(), []plan.SortKey{{Expr: price}}), 0, 3)

	want := "Limit 3\n" +
		"  Sort by price\n" +
		"    With x\n" +
		"      Project nope\n" +
		"        Scan trades/*.parquet"
	if got := n.Tree(); got != want {
		t.Errorf("the plan reads\n%s\nwant\n%s", got, want)
	}
}

// TestAPoisonedPlanIsRefusedWithWhatTheStepIsHolding is the mistake surfacing
// once, at the check, rather than at the call that made it.
func TestAPoisonedPlanIsRefusedWithWhatTheStepIsHolding(t *testing.T) {
	n := plan.Limit(plan.Sort(bad(), []plan.SortKey{{Expr: price}}), 0, 3)

	err := n.Validate()
	if err == nil {
		t.Fatal("a plan with a step that could not be built came back fine")
	}
	if !errors.Is(err, plan.ErrNoColumn) {
		t.Errorf("errors.Is does not find ErrNoColumn in %v", err)
	}

	want := "kuma: column \"nope\" not found in Project\n" +
		"  available: symbol, price, qty\n" +
		"\n" +
		"in the plan\n" +
		"  Limit 3\n" +
		"    Sort by price\n" +
		"      With x\n" +
		">       Project nope\n" +
		"          Scan trades/*.parquet"
	if got := err.Error(); got != want {
		t.Errorf("the error reads\n%s\nwant\n%s", got, want)
	}

	if _, got := n.Schema(); got == nil || got.Error() != err.Error() {
		t.Errorf("Schema says\n%v\nand Validate says\n%v", got, err)
	}
}

// TestTheMarkIsOnTheStepItselfWhenNothingBelowItIsWrong is the other half of
// where the mark goes. A step can fail to build for a reason that is not about
// an operator underneath, and then it is the one to point at.
func TestTheMarkIsOnTheStepItselfWhenNothingBelowItIsWrong(t *testing.T) {
	own := errors.New("kuma: Agg with nothing to aggregate")
	n := plan.Limit(plan.Poison(scan, "Agg", own), 0, 3)

	err := n.Validate()
	if err == nil {
		t.Fatal("a plan with a step that could not be built came back fine")
	}

	want := "kuma: Agg with nothing to aggregate\n" +
		"\n" +
		"in the plan\n" +
		"  Limit 3\n" +
		">   Agg\n" +
		"      Scan trades/*.parquet"
	if got := err.Error(); got != want {
		t.Errorf("the error reads\n%s\nwant\n%s", got, want)
	}
	if !errors.Is(err, own) {
		t.Errorf("errors.Is does not find the error the step was given in %v", err)
	}
}

// TestTheFirstStepThatCouldNotBeBuiltIsTheOneReported is the rule that a query
// carries its first mistake. The steps after it were written against something
// that did not happen, so what they have to say about it is not worth hearing.
func TestTheFirstStepThatCouldNotBeBuiltIsTheOneReported(t *testing.T) {
	first := errors.New("kuma: the first one")
	second := errors.New("kuma: the second one")
	n := plan.Poison(plan.Poison(scan, "With x", first), "With y", second)

	err := n.Validate()
	if err == nil {
		t.Fatal("a plan with two steps that could not be built came back fine")
	}
	if !errors.Is(err, first) {
		t.Errorf("the error is %v, want the first mistake", err)
	}
	if errors.Is(err, second) {
		t.Errorf("the error is %v, want it to leave the second mistake out", err)
	}
}

// TestAPoisonedStepOnTheRightOfAJoinIsFound is the second input, which is the
// one a walk that only ever goes left would miss.
func TestAPoisonedStepOnTheRightOfAJoinIsFound(t *testing.T) {
	own := errors.New("kuma: nothing to aggregate")
	n := plan.Join(scan, plan.Poison(quoted, "Agg", own), onSymbol, kernel.InnerJoin)

	if err := n.Validate(); !errors.Is(err, own) {
		t.Errorf("Validate = %v, want the mistake on the right of the join", err)
	}
}

// TestTwoQueriesOverOnePoisonedStepDoNotShareAnError is why the step holds the
// mistake in pieces and builds the error each time it is asked. One poisoned
// query can be the bottom of several, and an error says which plan it is about,
// so an error kept on the step would be two queries writing over each other.
func TestTwoQueriesOverOnePoisonedStepDoNotShareAnError(t *testing.T) {
	shared := bad()
	one := plan.Limit(shared, 0, 3)
	two := plan.Sort(shared, []plan.SortKey{{Expr: price}})

	first := one.Validate()
	second := two.Validate()
	if first == nil || second == nil {
		t.Fatal("a plan with a step that could not be built came back fine")
	}

	if got := first.Error(); !strings.Contains(got, "  Limit 3\n") {
		t.Errorf("the first error draws\n%s\nwant the plan it was asked about", got)
	}
	if got := second.Error(); !strings.Contains(got, "  Sort by price\n") {
		t.Errorf("the second error draws\n%s\nwant the plan it was asked about", got)
	}
}

// TestAStepPoisonedWithNoReasonIsStillRefused is the operator with a piece
// missing. Whatever built it left out the reason, which is a mistake in the
// builder rather than in the query, and taking it for a whole step would run a
// query the caller did not write.
func TestAStepPoisonedWithNoReasonIsStillRefused(t *testing.T) {
	err := plan.Poison(scan, "With x", nil).Validate()
	if err == nil {
		t.Fatal("a step poisoned with no reason came back fine")
	}
	if got := err.Error(); !strings.Contains(got, "With x") {
		t.Errorf("the error reads %q, want it to name the step", got)
	}
}

// TestOptimizeRefusesAPoisonedPlanWithoutNamingAPass is why the check is at the
// top of the optimizer rather than left to whichever pass asks for a schema
// first. Every pass names itself in what it returns, and "constant folding" in
// front of a wrong column name is a worse answer than the one the step already
// has.
func TestOptimizeRefusesAPoisonedPlanWithoutNamingAPass(t *testing.T) {
	_, err := plan.Optimize(plan.Limit(bad(), 0, 3), plan.Passes()...)
	if err == nil {
		t.Fatal("a plan with a step that could not be built was optimized")
	}
	for _, p := range plan.Passes() {
		if strings.Contains(err.Error(), p.Name) {
			t.Errorf("the error names the %s pass:\n%v", p.Name, err)
		}
	}
	if got, want := err.Error(), "kuma: column \"nope\" not found in Project"; !strings.HasPrefix(got, want) {
		t.Errorf("the error starts\n%s\nwant it to start with\n%s", got, want)
	}
}

// TestAPassLeavesAPoisonedStepAlone is the passes being safe to call one at a
// time, which they are meant to be. A pass that walked into a step that could
// not be built and took it for an operator it knows would read the wrong fields
// out of it.
func TestAPassLeavesAPoisonedStepAlone(t *testing.T) {
	n := plan.Limit(plan.Filter(bad(), plan.Compare(kernel.OpGt, price, plan.Lit(1.0))), 0, 3)

	for _, p := range plan.Passes() {
		t.Run(p.Name, func(t *testing.T) {
			out, err := p.Rewrite(n)
			if err != nil {
				// A pass that has to know what an operator produces asks for a
				// schema and is told the mistake, which is the right answer.
				if !errors.Is(err, plan.ErrNoColumn) {
					t.Errorf("the pass says %v, want the mistake the step is holding", err)
				}
				return
			}
			if got := out.Tree(); !strings.Contains(got, "With x") {
				t.Errorf("the plan that came out reads\n%s\nwant the step still in it", got)
			}
		})
	}
}

func BenchmarkPoisonedPlanIsFound(b *testing.B) {
	n := plan.Limit(plan.Sort(bad(), []plan.SortKey{{Expr: price}}), 0, 3)

	b.ReportAllocs()
	for b.Loop() {
		err := n.Validate()
		if err == nil {
			b.Fatal("a plan with a step that could not be built came back fine")
		}
	}
}
