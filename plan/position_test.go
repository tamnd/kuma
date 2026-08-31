package plan_test

// The tests for the part of an error that says where in a plan the mistake is.
// What each check catches is tested with the check, so what is left here is the
// pointing: that it lands on the operator that found the mistake rather than
// the one that was asked, that it survives a plan with a hole in it, and that
// wrapping the message in a plan does not stop errors.Is finding what it is.

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// deep is a plan with two filters in it, so that an error naming Filter is not
// on its own enough to say which one.
func deep() *plan.Node {
	inner := plan.Filter(scan, plan.Compare(kernel.OpGt, price, plan.Lit(100.0)))
	bad := plan.Filter(inner, plan.Compare(kernel.OpGt, plan.Col("prcie"), plan.Lit(1.0)))
	return plan.Project(bad, []plan.Projection{{Expr: symbol}})
}

func TestTheErrorMarksTheOperatorTheMistakeIsIn(t *testing.T) {
	err := deep().Validate()
	if err == nil {
		t.Fatal("a plan reading a column that is not there came back fine")
	}

	want := "kuma: column \"prcie\" not found in Filter\n" +
		"  available: symbol, price, qty\n" +
		"  did you mean: price?\n" +
		"\n" +
		"in the plan\n" +
		"  Project symbol\n" +
		">   Filter (prcie > 1)\n" +
		"      Filter (price > 100)\n" +
		"        Scan trades/*.parquet"
	if got := err.Error(); got != want {
		t.Errorf("the error reads\n%s\nwant\n%s", got, want)
	}
}

// TestTheMarkedPlanLinesUpWithThePlainOne is the reason the mark is in a gutter
// rather than in the line. A reader who has seen the plan printed anywhere else
// should be reading the same shape here.
func TestTheMarkedPlanLinesUpWithThePlainOne(t *testing.T) {
	n := deep()
	err := n.Validate()
	if err == nil {
		t.Fatal("a plan reading a column that is not there came back fine")
	}

	_, text, ok := strings.Cut(err.Error(), "in the plan\n")
	if !ok {
		t.Fatalf("the error does not print the plan:\n%s", err)
	}

	var got []string
	for line := range strings.SplitSeq(text, "\n") {
		if len(line) < 2 {
			t.Fatalf("the line %q has no room for the gutter", line)
		}
		if gutter := line[:2]; gutter != "  " && gutter != "> " {
			t.Errorf("the line %q starts with %q, want a gutter", line, gutter)
		}
		got = append(got, line[2:])
	}

	if want := strings.Split(n.Tree(), "\n"); !slices.Equal(got, want) {
		t.Errorf("with the gutter taken off the plan reads\n%s\nwant\n%s",
			strings.Join(got, "\n"), strings.Join(want, "\n"))
	}
}

// TestTheMistakeOnTheRightOfAJoinIsFound is the second input, which is the one
// a walk that only ever goes left would miss.
func TestTheMistakeOnTheRightOfAJoinIsFound(t *testing.T) {
	bad := plan.Filter(quoted, plan.Compare(kernel.OpGt, plan.Col("ask"), plan.Lit(1.0)))
	n := plan.Join(scan, bad, onSymbol, kernel.InnerJoin)

	err := n.Validate()
	if err == nil {
		t.Fatal("a join over a filter reading a column that is not there came back fine")
	}

	want := "in the plan\n" +
		"  Join inner on symbol\n" +
		"    Scan trades/*.parquet\n" +
		">   Filter (ask > 1)\n" +
		"      Scan quotes/*.parquet"
	if got := err.Error(); !strings.Contains(got, want) {
		t.Errorf("the error reads\n%s\nwant it to hold\n%s", got, want)
	}
}

// TestAPlanOfOneOperatorSaysNoMoreThanTheMessage is the case where drawing the
// plan would be pointing at the only line there is.
func TestAPlanOfOneOperatorSaysNoMoreThanTheMessage(t *testing.T) {
	err := plan.Scan(missing{}).Validate()
	if err == nil {
		t.Fatal("a scan of a source that cannot say what it holds came back fine")
	}
	if got := err.Error(); strings.Contains(got, "in the plan") {
		t.Errorf("the error draws a plan of one operator:\n%s", got)
	}
}

// TestTheMessageIsTheOneTheCheckGaveWhenTheOperatorIsAsked is the promise that
// this only adds. A caller who was reading the message before reads the same
// first lines now.
func TestTheMessageIsTheOneTheCheckGaveWhenTheOperatorIsAsked(t *testing.T) {
	n := deep()
	err := n.Validate()
	if err == nil {
		t.Fatal("a plan reading a column that is not there came back fine")
	}

	var oe *plan.OperatorError
	if !errors.As(err, &oe) {
		t.Fatalf("the error is %T, want one that says where it came from", err)
	}
	if got := oe.Err.Error(); strings.Contains(got, "in the plan") {
		t.Errorf("the error underneath already draws the plan:\n%s", got)
	}
	if !strings.HasPrefix(err.Error(), oe.Err.Error()) {
		t.Errorf("the message starts\n%s\nwant it to start with\n%s", err, oe.Err)
	}
	if oe.Plan != n {
		t.Errorf("the plan to print is %s, want the one that was asked", oe.Plan)
	}
	if oe.At == nil || oe.At.Op() != plan.OpFilter {
		t.Errorf("the mistake is put at %s, want the filter", oe.At)
	}
}

// TestWhatTheErrorStillIs is the part that would be easy to break, since a
// caller asking whether a name was wrong should not have to know that the
// answer now comes with a plan attached.
func TestWhatTheErrorStillIs(t *testing.T) {
	err := deep().Validate()
	if !errors.Is(err, plan.ErrNoColumn) {
		t.Errorf("errors.Is does not find ErrNoColumn in %v", err)
	}

	var ce *plan.ColumnError
	if !errors.As(err, &ce) {
		t.Fatalf("errors.As does not find a ColumnError in %v", err)
	}
	if ce.Name != "prcie" || ce.Op != "Filter" {
		t.Errorf("the column error is for %q in %q, want prcie in Filter", ce.Name, ce.Op)
	}
}

// TestSchemaAndValidateSayTheSameThingAboutWhere is the pair being one walk, so
// that a caller who wants the schema and a caller who wants to know whether the
// query is right are told the same story.
func TestSchemaAndValidateSayTheSameThingAboutWhere(t *testing.T) {
	n := deep()
	_, err := n.Schema()
	if err == nil {
		t.Fatal("a plan reading a column that is not there gave a schema")
	}
	if got := n.Validate(); got == nil || got.Error() != err.Error() {
		t.Errorf("Validate says\n%v\nand Schema says\n%v", got, err)
	}
}

// TestThePlanPrintsWithAHoleInIt is the printer being asked to draw the plans
// it is least likely to be given a whole one of. An operator with a piece
// missing is exactly the plan an error is about, so a printer that gives up on
// one is a printer that is never there when it is wanted.
func TestThePlanPrintsWithAHoleInIt(t *testing.T) {
	cases := []struct {
		name string
		node *plan.Node
		want string
	}{
		{"a filter with no condition", plan.Filter(scan, nil), "Filter ?"},
		{
			"a projection of nothing",
			plan.Project(scan, []plan.Projection{{Expr: symbol}, {}}),
			"Project symbol, ?",
		},
		{"a sort by nothing", plan.Sort(scan, []plan.SortKey{{}}), "Sort by ?"},
		{
			"a group by nothing",
			plan.Aggregate(scan, []*plan.Expr{nil}, []plan.Agg{{Func: plan.AggSize}}),
			"Aggregate by ?: Size()",
		},
		{
			"a join key with only one side",
			plan.Join(scan, quoted, []plan.JoinKey{{Left: symbol}}, kernel.InnerJoin),
			"Join inner on symbol = ?",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.node.String(); got != tt.want {
				t.Errorf("the operator prints as %q, want %q", got, tt.want)
			}
			// The check has something to say about every one of these, and
			// saying it means printing the plan, which is the walk that used to
			// be a panic.
			if err := tt.node.Validate(); err == nil {
				t.Error("a plan with a hole in it came back fine")
			} else if err.Error() == "" {
				t.Error("the error has nothing in it")
			}
		})
	}
}

func BenchmarkOperatorErrorMessage(b *testing.B) {
	err := deep().Validate()
	if err == nil {
		b.Fatal("a plan reading a column that is not there came back fine")
	}

	b.ReportAllocs()
	for b.Loop() {
		textSink = err.Error()
	}
}
