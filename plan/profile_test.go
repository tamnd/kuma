package plan_test

// The tests for printing a plan that has run. The measures here are written out
// rather than timed, because a test that reads a clock is a test that fails on
// a busy machine, and what is being checked is the arithmetic and the format
// rather than whether time passes.

import (
	"strings"
	"testing"
	"time"

	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// A cost is one operator's line of a made up profile: how long it and
// everything under it took, and how many rows it produced.
type cost struct {
	took time.Duration
	rows int64
}

// timed builds a measure over a plan out of the costs, in the order
// [plan.Node.Tree] prints the operators, which is the operator and then its
// inputs. It takes the costs by pointer because it walks the plan and the costs
// together and there is one list of costs for the whole walk.
func timed(t *testing.T, n *plan.Node, costs *[]cost) plan.Measure {
	t.Helper()
	if len(*costs) == 0 {
		t.Fatalf("the plan has more operators than there are costs for it")
	}

	c := (*costs)[0]
	*costs = (*costs)[1:]

	m := plan.Measure{Node: n, Took: c.took, Rows: c.rows}
	for _, in := range []*plan.Node{n.Input(), n.Right()} {
		if in != nil {
			m.Input = append(m.Input, timed(t, in, costs))
		}
	}
	return m
}

// ranAs optimizes the plan the way a query is optimized and hangs the costs off
// what came out, which is the plan that would have run.
func ranAs(t *testing.T, n *plan.Node, costs ...cost) plan.Measure {
	t.Helper()

	ran, err := plan.Optimize(n, plan.Passes()...)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	rest := costs
	m := timed(t, ran, &rest)
	if len(rest) > 0 {
		t.Fatalf("there are %d costs left over for a plan of %d operators", len(rest), len(costs)-len(rest))
	}
	return m
}

// TestProfileWritesWhatEachOperatorCostNextToIt is the format, written out. The
// times on the lines are what each operator spent on its own, so they add up to
// the total, and the rows on a line are the rows the line above it read.
func TestProfileWritesWhatEachOperatorCostNextToIt(t *testing.T) {
	q := plan.Limit(plan.Project(plan.Filter(scan, dear),
		[]plan.Projection{{Expr: symbol}, {Expr: price}}), 0, 20)

	ran := ranAs(t, q,
		cost{100 * time.Microsecond, 2},
		cost{90 * time.Microsecond, 2},
		cost{80 * time.Microsecond, 3},
		cost{30 * time.Microsecond, 4})

	want := "the query as written\n" +
		"  Limit 20\n" +
		"    Project symbol, price\n" +
		"      Filter (price > 100)\n" +
		"        Scan trades/*.parquet\n" +
		"\n" +
		"the query that ran\n" +
		"  Project symbol, price                         10.0us   2 rows\n" +
		"    Limit 20                                    10.0us   2 rows\n" +
		"      Filter (price > 100)                      50.0us   3 rows\n" +
		"        Scan trades/*.parquet [symbol, price]   30.0us   4 rows\n" +
		"\n" +
		"changed by slice pushdown and projection pushdown\n" +
		"ran in 100us\n"

	got, err := plan.Profile(q, ran, plan.Passes()...)
	if err != nil {
		t.Fatalf("Profile: %v", err)
	}
	if got != want {
		t.Errorf("Profile gave\n%s\nwant\n%s", got, want)
	}
}

// TestProfileOfAQueryNoPassChanges is the shorter shape, which is the same rule
// [plan.Explain] follows: a plan printed twice under two headings is a worse
// way of saying nothing happened than saying so.
func TestProfileOfAQueryNoPassChanges(t *testing.T) {
	q := plan.Filter(scan, dear)

	ran := ranAs(t, q,
		cost{4 * time.Millisecond, 3},
		cost{time.Millisecond, 4})

	want := "the query that ran\n" +
		"  Filter (price > 100)      3.00ms   3 rows\n" +
		"    Scan trades/*.parquet   1.00ms   4 rows\n" +
		"\n" +
		"nothing the optimizer does changes it\n" +
		"ran in 4.00ms\n"

	got, err := plan.Profile(q, ran, plan.Passes()...)
	if err != nil {
		t.Fatalf("Profile: %v", err)
	}
	if got != want {
		t.Errorf("Profile gave\n%s\nwant\n%s", got, want)
	}
}

// TestProfileOfBothSidesOfAJoin is the one operator that reads two things, and
// the reason a measure is a tree: the rows a join read are on the two lines
// under it rather than on its own.
func TestProfileOfBothSidesOfAJoin(t *testing.T) {
	q := plan.Join(scan, quoted, onSymbol, kernel.InnerJoin)

	ran := ranAs(t, q,
		cost{10 * time.Millisecond, 9},
		cost{2 * time.Millisecond, 4},
		cost{3 * time.Millisecond, 5})

	want := "the query that ran\n" +
		"  Join inner on symbol      5.00ms   9 rows\n" +
		"    Scan trades/*.parquet   2.00ms   4 rows\n" +
		"    Scan quotes/*.parquet   3.00ms   5 rows\n" +
		"\n" +
		"nothing the optimizer does changes it\n" +
		"ran in 10.0ms\n"

	got, err := plan.Profile(q, ran, plan.Passes()...)
	if err != nil {
		t.Fatalf("Profile: %v", err)
	}
	if got != want {
		t.Errorf("Profile gave\n%s\nwant\n%s", got, want)
	}
}

// TestProfileOfAnOperatorThatLooksFasterThanItsInput is the clamp. Two clock
// reads either side of a call that took no time at all can come back in the
// wrong order, and a profile that says an operator took less than nothing is
// one nobody believes the rest of.
func TestProfileOfAnOperatorThatLooksFasterThanItsInput(t *testing.T) {
	q := plan.Filter(scan, dear)

	ran := ranAs(t, q,
		cost{time.Microsecond, 3},
		cost{2 * time.Microsecond, 4})

	got, err := plan.Profile(q, ran, plan.Passes()...)
	if err != nil {
		t.Fatalf("Profile: %v", err)
	}
	if !strings.Contains(got, "0ns   3 rows") {
		t.Errorf("Profile gave\n%s\nwant the filter to have spent no time rather than less than none", got)
	}
}

// TestProfileCountsOneRowAsOne is the word after the number, which is the sort
// of thing nobody notices until it is wrong.
func TestProfileCountsOneRowAsOne(t *testing.T) {
	q := plan.Filter(scan, dear)

	ran := ranAs(t, q,
		cost{2 * time.Microsecond, 1},
		cost{time.Microsecond, 0})

	got, err := plan.Profile(q, ran, plan.Passes()...)
	if err != nil {
		t.Fatalf("Profile: %v", err)
	}
	if !strings.Contains(got, "1 row\n") {
		t.Errorf("Profile gave\n%s\nwant one row written as one row", got)
	}
	if !strings.Contains(got, "0 rows") {
		t.Errorf("Profile gave\n%s\nwant no rows written as no rows", got)
	}
}

// TestProfilePrintsATimeWithoutTheMicroSign is the rule the whole repository
// follows, and time.Duration does not, so this is the one place it has to be
// checked rather than assumed.
func TestProfilePrintsATimeWithoutTheMicroSign(t *testing.T) {
	cases := []struct {
		took time.Duration
		want string
	}{
		{0, "0ns"},
		{999 * time.Nanosecond, "999ns"},
		{time.Microsecond, "1.00us"},
		{1500 * time.Nanosecond, "1.50us"},
		{12345 * time.Nanosecond, "12.3us"},
		{999500 * time.Nanosecond, "1000us"},
		{time.Millisecond, "1.00ms"},
		{time.Second, "1.00s"},
		{90 * time.Second, "90.0s"},
	}

	for _, c := range cases {
		t.Run(c.want, func(t *testing.T) {
			ran := ranAs(t, scan, cost{c.took, 1})
			got, err := plan.Profile(scan, ran, plan.Passes()...)
			if err != nil {
				t.Fatalf("Profile: %v", err)
			}
			if !strings.HasSuffix(got, "ran in "+c.want+"\n") {
				t.Errorf("Profile of a run of %s gave\n%s\nwant it to end in %q", c.took, got, c.want)
			}
			for _, r := range got {
				if r > 126 || (r < 32 && r != '\n') {
					t.Fatalf("Profile gave %q, which is not a character a terminal has to guess at", r)
				}
			}
		})
	}
}

func TestProfileOfNoPlan(t *testing.T) {
	_, err := plan.Profile(nil, plan.Measure{Node: scan}, plan.Passes()...)
	if err == nil || !strings.Contains(err.Error(), "no operator") {
		t.Fatalf("Profile = %v, want a plan with no operator in it", err)
	}
}

func TestProfileOfARunThatDidNotHappen(t *testing.T) {
	_, err := plan.Profile(scan, plan.Measure{}, plan.Passes()...)
	if err == nil || !strings.Contains(err.Error(), "has not run") {
		t.Fatalf("Profile = %v, want a profile of a query that has not run", err)
	}
}

// TestProfileOfAQueryThatIsWrong cannot really happen, since a query that does
// not check does not run and so has nothing to profile, but the check is here
// because the alternative is printing a plan while the passes report an error
// about it.
func TestProfileOfAQueryThatIsWrong(t *testing.T) {
	q := plan.Filter(scan, plan.Compare(kernel.OpGt, plan.Col("volume"), plan.Lit(100.0)))

	_, err := plan.Profile(q, plan.Measure{Node: scan, Rows: 4}, plan.Passes()...)
	if err == nil || !strings.Contains(err.Error(), "volume") {
		t.Fatalf("Profile = %v, want the error to name the column", err)
	}
}

// BenchmarkProfile is what printing a profile costs, which is paid once by a
// query that has already run and so has room to be a little slower than an
// explain.
func BenchmarkProfile(b *testing.B) {
	q := plan.Limit(plan.Sort(
		plan.Aggregate(
			plan.Filter(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin),
				plan.Compare(kernel.OpGt, price, plan.Lit(100.0))),
			[]*plan.Expr{symbol},
			[]plan.Agg{{Func: plan.AggSum, Expr: qty, As: "volume"}}),
		[]plan.SortKey{{Expr: plan.Col("volume"), Descending: true}}), 0, 20)

	ran, err := plan.Optimize(q, plan.Passes()...)
	if err != nil {
		b.Fatalf("Optimize: %v", err)
	}
	m := allTook(ran, time.Millisecond)

	passes := plan.Passes()
	b.ReportAllocs()
	for b.Loop() {
		out, err := plan.Profile(q, m, passes...)
		if err != nil {
			b.Fatalf("Profile: %v", err)
		}
		textSink = out
	}
}

// allTook is a measure over a plan with the same cost on every operator, which
// is what a benchmark of the printing wants and no test would accept.
func allTook(n *plan.Node, d time.Duration) plan.Measure {
	m := plan.Measure{Node: n, Took: d, Rows: 1000}
	for _, in := range []*plan.Node{n.Input(), n.Right()} {
		if in != nil {
			m.Input = append(m.Input, allTook(in, d/2))
		}
	}
	return m
}
