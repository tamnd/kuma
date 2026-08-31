package plan_test

// The tests for the slice pushdown. Where a limit ended up is read off the
// printed plan, which is why a scan prints the run of rows it reads, and the
// operators the limit must not move under get a case each.

import (
	"math"
	"strings"
	"testing"

	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// slid moves the limits of a plan and fails the test if the pass reports
// anything, since none of the plans here are wrong.
func slid(t *testing.T, n *plan.Node) *plan.Node {
	t.Helper()
	out, err := plan.PushSlice(n)
	if err != nil {
		t.Fatalf("PushSlice: %v", err)
	}
	return out
}

func TestPushSliceSinksALimit(t *testing.T) {
	just := []plan.Projection{{Expr: symbol}}
	volume := []plan.Agg{{Func: plan.AggSum, Expr: qty, As: "volume"}}
	byPrice := []plan.SortKey{{Expr: price}}

	cases := []struct {
		name string
		plan *plan.Node
		want string
	}{
		{
			name: "a limit over a scan is written into the scan",
			plan: plan.Limit(scan, 0, 20),
			want: "Scan trades/*.parquet rows 0 to 20",
		},
		{
			name: "an offset goes into the scan with it",
			plan: plan.Limit(scan, 5, 20),
			want: "Scan trades/*.parquet rows 5 to 25",
		},
		{
			name: "a limit of no rows at all still goes in",
			plan: plan.Limit(scan, 0, 0),
			want: "Scan trades/*.parquet rows 0 to 0",
		},
		{
			name: "a limit goes under a projection, which works out the rows it keeps",
			plan: plan.Limit(plan.Project(scan, just), 0, 20),
			want: "Project symbol\n  Scan trades/*.parquet rows 0 to 20",
		},
		{
			name: "and under two of them",
			plan: plan.Limit(plan.Project(plan.Project(scan, just), just), 0, 20),
			want: "Project symbol\n  Project symbol\n    Scan trades/*.parquet rows 0 to 20",
		},
		{
			name: "a limit stays over a filter, which is not one row out for each row in",
			plan: plan.Limit(plan.Filter(scan, dear), 0, 20),
			want: "Limit 20\n  Filter (price > 100)\n    Scan trades/*.parquet",
		},
		{
			name: "a limit stays over a sort, since which rows the first twenty are is the sort",
			plan: plan.Limit(plan.Sort(scan, byPrice), 0, 20),
			want: "Limit 20\n  Sort by price\n    Scan trades/*.parquet",
		},
		{
			name: "a limit stays over an aggregate",
			plan: plan.Limit(plan.Aggregate(scan, []*plan.Expr{symbol}, volume), 0, 20),
			want: "Limit 20\n  Aggregate by symbol: Sum(qty) as volume\n    Scan trades/*.parquet",
		},
		{
			name: "a limit stays over a distinct",
			plan: plan.Limit(plan.Distinct(scan, []*plan.Expr{symbol}), 0, 20),
			want: "Limit 20\n  Distinct by symbol\n    Scan trades/*.parquet",
		},
		{
			name: "a limit stays over an explode, which turns one row into as many as the list is long",
			plan: plan.Limit(plan.Explode(listed, []string{"tags"}), 0, 20),
			want: "Limit 20\n  Explode tags\n    Scan orders/*.parquet",
		},
		{
			name: "a limit stays over a join",
			plan: plan.Limit(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin), 0, 20),
			want: "Limit 20\n  Join inner on symbol\n    Scan trades/*.parquet\n    Scan quotes/*.parquet",
		},
		{
			name: "a limit under a filter carries on down without the filter moving",
			plan: plan.Filter(plan.Limit(plan.Project(scan, just), 0, 20), dear),
			want: "Filter (price > 100)\n  Project symbol\n    Scan trades/*.parquet rows 0 to 20",
		},
		{
			name: "a limit on each side of a join is written into each scan",
			plan: plan.Join(plan.Limit(scan, 0, 20), plan.Limit(quoted, 3, 4), onSymbol, kernel.InnerJoin),
			want: "Join inner on symbol\n  Scan trades/*.parquet rows 0 to 20\n  Scan quotes/*.parquet rows 3 to 7",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if got := tree(slid(t, tt.plan)); got != tt.want {
				t.Errorf("the plan came out as\n%s\nwant\n%s", got, tt.want)
			}
		})
	}
}

// TestPushSliceMergesTwoLimits is the arithmetic of a slice of a slice, which
// is the half of this pass that is not about moving anything.
func TestPushSliceMergesTwoLimits(t *testing.T) {
	cases := []struct {
		name  string
		outer [2]int64
		inner [2]int64
		want  string
	}{
		{
			name:  "two heads keep the smaller",
			outer: [2]int64{0, 5},
			inner: [2]int64{0, 20},
			want:  "Scan trades/*.parquet rows 0 to 5",
		},
		{
			name:  "the smaller one being the outer one is the same answer",
			outer: [2]int64{0, 20},
			inner: [2]int64{0, 5},
			want:  "Scan trades/*.parquet rows 0 to 5",
		},
		{
			name:  "the offsets add up",
			outer: [2]int64{3, 4},
			inner: [2]int64{10, 20},
			want:  "Scan trades/*.parquet rows 13 to 17",
		},
		{
			name:  "an outer window that runs off the end of the inner one is cut to it",
			outer: [2]int64{8, 20},
			inner: [2]int64{10, 10},
			want:  "Scan trades/*.parquet rows 18 to 20",
		},
		{
			name:  "an outer offset past the end of the inner window leaves no rows",
			outer: [2]int64{30, 20},
			inner: [2]int64{10, 10},
			want:  "Scan trades/*.parquet rows 40 to 40",
		},
		{
			name:  "an offset that would run off the end of an int64 stops there",
			outer: [2]int64{math.MaxInt64, 1},
			inner: [2]int64{math.MaxInt64, math.MaxInt64},
			want:  "Scan trades/*.parquet rows 9223372036854775807 to 9223372036854775807",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			n := plan.Limit(plan.Limit(scan, tt.inner[0], tt.inner[1]), tt.outer[0], tt.outer[1])
			if got := tree(slid(t, n)); got != tt.want {
				t.Errorf("the plan came out as\n%s\nwant\n%s", got, tt.want)
			}
		})
	}
}

// TestPushSliceLeavesAPlanItFoundNothingIn is the contract every pass has to
// keep, since a pass that rebuilds a plan it did not change is a pass the
// optimizer never stops running.
func TestPushSliceLeavesAPlanItFoundNothingIn(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
	}{
		{"a plan with no limit in it", plan.Filter(scan, dear)},
		{"a limit that is already over the operator that stops it", plan.Limit(plan.Filter(scan, dear), 0, 20)},
		{"a limit over a sort", plan.Limit(plan.Sort(scan, []plan.SortKey{{Expr: price}}), 0, 20)},
		{"a limit over a join", plan.Limit(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin), 0, 20)},
		{"a scan that is already read as a run of rows", plan.ScanSlice(trades{}, 0, 20)},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			if got := slid(t, tt.plan); got != tt.plan {
				t.Errorf("the pass gave back a new plan\n%s\nfor one it changed nothing in\n%s", tree(got), tree(tt.plan))
			}
		})
	}
}

// TestPushSliceSettles runs the pass over its own output, which is what the
// optimizer does and what a pass that keeps finding the same thing to do would
// fail.
func TestPushSliceSettles(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
	}{
		{"a limit over a scan", plan.Limit(scan, 0, 20)},
		{"a limit over a projection", plan.Limit(plan.Project(scan, []plan.Projection{{Expr: symbol}}), 4, 20)},
		{"a limit over a limit", plan.Limit(plan.Limit(scan, 3, 30), 4, 20)},
		{"a limit over an operator that stops it", plan.Limit(plan.Filter(scan, dear), 0, 20)},
		{"a limit over a scan that is already read as a run", plan.Limit(plan.ScanSlice(trades{}, 10, 10), 2, 3)},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			once := slid(t, tt.plan)
			if twice := slid(t, once); twice != once {
				t.Errorf("running the pass again gave\n%s\nfrom\n%s", tree(twice), tree(once))
			}
		})
	}
}

// TestPushSliceKeepsWhatThePlanProduces is the check that matters most, since a
// pass is only allowed to change how a query runs and never what it answers.
func TestPushSliceKeepsWhatThePlanProduces(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
	}{
		{"a limit over a scan", plan.Limit(scan, 2, 20)},
		{"a limit over a projection", plan.Limit(plan.Project(scan, []plan.Projection{{Expr: symbol}}), 0, 20)},
		{"a limit over a join", plan.Limit(plan.Join(scan, quoted, onSymbol, kernel.InnerJoin), 0, 20)},
		{"a limit over an explode", plan.Limit(plan.Explode(listed, []string{"tags"}), 0, 20)},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			before, err := tt.plan.Schema()
			if err != nil {
				t.Fatalf("Schema: %v", err)
			}
			after, err := slid(t, tt.plan).Schema()
			if err != nil {
				t.Fatalf("Schema of the plan that came out: %v", err)
			}
			if !before.Equal(after) {
				t.Errorf("the plan produces %s, want %s", after, before)
			}
		})
	}
}

// TestPushSliceOfAPlanThatIsWrong is the pass meeting a query that was never
// going to run. Unlike the other two pushdowns it has nothing to ask a source,
// so it moves the limit and says nothing, and the plan is turned away where
// every other wrong plan is.
func TestPushSliceOfAPlanThatIsWrong(t *testing.T) {
	q := plan.Limit(plan.Scan(missing{}), 0, 20)

	out, err := plan.PushSlice(q)
	if err != nil {
		t.Fatalf("PushSlice = %v, want the pass to have nothing to say", err)
	}
	if _, err := out.Schema(); err == nil || !strings.Contains(err.Error(), "no such file") {
		t.Fatalf("the schema of what came out = %v, want what the source said", err)
	}
}

// TestTheThreePushdownsTogether is the shape a real query has, with all three
// passes run over it the way the optimizer runs them.
func TestTheThreePushdownsTogether(t *testing.T) {
	// Read the file, keep the first hundred rows, work out an amount, keep the
	// dear ones, and take twenty. All three passes find something and each of
	// them gives the next one more to find. The condition sinks under the
	// projection, since price is a column that goes through it unchanged, and
	// stops on the limit. The inner limit goes into the scan, and the outer one
	// follows it under the projection until the filter stops it, which is
	// exactly as far as it may go. Then the read is narrowed to the two columns
	// that are left, which are the ones the amount is worked out from.
	n := plan.Limit(
		plan.Filter(
			plan.Project(
				plan.Limit(scan, 0, 100),
				[]plan.Projection{{Expr: amount, As: "amount"}, {Expr: price}},
			),
			plan.Compare(kernel.OpGt, price, plan.Lit(100.0)),
		),
		0, 20)

	out, err := plan.Optimize(n, plan.Passes()...)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	want := "Project (price * (qty as float64)) as amount, price\n" +
		"  Limit 20\n" +
		"    Filter (price > 100)\n" +
		"      Scan trades/*.parquet [price, qty] rows 0 to 100"
	if got := tree(out); got != want {
		t.Errorf("the plan came out as\n%s\nwant\n%s", got, want)
	}
}

func BenchmarkPushSlice(b *testing.B) {
	n := plan.Limit(
		plan.Project(
			plan.Limit(plan.Filter(scan, dear), 0, 100),
			[]plan.Projection{{Expr: amount, As: "amount"}},
		),
		5, 20)

	b.ReportAllocs()
	for b.Loop() {
		out, err := plan.PushSlice(n)
		if err != nil {
			b.Fatalf("PushSlice: %v", err)
		}
		sink = out
	}
}
