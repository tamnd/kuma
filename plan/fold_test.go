package plan_test

// The tests for the constant folding pass. What was worked out is read off the
// printed plan, since a value the pass settled is a value written in the plan
// where a step used to be.

import (
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// narrow is a source with a small integer column, which is where the type of a
// step depends on what is beside it: 5 next to an int8 column is an int8 and 2
// plus 3 on their own are an int64.
type narrow struct{}

func (narrow) Name() string { return "narrow.csv" }

func (narrow) Schema() (dtype.Schema, error) {
	return dtype.Schema{Fields: []dtype.Field{
		{Name: "count", Type: dtype.Int8},
		{Name: "flag", Type: dtype.Bool},
	}}, nil
}

// The values these tests fold. They are built once because two expressions that
// say the same thing are the same expression, so a test that checks what became
// of one is checking the one it wrote.
var (
	count    = plan.Col("count")
	flag     = plan.Col("flag")
	narrowed = plan.Scan(narrow{})

	// A limit written as a sum, which is the shape a caller writes when the
	// number means something: a hundred with a tenth on top of it.
	sum = plan.Arith(kernel.OpAdd, plan.Lit(100.0), plan.Lit(10.0))

	// A condition that is true whatever the data says, which is what a
	// generated query leaves behind when the part that varied was not filled
	// in.
	always = plan.Compare(kernel.OpLt, plan.Lit(int64(1)), plan.Lit(int64(2)))
)

// settled runs the pass and fails the test if it reports anything, since none
// of the plans it is given here are wrong.
func settled(t *testing.T, n *plan.Node) *plan.Node {
	t.Helper()
	out, err := plan.Fold(n)
	if err != nil {
		t.Fatalf("Fold: %v", err)
	}
	return out
}

func TestFoldWorksOutWhatTheDataDoesNotDecide(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
		want string
	}{
		{
			name: "a sum of two values is the value",
			plan: plan.Filter(scan, plan.Compare(kernel.OpGt, price, sum)),
			want: "Filter (price > 110)\n  Scan trades/*.parquet",
		},
		{
			name: "a comparison of two values is the answer to it",
			plan: plan.Filter(scan, plan.And(dear, always)),
			want: "Filter (price > 100)\n  Scan trades/*.parquet",
		},
		{
			name: "an or with a condition that never holds is the other side",
			plan: plan.Filter(scan, plan.Or(dear, plan.Lit(false))),
			want: "Filter (price > 100)\n  Scan trades/*.parquet",
		},
		{
			name: "an and with a condition that always holds is the other side",
			plan: plan.Filter(scan, plan.And(plan.Lit(true), dear)),
			want: "Filter (price > 100)\n  Scan trades/*.parquet",
		},
		{
			name: "a not of a value is the other value",
			plan: plan.Filter(scan, plan.Or(dear, plan.Not(plan.Lit(true)))),
			want: "Filter (price > 100)\n  Scan trades/*.parquet",
		},
		{
			name: "an and over a column that already holds the conditions",
			plan: plan.Filter(narrowed, plan.And(flag, plan.Lit(true))),
			want: "Filter flag\n  Scan narrow.csv",
		},
		{
			name: "a not of a not is the condition itself",
			plan: plan.Filter(scan, plan.Not(plan.Not(dear))),
			want: "Filter (price > 100)\n  Scan trades/*.parquet",
		},
		{
			name: "a value is never missing",
			plan: plan.Filter(scan, plan.And(dear, plan.IsNotNull(plan.Lit(1.0)))),
			want: "Filter (price > 100)\n  Scan trades/*.parquet",
		},
		{
			name: "a cast to the type the column already has is nothing at all",
			plan: plan.Filter(scan, plan.Compare(kernel.OpGt,
				plan.Cast(dtype.Float64, price), plan.Lit(100.0))),
			want: "Filter (price > 100)\n  Scan trades/*.parquet",
		},
		{
			name: "a cast of a value is the value in the other type",
			plan: plan.Filter(scan, plan.Compare(kernel.OpGt,
				price, plan.Cast(dtype.Float64, plan.Lit(int64(100))))),
			want: "Filter (price > 100)\n  Scan trades/*.parquet",
		},
		{
			name: "inside a projection, which keeps the name it had",
			plan: plan.Project(scan, []plan.Projection{
				{Expr: plan.Cast(dtype.Float64, price)},
			}),
			want: "Project price as (price as float64)\n  Scan trades/*.parquet",
		},
		{
			name: "inside a projection that was named to begin with",
			plan: plan.Project(scan, []plan.Projection{
				{Expr: plan.Arith(kernel.OpMul, price, sum), As: "total"},
			}),
			want: "Project (price * 110) as total\n  Scan trades/*.parquet",
		},
		{
			name: "inside a sort, which has no name to keep",
			plan: plan.Sort(scan, []plan.SortKey{
				{Expr: plan.Arith(kernel.OpMul, price, sum)},
			}),
			want: "Sort by (price * 110)\n  Scan trades/*.parquet",
		},
		{
			name: "inside a distinct, which produces its input either way",
			plan: plan.Distinct(scan, []*plan.Expr{plan.Cast(dtype.Float64, price)}),
			want: "Distinct by price\n  Scan trades/*.parquet",
		},
		{
			name: "inside an aggregation, which keeps the name it had",
			plan: plan.Aggregate(scan, []*plan.Expr{symbol}, []plan.Agg{
				{Func: plan.AggSum, Expr: plan.Cast(dtype.Float64, price)},
			}),
			want: "Aggregate by symbol: Sum(price) as (price as float64)\n  Scan trades/*.parquet",
		},
		{
			name: "under an operator that holds no expression of its own",
			plan: plan.Limit(plan.Filter(scan, plan.Compare(kernel.OpGt, price, sum)), 0, 20),
			want: "Limit 20\n  Filter (price > 110)\n    Scan trades/*.parquet",
		},
		{
			name: "on both sides of a join",
			plan: plan.Join(
				plan.Filter(scan, plan.Compare(kernel.OpGt, price, sum)),
				plan.Filter(quoted, plan.Or(wide, plan.Lit(false))),
				onSymbol, kernel.InnerJoin,
			),
			want: "Join inner on symbol\n" +
				"  Filter (price > 110)\n" +
				"    Scan trades/*.parquet\n" +
				"  Filter (bid < 1)\n" +
				"    Scan quotes/*.parquet",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := tree(settled(t, c.plan)); got != c.want {
				t.Errorf("the plan folded to\n%s\nwant\n%s", got, c.want)
			}
		})
	}
}

func TestFoldLeavesAPlanItFoundNothingIn(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
	}{
		{
			name: "a condition over a column and a value, which is already as short as it gets",
			plan: plan.Filter(scan, dear),
		},
		{
			name: "a cast that does change the type",
			plan: plan.Project(scan, []plan.Projection{{Expr: amount, As: "amount"}}),
		},
		{
			name: "an and with a condition that never holds, which needs an operator that is not there yet",
			plan: plan.Filter(scan, plan.And(dear, plan.Lit(false))),
		},
		{
			name: "an or with a condition that always holds, for the same reason",
			plan: plan.Filter(scan, plan.Or(dear, plan.Lit(true))),
		},
		{
			name: "a division by zero, which is an error when the query runs and not before",
			plan: plan.Filter(scan, plan.Compare(kernel.OpGt, qty,
				plan.Arith(kernel.OpDiv, plan.Lit(int64(1)), plan.Lit(int64(0))))),
		},
		{
			name: "a group key, which is called after itself",
			plan: plan.Aggregate(scan, []*plan.Expr{plan.Cast(dtype.Float64, price)},
				[]plan.Agg{{Func: plan.AggSize}}),
		},
		{
			name: "a scan on its own",
			plan: scan,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if out := settled(t, c.plan); out != c.plan {
				t.Errorf("the pass rebuilt a plan it had nothing to do to:\n%s", tree(out))
			}
		})
	}
}

// TestFoldWillNotMakeAPlanThatDoesNotTypeType is the rule that keeps the pass
// honest. A value takes its type from what it is used with, so working two of
// them out can change what the step above them sees, and a plan that is wrong
// has to stay wrong rather than start working by accident.
func TestFoldWillNotMakeAPlanThatDoesNotTypeType(t *testing.T) {
	// An int8 column next to an int64 is a plan kuma refuses, since it does not
	// widen a column on its own. Folded to a 5 it would be an int8 and the plan
	// would run, which is a different query and not a faster one.
	written := plan.Arith(kernel.OpAdd, count, plan.Arith(kernel.OpAdd, plan.Lit(2), plan.Lit(3)))
	q := plan.Filter(narrowed, plan.Compare(kernel.OpGt, written, plan.Lit(0)))

	if _, err := q.Schema(); err == nil {
		t.Fatal("the plan this test is about types, so it is not the plan this test is about")
	}
	if out := settled(t, q); out != q {
		t.Errorf("the pass folded a plan that does not type:\n%s", tree(out))
	}
}

// TestFoldSettles is the rule every pass has to keep, since the optimizer runs
// them over and over until none of them moves anything.
func TestFoldSettles(t *testing.T) {
	q := plan.Filter(scan, plan.And(plan.Compare(kernel.OpGt, price, sum), always))

	once := settled(t, q)
	if once == q {
		t.Fatal("the pass found nothing to fold, so there is nothing to settle")
	}
	if twice := settled(t, once); twice != once {
		t.Errorf("the pass folded its own output:\n%s", tree(twice))
	}
}

// TestFoldOfAValueInsideAValue is the case that takes more than one step, since
// the sum has to be worked out before the product it is inside of can be.
func TestFoldOfAValueInsideAValue(t *testing.T) {
	inner := plan.Arith(kernel.OpMul, sum, plan.Lit(2.0))
	q := plan.Filter(scan, plan.Compare(kernel.OpGt, price, inner))

	want := "Filter (price > 220)\n  Scan trades/*.parquet"
	if got := tree(settled(t, q)); got != want {
		t.Errorf("the plan folded to\n%s\nwant\n%s", got, want)
	}
}

// TestFoldKeepsWhatThePlanProduces is the check that folding is not a rename.
// The columns a query answers with are named by the caller or named after what
// was written, and a pass that shortens what was written must not shorten the
// name with it.
func TestFoldKeepsWhatThePlanProduces(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
	}{
		{
			name: "a projection with no name of its own",
			plan: plan.Project(scan, []plan.Projection{
				{Expr: plan.Cast(dtype.Float64, price)},
				{Expr: symbol},
			}),
		},
		{
			name: "a projection the caller named",
			plan: plan.Project(scan, []plan.Projection{
				{Expr: plan.Arith(kernel.OpMul, price, sum), As: "total"},
			}),
		},
		{
			name: "an aggregation with no name of its own",
			plan: plan.Aggregate(scan, []*plan.Expr{symbol}, []plan.Agg{
				{Func: plan.AggSum, Expr: plan.Cast(dtype.Float64, price)},
				{Func: plan.AggSize},
			}),
		},
		{
			name: "a sort, which produces what it was given",
			plan: plan.Sort(scan, []plan.SortKey{{Expr: plan.Cast(dtype.Float64, price)}}),
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			before, err := c.plan.Schema()
			if err != nil {
				t.Fatalf("the plan does not type: %v", err)
			}
			after, err := settled(t, c.plan).Schema()
			if err != nil {
				t.Fatalf("the folded plan does not type: %v", err)
			}
			// The check is on the names and the types rather than on the whole
			// schema. A cast a column did not need was also the plan saying the
			// column might come out missing, and taking the cast away takes
			// that away with it, which is the plan being right rather than the
			// query being different.
			if !slices.Equal(after.Names(), before.Names()) {
				t.Fatalf("the folded plan produces %v, want %v", after.Names(), before.Names())
			}
			for i, f := range before.Fields {
				if !dtype.Equal(after.Fields[i].Type, f.Type) {
					t.Errorf("%s came out a %s, want a %s", f.Name, after.Fields[i].Type, f.Type)
				}
			}
		})
	}
}

// TestFoldOfAPlanThatIsWrong is the pass over a plan it cannot work out the
// columns of, which comes back as the error rather than as the plan as far as
// it got.
func TestFoldOfAPlanThatIsWrong(t *testing.T) {
	q := plan.Filter(plan.Scan(missing{}), dear)

	if _, err := plan.Fold(q); err == nil {
		t.Fatal("Fold gave no error over a source that cannot say what it holds")
	} else if !strings.Contains(err.Error(), "gone") {
		t.Errorf("Fold said %q, want it to name the source", err)
	}
}

// TestFoldWithTheOtherPasses is the pass where it actually runs, since a value
// worked out here is a value the pushdowns then move.
func TestFoldWithTheOtherPasses(t *testing.T) {
	q := plan.Limit(
		plan.Filter(
			plan.Project(scan, []plan.Projection{
				{Expr: price},
				{Expr: plan.Arith(kernel.OpMul, price, sum), As: "total"},
			}),
			plan.And(dear, always),
		),
		0, 20,
	)

	out, err := plan.Optimize(q, plan.Passes()...)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	want := "Project price, (price * 110) as total\n" +
		"  Limit 20\n" +
		"    Filter (price > 100)\n" +
		"      Scan trades/*.parquet [price]"
	if got := tree(out); got != want {
		t.Errorf("the query optimized to\n%s\nwant\n%s", got, want)
	}
}

func BenchmarkFold(b *testing.B) {
	q := plan.Filter(scan, plan.And(plan.Compare(kernel.OpGt, price, sum), always))

	b.ReportAllocs()
	for b.Loop() {
		out, err := plan.Fold(q)
		if err != nil {
			b.Fatal(err)
		}
		sink = out
	}
}
