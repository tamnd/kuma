package plan_test

// The tests for the common subexpression pass. What was hoisted is read off the
// printed plan, since the whole of what the pass does is move an expression into
// a projection underneath and leave a column name behind.

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// odd is a table with columns named after expressions, which is a name a
// parquet file is allowed to have and a name the pass has to leave room for.
// Both the product and the cast inside it are there, since the pass looks
// inside a value it cannot hold for one it can.
type odd struct{}

func (odd) Name() string { return "odd/*.parquet" }

func (odd) Schema() (dtype.Schema, error) {
	return dtype.Schema{Fields: []dtype.Field{
		{Name: "symbol", Type: dtype.String},
		{Name: "price", Type: dtype.Float64},
		{Name: "qty", Type: dtype.Int64},
		{Name: "(qty as float64)", Type: dtype.Float64},
		{Name: "(price * (qty as float64))", Type: dtype.Float64},
	}}, nil
}

// The values these tests share. They are built once because two expressions
// that say the same thing are the same expression, which is the fact the pass
// is built on.
var (
	taxed   = plan.Arith(kernel.OpMul, amount, plan.Lit(1.1))
	doubled = plan.Arith(kernel.OpMul, taxed, plan.Lit(2.0))
	oddish  = plan.Scan(odd{})
)

// hoisted runs the pass and fails the test if it reports anything, since none
// of the plans it is given here are wrong.
func hoisted(t *testing.T, n *plan.Node) *plan.Node {
	t.Helper()
	out, err := plan.HoistCommon(n)
	if err != nil {
		t.Fatalf("HoistCommon: %v", err)
	}
	return out
}

func TestHoistCommonPullsARepeatedValueOut(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
		want string
	}{
		{
			name: "a value written in two columns is worked out in one underneath",
			plan: plan.Project(scan, []plan.Projection{
				{Expr: amount, As: "amount"},
				{Expr: taxed, As: "taxed"},
			}),
			want: "Project (price * (qty as float64)) as amount, ((price * (qty as float64)) * 1.1) as taxed\n" +
				"  Project (price * (qty as float64))\n" +
				"    Scan trades/*.parquet",
		},
		{
			name: "the columns the rewritten expressions still read come with it",
			plan: plan.Project(scan, []plan.Projection{
				{Expr: symbol},
				{Expr: amount, As: "amount"},
				{Expr: taxed, As: "taxed"},
			}),
			want: "Project symbol, (price * (qty as float64)) as amount, ((price * (qty as float64)) * 1.1) as taxed\n" +
				"  Project symbol, (price * (qty as float64))\n" +
				"    Scan trades/*.parquet",
		},
		{
			name: "the outermost repeat is the one that moves, not the ones inside it",
			plan: plan.Project(scan, []plan.Projection{
				{Expr: taxed, As: "taxed"},
				{Expr: doubled, As: "doubled"},
			}),
			want: "Project ((price * (qty as float64)) * 1.1) as taxed, (((price * (qty as float64)) * 1.1) * 2) as doubled\n" +
				"  Project ((price * (qty as float64)) * 1.1)\n" +
				"    Scan trades/*.parquet",
		},
		{
			// The product is repeated three times and the taxed amount twice, so
			// both come out, and the taxed amount still works the product out
			// for itself. Running the pass again is what fixes that, which is
			// what the optimizer does anyway and what the next test is about.
			name: "two repeats, one inside the other, both come out in one go",
			plan: plan.Project(scan, []plan.Projection{
				{Expr: amount, As: "amount"},
				{Expr: taxed, As: "taxed"},
				{Expr: doubled, As: "doubled"},
			}),
			want: "Project (price * (qty as float64)) as amount, ((price * (qty as float64)) * 1.1) as taxed, " +
				"(((price * (qty as float64)) * 1.1) * 2) as doubled\n" +
				"  Project (price * (qty as float64)), ((price * (qty as float64)) * 1.1)\n" +
				"    Scan trades/*.parquet",
		},
		{
			name: "an aggregation that reads one value twice reads it once",
			plan: plan.Aggregate(scan, []*plan.Expr{symbol}, []plan.Agg{
				{Func: plan.AggSum, Expr: amount, As: "total"},
				{Func: plan.AggMax, Expr: amount, As: "top"},
			}),
			want: "Aggregate by symbol: Sum((price * (qty as float64))) as total, Max((price * (qty as float64))) as top\n" +
				"  Project symbol, (price * (qty as float64))\n" +
				"    Scan trades/*.parquet",
		},
		{
			name: "a group key and an aggregation over the same value count as a repeat",
			plan: plan.Aggregate(scan, []*plan.Expr{amount}, []plan.Agg{
				{Func: plan.AggSum, Expr: amount, As: "total"},
			}),
			want: "Aggregate by (price * (qty as float64)): Sum((price * (qty as float64))) as total\n" +
				"  Project (price * (qty as float64))\n" +
				"    Scan trades/*.parquet",
		},
		{
			name: "a size counts rows and shares nothing, so the rest still hoists",
			plan: plan.Aggregate(scan, []*plan.Expr{symbol}, []plan.Agg{
				{Func: plan.AggCount, As: "n"},
				{Func: plan.AggSum, Expr: amount, As: "total"},
				{Func: plan.AggMin, Expr: amount, As: "low"},
			}),
			want: "Aggregate by symbol: Count() as n, Sum((price * (qty as float64))) as total, " +
				"Min((price * (qty as float64))) as low\n" +
				"  Project symbol, (price * (qty as float64))\n" +
				"    Scan trades/*.parquet",
		},
		{
			name: "an operator deeper in the plan is reached too",
			plan: plan.Sort(plan.Project(scan, []plan.Projection{
				{Expr: amount, As: "amount"},
				{Expr: taxed, As: "taxed"},
			}), []plan.SortKey{{Expr: plan.Col("amount")}}),
			want: "Sort by amount\n" +
				"  Project (price * (qty as float64)) as amount, ((price * (qty as float64)) * 1.1) as taxed\n" +
				"    Project (price * (qty as float64))\n" +
				"      Scan trades/*.parquet",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := tree(hoisted(t, c.plan)); got != c.want {
				t.Errorf("HoistCommon gave\n%s\nwant\n%s", got, c.want)
			}
		})
	}
}

func TestHoistCommonLeavesAPlanItFoundNothingIn(t *testing.T) {
	cases := []struct {
		name string
		plan *plan.Node
	}{
		{
			name: "a value written once is worked out once already",
			plan: plan.Project(scan, []plan.Projection{
				{Expr: symbol},
				{Expr: amount, As: "amount"},
			}),
		},
		{
			name: "a column written twice is a column both times",
			plan: plan.Project(scan, []plan.Projection{
				{Expr: price, As: "a"},
				{Expr: price, As: "b"},
			}),
		},
		{
			name: "a literal written twice costs nothing to write twice",
			plan: plan.Project(scan, []plan.Projection{
				{Expr: plan.Lit(1.0), As: "a"},
				{Expr: plan.Lit(1.0), As: "b"},
			}),
		},
		{
			name: "a filter has one expression, so it has nothing to share it with",
			plan: plan.Filter(scan, plan.And(dear, dear)),
		},
		{
			name: "a plan of one scan has no expression at all",
			plan: scan,
		},
		{
			name: "a repeat whose name is a column of the input would hide that column",
			plan: plan.Project(oddish, []plan.Projection{
				{Expr: amount, As: "amount"},
				{Expr: taxed, As: "taxed"},
			}),
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if out := hoisted(t, c.plan); out != c.plan {
				t.Errorf("HoistCommon rebuilt a plan it found nothing in:\n%s", tree(out))
			}
		})
	}
}

func TestHoistCommonSettles(t *testing.T) {
	n := plan.Project(scan, []plan.Projection{
		{Expr: amount, As: "amount"},
		{Expr: taxed, As: "taxed"},
	})

	once := hoisted(t, n)
	if once == n {
		t.Fatal("HoistCommon found nothing in a plan that writes one value twice")
	}
	if again := hoisted(t, once); again != once {
		t.Errorf("HoistCommon of a plan it had already done gave\n%s\nwant the plan it was given", tree(again))
	}
}

func TestHoistCommonRunToFixpointFinishesANestedRepeat(t *testing.T) {
	// One round takes both repeats out and leaves the outer one working the
	// inner one out for itself. The round after that takes the inner one out
	// from under it, and the round after that finds nothing, which is where the
	// product is multiplied once for the three columns that read it.
	n := plan.Project(scan, []plan.Projection{
		{Expr: amount, As: "amount"},
		{Expr: taxed, As: "taxed"},
		{Expr: doubled, As: "doubled"},
	})

	out, err := plan.Optimize(n, plan.Pass{Name: "common subexpression elimination", Rewrite: plan.HoistCommon})
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	want := "Project (price * (qty as float64)) as amount, ((price * (qty as float64)) * 1.1) as taxed, " +
		"(((price * (qty as float64)) * 1.1) * 2) as doubled\n" +
		"  Project (price * (qty as float64)), ((price * (qty as float64)) * 1.1)\n" +
		"    Project (price * (qty as float64))\n" +
		"      Scan trades/*.parquet"
	if got := tree(out); got != want {
		t.Errorf("Optimize gave\n%s\nwant\n%s", got, want)
	}
}

func TestHoistCommonKeepsWhatThePlanProduces(t *testing.T) {
	cases := []*plan.Node{
		plan.Project(scan, []plan.Projection{
			{Expr: amount},
			{Expr: taxed},
		}),
		plan.Project(scan, []plan.Projection{
			{Expr: symbol},
			{Expr: amount, As: "amount"},
			{Expr: taxed, As: "taxed"},
		}),
		plan.Aggregate(scan, []*plan.Expr{symbol}, []plan.Agg{
			{Func: plan.AggSum, Expr: amount, As: "total"},
			{Func: plan.AggMax, Expr: amount, As: "top"},
		}),
		plan.Aggregate(scan, []*plan.Expr{amount}, []plan.Agg{
			{Func: plan.AggSum, Expr: amount, As: "total"},
		}),
	}

	for _, n := range cases {
		t.Run(n.String(), func(t *testing.T) {
			before, err := n.Schema()
			if err != nil {
				t.Fatalf("Schema of the plan as written: %v", err)
			}
			after, err := hoisted(t, n).Schema()
			if err != nil {
				t.Fatalf("Schema of the plan the pass gave: %v", err)
			}
			if !before.Equal(after) {
				t.Errorf("the plan produces %v, want %v", after.Names(), before.Names())
			}
		})
	}
}

func TestHoistCommonOfAPlanThatIsWrong(t *testing.T) {
	n := plan.Project(plan.Scan(missing{}), []plan.Projection{
		{Expr: amount, As: "amount"},
		{Expr: taxed, As: "taxed"},
	})

	_, err := plan.HoistCommon(n)
	if err == nil {
		t.Fatal("HoistCommon of a plan over a table that is not there gave no error")
	}
	if !strings.Contains(err.Error(), "no such file") {
		t.Errorf("HoistCommon said %q, want the error the table gave", err)
	}
}

func TestHoistCommonWithTheOtherPasses(t *testing.T) {
	// A query that filters, writes one value in two columns and asks for the
	// first twenty rows. Each pass has something to do and none of them undoes
	// another: the filter sinks to the scan, the limit stops over the filter,
	// the repeat comes out into a projection of its own, and the scan is left
	// reading the two columns that projection needs.
	n := plan.Limit(plan.Filter(plan.Project(scan, []plan.Projection{
		{Expr: price},
		{Expr: amount, As: "amount"},
		{Expr: taxed, As: "taxed"},
	}), dear), 0, 20)

	out, err := plan.Optimize(n, plan.Passes()...)
	if err != nil {
		t.Fatalf("Optimize: %v", err)
	}

	want := "Project price, (price * (qty as float64)) as amount, ((price * (qty as float64)) * 1.1) as taxed\n" +
		"  Project price, (price * (qty as float64))\n" +
		"    Limit 20\n" +
		"      Filter (price > 100)\n" +
		"        Scan trades/*.parquet [price, qty]"
	if got := tree(out); got != want {
		t.Errorf("Optimize gave\n%s\nwant\n%s", got, want)
	}
}

func BenchmarkHoistCommon(b *testing.B) {
	n := plan.Project(scan, []plan.Projection{
		{Expr: symbol},
		{Expr: amount, As: "amount"},
		{Expr: taxed, As: "taxed"},
		{Expr: doubled, As: "doubled"},
	})

	b.ReportAllocs()
	for b.Loop() {
		out, err := plan.HoistCommon(n)
		if err != nil {
			b.Fatal(err)
		}
		sink = out
	}
}
