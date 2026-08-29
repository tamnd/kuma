package plan_test

import (
	"testing"

	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

func TestTheAggregationsAreNamed(t *testing.T) {
	funcs := []plan.AggFunc{
		plan.AggSum, plan.AggMean, plan.AggMin, plan.AggMax, plan.AggCount,
		plan.AggSize, plan.AggFirst, plan.AggLast, plan.AggVar, plan.AggStd,
		plan.AggMedian, plan.AggQuantile, plan.AggNUnique,
	}
	want := []string{
		"Sum", "Mean", "Min", "Max", "Count", "Size", "First", "Last",
		"Var", "Std", "Median", "Quantile", "NUnique",
	}

	for i, f := range funcs {
		if got := f.String(); got != want[i] {
			t.Errorf("aggregation %d is called %q, want %q", i, got, want[i])
		}
	}
}

func TestAnAggregationPrintsItsArguments(t *testing.T) {
	cases := []struct {
		agg  plan.Agg
		want string
	}{
		{plan.Agg{Func: plan.AggSum, Expr: qty}, "Sum(qty)"},
		{plan.Agg{Func: plan.AggSum, Expr: qty, As: "total"}, "Sum(qty) as total"},
		{plan.Agg{Func: plan.AggSize}, "Size()"},
		{plan.Agg{Func: plan.AggVar, Expr: price, DDoF: 1}, "Var(price, 1)"},
		{plan.Agg{Func: plan.AggStd, Expr: price, DDoF: 0}, "Std(price, 0)"},
		{
			plan.Agg{Func: plan.AggQuantile, Expr: price, Q: 0.99, How: kernel.Linear},
			"Quantile(price, 0.99, linear)",
		},
	}

	for _, c := range cases {
		if got := c.agg.String(); got != c.want {
			t.Errorf("the aggregation reads %q, want %q", got, c.want)
		}
	}
}

// TestAnAggregationIsNamedAfterWhatItReads is what makes a query readable when
// nobody says otherwise, and it is the same rule the eager frame follows.
func TestAnAggregationIsNamedAfterWhatItReads(t *testing.T) {
	cases := []struct {
		agg  plan.Agg
		want string
	}{
		{plan.Agg{Func: plan.AggSum, Expr: qty}, "qty"},
		{plan.Agg{Func: plan.AggSum, Expr: qty, As: "total"}, "total"},
		{plan.Agg{Func: plan.AggSize}, "size"},
		{plan.Agg{Func: plan.AggSize, As: "rows"}, "rows"},
		{plan.Agg{Func: plan.AggMean, Expr: plan.Arith(kernel.OpMul, price, qty)}, "(price * qty)"},
	}

	for _, c := range cases {
		if got := c.agg.Name(); got != c.want {
			t.Errorf("%s is called %q, want %q", c.agg, got, c.want)
		}
	}
}
