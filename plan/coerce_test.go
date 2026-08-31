package plan_test

import (
	"strings"
	"testing"
	"time"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// mixed is a source with a column of each type a value written in a query turns
// into something else against: a float against an integer written plainly, an
// unsigned integer that is narrower than the one a literal has on its own, and
// a timestamp counted in seconds where a Go time counts nanoseconds.
type mixed struct{}

func (mixed) Name() string { return "mixed.parquet" }

func (mixed) Schema() (dtype.Schema, error) {
	return dtype.Schema{Fields: []dtype.Field{
		{Name: "price", Type: dtype.Float64},
		{Name: "qty", Type: dtype.Int64},
		{Name: "n", Type: dtype.Uint32},
		{Name: "at", Type: dtype.Timestamp{Unit: dtype.Second, Zone: "UTC"}},
	}}, nil
}

var mixture = plan.Scan(mixed{})

// TestCoerceWritesTheTypeAValueIsUsedAt is the pass itself, read off the plan it
// produces. A plan that says what type a comparison happens at is the whole
// point, so the check is the text and not the shape.
func TestCoerceWritesTheTypeAValueIsUsedAt(t *testing.T) {
	noon := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)

	cases := []struct {
		name string
		expr *plan.Expr
		want string
	}{
		{
			"a plain number against a float column",
			plan.Compare(kernel.OpGt, price, plan.Lit(100)),
			"Filter (price > (100 as float64))",
		},
		{
			"the value on the left instead",
			plan.Compare(kernel.OpLt, plan.Lit(100), price),
			"Filter ((100 as float64) < price)",
		},
		{
			"a number against a narrower integer column",
			plan.Compare(kernel.OpGt, small, plan.Lit(100)),
			"Filter (n > (100 as uint32))",
		},
		{
			"a time against a column counted in seconds",
			plan.Compare(kernel.OpGe, at, plan.Lit(noon)),
			"Filter (at >= (2026-08-25T12:00:00Z as timestamp[s, tz=UTC]))",
		},
		{
			"arithmetic, which coerces the same way a comparison does",
			plan.Compare(kernel.OpGt, plan.Arith(kernel.OpMul, price, plan.Lit(2)), plan.Lit(10)),
			"Filter ((price * (2 as float64)) > (10 as float64))",
		},
		{
			"a value already at the type it is used at",
			plan.Compare(kernel.OpGt, qty, plan.Lit(int64(100))),
			"Filter (qty > 100)",
		},
		{
			"a float written against a float column",
			plan.Compare(kernel.OpGt, price, plan.Lit(100.0)),
			"Filter (price > 100)",
		},
		{
			"a missing value, which is missing in every type",
			plan.Compare(kernel.OpEq, price, plan.Lit(nil)),
			"Filter (price == null)",
		},
		{
			"two columns, which have nothing to tell each other",
			plan.Compare(kernel.OpGt, price, price),
			"Filter (price > price)",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out, err := plan.Coerce(plan.Filter(mixture, c.expr))
			if err != nil {
				t.Fatalf("Coerce = %v", err)
			}
			got, _, ok := strings.Cut(out.Tree(), "\n")
			if !ok {
				t.Fatalf("the plan is %q and has no scan under it", out.Tree())
			}
			if got != c.want {
				t.Errorf("Coerce gave\n  %s\nwant\n  %s", got, c.want)
			}
		})
	}
}

// TestCoerceHandsBackThePlanWhenThereIsNothingToWrite is the rule every pass
// follows, which is what lets the optimizer stop.
func TestCoerceHandsBackThePlanWhenThereIsNothingToWrite(t *testing.T) {
	cases := []struct {
		name string
		node *plan.Node
	}{
		{"a value already at its type", plan.Filter(mixture, plan.Compare(kernel.OpGt, qty, plan.Lit(int64(1))))},
		{"nothing but columns", plan.Filter(mixture, plan.Compare(kernel.OpGt, price, price))},
		{"a limit, which holds no expression", plan.Limit(mixture, 0, 10)},
		{
			"an expression that does not type at all",
			plan.Filter(mixture, plan.Compare(kernel.OpGt, plan.Col("nope"), plan.Lit(1))),
		},
		{
			"a value the column cannot hold",
			plan.Filter(mixture, plan.Compare(kernel.OpGt, small, plan.Lit(-1))),
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out, err := plan.Coerce(c.node)
			if err != nil {
				t.Fatalf("Coerce = %v", err)
			}
			if out != c.node {
				t.Errorf("Coerce rebuilt the plan as %q, want the plan it was given", out.Tree())
			}
		})
	}
}

// TestCoerceLeavesTheErrorToTheCheck is the other half of the case above. A pass
// that cannot work a type out has to leave the expression exactly as written,
// because the error a caller reads should name what they wrote.
func TestCoerceLeavesTheErrorToTheCheck(t *testing.T) {
	q := plan.Filter(mixture, plan.Compare(kernel.OpGt, small, plan.Lit(-1)))

	out, err := plan.Coerce(q)
	if err != nil {
		t.Fatalf("Coerce = %v", err)
	}
	err = out.Validate()
	if err == nil {
		t.Fatal("the plan checks out, want the value it cannot hold to be refused")
	}
	if !strings.Contains(err.Error(), "-1 does not fit in uint32") {
		t.Errorf("the error is %q, want it to be about the value as written", err)
	}
}

// TestCoerceDoesNotMoveATypeWrittenByHand is what makes the pass safe to run
// over a plan that came from somewhere else. A type already written down is an
// answer somebody gave, and a pass asks the same question the plan asks.
func TestCoerceDoesNotMoveATypeWrittenByHand(t *testing.T) {
	q := plan.Filter(mixture, plan.Compare(kernel.OpGt, price, plan.LitAs(dtype.Int64, 100)))

	out, err := plan.Coerce(q)
	if err != nil {
		t.Fatalf("Coerce = %v", err)
	}
	if out != q {
		t.Fatalf("Coerce rewrote the plan to %q, want the type as written", out.Tree())
	}
	if err := out.Validate(); err == nil {
		t.Error("the plan checks out, want the int64 against a float64 column to be refused")
	}
}

// TestCoerceAgreesWithTheTypeChecker is the anti drift test. The pass writes
// down what the checker would have worked out, so a plan it has been over has to
// type exactly as the plan it was given did. If the two ever disagree the pass
// is quietly changing the query rather than describing it.
func TestCoerceAgreesWithTheTypeChecker(t *testing.T) {
	s, err := mixed{}.Schema()
	if err != nil {
		t.Fatal(err)
	}

	exprs := []*plan.Expr{
		plan.Compare(kernel.OpGt, price, plan.Lit(100)),
		plan.Compare(kernel.OpGt, price, plan.Lit(100.5)),
		plan.Compare(kernel.OpLt, plan.Lit(100), price),
		plan.Compare(kernel.OpGt, small, plan.Lit(100)),
		plan.Compare(kernel.OpEq, qty, plan.Lit(int8(3))),
		plan.Compare(kernel.OpGe, at, plan.Lit(time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC))),
		plan.Arith(kernel.OpMul, price, plan.Lit(2)),
		plan.Arith(kernel.OpAdd, small, plan.Lit(1)),
		plan.Compare(kernel.OpGt, plan.Arith(kernel.OpMul, price, plan.Lit(2)), plan.Lit(10)),
		plan.And(
			plan.Compare(kernel.OpGt, price, plan.Lit(100)),
			plan.Compare(kernel.OpLt, small, plan.Lit(5)),
		),
		plan.Not(plan.Compare(kernel.OpGt, price, plan.Lit(100))),
		plan.IsNull(plan.Arith(kernel.OpSub, price, plan.Lit(1))),
		plan.Cast(dtype.Float64, plan.Arith(kernel.OpAdd, small, plan.Lit(1))),
	}

	for _, e := range exprs {
		t.Run(e.String(), func(t *testing.T) {
			was, err := plan.TypeOf(e, s)
			if err != nil {
				t.Fatalf("TypeOf(%s) = %v, the expression has to type before the pass", e, err)
			}

			out, err := plan.Coerce(plan.Project(mixture, []plan.Projection{{Expr: e, As: "out"}}))
			if err != nil {
				t.Fatalf("Coerce = %v", err)
			}
			got := out.Columns()[0].Expr

			now, err := plan.TypeOf(got, s)
			if err != nil {
				t.Fatalf("TypeOf(%s) = %v, the pass turned a good expression into a bad one", got, err)
			}
			if !dtype.Equal(was, now) {
				t.Errorf("%s was a %s and %s is a %s, the pass has to keep the type", e, was, got, now)
			}
		})
	}
}

// TestCoerceRunsWithTheOtherPasses is the pass where it actually runs. It has to
// settle, it has to say it changed the plan, and the plan it leaves has to be
// the one an explain shows.
func TestCoerceRunsWithTheOtherPasses(t *testing.T) {
	q := plan.Limit(plan.Filter(mixture, plan.Compare(kernel.OpGt, price, plan.Lit(100))), 0, 10)

	text, err := plan.Explain(q, plan.Passes()...)
	if err != nil {
		t.Fatalf("Explain = %v", err)
	}
	if !strings.Contains(text, "(price > (100 as float64))") {
		t.Errorf("the explain is\n%s\nwant the comparison to say what type it happens at", text)
	}
	if !strings.Contains(text, "type coercion") {
		t.Errorf("the explain is\n%s\nwant it to name the pass that changed the plan", text)
	}
}

// TestCoerceIsIdempotent is the property the fixpoint depends on. Running the
// pass over its own answer has to find nothing, or the optimizer never stops.
func TestCoerceIsIdempotent(t *testing.T) {
	q := plan.Filter(mixture, plan.And(
		plan.Compare(kernel.OpGt, price, plan.Lit(100)),
		plan.Compare(kernel.OpLt, small, plan.Lit(5)),
	))

	once, err := plan.Coerce(q)
	if err != nil {
		t.Fatalf("Coerce = %v", err)
	}
	if once == q {
		t.Fatal("Coerce found nothing to write, want both values written down")
	}

	twice, err := plan.Coerce(once)
	if err != nil {
		t.Fatalf("Coerce = %v", err)
	}
	if twice != once {
		t.Errorf("Coerce rewrote its own answer to %q, want it to find nothing the second time", twice.Tree())
	}
}

// TestCoerceKeepsTheNameOfAProjection is the same rule constant folding follows.
// A projection with no name of its own is called after the expression it holds,
// so writing a type into that expression would rename the column it produces.
func TestCoerceKeepsTheNameOfAProjection(t *testing.T) {
	e := plan.Arith(kernel.OpMul, price, plan.Lit(2))
	q := plan.Project(mixture, []plan.Projection{{Expr: e}})

	out, err := plan.Coerce(q)
	if err != nil {
		t.Fatalf("Coerce = %v", err)
	}
	if out == q {
		t.Fatal("Coerce found nothing to write, want the two written down")
	}

	s, err := out.Schema()
	if err != nil {
		t.Fatalf("Schema = %v", err)
	}
	if got := s.Names(); len(got) != 1 || got[0] != e.String() {
		t.Errorf("the projection is called %q, want %q", got, e.String())
	}
}

// BenchmarkCoerce is what a query pays for the plan to say what type each value
// in it is used at. It is paid once per query, and both halves are worth having:
// the one that finds something is the cost of the rewrite and the one that finds
// nothing is what every query that was already explicit pays.
func BenchmarkCoerce(b *testing.B) {
	cases := []struct {
		name string
		node *plan.Node
	}{
		{
			"two values to write down",
			plan.Filter(mixture, plan.And(
				plan.Compare(kernel.OpGt, price, plan.Lit(100)),
				plan.Compare(kernel.OpLt, small, plan.Lit(5)),
			)),
		},
		{
			"nothing to write down",
			plan.Filter(mixture, plan.And(
				plan.Compare(kernel.OpGt, price, plan.Lit(100.0)),
				plan.Compare(kernel.OpLt, qty, plan.Lit(int64(5))),
			)),
		},
	}

	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				out, err := plan.Coerce(c.node)
				if err != nil {
					b.Fatal(err)
				}
				sink = out
			}
		})
	}
}
