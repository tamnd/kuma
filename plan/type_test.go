package plan_test

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// typed is a schema with a column of each kind these tests need: text, a float,
// an integer that is not the one a literal has on its own, a boolean, a point
// in time and some bytes.
var typed = dtype.Schema{Fields: []dtype.Field{
	{Name: "symbol", Type: dtype.String},
	{Name: "price", Type: dtype.Float64},
	{Name: "qty", Type: dtype.Int64},
	{Name: "n", Type: dtype.Uint32},
	{Name: "live", Type: dtype.Bool},
	{Name: "at", Type: dtype.Timestamp{Unit: dtype.Second, Zone: "UTC"}},
	{Name: "blob", Type: dtype.Binary},
}}

var (
	small = plan.Col("n")
	live  = plan.Col("live")
	at    = plan.Col("at")
	blob  = plan.Col("blob")
)

func TestTypeOfEveryStep(t *testing.T) {
	seconds := dtype.Timestamp{Unit: dtype.Second, Zone: "UTC"}

	cases := []struct {
		expr *plan.Expr
		want dtype.DataType
	}{
		{price, dtype.Float64},
		{plan.Lit(100.0), dtype.Float64},
		{plan.Lit(nil), dtype.Null},
		{plan.Compare(kernel.OpGt, price, plan.Lit(100.0)), dtype.Bool},
		{plan.Arith(kernel.OpMul, price, price), dtype.Float64},
		{plan.Arith(kernel.OpAdd, qty, plan.Lit(1)), dtype.Int64},
		{plan.And(live, plan.Lit(true)), dtype.Bool},
		{plan.Or(live, plan.Compare(kernel.OpLt, price, plan.Lit(1.0))), dtype.Bool},
		{plan.Not(live), dtype.Bool},
		{plan.IsNull(symbol), dtype.Bool},
		{plan.IsNotNull(blob), dtype.Bool},
		{plan.Cast(dtype.Int32, price), dtype.Int32},
		{plan.Compare(kernel.OpEq, at, plan.Lit(time.Unix(0, 0))), dtype.Bool},
		{at, seconds},
	}

	for _, c := range cases {
		got, err := plan.TypeOf(c.expr, typed)
		if err != nil {
			t.Errorf("%s: %v", c.expr, err)
			continue
		}
		if !dtype.Equal(got, c.want) {
			t.Errorf("%s is a %s, want a %s", c.expr, got, c.want)
		}
	}
}

// TestALiteralTakesTheColumnType is the rule that keeps a comparison against a
// narrow column a comparison in that column's own type, since widening a whole
// column to suit the number somebody typed is the expensive way round.
func TestALiteralTakesTheColumnType(t *testing.T) {
	cases := []struct {
		expr *plan.Expr
		want dtype.DataType
	}{
		{plan.Arith(kernel.OpAdd, small, plan.Lit(1)), dtype.Uint32},
		{plan.Arith(kernel.OpAdd, plan.Lit(1), small), dtype.Uint32},
		{plan.Arith(kernel.OpAdd, small, plan.Lit(nil)), dtype.Uint32},
		{plan.Arith(kernel.OpAdd, plan.Lit(1), plan.Lit(2)), dtype.Int64},
	}

	for _, c := range cases {
		got, err := plan.TypeOf(c.expr, typed)
		if err != nil {
			t.Errorf("%s: %v", c.expr, err)
			continue
		}
		if !dtype.Equal(got, c.want) {
			t.Errorf("%s is a %s, want a %s", c.expr, got, c.want)
		}
	}
}

// TestALiteralHasToFitTheColumnAndNotJustItsType is the other half of the rule
// above. Taking the column's type is what makes the comparison cheap, and it is
// also what a value outside that type falls foul of, so the value is checked
// here rather than on the first row of the first file.
func TestALiteralHasToFitTheColumnAndNotJustItsType(t *testing.T) {
	cases := []struct {
		name string
		expr *plan.Expr
		fits bool
	}{
		{"the top of the column type", plan.Compare(kernel.OpGt, small, plan.Lit(4294967295)), true},
		{"one past the top", plan.Compare(kernel.OpGt, small, plan.Lit(4294967296)), false},
		{"below the bottom", plan.Compare(kernel.OpGt, small, plan.Lit(-1)), false},
		{"the literal on the left", plan.Compare(kernel.OpLt, plan.Lit(-1), small), false},
		{"under arithmetic", plan.Arith(kernel.OpAdd, small, plan.Lit(-1)), false},
		{"a number the int64 column holds", plan.Compare(kernel.OpGt, qty, plan.Lit(-1)), true},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := plan.TypeOf(c.expr, typed)
			if c.fits {
				if err != nil {
					t.Fatalf("%s: %v", c.expr, err)
				}
				return
			}
			if err == nil {
				t.Fatalf("%s was checked and passed", c.expr)
			}
			if !errors.Is(err, plan.ErrWrongType) {
				t.Errorf("the error is %v, want one errors.Is finds ErrWrongType through", err)
			}
			if !strings.Contains(err.Error(), "cast the column or write a value it holds") {
				t.Errorf("the error is %q, want it to say what to do about it", err)
			}
		})
	}
}

// TestATimeLiteralTakesTheColumnUnit is the same rule for a time, which is the
// one value with no column type of its own to coerce.
func TestATimeLiteralTakesTheColumnUnit(t *testing.T) {
	when := plan.Lit(time.Date(2026, 8, 25, 9, 30, 0, 0, time.UTC))

	got, err := plan.LiteralTypeAgainst(when.Value(), dtype.Timestamp{Unit: dtype.Second, Zone: "UTC"})
	if err != nil {
		t.Fatalf("LiteralTypeAgainst: %v", err)
	}
	if want := (dtype.Timestamp{Unit: dtype.Second, Zone: "UTC"}); !dtype.Equal(got, want) {
		t.Errorf("against a column of seconds a time literal is a %s, want a %s", got, want)
	}

	got, err = plan.LiteralTypeAgainst(when.Value(), nil)
	if err != nil {
		t.Fatalf("LiteralTypeAgainst: %v", err)
	}
	if want := (dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "UTC"}); !dtype.Equal(got, want) {
		t.Errorf("on its own a time literal is a %s, want a %s", got, want)
	}
}

func TestTypeOfAColumnThatIsNotThere(t *testing.T) {
	_, err := plan.TypeOf(plan.Compare(kernel.OpGt, plan.Col("sym"), plan.Lit("A")), typed)
	if err == nil {
		t.Fatal("a plan over a column that is not there was checked and passed")
	}
	if !errors.Is(err, plan.ErrNoColumn) {
		t.Errorf("the error is %v, want an ErrNoColumn", err)
	}
	if !strings.Contains(err.Error(), "did you mean: symbol?") {
		t.Errorf("the error is %q, which does not suggest the column that is there", err)
	}
}

// TestTypeOfRefuses is the list of things that are wrong with a plan and are
// caught before any data is read, which is the whole point of checking one.
func TestTypeOfRefuses(t *testing.T) {
	cases := []struct {
		name string
		expr *plan.Expr
		says string
	}{
		{
			"two types with nothing in common",
			plan.Arith(kernel.OpAdd, price, qty),
			"cannot combine",
		},
		{
			"a comparison of two types with nothing in common",
			plan.Compare(kernel.OpLt, symbol, qty),
			"cannot compare",
		},
		{
			"a fractional literal against an integer column",
			plan.Compare(kernel.OpGt, qty, plan.Lit(1.5)),
			"cannot use a float64 literal with a int64 column",
		},
		{
			"a value the column cannot hold",
			plan.Compare(kernel.OpGt, small, plan.Lit(-1)),
			"-1 does not fit in uint32, which holds 0 to 4294967295",
		},
		{
			"a filter on something that is not a condition",
			plan.And(live, price),
			"not a condition",
		},
		{
			"a cast that has no meaning",
			plan.Cast(dtype.Int64, blob),
			"cannot be cast",
		},
		{
			"arithmetic on text",
			plan.Arith(kernel.OpAdd, symbol, symbol),
			"no arithmetic",
		},
		{
			"a column that is not there, under an operator",
			plan.Not(plan.Col("alive")),
			"not found",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := plan.TypeOf(c.expr, typed)
			if err == nil {
				t.Fatalf("%s was checked and passed", c.expr)
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Errorf("the error is %q, want one saying %q", err, c.says)
			}
		})
	}
}

func TestLiteralTypeOfEveryValue(t *testing.T) {
	cases := []struct {
		lit  any
		want dtype.DataType
	}{
		{nil, dtype.Null},
		{true, dtype.Bool},
		{7, dtype.Int64},
		{int8(7), dtype.Int8},
		{int16(7), dtype.Int16},
		{int32(7), dtype.Int32},
		{int64(7), dtype.Int64},
		{uint(7), dtype.Uint64},
		{uint8(7), dtype.Uint8},
		{uint16(7), dtype.Uint16},
		{uint32(7), dtype.Uint32},
		{uint64(7), dtype.Uint64},
		{float32(7), dtype.Float32},
		{7.0, dtype.Float64},
		{"A", dtype.String},
		{[]byte("A"), dtype.Binary},
		{time.Unix(0, 0), dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "UTC"}},
	}

	for _, c := range cases {
		got, ok := plan.LiteralType(c.lit)
		if !ok {
			t.Errorf("a %T is not a value a column can hold, and it is", c.lit)
			continue
		}
		if !dtype.Equal(got, c.want) {
			t.Errorf("a %T literal is a %s, want a %s", c.lit, got, c.want)
		}
	}
}

func TestLiteralTypeOfSomethingNoColumnHolds(t *testing.T) {
	type point struct{ X, Y int }

	if dt, ok := plan.LiteralType(point{}); ok {
		t.Fatalf("a struct is held by a %s column", dt)
	}

	_, err := plan.LiteralTypeAgainst(point{}, dtype.Int64)
	if !errors.Is(err, plan.ErrWrongType) {
		t.Errorf("a struct literal gives %v, want an ErrWrongType", err)
	}
}

// TestTypeOfReadsTheSchemaOfTheSource is the way a plan gets one, since the
// columns of a scan are whatever the files turn out to hold.
func TestTypeOfReadsTheSchemaOfTheSource(t *testing.T) {
	s, err := plan.Scan(trades{}).Source().Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}

	got, err := plan.TypeOf(plan.Arith(kernel.OpMul, price, qty), s)
	if err == nil {
		t.Fatalf("a float64 column was multiplied by an int64 one, giving a %s", got)
	}
	if !strings.Contains(err.Error(), "cannot combine") {
		t.Errorf("the error is %q, want one saying the two do not combine", err)
	}
}

// BenchmarkTypeOf is what checking an expression costs, which is what a query
// pays once for each of them however many rows it goes on to read.
func BenchmarkTypeOf(b *testing.B) {
	e := plan.And(
		plan.Compare(kernel.OpGt, plan.Arith(kernel.OpMul, price, plan.Lit(2.0)), plan.Lit(100.0)),
		plan.Compare(kernel.OpEq, symbol, plan.Lit("AAPL")),
	)

	for b.Loop() {
		dt, err := plan.TypeOf(e, typed)
		if err != nil {
			b.Fatal(err)
		}
		typeSink = dt
	}
}

// typeSink keeps the benchmark above from being optimized away.
var typeSink dtype.DataType
