package kuma_test

import (
	"fmt"
	"testing"
	"time"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// The tests here walk every method of every expression family, one row per
// method. Each of those methods is one line that picks an operator, so the
// mistake they are written for is the one a table catches and a reader does
// not: a Le that was pasted from Lt and never changed.
//
// A case says what the expression reads as and what it answers over the three
// rows of opsFrame, since either on its own would leave half of the method
// unchecked.
type opsCase struct {
	expr kuma.Expr[kuma.Dynamic]
	text string
	want string
}

// opsBase is the moment the timestamp column starts at.
var opsBase = time.Date(2026, 8, 25, 9, 30, 0, 0, time.UTC)

// opsFrame is three rows wide enough to answer every comparison. The second
// column of each pair holds the middle value of the first, so a comparison
// against a column has the same answers as the comparison against the literal
// beside it.
func opsFrame(t *testing.T) *kuma.Frame[kuma.Dynamic] {
	t.Helper()

	f, err := kuma.NewFrame(
		kuma.NewSeries("price", 1.0, 2.0, 3.0).Column(),
		kuma.NewSeries("limit", 2.0, 2.0, 2.0).Column(),
		kuma.NewSeries("qty", int64(1), 2, 3).Column(),
		kuma.NewSeries("cap", int64(2), 2, 2).Column(),
		kuma.NewSeries("sym", "a", "b", "c").Column(),
		kuma.NewSeries("mid", "b", "b", "b").Column(),
		kuma.NewSeries("flag", true, false, true).Column(),
		kuma.NewSeries("mark", true, true, false).Column(),
		opsTimes(t, "ts", 0, 60, 120),
		opsTimes(t, "noon", 60, 60, 60),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	return f
}

// opsTimes builds a timestamp column held in seconds, offsets seconds after
// opsBase.
func opsTimes(t *testing.T, name string, offsets ...int64) kuma.Column {
	t.Helper()

	dt := dtype.Timestamp{Unit: dtype.Second, Zone: "UTC"}
	b, err := array.NewBuilder(dt)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	for _, off := range offsets {
		b.Append(opsBase.Unix() + off)
	}
	data, err := array.NewChunked(dt, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	c, err := kuma.NewColumn(name, data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	return c
}

// runOps is the body of each of the tests below.
func runOps(t *testing.T, cases []opsCase) {
	t.Helper()

	f := opsFrame(t)
	for _, tt := range cases {
		t.Run(tt.text, func(t *testing.T) {
			if got := tt.expr.String(); got != tt.text {
				t.Errorf("the expression reads as %q, want %q", got, tt.text)
			}
			got, err := f.Eval(tt.expr)
			if err != nil {
				t.Fatalf("Eval: %v", err)
			}
			if have := opsAnswers(t, got); have != tt.want {
				t.Errorf("%s gives %s, want %s", tt.text, have, tt.want)
			}
		})
	}
}

// opsAnswers renders a result column, as t, f and n per row for a condition and
// as the values for anything else.
func opsAnswers(t *testing.T, c kuma.Column) string {
	t.Helper()

	switch c.DType().Kind() {
	case dtype.BoolKind:
		out := make([]byte, c.Len())
		s := c.MustAs[bool]()
		for i := range out {
			switch {
			case s.IsNull(i):
				out[i] = 'n'
			case s.Value(i):
				out[i] = 't'
			default:
				out[i] = 'f'
			}
		}
		return string(out)
	case dtype.Float64Kind:
		return fmt.Sprint(c.MustAs[float64]().Values())
	case dtype.Int64Kind:
		return fmt.Sprint(c.MustAs[int64]().Values())
	case dtype.StringKind:
		return fmt.Sprint(c.MustAs[string]().Values())
	default:
		t.Fatalf("no way to render a %s column", c.DType())
		return ""
	}
}

func TestF64Ops(t *testing.T) {
	price, limit := kuma.F64("price"), kuma.F64("limit")

	runOps(t, []opsCase{
		{price.Eq(2), "(price == 2)", "ftf"},
		{price.Ne(2), "(price != 2)", "tft"},
		{price.Lt(2), "(price < 2)", "tff"},
		{price.Le(2), "(price <= 2)", "ttf"},
		{price.Gt(2), "(price > 2)", "fft"},
		{price.Ge(2), "(price >= 2)", "ftt"},

		{price.EqExpr(limit), "(price == limit)", "ftf"},
		{price.NeExpr(limit), "(price != limit)", "tft"},
		{price.LtExpr(limit), "(price < limit)", "tff"},
		{price.LeExpr(limit), "(price <= limit)", "ttf"},
		{price.GtExpr(limit), "(price > limit)", "fft"},
		{price.GeExpr(limit), "(price >= limit)", "ftt"},

		{price.Add(2), "(price + 2)", "[3 4 5]"},
		{price.Sub(2), "(price - 2)", "[-1 0 1]"},
		{price.Mul(2), "(price * 2)", "[2 4 6]"},
		{price.Div(2), "(price / 2)", "[0.5 1 1.5]"},
		{price.Mod(2), "(price % 2)", "[1 0 1]"},

		{price.AddExpr(limit), "(price + limit)", "[3 4 5]"},
		{price.SubExpr(limit), "(price - limit)", "[-1 0 1]"},
		{price.MulExpr(limit), "(price * limit)", "[2 4 6]"},
		{price.DivExpr(limit), "(price / limit)", "[0.5 1 1.5]"},
		{price.ModExpr(limit), "(price % limit)", "[1 0 1]"},

		{price.AsI64(), "(price as int64)", "[1 2 3]"},
		{price.IsNull(), "(price is null)", "fff"},
		{price.IsNotNull(), "(price is not null)", "ttt"},
	})
}

func TestI64Ops(t *testing.T) {
	qty, size := kuma.I64("qty"), kuma.I64("cap")

	runOps(t, []opsCase{
		{qty.Eq(2), "(qty == 2)", "ftf"},
		{qty.Ne(2), "(qty != 2)", "tft"},
		{qty.Lt(2), "(qty < 2)", "tff"},
		{qty.Le(2), "(qty <= 2)", "ttf"},
		{qty.Gt(2), "(qty > 2)", "fft"},
		{qty.Ge(2), "(qty >= 2)", "ftt"},

		{qty.EqExpr(size), "(qty == cap)", "ftf"},
		{qty.NeExpr(size), "(qty != cap)", "tft"},
		{qty.LtExpr(size), "(qty < cap)", "tff"},
		{qty.LeExpr(size), "(qty <= cap)", "ttf"},
		{qty.GtExpr(size), "(qty > cap)", "fft"},
		{qty.GeExpr(size), "(qty >= cap)", "ftt"},

		{qty.Add(2), "(qty + 2)", "[3 4 5]"},
		{qty.Sub(2), "(qty - 2)", "[-1 0 1]"},
		{qty.Mul(2), "(qty * 2)", "[2 4 6]"},
		{qty.Div(2), "(qty / 2)", "[0 1 1]"},
		{qty.Mod(2), "(qty % 2)", "[1 0 1]"},

		{qty.AddExpr(size), "(qty + cap)", "[3 4 5]"},
		{qty.SubExpr(size), "(qty - cap)", "[-1 0 1]"},
		{qty.MulExpr(size), "(qty * cap)", "[2 4 6]"},
		{qty.DivExpr(size), "(qty / cap)", "[0 1 1]"},
		{qty.ModExpr(size), "(qty % cap)", "[1 0 1]"},

		{qty.AsF64(), "(qty as float64)", "[1 2 3]"},
		{qty.IsNull(), "(qty is null)", "fff"},
		{qty.IsNotNull(), "(qty is not null)", "ttt"},
	})
}

func TestStrOps(t *testing.T) {
	sym, mid := kuma.Str("sym"), kuma.Str("mid")

	runOps(t, []opsCase{
		{sym.Eq("b"), `(sym == "b")`, "ftf"},
		{sym.Ne("b"), `(sym != "b")`, "tft"},
		{sym.Lt("b"), `(sym < "b")`, "tff"},
		{sym.Le("b"), `(sym <= "b")`, "ttf"},
		{sym.Gt("b"), `(sym > "b")`, "fft"},
		{sym.Ge("b"), `(sym >= "b")`, "ftt"},

		{sym.EqExpr(mid), "(sym == mid)", "ftf"},
		{sym.NeExpr(mid), "(sym != mid)", "tft"},
		{sym.LtExpr(mid), "(sym < mid)", "tff"},
		{sym.LeExpr(mid), "(sym <= mid)", "ttf"},
		{sym.GtExpr(mid), "(sym > mid)", "fft"},
		{sym.GeExpr(mid), "(sym >= mid)", "ftt"},

		{sym.IsNull(), "(sym is null)", "fff"},
		{sym.IsNotNull(), "(sym is not null)", "ttt"},
	})
}

func TestBoolOps(t *testing.T) {
	flag, mark := kuma.Bool("flag"), kuma.Bool("mark")

	runOps(t, []opsCase{
		{flag.And(mark), "(flag and mark)", "tff"},
		{flag.Or(mark), "(flag or mark)", "ttt"},
		{flag.Not(), "(not flag)", "ftf"},

		{flag.Eq(true), "(flag == true)", "tft"},
		{flag.Ne(true), "(flag != true)", "ftf"},
		{flag.EqExpr(mark), "(flag == mark)", "tff"},
		{flag.NeExpr(mark), "(flag != mark)", "ftt"},

		{flag.IsNull(), "(flag is null)", "fff"},
		{flag.IsNotNull(), "(flag is not null)", "ttt"},
	})
}

func TestTimeOps(t *testing.T) {
	ts, noon := kuma.Time("ts"), kuma.Time("noon")
	at := opsBase.Add(time.Minute)
	text := at.Format(time.RFC3339Nano)

	runOps(t, []opsCase{
		{ts.Eq(at), "(ts == " + text + ")", "ftf"},
		{ts.Ne(at), "(ts != " + text + ")", "tft"},
		{ts.Before(at), "(ts < " + text + ")", "tff"},
		{ts.AtOrBefore(at), "(ts <= " + text + ")", "ttf"},
		{ts.After(at), "(ts > " + text + ")", "fft"},
		{ts.AtOrAfter(at), "(ts >= " + text + ")", "ftt"},

		{ts.EqExpr(noon), "(ts == noon)", "ftf"},
		{ts.NeExpr(noon), "(ts != noon)", "tft"},
		{ts.BeforeExpr(noon), "(ts < noon)", "tff"},
		{ts.AtOrBeforeExpr(noon), "(ts <= noon)", "ttf"},
		{ts.AfterExpr(noon), "(ts > noon)", "fft"},
		{ts.AtOrAfterExpr(noon), "(ts >= noon)", "ftt"},

		{ts.IsNull(), "(ts is null)", "fff"},
		{ts.IsNotNull(), "(ts is not null)", "ttt"},
	})
}

func TestAnyOps(t *testing.T) {
	price, limit := kuma.Dyn("price"), kuma.Dyn("limit")

	runOps(t, []opsCase{
		{price.Eq(2.0), "(price == 2)", "ftf"},
		{price.Ne(2.0), "(price != 2)", "tft"},
		{price.Lt(2.0), "(price < 2)", "tff"},
		{price.Le(2.0), "(price <= 2)", "ttf"},
		{price.Gt(2.0), "(price > 2)", "fft"},
		{price.Ge(2.0), "(price >= 2)", "ftt"},

		{price.EqExpr(limit), "(price == limit)", "ftf"},
		{price.NeExpr(limit), "(price != limit)", "tft"},
		{price.LtExpr(limit), "(price < limit)", "tff"},
		{price.LeExpr(limit), "(price <= limit)", "ttf"},
		{price.GtExpr(limit), "(price > limit)", "fft"},
		{price.GeExpr(limit), "(price >= limit)", "ftt"},

		{price.Add(2.0), "(price + 2)", "[3 4 5]"},
		{price.Sub(2.0), "(price - 2)", "[-1 0 1]"},
		{price.Mul(2.0), "(price * 2)", "[2 4 6]"},
		{price.Div(2.0), "(price / 2)", "[0.5 1 1.5]"},
		{price.Mod(2.0), "(price % 2)", "[1 0 1]"},

		{price.AddExpr(limit), "(price + limit)", "[3 4 5]"},
		{price.SubExpr(limit), "(price - limit)", "[-1 0 1]"},
		{price.MulExpr(limit), "(price * limit)", "[2 4 6]"},
		{price.DivExpr(limit), "(price / limit)", "[0.5 1 1.5]"},
		{price.ModExpr(limit), "(price % limit)", "[1 0 1]"},

		{price.IsNull(), "(price is null)", "fff"},
		{price.IsNotNull(), "(price is not null)", "ttt"},
	})
}

// TestBareHandles is a handle used as an expression on its own, which is the
// column itself and is how a frame is asked for a column it already has.
func TestBareHandles(t *testing.T) {
	runOps(t, []opsCase{
		{kuma.F64("price"), "price", "[1 2 3]"},
		{kuma.I64("qty"), "qty", "[1 2 3]"},
		{kuma.Str("sym"), "sym", "[a b c]"},
		{kuma.Bool("flag"), "flag", "tft"},
		{kuma.Dyn("price"), "price", "[1 2 3]"},
	})

	if got := kuma.Time("ts").String(); got != "ts" {
		t.Errorf("a time handle reads as %q, want %q", got, "ts")
	}
}

// TestLiteralOnTheLeft is the operand order the evaluator has to work out
// backwards, since the column that decides what the literal becomes is on the
// other side.
func TestLiteralOnTheLeft(t *testing.T) {
	runOps(t, []opsCase{
		{kuma.Lit(2.0).LtExpr(kuma.Dyn("price")), "(2 < price)", "fft"},
		{kuma.Lit(2.0).SubExpr(kuma.Dyn("price")), "(2 - price)", "[1 0 -1]"},
	})
}

// TestOpsSeries walks the handles that read a column straight out of a frame,
// which is the other half of what a handle is for.
func TestOpsSeries(t *testing.T) {
	f := opsFrame(t)

	flag, err := kuma.Bool("flag").Series(f)
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := flag.Value(0); !got {
		t.Error("flag 0 is false, want true")
	}

	ts, err := kuma.Time("ts").Series(f)
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := ts.Value(1); !got.Equal(opsBase.Add(time.Minute)) {
		t.Errorf("ts 1 is %s, want %s", got, opsBase.Add(time.Minute))
	}

	if _, err := kuma.Bool("price").Series(f); err == nil {
		t.Error("the price column was read as a condition")
	}
	if _, err := kuma.Time("qty").Series(f); err == nil {
		t.Error("the quantity column was read as a time")
	}
}
