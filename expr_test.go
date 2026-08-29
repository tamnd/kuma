package kuma_test

import (
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// Trade is the schema the typed tests are written against. It is what a caller
// would write for their own data and what kumagen will read.
type Trade struct {
	Symbol string  `kuma:"symbol"`
	Price  float64 `kuma:"price"`
	Qty    int64   `kuma:"qty"`
}

// tradeCols is what kumagen generates: one handle per column, built once.
var tradeCols = struct {
	Symbol kuma.StrCol[Trade]
	Price  kuma.F64Col[Trade]
	Qty    kuma.I64Col[Trade]
}{
	Symbol: kuma.NewStrCol[Trade]("symbol"),
	Price:  kuma.NewF64Col[Trade]("price"),
	Qty:    kuma.NewI64Col[Trade]("qty"),
}

// typedTrades is the frame the other tests use, bound to Trade so that the
// handles above can be used on it.
func typedTrades(t *testing.T) *kuma.Frame[Trade] {
	t.Helper()

	f, err := kuma.Bind[Trade](trades(t))
	if err != nil {
		t.Fatalf("Bind: %v", err)
	}
	return f
}

// symbolsOf is the symbol column of a result, which is what most of these tests
// check, since knowing which rows came back is knowing whether the condition
// was right.
func symbolsOf[S any](t *testing.T, f *kuma.Frame[S]) []string {
	t.Helper()

	s, err := f.Series[string]("symbol")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	return s.Values()
}

func TestFilter(t *testing.T) {
	c := tradeCols

	tests := []struct {
		name string
		cond kuma.BoolValue[Trade]
		want []string
	}{
		{
			name: "greater than a literal",
			cond: c.Price.Gt(189.5),
			want: []string{"MSFT", "AAPL"},
		},
		{
			name: "at least a literal",
			cond: c.Price.Ge(189.5),
			want: []string{"AAPL", "MSFT", "AAPL"},
		},
		{
			name: "a string equals",
			cond: c.Symbol.Eq("AAPL"),
			want: []string{"AAPL", "AAPL"},
		},
		{
			name: "and",
			cond: c.Price.Gt(150).And(c.Symbol.Ne("MSFT")),
			want: []string{"AAPL", "AAPL"},
		},
		{
			name: "or",
			cond: c.Symbol.Eq("NVDA").Or(c.Qty.Ge(100)),
			want: []string{"AAPL", "NVDA"},
		},
		{
			name: "not",
			cond: c.Symbol.Eq("AAPL").Not(),
			want: []string{"MSFT", "NVDA"},
		},
		{
			name: "an integer column against a whole number",
			cond: c.Qty.Lt(100),
			want: []string{"MSFT", "AAPL"},
		},
		{
			name: "arithmetic on the way",
			cond: c.Price.MulExpr(c.Qty.AsF64()).Gt(20000),
			want: []string{"MSFT", "NVDA"},
		},
		{
			name: "a column against another column",
			cond: c.Price.GtExpr(c.Qty.AsF64()),
			want: []string{"AAPL", "MSFT", "AAPL"},
		},
		{
			name: "the whole frame",
			cond: c.Price.Gt(0),
			want: []string{"AAPL", "MSFT", "AAPL", "NVDA"},
		},
		{
			name: "nothing at all",
			cond: c.Price.Lt(0),
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := typedTrades(t)

			got, err := f.Filter(tt.cond)
			if err != nil {
				t.Fatalf("Filter: %v", err)
			}
			if have := symbolsOf(t, got); !slices.Equal(have, tt.want) {
				t.Errorf("%s kept %v, want %v", tt.cond, have, tt.want)
			}
		})
	}
}

// TestFilterByABoolColumn is the case Filter takes a handle rather than an
// expression, which is what a frame that already holds the answers has.
func TestFilterByABoolColumn(t *testing.T) {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "NVDA").Column(),
		kuma.NewSeries("listed", true, false, true).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	got, err := f.Filter(kuma.Bool("listed"))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if have, want := symbolsOf(t, got), []string{"AAPL", "NVDA"}; !slices.Equal(have, want) {
		t.Errorf("the listed symbols are %v, want %v", have, want)
	}
}

// TestFilterDropsTheRowsNobodyCanAnswerFor is the rule that makes a filter on a
// condition and a filter on its negation not add up to the whole frame.
func TestFilterDropsTheRowsNobodyCanAnswerFor(t *testing.T) {
	f := gaps(t)
	qty := kuma.I64("qty")

	small, err := f.Filter(qty.Lt(3))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	big, err := f.Filter(qty.Lt(3).Not())
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if small.NumRows() != 1 || big.NumRows() != 1 {
		t.Fatalf("a condition kept %d rows and its negation %d, want 1 and 1 of the 4",
			small.NumRows(), big.NumRows())
	}

	missing, err := f.Filter(qty.IsNull())
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if missing.NumRows() != 2 {
		t.Errorf("%d rows have no quantity, want 2", missing.NumRows())
	}
}

func TestFilterErrors(t *testing.T) {
	f := typedTrades(t)
	c := tradeCols

	tests := []struct {
		name string
		cond kuma.BoolValue[Trade]
		want error
		says string
	}{
		{
			name: "a column that is not there",
			cond: kuma.NewF64Col[Trade]("prive").Gt(1),
			want: kuma.ErrNoColumn,
			says: "did you mean: price?",
		},
		{
			name: "a column that is not a condition",
			cond: kuma.NewBoolCol[Trade]("symbol").And(c.Price.Gt(1)),
			want: nil,
			says: "not a condition",
		},
		{
			name: "a string literal against a float column",
			cond: kuma.NewStrCol[Trade]("price").Eq("AAPL"),
			want: nil,
			says: "cannot use a string literal with a float64 column",
		},
		{
			name: "a float literal against an integer column",
			cond: kuma.NewF64Col[Trade]("qty").Gt(1.5),
			want: nil,
			says: "cast the column",
		},
		{
			name: "a literal on the left of a column it does not fit",
			cond: kuma.NewLit[Trade]("AAPL").LtExpr(kuma.NewAnyCol[Trade]("price")),
			want: nil,
			says: "cannot use a string literal with a float64 column",
		},
		{
			name: "a time literal against a column of numbers",
			cond: kuma.NewAnyCol[Trade]("price").Eq(time.Date(2026, 8, 25, 0, 0, 0, 0, time.UTC)),
			want: nil,
			says: "cannot compare float64 with timestamp",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := f.Filter(tt.cond)
			if err == nil {
				t.Fatal("the condition was accepted")
			}
			if tt.want != nil && !errors.Is(err, tt.want) {
				t.Errorf("the error is %v, which is not the one to test for", err)
			}
			if !strings.Contains(err.Error(), tt.says) {
				t.Errorf("the error is %q, which does not say %q", err, tt.says)
			}
		})
	}
}

// TestFilterOnNoColumnAtAll is the condition that never looks at the frame, so
// it has one answer where the frame has four rows and there is no honest way to
// say which of them it kept.
func TestFilterOnNoColumnAtAll(t *testing.T) {
	f := trades(t)

	_, err := f.Filter(kuma.Lit(1).Lt(2))
	if !errors.Is(err, kuma.ErrLength) {
		t.Fatalf("a condition that reads no column gave %v, want an ErrLength", err)
	}

	_, err = f.Eval(kuma.Lit(1).Add(2))
	if !errors.Is(err, kuma.ErrLength) {
		t.Fatalf("an expression that reads no column gave %v, want an ErrLength", err)
	}
}

func TestEval(t *testing.T) {
	f := typedTrades(t)
	c := tradeCols

	notional, err := f.Eval(c.Price.MulExpr(c.Qty.AsF64()))
	if err != nil {
		t.Fatalf("Eval: %v", err)
	}
	if want := "(price * (qty as float64))"; notional.Name() != want {
		t.Errorf("the column is called %q, want %q", notional.Name(), want)
	}
	if !dtype.Equal(notional.DType(), dtype.Float64) {
		t.Errorf("the column is a %s, want a float64", notional.DType())
	}

	values := notional.MustAs[float64]().Values()
	want := []float64{18950, 20560, 4752.5, 48400}
	if !slices.Equal(values, want) {
		t.Errorf("the notionals are %v, want %v", values, want)
	}
}

func TestWithExpr(t *testing.T) {
	f := typedTrades(t)
	c := tradeCols

	got, err := f.WithExpr("cheap", c.Price.Lt(150))
	if err != nil {
		t.Fatalf("WithExpr: %v", err)
	}
	if got.NumCols() != 4 {
		t.Fatalf("%s, want 4 columns", got)
	}

	cheap, err := got.Series[bool]("cheap")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if want := []bool{false, false, false, true}; !slices.Equal(cheap.Values(), want) {
		t.Errorf("the answers are %v, want %v", cheap.Values(), want)
	}
}

// TestTypedAndDynamicAgree is the milestone's own exit criterion, one level
// down from the plan it is written about: the same query written both ways is
// the same expression and gives the same rows.
func TestTypedAndDynamicAgree(t *testing.T) {
	f := trades(t)

	typed := kuma.NewF64Col[kuma.Dynamic]("price").Gt(150).And(kuma.NewStrCol[kuma.Dynamic]("symbol").Eq("AAPL"))
	dynamic := kuma.Dyn("price").Gt(150).And(kuma.Dyn("symbol").Eq("AAPL"))

	if typed.String() != dynamic.String() {
		t.Fatalf("the typed query is %s and the dynamic one is %s", typed, dynamic)
	}

	a, err := f.Filter(typed)
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	b, err := f.Filter(dynamic)
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if !slices.Equal(symbolsOf(t, a), symbolsOf(t, b)) {
		t.Errorf("the typed query kept %v and the dynamic one %v",
			symbolsOf(t, a), symbolsOf(t, b))
	}
}

// TestLiteralOfEveryType walks every Go value a query can be written with,
// against a column of the type that value belongs in. Each one is compared to
// itself, so the answer is true unless the value went in wrong or came out as
// something else.
func TestLiteralOfEveryType(t *testing.T) {
	when := time.Date(2026, 8, 25, 9, 30, 0, 0, time.UTC)

	tests := []struct {
		name string
		dt   dtype.DataType
		put  func(*array.Builder)
		lit  any
	}{
		{"bool", dtype.Bool, func(b *array.Builder) { b.AppendBool(true) }, true},
		{"int", dtype.Int64, func(b *array.Builder) { b.Append(int64(7)) }, 7},
		{"int8", dtype.Int8, func(b *array.Builder) { b.Append(int8(7)) }, int8(7)},
		{"int16", dtype.Int16, func(b *array.Builder) { b.Append(int16(7)) }, int16(7)},
		{"int32", dtype.Int32, func(b *array.Builder) { b.Append(int32(7)) }, int32(7)},
		{"int64", dtype.Int64, func(b *array.Builder) { b.Append(int64(7)) }, int64(7)},
		{"uint", dtype.Uint64, func(b *array.Builder) { b.Append(uint64(7)) }, uint(7)},
		{"uint8", dtype.Uint8, func(b *array.Builder) { b.Append(uint8(7)) }, uint8(7)},
		{"uint16", dtype.Uint16, func(b *array.Builder) { b.Append(uint16(7)) }, uint16(7)},
		{"uint32", dtype.Uint32, func(b *array.Builder) { b.Append(uint32(7)) }, uint32(7)},
		{"uint64", dtype.Uint64, func(b *array.Builder) { b.Append(uint64(7)) }, uint64(7)},
		{"float32", dtype.Float32, func(b *array.Builder) { b.Append(float32(1.5)) }, float32(1.5)},
		{"float64", dtype.Float64, func(b *array.Builder) { b.Append(1.5) }, 1.5},
		{"string", dtype.String, func(b *array.Builder) { b.AppendString("x") }, "x"},
		{"binary", dtype.Binary, func(b *array.Builder) { b.AppendBytes([]byte("x")) }, []byte("x")},
		{
			"a time in seconds",
			dtype.Timestamp{Unit: dtype.Second, Zone: "UTC"},
			func(b *array.Builder) { b.Append(when.Unix()) },
			when,
		},
		{
			"a time in milliseconds",
			dtype.Timestamp{Unit: dtype.Millisecond, Zone: "UTC"},
			func(b *array.Builder) { b.Append(when.UnixMilli()) },
			when,
		},
		{
			"a time in microseconds",
			dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"},
			func(b *array.Builder) { b.Append(when.UnixMicro()) },
			when,
		},
		{
			"a time in nanoseconds",
			dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "UTC"},
			func(b *array.Builder) { b.Append(when.UnixNano()) },
			when,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := oneValueFrame(t, tt.dt, tt.put)

			got, err := f.Eval(kuma.Dyn("v").Eq(tt.lit))
			if err != nil {
				t.Fatalf("Eval: %v", err)
			}
			if !got.MustAs[bool]().Value(0) {
				t.Errorf("%s does not equal the value it was written from", tt.lit)
			}

			// A missing literal is the missing value, which is not equal to
			// anything, including the value that is there.
			none, err := f.Eval(kuma.Dyn("v").Eq(nil))
			if err != nil {
				t.Fatalf("Eval against null: %v", err)
			}
			if !none.IsNull(0) {
				t.Error("comparing against null gave an answer, want a null")
			}
		})
	}
}

// oneValueFrame returns a frame whose one column is called v and holds the one
// value put writes.
func oneValueFrame(t *testing.T, dt dtype.DataType, put func(*array.Builder)) *kuma.Frame[kuma.Dynamic] {
	t.Helper()

	b, err := array.NewBuilder(dt)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	put(b)

	data, err := array.NewChunked(dt, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	c, err := kuma.NewColumn("v", data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	f, err := kuma.NewFrame(c)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	return f
}

// TestLiteralOfANoneSuchType is a Go value there is no column for, which is
// only reachable through the dynamic handles and is an error rather than a
// panic because a query is data as often as it is code.
func TestLiteralOfANoneSuchType(t *testing.T) {
	f := trades(t)

	_, err := f.Filter(kuma.Dyn("price").Eq(struct{ A int }{1}))
	if !errors.Is(err, kuma.ErrWrongType) {
		t.Fatalf("comparing against a struct gave %v, want an ErrWrongType", err)
	}
}

// TestLiteralTextOfEveryType is what an expression looks like when it is read
// back, for the literals whose text is not just the number.
func TestLiteralTextOfEveryType(t *testing.T) {
	tests := []struct {
		expr kuma.Expr[kuma.Dynamic]
		want string
	}{
		{kuma.Dyn("v").Eq(nil), "(v == null)"},
		{kuma.Dyn("v").Eq(true), "(v == true)"},
		{kuma.Dyn("v").Eq("x"), `(v == "x")`},
		{kuma.Dyn("v").Eq([]byte("x")), `(v == "x")`},
		{kuma.Dyn("v").Eq(float32(1.5)), "(v == 1.5)"},
		{kuma.Dyn("v").Eq(1.5), "(v == 1.5)"},
		{kuma.Dyn("v").Eq(7), "(v == 7)"},
		{kuma.Dyn("v").Eq(uint8(7)), "(v == 7)"},
	}

	for _, tt := range tests {
		if got := tt.expr.String(); got != tt.want {
			t.Errorf("the expression reads as %q, want %q", got, tt.want)
		}
	}
}

// TestLiteralTakesTheColumnType is the rule that keeps a comparison against a
// number from widening the column it is compared against.
func TestLiteralTakesTheColumnType(t *testing.T) {
	f := smallInts(t)

	got, err := f.Eval(kuma.Dyn("n").Add(1))
	if err != nil {
		t.Fatalf("Eval: %v", err)
	}
	if !dtype.Equal(got.DType(), dtype.Uint8) {
		t.Errorf("adding 1 to a uint8 column gives a %s column, want a uint8 one", got.DType())
	}

	kept, err := f.Filter(kuma.Dyn("n").Gt(1))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if kept.NumRows() != 2 {
		t.Errorf("%d values are over 1, want 2", kept.NumRows())
	}
}

// TestLiteralThatDoesNotFit is the other half of that rule. The type of the
// literal fits the column and the value does not, which is a question about the
// value and is answered where the value is.
func TestLiteralThatDoesNotFit(t *testing.T) {
	f := smallInts(t)

	_, err := f.Filter(kuma.Dyn("n").Gt(300))
	if err == nil {
		t.Fatal("300 was compared against a uint8 column")
	}
	if !strings.Contains(err.Error(), "300") {
		t.Errorf("the error is %q, which does not say which value did not fit", err)
	}
}

// smallInts returns a frame whose one column is a uint8, which is the type a
// literal is most likely to be wrong about.
func smallInts(t *testing.T) *kuma.Frame[kuma.Dynamic] {
	t.Helper()

	b, err := array.NewBuilder(dtype.Uint8)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	for _, v := range []uint8{1, 2, 3} {
		b.Append(v)
	}
	data, err := array.NewChunked(dtype.Uint8, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	c, err := kuma.NewColumn("n", data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	f, err := kuma.NewFrame(c)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	return f
}

func TestTimeLiteral(t *testing.T) {
	f := minutes(t)
	minute := kuma.Time("minute")

	nine30 := time.Date(2026, 8, 25, 9, 30, 0, 0, time.UTC)

	got, err := f.Filter(minute.After(nine30))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if got.NumRows() != 2 {
		t.Errorf("%d rows are after 9:30, want 2", got.NumRows())
	}

	same, err := f.Filter(minute.Eq(nine30))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if same.NumRows() != 1 {
		t.Errorf("%d rows are at 9:30, want 1", same.NumRows())
	}
}

// TestTimeLiteralThatDoesNotFitTheUnit is the rule that a literal lands on the
// column's unit exactly or not at all.
func TestTimeLiteralThatDoesNotFitTheUnit(t *testing.T) {
	f := minutes(t)

	odd := time.Date(2026, 8, 25, 9, 30, 0, 500, time.UTC)
	_, err := f.Filter(kuma.Time("minute").After(odd))
	if err == nil {
		t.Fatal("a time with nanoseconds in it was used with a column of seconds")
	}
	if !strings.Contains(err.Error(), "truncate") {
		t.Errorf("the error is %q, which does not say what to do about it", err)
	}
}

// minutes returns a frame of timestamps stored in seconds, which is the unit a
// wall clock literal is most likely to be finer than.
func minutes(t *testing.T) *kuma.Frame[kuma.Dynamic] {
	t.Helper()

	dt := dtype.Timestamp{Unit: dtype.Second, Zone: "UTC"}
	b, err := array.NewBuilder(dt)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	base := time.Date(2026, 8, 25, 9, 30, 0, 0, time.UTC).Unix()
	for _, v := range []int64{base, base + 60, base + 120} {
		b.Append(v)
	}
	data, err := array.NewChunked(dt, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	c, err := kuma.NewColumn("minute", data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	f, err := kuma.NewFrame(c)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	return f
}

func TestExprString(t *testing.T) {
	c := tradeCols

	tests := []struct {
		expr kuma.Expr[Trade]
		want string
	}{
		{c.Price, "price"},
		{c.Price.Gt(100), "(price > 100)"},
		{c.Symbol.Eq("AAPL"), `(symbol == "AAPL")`},
		{c.Price.Gt(100).And(c.Symbol.Ne("MSFT")), `((price > 100) and (symbol != "MSFT"))`},
		{c.Price.Gt(100).Not(), "(not (price > 100))"},
		{c.Price.Add(1).Mul(2), "((price + 1) * 2)"},
		{c.Qty.Div(2), "(qty / 2)"},
		{c.Qty.Mod(2), "(qty % 2)"},
		{c.Price.IsNull(), "(price is null)"},
		{c.Price.IsNotNull(), "(price is not null)"},
		{c.Qty.AsF64(), "(qty as float64)"},
		{c.Price.AsI64(), "(price as int64)"},
		{c.Price.Sub(0.5), "(price - 0.5)"},
		{c.Qty.SubExpr(c.Qty), "(qty - qty)"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			got, ok := tt.expr.(interface{ String() string })
			if !ok {
				t.Fatalf("a %T does not print", tt.expr)
			}
			if got.String() != tt.want {
				t.Errorf("the expression prints as %q, want %q", got.String(), tt.want)
			}
		})
	}
}

func TestHandleSeries(t *testing.T) {
	f := typedTrades(t)
	c := tradeCols

	prices, err := c.Price.Series(f)
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got, want := prices.Value(1), 411.2; got != want {
		t.Errorf("price 1 is %v, want %v", got, want)
	}

	qty, err := c.Qty.Series(f)
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got, want := qty.Value(0), int64(100); got != want {
		t.Errorf("quantity 0 is %d, want %d", got, want)
	}

	symbols, err := c.Symbol.Series(f)
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got, want := symbols.Value(3), "NVDA"; got != want {
		t.Errorf("symbol 3 is %q, want %q", got, want)
	}

	if _, err := kuma.NewI64Col[Trade]("price").Series(f); !errors.Is(err, kuma.ErrWrongType) {
		t.Errorf("reading the price column as an int64 gave %v, want an ErrWrongType", err)
	}
}

func TestHandleName(t *testing.T) {
	tests := []struct {
		got  string
		want string
	}{
		{tradeCols.Price.Name(), "price"},
		{tradeCols.Qty.Name(), "qty"},
		{tradeCols.Symbol.Name(), "symbol"},
		{kuma.Bool("listed").Name(), "listed"},
		{kuma.Time("minute").Name(), "minute"},
		{kuma.Dyn("whatever").Name(), "whatever"},
	}

	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("the handle names %q, want %q", tt.got, tt.want)
		}
	}
}

// TestDynColumn is the way back from a dynamic handle to the column itself.
func TestDynColumn(t *testing.T) {
	f := trades(t)

	c, err := kuma.Dyn("price").Column(f)
	if err != nil {
		t.Fatalf("Column: %v", err)
	}
	if c.Name() != "price" || c.Len() != 4 {
		t.Errorf("%s, want the price column of 4 values", c)
	}
	if _, err := kuma.Dyn("nope").Column(f); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("a column that is not there gave %v, want an ErrNoColumn", err)
	}
}

// TestDynArithmetic walks the dynamic path over the operators, where the type
// of everything is settled against the frame rather than at compile time.
func TestDynArithmetic(t *testing.T) {
	f := trades(t)

	tests := []struct {
		name string
		expr kuma.Expr[kuma.Dynamic]
		want []int64
	}{
		{"add", kuma.Dyn("qty").Add(1), []int64{101, 51, 26, 401}},
		{"subtract", kuma.Dyn("qty").Sub(1), []int64{99, 49, 24, 399}},
		{"multiply", kuma.Dyn("qty").Mul(2), []int64{200, 100, 50, 800}},
		{"divide", kuma.Dyn("qty").Div(3), []int64{33, 16, 8, 133}},
		{"remainder", kuma.Dyn("qty").Mod(3), []int64{1, 2, 1, 1}},
		{"a column against a column", kuma.Dyn("qty").SubExpr(kuma.Dyn("qty")), []int64{0, 0, 0, 0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := f.Eval(tt.expr)
			if err != nil {
				t.Fatalf("Eval: %v", err)
			}
			if have := got.MustAs[int64]().Values(); !slices.Equal(have, tt.want) {
				t.Errorf("%s gives %v, want %v", tt.expr, have, tt.want)
			}
		})
	}
}

func TestDivideByZero(t *testing.T) {
	f := trades(t)

	_, err := f.Eval(kuma.Dyn("qty").Div(0))
	if err == nil {
		t.Fatal("dividing an integer column by zero was allowed")
	}
	if !strings.Contains(err.Error(), "division by zero") {
		t.Errorf("the error is %q, which does not say what went wrong", err)
	}
}

// TestCast walks the two casts a numeric handle offers, which are the same
// conversions Go writes with int64(x) and float64(x).
func TestCast(t *testing.T) {
	f := typedTrades(t)

	whole, err := f.Eval(tradeCols.Price.AsI64())
	if err != nil {
		t.Fatalf("Eval: %v", err)
	}
	if want := []int64{189, 411, 190, 121}; !slices.Equal(whole.MustAs[int64]().Values(), want) {
		t.Errorf("the prices came out as %v, want %v", whole.MustAs[int64]().Values(), want)
	}

	as64, err := f.Eval(tradeCols.Qty.AsF64())
	if err != nil {
		t.Fatalf("Eval: %v", err)
	}
	if want := []float64{100, 50, 25, 400}; !slices.Equal(as64.MustAs[float64]().Values(), want) {
		t.Errorf("the quantities came out as %v, want %v", as64.MustAs[float64]().Values(), want)
	}
}

// TestEvalCarriesTheErrorUp puts a column that is not there under each kind of
// step there is, since an error found halfway down a tree has to come back out
// of it rather than being lost or turned into an empty column.
func TestEvalCarriesTheErrorUp(t *testing.T) {
	f := trades(t)
	gone, ok := kuma.Dyn("nope"), kuma.Dyn("price")

	tests := []struct {
		name string
		expr kuma.Expr[kuma.Dynamic]
	}{
		{"a comparison", gone.Gt(1)},
		{"the right of a comparison", ok.GtExpr(gone)},
		{"the right of a comparison under a literal", kuma.Lit(1).LtExpr(gone)},
		{"arithmetic", gone.Add(1)},
		{"the right of arithmetic", ok.AddExpr(gone)},
		{"and", gone.Gt(1).And(ok.Gt(1))},
		{"the right of and", ok.Gt(1).And(gone.Gt(1))},
		{"or", gone.Gt(1).Or(ok.Gt(1))},
		{"not", gone.Gt(1).Not()},
		{"is null", gone.IsNull()},
		{"is not null", gone.IsNotNull()},
		{"a cast", kuma.I64("nope").AsF64()},
		{"two steps down", ok.Gt(1).And(ok.Lt(1).Or(gone.Gt(1)))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := f.Eval(tt.expr); !errors.Is(err, kuma.ErrNoColumn) {
				t.Errorf("%s gave %v, want an ErrNoColumn", tt.expr, err)
			}
		})
	}

	if _, err := f.WithExpr("x", gone.Gt(1)); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("WithExpr gave %v, want an ErrNoColumn", err)
	}
}

// TestEvalOnAFrameWithoutTheColumn checks that an expression names the
// operation in the error, so that a wrong name in a chain says where it was.
func TestEvalOnAFrameWithoutTheColumn(t *testing.T) {
	f := trades(t)

	_, err := f.Eval(kuma.Dyn("prices").Gt(1))
	if !errors.Is(err, kuma.ErrNoColumn) {
		t.Fatalf("a column that is not there gave %v, want an ErrNoColumn", err)
	}
	if !strings.Contains(err.Error(), "in Eval") {
		t.Errorf("the error is %q, which does not say what was running", err)
	}
}

// TestExprIsReusable checks that an expression is a value rather than something
// that is used up, since a package level handle is the whole point of the
// design and a query is often run over several frames.
func TestExprIsReusable(t *testing.T) {
	cond := tradeCols.Price.Gt(150)

	for range 3 {
		f := typedTrades(t)
		got, err := f.Filter(cond)
		if err != nil {
			t.Fatalf("Filter: %v", err)
		}
		if got.NumRows() != 3 {
			t.Fatalf("the same condition kept %d rows this time, want 3", got.NumRows())
		}
	}
}

// TestThePlanTypeIsTheColumnThatComesOut is the promise plan.TypeOf makes: the
// type a plan is checked to have is the type the values arrive as. The two are
// worked out by different code over different things, one over the schema
// alone and one over the values, so this is what keeps them in step.
func TestThePlanTypeIsTheColumnThatComesOut(t *testing.T) {
	f := trades(t)
	s := f.Schema()

	tests := []struct {
		eager kuma.Expr[kuma.Dynamic]
		node  *plan.Expr
	}{
		{
			kuma.Dyn("price").Gt(150),
			plan.Compare(kernel.OpGt, plan.Col("price"), plan.Lit(150)),
		},
		{
			kuma.Dyn("qty").Add(1),
			plan.Arith(kernel.OpAdd, plan.Col("qty"), plan.Lit(1)),
		},
		{
			kuma.Dyn("price").MulExpr(kuma.Lit(2.0)),
			plan.Arith(kernel.OpMul, plan.Col("price"), plan.Lit(2.0)),
		},
		{
			kuma.Dyn("price").Gt(150).And(kuma.Dyn("symbol").Eq("AAPL")),
			plan.And(
				plan.Compare(kernel.OpGt, plan.Col("price"), plan.Lit(150)),
				plan.Compare(kernel.OpEq, plan.Col("symbol"), plan.Lit("AAPL")),
			),
		},
		{
			kuma.Dyn("symbol").IsNull(),
			plan.IsNull(plan.Col("symbol")),
		},
		{
			kuma.NewF64Col[kuma.Dynamic]("price").AsI64(),
			plan.Cast(dtype.Int64, plan.Col("price")),
		},
	}

	for _, tt := range tests {
		if tt.eager.String() != tt.node.String() {
			t.Errorf("the query reads %s and the plan reads %s", tt.eager, tt.node)
			continue
		}

		got, err := f.Eval(tt.eager)
		if err != nil {
			t.Errorf("Eval %s: %v", tt.eager, err)
			continue
		}
		want, err := plan.TypeOf(tt.node, s)
		if err != nil {
			t.Errorf("TypeOf %s: %v", tt.node, err)
			continue
		}
		if !dtype.Equal(got.DType(), want) {
			t.Errorf("%s was checked as a %s and came out as a %s", tt.node, want, got.DType())
		}
	}
}

// TestThePlanCatchesWhatTheFrameWould is the other half of it. An expression
// the check turns away is one the frame turns away too, so a query is refused
// for the same reason wherever it is run.
func TestThePlanCatchesWhatTheFrameWould(t *testing.T) {
	f := trades(t)
	s := f.Schema()

	tests := []struct {
		eager kuma.Expr[kuma.Dynamic]
		node  *plan.Expr
	}{
		{
			kuma.Dyn("price").AddExpr(kuma.Dyn("qty")),
			plan.Arith(kernel.OpAdd, plan.Col("price"), plan.Col("qty")),
		},
		{
			kuma.Dyn("qty").Gt(1.5),
			plan.Compare(kernel.OpGt, plan.Col("qty"), plan.Lit(1.5)),
		},
		{
			kuma.Dyn("nope").Gt(1),
			plan.Compare(kernel.OpGt, plan.Col("nope"), plan.Lit(1)),
		},
	}

	for _, tt := range tests {
		if _, err := f.Eval(tt.eager); err == nil {
			t.Errorf("the frame ran %s, which the plan should have turned away", tt.eager)
		}
		if _, err := plan.TypeOf(tt.node, s); err == nil {
			t.Errorf("the plan passed %s, which the frame turns away", tt.node)
		}
	}
}
