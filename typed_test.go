package kuma_test

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
)

// Quote is two of the three columns the trades frame holds, written in the
// other order, so that a test can tell the columns the struct named from the
// columns that were there.
type Quote struct {
	Price  float64 `kuma:"price"`
	Symbol string  `kuma:"symbol"`
}

// quoteCols is what kumagen writes for Quote.
var quoteCols = struct {
	Price  kuma.F64Col[Quote]
	Symbol kuma.StrCol[Quote]
}{
	Price:  kuma.NewF64Col[Quote]("price"),
	Symbol: kuma.NewStrCol[Quote]("symbol"),
}

func TestSelectAs(t *testing.T) {
	f := trades(t)

	got, err := f.SelectAs[Quote]()
	if err != nil {
		t.Fatalf("SelectAs: %v", err)
	}

	// The struct decides which columns and in which order, so the quantity is
	// gone and the price comes first even though the frame had it second.
	if want := []string{"price", "symbol"}; !slices.Equal(got.Names(), want) {
		t.Errorf("SelectAs kept %v, want %v", got.Names(), want)
	}
	if got.NumRows() != f.NumRows() {
		t.Errorf("SelectAs gave %d rows, want the %d that went in", got.NumRows(), f.NumRows())
	}

	// The frame is typed, so a handle written for Quote reads it.
	prices, err := quoteCols.Price.Series(got)
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if want := []float64{189.5, 411.2, 190.1, 121.0}; !slices.Equal(prices.Values(), want) {
		t.Errorf("the prices are %v, want %v", prices.Values(), want)
	}
}

// TestSelectAsSharesColumns is the promise that this costs a slice header. The
// columns that came out are the ones that went in rather than copies of them.
func TestSelectAsSharesColumns(t *testing.T) {
	f := trades(t)

	got, err := f.SelectAs[Quote]()
	if err != nil {
		t.Fatalf("SelectAs: %v", err)
	}

	for _, name := range got.Names() {
		before, err := f.Column(name)
		if err != nil {
			t.Fatalf("Column: %v", err)
		}
		after, err := got.Column(name)
		if err != nil {
			t.Fatalf("Column: %v", err)
		}
		if before.Data() != after.Data() {
			t.Errorf("the %s column was rebuilt, want the one that went in", name)
		}
	}
}

// TestSelectAsSkipsFields is the rule Bind follows about which fields are
// columns, checked here because a field that is not a column must not turn into
// one that is missing.
func TestSelectAsSkipsFields(t *testing.T) {
	type Row struct {
		Symbol string `kuma:"symbol"`
		Note   string `kuma:"-"`
		cache  []byte
	}

	got, err := trades(t).SelectAs[Row]()
	if err != nil {
		t.Fatalf("SelectAs: %v", err)
	}
	if want := []string{"symbol"}; !slices.Equal(got.Names(), want) {
		t.Errorf("SelectAs kept %v, want %v", got.Names(), want)
	}

	// The unexported field is read here so that the compiler and the linters
	// agree it is part of the struct rather than a leftover.
	var r Row
	if len(r.cache) != 0 {
		t.Error("a zero Row has a cache in it")
	}
}

// TestSelectAsUntaggedField is the other half of the naming rule, which is the
// field name in snake case when there is no tag.
func TestSelectAsUntaggedField(t *testing.T) {
	type Row struct {
		Symbol string
	}

	got, err := trades(t).SelectAs[Row]()
	if err != nil {
		t.Fatalf("SelectAs: %v", err)
	}
	if want := []string{"symbol"}; !slices.Equal(got.Names(), want) {
		t.Errorf("SelectAs kept %v, want %v", got.Names(), want)
	}
}

func TestSelectAsMistakes(t *testing.T) {
	f := trades(t)

	type missing struct {
		Venue string `kuma:"venue"`
	}
	if _, err := f.SelectAs[missing](); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("SelectAs to a struct naming a column that is not there = %v, want ErrNoColumn", err)
	}

	type wrong struct {
		Symbol int64 `kuma:"symbol"`
	}
	if _, err := f.SelectAs[wrong](); !errors.Is(err, kuma.ErrWrongType) {
		t.Errorf("SelectAs to a struct wanting a string column as an int64 = %v, want ErrWrongType", err)
	}

	type nothing struct {
		Note string `kuma:"-"`
	}
	if _, err := f.SelectAs[nothing](); !errors.Is(err, kuma.ErrNoValues) {
		t.Errorf("SelectAs to a struct that names no columns = %v, want ErrNoValues", err)
	}

	type twice struct {
		Symbol string `kuma:"symbol"`
		Also   string `kuma:"symbol"`
	}
	if _, err := f.SelectAs[twice](); !errors.Is(err, kuma.ErrDuplicateColumn) {
		t.Errorf("SelectAs to a struct naming one column twice = %v, want ErrDuplicateColumn", err)
	}

	if _, err := f.SelectAs[int](); !errors.Is(err, kuma.ErrWrongType) {
		t.Errorf("SelectAs to a type that is not a struct = %v, want ErrWrongType", err)
	}
}

// TestSelectAsSaysWhatWentWrong is the message a caller reads, since a struct
// and a file that disagree is the thing this step exists to catch.
func TestSelectAsSaysWhatWentWrong(t *testing.T) {
	type wrong struct {
		Symbol int64 `kuma:"symbol"`
	}

	_, err := trades(t).SelectAs[wrong]()
	if err == nil {
		t.Fatal("SelectAs on a field of the wrong type succeeded")
	}
	for _, want := range []string{"SelectAs", "Symbol", "symbol", "int64", "string"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the message is %q, want %q in it", err.Error(), want)
		}
	}
}

func TestLazySelectAs(t *testing.T) {
	f := typedTrades(t)

	want, err := f.SelectAs[Quote]()
	if err != nil {
		t.Fatalf("SelectAs: %v", err)
	}
	got, err := f.Lazy().SelectAs[Quote]().Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	if got.String() != want.String() {
		t.Errorf("the query gave\n%s\nwant\n%s", got, want)
	}
}

// TestLazySelectAsKeepsTheSchemaType is the point of the step. What comes back
// is a query over Quote, so the rest of it can be written against the handles.
func TestLazySelectAsKeepsTheSchemaType(t *testing.T) {
	out, err := typedTrades(t).Lazy().
		SelectAs[Quote]().
		Filter(quoteCols.Price.Gt(200)).
		Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	symbols, err := quoteCols.Symbol.Series(out)
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if want := []string{"MSFT"}; !slices.Equal(symbols.Values(), want) {
		t.Errorf("the query gave %v, want %v", symbols.Values(), want)
	}
}

// TestLazySelectAsSchema is the schema of the query, worked out without running
// it, which is the struct's columns in the struct's order.
func TestLazySelectAsSchema(t *testing.T) {
	s, err := trades(t).Lazy().SelectAs[Quote]().Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}
	if want := []string{"price", "symbol"}; !slices.Equal(s.Names(), want) {
		t.Errorf("the query produces %v, want %v", s.Names(), want)
	}
}

// TestLazySelectAsIsCheckedWhereItIsWritten is what this step has that the
// others do not. The struct and the plan can be compared at the point the step
// is written, so the mistake is known before Collect is reached.
func TestLazySelectAsIsCheckedWhereItIsWritten(t *testing.T) {
	type missing struct {
		Venue string `kuma:"venue"`
	}

	q := trades(t).Lazy().SelectAs[missing]()
	if err := q.Validate(); !errors.Is(err, kuma.ErrNoColumn) {
		t.Fatalf("Validate = %v, want ErrNoColumn", err)
	}

	// The steps written after it build nothing, so what Collect reports is the
	// mistake that was made rather than the one it led to.
	if _, err := q.Head(1).Collect(t.Context()); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("Collect = %v, want ErrNoColumn", err)
	}
}

// TestLazySelectAsCarriesTheEarlierMistake is the other direction, which is a
// query that was already wrong staying wrong through a typed step.
func TestLazySelectAsCarriesTheEarlierMistake(t *testing.T) {
	q := trades(t).Lazy().Drop("venue").SelectAs[Quote]()

	if _, err := q.Collect(t.Context()); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("Collect = %v, want the mistake the Drop made, got %v", err, err)
	}
}

// TestLazySelectAsString is the plan it builds, which is the projection a select
// by name builds and nothing else. The struct is gone by the time the plan is
// written, which is why the same query written both ways optimizes the same way.
func TestLazySelectAsString(t *testing.T) {
	got := trades(t).Lazy().SelectAs[Quote]().String()
	want := "Project price, symbol\n  Scan frame"
	if got != want {
		t.Errorf("the plan is\n%s\nwant\n%s", got, want)
	}
}

// Total is the result of a group by symbol with a sum of the quantity, written
// with the aggregation first so that a test can tell the struct's order from the
// keys first order the untyped Agg produces.
type Total struct {
	Qty    int64  `kuma:"qty"`
	Symbol string `kuma:"symbol"`
}

func TestAggAs(t *testing.T) {
	g, err := trades(t).GroupBy("symbol")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}

	got, err := g.AggAs[Total](kuma.Sum("qty"))
	if err != nil {
		t.Fatalf("AggAs: %v", err)
	}

	if want := []string{"qty", "symbol"}; !slices.Equal(got.Names(), want) {
		t.Errorf("AggAs gave %v, want %v", got.Names(), want)
	}
	if r := rows(t, got); !slices.Equal(r, []string{"125 AAPL", "50 MSFT", "400 NVDA"}) {
		t.Errorf("AggAs gave %v", r)
	}
}

// TestAggAsLeavesOutWhatTheStructDoesNotName is the rule a select follows, here
// applied to an aggregation that was worked out and then not wanted. Asking for
// one and not keeping it is allowed, since a struct is a list of what the result
// holds rather than a list of what to compute.
func TestAggAsLeavesOutWhatTheStructDoesNotName(t *testing.T) {
	g, err := trades(t).GroupBy("symbol")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}

	got, err := g.AggAs[Total](kuma.Sum("qty"), kuma.Mean("price").As("avg"))
	if err != nil {
		t.Fatalf("AggAs: %v", err)
	}
	if want := []string{"qty", "symbol"}; !slices.Equal(got.Names(), want) {
		t.Errorf("AggAs gave %v, want %v", got.Names(), want)
	}
}

// TestAggAsMistakes is the one a caller makes, which is naming a column that no
// aggregation produced under that name. The message says AggAs rather than the
// select underneath, since that is the step that was written.
func TestAggAsMistakes(t *testing.T) {
	g, err := trades(t).GroupBy("symbol")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}

	type wanted struct {
		Symbol string `kuma:"symbol"`
		Total  int64  `kuma:"total"`
	}
	_, err = g.AggAs[wanted](kuma.Sum("qty"))
	if !errors.Is(err, kuma.ErrNoColumn) {
		t.Fatalf("AggAs = %v, want ErrNoColumn", err)
	}
	if !strings.Contains(err.Error(), "AggAs") {
		t.Errorf("the message is %q, want it to name the step", err.Error())
	}

	// The mistake the aggregation itself makes still comes from there.
	if _, err := g.AggAs[Total](kuma.Sum("smybol")); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("AggAs on a column that is not there = %v, want ErrNoColumn", err)
	}
	if _, err := g.AggAs[Total](); !errors.Is(err, kuma.ErrLength) {
		t.Errorf("AggAs with nothing to aggregate = %v, want ErrLength", err)
	}
}

func TestLazyAggAs(t *testing.T) {
	f := typedTrades(t)

	g, err := f.GroupBy("symbol")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	want, err := g.AggAs[Total](kuma.Sum("qty"))
	if err != nil {
		t.Fatalf("AggAs: %v", err)
	}

	got, err := f.Lazy().GroupBy("symbol").AggAs[Total](kuma.Sum("qty")).Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if got.String() != want.String() {
		t.Errorf("the query gave\n%s\nwant\n%s", got, want)
	}
}

// TestLazyAggAsKeepsTheSchemaType is the point of it, which is that the steps
// after a group by are written against the struct rather than against strings.
func TestLazyAggAsKeepsTheSchemaType(t *testing.T) {
	qty := kuma.NewI64Col[Total]("qty")

	out, err := typedTrades(t).Lazy().
		GroupBy("symbol").
		AggAs[Total](kuma.Sum("qty")).
		Filter(qty.Ge(125)).
		Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if r := rows(t, out); !slices.Equal(r, []string{"125 AAPL", "400 NVDA"}) {
		t.Errorf("the query gave %v", r)
	}
}

// TestLazyAggAsIsCheckedWhereItIsWritten is the same promise the typed select
// makes, here over a step whose columns did not exist until it ran.
func TestLazyAggAsIsCheckedWhereItIsWritten(t *testing.T) {
	type wanted struct {
		Symbol string `kuma:"symbol"`
		Total  int64  `kuma:"total"`
	}

	q := trades(t).Lazy().GroupBy("symbol").AggAs[wanted](kuma.Sum("qty"))
	if err := q.Validate(); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("Validate = %v, want ErrNoColumn", err)
	}
}

// TestLazyAggAsString is the plan, which is a projection over the aggregate. The
// struct is gone by then, so this optimizes the way the same query written by
// name does.
func TestLazyAggAsString(t *testing.T) {
	got := trades(t).Lazy().GroupBy("symbol").AggAs[Total](kuma.Sum("qty")).String()
	want := "Project qty, symbol\n  Aggregate by symbol: Sum(qty)\n    Scan frame"
	if got != want {
		t.Errorf("the plan is\n%s\nwant\n%s", got, want)
	}
}

// Sector is the schema of the frame the trades are joined to, so that a test can
// join two typed frames rather than a typed one and a dynamic one.
type Sector struct {
	Symbol string `kuma:"symbol"`
	Sector string `kuma:"sector"`
}

// Enriched is the result of that join, which is one column from the left, one
// from the right and the key they share.
type Enriched struct {
	Symbol string  `kuma:"symbol"`
	Price  float64 `kuma:"price"`
	Sector string  `kuma:"sector"`
}

// typedSectors is the sectors frame bound to Sector.
func typedSectors(t *testing.T) *kuma.Frame[Sector] {
	t.Helper()

	f, err := kuma.Bind[Sector](sectors(t))
	if err != nil {
		t.Fatalf("Bind: %v", err)
	}
	return f
}

func TestJoinAs(t *testing.T) {
	got, err := typedTrades(t).JoinAs[Enriched](typedSectors(t), kuma.Using("symbol"), kuma.InnerJoin)
	if err != nil {
		t.Fatalf("JoinAs: %v", err)
	}

	if want := []string{"symbol", "price", "sector"}; !slices.Equal(got.Names(), want) {
		t.Errorf("JoinAs gave %v, want %v", got.Names(), want)
	}
	want := []string{
		"AAPL 189.5 hardware",
		"MSFT 411.2 software",
		"AAPL 190.1 hardware",
	}
	if r := rows(t, got); !slices.Equal(r, want) {
		t.Errorf("JoinAs gave %v, want %v", r, want)
	}
}

// TestJoinAsLeavesOutWhatTheStructDoesNotName is the wide join whose result is a
// handful of columns, which is one step here rather than a join and a select.
func TestJoinAsLeavesOutWhatTheStructDoesNotName(t *testing.T) {
	type justSector struct {
		Sector string `kuma:"sector"`
	}

	got, err := trades(t).JoinAs[justSector](sectors(t), kuma.Using("symbol"), kuma.InnerJoin)
	if err != nil {
		t.Fatalf("JoinAs: %v", err)
	}
	if want := []string{"sector"}; !slices.Equal(got.Names(), want) {
		t.Errorf("JoinAs gave %v, want %v", got.Names(), want)
	}
}

func TestJoinAsMistakes(t *testing.T) {
	left, right := trades(t), sectors(t)

	type wanted struct {
		Industry string `kuma:"industry"`
	}
	_, err := left.JoinAs[wanted](right, kuma.Using("symbol"), kuma.InnerJoin)
	if !errors.Is(err, kuma.ErrNoColumn) {
		t.Fatalf("JoinAs = %v, want ErrNoColumn", err)
	}
	if !strings.Contains(err.Error(), "JoinAs") {
		t.Errorf("the message is %q, want it to name the step", err.Error())
	}

	// A semi join takes no columns from the right frame, so a struct that asks
	// for one is asking for a column the step did not produce rather than one
	// that was never there.
	if _, err := left.JoinAs[Enriched](right, kuma.Using("symbol"), kuma.SemiJoin); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("JoinAs over a semi join = %v, want ErrNoColumn", err)
	}

	// The mistake the join itself makes still comes from there.
	if _, err := left.JoinAs[Enriched](right, kuma.Using("ticker"), kuma.InnerJoin); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("JoinAs on a key that is not there = %v, want ErrNoColumn", err)
	}
}

// TestJoinTakesAnySchema is the change that made the typed join possible, which
// is that the frame being joined to no longer has to be a dynamic one.
func TestJoinTakesAnySchema(t *testing.T) {
	got, err := typedTrades(t).InnerJoin(typedSectors(t), "symbol")
	if err != nil {
		t.Fatalf("InnerJoin: %v", err)
	}
	if want := []string{"symbol", "price", "qty", "sector"}; !slices.Equal(got.Names(), want) {
		t.Errorf("InnerJoin gave %v, want %v", got.Names(), want)
	}
}

func TestLazyJoinAs(t *testing.T) {
	left, right := typedTrades(t), typedSectors(t)

	want, err := left.JoinAs[Enriched](right, kuma.Using("symbol"), kuma.InnerJoin)
	if err != nil {
		t.Fatalf("JoinAs: %v", err)
	}

	got, err := left.Lazy().
		JoinAs[Enriched](right.Lazy(), kuma.Using("symbol"), kuma.InnerJoin).
		Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if got.String() != want.String() {
		t.Errorf("the query gave\n%s\nwant\n%s", got, want)
	}
}

// TestLazyJoinAsKeepsTheSchemaType is the point of it, which is a query that
// goes on being written against handles after the join.
func TestLazyJoinAsKeepsTheSchemaType(t *testing.T) {
	price := kuma.NewF64Col[Enriched]("price")

	out, err := typedTrades(t).Lazy().
		JoinAs[Enriched](typedSectors(t).Lazy(), kuma.Using("symbol"), kuma.InnerJoin).
		Filter(price.Gt(200)).
		Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if r := rows(t, out); !slices.Equal(r, []string{"MSFT 411.2 software"}) {
		t.Errorf("the query gave %v", r)
	}
}

// TestLazyJoinAsIsCheckedWhereItIsWritten is the check the join gets the most
// out of, since which columns a join produces is a rule with three parts to it.
func TestLazyJoinAsIsCheckedWhereItIsWritten(t *testing.T) {
	q := trades(t).Lazy().
		JoinAs[Enriched](sectors(t).Lazy(), kuma.Using("symbol"), kuma.SemiJoin)

	if err := q.Validate(); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("Validate = %v, want ErrNoColumn", err)
	}
}

// TestLazyJoinAsString is the plan, which is a projection over the join.
func TestLazyJoinAsString(t *testing.T) {
	got := trades(t).Lazy().
		JoinAs[Enriched](sectors(t).Lazy(), kuma.Using("symbol"), kuma.InnerJoin).
		String()
	want := "Project symbol, price, sector\n  Join inner on symbol\n    Scan frame\n    Scan frame"
	if got != want {
		t.Errorf("the plan is\n%s\nwant\n%s", got, want)
	}
}
