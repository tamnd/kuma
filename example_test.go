package kuma_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/dtype"
)

func ExampleNewSeries() {
	s := kuma.NewSeries("price", 189.5, 411.2, 190.1)

	fmt.Println(s.Name(), s.Len(), s.DType())
	fmt.Println(s.Value(1))
	// Output:
	// price 3 float64
	// 411.2
}

// ExampleSeries_Values shows the door out to a hand written kernel. The slice
// is the column's own memory rather than a copy of it.
func ExampleSeries_Values() {
	s := kuma.NewSeries("qty", int64(100), 50, 25, 400)

	var total int64
	for _, v := range s.Values() {
		total += v
	}
	fmt.Println(total)
	// Output:
	// 575
}

func ExampleSeries_Head() {
	s := kuma.NewSeries("qty", int64(1), 2, 3, 4, 5)

	fmt.Println(s.Head(2).Values())
	fmt.Println(s.Tail(2).Values())
	fmt.Println(s.Head(-1).Values())
	// Output:
	// [1 2]
	// [4 5]
	// [1 2 3 4]
}

// ExampleSeriesFrom shows a column stored as one type being read as another.
// A timestamp is int64 values with a meaning attached, so it reads as a
// time.Time and as the int64 underneath, and neither of those copies anything.
func ExampleSeriesFrom() {
	ts := kuma.NewSeries("ts",
		time.Date(2026, 8, 25, 9, 30, 0, 0, time.UTC),
		time.Date(2026, 8, 25, 9, 31, 0, 0, time.UTC),
	)

	nanos, err := kuma.SeriesFrom[int64]("ts", ts.Data())
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(ts.Value(0).Format(time.RFC3339))
	fmt.Println(nanos.Value(0))
	// Output:
	// 2026-08-25T09:30:00Z
	// 1787650200000000000
}

func ExampleNewFrame() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "NVDA").Column(),
		kuma.NewSeries("price", 189.5, 411.2, 121.0).Column(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(f)
	// Output:
	// kuma.Frame[kuma.Dynamic] 3 rows x 2 cols
	//
	//   symbol |   price
	//   string | float64
	// ---------+--------
	//   AAPL   |   189.5
	//   MSFT   |   411.2
	//   NVDA   |     121
}

func ExampleFrame_Series() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "NVDA").Column(),
		kuma.NewSeries("price", 189.5, 411.2, 121.0).Column(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	prices, err := f.Series[float64]("price")
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(prices.Values())
	// Output:
	// [189.5 411.2 121]
}

func ExampleFrame_Select() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT").Column(),
		kuma.NewSeries("price", 189.5, 411.2).Column(),
		kuma.NewSeries("qty", int64(100), 50).Column(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	out, err := f.Select("qty", "symbol")
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(out.Names())
	// Output:
	// [qty symbol]
}

func ExampleFrame_WithColumn() {
	f, err := kuma.NewFrame(kuma.NewSeries("price", 10.0, 20.0).Column())
	if err != nil {
		fmt.Println(err)
		return
	}

	// The values of a new column are worked out in Go and handed back as a
	// column, which is what an expression will do for you once M3 lands.
	prices, err := f.Series[float64]("price")
	if err != nil {
		fmt.Println(err)
		return
	}
	taxed := make([]float64, prices.Len())
	for i, p := range prices.Values() {
		taxed[i] = p * 1.1
	}

	out, err := f.WithColumn(kuma.NewSeries("taxed", taxed...).Column())
	if err != nil {
		fmt.Println(err)
		return
	}

	taxes, err := out.Series[float64]("taxed")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out.Names(), taxes.Values())
	// Output:
	// [price taxed] [11 22]
}

func ExampleFrame_Head() {
	f, err := kuma.NewFrame(kuma.NewSeries("qty", int64(1), 2, 3, 4, 5).Column())
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(f.Head(2).NumRows(), f.Tail(1).NumRows(), f.Slice(1, 4).NumRows())
	// Output:
	// 2 1 3
}

// ExampleColumnError is the error a wrong column name gives. It lists what the
// frame holds and points at the name that is one letter away, because that is
// what turns a five minute detour into a five second one.
func ExampleColumnError() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL").Column(),
		kuma.NewSeries("price", 189.5).Column(),
		kuma.NewSeries("qty", int64(100)).Column(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	_, err = f.Select("symbol", "prices")
	fmt.Println(err)
	fmt.Println(errors.Is(err, kuma.ErrNoColumn))
	// Output:
	// kuma: column "prices" not found in Select
	//   available: symbol, price, qty
	//   did you mean: price?
	// true
}

func ExampleColumn_As() {
	c := kuma.NewSeries("qty", int64(100), 50).Column()

	if _, err := c.As[float64](); err != nil {
		fmt.Println(err)
	}

	s, err := c.As[int64]()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(s.Values())
	// Output:
	// kuma: column "qty" is a int64 column, which does not read as a float64: wrong type
	// [100 50]
}

func ExampleDTypeOf() {
	fmt.Println(kuma.DTypeOf[float64]())
	fmt.Println(kuma.DTypeOf[time.Time]())
	// Output:
	// float64
	// timestamp[ns, tz=UTC]
}

func ExampleCanRead() {
	// A date is int32 days since the epoch, so it reads as an int32 without
	// anything being copied or converted.
	fmt.Println(kuma.CanRead[int32](dtype.Date32))
	fmt.Println(kuma.CanRead[int64](dtype.Float64))
	// Output:
	// true
	// false
}

func ExampleColumnName() {
	// This is the column a field of each of these names binds to when it carries
	// no kuma tag, and it is the name kumagen writes into the handle.
	fmt.Println(kuma.ColumnName("Price"))
	fmt.Println(kuma.ColumnName("OrderID"))
	fmt.Println(kuma.ColumnName("HTTPCode"))
	// Output:
	// price
	// order_id
	// http_code
}

func ExampleFrame_Filter() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "NVDA").Column(),
		kuma.NewSeries("price", 189.5, 411.2, 121.0).Column(),
	)
	if err != nil {
		panic(err)
	}

	// The handles would normally be a package level variable written once, or
	// generated from the struct the rows are read into.
	symbol, price := kuma.Str("symbol"), kuma.F64("price")

	dear, err := f.Filter(price.Gt(150).And(symbol.Ne("MSFT")))
	if err != nil {
		panic(err)
	}

	symbols, err := symbol.Series(dear)
	if err != nil {
		panic(err)
	}
	fmt.Println(symbols.Values())
	// Output:
	// [AAPL]
}

func ExampleFrame_FilterMask() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "NVDA").Column(),
		kuma.NewSeries("price", 189.5, 411.2, 121.0).Column(),
	)
	if err != nil {
		panic(err)
	}

	// A mask is an ordinary boolean column, which is what a caller who worked
	// out the rows some other way already has.
	prices, err := f.Series[float64]("price")
	if err != nil {
		panic(err)
	}
	keep := make([]bool, prices.Len())
	for i, p := range prices.Values() {
		keep[i] = p > 150
	}

	dear, err := f.FilterMask(kuma.NewSeries("keep", keep...))
	if err != nil {
		panic(err)
	}

	symbols, err := dear.Series[string]("symbol")
	if err != nil {
		panic(err)
	}
	fmt.Println(symbols.Values())
	// Output:
	// [AAPL MSFT]
}

func ExampleFrame_Take() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "NVDA").Column(),
		kuma.NewSeries("qty", int64(100), 50, 400).Column(),
	)
	if err != nil {
		panic(err)
	}

	// The order some other operation worked out, and a position below zero for
	// a row that matched nothing.
	got := f.Take([]int{2, -1, 0})

	symbols, err := got.Series[string]("symbol")
	if err != nil {
		panic(err)
	}
	for i := range symbols.Len() {
		if symbols.IsNull(i) {
			fmt.Println("null")
			continue
		}
		fmt.Println(symbols.Value(i))
	}
	// Output:
	// NVDA
	// null
	// AAPL
}

func ExampleSeries_Take() {
	s := kuma.NewSeries("qty", int64(100), 50, 400)

	fmt.Println(s.Take([]int{2, 0, 2}).Values())
	// Output:
	// [400 100 400]
}

func ExampleSeries_Cast() {
	s := kuma.NewSeries("qty", int64(100), 50, 400)

	// The type argument is what the values come back as, and the argument is
	// what they are stored as.
	small, err := s.Cast[int32](dtype.Int32)
	if err != nil {
		panic(err)
	}
	fmt.Println(small.DType(), small.Values())

	// An int8 holds up to 127, and one of the three rows does not.
	if _, err := s.Cast[int8](dtype.Int8); err != nil {
		fmt.Println(err)
	}
	// Output:
	// int32 [100 50 400]
	// kernel: cannot cast int64 to int8: row 2 is 400: value out of range
}

func ExampleSeries_Cast_asTime() {
	// Seconds since the epoch, which is how a file that came out of a database
	// export usually holds them.
	s := kuma.NewSeries("seen", int64(1767225600), 1767225660)

	seen, err := s.Cast[time.Time](dtype.Timestamp{Unit: dtype.Second, Zone: "UTC"})
	if err != nil {
		panic(err)
	}
	for i := range seen.Len() {
		fmt.Println(seen.Value(i).Format(time.RFC3339))
	}
	// Output:
	// 2026-01-01T00:00:00Z
	// 2026-01-01T00:01:00Z
}

func ExampleColumn_TryCast() {
	c := kuma.NewSeries("qty", "100", "n/a", "400").Column()

	// The row that will not parse becomes a null rather than an error, which is
	// what makes a file of a million rows survive the one that says n/a.
	got, err := c.TryCast(dtype.Int64)
	if err != nil {
		panic(err)
	}

	s, err := got.As[int64]()
	if err != nil {
		panic(err)
	}
	fmt.Println(s.Values(), "with", got.NullCount(), "null")
	// Output:
	// [100 0 400] with 1 null
}

func ExampleFrame_Cast() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT").Column(),
		kuma.NewSeries("qty", int64(100), 50).Column(),
	)
	if err != nil {
		panic(err)
	}

	got, err := f.Cast("qty", dtype.Float64)
	if err != nil {
		panic(err)
	}
	for _, c := range got.Columns() {
		fmt.Println(c.Name(), c.DType())
	}
	// Output:
	// symbol string
	// qty float64
}

func ExampleFrame_SortBy() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "NVDA", "AAPL", "MSFT").Column(),
		kuma.NewSeries("qty", int64(400), 100, 50).Column(),
	)
	if err != nil {
		panic(err)
	}

	got, err := f.SortBy("symbol")
	if err != nil {
		panic(err)
	}

	symbols, err := got.Series[string]("symbol")
	if err != nil {
		panic(err)
	}
	fmt.Println(symbols.Values())
	// Output:
	// [AAPL MSFT NVDA]
}

// The first key decides and the later ones break its ties, and each key runs
// the way it says. This is the trades of each symbol, biggest first.
func ExampleFrame_Sort() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "b", "a", "b", "a").Column(),
		kuma.NewSeries("qty", int64(2), 9, 7, 1).Column(),
	)
	if err != nil {
		panic(err)
	}

	got, err := f.Sort(kuma.Asc("symbol"), kuma.Desc("qty"))
	if err != nil {
		panic(err)
	}

	symbols, err := got.Series[string]("symbol")
	if err != nil {
		panic(err)
	}
	qty, err := got.Series[int64]("qty")
	if err != nil {
		panic(err)
	}
	for i := range got.NumRows() {
		fmt.Println(symbols.Value(i), qty.Value(i))
	}
	// Output:
	// a 9
	// a 1
	// b 7
	// b 2
}

func ExampleSeries_Sort() {
	s := kuma.NewSeries("qty", int64(400), 100, 50)

	got, err := s.Sort(kuma.Order{Descending: true})
	if err != nil {
		panic(err)
	}
	fmt.Println(got.Values())
	// Output:
	// [400 100 50]
}

// SortIndex works out the order without moving anything, which is what to reach
// for when the order is what you are after.
func ExampleSeries_SortIndex() {
	s := kuma.NewSeries("qty", int64(400), 100, 50)

	idx, err := s.SortIndex(kuma.Order{})
	if err != nil {
		panic(err)
	}
	fmt.Println(idx)
	// Output:
	// [2 1 0]
}

func ExampleFrame_GroupBy() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "AAPL", "NVDA", "MSFT").Column(),
		kuma.NewSeries("qty", int64(100), 50, 25, 400, 75).Column(),
		kuma.NewSeries("price", 189.5, 411.2, 190.1, 121.0, 410.0).Column(),
	)
	if err != nil {
		panic(err)
	}

	g, err := f.GroupBy("symbol")
	if err != nil {
		panic(err)
	}
	got, err := g.Agg(
		kuma.Sum("qty").As("total"),
		kuma.Mean("price").As("avg"),
		kuma.Size(),
	)
	if err != nil {
		panic(err)
	}

	fmt.Println(strings.Join(got.Names(), " "))
	for i := range got.NumRows() {
		fmt.Println(string(got.ColumnAt(0).Data().Bytes(i)),
			got.ColumnAt(1).Data().Value[int64](i),
			got.ColumnAt(2).Data().Value[float64](i),
			got.ColumnAt(3).Data().Value[int64](i))
	}
	// Output:
	// symbol total avg size
	// AAPL 125 189.8 2
	// MSFT 125 410.6 2
	// NVDA 400 121 1
}

// Groups come out in the order they first appear, which is deterministic
// without being sorted. Sort the result when the order matters.
func ExampleFrame_GroupBy_sorted() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("region", "west", "east", "west", "north").Column(),
		kuma.NewSeries("sales", 10.0, 40.0, 30.0, 20.0).Column(),
	)
	if err != nil {
		panic(err)
	}

	g, err := f.GroupBy("region")
	if err != nil {
		panic(err)
	}
	totals, err := g.Agg(kuma.Sum("sales"))
	if err != nil {
		panic(err)
	}
	got, err := totals.SortDesc("sales")
	if err != nil {
		panic(err)
	}

	for i := range got.NumRows() {
		fmt.Println(string(got.ColumnAt(0).Data().Bytes(i)),
			got.ColumnAt(1).Data().Value[float64](i))
	}
	// Output:
	// west 40
	// east 40
	// north 20
}

// A grouping is worked out once and answers as many questions as it is asked,
// which is why GroupBy hands one back instead of doing everything at once.
func ExampleGroupedFrame_Agg() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("host", "a", "b", "a", "b", "a").Column(),
		kuma.NewSeries("ms", 12.0, 240.0, 15.0, 11.0, 19.0).Column(),
	)
	if err != nil {
		panic(err)
	}

	g, err := f.GroupBy("host")
	if err != nil {
		panic(err)
	}
	got, err := g.Agg(
		kuma.Median("ms").As("p50"),
		kuma.Quantile("ms", 0.9, kuma.Linear).As("p90"),
		kuma.Max("ms").As("worst"),
		kuma.Count("ms").As("n"),
	)
	if err != nil {
		panic(err)
	}

	fmt.Println(strings.Join(got.Names(), " "))
	for i := range got.NumRows() {
		fmt.Println(string(got.ColumnAt(0).Data().Bytes(i)),
			got.ColumnAt(1).Data().Value[float64](i),
			got.ColumnAt(2).Data().Value[float64](i),
			got.ColumnAt(3).Data().Value[float64](i),
			got.ColumnAt(4).Data().Value[int64](i))
	}
	// Output:
	// host p50 p90 worst n
	// a 15 18.2 19 3
	// b 125.5 217.1 240 2
}

// Aggregating two things about one column needs at least one of them named,
// because otherwise both result columns want to be called price.
func ExampleAggregation_As() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("day", int32(1), 1, 2, 2).Column(),
		kuma.NewSeries("price", 9.0, 11.0, 4.0, 8.0).Column(),
	)
	if err != nil {
		panic(err)
	}

	g, err := f.GroupBy("day")
	if err != nil {
		panic(err)
	}
	if _, clash := g.Agg(kuma.Min("price"), kuma.Max("price")); clash != nil {
		fmt.Println(clash)
	}

	got, err := g.Agg(kuma.Min("price").As("low"), kuma.Max("price").As("high"))
	if err != nil {
		panic(err)
	}
	fmt.Println(strings.Join(got.Names(), " "))
	// Output:
	// kuma: two columns are called "price": duplicate column
	// day low high
}

func ExampleGroupedFrame_Count() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("status", "ok", "ok", "error", "ok", "error").Column(),
	)
	if err != nil {
		panic(err)
	}

	g, err := f.GroupBy("status")
	if err != nil {
		panic(err)
	}
	got, err := g.Count()
	if err != nil {
		panic(err)
	}

	for i := range got.NumRows() {
		fmt.Println(string(got.ColumnAt(0).Data().Bytes(i)),
			got.ColumnAt(1).Data().Value[int64](i))
	}
	// Output:
	// ok 3
	// error 2
}

// A join puts two frames together on the columns they share. This is an inner
// join, so a trade in a symbol the reference data has never heard of is not in
// the answer.
func ExampleFrame_Join() {
	trades, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "AAPL", "NVDA").Column(),
		kuma.NewSeries("qty", int64(100), 50, 25, 400).Column(),
	)
	if err != nil {
		panic(err)
	}
	sectors, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "MSFT", "AAPL").Column(),
		kuma.NewSeries("sector", "software", "hardware").Column(),
	)
	if err != nil {
		panic(err)
	}

	got, err := trades.InnerJoin(sectors, "symbol")
	if err != nil {
		panic(err)
	}

	fmt.Println(strings.Join(got.Names(), " "))
	for i := range got.NumRows() {
		fmt.Println(string(got.ColumnAt(0).Data().Bytes(i)),
			got.ColumnAt(1).Data().Value[int64](i),
			string(got.ColumnAt(2).Data().Bytes(i)))
	}
	// Output:
	// symbol qty sector
	// AAPL 100 hardware
	// MSFT 50 software
	// AAPL 25 hardware
}

// A left join keeps every left row whether it matched or not, and the columns
// that came from the right side are missing where nothing did.
func ExampleFrame_LeftJoin() {
	trades, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "NVDA").Column(),
		kuma.NewSeries("qty", int64(100), 400).Column(),
	)
	if err != nil {
		panic(err)
	}
	sectors, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL").Column(),
		kuma.NewSeries("sector", "hardware").Column(),
	)
	if err != nil {
		panic(err)
	}

	got, err := trades.LeftJoin(sectors, "symbol")
	if err != nil {
		panic(err)
	}

	for i := range got.NumRows() {
		sector := "unknown"
		if !got.ColumnAt(2).IsNull(i) {
			sector = string(got.ColumnAt(2).Data().Bytes(i))
		}
		fmt.Println(string(got.ColumnAt(0).Data().Bytes(i)), sector)
	}
	// Output:
	// AAPL hardware
	// NVDA unknown
}

// A semi join answers which of these have one, and takes nothing from the right
// side. An anti join is the other half of the same question.
func ExampleFrame_Join_semi() {
	orders, err := kuma.NewFrame(
		kuma.NewSeries("id", int64(1), 2, 3).Column(),
		kuma.NewSeries("customer", "ann", "bob", "cat").Column(),
	)
	if err != nil {
		panic(err)
	}
	shipped, err := kuma.NewFrame(kuma.NewSeries("id", int64(3), 1).Column())
	if err != nil {
		panic(err)
	}

	for _, how := range []kuma.JoinType{kuma.SemiJoin, kuma.AntiJoin} {
		got, err := orders.Join(shipped, kuma.Using("id"), how)
		if err != nil {
			panic(err)
		}
		fmt.Print(how, ":")
		for i := range got.NumRows() {
			fmt.Print(" ", string(got.ColumnAt(1).Data().Bytes(i)))
		}
		fmt.Println()
	}
	// Output:
	// semi: ann cat
	// anti: bob
}

// When the two sides call the key different things, name both. Both columns are
// kept, since a caller who wrote two names probably wants to see both.
func ExampleOn() {
	trades, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL").Column(),
		kuma.NewSeries("qty", int64(100)).Column(),
	)
	if err != nil {
		panic(err)
	}
	sectors, err := kuma.NewFrame(
		kuma.NewSeries("ticker", "AAPL").Column(),
		kuma.NewSeries("sector", "hardware").Column(),
	)
	if err != nil {
		panic(err)
	}

	got, err := trades.Join(sectors,
		[]kuma.On{{Left: "symbol", Right: "ticker"}}, kuma.InnerJoin)
	if err != nil {
		panic(err)
	}
	fmt.Println(strings.Join(got.Names(), " "))
	// Output:
	// symbol qty ticker sector
}

// Concat stacks frames on top of each other. Nothing is copied: a column is a
// list of chunks, so stacking two frames puts the two lists together and the
// values stay where they are.
func ExampleConcat() {
	monday, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT").Column(),
		kuma.NewSeries("qty", int64(100), 50).Column(),
	)
	if err != nil {
		panic(err)
	}
	tuesday, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "NVDA").Column(),
		kuma.NewSeries("qty", int64(400)).Column(),
	)
	if err != nil {
		panic(err)
	}

	week, err := kuma.Concat(monday, tuesday)
	if err != nil {
		panic(err)
	}

	qty, err := week.Series[int64]("qty")
	if err != nil {
		panic(err)
	}
	fmt.Println(week.NumRows(), qty.Values())
	// Output:
	// 3 [100 50 400]
}

// ConcatUnion is for frames that do not hold the same columns. The result has
// every column any of them has, and a frame that lacks one contributes nulls.
func ExampleConcatUnion() {
	old, err := kuma.NewFrame(kuma.NewSeries("qty", int64(100), 50).Column())
	if err != nil {
		panic(err)
	}
	later, err := kuma.NewFrame(
		kuma.NewSeries("qty", int64(25)).Column(),
		kuma.NewSeries("fee", 0.5).Column(),
	)
	if err != nil {
		panic(err)
	}

	got, err := kuma.ConcatUnion(old, later)
	if err != nil {
		panic(err)
	}

	fmt.Println(got.Names())
	for i := range got.NumRows() {
		if got.ColumnAt(1).IsNull(i) {
			fmt.Println("no fee")
			continue
		}
		fmt.Println(got.ColumnAt(1).Data().Value[float64](i))
	}
	// Output:
	// [qty fee]
	// no fee
	// no fee
	// 0.5
}

// HStack puts frames side by side, matching them up by position. Use a join
// when the rows should be matched by a key instead.
func ExampleHStack() {
	symbols, err := kuma.NewFrame(kuma.NewSeries("symbol", "AAPL", "MSFT").Column())
	if err != nil {
		panic(err)
	}
	prices, err := kuma.NewFrame(kuma.NewSeries("price", 189.5, 411.2).Column())
	if err != nil {
		panic(err)
	}

	got, err := kuma.HStack(symbols, prices)
	if err != nil {
		panic(err)
	}
	fmt.Println(got.Names(), got.NumRows())
	// Output:
	// [symbol price] 2
}

// withGaps returns a frame with a hole in it, built the way a hole usually
// turns up, which is data from two places that do not agree on the columns.
func withGaps() *kuma.Frame[kuma.Dynamic] {
	old, err := kuma.NewFrame(kuma.NewSeries("qty", int64(100), 50).Column())
	if err != nil {
		panic(err)
	}
	later, err := kuma.NewFrame(
		kuma.NewSeries("qty", int64(25)).Column(),
		kuma.NewSeries("fee", 0.5).Column(),
	)
	if err != nil {
		panic(err)
	}

	f, err := kuma.ConcatUnion(old, later)
	if err != nil {
		panic(err)
	}
	return f
}

func ExampleFrame_FillNull() {
	got, err := withGaps().FillNull("fee", 0.0)
	if err != nil {
		panic(err)
	}

	fees, err := got.Series[float64]("fee")
	if err != nil {
		panic(err)
	}
	fmt.Println(fees.Values(), got.HasNulls())
	// Output:
	// [0 0 0.5] false
}

func ExampleFrame_DropNulls() {
	got, err := withGaps().DropNulls()
	if err != nil {
		panic(err)
	}

	qty, err := got.Series[int64]("qty")
	if err != nil {
		panic(err)
	}
	fmt.Println(qty.Values())
	// Output:
	// [25]
}

func ExampleFrame_KeepAtLeast() {
	// One value present out of the two columns is enough, which is the pandas
	// how="all" and keeps every row that is not entirely empty.
	got, err := withGaps().KeepAtLeast(1)
	if err != nil {
		panic(err)
	}
	fmt.Println(got.NumRows())
	// Output:
	// 3
}

func ExampleFrame_NullCounts() {
	f := withGaps()
	fmt.Println(f.Names(), f.NullCounts())
	// Output:
	// [qty fee] [0 2]
}

func ExampleSeries_FillNull() {
	fees, err := withGaps().Series[float64]("fee")
	if err != nil {
		panic(err)
	}
	fmt.Println(fees.FillNull(0.25).Values())
	// Output:
	// [0.25 0.25 0.5]
}

func ExampleSeries_ValidMask() {
	f := withGaps()

	fees, err := f.Series[float64]("fee")
	if err != nil {
		panic(err)
	}

	// The mask goes straight back into Filter, which is the whole point of it
	// being a series rather than a slice of bool.
	got, err := f.FilterMask(fees.ValidMask())
	if err != nil {
		panic(err)
	}
	fmt.Println(got.NumRows())
	// Output:
	// 1
}

func ExampleReadCSV() {
	in := `sym,qty,px
AAPL,100,182.5
MSFT,,411.2
GOOG,300,141.8
`

	f, err := kuma.ReadCSV(strings.NewReader(in), nil)
	if err != nil {
		panic(err)
	}
	fmt.Println(f)

	// The columns arrive as themselves, so this is a sum and not a parse.
	qty, err := f.Series[int64]("qty")
	if err != nil {
		panic(err)
	}
	fmt.Println(qty.DropNulls().Values())
	// Output:
	// kuma.Frame[kuma.Dynamic] 3 rows x 3 cols
	//
	//   sym    |   qty |      px
	//   string | int64 | float64
	// ---------+-------+--------
	//   AAPL   |   100 |   182.5
	//   MSFT   |  null |   411.2
	//   GOOG   |   300 |   141.8
	// [100 300]
}

func ExampleFrame_WriteCSV() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("sym", "AAPL", "MSFT", "GOOG").Column(),
		kuma.NewSeries[int64]("qty", 100, 200, 300).Column(),
	)
	if err != nil {
		panic(err)
	}

	if err := f.WriteCSV(os.Stdout, nil); err != nil {
		panic(err)
	}
	// Output:
	// sym,qty
	// AAPL,100
	// MSFT,200
	// GOOG,300
}

func ExampleReadDataset() {
	// A dataset is a tree of files whose directories are named key=value, which
	// is the layout Hive wrote and every engine since has read. The directory
	// names are data, so the year and the month below are columns of the frame
	// and are in none of the files.
	root, err := os.MkdirTemp("", "orders")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.RemoveAll(root)

	for dir, rows := range map[string]string{
		"year=2024/month=01": `{"sym":"AAPL","qty":100}` + "\n",
		"year=2024/month=02": `{"sym":"MSFT","qty":50}` + "\n",
		"year=2025/month=01": `{"sym":"GOOG","qty":25}` + "\n",
	} {
		if err = os.MkdirAll(filepath.Join(root, dir), 0o750); err != nil {
			fmt.Println(err)
			return
		}
		p := filepath.Join(root, dir, "part-0.ndjson")
		if err = os.WriteFile(p, []byte(rows), 0o600); err != nil {
			fmt.Println(err)
			return
		}
	}

	f, err := kuma.ReadDataset(root, nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(f.Names())
	if err = f.WriteCSV(os.Stdout, nil); err != nil {
		fmt.Println(err)
	}
	// Output:
	// [sym qty year month]
	// sym,qty,year,month
	// AAPL,100,2024,01
	// MSFT,50,2024,02
	// GOOG,25,2025,01
}

// ExampleFrame_Lazy shows a query written down in one expression and run at the
// end. What comes back is what the eager methods give, worked out from a plan
// that could be printed, checked or optimized before any of it ran.
func ExampleFrame_Lazy() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "AAPL", "NVDA").Column(),
		kuma.NewSeries("price", 189.5, 411.2, 190.1, 121.0).Column(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	q := f.Lazy().Filter(kuma.F64("price").Gt(150)).SortDesc("price").Head(2)
	fmt.Println(q)

	out, err := q.Collect(context.Background())
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out)
	// Output:
	// Limit 2
	//   Sort by price desc
	//     Filter (price > 150)
	//       Scan frame
	// kuma.Frame[kuma.Dynamic] 2 rows x 2 cols
	//
	//   symbol |   price
	//   string | float64
	// ---------+--------
	//   MSFT   |   411.2
	//   AAPL   |   190.1
}

// ExampleLazyFrame_Schema shows what a query would produce being asked for
// before it runs, and a query that names a column that is not there being
// turned away without anything being read.
func ExampleLazyFrame_Schema() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT").Column(),
		kuma.NewSeries("price", 189.5, 411.2).Column(),
		kuma.NewSeries("qty", int64(100), 50).Column(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	s, err := f.Lazy().Drop("qty").With("half", kuma.F64("price").Div(2)).Schema()
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(s)

	_, err = f.Lazy().Filter(kuma.F64("prcie").Gt(150)).Schema()
	fmt.Println(err)
	fmt.Println(errors.Is(err, kuma.ErrNoColumn))
	// Output:
	// schema<symbol: string not null, price: float64 not null, half: float64>
	// kuma: column "prcie" not found in Filter
	//   available: symbol, price, qty
	// true
}

// ExampleLazyGroupBy_Agg shows a group by written as part of a query, which
// gives the same table the eager [kuma.GroupedFrame.Agg] does and does none of
// the work until the query is collected.
func ExampleLazyGroupBy_Agg() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "AAPL", "NVDA").Column(),
		kuma.NewSeries("price", 189.5, 411.2, 190.1, 121.0).Column(),
		kuma.NewSeries("qty", int64(100), 50, 25, 400).Column(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	q := f.Lazy().GroupBy("symbol").Agg(
		kuma.Sum("qty").As("total"),
		kuma.Mean("price").As("avg"),
		kuma.Size(),
	)
	fmt.Println(q)

	out, err := q.Collect(context.Background())
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out)
	// Output:
	// Aggregate by symbol: Sum(qty) as total, Mean(price) as avg, Size() as size
	//   Scan frame
	// kuma.Frame[kuma.Dynamic] 3 rows x 4 cols
	//
	//   symbol | total |     avg |  size
	//   string | int64 | float64 | int64
	// ---------+-------+---------+------
	//   AAPL   |   125 |   189.8 |     2
	//   MSFT   |    50 |   411.2 |     1
	//   NVDA   |   400 |     121 |     1
}

// ExampleLazyFrame_InnerJoin shows a join where the right side is a query of
// its own. The filter on it runs before the join sees it, so the join is over
// the rows that survived rather than over everything the frame held.
func ExampleLazyFrame_InnerJoin() {
	trades, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "NVDA").Column(),
		kuma.NewSeries("qty", int64(100), 50, 400).Column(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}
	sectors, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "NVDA").Column(),
		kuma.NewSeries("sector", "hardware", "software", "hardware").Column(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	out, err := trades.Lazy().
		InnerJoin(sectors.Lazy().Filter(kuma.Str("sector").Eq("hardware")), "symbol").
		Collect(context.Background())
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out)
	// Output:
	// kuma.Frame[kuma.Dynamic] 2 rows x 3 cols
	//
	//   symbol |   qty | sector
	//   string | int64 | string
	// ---------+-------+---------
	//   AAPL   |   100 | hardware
	//   NVDA   |   400 | hardware
}

// ExampleFrame_Distinct shows a drop duplicates over one column, which keeps
// the first row of each symbol and the values that row carried.
func ExampleFrame_Distinct() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "AAPL", "MSFT").Column(),
		kuma.NewSeries("venue", "NYSE", "NASDAQ", "NASDAQ", "NASDAQ").Column(),
		kuma.NewSeries("qty", int64(100), 50, 25, 400).Column(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	once, err := f.Distinct("symbol")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(once)

	pairs, err := f.Distinct("symbol", "venue")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(pairs)
	// Output:
	// kuma.Frame[kuma.Dynamic] 2 rows x 3 cols
	//
	//   symbol | venue  |   qty
	//   string | string | int64
	// ---------+--------+------
	//   AAPL   | NYSE   |   100
	//   MSFT   | NASDAQ |    50
	// kuma.Frame[kuma.Dynamic] 3 rows x 3 cols
	//
	//   symbol | venue  |   qty
	//   string | string | int64
	// ---------+--------+------
	//   AAPL   | NYSE   |   100
	//   MSFT   | NASDAQ |    50
	//   AAPL   | NASDAQ |    25
}

// ExampleLazyFrame_Distinct shows the same thing written as a query, over the
// columns that were selected rather than over all of them.
func ExampleLazyFrame_Distinct() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "AAPL", "NVDA").Column(),
		kuma.NewSeries("qty", int64(100), 50, 25, 400).Column(),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	q := f.Lazy().Select("symbol").Distinct()
	fmt.Println(q)

	out, err := q.Collect(context.Background())
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(out)
	// Output:
	// Distinct
	//   Project symbol
	//     Scan frame
	// kuma.Frame[kuma.Dynamic] 3 rows x 1 cols
	//
	//   symbol
	//   string
	// --------
	//   AAPL
	//   MSFT
	//   NVDA
}
