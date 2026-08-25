package kuma_test

import (
	"errors"
	"fmt"
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
	//   symbol: string
	//   price: float64
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

func ExampleFrame_Filter() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "NVDA").Column(),
		kuma.NewSeries("price", 189.5, 411.2, 121.0).Column(),
	)
	if err != nil {
		panic(err)
	}

	prices, err := f.Series[float64]("price")
	if err != nil {
		panic(err)
	}

	// A mask is an ordinary boolean column, so building one by hand is what
	// there is until the expression API arrives.
	keep := make([]bool, prices.Len())
	for i, p := range prices.Values() {
		keep[i] = p > 150
	}

	dear, err := f.Filter(kuma.NewSeries("keep", keep...))
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
