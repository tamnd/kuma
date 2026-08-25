package kernel_test

import (
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

func ExampleTake() {
	prices, _ := array.NewChunked(dtype.Float64, array.Of(189.5, 411.2, 190.1, 12.75))

	// The order a sort or a join worked out, somewhere else.
	got := kernel.Take(prices, []int{3, 0, 2})
	for i := range got.Len() {
		fmt.Println(got.Value[float64](i))
	}
	// Output:
	// 12.75
	// 189.5
	// 190.1
}

// A position below zero is a null in the result, which is how an outer join
// says that a row matched nothing.
func ExampleTake_unmatched() {
	names, _ := array.NewChunked(dtype.String, array.OfStrings("AAPL", "MSFT"))

	got := kernel.Take(names, []int{1, -1, 0})
	for i := range got.Len() {
		if got.IsNull(i) {
			fmt.Println("null")
			continue
		}
		fmt.Println(string(got.Bytes(i)))
	}
	// Output:
	// MSFT
	// null
	// AAPL
}

func ExampleFilter() {
	prices, _ := array.NewChunked(dtype.Float64, array.Of(189.5, 411.2, 190.1))
	mask, _ := array.NewChunked(dtype.Bool, array.OfBools(true, false, true))

	got := kernel.Filter(prices, mask)
	for i := range got.Len() {
		fmt.Println(got.Value[float64](i))
	}
	// Output:
	// 189.5
	// 190.1
}

func ExampleIndices() {
	mask, _ := array.NewChunked(dtype.Bool, array.OfBools(false, true, true, false, true))

	fmt.Println(kernel.Indices(mask))
	// Output:
	// [1 2 4]
}

func ExampleCast() {
	read, _ := array.NewChunked(dtype.String, array.OfStrings("189", "411", "190"))

	prices, err := kernel.Cast(read, dtype.Int32)
	if err != nil {
		fmt.Println(err)
		return
	}
	for i := range prices.Len() {
		fmt.Println(prices.Value[int32](i))
	}
	// Output:
	// 189
	// 411
	// 190
}

// A value with no answer in the new type stops the cast and says which row it
// was, which is what makes a bad file findable.
func ExampleCast_doesNotFit() {
	read, _ := array.NewChunked(dtype.String, array.OfStrings("189", "n/a", "190"))

	_, err := kernel.Cast(read, dtype.Int32)
	fmt.Println(err)
	// Output:
	// kernel: cannot cast string to int32: row 1 is "n/a": invalid syntax
}

// TryCast is the same cast with the bad row becoming a null, which is what to
// reach for when the plan is to count them afterwards.
func ExampleTryCast() {
	read, _ := array.NewChunked(dtype.String, array.OfStrings("189", "n/a", "190"))

	prices, err := kernel.TryCast(read, dtype.Int32)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(prices.Len(), "values,", prices.NullCount(), "of them missing")
	// Output:
	// 3 values, 1 of them missing
}

// SortIndex works out the order and Take applies it, which is how a sort of a
// table is one order applied to every column.
func ExampleSortIndex() {
	symbol, _ := array.NewChunked(dtype.String, array.OfStrings("NVDA", "AAPL", "MSFT"))

	idx, err := kernel.SortIndex(kernel.Order{Column: symbol})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(idx)

	sorted := kernel.Take(symbol, idx)
	for i := range sorted.Len() {
		fmt.Println(string(sorted.Bytes(i)))
	}
	// Output:
	// [1 2 0]
	// AAPL
	// MSFT
	// NVDA
}

// The first key decides and the later ones break its ties, and each key has its
// own direction.
func ExampleSortIndex_severalKeys() {
	symbol, _ := array.NewChunked(dtype.String, array.OfStrings("b", "a", "b", "a"))
	qty, _ := array.NewChunked(dtype.Int64, array.Of[int64](2, 9, 7, 1))

	idx, err := kernel.SortIndex(
		kernel.Order{Column: symbol},
		kernel.Order{Column: qty, Descending: true},
	)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(idx)
	// Output:
	// [1 3 2 0]
}

func ExampleGroupBy() {
	symbol, err := array.NewChunked(dtype.String,
		array.OfStrings("NVDA", "AAPL", "NVDA", "AAPL", "NVDA"))
	if err != nil {
		panic(err)
	}
	qty, err := array.NewChunked(dtype.Int64, array.Of[int64](10, 3, 20, 4, 30))
	if err != nil {
		panic(err)
	}

	g, err := kernel.GroupBy(symbol)
	if err != nil {
		panic(err)
	}
	total, err := kernel.Sum(qty, g)
	if err != nil {
		panic(err)
	}

	for i := range g.NumGroups() {
		fmt.Println(string(g.Keys()[0].Bytes(i)), total.Value[int64](i))
	}
	// Output:
	// NVDA 60
	// AAPL 7
}

// A grouping is worked out once and handed to as many aggregations as the
// caller wants, which is the reason it is a value rather than an argument.
func ExampleGroups() {
	day, err := array.NewChunked(dtype.Int32, array.Of[int32](1, 1, 2, 2, 2))
	if err != nil {
		panic(err)
	}
	price, err := array.NewChunked(dtype.Float64, array.Of(9.0, 11.0, 4.0, 6.0, 8.0))
	if err != nil {
		panic(err)
	}

	g, err := kernel.GroupBy(day)
	if err != nil {
		panic(err)
	}
	mean, err := kernel.Mean(price, g)
	if err != nil {
		panic(err)
	}
	high, err := kernel.Max(price, g)
	if err != nil {
		panic(err)
	}
	n := kernel.Count(price, g)

	for i := range g.NumGroups() {
		fmt.Println(g.Keys()[0].Value[int32](i), mean.Value[float64](i),
			high.Value[float64](i), n.Value[int64](i))
	}
	// Output:
	// 1 10 11 2
	// 2 6 8 3
}

func ExampleQuantile() {
	latency, err := array.NewChunked(dtype.Float64,
		array.Of(12.0, 19.0, 15.0, 240.0, 14.0, 17.0, 13.0, 16.0, 18.0, 11.0))
	if err != nil {
		panic(err)
	}

	whole := kernel.OneGroup(latency.Len())
	for _, q := range []float64{0.5, 0.9, 0.99} {
		got, err := kernel.Quantile(latency, whole, q, kernel.Linear)
		if err != nil {
			panic(err)
		}
		fmt.Printf("%.2f %.2f\n", q, got.Value[float64](0))
	}
	// Output:
	// 0.50 15.50
	// 0.90 41.10
	// 0.99 220.11
}

// The five interpolations differ only when the quantile falls between two
// values, which is most of the time on a small group. Lower and higher pick one
// of the neighbors, midpoint splits them evenly whatever the fraction was, and
// nearest picks the closer one.
func ExampleInterpolation() {
	c, err := array.NewChunked(dtype.Float64, array.Of(1.0, 2.0, 3.0, 4.0))
	if err != nil {
		panic(err)
	}

	g := kernel.OneGroup(c.Len())
	for _, how := range []kernel.Interpolation{
		kernel.Linear, kernel.Lower, kernel.Higher, kernel.Nearest, kernel.Midpoint,
	} {
		got, err := kernel.Quantile(c, g, 0.25, how)
		if err != nil {
			panic(err)
		}
		fmt.Println(how, got.Value[float64](0))
	}
	// Output:
	// linear 1.75
	// lower 1
	// higher 2
	// nearest 2
	// midpoint 1.5
}

// A ddof of one is the sample standard deviation and a ddof of zero is the
// population one. The first divides by the number of values less one, which is
// the right thing when the values are a sample of something larger, and it is
// what pandas and Polars do when nobody says otherwise.
func ExampleStd() {
	c, err := array.NewChunked(dtype.Float64, array.Of(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0))
	if err != nil {
		panic(err)
	}

	g := kernel.OneGroup(c.Len())
	sample, err := kernel.Std(c, g, 1)
	if err != nil {
		panic(err)
	}
	population, err := kernel.Std(c, g, 0)
	if err != nil {
		panic(err)
	}

	fmt.Println(sample.Value[float64](0), population.Value[float64](0))
	// Output:
	// 2.138089935299395 2
}

func ExampleNUnique() {
	visitor, err := array.NewChunked(dtype.String,
		array.OfStrings("ana", "bo", "ana", "cy", "bo", "ana"))
	if err != nil {
		panic(err)
	}
	day, err := array.NewChunked(dtype.Int32, array.Of[int32](1, 1, 1, 2, 2, 2))
	if err != nil {
		panic(err)
	}

	g, err := kernel.GroupBy(day)
	if err != nil {
		panic(err)
	}
	seen, err := kernel.NUnique(visitor, g)
	if err != nil {
		panic(err)
	}
	n := kernel.Count(visitor, g)

	for i := range g.NumGroups() {
		fmt.Println(g.Keys()[0].Value[int32](i), n.Value[int64](i), seen.Value[int64](i))
	}
	// Output:
	// 1 3 2
	// 2 3 3
}

func ExampleJoin() {
	trades, err := array.NewChunked(dtype.String,
		array.OfStrings("AAPL", "MSFT", "TSLA"))
	if err != nil {
		panic(err)
	}
	listed, err := array.NewChunked(dtype.String, array.OfStrings("MSFT", "AAPL"))
	if err != nil {
		panic(err)
	}
	sector, err := array.NewChunked(dtype.String, array.OfStrings("software", "hardware"))
	if err != nil {
		panic(err)
	}

	p, err := kernel.Join(
		kernel.Side{Rows: trades.Len(), Keys: []*array.Chunked{trades}},
		kernel.Side{Rows: listed.Len(), Keys: []*array.Chunked{listed}},
		kernel.LeftJoin)
	if err != nil {
		panic(err)
	}

	// A position below zero is a null, so the row that matched nothing needs no
	// handling of its own.
	symbols := kernel.Take(trades, p.Left)
	sectors := kernel.Take(sector, p.Right)
	for i := range p.Len() {
		if sectors.IsNull(i) {
			fmt.Println(string(symbols.Bytes(i)), "unknown")
			continue
		}
		fmt.Println(string(symbols.Bytes(i)), string(sectors.Bytes(i)))
	}
	// Output:
	// AAPL hardware
	// MSFT software
	// TSLA unknown
}

// A missing key matches nothing, including another missing key, which is what
// SQL says. It is not what pandas does, where merging on a column with blanks
// in it pairs the blanks up with each other.
func ExampleJoin_missingKeys() {
	left, err := array.NewChunked(dtype.Int64, array.Of[int64](1, 2))
	if err != nil {
		panic(err)
	}

	b, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		panic(err)
	}
	b.AppendNull()
	b.Append[int64](2)
	right, err := array.NewChunked(dtype.Int64, b.Finish())
	if err != nil {
		panic(err)
	}

	// The left side has no null in it, so this is the right side's null being
	// asked to match, and the only pair is the two twos.
	p, err := kernel.Join(
		kernel.Side{Rows: left.Len(), Keys: []*array.Chunked{left}},
		kernel.Side{Rows: right.Len(), Keys: []*array.Chunked{right}},
		kernel.InnerJoin)
	if err != nil {
		panic(err)
	}
	fmt.Println(p.Left, p.Right)
	// Output:
	// [1] [1]
}

// A cross join takes row counts rather than keys, because it looks at no values
// at all. It is the one join that turns two small tables into a large one, so
// it has to be asked for by name.
func ExampleJoin_cross() {
	p, err := kernel.Join(kernel.Side{Rows: 2}, kernel.Side{Rows: 3}, kernel.CrossJoin)
	if err != nil {
		panic(err)
	}
	fmt.Println(p.Len(), p.Left, p.Right)
	// Output:
	// 6 [0 0 0 1 1 1] [0 1 2 0 1 2]
}

func ExampleIsNull() {
	b, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		panic(err)
	}
	b.Append(int64(1))
	b.AppendNull()
	b.Append(int64(3))

	qty, err := array.NewChunked(dtype.Int64, b.Finish())
	if err != nil {
		panic(err)
	}

	missing := kernel.IsNull(qty)
	for i := range missing.Len() {
		fmt.Println(missing.Bool(i))
	}
	// Output:
	// false
	// true
	// false
}

func ExampleFillNull() {
	b, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		panic(err)
	}
	b.Append(int64(1))
	b.AppendNull()
	b.Append(int64(3))

	qty, err := array.NewChunked(dtype.Int64, b.Finish())
	if err != nil {
		panic(err)
	}

	filled, err := kernel.FillNull(qty, array.Of(int64(0)))
	if err != nil {
		panic(err)
	}
	for i := range filled.Len() {
		fmt.Println(filled.Value[int64](i))
	}
	// Output:
	// 1
	// 0
	// 3
}

// KeepIndex answers over several columns at once, which is what makes it one
// call rather than a mask per column and then an and.
func ExampleKeepIndex() {
	b, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		panic(err)
	}
	b.Append(int64(1))
	b.AppendNull()
	b.Append(int64(3))
	qty, err := array.NewChunked(dtype.Int64, b.Finish())
	if err != nil {
		panic(err)
	}

	b.AppendNull()
	b.Append(int64(2))
	b.Append(int64(3))
	fee, err := array.NewChunked(dtype.Int64, b.Finish())
	if err != nil {
		panic(err)
	}

	cols := []*array.Chunked{qty, fee}
	fmt.Println(kernel.KeepIndex(cols, 3, 2))
	fmt.Println(kernel.KeepIndex(cols, 3, 1))
	// Output:
	// [2]
	// [0 1 2]
}
