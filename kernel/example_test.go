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
