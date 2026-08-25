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
