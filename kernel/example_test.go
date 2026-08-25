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
