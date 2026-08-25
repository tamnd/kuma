package array_test

import (
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
)

func ExampleOf() {
	a := array.Of[int64](10, 20, 30)

	fmt.Println(a)
	fmt.Println(a.Values[int64]())
	// Output:
	// array.Array{int64, len 3, nulls 0, offset 0}
	// [10 20 30]
}

// ExampleNew builds a column the way a reader would, by filling a buffer and a
// validity bitmap and handing both over.
func ExampleNew() {
	values := buffer.New(4 * 8)
	for i, v := range []int64{1, 0, 3, 4} {
		for k := range 8 {
			values.Bytes()[i*8+k] = byte(uint64(v) >> (8 * k))
		}
	}

	valid := bitmap.NewSet(4)
	valid.Set(1, false) // the second value is missing

	a, err := array.New(dtype.Int64, 4, values, valid)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(a.Len(), a.NullCount())
	for i := range a.Len() {
		if a.IsNull(i) {
			fmt.Println("null")
			continue
		}
		fmt.Println(a.Value[int64](i))
	}
	// Output:
	// 4 1
	// 1
	// null
	// 3
	// 4
}

// ExampleArray_Slice shows the two things a slice does: it shares the memory it
// came from, and it recounts the nulls in its own range.
func ExampleArray_Slice() {
	valid := bitmap.NewSet(6)
	valid.Set(0, false)
	valid.Set(4, false)

	a, err := array.New(dtype.Int64, 6, buffer.New(6*8), valid)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(a)
	fmt.Println(a.Slice(1, 4))
	fmt.Println(a.Slice(1, 4).Slice(2, 3))
	// Output:
	// array.Array{int64, len 6, nulls 2, offset 0}
	// array.Array{int64, len 3, nulls 0, offset 1}
	// array.Array{int64, len 1, nulls 0, offset 3}
}

func ExampleArray_Values() {
	a := array.Of[float64](1.5, 2.5, 3.5, 4.5)

	sum := 0.0
	for _, v := range a.Slice(1, 3).Values[float64]() {
		sum += v
	}
	fmt.Println(sum)
	// Output: 6
}

func ExampleOfStrings() {
	a := array.OfStrings("kuma", "bear", "a value that is too long to live inside its view")

	for i := range a.Len() {
		fmt.Printf("%d %s\n", len(a.Bytes(i)), a.Bytes(i))
	}
	// Output:
	// 4 kuma
	// 4 bear
	// 48 a value that is too long to live inside its view
}

func ExampleNewNull() {
	a := array.NewNull(1000)

	fmt.Println(a)
	fmt.Println(a.IsValid(0), a.Validity() == nil)
	// Output:
	// array.Array{null, len 1000, nulls 1000, offset 0}
	// false true
}
