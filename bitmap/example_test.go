package bitmap_test

import (
	"fmt"

	"github.com/tamnd/kuma/bitmap"
)

func ExampleBitmap() {
	// A column of five values where the third one is missing.
	valid := bitmap.NewSet(5)
	valid.Set(2, false)

	fmt.Println(valid.Len(), valid.CountOnes())
	fmt.Println(valid.Get(1), valid.Get(2))
	// Output:
	// 5 4
	// true false
}

func ExampleBitmap_And() {
	// Adding two columns: the result is valid only where both inputs were.
	a := bitmap.NewSet(4)
	a.Set(1, false)

	b := bitmap.NewSet(4)
	b.Set(3, false)

	a.And(b)

	for i := range a.Len() {
		fmt.Print(a.Get(i), " ")
	}
	fmt.Println()
	// Output:
	// true false true false
}

func ExampleBitmap_Append() {
	// The zero value is ready to use.
	var b bitmap.Bitmap
	for _, v := range []bool{true, false, true} {
		b.Append(v)
	}

	fmt.Println(b.Len(), b.CountOnes())
	// Output:
	// 3 2
}
