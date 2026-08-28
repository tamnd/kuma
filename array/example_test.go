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

// ExampleNewBuilder is the way a reader builds a column: make room for the
// chunk, append values and nulls in the order they arrive, and finish.
func ExampleNewBuilder() {
	b, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		fmt.Println(err)
		return
	}

	b.Grow(4)
	b.AppendValues([]int64{10, 20})
	b.AppendNull()
	b.Append[int64](40)

	a := b.Finish()
	fmt.Println(a)
	for i := range a.Len() {
		if a.IsNull(i) {
			fmt.Println("null")
			continue
		}
		fmt.Println(a.Value[int64](i))
	}
	// Output:
	// array.Array{int64, len 4, nulls 1, offset 0}
	// 10
	// 20
	// null
	// 40
}

// ExampleBuilder_Finish shows what makes a builder worth reusing: it comes back
// empty, so the next column starts from nothing rather than from memory the
// last one is still reading.
func ExampleBuilder_Finish() {
	b, err := array.NewBuilder(dtype.String)
	if err != nil {
		fmt.Println(err)
		return
	}

	for _, chunk := range [][]string{{"kuma", "bear"}, {"one", "two", "three"}} {
		for _, s := range chunk {
			b.AppendString(s)
		}
		fmt.Println(b.Finish())
	}
	// Output:
	// array.Array{string, len 2, nulls 0, offset 0}
	// array.Array{string, len 3, nulls 0, offset 0}
}

func ExampleNewNull() {
	a := array.NewNull(1000)

	fmt.Println(a)
	fmt.Println(a.IsValid(0), a.Validity() == nil)
	// Output:
	// array.Array{null, len 1000, nulls 1000, offset 0}
	// false true
}

// ExampleNewChunked shows the shape a reader produces: one array per batch, all
// of them one column.
func ExampleNewChunked() {
	c, err := array.NewChunked(dtype.Int64,
		array.Of[int64](1, 2, 3),
		array.Of[int64](4, 5),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(c)
	fmt.Println(c.Len(), c.Value[int64](4))
	// Output:
	// array.Chunked{int64, len 5, nulls 0, chunks 2}
	// 5 5
}

// ExampleChunked_Slice shows what a slice of a chunked column costs. The chunks
// the range covers whole are the same arrays, shared rather than copied, and
// only the ones at the ends are cut.
func ExampleChunked_Slice() {
	c, err := array.NewChunked(dtype.Int64,
		array.Of[int64](0, 1, 2),
		array.Of[int64](3, 4, 5),
		array.Of[int64](6, 7, 8),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	s := c.Slice(1, 8)
	fmt.Println(s)
	fmt.Println(s.Chunk(1) == c.Chunk(1))
	for _, chunk := range s.Chunks() {
		fmt.Println(chunk.Values[int64]())
	}
	// Output:
	// array.Chunked{int64, len 7, nulls 0, chunks 3}
	// true
	// [1 2]
	// [3 4 5]
	// [6 7]
}

// ExampleChunked_Chunks is the loop a kernel runs. Each chunk is a plain Go
// slice, and the work happens there rather than one value at a time through the
// column.
func ExampleChunked_Chunks() {
	c, err := array.NewChunked(dtype.Float64,
		array.Of[float64](1.5, 2.5),
		array.Of[float64](3.5, 4.5, 5.5),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	sum := 0.0
	for _, chunk := range c.Chunks() {
		for _, v := range chunk.Values[float64]() {
			sum += v
		}
	}
	fmt.Println(sum)
	// Output: 17.5
}

// ExampleNewDictionary shows a column stored as indices into a shared set of
// values, which is what a Parquet file mostly holds and what pandas calls a
// Categorical.
func ExampleNewDictionary() {
	regions, err := array.NewDictionary(
		array.Of[int32](1, 0, 0, 1, 2),
		array.OfStrings("north", "south", "east"),
	)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(regions.Len(), regions.Dictionary().Len())
	for i := range regions.Len() {
		fmt.Print(string(regions.Dictionary().Bytes(regions.Index(i))), " ")
	}
	fmt.Println()
	// Output:
	// 5 3
	// south north north south east
}
