package array_test

import (
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
)

// The columns here are 65536 values, which is the chunk size the executor is
// meant to work in, so a per operation number can be divided by that to get a
// per value cost.
const benchLen = 1 << 16

var (
	intSink   int64
	boolSink  bool
	countSink int
	arraySink *array.Array
)

func benchInts(b *testing.B) *array.Array {
	b.Helper()
	values := make([]int64, benchLen)
	for i := range values {
		values[i] = int64(i)
	}
	return array.Of[int64](values...)
}

// benchIntsWithNulls is the same column with every seventh value missing, which
// is what makes the null count of a slice cost a popcount.
func benchIntsWithNulls(b *testing.B) *array.Array {
	b.Helper()
	valid := bitmap.NewSet(benchLen)
	for i := range benchLen {
		if i%7 == 0 {
			valid.Set(i, false)
		}
	}

	a, err := array.New(dtype.Int64, benchLen, buffer.New(benchLen*8), valid)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	return a
}

// BenchmarkValues is the whole point of the layout. The type check happens once
// and then the loop is a loop over an ordinary Go slice, which is what a kernel
// gets to compile down to.
func BenchmarkValues(b *testing.B) {
	a := benchInts(b)
	b.SetBytes(benchLen * 8)
	for b.Loop() {
		var sum int64
		for _, v := range a.Values[int64]() {
			sum += v
		}
		intSink = sum
	}
}

// BenchmarkValue is the same sum one value at a time, where the type check is
// paid on every element. The gap between this and the one above is the reason
// Values exists at all.
func BenchmarkValue(b *testing.B) {
	a := benchInts(b)
	b.SetBytes(benchLen * 8)
	for b.Loop() {
		var sum int64
		for i := range a.Len() {
			sum += a.Value[int64](i)
		}
		intSink = sum
	}
}

// BenchmarkIsValid is the branch every kernel with nulls takes per value.
func BenchmarkIsValid(b *testing.B) {
	a := benchIntsWithNulls(b)
	for b.Loop() {
		n := 0
		for i := range a.Len() {
			if a.IsValid(i) {
				n++
			}
		}
		countSink = n
	}
}

// BenchmarkIsValidNoNulls is the same loop on a column with no nulls, where the
// answer comes from the null count rather than from a bitmap.
func BenchmarkIsValidNoNulls(b *testing.B) {
	a := benchInts(b)
	for b.Loop() {
		n := 0
		for i := range a.Len() {
			if a.IsValid(i) {
				n++
			}
		}
		countSink = n
	}
}

// BenchmarkSlice is what the executor does to hand a morsel to a worker. It has
// to be small enough that splitting a chunk is not worth thinking about.
func BenchmarkSlice(b *testing.B) {
	a := benchInts(b)
	for b.Loop() {
		arraySink = a.Slice(1000, 9192)
	}
}

// BenchmarkSliceNulls is the same slice on a column that has nulls, where the
// count over the range is the only work there is.
func BenchmarkSliceNulls(b *testing.B) {
	a := benchIntsWithNulls(b)
	for b.Loop() {
		arraySink = a.Slice(1000, 9192)
	}
}

func BenchmarkClone(b *testing.B) {
	a := benchIntsWithNulls(b)
	b.SetBytes(benchLen * 8)
	for b.Loop() {
		arraySink = a.Clone()
	}
}

// BenchmarkBool reads a boolean column one value at a time, which is the price
// of asking about one row.
func BenchmarkBool(b *testing.B) {
	values := make([]bool, benchLen)
	for i := range values {
		values[i] = i%3 == 0
	}
	a := array.OfBools(values...)

	for b.Loop() {
		var v bool
		for i := range a.Len() {
			v = v != a.Bool(i)
		}
		boolSink = v
	}
}

// BenchmarkBools counts the true values a word at a time instead, which is the
// reason booleans are packed into bits in the first place.
func BenchmarkBools(b *testing.B) {
	values := make([]bool, benchLen)
	for i := range values {
		values[i] = i%3 == 0
	}
	a := array.OfBools(values...)

	for b.Loop() {
		countSink = a.Bools().CountOnes()
	}
}

func BenchmarkBytes(b *testing.B) {
	values := make([]string, benchLen)
	for i := range values {
		values[i] = "kuma"
	}
	a := array.OfStrings(values...)

	for b.Loop() {
		n := 0
		for i := range a.Len() {
			n += len(a.Bytes(i))
		}
		countSink = n
	}
}
