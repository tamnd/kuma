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
	intSink     int64
	boolSink    bool
	countSink   int
	arraySink   *array.Array
	chunkedSink *array.Chunked
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

// benchBuilder returns a builder for dt, or fails the benchmark.
func benchBuilder(b *testing.B, dt dtype.DataType) *array.Builder {
	b.Helper()
	bl, err := array.NewBuilder(dt)
	if err != nil {
		b.Fatalf("NewBuilder(%s): %v", dt, err)
	}
	return bl
}

// BenchmarkBuilderAppend builds a chunk one value at a time, which is what a
// row oriented reader such as CSV does.
func BenchmarkBuilderAppend(b *testing.B) {
	bl := benchBuilder(b, dtype.Int64)
	b.SetBytes(benchLen * 8)
	for b.Loop() {
		bl.Grow(benchLen)
		for i := range benchLen {
			bl.Append(int64(i))
		}
		arraySink = bl.Finish()
	}
}

// BenchmarkBuilderAppendNulls is the same loop with every seventh value
// missing, which is what turns the validity bitmap on and makes every append
// after it write a bit.
func BenchmarkBuilderAppendNulls(b *testing.B) {
	bl := benchBuilder(b, dtype.Int64)
	b.SetBytes(benchLen * 8)
	for b.Loop() {
		bl.Grow(benchLen)
		for i := range benchLen {
			if i%7 == 0 {
				bl.AppendNull()
				continue
			}
			bl.Append(int64(i))
		}
		arraySink = bl.Finish()
	}
}

// BenchmarkBuilderAppendValues hands the same values over as one slice, which
// is what a columnar reader such as Parquet or Arrow IPC does. The type is
// checked once and the values are one copy.
func BenchmarkBuilderAppendValues(b *testing.B) {
	values := make([]int64, benchLen)
	for i := range values {
		values[i] = int64(i)
	}

	bl := benchBuilder(b, dtype.Int64)
	b.SetBytes(benchLen * 8)
	for b.Loop() {
		bl.Grow(benchLen)
		bl.AppendValues(values)
		arraySink = bl.Finish()
	}
}

// BenchmarkBuilderAppendString is the string path, where the work is the view
// per value and the copy of anything longer than twelve bytes.
func BenchmarkBuilderAppendString(b *testing.B) {
	bl := benchBuilder(b, dtype.String)
	for b.Loop() {
		bl.Grow(benchLen)
		for range benchLen {
			bl.AppendString("kuma")
		}
		arraySink = bl.Finish()
	}
}

// BenchmarkBuilderAppendBool is the packed path, one bit a value.
func BenchmarkBuilderAppendBool(b *testing.B) {
	bl := benchBuilder(b, dtype.Bool)
	for b.Loop() {
		bl.Grow(benchLen)
		for i := range benchLen {
			bl.AppendBool(i%3 == 0)
		}
		arraySink = bl.Finish()
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

// benchChunked returns the same 65536 values as benchInts, held as chunks of
// 8192, which is the morsel size the executor hands to a worker.
func benchChunked(b *testing.B) *array.Chunked {
	b.Helper()

	const per = 8192
	var chunks []*array.Array
	for start := 0; start < benchLen; start += per {
		values := make([]int64, per)
		for i := range values {
			values[i] = int64(start + i)
		}
		chunks = append(chunks, array.Of[int64](values...))
	}

	c, err := array.NewChunked(dtype.Int64, chunks...)
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	return c
}

// BenchmarkChunkedChunks is the loop a kernel runs, which is the same loop as
// BenchmarkValues once per chunk and should cost about the same per value.
func BenchmarkChunkedChunks(b *testing.B) {
	c := benchChunked(b)
	b.SetBytes(benchLen * 8)
	for b.Loop() {
		var sum int64
		for _, chunk := range c.Chunks() {
			for _, v := range chunk.Values[int64]() {
				sum += v
			}
		}
		intSink = sum
	}
}

// BenchmarkChunkedValue is the same sum asked for one value at a time, where
// every value costs a binary search over the chunks. The gap between this and
// the one above is what the doc comment on Value is warning about.
func BenchmarkChunkedValue(b *testing.B) {
	c := benchChunked(b)
	b.SetBytes(benchLen * 8)
	for b.Loop() {
		var sum int64
		for i := range c.Len() {
			sum += c.Value[int64](i)
		}
		intSink = sum
	}
}

// BenchmarkChunkedSlice cuts a range that spans several chunks, where the work
// is a binary search and two partial chunks however long the range is.
func BenchmarkChunkedSlice(b *testing.B) {
	c := benchChunked(b)
	for b.Loop() {
		chunkedSink = c.Slice(1000, 60000)
	}
}

// benchLists returns a list column of benchLen rows of four elements each,
// which is the shape a repeated field in a file usually has.
func benchLists(b *testing.B) *array.Array {
	b.Helper()

	lb, err := array.NewListBuilder(dtype.List{Elem: dtype.Int64})
	if err != nil {
		b.Fatalf("NewListBuilder: %v", err)
	}
	lb.Grow(benchLen)
	lb.Elem().Grow(benchLen * 4)

	for i := range benchLen {
		lb.Elem().AppendValues([]int64{int64(i), int64(i + 1), int64(i + 2), int64(i + 3)})
		lb.Append()
	}
	return lb.Finish()
}

// BenchmarkListSum reads every element of every row one row at a time, which is
// two offsets and a slice per row and no copying at all.
func BenchmarkListSum(b *testing.B) {
	a := benchLists(b)
	b.SetBytes(benchLen * 4 * 8)
	for b.Loop() {
		var sum int64
		for i := range a.Len() {
			for _, v := range a.List(i).Values[int64]() {
				sum += v
			}
		}
		intSink = sum
	}
}

// BenchmarkListSumChild is the same sum over the child, ignoring where the rows
// begin, which is what a kernel that does not care about them gets to do. The
// gap between this and the one above is the whole cost of the row boundaries.
func BenchmarkListSumChild(b *testing.B) {
	a := benchLists(b)
	b.SetBytes(benchLen * 4 * 8)
	for b.Loop() {
		var sum int64
		for _, v := range a.Child().Values[int64]() {
			sum += v
		}
		intSink = sum
	}
}

// BenchmarkListBuilder is building the column, which is one offset per row and
// the elements going into an ordinary builder.
func BenchmarkListBuilder(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		arraySink = benchLists(b)
	}
}
