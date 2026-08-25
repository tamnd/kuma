package kernel_test

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// benchLen is the same length the root package benchmarks use, so that a number
// from here is comparable with a number from there.
const benchLen = 1 << 16

var chunkedSink *array.Chunked

var indicesSink []int

var indexSink []int

// benchInts returns a column of benchLen int64 values in the given number of
// chunks, with no nulls.
func benchInts(b *testing.B, chunks int) *array.Chunked {
	b.Helper()

	bd, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		b.Fatalf("NewBuilder: %v", err)
	}

	per := benchLen / chunks
	arrays := make([]*array.Array, chunks)
	for c := range chunks {
		for i := range per {
			bd.Append(int64(c*per + i))
		}
		arrays[c] = bd.Finish()
	}

	out, err := array.NewChunked(dtype.Int64, arrays...)
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	return out
}

// benchStrings returns a column of benchLen short strings, which is the case
// that goes through the byte gather rather than the value gather.
func benchStrings(b *testing.B) *array.Chunked {
	b.Helper()

	bd, err := array.NewBuilder(dtype.String)
	if err != nil {
		b.Fatalf("NewBuilder: %v", err)
	}
	for i := range benchLen {
		bd.AppendString(string(rune('a' + i%26)))
	}

	out, err := array.NewChunked(dtype.String, bd.Finish())
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	return out
}

// order returns a position for every value of a benchmark column. Sorted
// positions are what a filter or a merge produces and shuffled ones are what a
// sort or a hash join produces, and the two cost different amounts on real
// hardware, so both are measured.
func order(shuffled bool) []int {
	idx := make([]int, benchLen)
	for i := range idx {
		idx[i] = i
	}
	if shuffled {
		r := rand.New(rand.NewPCG(1, 2))
		r.Shuffle(len(idx), func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })
	}
	return idx
}

func BenchmarkTake(b *testing.B) {
	src := benchInts(b, 1)
	idx := order(false)

	b.SetBytes(benchLen * 8)
	for b.Loop() {
		chunkedSink = kernel.Take(src, idx)
	}
}

func BenchmarkTakeShuffled(b *testing.B) {
	src := benchInts(b, 1)
	idx := order(true)

	b.SetBytes(benchLen * 8)
	for b.Loop() {
		chunkedSink = kernel.Take(src, idx)
	}
}

func BenchmarkTakeChunked(b *testing.B) {
	src := benchInts(b, 16)
	idx := order(false)

	b.SetBytes(benchLen * 8)
	for b.Loop() {
		chunkedSink = kernel.Take(src, idx)
	}
}

func BenchmarkTakeChunkedShuffled(b *testing.B) {
	src := benchInts(b, 16)
	idx := order(true)

	b.SetBytes(benchLen * 8)
	for b.Loop() {
		chunkedSink = kernel.Take(src, idx)
	}
}

func BenchmarkTakeStrings(b *testing.B) {
	src := benchStrings(b)
	idx := order(true)

	for b.Loop() {
		chunkedSink = kernel.Take(src, idx)
	}
}

func BenchmarkFilter(b *testing.B) {
	src := benchInts(b, 1)
	mask := benchMask(b, 2)

	b.SetBytes(benchLen * 8)
	for b.Loop() {
		chunkedSink = kernel.Filter(src, mask)
	}
}

func BenchmarkIndices(b *testing.B) {
	mask := benchMask(b, 2)

	for b.Loop() {
		indicesSink = kernel.Indices(mask)
	}
}

// benchMask returns a mask that keeps one value in every n, with no nulls.
func benchMask(b *testing.B, n int) *array.Chunked {
	b.Helper()

	bd, err := array.NewBuilder(dtype.Bool)
	if err != nil {
		b.Fatalf("NewBuilder: %v", err)
	}
	for i := range benchLen {
		bd.AppendBool(i%n == 0)
	}

	out, err := array.NewChunked(dtype.Bool, bd.Finish())
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	return out
}

func BenchmarkCastWiden(b *testing.B) {
	src := benchInts(b, 1)

	b.SetBytes(benchLen * 8)
	for b.Loop() {
		chunkedSink = mustCast(b, src, dtype.Float64)
	}
}

func BenchmarkCastNarrow(b *testing.B) {
	src := benchInts(b, 1)

	b.SetBytes(benchLen * 8)
	for b.Loop() {
		chunkedSink = mustCast(b, src, dtype.Uint32)
	}
}

// BenchmarkCastReinterpret is the cheapest cast there is, since the bytes do
// not change at all and the only work is the range check and the copy.
func BenchmarkCastReinterpret(b *testing.B) {
	src := benchInts(b, 1)

	b.SetBytes(benchLen * 8)
	for b.Loop() {
		chunkedSink = mustCast(b, src, dtype.Timestamp{Unit: dtype.Nanosecond})
	}
}

func BenchmarkCastToText(b *testing.B) {
	src := benchInts(b, 1)

	for b.Loop() {
		chunkedSink = mustCast(b, src, dtype.String)
	}
}

// BenchmarkCastParse is the one that matters for a reader, since every number
// in a CSV file arrives as text and goes through here.
func BenchmarkCastParse(b *testing.B) {
	src := mustCast(b, benchInts(b, 1), dtype.String)

	for b.Loop() {
		chunkedSink = mustCast(b, src, dtype.Int64)
	}
}

// mustCast is Cast where a failure is a broken benchmark rather than a result.
func mustCast(b *testing.B, c *array.Chunked, to dtype.DataType) *array.Chunked {
	b.Helper()

	out, err := kernel.Cast(c, to)
	if err != nil {
		b.Fatalf("Cast to %s: %v", to, err)
	}
	return out
}

// benchShuffled returns the benchmark column with its values in no particular
// order, which is what a sort is normally handed.
func benchShuffled(b *testing.B) *array.Chunked {
	b.Helper()

	bd, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		b.Fatalf("NewBuilder: %v", err)
	}

	// A fixed seed, so that every run of the benchmark sorts the same values in
	// the same order and a number from one run means something next to a number
	// from another.
	r := rand.New(rand.NewPCG(1, 2))
	for range benchLen {
		bd.Append(r.Int64())
	}

	out, err := array.NewChunked(dtype.Int64, bd.Finish())
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	return out
}

// benchLowCardinality returns a column of eight distinct values, which is what
// a first sort key usually is: a symbol, a day, a customer.
func benchLowCardinality(b *testing.B) *array.Chunked {
	b.Helper()

	bd, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		b.Fatalf("NewBuilder: %v", err)
	}
	for i := range benchLen {
		bd.Append(int64(i % 8))
	}

	out, err := array.NewChunked(dtype.Int64, bd.Finish())
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	return out
}

// BenchmarkSortInts is the sort everything else is measured against: one
// numeric key, no nulls, values in no particular order.
func BenchmarkSortInts(b *testing.B) {
	c := benchShuffled(b)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		idx, err := kernel.SortIndex(kernel.Order{Column: c})
		if err != nil {
			b.Fatalf("SortIndex: %v", err)
		}
		indexSink = idx
	}
}

// BenchmarkSortSorted is the same column already in order, which is the case a
// pattern defeating sort is supposed to notice.
func BenchmarkSortSorted(b *testing.B) {
	c := benchInts(b, 1)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		idx, err := kernel.SortIndex(kernel.Order{Column: c})
		if err != nil {
			b.Fatalf("SortIndex: %v", err)
		}
		indexSink = idx
	}
}

// BenchmarkSortChunked is the column BenchmarkSortSorted uses, cut into sixteen
// chunks. The gap between the two is what the binary search per comparison
// costs, since the values and the order they are in are the same.
func BenchmarkSortChunked(b *testing.B) {
	c := benchInts(b, 16)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		idx, err := kernel.SortIndex(kernel.Order{Column: c})
		if err != nil {
			b.Fatalf("SortIndex: %v", err)
		}
		indexSink = idx
	}
}

// BenchmarkSortStrings is a sort whose comparison reads two values out of the
// string data rather than out of a slice of numbers.
func BenchmarkSortStrings(b *testing.B) {
	c := benchStrings(b)

	b.ReportAllocs()
	for b.Loop() {
		idx, err := kernel.SortIndex(kernel.Order{Column: c})
		if err != nil {
			b.Fatalf("SortIndex: %v", err)
		}
		indexSink = idx
	}
}

// BenchmarkSortTwoKeys is what the second key costs, on a first key of a
// handful of distinct values so that the second one is doing real work.
func BenchmarkSortTwoKeys(b *testing.B) {
	first := benchLowCardinality(b)
	second := benchShuffled(b)

	b.ReportAllocs()
	for b.Loop() {
		idx, err := kernel.SortIndex(
			kernel.Order{Column: first},
			kernel.Order{Column: second, Descending: true},
		)
		if err != nil {
			b.Fatalf("SortIndex: %v", err)
		}
		indexSink = idx
	}
}

// BenchmarkSortLowCardinality is a key of eight distinct values, which is what
// a group by sorts on and where a sort spends most of its time comparing values
// that are equal.
func BenchmarkSortLowCardinality(b *testing.B) {
	c := benchLowCardinality(b)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		idx, err := kernel.SortIndex(kernel.Order{Column: c})
		if err != nil {
			b.Fatalf("SortIndex: %v", err)
		}
		indexSink = idx
	}
}

var groupsSink *kernel.Groups

// benchKeys returns a key column of n distinct values spread over the benchmark
// length, which is the shape a group by meets: a symbol, a day, a customer.
func benchKeys(b *testing.B, n int) *array.Chunked {
	b.Helper()

	bd, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		b.Fatalf("NewBuilder: %v", err)
	}
	for i := range benchLen {
		bd.Append(int64(i % n))
	}

	out, err := array.NewChunked(dtype.Int64, bd.Finish())
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	return out
}

// benchStringKeys is the same spread of distinct values as text, which is the
// case that goes through the length prefixed encoding.
func benchStringKeys(b *testing.B, n int) *array.Chunked {
	b.Helper()

	bd, err := array.NewBuilder(dtype.String)
	if err != nil {
		b.Fatalf("NewBuilder: %v", err)
	}
	for i := range benchLen {
		bd.AppendString(fmt.Sprintf("key-%06d", i%n))
	}

	out, err := array.NewChunked(dtype.String, bd.Finish())
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	return out
}

// BenchmarkGroupBy is the ordinary case: one integer key of a hundred distinct
// values, which is what db-benchmark calls a low cardinality group by.
func BenchmarkGroupBy(b *testing.B) {
	c := benchKeys(b, 100)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		g, err := kernel.GroupBy(c)
		if err != nil {
			b.Fatalf("GroupBy: %v", err)
		}
		groupsSink = g
	}
}

// BenchmarkGroupByHighCardinality is a key that is nearly unique, so almost
// every row builds a group and the map does the most work it can.
func BenchmarkGroupByHighCardinality(b *testing.B) {
	c := benchKeys(b, benchLen)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		g, err := kernel.GroupBy(c)
		if err != nil {
			b.Fatalf("GroupBy: %v", err)
		}
		groupsSink = g
	}
}

func BenchmarkGroupByStrings(b *testing.B) {
	c := benchStringKeys(b, 100)

	b.ReportAllocs()
	for b.Loop() {
		g, err := kernel.GroupBy(c)
		if err != nil {
			b.Fatalf("GroupBy: %v", err)
		}
		groupsSink = g
	}
}

// BenchmarkGroupByTwoKeys is what the second key costs, which is the encoding
// and the wider map key rather than a second pass.
func BenchmarkGroupByTwoKeys(b *testing.B) {
	first := benchKeys(b, 100)
	second := benchStringKeys(b, 10)

	b.ReportAllocs()
	for b.Loop() {
		g, err := kernel.GroupBy(first, second)
		if err != nil {
			b.Fatalf("GroupBy: %v", err)
		}
		groupsSink = g
	}
}

func BenchmarkSum(b *testing.B) {
	c := benchInts(b, 1)
	g := mustGroup(b, benchKeys(b, 100))

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		out, err := kernel.Sum(c, g)
		if err != nil {
			b.Fatalf("Sum: %v", err)
		}
		chunkedSink = out
	}
}

func BenchmarkMean(b *testing.B) {
	c := benchInts(b, 1)
	g := mustGroup(b, benchKeys(b, 100))

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		out, err := kernel.Mean(c, g)
		if err != nil {
			b.Fatalf("Mean: %v", err)
		}
		chunkedSink = out
	}
}

// BenchmarkMax is the aggregation that goes through the sort comparison, so it
// is the slow one and the gap to Sum is what that costs.
func BenchmarkMax(b *testing.B) {
	c := benchInts(b, 1)
	g := mustGroup(b, benchKeys(b, 100))

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		out, err := kernel.Max(c, g)
		if err != nil {
			b.Fatalf("Max: %v", err)
		}
		chunkedSink = out
	}
}

func BenchmarkCount(b *testing.B) {
	c := benchInts(b, 1)
	g := mustGroup(b, benchKeys(b, 100))

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		chunkedSink = kernel.Count(c, g)
	}
}

// mustGroup is GroupBy where a failure is a broken benchmark rather than a
// result.
func mustGroup(b *testing.B, keys ...*array.Chunked) *kernel.Groups {
	b.Helper()

	g, err := kernel.GroupBy(keys...)
	if err != nil {
		b.Fatalf("GroupBy: %v", err)
	}
	return g
}
