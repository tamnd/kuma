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

// BenchmarkStd is Welford's method, which costs a divide per value, and the gap
// to Mean is what that divide is worth paying.
func BenchmarkStd(b *testing.B) {
	c := benchInts(b, 1)
	g := mustGroup(b, benchKeys(b, 100))

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		out, err := kernel.Std(c, g, 1)
		if err != nil {
			b.Fatalf("Std: %v", err)
		}
		chunkedSink = out
	}
}

// BenchmarkMedian sorts every group, so this is the one aggregation that is not
// a single pass and the only one whose cost has a log in it.
func BenchmarkMedian(b *testing.B) {
	c := benchInts(b, 1)
	g := mustGroup(b, benchKeys(b, 100))

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		out, err := kernel.Median(c, g)
		if err != nil {
			b.Fatalf("Median: %v", err)
		}
		chunkedSink = out
	}
}

// BenchmarkNUnique goes back through the group by encoder and a map, so it is
// the aggregation that costs about what the grouping did.
func BenchmarkNUnique(b *testing.B) {
	c := benchInts(b, 1)
	g := mustGroup(b, benchKeys(b, 100))

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		out, err := kernel.NUnique(c, g)
		if err != nil {
			b.Fatalf("NUnique: %v", err)
		}
		chunkedSink = out
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

// benchSide returns a join side of benchLen rows over one int64 key of n
// distinct values.
func benchSide(b *testing.B, n int) kernel.Side {
	b.Helper()

	c := benchKeys(b, n)
	return kernel.Side{Rows: c.Len(), Keys: []*array.Chunked{c}}
}

// BenchmarkJoin is the ordinary case: a key that is unique on the right, so
// every left row matches once and the output is the size of the left side.
func BenchmarkJoin(b *testing.B) {
	left := benchSide(b, benchLen/8)
	right := benchSide(b, benchLen)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		p, err := kernel.Join(left, right, kernel.InnerJoin)
		if err != nil {
			b.Fatalf("Join: %v", err)
		}
		indexSink = p.Left
	}
}

// BenchmarkJoinLeft is the same join keeping the rows that matched nothing, so
// the gap to BenchmarkJoin is what the bookkeeping for an unmatched row costs.
func BenchmarkJoinLeft(b *testing.B) {
	left := benchSide(b, benchLen/8)
	right := benchSide(b, benchLen)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		p, err := kernel.Join(left, right, kernel.LeftJoin)
		if err != nil {
			b.Fatalf("Join: %v", err)
		}
		indexSink = p.Left
	}
}

// BenchmarkJoinManyMatches is a thousand distinct keys on both sides, so every
// left row matches sixty five right rows and the output is sixty five times
// either input. The cost of a join like this is writing the answer down and
// almost nothing else, which is the case worth knowing about before somebody
// joins on a low cardinality column by accident.
func BenchmarkJoinManyMatches(b *testing.B) {
	left := benchSide(b, 1000)
	right := benchSide(b, 1000)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		p, err := kernel.Join(left, right, kernel.InnerJoin)
		if err != nil {
			b.Fatalf("Join: %v", err)
		}
		indexSink = p.Left
	}
}

// BenchmarkJoinSemi takes nothing from the right side and one row per left row
// at most, so it is the cheapest join and the one a filter should compile to.
func BenchmarkJoinSemi(b *testing.B) {
	left := benchSide(b, benchLen/8)
	right := benchSide(b, benchLen)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		p, err := kernel.Join(left, right, kernel.SemiJoin)
		if err != nil {
			b.Fatalf("Join: %v", err)
		}
		indexSink = p.Left
	}
}

// benchGappy returns a column of benchLen int64 values where one in eight is
// missing, which is roughly what a file of real data looks like.
func benchGappy(b *testing.B) *array.Chunked {
	b.Helper()

	bd, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		b.Fatalf("NewBuilder: %v", err)
	}
	bd.Grow(benchLen)
	for i := range benchLen {
		if i%8 == 0 {
			bd.AppendNull()
			continue
		}
		bd.Append(int64(i))
	}

	c, err := array.NewChunked(dtype.Int64, bd.Finish())
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	return c
}

// BenchmarkIsNull is a copy of the validity bitmap with the words inverted, so
// it is a byte of work per eight rows and the number to watch is the bandwidth
// rather than the time.
func BenchmarkIsNull(b *testing.B) {
	c := benchGappy(b)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		chunkedSink = kernel.IsNull(c)
	}
}

// BenchmarkIsNotNullComplete is the same mask over a column with nothing
// missing, which has no bitmap to read at all.
func BenchmarkIsNotNullComplete(b *testing.B) {
	c := benchInts(b, 1)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		chunkedSink = kernel.IsNotNull(c)
	}
}

// BenchmarkFillNull is the gather that does the filling, which reads every row
// rather than only the missing ones. That is what buys it one piece of code for
// every type.
func BenchmarkFillNull(b *testing.B) {
	c := benchGappy(b)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		out, err := kernel.FillNull(c, array.Of(int64(0)))
		if err != nil {
			b.Fatalf("FillNull: %v", err)
		}
		chunkedSink = out
	}
}

// BenchmarkKeepIndex is one column that can fail, which is the common case and
// the one that answers without counting anything.
func BenchmarkKeepIndex(b *testing.B) {
	cols := []*array.Chunked{benchGappy(b), benchInts(b, 1)}

	b.SetBytes(benchLen * 16)
	b.ReportAllocs()
	for b.Loop() {
		indexSink = kernel.KeepIndex(cols, benchLen, 2)
	}
}

// BenchmarkKeepIndexCounted is two columns that can fail, which is the case
// that has to add up a count per row before it can answer.
func BenchmarkKeepIndexCounted(b *testing.B) {
	cols := []*array.Chunked{benchGappy(b), benchGappy(b)}

	b.SetBytes(benchLen * 16)
	b.ReportAllocs()
	for b.Loop() {
		indexSink = kernel.KeepIndex(cols, benchLen, 2)
	}
}

// BenchmarkCompare is a column against a literal, which is what a filter on a
// threshold turns into and by far the most common comparison there is.
func BenchmarkCompare(b *testing.B) {
	src := benchInts(b, 1)
	lit := benchLiteral(b, benchLen/2)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		chunkedSink = mustCompare(b, src, lit)
	}
}

// BenchmarkCompareColumns is two columns of the same length, where the cursor
// has to walk both sides rather than sitting still on one of them.
func BenchmarkCompareColumns(b *testing.B) {
	src, other := benchInts(b, 1), benchInts(b, 1)

	b.SetBytes(benchLen * 16)
	b.ReportAllocs()
	for b.Loop() {
		chunkedSink = mustCompare(b, src, other)
	}
}

// BenchmarkCompareChunked is the case the cursor exists for, where both sides
// are in many pieces and a walk through either one of them would otherwise cost
// a binary search per value.
func BenchmarkCompareChunked(b *testing.B) {
	src, other := benchInts(b, 64), benchInts(b, 16)

	b.SetBytes(benchLen * 16)
	b.ReportAllocs()
	for b.Loop() {
		chunkedSink = mustCompare(b, src, other)
	}
}

func BenchmarkCompareGappy(b *testing.B) {
	src := benchGappy(b)
	lit := benchLiteral(b, benchLen/2)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		chunkedSink = mustCompare(b, src, lit)
	}
}

func BenchmarkArith(b *testing.B) {
	src, other := benchInts(b, 1), benchInts(b, 1)

	b.SetBytes(benchLen * 16)
	b.ReportAllocs()
	for b.Loop() {
		out, err := kernel.Arith(src, other, kernel.OpAdd)
		if err != nil {
			b.Fatalf("Arith: %v", err)
		}
		chunkedSink = out
	}
}

func BenchmarkAnd(b *testing.B) {
	x, y := benchMask(b, 2), benchMask(b, 3)

	b.SetBytes(benchLen * 2)
	b.ReportAllocs()
	for b.Loop() {
		out, err := kernel.And(x, y)
		if err != nil {
			b.Fatalf("And: %v", err)
		}
		chunkedSink = out
	}
}

// benchLiteral returns a column of one value, which is what a comparison
// against a number is given.
func benchLiteral(b *testing.B, v int64) *array.Chunked {
	b.Helper()

	out, err := array.NewChunked(dtype.Int64, array.Of(v))
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	return out
}

// mustCompare is a greater than, where a type error would be a mistake in the
// benchmark rather than something being measured. The operator is not a
// parameter because it is not a variable a benchmark here changes: which of the
// six it is decides one comparison of an int at the bottom of a loop that reads
// two columns, so measuring all six would measure the same thing six times.
func mustCompare(b *testing.B, x, y *array.Chunked) *array.Chunked {
	b.Helper()

	out, err := kernel.Compare(x, y, kernel.OpGt)
	if err != nil {
		b.Fatalf("Compare: %v", err)
	}
	return out
}
