package kuma_test

import (
	"fmt"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// benchLen is the number of rows a benchmark works on. It is large enough that
// the per row cost is what is being measured and small enough that the column
// stays in cache, which is the case a kernel is written for.
const benchLen = 1 << 16

var (
	intSink     int64
	stringSink  string
	timeSink    time.Time
	valuesSink  []int64
	seriesSink  kuma.Series[int64]
	columnSink  kuma.Column
	frameSink   *kuma.Frame[kuma.Dynamic]
	stringsSink []string

	floatSeriesSink kuma.Series[float64]
	indexSink       []int
	groupSink       *kuma.GroupedFrame[kuma.Dynamic]
	boundSink       *kuma.Frame[benchRow]
	totalSink       *kuma.Frame[benchTotal]
)

// benchInts returns a column of benchLen int64 values in the given number of
// chunks.
func benchInts(b *testing.B, chunks int) kuma.Series[int64] {
	b.Helper()

	per := benchLen / chunks
	arrays := make([]*array.Array, chunks)
	for i := range arrays {
		values := make([]int64, per)
		for j := range values {
			values[j] = int64(i*per + j)
		}
		arrays[i] = array.Of(values...)
	}

	c, err := array.NewChunked(dtype.Int64, arrays...)
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	s, err := kuma.SeriesFrom[int64]("qty", c)
	if err != nil {
		b.Fatalf("SeriesFrom: %v", err)
	}
	return s
}

// benchFrame returns a frame of sixteen columns of benchLen rows.
func benchFrame(b *testing.B) *kuma.Frame[kuma.Dynamic] {
	b.Helper()

	cols := make([]kuma.Column, 16)
	for i := range cols {
		cols[i] = benchInts(b, 1).Rename(fmt.Sprintf("c%02d", i)).Column()
	}

	f, err := kuma.NewFrame(cols...)
	if err != nil {
		b.Fatalf("NewFrame: %v", err)
	}
	return f
}

func BenchmarkSeriesValue(b *testing.B) {
	s := benchInts(b, 1)

	b.ReportAllocs()
	for b.Loop() {
		var sum int64
		for i := range s.Len() {
			sum += s.Value(i)
		}
		intSink = sum
	}
}

// BenchmarkSeriesValueChunked is the same read on a column in sixteen chunks,
// which is what a file read in record batches gives you. The difference between
// this and the one above is the binary search over the chunk starts.
func BenchmarkSeriesValueChunked(b *testing.B) {
	s := benchInts(b, 16)

	b.ReportAllocs()
	for b.Loop() {
		var sum int64
		for i := range s.Len() {
			sum += s.Value(i)
		}
		intSink = sum
	}
}

// BenchmarkSeriesValues is the read a kernel does, and it is the number that
// matters. It should be the cost of the loop and nothing else.
func BenchmarkSeriesValues(b *testing.B) {
	s := benchInts(b, 1)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		var sum int64
		for _, v := range s.Values() {
			sum += v
		}
		intSink = sum
	}
}

func BenchmarkSeriesValuesChunked(b *testing.B) {
	s := benchInts(b, 16)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		valuesSink = s.Values()
	}
}

func BenchmarkSeriesValueString(b *testing.B) {
	values := make([]string, benchLen)
	for i := range values {
		values[i] = fmt.Sprintf("symbol-%d", i)
	}
	s := kuma.NewSeries("symbol", values...)

	b.ReportAllocs()
	for b.Loop() {
		for i := range s.Len() {
			stringSink = s.Value(i)
		}
	}
}

func BenchmarkSeriesValuesString(b *testing.B) {
	values := make([]string, benchLen)
	for i := range values {
		values[i] = fmt.Sprintf("symbol-%d", i)
	}
	s := kuma.NewSeries("symbol", values...)

	b.ReportAllocs()
	for b.Loop() {
		stringsSink = s.Values()
	}
}

func BenchmarkNewSeries(b *testing.B) {
	values := make([]int64, benchLen)
	for i := range values {
		values[i] = int64(i)
	}

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		seriesSink = kuma.NewSeries("qty", values...)
	}
}

func BenchmarkSeriesSlice(b *testing.B) {
	s := benchInts(b, 16)

	b.ReportAllocs()
	for b.Loop() {
		seriesSink = s.Slice(1, benchLen-1)
	}
}

// BenchmarkFrameColumn is the name lookup, which happens once per column per
// operation and is the reason the frame keeps a map rather than scanning the
// schema.
func BenchmarkFrameColumn(b *testing.B) {
	f := benchFrame(b)

	b.ReportAllocs()
	for b.Loop() {
		c, err := f.Column("c15")
		if err != nil {
			b.Fatalf("Column: %v", err)
		}
		columnSink = c
	}
}

func BenchmarkFrameSeries(b *testing.B) {
	f := benchFrame(b)

	b.ReportAllocs()
	for b.Loop() {
		s, err := f.Series[int64]("c15")
		if err != nil {
			b.Fatalf("Series: %v", err)
		}
		seriesSink = s
	}
}

func BenchmarkFrameSelect(b *testing.B) {
	f := benchFrame(b)

	b.ReportAllocs()
	for b.Loop() {
		out, err := f.Select("c01", "c05", "c09")
		if err != nil {
			b.Fatalf("Select: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkFrameSlice is the one that has to stay flat as the row count grows.
// Slicing sixteen columns is sixteen slices whatever the number of rows.
func BenchmarkFrameSlice(b *testing.B) {
	f := benchFrame(b)

	b.ReportAllocs()
	for b.Loop() {
		frameSink = f.Slice(1, benchLen-1)
	}
}

func BenchmarkFrameWithColumn(b *testing.B) {
	f := benchFrame(b)
	c := benchInts(b, 1).Rename("extra").Column()

	b.ReportAllocs()
	for b.Loop() {
		out, err := f.WithColumn(c)
		if err != nil {
			b.Fatalf("WithColumn: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkSeriesTake is a gather over one column, which is the cost every sort
// and every join is really made of.
func BenchmarkSeriesTake(b *testing.B) {
	s := benchInts(b, 1)
	idx := benchOrder(benchLen)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		seriesSink = s.Take(idx)
	}
}

// BenchmarkFrameTake is the same gather across sixteen columns, which is what a
// row reordering costs on a table rather than on a column.
func BenchmarkFrameTake(b *testing.B) {
	f := benchFrame(b)
	idx := benchOrder(benchLen)

	b.ReportAllocs()
	for b.Loop() {
		frameSink = f.Take(idx)
	}
}

// BenchmarkFrameFilterMask is a filter by a mask that is already worked out,
// which is the gather on its own with nothing in front of it.
func BenchmarkFrameFilterMask(b *testing.B) {
	f := benchFrame(b)

	keep := make([]bool, benchLen)
	for i := range keep {
		keep[i] = i%2 == 0
	}
	mask := kuma.NewSeries("keep", keep...)

	b.ReportAllocs()
	for b.Loop() {
		out, err := f.FilterMask(mask)
		if err != nil {
			b.Fatalf("FilterMask: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkFrameFilterExpr is the same gather with a condition in front of it,
// so the gap between this and BenchmarkFrameFilterMask is what working the
// condition out costs. Half the rows are kept in both, since the gather is
// paid per row that comes back.
func BenchmarkFrameFilterExpr(b *testing.B) {
	f := benchFrame(b)
	cond := kuma.I64("c00").Lt(benchLen / 2)

	b.ReportAllocs()
	for b.Loop() {
		out, err := f.Filter(cond)
		if err != nil {
			b.Fatalf("Filter: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkFrameFilterExprTwo is two conditions and an and, which is the shape
// most predicates have and the one that has to walk the tree.
func BenchmarkFrameFilterExprTwo(b *testing.B) {
	f := benchFrame(b)
	cond := kuma.I64("c00").Lt(benchLen / 2).And(kuma.I64("c01").Ge(1))

	b.ReportAllocs()
	for b.Loop() {
		out, err := f.Filter(cond)
		if err != nil {
			b.Fatalf("Filter: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkFrameEval is arithmetic over two columns, which is the other half of
// what an expression is for.
func BenchmarkFrameEval(b *testing.B) {
	f := benchFrame(b)
	e := kuma.I64("c00").MulExpr(kuma.I64("c01"))

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		out, err := f.Eval(e)
		if err != nil {
			b.Fatalf("Eval: %v", err)
		}
		columnSink = out
	}
}

// BenchmarkBind is the check a frame goes through on its way into the typed
// world. It is per column rather than per row, and it is here to keep it that
// way.
func BenchmarkBind(b *testing.B) {
	f := benchFrame(b)

	b.ReportAllocs()
	for b.Loop() {
		out, err := kuma.Bind[benchRow](f)
		if err != nil {
			b.Fatalf("Bind: %v", err)
		}
		boundSink = out
	}
}

// BenchmarkFrameSelectAs is the same check with the columns the schema does not
// name left out, which is the step a program that reads a wide file and works on
// three of its columns writes.
func BenchmarkFrameSelectAs(b *testing.B) {
	f := benchFrame(b)

	b.ReportAllocs()
	for b.Loop() {
		out, err := f.SelectAs[benchRow]()
		if err != nil {
			b.Fatalf("SelectAs: %v", err)
		}
		boundSink = out
	}
}

// BenchmarkFrameSelectAndBind is the two steps that were the way to write the
// one above, and the gap between them is what writing it once is worth. Both of
// them are per column rather than per row, so both are small and the gap is the
// second frame the first one builds.
func BenchmarkFrameSelectAndBind(b *testing.B) {
	f := benchFrame(b)

	b.ReportAllocs()
	for b.Loop() {
		sel, err := f.Select("c00", "c01", "c02")
		if err != nil {
			b.Fatalf("Select: %v", err)
		}
		out, err := kuma.Bind[benchRow](sel)
		if err != nil {
			b.Fatalf("Bind: %v", err)
		}
		boundSink = out
	}
}

// benchRow is the schema BenchmarkBind binds to, which is three of the sixteen
// columns benchFrame has.
type benchRow struct {
	C00 int64 `kuma:"c00"`
	C01 int64 `kuma:"c01"`
	C02 int64 `kuma:"c02"`
}

// benchOrder returns the positions of every row, shuffled, which is the order a
// sort or a hash join hands to a gather and the one that costs the most.
func benchOrder(n int) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	r := rand.New(rand.NewPCG(1, 2))
	r.Shuffle(len(idx), func(i, j int) { idx[i], idx[j] = idx[j], idx[i] })
	return idx
}

// BenchmarkSeriesCast is a cast through the root package, which is the same
// kernel with a series wrapped around it. The gap between this and the kernel
// benchmark of the same name is what the wrapping costs.
func BenchmarkSeriesCast(b *testing.B) {
	s := benchInts(b, 1)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		got, err := s.Cast[float64](dtype.Float64)
		if err != nil {
			b.Fatalf("Cast: %v", err)
		}
		floatSeriesSink = got
	}
}

// BenchmarkFrameCast casts one column of sixteen. The other fifteen are shared
// with the frame that went in, so this measures a cast plus the cost of a new
// frame rather than sixteen casts.
func BenchmarkFrameCast(b *testing.B) {
	f := benchFrame(b)

	b.ReportAllocs()
	for b.Loop() {
		got, err := f.Cast("c00", dtype.Float64)
		if err != nil {
			b.Fatalf("Cast: %v", err)
		}
		frameSink = got
	}
}

// BenchmarkFrameSortBy is a sort of a table, which is the order worked out over
// one column and then applied to sixteen.
func BenchmarkFrameSortBy(b *testing.B) {
	f := benchFrame(b)

	b.ReportAllocs()
	for b.Loop() {
		got, err := f.SortBy("c00")
		if err != nil {
			b.Fatalf("SortBy: %v", err)
		}
		frameSink = got
	}
}

// BenchmarkFrameSortIndex is the same sort without moving anything, so the gap
// between this and the one above is what the gather costs.
func BenchmarkFrameSortIndex(b *testing.B) {
	f := benchFrame(b)

	b.ReportAllocs()
	for b.Loop() {
		idx, err := f.SortIndex(kuma.Asc("c00"))
		if err != nil {
			b.Fatalf("SortIndex: %v", err)
		}
		indexSink = idx
	}
}

// BenchmarkSeriesSort is one column, which is the kernel with a series wrapped
// around it.
func BenchmarkSeriesSort(b *testing.B) {
	s := benchInts(b, 1)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		got, err := s.Sort(kuma.Order{})
		if err != nil {
			b.Fatalf("Sort: %v", err)
		}
		seriesSink = got
	}
}

// benchGrouped returns a frame of a key column of a hundred distinct values and
// two columns to aggregate, already grouped, which is the shape of every group
// by anybody writes.
func benchGrouped(b *testing.B) *kuma.GroupedFrame[kuma.Dynamic] {
	b.Helper()

	keys := make([]int64, benchLen)
	for i := range keys {
		keys[i] = int64(i % 100)
	}
	k, err := array.NewChunked(dtype.Int64, array.Of(keys...))
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	key, err := kuma.NewColumn("k", k)
	if err != nil {
		b.Fatalf("NewColumn: %v", err)
	}

	f, err := kuma.NewFrame(key,
		benchInts(b, 1).Rename("qty").Column(),
		benchInts(b, 1).Rename("price").Column())
	if err != nil {
		b.Fatalf("NewFrame: %v", err)
	}

	g, err := f.GroupBy("k")
	if err != nil {
		b.Fatalf("GroupBy: %v", err)
	}
	return g
}

// BenchmarkFrameGroupBy is the grouping on its own, which is the expensive half
// of a group by and the half that does not get cheaper as more is asked of it.
func BenchmarkFrameGroupBy(b *testing.B) {
	f := benchGrouped(b).Frame()

	b.ReportAllocs()
	for b.Loop() {
		g, err := f.GroupBy("k")
		if err != nil {
			b.Fatalf("GroupBy: %v", err)
		}
		groupSink = g
	}
}

// BenchmarkFrameAgg is one aggregation over a grouping that already exists, so
// the gap to BenchmarkFrameGroupBy is what asking a second question costs.
func BenchmarkFrameAgg(b *testing.B) {
	g := benchGrouped(b)

	b.ReportAllocs()
	for b.Loop() {
		out, err := g.Agg(kuma.Sum("qty"))
		if err != nil {
			b.Fatalf("Agg: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkFrameAggSeveral asks four questions of one grouping, which is what
// the whole design is for: the grouping is paid for once.
func BenchmarkFrameAggSeveral(b *testing.B) {
	g := benchGrouped(b)

	b.ReportAllocs()
	for b.Loop() {
		out, err := g.Agg(
			kuma.Sum("qty").As("total"),
			kuma.Mean("price").As("avg"),
			kuma.Max("price").As("high"),
			kuma.Size(),
		)
		if err != nil {
			b.Fatalf("Agg: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkFrameGroupByAndAgg is the whole thing end to end, which is the
// number to compare against a pandas groupby.
func BenchmarkFrameGroupByAndAgg(b *testing.B) {
	f := benchGrouped(b).Frame()

	b.ReportAllocs()
	for b.Loop() {
		g, err := f.GroupBy("k")
		if err != nil {
			b.Fatalf("GroupBy: %v", err)
		}
		out, err := g.Agg(kuma.Sum("qty").As("total"), kuma.Mean("price").As("avg"))
		if err != nil {
			b.Fatalf("Agg: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkFrameAggAs is the same group by and the same two aggregations with a
// struct saying what the result holds, so the gap to BenchmarkFrameGroupByAndAgg
// is what keeping the frame typed costs. It is the check of three columns and
// the frame built from them, over a hundred groups.
func BenchmarkFrameAggAs(b *testing.B) {
	f := benchGrouped(b).Frame()

	b.ReportAllocs()
	for b.Loop() {
		g, err := f.GroupBy("k")
		if err != nil {
			b.Fatalf("GroupBy: %v", err)
		}
		out, err := g.AggAs[benchTotal](kuma.Sum("qty").As("total"), kuma.Mean("price").As("avg"))
		if err != nil {
			b.Fatalf("AggAs: %v", err)
		}
		totalSink = out
	}
}

// benchTotal is the schema BenchmarkFrameAggAs asks for, which is the key and
// the two aggregations.
type benchTotal struct {
	K     int64   `kuma:"k"`
	Total int64   `kuma:"total"`
	Avg   float64 `kuma:"avg"`
}

// BenchmarkFrameDistinct is a drop duplicates over one key column, which is the
// pass a group by makes over the rows with everything the aggregations need
// left off. The gap to BenchmarkFrameGroupBy is what working out the group of
// every row and gathering the key columns costs.
func BenchmarkFrameDistinct(b *testing.B) {
	f := benchGrouped(b).Frame()

	b.ReportAllocs()
	for b.Loop() {
		out, err := f.Distinct("k")
		if err != nil {
			b.Fatalf("Distinct: %v", err)
		}
		frameSink = out
	}
}

// benchJoinSides returns two frames to join: a large left one with a key that
// repeats and a small right one with each key once, which is the shape almost
// every real join has.
func benchJoinSides(b *testing.B) (left, right *kuma.Frame[kuma.Dynamic]) {
	b.Helper()

	const groups = 4096

	lk := make([]int64, benchLen)
	for i := range lk {
		lk[i] = int64(i % groups)
	}
	rk := make([]int64, groups)
	for i := range rk {
		rk[i] = int64(i)
	}

	left, err := kuma.NewFrame(
		benchKey(b, "k", lk),
		benchInts(b, 1).Rename("qty").Column(),
	)
	if err != nil {
		b.Fatalf("NewFrame: %v", err)
	}

	right, err = kuma.NewFrame(
		benchKey(b, "k", rk),
		kuma.NewSeries("rate", make([]float64, groups)...).Column(),
	)
	if err != nil {
		b.Fatalf("NewFrame: %v", err)
	}
	return left, right
}

// benchKey builds a named int64 column out of the values given.
func benchKey(b *testing.B, name string, vals []int64) kuma.Column {
	b.Helper()

	data, err := array.NewChunked(dtype.Int64, array.Of(vals...))
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	c, err := kuma.NewColumn(name, data)
	if err != nil {
		b.Fatalf("NewColumn: %v", err)
	}
	return c
}

// BenchmarkFrameInnerJoin is the whole join end to end, the kernel working out
// the pairs and then a gather over every column of both sides.
func BenchmarkFrameInnerJoin(b *testing.B) {
	left, right := benchJoinSides(b)

	b.ReportAllocs()
	for b.Loop() {
		out, err := left.InnerJoin(right, "k")
		if err != nil {
			b.Fatalf("InnerJoin: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkFrameLeftJoin is the same join keeping the left rows that matched
// nothing, which here is none of them, so the difference from the inner join is
// the cost of the bookkeeping rather than of any extra rows.
func BenchmarkFrameLeftJoin(b *testing.B) {
	left, right := benchJoinSides(b)

	b.ReportAllocs()
	for b.Loop() {
		out, err := left.LeftJoin(right, "k")
		if err != nil {
			b.Fatalf("LeftJoin: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkFrameSemiJoin takes nothing from the right side, so it is the join
// without the gather, and the gap between it and the inner join is what
// building the table costs.
func BenchmarkFrameSemiJoin(b *testing.B) {
	left, right := benchJoinSides(b)

	b.ReportAllocs()
	for b.Loop() {
		out, err := left.Join(right, kuma.Using("k"), kuma.SemiJoin)
		if err != nil {
			b.Fatalf("Join: %v", err)
		}
		frameSink = out
	}
}

// benchParts returns eight frames of the same shape, which is the day of files
// a concat usually gets handed.
func benchParts(b *testing.B) []*kuma.Frame[kuma.Dynamic] {
	b.Helper()

	const parts = 8

	prices := make([]float64, benchLen)
	for i := range prices {
		prices[i] = float64(i) / 8
	}

	out := make([]*kuma.Frame[kuma.Dynamic], parts)
	for i := range out {
		f, err := kuma.NewFrame(
			benchInts(b, 1).Rename("qty").Column(),
			kuma.NewSeries("price", prices...).Column(),
		)
		if err != nil {
			b.Fatalf("NewFrame: %v", err)
		}
		out[i] = f
	}
	return out
}

// BenchmarkConcat is the point of storing a column as a list of chunks. Eight
// frames of 65536 rows stack into one frame of half a million, and the cost is
// the two column loops rather than anything to do with the rows.
func BenchmarkConcat(b *testing.B) {
	parts := benchParts(b)

	b.ReportAllocs()
	for b.Loop() {
		out, err := kuma.Concat(parts...)
		if err != nil {
			b.Fatalf("Concat: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkConcatUnion is the same stacking with a column one frame does not
// have, which is the case that has to build something: the nulls that stand in
// for it.
func BenchmarkConcatUnion(b *testing.B) {
	parts := benchParts(b)
	odd, err := parts[0].Drop("price")
	if err != nil {
		b.Fatalf("Drop: %v", err)
	}
	parts[0] = odd

	b.ReportAllocs()
	for b.Loop() {
		out, err := kuma.ConcatUnion(parts...)
		if err != nil {
			b.Fatalf("ConcatUnion: %v", err)
		}
		frameSink = out
	}
}

// benchGappy returns a frame of benchLen rows where one column in eight has a
// missing value, which is roughly what a file of real data looks like. A column
// with nothing missing takes a different path through all of this, so the other
// column is complete on purpose.
func benchGappy(b *testing.B) *kuma.Frame[kuma.Dynamic] {
	b.Helper()

	builder, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		b.Fatalf("NewBuilder: %v", err)
	}
	builder.Grow(benchLen)
	for i := range benchLen {
		if i%8 == 0 {
			builder.AppendNull()
			continue
		}
		builder.Append(int64(i))
	}

	data, err := array.NewChunked(dtype.Int64, builder.Finish())
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	qty, err := kuma.NewColumn("qty", data)
	if err != nil {
		b.Fatalf("NewColumn: %v", err)
	}

	prices := make([]float64, benchLen)
	for i := range prices {
		prices[i] = float64(i) / 8
	}

	f, err := kuma.NewFrame(qty, kuma.NewSeries("price", prices...).Column())
	if err != nil {
		b.Fatalf("NewFrame: %v", err)
	}
	return f
}

// benchGappyColumn is the column of benchGappy that has the holes in it.
func benchGappyColumn(b *testing.B) kuma.Column {
	b.Helper()

	c, err := benchGappy(b).Column("qty")
	if err != nil {
		b.Fatalf("Column: %v", err)
	}
	return c
}

// BenchmarkColumnNullMask is one bitmap read and one boolean write per row.
func BenchmarkColumnNullMask(b *testing.B) {
	c := benchGappyColumn(b)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		columnSink = c.NullMask()
	}
}

// BenchmarkColumnValidMaskComplete is the same mask over a column with nothing
// missing, which skips the bitmap entirely and is the common case.
func BenchmarkColumnValidMaskComplete(b *testing.B) {
	f := benchGappy(b)
	c, err := f.Column("price")
	if err != nil {
		b.Fatalf("Column: %v", err)
	}

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		columnSink = c.ValidMask()
	}
}

// BenchmarkColumnFillNull is the gather that does the filling. It reads every
// row rather than only the missing ones, which is what buys it the same code
// for every type.
func BenchmarkColumnFillNull(b *testing.B) {
	c := benchGappyColumn(b)

	b.SetBytes(benchLen * 8)
	b.ReportAllocs()
	for b.Loop() {
		out, err := c.FillNull(int64(0))
		if err != nil {
			b.Fatalf("FillNull: %v", err)
		}
		columnSink = out
	}
}

// BenchmarkFrameDropNulls counts the values present in each row and then
// gathers the rows that passed, over two columns of which only one can fail.
func BenchmarkFrameDropNulls(b *testing.B) {
	f := benchGappy(b)

	b.SetBytes(benchLen * 16)
	b.ReportAllocs()
	for b.Loop() {
		out, err := f.DropNulls()
		if err != nil {
			b.Fatalf("DropNulls: %v", err)
		}
		frameSink = out
	}
}

// BenchmarkFrameIsNull is the mask over every column of the frame at once.
func BenchmarkFrameIsNull(b *testing.B) {
	f := benchGappy(b)

	b.SetBytes(benchLen * 16)
	b.ReportAllocs()
	for b.Loop() {
		frameSink = f.IsNull()
	}
}

// BenchmarkFrameString is a fmt.Println of a frame of sixteen columns and
// sixty five thousand rows.
//
// What it should show is a cost that has nothing to do with how big the frame
// is, since ten rows are rendered whether the frame holds ten or ten million.
// A printer that walked the whole frame to work out its widths would show up
// here as a number that grows with benchLen.
func BenchmarkFrameString(b *testing.B) {
	f := benchFrame(b)

	b.SetBytes(int64(len(f.String())))
	b.ReportAllocs()
	for b.Loop() {
		stringSink = f.String()
	}
}

// BenchmarkFrameRender is the whole frame rather than a window on it, which is
// what writing one to a report does. This one is per row and is meant to be.
func BenchmarkFrameRender(b *testing.B) {
	f := benchFrame(b).Head(1000)
	opts := &kuma.PrintOptions{MaxRows: -1, MaxCols: -1}

	b.SetBytes(int64(len(f.Render(opts))))
	b.ReportAllocs()
	for b.Loop() {
		stringSink = f.Render(opts)
	}
}
