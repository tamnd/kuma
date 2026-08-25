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

func BenchmarkFrameFilter(b *testing.B) {
	f := benchFrame(b)

	keep := make([]bool, benchLen)
	for i := range keep {
		keep[i] = i%2 == 0
	}
	mask := kuma.NewSeries("keep", keep...)

	b.ReportAllocs()
	for b.Loop() {
		out, err := f.Filter(mask)
		if err != nil {
			b.Fatalf("Filter: %v", err)
		}
		frameSink = out
	}
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
