package kernel_test

import (
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
