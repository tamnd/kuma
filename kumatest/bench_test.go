package kumatest_test

import (
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kumatest"
)

// benchRows is the size of the frames these benchmarks compare. It is large
// enough that the walk over the values is what is being timed rather than the
// cost of setting up the report.
const benchRows = 100_000

// BenchmarkEqualFrames is the case that runs on every green test run, where the
// two frames are the same and every value has to be looked at to find that out.
func BenchmarkEqualFrames(b *testing.B) {
	opts := &kumatest.RandomOptions{Rows: benchRows, Nulls: 0.1, Seed: 1}
	got := kumatest.Random(opts)
	want := kumatest.Random(opts)
	var t discard

	b.SetBytes(int64(benchRows))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		kumatest.EqualFrames(&t, got, want, nil)
	}
	if t.reports != 0 {
		b.Fatalf("the two frames were reported as different %d times", t.reports)
	}
}

// BenchmarkDiffFrames is the failing case, where the report fills up in the
// first few rows and the rest of the walk is counting.
func BenchmarkDiffFrames(b *testing.B) {
	got := kumatest.Random(&kumatest.RandomOptions{Rows: benchRows, Seed: 1})
	want := kumatest.Random(&kumatest.RandomOptions{Rows: benchRows, Seed: 2})

	b.SetBytes(int64(benchRows))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if kumatest.DiffFrames(got, want, nil) == "" {
			b.Fatal("the two frames were reported as equal")
		}
	}
}

// BenchmarkEqualSeriesOfInt64 times one column of one type, which is the
// number to watch when the comparison of a type changes.
func BenchmarkEqualSeriesOfInt64(b *testing.B) {
	opts := &kumatest.RandomOptions{
		Rows:  benchRows,
		Types: []dtype.DataType{dtype.Int64},
		Seed:  1,
	}
	got := series(b, kumatest.Random(opts))
	want := series(b, kumatest.Random(opts))
	var t discard

	b.SetBytes(int64(benchRows) * 8)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		kumatest.EqualSeries(&t, got, want, nil)
	}
	if t.reports != 0 {
		b.Fatalf("the two series were reported as different %d times", t.reports)
	}
}

// BenchmarkRandom times building the data, since a benchmark elsewhere that
// builds a frame per run is paying this.
func BenchmarkRandom(b *testing.B) {
	opts := &kumatest.RandomOptions{Rows: benchRows, Nulls: 0.1, Seed: 1}

	b.SetBytes(int64(benchRows))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		kumatest.Random(opts)
	}
}

// discard is a [kumatest.TB] that counts what was reported rather than printing
// it, so that a benchmark is not timing the test log.
type discard struct{ reports int }

func (*discard) Helper() {}

func (d *discard) Errorf(string, ...any) { d.reports++ }

// series pulls the one column out of a frame.
func series(b *testing.B, f *kuma.Frame[kuma.Dynamic]) kuma.Series[int64] {
	b.Helper()

	s, err := f.Series[int64]("column_1")
	if err != nil {
		b.Fatalf("Series: %v", err)
	}
	return s
}
