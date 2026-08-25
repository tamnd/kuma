package bitmap_test

import (
	"testing"

	"github.com/tamnd/kuma/bitmap"
)

// reference is the model the real implementation is checked against. It is
// deliberately the slowest possible correct implementation.
type reference []bool

func (r reference) countOnes() int {
	n := 0
	for _, v := range r {
		if v {
			n++
		}
	}
	return n
}

// equalRef is checkEqual without the reporting, for the cases that compare
// thousands of small bitmaps and only want to know which one differed.
func equalRef(got *bitmap.Bitmap, want reference) bool {
	if got.Len() != len(want) {
		return false
	}
	for i := range want {
		if got.Get(i) != want[i] {
			return false
		}
	}
	return got.CountOnes() == want.countOnes()
}

func checkEqual(t *testing.T, got *bitmap.Bitmap, want reference) {
	t.Helper()
	if got.Len() != len(want) {
		t.Fatalf("Len() = %d, want %d", got.Len(), len(want))
	}
	for i := range want {
		if got.Get(i) != want[i] {
			t.Fatalf("Get(%d) = %v, want %v", i, got.Get(i), want[i])
		}
	}
	if got.CountOnes() != want.countOnes() {
		t.Fatalf("CountOnes() = %d, want %d", got.CountOnes(), want.countOnes())
	}
}

func TestNew(t *testing.T) {
	for _, n := range []int{0, 1, 7, 8, 9, 63, 64, 65, 1000} {
		b := bitmap.New(n)
		if b.Len() != n {
			t.Errorf("New(%d).Len() = %d", n, b.Len())
		}
		if b.CountOnes() != 0 {
			t.Errorf("New(%d).CountOnes() = %d, want 0", n, b.CountOnes())
		}
	}
}

// TestNewSetPadding is the test that matters most in this file. A naive
// implementation of NewSet fills every byte with 0xFF and reports 8 set bits
// for a 5 bit bitmap.
func TestNewSetPadding(t *testing.T) {
	for _, n := range []int{0, 1, 5, 7, 8, 9, 15, 16, 17, 1000} {
		b := bitmap.NewSet(n)
		if b.CountOnes() != n {
			t.Errorf("NewSet(%d).CountOnes() = %d, want %d", n, b.CountOnes(), n)
		}
		for i := range n {
			if !b.Get(i) {
				t.Fatalf("NewSet(%d).Get(%d) = false", n, i)
			}
		}
	}
}

func TestSetGet(t *testing.T) {
	const n = 100
	b := bitmap.New(n)
	want := make(reference, n)

	for i := range n {
		if i%3 == 0 {
			b.Set(i, true)
			want[i] = true
		}
	}
	checkEqual(t, b, want)

	// Clearing has to work too, and it is the half people forget to test.
	for i := range n {
		if i%2 == 0 {
			b.Set(i, false)
			want[i] = false
		}
	}
	checkEqual(t, b, want)
}

func TestAppend(t *testing.T) {
	var b bitmap.Bitmap // the zero value must be usable
	var want reference

	for i := range 200 {
		v := i%7 == 0
		b.Append(v)
		want = append(want, v)
	}
	checkEqual(t, &b, want)
}

func TestBooleanOps(t *testing.T) {
	const n = 37 // deliberately not a multiple of 8

	x := bitmap.New(n)
	y := bitmap.New(n)
	rx := make(reference, n)
	ry := make(reference, n)
	for i := range n {
		rx[i] = i%2 == 0
		ry[i] = i%3 == 0
		x.Set(i, rx[i])
		y.Set(i, ry[i])
	}

	tests := []struct {
		name string
		op   func(a, b *bitmap.Bitmap)
		want func(a, b bool) bool
	}{
		{"And", (*bitmap.Bitmap).And, func(a, b bool) bool { return a && b }},
		{"Or", (*bitmap.Bitmap).Or, func(a, b bool) bool { return a || b }},
		{"AndNot", (*bitmap.Bitmap).AndNot, func(a, b bool) bool { return a && !b }},
		{"Xor", (*bitmap.Bitmap).Xor, func(a, b bool) bool { return a != b }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := x.Clone()
			tt.op(got, y)

			want := make(reference, n)
			for i := range n {
				want[i] = tt.want(rx[i], ry[i])
			}
			checkEqual(t, got, want)
		})
	}
}

func TestNot(t *testing.T) {
	const n = 37
	b := bitmap.New(n)
	want := make(reference, n)
	for i := range n {
		want[i] = i%5 == 0
		b.Set(i, want[i])
	}

	b.Not()
	for i := range n {
		want[i] = !want[i]
	}
	checkEqual(t, b, want)

	// Not must not leave the padding bits set, or CountOnes over-reports.
	if b.CountOnes() != want.countOnes() {
		t.Errorf("CountOnes() after Not = %d, want %d", b.CountOnes(), want.countOnes())
	}
}

func TestClone(t *testing.T) {
	b := bitmap.NewSet(20)
	c := b.Clone()
	c.Set(0, false)

	if !b.Get(0) {
		t.Error("modifying the clone modified the original")
	}
}

func TestFromBytes(t *testing.T) {
	buf := []byte{0b10101010, 0b00000001}
	b := bitmap.FromBytes(buf, 9)

	if b.Len() != 9 {
		t.Fatalf("Len() = %d, want 9", b.Len())
	}
	for i, want := range []bool{false, true, false, true, false, true, false, true, true} {
		if b.Get(i) != want {
			t.Errorf("Get(%d) = %v, want %v", i, b.Get(i), want)
		}
	}
}

// TestSlice checks every start and end position against a model, because the
// interesting case is the unaligned one where each output byte is stitched
// together from two input bytes.
func TestSlice(t *testing.T) {
	const n = 70
	src := bitmap.New(n)
	model := make(reference, n)
	for i := range n {
		model[i] = i%3 == 0 || i%7 == 1
		src.Set(i, model[i])
	}

	for i := range n + 1 {
		for j := i; j <= n; j++ {
			got := src.Slice(i, j)
			if !equalRef(got, model[i:j]) {
				t.Fatalf("Slice(%d, %d) disagrees with the model", i, j)
			}
		}
	}

	// The source must be untouched, since Slice copies.
	checkEqual(t, src, model)
}

func TestSlicePadding(t *testing.T) {
	src := bitmap.NewSet(64)
	for i := range 64 {
		for j := i; j <= 64; j++ {
			got := src.Slice(i, j)
			if got.CountOnes() != j-i {
				t.Fatalf("Slice(%d, %d).CountOnes() = %d, want %d",
					i, j, got.CountOnes(), j-i)
			}
		}
	}
}

// TestCountOnesRange checks every start and end position against the model.
// The masking of the first and last byte is the whole of this function, and a
// mask that is off by one bit is an answer that is off by one null, which is
// the kind of wrong that reaches a user as a row count rather than as a crash.
func TestCountOnesRange(t *testing.T) {
	const n = 200
	src := bitmap.New(n)
	model := make(reference, n)
	for i := range n {
		model[i] = i%3 == 0 || i%7 == 1
		src.Set(i, model[i])
	}

	for i := range n + 1 {
		for j := i; j <= n; j++ {
			if got, want := src.CountOnesRange(i, j), model[i:j].countOnes(); got != want {
				t.Fatalf("CountOnesRange(%d, %d) = %d, want %d", i, j, got, want)
			}
		}
	}

	// The whole range has to agree with CountOnes, since they are two routes to
	// the same number and one of them is the one everything else calls.
	if got, want := src.CountOnesRange(0, n), src.CountOnes(); got != want {
		t.Errorf("CountOnesRange(0, %d) = %d but CountOnes() = %d", n, got, want)
	}
}

// TestCountOnesRangeAllSet is the case where every mask bug shows up as a
// number that is too large, since there is no clear bit to hide behind.
func TestCountOnesRangeAllSet(t *testing.T) {
	for _, n := range []int{1, 7, 8, 9, 63, 64, 65, 511, 512, 513} {
		src := bitmap.NewSet(n)
		for i := range n + 1 {
			for j := i; j <= n; j++ {
				if got := src.CountOnesRange(i, j); got != j-i {
					t.Fatalf("on %d bits all set, CountOnesRange(%d, %d) = %d, want %d",
						n, i, j, got, j-i)
				}
			}
		}
	}
}

func TestPanics(t *testing.T) {
	tests := []struct {
		name string
		fn   func()
	}{
		{"negative length", func() { bitmap.New(-1) }},
		{"short buffer", func() { bitmap.FromBytes([]byte{0}, 9) }},
		{"negative length from bytes", func() { bitmap.FromBytes([]byte{0}, -1) }},
		{"get out of range", func() { bitmap.New(8).Get(8) }},
		{"get negative", func() { bitmap.New(8).Get(-1) }},
		{"set out of range", func() { bitmap.New(8).Set(8, true) }},
		{"length mismatch", func() { bitmap.New(8).And(bitmap.New(9)) }},
		{"xor length mismatch", func() { bitmap.New(8).Xor(bitmap.New(9)) }},
		{"slice past the end", func() { bitmap.New(8).Slice(0, 9) }},
		{"slice backwards", func() { bitmap.New(8).Slice(5, 4) }},
		{"slice negative", func() { bitmap.New(8).Slice(-1, 4) }},
		{"count past the end", func() { bitmap.New(8).CountOnesRange(0, 9) }},
		{"count backwards", func() { bitmap.New(8).CountOnesRange(5, 4) }},
		{"count negative", func() { bitmap.New(8).CountOnesRange(-1, 4) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Error("did not panic")
				}
			}()
			tt.fn()
		})
	}
}

func BenchmarkCountOnes(b *testing.B) {
	bm := bitmap.NewSet(1 << 20)
	b.SetBytes(int64(len(bm.Bytes())))
	for b.Loop() {
		intSink = bm.CountOnes()
	}
}

// BenchmarkCountOnesRange is the same work reached the other way. The aligned
// case is a chunk boundary that fell on a byte, which is the common one, and
// the unaligned case pays for masking two partial bytes at the ends.
func BenchmarkCountOnesRange(b *testing.B) {
	bm := bitmap.NewSet(1 << 20)
	b.SetBytes(int64(len(bm.Bytes())))
	for b.Loop() {
		intSink = bm.CountOnesRange(0, 1<<20)
	}
}

func BenchmarkCountOnesRangeUnaligned(b *testing.B) {
	bm := bitmap.NewSet(1 << 20)
	b.SetBytes(int64(len(bm.Bytes())))
	for b.Loop() {
		intSink = bm.CountOnesRange(3, 1<<20-5)
	}
}

// BenchmarkCountOnesRangeSmall is the size a null count over one chunk actually
// runs at, which is where the per call overhead shows up rather than the
// throughput.
func BenchmarkCountOnesRangeSmall(b *testing.B) {
	bm := bitmap.NewSet(1 << 20)
	for b.Loop() {
		intSink = bm.CountOnesRange(1000, 9192)
	}
}

// intSink keeps the benchmarks from being optimized away.
var intSink int

func BenchmarkAnd(b *testing.B) {
	x := bitmap.NewSet(1 << 20)
	y := bitmap.NewSet(1 << 20)
	b.SetBytes(int64(len(x.Bytes())))
	for b.Loop() {
		x.And(y)
	}
}
