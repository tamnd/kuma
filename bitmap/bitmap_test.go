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

func TestPanics(t *testing.T) {
	tests := []struct {
		name string
		fn   func()
	}{
		{"negative length", func() { bitmap.New(-1) }},
		{"short buffer", func() { bitmap.FromBytes([]byte{0}, 9) }},
		{"get out of range", func() { bitmap.New(8).Get(8) }},
		{"get negative", func() { bitmap.New(8).Get(-1) }},
		{"set out of range", func() { bitmap.New(8).Set(8, true) }},
		{"length mismatch", func() { bitmap.New(8).And(bitmap.New(9)) }},
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
		_ = bm.CountOnes()
	}
}

func BenchmarkAnd(b *testing.B) {
	x := bitmap.NewSet(1 << 20)
	y := bitmap.NewSet(1 << 20)
	b.SetBytes(int64(len(x.Bytes())))
	for b.Loop() {
		x.And(y)
	}
}
