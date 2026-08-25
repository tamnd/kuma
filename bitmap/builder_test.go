package bitmap_test

import (
	"testing"

	"github.com/tamnd/kuma/bitmap"
)

func TestBuilder(t *testing.T) {
	var b bitmap.Builder // the zero value must be usable
	var want reference

	for i := range 200 {
		v := i%7 == 0
		b.Append(v)
		want = append(want, v)

		if b.Len() != len(want) {
			t.Fatalf("Len() = %d, want %d", b.Len(), len(want))
		}
	}
	checkEqual(t, b.Finish(), want)
}

// TestBuilderAppendMany covers every alignment of the start and the end of the
// run against a byte boundary, which is where the three loops inside
// AppendMany hand over to each other.
func TestBuilderAppendMany(t *testing.T) {
	for start := range 17 {
		for count := range 40 {
			var b bitmap.Builder
			var want reference

			b.AppendMany(false, start)
			for range start {
				want = append(want, false)
			}
			b.AppendMany(true, count)
			for range count {
				want = append(want, true)
			}
			b.AppendMany(false, 3)
			want = append(want, false, false, false)

			checkEqual(t, b.Finish(), want)
		}
	}
}

func TestBuilderAppendBools(t *testing.T) {
	vals := []bool{true, false, true, true, false, false, false, true, true}

	var b bitmap.Builder
	b.AppendBools(vals)
	b.AppendBools(nil)
	b.AppendBools(vals)

	checkEqual(t, b.Finish(), reference(append(append([]bool{}, vals...), vals...)))
}

// TestBuilderReuse is the test that catches a builder that hands over a buffer
// it is still writing into, and a Reset that leaves the previous run's bits
// behind.
func TestBuilderReuse(t *testing.T) {
	var b bitmap.Builder

	b.AppendMany(true, 40)
	first := b.Finish()

	b.AppendMany(false, 40)
	second := b.Finish()

	if first.CountOnes() != 40 {
		t.Errorf("the first bitmap changed under us: CountOnes() = %d, want 40", first.CountOnes())
	}
	if second.CountOnes() != 0 {
		t.Errorf("second.CountOnes() = %d, want 0", second.CountOnes())
	}

	b.AppendMany(true, 40)
	b.Reset()
	b.AppendMany(false, 5)
	if third := b.Finish(); third.CountOnes() != 0 {
		t.Errorf("Reset left bits behind: CountOnes() = %d, want 0", third.CountOnes())
	}
}

func TestBuilderGrow(t *testing.T) {
	var b bitmap.Builder
	b.Grow(1000)
	if b.Len() != 0 {
		t.Errorf("Grow changed the length to %d", b.Len())
	}
	b.AppendMany(true, 1000)
	if got := b.Finish().CountOnes(); got != 1000 {
		t.Errorf("CountOnes() = %d, want 1000", got)
	}
}

func TestBuilderEmpty(t *testing.T) {
	var b bitmap.Builder
	got := b.Finish()
	if got.Len() != 0 {
		t.Errorf("Len() = %d, want 0", got.Len())
	}
	if len(got.Bytes()) != 0 {
		t.Errorf("Bytes() = %v, want empty", got.Bytes())
	}
}

func TestBuilderPanics(t *testing.T) {
	tests := []struct {
		name string
		fn   func()
	}{
		{"negative grow", func() { new(bitmap.Builder).Grow(-1) }},
		{"negative count", func() { new(bitmap.Builder).AppendMany(true, -1) }},
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

func BenchmarkBuilderAppend(b *testing.B) {
	const n = 1 << 16
	var bld bitmap.Builder
	b.SetBytes(n / 8)
	for b.Loop() {
		bld.Reset()
		for i := range n {
			bld.Append(i%7 == 0)
		}
	}
}

// BenchmarkBuilderAppendMany is the comparison that justifies the type. A
// column with no nulls goes through this path.
func BenchmarkBuilderAppendMany(b *testing.B) {
	const n = 1 << 16
	var bld bitmap.Builder
	b.SetBytes(n / 8)
	for b.Loop() {
		bld.Reset()
		bld.AppendMany(true, n)
	}
}
