package bitmap_test

import (
	"testing"

	"github.com/tamnd/kuma/bitmap"
)

// FuzzBitmap runs a random sequence of operations against a []bool model and
// checks that the two agree after every step.
//
// This is the test to write first for anything in this package. The failure
// mode of bit packed code is almost never the middle of the loop, it is the
// final partial byte, and a fuzzer finds that in seconds while a hand written
// table test finds it in production.
func FuzzBitmap(f *testing.F) {
	f.Add([]byte{0}, uint8(0))
	f.Add([]byte{1, 2, 3}, uint8(7))
	f.Add([]byte{255, 0, 128, 64}, uint8(63))

	f.Fuzz(func(t *testing.T, ops []byte, size uint8) {
		n := int(size)
		got := bitmap.New(n)
		want := make(reference, n)

		for i, op := range ops {
			switch op % 6 {
			case 0: // set a bit
				if n > 0 {
					idx := int(op>>3) % n
					got.Set(idx, true)
					want[idx] = true
				}
			case 1: // clear a bit
				if n > 0 {
					idx := int(op>>3) % n
					got.Set(idx, false)
					want[idx] = false
				}
			case 2: // invert everything
				got.Not()
				for j := range want {
					want[j] = !want[j]
				}
			case 3: // intersect with a clone of itself, which must be a no-op
				got.And(got.Clone())
			case 4: // xor with itself, which must clear everything
				got.Xor(got.Clone())
				for j := range want {
					want[j] = false
				}
			case 5: // xor with the inverse, which must set everything
				inv := got.Clone()
				inv.Not()
				got.Xor(inv)
				for j := range want {
					want[j] = true
				}
			}

			if got.CountOnes() != want.countOnes() {
				t.Fatalf("after op %d (%d): CountOnes() = %d, want %d",
					i, op, got.CountOnes(), want.countOnes())
			}
		}

		checkEqual(t, got, want)
	})
}

// FuzzAppend checks that appending bits one at a time agrees with the model,
// including across every byte boundary.
func FuzzAppend(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1})
	f.Add([]byte{0, 1, 1, 0, 1, 0, 0, 1, 1})

	f.Fuzz(func(t *testing.T, in []byte) {
		var got bitmap.Bitmap
		var want reference

		for _, v := range in {
			b := v%2 == 1
			got.Append(b)
			want = append(want, b)
		}

		checkEqual(t, &got, want)

		checkPadding(t, &got)
	})
}

// FuzzSlice checks that slicing agrees with slicing the model. The unaligned
// case stitches each output byte out of two input bytes, which is the shift
// arithmetic most likely to be wrong at one end or the other.
func FuzzSlice(f *testing.F) {
	f.Add([]byte{0b10110101, 0b00001111}, uint8(3), uint8(9))
	f.Add([]byte{255, 255, 255}, uint8(0), uint8(24))
	f.Add([]byte{}, uint8(0), uint8(0))

	f.Fuzz(func(t *testing.T, data []byte, from, to uint8) {
		n := len(data) * 8
		if n == 0 {
			return
		}
		i := int(from) % (n + 1)
		j := int(to) % (n + 1)
		if i > j {
			i, j = j, i
		}

		src := bitmap.FromBytes(data, n)
		model := make(reference, n)
		for k := range n {
			model[k] = src.Get(k)
		}

		got := src.Slice(i, j)
		checkEqual(t, got, model[i:j])
		checkPadding(t, got)
	})
}

// FuzzBuilder checks a builder against the model across a random mix of single
// bits and runs, which is what puts the boundary between AppendMany's three
// loops in every possible place.
func FuzzBuilder(f *testing.F) {
	f.Add([]byte{})
	f.Add([]byte{1, 2, 3})
	f.Add([]byte{255, 0, 7, 64, 33})

	f.Fuzz(func(t *testing.T, ops []byte) {
		var b bitmap.Builder
		var want reference

		for _, op := range ops {
			v := op&1 == 1
			switch (op >> 1) % 3 {
			case 0:
				b.Append(v)
				want = append(want, v)
			case 1:
				count := int(op>>3) % 20
				b.AppendMany(v, count)
				for range count {
					want = append(want, v)
				}
			case 2:
				vals := make([]bool, int(op>>3)%20)
				for k := range vals {
					vals[k] = k%2 == 0
				}
				b.AppendBools(vals)
				want = append(want, vals...)
			}

			if b.Len() != len(want) {
				t.Fatalf("Len() = %d, want %d", b.Len(), len(want))
			}
		}

		got := b.Finish()
		checkEqual(t, got, want)
		checkPadding(t, got)
	})
}

// checkPadding asserts the invariant the whole package rests on. Arrow requires
// the bits past the length to be zero for a buffer to round trip through IPC,
// and CountOnes over-reports by up to seven if they are not.
func checkPadding(t *testing.T, b *bitmap.Bitmap) {
	t.Helper()
	buf := b.Bytes()
	rem := b.Len() & 7
	if len(buf) == 0 || rem == 0 {
		return
	}
	if buf[len(buf)-1]&^(byte(1)<<uint(rem)-1) != 0 {
		t.Errorf("padding bits set in final byte %08b at length %d",
			buf[len(buf)-1], b.Len())
	}
}
