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
			switch op % 4 {
			case 0: // set a bit
				if n > 0 {
					idx := int(op>>2) % n
					got.Set(idx, true)
					want[idx] = true
				}
			case 1: // clear a bit
				if n > 0 {
					idx := int(op>>2) % n
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

		// The padding bits must be clear whatever the length, or the buffer
		// will not round trip through Arrow IPC.
		if buf := got.Bytes(); len(buf) > 0 {
			if rem := got.Len() & 7; rem != 0 {
				if buf[len(buf)-1]&^(byte(1)<<uint(rem)-1) != 0 {
					t.Errorf("padding bits set in final byte %08b at length %d",
						buf[len(buf)-1], got.Len())
				}
			}
		}
	})
}
