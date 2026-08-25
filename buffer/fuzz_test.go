package buffer_test

import (
	"bytes"
	"testing"

	"github.com/tamnd/kuma/buffer"
)

// FuzzBuffer runs a random sequence of operations against a []byte model and
// checks that the two agree after every step, along with the alignment and
// padding invariants that the kernels are going to rely on.
//
// The interesting failures in this package are not in any single method. They
// are in the interaction between length and capacity: a Resize that grows into
// memory a previous Resize shrank away from, an Append that lands exactly on a
// class boundary, a Grow that decided it did not need to reallocate and was
// wrong. Those are sequences, so the test that finds them has to be a sequence
// too.
func FuzzBuffer(f *testing.F) {
	f.Add([]byte{0}, []byte("a"))
	f.Add([]byte{1, 2, 3, 4}, []byte("kuma"))
	f.Add([]byte{2, 2, 2, 2, 2, 2}, bytes.Repeat([]byte("x"), 100))

	f.Fuzz(func(t *testing.T, ops, data []byte) {
		var got buffer.Buffer
		var want []byte

		check := func(step int, op byte) {
			t.Helper()
			if !bytes.Equal(got.Bytes(), want) {
				t.Fatalf("step %d op %d: buffer holds %q, want %q", step, op, got.Bytes(), want)
			}
			if got.Len() != len(want) {
				t.Fatalf("step %d op %d: Len is %d, want %d", step, op, got.Len(), len(want))
			}
			if got.Cap() < got.Len() {
				t.Fatalf("step %d op %d: Cap %d is below Len %d", step, op, got.Cap(), got.Len())
			}
			if !got.Aligned() {
				t.Fatalf("step %d op %d: buffer is no longer aligned", step, op)
			}
			if len(got.Padded())%buffer.Alignment != 0 {
				t.Fatalf("step %d op %d: Padded is %d bytes, not a whole number of blocks",
					step, op, len(got.Padded()))
			}
		}

		for step, op := range ops {
			// The operand is derived from the operation byte so that the
			// fuzzer only has one thing to mutate per step. Sizes stay small
			// because the bugs live at boundaries rather than at scale.
			k := int(op >> 3)

			switch op % 6 {
			case 0: // append a piece of the data
				n := min(k, len(data))
				got.Append(data[:n])
				want = append(want, data[:n]...)
			case 1: // grow to a shorter length
				n := len(want) - k
				if n < 0 {
					n = 0
				}
				got.Resize(n)
				want = want[:n]
			case 2: // resize to a longer length, which zeroes what it adds
				n := len(want) + k
				got.Resize(n)
				for len(want) < n {
					want = append(want, 0)
				}
			case 3: // reserve room without using it
				got.Grow(k)
			case 4: // start over
				got.Reset()
				want = want[:0]
			case 5: // write over what is there
				if len(want) > 0 {
					copy(got.Bytes(), data)
					copy(want, data)
				}
			}
			check(step, op)
		}

		// A clone matches and is independent, whatever state the sequence left
		// the buffer in.
		clone := got.Clone()
		if !bytes.Equal(clone.Bytes(), want) {
			t.Fatalf("clone holds %q, want %q", clone.Bytes(), want)
		}
		clone.Zero()
		if !bytes.Equal(got.Bytes(), want) {
			t.Fatalf("zeroing the clone changed the original to %q", got.Bytes())
		}

		// And a buffer from the pool is usable for the same length, since the
		// executor is going to reach for one in place of every New here.
		var p buffer.Pool
		pooled := p.Get(len(want))
		copy(pooled.Bytes(), want)
		if !bytes.Equal(pooled.Bytes(), want) {
			t.Fatalf("pooled buffer holds %q, want %q", pooled.Bytes(), want)
		}
		p.Put(pooled)
	})
}
