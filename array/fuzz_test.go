package array_test

import (
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
)

// FuzzSlice checks a sliced column against a plain model built from the same
// bytes. Slicing is where the null count, the offset and the values all have to
// agree with each other, and the way they go wrong is quietly: a null count
// that is off by one is a row count a user sees, not a crash anyone sees.
//
// The input is read as the validity bits and the values at once, so the fuzzer
// gets to choose both the pattern of nulls and where the range falls in it.
func FuzzSlice(f *testing.F) {
	f.Add([]byte{0b10110101}, 8, 0, 8)
	f.Add([]byte{0xFF, 0x00, 0xAA}, 20, 3, 17)
	f.Add([]byte{}, 0, 0, 0)
	f.Add([]byte{0x01}, 1, 1, 1)

	f.Fuzz(func(t *testing.T, bits []byte, length, i, j int) {
		if length < 0 || length > 8*len(bits) {
			t.Skip()
		}
		if i < 0 || j < i || j > length {
			t.Skip()
		}

		valid := bitmap.New(length)
		values := buffer.New(length * 4)
		present := make([]bool, length)
		for k := range length {
			present[k] = bits[k/8]&(1<<(k%8)) != 0
			valid.Set(k, present[k])
			values.Bytes()[k*4] = byte(k)
		}

		a, err := array.New(dtype.Int32, length, values, valid)
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		s := a.Slice(i, j)
		if s.Len() != j-i {
			t.Fatalf("Slice(%d, %d).Len() = %d, want %d", i, j, s.Len(), j-i)
		}
		if s.Offset() != i {
			t.Fatalf("Slice(%d, %d).Offset() = %d, want %d", i, j, s.Offset(), i)
		}

		nulls := 0
		for k := i; k < j; k++ {
			if !present[k] {
				nulls++
			}
		}
		if s.NullCount() != nulls {
			t.Fatalf("Slice(%d, %d).NullCount() = %d, want %d", i, j, s.NullCount(), nulls)
		}

		for k := range s.Len() {
			if s.IsValid(k) != present[i+k] {
				t.Fatalf("Slice(%d, %d).IsValid(%d) = %v, want %v",
					i, j, k, s.IsValid(k), present[i+k])
			}
			if got := s.Value[int32](k); got != int32(byte(i+k)) {
				t.Fatalf("Slice(%d, %d).Value(%d) = %d, want %d",
					i, j, k, got, int32(byte(i+k)))
			}
		}

		// A clone holds the same values, the same nulls and none of the memory,
		// so it is the other half of the same property.
		c := s.Clone()
		if c.Len() != s.Len() || c.NullCount() != s.NullCount() || c.Offset() != 0 {
			t.Fatalf("Clone() of %s gave %s", s, c)
		}
		for k := range c.Len() {
			if c.IsValid(k) != s.IsValid(k) || c.Value[int32](k) != s.Value[int32](k) {
				t.Fatalf("Clone() disagrees with the slice at %d", k)
			}
		}
	})
}

// FuzzBuilder drives a builder from a program of opcodes and checks what comes
// out against a model of the same appends. The interesting part is where the
// first null lands, because that is where the bitmap appears and everything
// before it has to be filled in as present, so the fuzzer choosing the position
// is choosing the case.
func FuzzBuilder(f *testing.F) {
	f.Add([]byte{1, 1, 0, 1})
	f.Add([]byte{0, 0, 0})
	f.Add([]byte{1, 2, 1})
	f.Add([]byte{3, 1, 0, 2, 1})
	f.Add([]byte{})

	f.Fuzz(func(t *testing.T, ops []byte) {
		b, err := array.NewBuilder(dtype.Int32)
		if err != nil {
			t.Fatalf("NewBuilder: %v", err)
		}

		var (
			values  []int32
			present []bool
		)
		for _, op := range ops {
			switch op % 4 {
			case 0:
				b.AppendNull()
				values = append(values, 0)
				present = append(present, false)
			case 1:
				v := int32(op) * 1000
				b.Append(v)
				values = append(values, v)
				present = append(present, true)
			case 2:
				n := int(op) % 9
				b.AppendNulls(n)
				for range n {
					values = append(values, 0)
					present = append(present, false)
				}
			case 3:
				vs := make([]int32, int(op)%5)
				for k := range vs {
					vs[k] = int32(op) + int32(k)
				}
				b.AppendValues(vs)
				values = append(values, vs...)
				for range vs {
					present = append(present, true)
				}
			}

			if b.Len() != len(values) {
				t.Fatalf("after op %d the builder has %d values, want %d", op, b.Len(), len(values))
			}
		}

		a := b.Finish()
		if a.Len() != len(values) {
			t.Fatalf("%s, want %d values", a, len(values))
		}

		nulls := 0
		for i := range present {
			if !present[i] {
				nulls++
			}
			if a.IsValid(i) != present[i] {
				t.Fatalf("IsValid(%d) = %v, want %v", i, a.IsValid(i), present[i])
			}
			if got := a.Value[int32](i); got != values[i] {
				t.Fatalf("Value(%d) = %d, want %d", i, got, values[i])
			}
		}
		if a.NullCount() != nulls {
			t.Fatalf("NullCount() = %d, want %d", a.NullCount(), nulls)
		}
		if nulls == 0 && a.Validity() != nil {
			t.Fatal("a column with no nulls came out with a validity bitmap")
		}

		// The builder is empty again, whatever it was handed.
		if b.Len() != 0 || b.NullCount() != 0 {
			t.Fatalf("after Finish the builder has %d values and %d nulls", b.Len(), b.NullCount())
		}
	})
}

// FuzzBools is the same property for the one type whose values are bits, where
// a slice that does not begin on a byte boundary has to shift them.
func FuzzBools(f *testing.F) {
	f.Add([]byte{0b10110101}, 8, 0, 8)
	f.Add([]byte{0xFF, 0x00, 0xAA}, 20, 3, 17)
	f.Add([]byte{0x0F}, 5, 1, 4)

	f.Fuzz(func(t *testing.T, bits []byte, length, i, j int) {
		if length < 0 || length > 8*len(bits) {
			t.Skip()
		}
		if i < 0 || j < i || j > length {
			t.Skip()
		}

		values := make([]bool, length)
		for k := range length {
			values[k] = bits[k/8]&(1<<(k%8)) != 0
		}

		s := array.OfBools(values...).Slice(i, j)
		for k := range s.Len() {
			if s.Bool(k) != values[i+k] {
				t.Fatalf("Slice(%d, %d).Bool(%d) = %v, want %v",
					i, j, k, s.Bool(k), values[i+k])
			}
		}

		c := s.Clone()
		for k := range c.Len() {
			if c.Bool(k) != values[i+k] {
				t.Fatalf("Clone() of Slice(%d, %d) has Bool(%d) = %v, want %v",
					i, j, k, c.Bool(k), values[i+k])
			}
		}
	})
}
