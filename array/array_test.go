package array_test

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/strview"
)

// validity returns a bitmap of the given bits, where true means the value is
// present. It is how every test here says which values are missing.
func validity(bits ...bool) *bitmap.Bitmap {
	b := bitmap.New(len(bits))
	for i, v := range bits {
		b.Set(i, v)
	}
	return b
}

// int64s returns a buffer holding the given values, the long way, so that the
// tests do not depend on the unsafe conversion the package itself uses.
func int64s(t *testing.T, values ...int64) *buffer.Buffer {
	t.Helper()
	buf := buffer.New(len(values) * 8)
	p := buf.Bytes()
	for i, v := range values {
		u := uint64(v)
		for k := range 8 {
			p[i*8+k] = byte(u >> (8 * k))
		}
	}
	return buf
}

func mustNew(t *testing.T, dt dtype.DataType, length int, values *buffer.Buffer, valid *bitmap.Bitmap) *array.Array {
	t.Helper()
	a, err := array.New(dt, length, values, valid)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return a
}

func TestNew(t *testing.T) {
	a := mustNew(t, dtype.Int64, 3, int64s(t, 10, 20, 30), nil)

	if !dtype.Equal(a.DType(), dtype.Int64) {
		t.Errorf("DType() = %s, want int64", a.DType())
	}
	if a.Len() != 3 {
		t.Errorf("Len() = %d, want 3", a.Len())
	}
	if a.NullCount() != 0 {
		t.Errorf("NullCount() = %d, want 0", a.NullCount())
	}
	if a.Offset() != 0 {
		t.Errorf("Offset() = %d, want 0", a.Offset())
	}
	if a.Validity() != nil {
		t.Error("Validity() is not nil for a column with no nulls")
	}
	if a.Strings() != nil {
		t.Error("Strings() is not nil for an int64 column")
	}
	if a.Buffer() == nil {
		t.Error("Buffer() is nil")
	}
	for i := range 3 {
		if !a.IsValid(i) || a.IsNull(i) {
			t.Errorf("value %d reads as missing", i)
		}
	}
}

// TestNewLongBuffer checks that a buffer with room to spare is accepted, since
// a buffer sized in whole bytes or rounded up to an alignment usually has some.
func TestNewLongBuffer(t *testing.T) {
	a := mustNew(t, dtype.Int64, 2, int64s(t, 1, 2, 3, 4), nil)
	if a.Len() != 2 {
		t.Errorf("Len() = %d, want 2", a.Len())
	}
	if got := a.Value[int64](1); got != 2 {
		t.Errorf("Value(1) = %d, want 2", got)
	}
}

func TestNewNulls(t *testing.T) {
	a := mustNew(t, dtype.Int64, 4, int64s(t, 1, 0, 3, 0), validity(true, false, true, false))

	if a.NullCount() != 2 {
		t.Errorf("NullCount() = %d, want 2", a.NullCount())
	}
	if a.Validity() == nil {
		t.Fatal("Validity() is nil for a column with nulls")
	}
	for i, want := range []bool{true, false, true, false} {
		if a.IsValid(i) != want {
			t.Errorf("IsValid(%d) = %v, want %v", i, a.IsValid(i), want)
		}
		if a.IsNull(i) == want {
			t.Errorf("IsNull(%d) = %v, want %v", i, a.IsNull(i), !want)
		}
	}
}

// TestNewAllValidBitmapDropped is the case where the caller hands over a bitmap
// with every bit set. It says the same thing as no bitmap at all and no bitmap
// is the one every kernel reads faster, so it is not kept.
func TestNewAllValidBitmapDropped(t *testing.T) {
	a := mustNew(t, dtype.Int64, 3, int64s(t, 1, 2, 3), bitmap.NewSet(3))

	if a.Validity() != nil {
		t.Error("Validity() kept a bitmap with every bit set")
	}
	if a.NullCount() != 0 {
		t.Errorf("NullCount() = %d, want 0", a.NullCount())
	}
}

// TestNewLongValidity checks that a bitmap longer than the column is accepted,
// because a bitmap is sized in whole bytes and a column is not.
func TestNewLongValidity(t *testing.T) {
	valid := validity(true, false, true, true, true, true, true, true)
	a := mustNew(t, dtype.Int64, 3, int64s(t, 1, 0, 3), valid)

	if a.NullCount() != 1 {
		t.Errorf("NullCount() = %d, want 1", a.NullCount())
	}
}

func TestNewErrors(t *testing.T) {
	tests := []struct {
		name string
		fn   func() (*array.Array, error)
		want string
	}{
		{"nil dtype", func() (*array.Array, error) {
			return array.New(nil, 0, buffer.New(0), nil)
		}, "nil dtype"},
		{"negative length", func() (*array.Array, error) {
			return array.New(dtype.Int64, -1, buffer.New(0), nil)
		}, "negative length"},
		{"nil buffer", func() (*array.Array, error) {
			return array.New(dtype.Int64, 1, nil, nil)
		}, "nil value buffer"},
		{"short buffer", func() (*array.Array, error) {
			return array.New(dtype.Int64, 4, buffer.New(16), nil)
		}, "needs 32 bytes"},
		{"short bool buffer", func() (*array.Array, error) {
			return array.New(dtype.Bool, 9, buffer.New(1), nil)
		}, "needs 2 bytes"},
		{"short validity", func() (*array.Array, error) {
			return array.New(dtype.Int64, 3, buffer.New(24), bitmap.New(2))
		}, "validity bitmap of 2 bits"},
		{"null column", func() (*array.Array, error) {
			return array.New(dtype.Null, 3, buffer.New(0), nil)
		}, "use NewNull"},
		{"string column", func() (*array.Array, error) {
			return array.New(dtype.String, 3, buffer.New(48), nil)
		}, "use NewStrings"},
		{"large string column", func() (*array.Array, error) {
			return array.New(dtype.LargeString, 3, buffer.New(48), nil)
		}, "store it as a string"},
		{"large binary column", func() (*array.Array, error) {
			return array.New(dtype.LargeBinary, 3, buffer.New(48), nil)
		}, "store it as a binary"},
		{"nested column", func() (*array.Array, error) {
			return array.New(dtype.List{Elem: dtype.Int64}, 3, buffer.New(48), nil)
		}, "not supported yet"},
		{"sub byte column", func() (*array.Array, error) {
			return array.New(nibble{}, 8, buffer.New(8), nil)
		}, "not a whole number of bytes"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := tt.fn()
			if err == nil {
				t.Fatalf("New returned %v, want an error", a)
			}
			if a != nil {
				t.Errorf("New returned both an array and the error %v", err)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error is %q, want it to mention %q", err, tt.want)
			}
		})
	}
}

// nibble is a type that claims to be four bits wide, which no dtype in kuma is.
// It is here to reach the check that refuses a width which is not a whole
// number of bytes. That check cannot be reached with a real type, since Bool is
// the only one narrower than a byte and it is handled before the check, but
// dtype.DataType is an interface and this is what someone implementing it
// themselves would run into.
type nibble struct{}

func (nibble) Kind() dtype.Kind { return dtype.InvalidKind }
func (nibble) String() string   { return "nibble" }
func (nibble) Bits() int        { return 4 }

func TestNewNull(t *testing.T) {
	a := array.NewNull(5)

	if a.Len() != 5 {
		t.Errorf("Len() = %d, want 5", a.Len())
	}
	if a.NullCount() != 5 {
		t.Errorf("NullCount() = %d, want 5", a.NullCount())
	}
	if a.Validity() != nil {
		t.Error("a null column allocated a bitmap")
	}
	if a.Buffer() != nil {
		t.Error("a null column allocated a value buffer")
	}
	for i := range 5 {
		if a.IsValid(i) {
			t.Errorf("IsValid(%d) = true on a null column", i)
		}
	}

	// The slice of a null column is a shorter null column, which is the one
	// case where the null count is not a popcount over anything.
	s := a.Slice(1, 4)
	if s.Len() != 3 || s.NullCount() != 3 {
		t.Errorf("Slice(1, 4) has len %d and %d nulls, want 3 and 3", s.Len(), s.NullCount())
	}
	if c := a.Clone(); c.Len() != 5 || c.NullCount() != 5 || c.Buffer() != nil {
		t.Errorf("Clone() = %s, want the same null column", c)
	}
}

func TestNewNullNegative(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("NewNull(-1) did not panic")
		}
	}()
	array.NewNull(-1)
}

func TestNewStrings(t *testing.T) {
	var b strview.Builder
	b.Append([]byte("kuma"))
	b.Append([]byte("a value that is too long to live inside its view"))
	b.Append(nil)

	a, err := array.NewStrings(dtype.String, b.Build(), validity(true, true, false))
	if err != nil {
		t.Fatalf("NewStrings: %v", err)
	}
	if a.Len() != 3 {
		t.Errorf("Len() = %d, want 3", a.Len())
	}
	if a.NullCount() != 1 {
		t.Errorf("NullCount() = %d, want 1", a.NullCount())
	}
	if a.Strings() == nil {
		t.Error("Strings() is nil for a string column")
	}
	if a.Buffer() != nil {
		t.Error("Buffer() is not nil for a string column")
	}
	if got := string(a.Bytes(0)); got != "kuma" {
		t.Errorf("Bytes(0) = %q, want %q", got, "kuma")
	}
}

// TestNewStringsKinds checks the two types the view layout serves. The large
// variants are not among them, since they are converted at the IPC boundary.
func TestNewStringsKinds(t *testing.T) {
	for _, dt := range []dtype.DataType{dtype.String, dtype.Binary} {
		var b strview.Builder
		b.AppendString("kuma")

		a, err := array.NewStrings(dt, b.Build(), nil)
		if err != nil {
			t.Fatalf("NewStrings(%s): %v", dt, err)
		}
		if !dtype.Equal(a.DType(), dt) {
			t.Errorf("DType() = %s, want %s", a.DType(), dt)
		}
	}
}

func TestNewStringsErrors(t *testing.T) {
	var b strview.Builder
	b.AppendString("kuma")
	d := b.Build()

	tests := []struct {
		name string
		fn   func() (*array.Array, error)
		want string
	}{
		{"nil dtype", func() (*array.Array, error) {
			return array.NewStrings(nil, d, nil)
		}, "nil dtype"},
		{"wrong dtype", func() (*array.Array, error) {
			return array.NewStrings(dtype.Int64, d, nil)
		}, "NewStrings on a int64 column"},
		{"nil values", func() (*array.Array, error) {
			return array.NewStrings(dtype.String, nil, nil)
		}, "nil values"},
		{"short validity", func() (*array.Array, error) {
			return array.NewStrings(dtype.String, d, bitmap.New(0))
		}, "validity bitmap of 0 bits"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := tt.fn()
			if err == nil {
				t.Fatalf("NewStrings returned %v, want an error", a)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error is %q, want it to mention %q", err, tt.want)
			}
		})
	}
}

func TestOf(t *testing.T) {
	tests := []struct {
		name string
		a    *array.Array
		dt   dtype.DataType
	}{
		{"int8", array.Of[int8](1, 2), dtype.Int8},
		{"int16", array.Of[int16](1, 2), dtype.Int16},
		{"int32", array.Of[int32](1, 2), dtype.Int32},
		{"int64", array.Of[int64](1, 2), dtype.Int64},
		{"uint8", array.Of[uint8](1, 2), dtype.Uint8},
		{"uint16", array.Of[uint16](1, 2), dtype.Uint16},
		{"uint32", array.Of[uint32](1, 2), dtype.Uint32},
		{"uint64", array.Of[uint64](1, 2), dtype.Uint64},
		{"float32", array.Of[float32](1, 2), dtype.Float32},
		{"float64", array.Of[float64](1, 2), dtype.Float64},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !dtype.Equal(tt.a.DType(), tt.dt) {
				t.Errorf("DType() = %s, want %s", tt.a.DType(), tt.dt)
			}
			if tt.a.Len() != 2 {
				t.Errorf("Len() = %d, want 2", tt.a.Len())
			}
			if tt.a.NullCount() != 0 {
				t.Errorf("NullCount() = %d, want 0", tt.a.NullCount())
			}
		})
	}

	if got := array.Of[int64](7, 8, 9).Values[int64](); got[2] != 9 {
		t.Errorf("Values() = %v, want the values it was given", got)
	}
	if got := array.Of[float64]().Len(); got != 0 {
		t.Errorf("Of() with no values has length %d, want 0", got)
	}
}

func TestOfBools(t *testing.T) {
	a := array.OfBools(true, false, true, true, false, false, true, false, true)

	if a.Len() != 9 {
		t.Fatalf("Len() = %d, want 9", a.Len())
	}
	for i, want := range []bool{true, false, true, true, false, false, true, false, true} {
		if a.Bool(i) != want {
			t.Errorf("Bool(%d) = %v, want %v", i, a.Bool(i), want)
		}
	}
	if got := array.OfBools().Len(); got != 0 {
		t.Errorf("OfBools() with no values has length %d, want 0", got)
	}
}

func TestOfStrings(t *testing.T) {
	a := array.OfStrings("kuma", "", "a value that is too long to live inside its view")

	if a.Len() != 3 {
		t.Fatalf("Len() = %d, want 3", a.Len())
	}
	for i, want := range []string{"kuma", "", "a value that is too long to live inside its view"} {
		if got := string(a.Bytes(i)); got != want {
			t.Errorf("Bytes(%d) = %q, want %q", i, got, want)
		}
	}
}

// TestSlice walks every start and end position on a column with nulls, because
// the null count of a slice is a popcount over a range that does not begin on a
// byte boundary and that is the part with somewhere to go wrong.
func TestSlice(t *testing.T) {
	const n = 70

	values := make([]int64, n)
	present := make([]bool, n)
	for i := range n {
		values[i] = int64(i)
		present[i] = i%3 != 0
	}
	a := mustNew(t, dtype.Int64, n, int64s(t, values...), validity(present...))

	for i := range n + 1 {
		for j := i; j <= n; j++ {
			s := a.Slice(i, j)

			if s.Len() != j-i {
				t.Fatalf("Slice(%d, %d).Len() = %d, want %d", i, j, s.Len(), j-i)
			}
			if s.Offset() != i {
				t.Fatalf("Slice(%d, %d).Offset() = %d, want %d", i, j, s.Offset(), i)
			}

			want := 0
			for k := i; k < j; k++ {
				if !present[k] {
					want++
				}
			}
			if s.NullCount() != want {
				t.Fatalf("Slice(%d, %d).NullCount() = %d, want %d", i, j, s.NullCount(), want)
			}
			for k := range s.Len() {
				if s.IsValid(k) != present[i+k] {
					t.Fatalf("Slice(%d, %d).IsValid(%d) disagrees with the model", i, j, k)
				}
				if s.IsValid(k) && s.Value[int64](k) != values[i+k] {
					t.Fatalf("Slice(%d, %d).Value(%d) = %d, want %d",
						i, j, k, s.Value[int64](k), values[i+k])
				}
			}
		}
	}

	// The original is untouched, since slicing shares rather than copies.
	if a.Len() != n || a.Offset() != 0 {
		t.Errorf("slicing changed the array it was called on")
	}
}

// TestSliceOfSlice checks that offsets add up rather than replace each other.
func TestSliceOfSlice(t *testing.T) {
	a := mustNew(t, dtype.Int64, 10, int64s(t, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9), nil)

	s := a.Slice(2, 8).Slice(1, 4)
	if s.Offset() != 3 || s.Len() != 3 {
		t.Fatalf("offset %d length %d, want 3 and 3", s.Offset(), s.Len())
	}
	for i, want := range []int64{3, 4, 5} {
		if got := s.Value[int64](i); got != want {
			t.Errorf("Value(%d) = %d, want %d", i, got, want)
		}
	}
}

// TestSliceNoNulls is the fast path, where there is nothing to count.
func TestSliceNoNulls(t *testing.T) {
	a := mustNew(t, dtype.Int64, 10, int64s(t, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9), nil)

	s := a.Slice(3, 7)
	if s.NullCount() != 0 || s.Validity() != nil {
		t.Errorf("slicing a column with no nulls produced %d nulls", s.NullCount())
	}
}

func TestSliceStrings(t *testing.T) {
	a := array.OfStrings("one", "two", "three", "four")

	s := a.Slice(1, 3)
	if s.Len() != 2 {
		t.Fatalf("Len() = %d, want 2", s.Len())
	}
	for i, want := range []string{"two", "three"} {
		if got := string(s.Bytes(i)); got != want {
			t.Errorf("Bytes(%d) = %q, want %q", i, got, want)
		}
	}
}

func TestSliceBools(t *testing.T) {
	model := []bool{true, false, true, true, false, false, true, false, true, true, false}
	a := array.OfBools(model...)

	for i := range len(model) + 1 {
		for j := i; j <= len(model); j++ {
			s := a.Slice(i, j)
			for k := range s.Len() {
				if s.Bool(k) != model[i+k] {
					t.Fatalf("Slice(%d, %d).Bool(%d) disagrees with the model", i, j, k)
				}
			}
		}
	}
}

func TestSlicePanics(t *testing.T) {
	a := mustNew(t, dtype.Int64, 4, int64s(t, 1, 2, 3, 4), nil)

	tests := []struct {
		name string
		fn   func()
	}{
		{"past the end", func() { a.Slice(0, 5) }},
		{"backwards", func() { a.Slice(3, 2) }},
		{"negative", func() { a.Slice(-1, 2) }},
		{"index past the end", func() { a.IsValid(4) }},
		{"index negative", func() { a.IsValid(-1) }},
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

// TestClone checks that a clone holds the values in range and nothing else, and
// that it shares no memory with what it came from.
func TestClone(t *testing.T) {
	const n = 70

	values := make([]int64, n)
	present := make([]bool, n)
	for i := range n {
		values[i] = int64(i)
		present[i] = i%5 != 0
	}
	a := mustNew(t, dtype.Int64, n, int64s(t, values...), validity(present...))

	// An unaligned range, so that the validity bits have to be shifted.
	s := a.Slice(11, 60)
	c := s.Clone()

	if c.Offset() != 0 {
		t.Errorf("Clone().Offset() = %d, want 0", c.Offset())
	}
	if c.Len() != s.Len() {
		t.Fatalf("Clone().Len() = %d, want %d", c.Len(), s.Len())
	}
	if c.NullCount() != s.NullCount() {
		t.Errorf("Clone().NullCount() = %d, want %d", c.NullCount(), s.NullCount())
	}
	if c.Buffer().Len() != c.Len()*8 {
		t.Errorf("Clone() kept %d bytes for %d values", c.Buffer().Len(), c.Len())
	}
	for i := range c.Len() {
		if c.IsValid(i) != s.IsValid(i) {
			t.Fatalf("Clone().IsValid(%d) disagrees with the original", i)
		}
		if c.IsValid(i) && c.Value[int64](i) != s.Value[int64](i) {
			t.Fatalf("Clone().Value(%d) = %d, want %d", i, c.Value[int64](i), s.Value[int64](i))
		}
	}

	// Writing through the clone must not reach the original. Nothing in the
	// package modifies an array, so this reaches around it on purpose.
	c.Buffer().Bytes()[0] = 0xFF
	if s.Value[int64](0) != 11 {
		t.Error("writing to the clone changed the array it came from")
	}
}

func TestCloneBools(t *testing.T) {
	model := []bool{true, false, true, true, false, false, true, false, true, true, false}
	a := array.OfBools(model...)

	c := a.Slice(3, 10).Clone()
	if c.Len() != 7 {
		t.Fatalf("Len() = %d, want 7", c.Len())
	}
	if c.Offset() != 0 {
		t.Errorf("Offset() = %d, want 0", c.Offset())
	}
	for i := range c.Len() {
		if c.Bool(i) != model[3+i] {
			t.Errorf("Bool(%d) = %v, want %v", i, c.Bool(i), model[3+i])
		}
	}
}

// TestCloneStrings checks that cloning a slice of a string column leaves the
// blocks of the values outside the slice behind, which is the reason to clone
// at all.
func TestCloneStrings(t *testing.T) {
	long := strings.Repeat("x", 1000)
	a := array.OfStrings(long, "kuma", long, "bear", long)

	c := a.Slice(3, 4).Clone()
	if c.Len() != 1 {
		t.Fatalf("Len() = %d, want 1", c.Len())
	}
	if got := string(c.Bytes(0)); got != "bear" {
		t.Errorf("Bytes(0) = %q, want %q", got, "bear")
	}

	var held int
	for _, b := range c.Strings().Blocks() {
		held += b.Len()
	}
	if held > 100 {
		t.Errorf("the clone is holding %d bytes of blocks for one four byte value", held)
	}
}

func TestCloneNoValidity(t *testing.T) {
	c := array.Of[int64](1, 2, 3).Clone()
	if c.Validity() != nil || c.NullCount() != 0 {
		t.Error("cloning a column with no nulls invented a bitmap")
	}
}

func TestString(t *testing.T) {
	a := mustNew(t, dtype.Int64, 4, int64s(t, 1, 0, 3, 4), validity(true, false, true, true))

	if got, want := a.String(), "array.Array{int64, len 4, nulls 1, offset 0}"; got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
	if got, want := a.Slice(2, 4).String(), "array.Array{int64, len 2, nulls 0, offset 2}"; got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
}
