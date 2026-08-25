package array_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

func mustBuilder(t *testing.T, dt dtype.DataType) *array.Builder {
	t.Helper()
	b, err := array.NewBuilder(dt)
	if err != nil {
		t.Fatalf("NewBuilder(%s): %v", dt, err)
	}
	return b
}

func TestBuilder(t *testing.T) {
	b := mustBuilder(t, dtype.Int64)

	b.Append[int64](10)
	b.AppendNull()
	b.Append[int64](30)
	b.AppendNulls(2)
	b.AppendValues([]int64{60, 70})

	if b.Len() != 7 || b.NullCount() != 3 {
		t.Fatalf("the builder has %d values and %d nulls, want 7 and 3", b.Len(), b.NullCount())
	}

	a := b.Finish()
	if a.Len() != 7 || a.NullCount() != 3 {
		t.Fatalf("%s, want 7 values and 3 nulls", a)
	}

	present := []bool{true, false, true, false, false, true, true}
	values := []int64{10, 0, 30, 0, 0, 60, 70}
	for i := range a.Len() {
		if a.IsValid(i) != present[i] {
			t.Errorf("IsValid(%d) = %v, want %v", i, a.IsValid(i), present[i])
		}
		if got := a.Value[int64](i); got != values[i] {
			t.Errorf("Value(%d) = %d, want %d", i, got, values[i])
		}
	}

	// Finish resets, so the builder is ready for the next chunk.
	if b.Len() != 0 || b.NullCount() != 0 {
		t.Errorf("after Finish the builder has %d values and %d nulls", b.Len(), b.NullCount())
	}
}

// TestBuilderNoNulls is the case the lazy bitmap is for. A column with nothing
// missing should come out with no validity bitmap at all.
func TestBuilderNoNulls(t *testing.T) {
	b := mustBuilder(t, dtype.Int64)
	b.AppendValues([]int64(nil))
	b.AppendValues([]int64{1, 2, 3})

	a := b.Finish()
	if a.Len() != 3 {
		t.Errorf("Len() = %d, want 3, so an empty AppendValues added a value", a.Len())
	}
	if a.Validity() != nil {
		t.Error("a column with no nulls came out with a validity bitmap")
	}
	if a.NullCount() != 0 {
		t.Errorf("NullCount() = %d, want 0", a.NullCount())
	}
}

// TestBuilderNullAfterValues is the other half of that. The values appended
// before the first null were all present, and their bits have to be filled in
// when the bitmap finally appears.
func TestBuilderNullAfterValues(t *testing.T) {
	for _, before := range []int{1, 7, 8, 9, 63, 64, 65} {
		b := mustBuilder(t, dtype.Int32)
		for i := range before {
			b.Append(int32(i))
		}
		b.AppendNull()

		a := b.Finish()
		if a.NullCount() != 1 {
			t.Fatalf("with %d values before the null, NullCount() = %d, want 1", before, a.NullCount())
		}
		for i := range before {
			if !a.IsValid(i) {
				t.Fatalf("with %d values before the null, value %d reads as missing", before, i)
			}
			if got := a.Value[int32](i); got != int32(i) {
				t.Fatalf("Value(%d) = %d, want %d", i, got, i)
			}
		}
		if a.IsValid(before) {
			t.Fatalf("with %d values before it, the null reads as present", before)
		}
	}
}

func TestBuilderNumericTypes(t *testing.T) {
	t.Run("int8", func(t *testing.T) {
		b := mustBuilder(t, dtype.Int8)
		b.AppendValues([]int8{-1, 2})
		if got := b.Finish().Values[int8](); got[0] != -1 || got[1] != 2 {
			t.Errorf("got %v", got)
		}
	})
	t.Run("int16", func(t *testing.T) {
		b := mustBuilder(t, dtype.Int16)
		b.AppendValues([]int16{-1, 2})
		if got := b.Finish().Values[int16](); got[0] != -1 || got[1] != 2 {
			t.Errorf("got %v", got)
		}
	})
	t.Run("uint32", func(t *testing.T) {
		b := mustBuilder(t, dtype.Uint32)
		b.AppendValues([]uint32{1, 4294967295})
		if got := b.Finish().Values[uint32](); got[0] != 1 || got[1] != 4294967295 {
			t.Errorf("got %v", got)
		}
	})
	t.Run("float64", func(t *testing.T) {
		b := mustBuilder(t, dtype.Float64)
		b.Append[float64](1.5)
		b.Append[float64](-2.25)
		if got := b.Finish().Values[float64](); got[0] != 1.5 || got[1] != -2.25 {
			t.Errorf("got %v", got)
		}
	})
	t.Run("timestamp takes int64", func(t *testing.T) {
		b := mustBuilder(t, dtype.Timestamp{Unit: dtype.Microsecond})
		b.AppendValues([]int64{1700000000000000, 1700000001000000})
		if got := b.Finish().Values[int64](); got[1]-got[0] != 1000000 {
			t.Errorf("got %v", got)
		}
	})
}

func TestBuilderBools(t *testing.T) {
	model := []bool{true, false, true, true, false, false, true, false, true, true}

	b := mustBuilder(t, dtype.Bool)
	b.Grow(len(model))
	b.AppendBool(model[0])
	b.AppendBools(model[1:5])
	b.AppendNull()
	b.AppendBools(model[5:])

	a := b.Finish()
	if a.Len() != len(model)+1 || a.NullCount() != 1 {
		t.Fatalf("%s, want %d values and 1 null", a, len(model)+1)
	}
	if a.IsValid(5) {
		t.Error("the null reads as present")
	}
	if a.Bool(5) {
		t.Error("the value under the null is not false")
	}
	for i := range 5 {
		if a.Bool(i) != model[i] {
			t.Errorf("Bool(%d) = %v, want %v", i, a.Bool(i), model[i])
		}
	}
	for i := 5; i < len(model); i++ {
		if a.Bool(i+1) != model[i] {
			t.Errorf("Bool(%d) = %v, want %v", i+1, a.Bool(i+1), model[i])
		}
	}
}

func TestBuilderStrings(t *testing.T) {
	long := strings.Repeat("x", 100)

	for _, dt := range []dtype.DataType{dtype.String, dtype.Binary} {
		b := mustBuilder(t, dt)
		b.Grow(4)
		b.AppendString("kuma")
		b.AppendBytes([]byte(long))
		b.AppendNull()
		b.AppendString("")

		a := b.Finish()
		if a.Len() != 4 || a.NullCount() != 1 {
			t.Fatalf("%s, want 4 values and 1 null", a)
		}
		for i, want := range []string{"kuma", long, "", ""} {
			if got := string(a.Bytes(i)); got != want {
				t.Errorf("%s: Bytes(%d) = %q, want %q", dt, i, got, want)
			}
		}
		if a.IsValid(2) {
			t.Errorf("%s: the null reads as present", dt)
		}
	}
}

// TestBuilderAppendBytesCopies is the promise that lets a reader append out of
// the buffer it reads into.
func TestBuilderAppendBytesCopies(t *testing.T) {
	scratch := []byte("kuma")

	b := mustBuilder(t, dtype.String)
	b.AppendBytes(scratch)
	copy(scratch, "bear")

	if got := string(b.Finish().Bytes(0)); got != "kuma" {
		t.Errorf("Bytes(0) = %q, want %q, so AppendBytes kept the caller's slice", got, "kuma")
	}
}

func TestBuilderFixedWidthBytes(t *testing.T) {
	dt := dtype.FixedSizeBinary{ByteWidth: 3}

	b := mustBuilder(t, dt)
	b.AppendBytes([]byte{1, 2, 3})
	b.AppendNull()
	b.AppendBytes([]byte{7, 8, 9})

	a := b.Finish()
	if a.Len() != 3 || a.NullCount() != 1 {
		t.Fatalf("%s, want 3 values and 1 null", a)
	}
	for i, want := range [][]byte{{1, 2, 3}, {0, 0, 0}, {7, 8, 9}} {
		if got := a.Bytes(i); !bytes.Equal(got, want) {
			t.Errorf("Bytes(%d) = %v, want %v", i, got, want)
		}
	}
}

func TestBuilderNullColumn(t *testing.T) {
	b := mustBuilder(t, dtype.Null)
	b.AppendNulls(5)

	a := b.Finish()
	if a.Len() != 5 || a.NullCount() != 5 {
		t.Fatalf("%s, want 5 values and 5 nulls", a)
	}
	if a.Validity() != nil {
		t.Error("a null column came out with a validity bitmap")
	}
	if a.IsValid(0) {
		t.Error("a value of a null column reads as present")
	}
}

// TestBuilderFinishHandsOverMemory checks that two columns out of one builder
// do not share anything, since the second one starts from an empty buffer
// rather than from the one already handed away.
func TestBuilderFinishHandsOverMemory(t *testing.T) {
	b := mustBuilder(t, dtype.Int64)

	b.AppendValues([]int64{1, 2, 3})
	first := b.Finish()

	b.AppendValues([]int64{4, 5, 6})
	second := b.Finish()

	if got := first.Values[int64](); got[0] != 1 || got[2] != 3 {
		t.Errorf("the first column changed when the second was built: %v", got)
	}
	if got := second.Values[int64](); got[0] != 4 || got[2] != 6 {
		t.Errorf("the second column is %v, want the values it was given", got)
	}
	if &first.Values[int64]()[0] == &second.Values[int64]()[0] {
		t.Error("both columns are looking at the same memory")
	}
}

func TestBuilderReset(t *testing.T) {
	b := mustBuilder(t, dtype.Int64)
	b.AppendValues([]int64{1, 2, 3})
	b.AppendNull()
	b.Reset()

	if b.Len() != 0 || b.NullCount() != 0 {
		t.Fatalf("after Reset the builder has %d values and %d nulls", b.Len(), b.NullCount())
	}

	b.AppendValues([]int64{9})
	a := b.Finish()
	if a.Len() != 1 || a.NullCount() != 0 || a.Value[int64](0) != 9 {
		t.Errorf("%s holds %d, want one value of 9 and no nulls", a, a.Value[int64](0))
	}
}

func TestBuilderGrow(t *testing.T) {
	for _, dt := range []dtype.DataType{dtype.Int64, dtype.Bool, dtype.String, dtype.Null} {
		b := mustBuilder(t, dt)
		b.AppendNull() // so that the validity bitmap exists and is grown too
		b.Grow(1000)

		if b.Len() != 1 {
			t.Errorf("%s: Grow changed the length to %d", dt, b.Len())
		}
		if a := b.Finish(); a.Len() != 1 || a.NullCount() != 1 {
			t.Errorf("%s: %s, want one value and one null", dt, a)
		}
	}
}

func TestNewBuilderErrors(t *testing.T) {
	tests := []struct {
		name string
		dt   dtype.DataType
		want string
	}{
		{"nil dtype", nil, "nil dtype"},
		{"nested", dtype.List{Elem: dtype.Int64}, "not supported yet"},
		{"large string", dtype.LargeString, "store it as a string"},
		{"sub byte", nibble{}, "not a whole number of bytes"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := array.NewBuilder(tt.dt)
			if err == nil {
				t.Fatalf("NewBuilder returned %v, want an error", b)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error is %q, want it to mention %q", err, tt.want)
			}
		})
	}
}

func TestBuilderPanics(t *testing.T) {
	tests := []struct {
		name string
		fn   func()
		want string
	}{
		{"wrong type", func() {
			mustBuilder(t, dtype.Int64).Append[int32](1)
		}, "cannot append int32 to a int64 column"},
		{"wrong type in bulk", func() {
			mustBuilder(t, dtype.Int64).AppendValues([]float64{1})
		}, "cannot append float64 to a int64 column"},
		{"numeric on a string column", func() {
			mustBuilder(t, dtype.String).Append[int64](1)
		}, "cannot append int64 to a string column"},
		{"bool on an int64 column", func() {
			mustBuilder(t, dtype.Int64).AppendBool(true)
		}, "AppendBool on a int64 column"},
		{"bools on an int64 column", func() {
			mustBuilder(t, dtype.Int64).AppendBools([]bool{true})
		}, "AppendBools on a int64 column"},
		{"string on an int64 column", func() {
			mustBuilder(t, dtype.Int64).AppendString("kuma")
		}, "AppendString on a int64 column"},
		{"bytes on an int64 column", func() {
			mustBuilder(t, dtype.Int64).AppendBytes([]byte{1})
		}, "AppendBytes on a int64 column"},
		{"bytes of the wrong width", func() {
			mustBuilder(t, dtype.FixedSizeBinary{ByteWidth: 3}).AppendBytes([]byte{1, 2})
		}, "is 3 bytes, got 2"},
		{"negative grow", func() {
			mustBuilder(t, dtype.Int64).Grow(-1)
		}, "negative grow"},
		{"negative nulls", func() {
			mustBuilder(t, dtype.Int64).AppendNulls(-1)
		}, "negative count"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("did not panic")
				}
				msg, ok := r.(string)
				if !ok {
					t.Fatalf("panicked with %T, want a string", r)
				}
				if !strings.Contains(msg, tt.want) {
					t.Errorf("panicked with %q, want it to mention %q", msg, tt.want)
				}
			}()
			tt.fn()
		})
	}
}

// TestBuilderEmpty checks that a builder that was handed nothing still produces
// a usable column, since a reader that hits end of file on a chunk boundary
// finishes an empty one.
func TestBuilderEmpty(t *testing.T) {
	for _, dt := range []dtype.DataType{dtype.Int64, dtype.Bool, dtype.String, dtype.Null} {
		b := mustBuilder(t, dt)
		b.AppendNulls(0)
		a := b.Finish()

		if a.Len() != 0 || a.NullCount() != 0 {
			t.Errorf("%s: %s, want an empty column", dt, a)
		}
		if !dtype.Equal(a.DType(), dt) {
			t.Errorf("%s: DType() = %s", dt, a.DType())
		}
	}
}

func TestBuilderDType(t *testing.T) {
	dt := dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "Europe/London"}
	if got := mustBuilder(t, dt).DType(); !dtype.Equal(got, dt) {
		t.Errorf("DType() = %s, want %s", got, dt)
	}
}
