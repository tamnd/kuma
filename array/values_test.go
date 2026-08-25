package array_test

import (
	"bytes"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
)

func TestValues(t *testing.T) {
	a := array.Of[int64](10, 20, 30, 40)

	got := a.Values[int64]()
	if len(got) != 4 {
		t.Fatalf("len(Values()) = %d, want 4", len(got))
	}
	for i, want := range []int64{10, 20, 30, 40} {
		if got[i] != want {
			t.Errorf("Values()[%d] = %d, want %d", i, got[i], want)
		}
		if v := a.Value[int64](i); v != want {
			t.Errorf("Value(%d) = %d, want %d", i, v, want)
		}
	}
}

// TestValuesAliases is the property the doc comment promises and the reason
// reading a column costs nothing. The slice is the buffer, not a copy of it.
func TestValuesAliases(t *testing.T) {
	a := array.Of[int64](1, 2, 3)

	a.Values[int64]()[1] = 99
	if got := a.Value[int64](1); got != 99 {
		t.Errorf("Value(1) = %d after writing through Values, want 99", got)
	}
}

// TestValuesSliced checks that a slice reads from its own offset rather than
// from the start of the buffer it shares.
func TestValuesSliced(t *testing.T) {
	a := array.Of[int32](0, 1, 2, 3, 4, 5, 6, 7)

	s := a.Slice(3, 6)
	got := s.Values[int32]()
	if len(got) != 3 {
		t.Fatalf("len(Values()) = %d, want 3", len(got))
	}
	for i, want := range []int32{3, 4, 5} {
		if got[i] != want {
			t.Errorf("Values()[%d] = %d, want %d", i, got[i], want)
		}
	}
}

func TestValuesEmpty(t *testing.T) {
	if got := array.Of[int64]().Values[int64](); got != nil {
		t.Errorf("Values() on an empty column = %v, want nil", got)
	}
	if got := array.Of[int64](1, 2, 3).Slice(2, 2).Values[int64](); got != nil {
		t.Errorf("Values() on an empty slice = %v, want nil", got)
	}
}

func TestValuesEveryNumericType(t *testing.T) {
	if got := array.Of[int8](-1, 2).Values[int8](); got[0] != -1 || got[1] != 2 {
		t.Errorf("int8 round trip gave %v", got)
	}
	if got := array.Of[int16](-1, 2).Values[int16](); got[0] != -1 || got[1] != 2 {
		t.Errorf("int16 round trip gave %v", got)
	}
	if got := array.Of[int32](-1, 2).Values[int32](); got[0] != -1 || got[1] != 2 {
		t.Errorf("int32 round trip gave %v", got)
	}
	if got := array.Of[int64](-1, 2).Values[int64](); got[0] != -1 || got[1] != 2 {
		t.Errorf("int64 round trip gave %v", got)
	}
	if got := array.Of[uint8](1, 255).Values[uint8](); got[0] != 1 || got[1] != 255 {
		t.Errorf("uint8 round trip gave %v", got)
	}
	if got := array.Of[uint16](1, 65535).Values[uint16](); got[0] != 1 || got[1] != 65535 {
		t.Errorf("uint16 round trip gave %v", got)
	}
	if got := array.Of[uint32](1, 4294967295).Values[uint32](); got[0] != 1 || got[1] != 4294967295 {
		t.Errorf("uint32 round trip gave %v", got)
	}
	if got := array.Of[uint64](1, 18446744073709551615).Values[uint64](); got[0] != 1 || got[1] != 18446744073709551615 {
		t.Errorf("uint64 round trip gave %v", got)
	}
	if got := array.Of[float32](-1.5, 2.25).Values[float32](); got[0] != -1.5 || got[1] != 2.25 {
		t.Errorf("float32 round trip gave %v", got)
	}
	if got := array.Of[float64](-1.5, 2.25).Values[float64](); got[0] != -1.5 || got[1] != 2.25 {
		t.Errorf("float64 round trip gave %v", got)
	}
}

// TestValuesTemporal is the reason Values checks the layout rather than the
// type. A timestamp column is int64 microseconds, and a kernel that adds a day
// to every value is an integer kernel, so it gets an []int64 and not a copy.
func TestValuesTemporal(t *testing.T) {
	tests := []struct {
		name string
		dt   dtype.DataType
		bits int
	}{
		{"timestamp", dtype.Timestamp{Unit: dtype.Microsecond}, 64},
		{"timestamp with a zone", dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "Europe/London"}, 64},
		{"duration", dtype.Duration{Unit: dtype.Second}, 64},
		{"time64", dtype.Time64{Unit: dtype.Nanosecond}, 64},
		{"date64", dtype.Date64, 64},
		{"date32", dtype.Date32, 32},
		{"time32", dtype.Time32{Unit: dtype.Second}, 32},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := mustNew(t, tt.dt, 2, buffer.New(2*tt.bits/8), nil)

			switch tt.bits {
			case 64:
				a.Values[int64]()[1] = 1234567
				if got := a.Value[int64](1); got != 1234567 {
					t.Errorf("Value(1) = %d, want 1234567", got)
				}
			case 32:
				a.Values[int32]()[1] = 19000
				if got := a.Value[int32](1); got != 19000 {
					t.Errorf("Value(1) = %d, want 19000", got)
				}
			}
		})
	}
}

// TestValuesWrongType covers the mistake this whole check exists for. Reading a
// float64 column as an int64 gives numbers that are not wrong so much as
// unrelated, and it is the kind of thing that reads fine in a diff.
func TestValuesWrongType(t *testing.T) {
	tests := []struct {
		name string
		fn   func()
		want string
	}{
		{"float64 as int64", func() { array.Of[float64](1).Values[int64]() },
			"cannot read a float64 column as int64"},
		{"int64 as float64", func() { array.Of[int64](1).Values[float64]() },
			"cannot read a int64 column as float64"},
		{"int32 as int64", func() { array.Of[int32](1).Values[int64]() },
			"cannot read a int32 column as int64"},
		{"int64 as uint64", func() { array.Of[int64](1).Values[uint64]() },
			"cannot read a int64 column as uint64"},
		{"bool as uint8", func() { array.OfBools(true).Values[uint8]() },
			"cannot read a bool column as uint8"},
		{"string as uint8", func() { array.OfStrings("kuma").Values[uint8]() },
			"cannot read a string column as uint8"},
		{"null as int64", func() { array.NewNull(1).Values[int64]() },
			"cannot read a null column as int64"},
		{"timestamp as int32", func() {
			a, err := array.New(dtype.Timestamp{Unit: dtype.Second}, 1, buffer.New(8), nil)
			if err != nil {
				panic(err)
			}
			a.Values[int32]()
		}, "as int32"},
		{"value of the wrong type", func() { array.Of[int64](1).Value[int32](0) },
			"cannot read a int64 column as int32"},
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

func TestValueOutOfRange(t *testing.T) {
	a := array.Of[int64](1, 2, 3)

	tests := []struct {
		name string
		fn   func()
	}{
		{"past the end", func() { a.Value[int64](3) }},
		{"negative", func() { a.Value[int64](-1) }},
		{"past the end of a slice", func() { a.Slice(0, 2).Value[int64](2) }},
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

func TestBools(t *testing.T) {
	model := []bool{true, false, true, true, false, false, true, false, true, true, false, true, false}
	a := array.OfBools(model...)

	bits := a.Bools()
	if bits.Len() != len(model) {
		t.Fatalf("Bools().Len() = %d, want %d", bits.Len(), len(model))
	}
	for i, want := range model {
		if bits.Get(i) != want {
			t.Errorf("Bools().Get(%d) = %v, want %v", i, bits.Get(i), want)
		}
	}

	// On a slice the bitmap covers the shared buffer, so value i of the array is
	// bit Offset()+i of the result. A kernel working a word at a time has to
	// know that, and this is the test that says so.
	s := a.Slice(5, 11)
	bits = s.Bools()
	if bits.Len() != s.Offset()+s.Len() {
		t.Fatalf("Bools().Len() = %d, want %d", bits.Len(), s.Offset()+s.Len())
	}
	for i := range s.Len() {
		if bits.Get(s.Offset()+i) != model[5+i] {
			t.Errorf("bit %d of the slice disagrees with the model", i)
		}
	}
}

func TestBoolPanics(t *testing.T) {
	tests := []struct {
		name string
		fn   func()
	}{
		{"bool on an int64 column", func() { array.Of[int64](1).Bool(0) }},
		{"bools on an int64 column", func() { array.Of[int64](1).Bools() }},
		{"bool past the end", func() { array.OfBools(true).Bool(1) }},
		{"bool negative", func() { array.OfBools(true).Bool(-1) }},
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

func TestBytesFixedWidth(t *testing.T) {
	tests := []struct {
		name  string
		dt    dtype.DataType
		width int
	}{
		{"fixed size binary", dtype.FixedSizeBinary{ByteWidth: 3}, 3},
		{"decimal128", dtype.Decimal128{Precision: 18, Scale: 2}, 16},
		{"decimal256", dtype.Decimal256{Precision: 50, Scale: 4}, 32},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			const n = 4
			buf := buffer.New(n * tt.width)
			for i := range n * tt.width {
				buf.Bytes()[i] = byte(i)
			}
			a := mustNew(t, tt.dt, n, buf, nil)

			for i := range n {
				want := buf.Bytes()[i*tt.width : (i+1)*tt.width]
				if got := a.Bytes(i); !bytes.Equal(got, want) {
					t.Errorf("Bytes(%d) = %v, want %v", i, got, want)
				}
			}

			// A slice reads from its own offset, and the result must not run
			// past the value it belongs to.
			s := a.Slice(2, 4)
			if got, want := s.Bytes(0), buf.Bytes()[2*tt.width:3*tt.width]; !bytes.Equal(got, want) {
				t.Errorf("Bytes(0) on a slice = %v, want %v", got, want)
			}
			if got := s.Bytes(0); cap(got) != tt.width {
				t.Errorf("Bytes(0) has capacity %d, want %d, so appending to it would reach the next value",
					cap(got), tt.width)
			}
		})
	}
}

func TestBytesPanics(t *testing.T) {
	tests := []struct {
		name string
		fn   func()
	}{
		{"on an int64 column", func() { array.Of[int64](1).Bytes(0) }},
		{"on a bool column", func() { array.OfBools(true).Bytes(0) }},
		{"past the end", func() { array.OfStrings("kuma").Bytes(1) }},
		{"negative", func() { array.OfStrings("kuma").Bytes(-1) }},
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
