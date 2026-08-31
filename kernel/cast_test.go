package kernel_test

import (
	"errors"
	"math"
	"strconv"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// TestCast is one row per pair of types, with the values chosen so that the
// answer says something. A number that survives every width is boring and the
// ones here do not.
func TestCast(t *testing.T) {
	tests := []struct {
		name string
		from dtype.DataType
		to   dtype.DataType
		in   []any
		want []any
	}{{
		name: "widening an integer",
		from: dtype.Int8, to: dtype.Int64,
		in:   []any{int8(-128), int8(0), nil, int8(127)},
		want: []any{int64(-128), int64(0), nil, int64(127)},
	}, {
		name: "narrowing an integer that fits",
		from: dtype.Int64, to: dtype.Int8,
		in:   []any{int64(-128), int64(127)},
		want: []any{int8(-128), int8(127)},
	}, {
		name: "signed to unsigned",
		from: dtype.Int32, to: dtype.Uint64,
		in:   []any{int32(0), int32(1), int32(math.MaxInt32)},
		want: []any{uint64(0), uint64(1), uint64(math.MaxInt32)},
	}, {
		name: "the largest uint64 into a float64",
		from: dtype.Uint64, to: dtype.Float64,
		in:   []any{uint64(math.MaxUint64)},
		want: []any{float64(math.MaxUint64)},
	}, {
		name: "a float becomes an integer by losing the fraction",
		from: dtype.Float64, to: dtype.Int64,
		in:   []any{3.9, -3.9, 0.5, -0.5},
		want: []any{int64(3), int64(-3), int64(0), int64(0)},
	}, {
		name: "an integer into a float32 that cannot hold all of it",
		from: dtype.Int64, to: dtype.Float32,
		in:   []any{int64(16777217)},
		want: []any{float32(16777216)},
	}, {
		name: "float64 to float32 keeps the infinities",
		from: dtype.Float64, to: dtype.Float32,
		in:   []any{math.Inf(1), math.Inf(-1)},
		want: []any{float32(math.Inf(1)), float32(math.Inf(-1))},
	}, {
		name: "a number becomes a boolean by being nonzero",
		from: dtype.Int64, to: dtype.Bool,
		in:   []any{int64(0), int64(1), int64(-1), nil},
		want: []any{false, true, true, nil},
	}, {
		name: "a float NaN is not zero and so is true",
		from: dtype.Float64, to: dtype.Bool,
		in:   []any{math.NaN(), 0.0, math.Copysign(0, -1)},
		want: []any{true, false, false},
	}, {
		name: "a boolean becomes zero or one",
		from: dtype.Bool, to: dtype.Int8,
		in:   []any{true, false, nil},
		want: []any{int8(1), int8(0), nil},
	}, {
		name: "a boolean becomes its own name",
		from: dtype.Bool, to: dtype.String,
		in:   []any{true, false},
		want: []any{"true", "false"},
	}, {
		name: "an integer becomes its decimal text",
		from: dtype.Int64, to: dtype.String,
		in:   []any{int64(math.MinInt64), int64(0), nil},
		want: []any{"-9223372036854775808", "0", nil},
	}, {
		name: "a float prints the shortest text that reads back",
		from: dtype.Float64, to: dtype.String,
		in:   []any{0.1, 1e21, math.Inf(-1), math.NaN()},
		want: []any{"0.1", "1e+21", "-Inf", "NaN"},
	}, {
		name: "a float32 prints as a float32 rather than as what it widens to",
		from: dtype.Float32, to: dtype.String,
		in:   []any{float32(0.1)},
		want: []any{"0.1"},
	}, {
		name: "text becomes an integer",
		from: dtype.String, to: dtype.Int32,
		in:   []any{"-2147483648", "0", "2147483647", nil},
		want: []any{int32(-2147483648), int32(0), int32(2147483647), nil},
	}, {
		name: "text becomes an unsigned integer",
		from: dtype.String, to: dtype.Uint64,
		in:   []any{"18446744073709551615"},
		want: []any{uint64(math.MaxUint64)},
	}, {
		name: "text becomes a float",
		from: dtype.String, to: dtype.Float64,
		in:   []any{"0.1", "1e21", "-Inf", "NaN"},
		want: []any{0.1, 1e21, math.Inf(-1), math.NaN()},
	}, {
		name: "text becomes a boolean the way strconv reads one",
		from: dtype.String, to: dtype.Bool,
		in:   []any{"true", "TRUE", "1", "false", "0", nil},
		want: []any{true, true, true, false, false, nil},
	}, {
		name: "text becomes bytes for nothing",
		from: dtype.String, to: dtype.Binary,
		in:   []any{"one", ""},
		want: []any{"one", ""},
	}, {
		name: "bytes become text once they are checked",
		from: dtype.Binary, to: dtype.String,
		in:   []any{[]byte("one"), []byte{}},
		want: []any{"one", ""},
	}, {
		name: "bytes of a fixed width become text",
		from: dtype.FixedSizeBinary{ByteWidth: 3}, to: dtype.String,
		in:   []any{[]byte("abc"), nil},
		want: []any{"abc", nil},
	}, {
		name: "text of the right width becomes bytes of a fixed width",
		from: dtype.String, to: dtype.FixedSizeBinary{ByteWidth: 3},
		in:   []any{"abc"},
		want: []any{"abc"},
	}, {
		name: "a timestamp is the count it is stored as",
		from: dtype.Timestamp{Unit: dtype.Microsecond}, to: dtype.Int64,
		in:   []any{int64(1700000000000000), nil},
		want: []any{int64(1700000000000000), nil},
	}, {
		name: "and a count becomes a timestamp",
		from: dtype.Int64, to: dtype.Timestamp{Unit: dtype.Microsecond},
		in:   []any{int64(1700000000000000)},
		want: []any{int64(1700000000000000)},
	}, {
		name: "a date is a count of days",
		from: dtype.Date32, to: dtype.Int32,
		in:   []any{int32(19000), int32(-1)},
		want: []any{int32(19000), int32(-1)},
	}, {
		name: "a duration reinterpreted into a wider integer",
		from: dtype.Duration{Unit: dtype.Second}, to: dtype.Float64,
		in:   []any{int64(90)},
		want: []any{90.0},
	}, {
		name: "everything becomes nothing",
		from: dtype.Int64, to: dtype.Null,
		in:   []any{int64(1), int64(2)},
		want: []any{nil, nil},
	}, {
		name: "nothing becomes everything",
		from: dtype.Null, to: dtype.Int64,
		in:   []any{nil, nil},
		want: []any{nil, nil},
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := col(t, tt.from, tt.in)

			got, err := kernel.Cast(src, tt.to)
			if err != nil {
				t.Fatalf("Cast(%s -> %s): %v", tt.from, tt.to, err)
			}
			checkValues(t, got, tt.to, tt.want)

			// The loose cast has nothing to be loose about here, so it has to
			// agree value for value.
			got, err = kernel.TryCast(src, tt.to)
			if err != nil {
				t.Fatalf("TryCast(%s -> %s): %v", tt.from, tt.to, err)
			}
			checkValues(t, got, tt.to, tt.want)
		})
	}
}

// TestCastDoesNotFit is the other half: the values that have no answer in the
// destination type. Each one is an error from Cast and a null from TryCast, and
// the two are checked together because the whole difference between them is
// this.
func TestCastDoesNotFit(t *testing.T) {
	tests := []struct {
		name string
		from dtype.DataType
		to   dtype.DataType
		in   []any
		row  int
		msg  string
	}{{
		name: "an integer too large for the destination",
		from: dtype.Int64, to: dtype.Int8,
		in:  []any{int64(1), int64(128)},
		row: 1, msg: "128",
	}, {
		name: "an integer too small for the destination",
		from: dtype.Int64, to: dtype.Int8,
		in:  []any{int64(-129)},
		row: 0, msg: "-129",
	}, {
		name: "a negative number into an unsigned column",
		from: dtype.Int8, to: dtype.Uint8,
		in:  []any{int8(-1)},
		row: 0, msg: "-1",
	}, {
		name: "an unsigned number past the signed range",
		from: dtype.Uint64, to: dtype.Int64,
		in:  []any{uint64(math.MaxInt64) + 1},
		row: 0, msg: "9223372036854775808",
	}, {
		name: "a float past the integer range",
		from: dtype.Float64, to: dtype.Int64,
		in:  []any{math.Ldexp(1, 63)},
		row: 0, msg: "9.223372036854776e+18",
	}, {
		name: "an unsigned number too large for a narrower unsigned column",
		from: dtype.Uint64, to: dtype.Uint8,
		in:  []any{uint64(255), uint64(256)},
		row: 1, msg: "256",
	}, {
		name: "a negative float into an unsigned column",
		from: dtype.Float64, to: dtype.Uint8,
		in:  []any{-1.5},
		row: 0, msg: "-1.5",
	}, {
		name: "a NaN is not an integer",
		from: dtype.Float64, to: dtype.Int32,
		in:  []any{math.NaN()},
		row: 0, msg: "NaN",
	}, {
		name: "an infinity is not an integer either",
		from: dtype.Float64, to: dtype.Int32,
		in:  []any{math.Inf(1)},
		row: 0, msg: "+Inf",
	}, {
		name: "a float64 too large for a float32",
		from: dtype.Float64, to: dtype.Float32,
		in:  []any{1e300},
		row: 0, msg: "1e+300",
	}, {
		name: "text that is not a number",
		from: dtype.String, to: dtype.Int64,
		in:  []any{"12", "n/a"},
		row: 1, msg: `"n/a"`,
	}, {
		name: "text that is a number of the wrong shape",
		from: dtype.String, to: dtype.Int64,
		in:  []any{"3.9"},
		row: 0, msg: `"3.9"`,
	}, {
		name: "text that is a number too large for the column",
		from: dtype.String, to: dtype.Int8,
		in:  []any{"200"},
		row: 0, msg: `"200"`,
	}, {
		name: "text that is not a truth value",
		from: dtype.String, to: dtype.Bool,
		in:  []any{"yes"},
		row: 0, msg: `"yes"`,
	}, {
		name: "bytes that are not valid UTF-8",
		from: dtype.Binary, to: dtype.String,
		in:  []any{[]byte("ok"), []byte{0xff, 0xfe}},
		row: 1, msg: "not valid UTF-8",
	}, {
		name: "text of the wrong width for a fixed width column",
		from: dtype.String, to: dtype.FixedSizeBinary{ByteWidth: 3},
		in:  []any{"abc", "ab"},
		row: 1, msg: "2 bytes",
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := col(t, tt.from, tt.in)

			_, err := kernel.Cast(src, tt.to)
			if err == nil {
				t.Fatalf("Cast(%s -> %s) succeeded, want an error", tt.from, tt.to)
			}
			var ce *kernel.CastError
			if !errors.As(err, &ce) {
				t.Fatalf("Cast returned a %T, want a *kernel.CastError", err)
			}
			if ce.Row != tt.row {
				t.Errorf("the error names row %d, want %d", ce.Row, tt.row)
			}
			if !strings.Contains(err.Error(), tt.msg) {
				t.Errorf("the message is %q, want it to mention %q", err.Error(), tt.msg)
			}
			if !dtype.Equal(ce.From, tt.from) || !dtype.Equal(ce.To, tt.to) {
				t.Errorf("the error says %s to %s, want %s to %s", ce.From, ce.To, tt.from, tt.to)
			}

			got, err := kernel.TryCast(src, tt.to)
			if err != nil {
				t.Fatalf("TryCast(%s -> %s): %v", tt.from, tt.to, err)
			}
			if got.Len() != len(tt.in) {
				t.Fatalf("TryCast returned %d values, want %d", got.Len(), len(tt.in))
			}
			if !got.IsNull(tt.row) {
				t.Errorf("row %d is %v, want a null", tt.row, valueAt(t, got, tt.row))
			}
			if got.NullCount() != 1 {
				t.Errorf("TryCast made %d nulls, want 1", got.NullCount())
			}
		})
	}
}

// TestCastRangeEdges walks the boundary of every integer width from both sides,
// which is the arithmetic most likely to be off by one and the least likely to
// be noticed.
func TestCastRangeEdges(t *testing.T) {
	widths := []struct {
		to       dtype.DataType
		lo, hi   int64
		unsigned bool
	}{
		{dtype.Int8, math.MinInt8, math.MaxInt8, false},
		{dtype.Int16, math.MinInt16, math.MaxInt16, false},
		{dtype.Int32, math.MinInt32, math.MaxInt32, false},
		{dtype.Uint8, 0, math.MaxUint8, true},
		{dtype.Uint16, 0, math.MaxUint16, true},
		{dtype.Uint32, 0, math.MaxUint32, true},
	}

	for _, w := range widths {
		t.Run(w.to.String(), func(t *testing.T) {
			// Two values that fit and two that do not, from an int64 source and
			// from a float64 one, since the two take different routes in.
			fits := []any{w.lo, w.hi}
			over := []any{w.lo - 1, w.hi + 1}

			if _, err := kernel.Cast(col(t, dtype.Int64, fits), w.to); err != nil {
				t.Errorf("the ends of the range do not fit: %v", err)
			}
			for _, v := range over {
				if _, err := kernel.Cast(col(t, dtype.Int64, []any{v}), w.to); err == nil {
					t.Errorf("%v fits in a %s, it should not", v, w.to)
				}
			}

			asFloat := func(vs []any) []any {
				out := make([]any, len(vs))
				for i, v := range vs {
					out[i] = float64(v.(int64))
				}
				return out
			}
			if _, err := kernel.Cast(col(t, dtype.Float64, asFloat(fits)), w.to); err != nil {
				t.Errorf("the ends of the range do not fit from a float: %v", err)
			}
			for _, v := range asFloat(over) {
				if _, err := kernel.Cast(col(t, dtype.Float64, []any{v}), w.to); err == nil {
					t.Errorf("the float %v fits in a %s, it should not", v, w.to)
				}
			}
		})
	}
}

// TestCastInt64Edges is the same for the two widest types, which do not fit in
// the table above because their limits are the limits of the loop counter.
func TestCastInt64Edges(t *testing.T) {
	// float64 has no exact MaxInt64, so the largest float that is an int64 is
	// two to the sixty three minus one step, and the next one up is not.
	fits := math.Ldexp(1, 63) - 1024
	over := math.Ldexp(1, 63)

	if _, err := kernel.Cast(col(t, dtype.Float64, []any{fits, math.Ldexp(-1, 63)}), dtype.Int64); err != nil {
		t.Errorf("the ends of the int64 range do not fit: %v", err)
	}
	for _, v := range []any{over, -math.Ldexp(1, 63) - 4096} {
		if _, err := kernel.Cast(col(t, dtype.Float64, []any{v}), dtype.Int64); err == nil {
			t.Errorf("the float %v fits in an int64, it should not", v)
		}
	}

	if _, err := kernel.Cast(col(t, dtype.Float64, []any{math.Ldexp(1, 64) - 2048}), dtype.Uint64); err != nil {
		t.Errorf("the top of the uint64 range does not fit: %v", err)
	}
	if _, err := kernel.Cast(col(t, dtype.Float64, []any{math.Ldexp(1, 64)}), dtype.Uint64); err == nil {
		t.Error("two to the sixty four fits in a uint64, it should not")
	}
}

// TestCastKeepsTheChunks checks that a cast is a value at a time and moves no
// rows, so the caller keeps whatever chunking they had.
func TestCastKeepsTheChunks(t *testing.T) {
	src := col(t, dtype.Int32,
		[]any{int32(1), int32(2)},
		[]any{},
		[]any{int32(3), nil, int32(5)},
	)

	got, err := kernel.Cast(src, dtype.Int64)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	if got.NumChunks() != src.NumChunks() {
		t.Errorf("the result has %d chunks, want %d", got.NumChunks(), src.NumChunks())
	}
	for i := range src.NumChunks() {
		if got.Chunk(i).Len() != src.Chunk(i).Len() {
			t.Errorf("chunk %d has %d values, want %d",
				i, got.Chunk(i).Len(), src.Chunk(i).Len())
		}
	}
	checkValues(t, got, dtype.Int64, []any{int64(1), int64(2), int64(3), nil, int64(5)})
}

// TestCastCountsRowsAcrossChunks is the reason the row in a CastError is what it
// is. Chunk boundaries are an accident of how the data was read and nobody
// hunting for the bad row in a file cares where they fell.
func TestCastCountsRowsAcrossChunks(t *testing.T) {
	src := col(t, dtype.Int64,
		[]any{int64(1), int64(2)},
		[]any{int64(3), int64(400)},
	)

	_, err := kernel.Cast(src, dtype.Int8)
	var ce *kernel.CastError
	if !errors.As(err, &ce) {
		t.Fatalf("Cast returned %v, want a *kernel.CastError", err)
	}
	if ce.Row != 3 {
		t.Errorf("the error names row %d, want 3", ce.Row)
	}
}

// TestCastToTheSameType is free and returns the column it was given, since
// there is nothing to do and copying a column to prove it would be a waste.
func TestCastToTheSameType(t *testing.T) {
	src := col(t, dtype.Timestamp{Unit: dtype.Microsecond}, []any{int64(1), int64(2)})

	got, err := kernel.Cast(src, dtype.Timestamp{Unit: dtype.Microsecond})
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	if got != src {
		t.Error("a cast to the type the column already is copied it")
	}
}

// TestCastRefused is the pairs that never produce a column. The two reasons are
// different and the messages say which is which, because one of them is the
// caller's mistake and the other is a gap in this package.
func TestCastRefused(t *testing.T) {
	tests := []struct {
		name string
		from dtype.DataType
		to   dtype.DataType
		msg  string
	}{{
		name: "bytes are not a number",
		from: dtype.Binary, to: dtype.Int64,
		msg: "nothing to say to each other",
	}, {
		name: "a truth value is not a moment",
		from: dtype.Bool, to: dtype.Timestamp{Unit: dtype.Second},
		msg: "nothing to say to each other",
	}, {
		name: "a change of unit is arithmetic",
		from: dtype.Timestamp{Unit: dtype.Second}, to: dtype.Timestamp{Unit: dtype.Millisecond},
		msg: "not implemented yet",
	}, {
		name: "a date has to be formatted",
		from: dtype.Date32, to: dtype.String,
		msg: "not implemented yet",
	}, {
		name: "and parsed",
		from: dtype.String, to: dtype.Date32,
		msg: "not implemented yet",
	}, {
		name: "a decimal is a wide integer nobody here can read",
		from: dtype.Decimal128{Precision: 10, Scale: 2}, to: dtype.Int64,
		msg: "not implemented yet",
	}}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := col(t, tt.from, []any{})

			_, err := kernel.Cast(src, tt.to)
			if err == nil {
				t.Fatalf("Cast(%s -> %s) succeeded, want an error", tt.from, tt.to)
			}
			if !strings.Contains(err.Error(), tt.msg) {
				t.Errorf("the message is %q, want it to mention %q", err.Error(), tt.msg)
			}

			// A refusal is about the pair of types, so being loose about the
			// values changes nothing.
			if _, err := kernel.TryCast(src, tt.to); err == nil {
				t.Errorf("TryCast(%s -> %s) succeeded, want an error", tt.from, tt.to)
			}
		})
	}
}

// TestCastErrorUnwraps checks that the reason a value did not fit survives, so
// that a caller can tell a number that was too big from text that was never a
// number.
func TestCastErrorUnwraps(t *testing.T) {
	_, err := kernel.Cast(col(t, dtype.String, []any{"200"}), dtype.Int8)
	if !errors.Is(err, strconv.ErrRange) {
		t.Errorf("a number too big for its column is %v, want strconv.ErrRange", err)
	}

	_, err = kernel.Cast(col(t, dtype.String, []any{"n/a"}), dtype.Int8)
	if !errors.Is(err, strconv.ErrSyntax) {
		t.Errorf("text that is not a number is %v, want strconv.ErrSyntax", err)
	}

	_, err = kernel.Cast(col(t, dtype.Int64, []any{int64(200)}), dtype.Int8)
	if !errors.Is(err, strconv.ErrRange) {
		t.Errorf("a number too big for its column is %v, want strconv.ErrRange", err)
	}
}

func TestCastPanics(t *testing.T) {
	c := col(t, dtype.Int64, []any{int64(1)})

	tests := []struct {
		name string
		call func()
	}{
		{"a nil column", func() { _, _ = kernel.Cast(nil, dtype.Int64) }},
		{"a nil type", func() { _, _ = kernel.Cast(c, nil) }},
		{"a nil column in the loose cast", func() { _, _ = kernel.TryCast(nil, dtype.Int64) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Error("it did not panic")
				}
			}()
			tt.call()
		})
	}
}

// checkValues compares a column against the values it should hold, where nil
// means missing.
func checkValues(t *testing.T, got *array.Chunked, want dtype.DataType, values []any) {
	t.Helper()

	if !dtype.Equal(got.DType(), want) {
		t.Errorf("the result is a %s column, want %s", got.DType(), want)
	}
	if got.Len() != len(values) {
		t.Fatalf("the result has %d values, want %d", got.Len(), len(values))
	}

	nulls := 0
	for i, w := range values {
		if w == nil {
			nulls++
		}
		have := valueAt(t, got, i)
		if !sameValue(have, w) {
			t.Errorf("value %d is %#v, want %#v", i, have, w)
		}
	}
	if got.NullCount() != nulls {
		t.Errorf("the result counts %d nulls, want %d", got.NullCount(), nulls)
	}
}

// sameValue is equality that says a NaN is the NaN that was asked for, which is
// what a test of a cast wants and is not what == says.
func sameValue(a, b any) bool {
	switch x := a.(type) {
	case float64:
		y, ok := b.(float64)
		return ok && (x == y || (math.IsNaN(x) && math.IsNaN(y)))
	case float32:
		y, ok := b.(float32)
		return ok && (x == y || (math.IsNaN(float64(x)) && math.IsNaN(float64(y))))
	}
	return a == b
}

// storedTypes is every type this package can read and write as a number. The
// units are the ones that are valid for their type and are otherwise arbitrary,
// since a cast to or from a number does not look at them.
var storedTypes = []dtype.DataType{
	dtype.Int8, dtype.Int16, dtype.Int32, dtype.Int64,
	dtype.Uint8, dtype.Uint16, dtype.Uint32, dtype.Uint64,
	dtype.Float32, dtype.Float64,
	dtype.Date32, dtype.Date64,
	dtype.Time32{Unit: dtype.Millisecond}, dtype.Time64{Unit: dtype.Nanosecond},
	dtype.Timestamp{Unit: dtype.Microsecond}, dtype.Duration{Unit: dtype.Second},
}

// TestCastEveryPair casts the value one between every pair of types that this
// package claims to handle, and checks it arrives as one.
//
// The table above this is chosen for what the answer says, which means it says
// nothing about the pairs where the answer is boring. This is the other half:
// no value is interesting and every arm of every switch is reached, so a type
// added to one of them and forgotten in another shows up here rather than in
// somebody's file.
func TestCastEveryPair(t *testing.T) {
	for _, from := range storedTypes {
		for _, to := range storedTypes {
			if dtype.Equal(from, to) {
				continue
			}
			// A temporal on both sides is a change of unit, which is
			// arithmetic and is not in this package yet.
			if dtype.IsTemporal(from) && dtype.IsTemporal(to) {
				continue
			}

			t.Run(from.String()+" to "+to.String(), func(t *testing.T) {
				got, err := kernel.Cast(col(t, from, []any{oneOf(t, from)}), to)
				if err != nil {
					t.Fatalf("Cast: %v", err)
				}
				checkValues(t, got, to, []any{oneOf(t, to)})
			})
		}
	}

	// The two destinations that are not numbers, and the source that is not one
	// either. A temporal type is left out of all three because formatting a
	// date and parsing one are arithmetic as well.
	for _, from := range storedTypes {
		if dtype.IsTemporal(from) {
			continue
		}

		t.Run(from.String()+" to bool", func(t *testing.T) {
			got, err := kernel.Cast(col(t, from, []any{oneOf(t, from)}), dtype.Bool)
			if err != nil {
				t.Fatalf("Cast: %v", err)
			}
			checkValues(t, got, dtype.Bool, []any{true})
		})

		t.Run(from.String()+" to text", func(t *testing.T) {
			got, err := kernel.Cast(col(t, from, []any{oneOf(t, from)}), dtype.String)
			if err != nil {
				t.Fatalf("Cast: %v", err)
			}
			checkValues(t, got, dtype.String, []any{"1"})
		})

		t.Run("text to "+from.String(), func(t *testing.T) {
			got, err := kernel.Cast(col(t, dtype.String, []any{"1"}), from)
			if err != nil {
				t.Fatalf("Cast: %v", err)
			}
			checkValues(t, got, from, []any{oneOf(t, from)})
		})
	}
}

// TestCastToATypeNothingCanBuild is the one way a cast fails before it reads a
// value that is not about the pair of types being meaningless. A fixed width of
// less than nothing is a type that passes every check this package makes and
// that no column can be built for.
func TestCastToATypeNothingCanBuild(t *testing.T) {
	_, err := kernel.Cast(col(t, dtype.String, []any{"ab"}), dtype.FixedSizeBinary{ByteWidth: -1})
	if err == nil {
		t.Fatal("a cast to a column of minus one bytes succeeded")
	}
	if !strings.Contains(err.Error(), "fixed_size_binary[-1]") {
		t.Errorf("the message is %q, want it to name the type", err.Error())
	}
}

// oneOf returns the value one, in whatever Go type the given column stores.
func oneOf(t *testing.T, dt dtype.DataType) any {
	t.Helper()

	switch dt.Kind() {
	case dtype.Int8Kind:
		return int8(1)
	case dtype.Int16Kind:
		return int16(1)
	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		return int32(1)
	case dtype.Int64Kind, dtype.Date64Kind, dtype.Time64Kind,
		dtype.TimestampKind, dtype.DurationKind:
		return int64(1)
	case dtype.Uint8Kind:
		return uint8(1)
	case dtype.Uint16Kind:
		return uint16(1)
	case dtype.Uint32Kind:
		return uint32(1)
	case dtype.Uint64Kind:
		return uint64(1)
	case dtype.Float32Kind:
		return float32(1)
	case dtype.Float64Kind:
		return float64(1)
	default:
		t.Fatalf("no value one for a %s column", dt)
		return nil
	}
}

// TestFits is the range check on its own, which is what a value written in a
// query is put to before the query runs.
func TestFits(t *testing.T) {
	cases := []struct {
		name string
		dt   dtype.DataType
		v    any
		want string // the part of the message that matters, or empty for a fit
	}{
		{"an int64 in an int64", dtype.Int64, int64(300), ""},
		{"an int64 in an int8", dtype.Int8, int64(300), "300 does not fit in int8, which holds -128 to 127"},
		{"the top of an int8", dtype.Int8, int64(127), ""},
		{"one past the top of an int8", dtype.Int8, int64(128), "does not fit in int8"},
		{"the bottom of an int8", dtype.Int8, int64(-128), ""},
		{"one past the bottom of an int8", dtype.Int8, int64(-129), "does not fit in int8"},
		{"a negative in a uint32", dtype.Uint32, -1, "-1 does not fit in uint32, which holds 0 to 4294967295"},
		{"the top of a uint64", dtype.Uint64, uint64(math.MaxUint64), ""},
		{"a uint64 above the int64 range", dtype.Int64, uint64(math.MaxInt64) + 1,
			"9223372036854775808 does not fit in int64"},
		{"a float in a float32", dtype.Float32, 1.5, ""},
		{"a float too big for a float32", dtype.Float32, 1e300, "1e+300 does not fit in float32"},
		{"a float in a float64", dtype.Float64, 1e300, ""},
		{"an int in a float32", dtype.Float32, int64(1) << 60, ""},

		// Two to the sixty three is the first float past the int64 range and it
		// is also what float64(math.MaxInt64) rounds to, so a bound written as
		// the top of the range instead of the power of two past it lets this
		// one through. The same holds one width up for the unsigned side.
		{"the largest float that is an int64", dtype.Int64, math.Ldexp(1, 63) - 1024, ""},
		{"the first float past the int64 range", dtype.Int64, math.Ldexp(1, 63),
			"9.223372036854776e+18 does not fit in int64"},
		{"the largest float that is a uint64", dtype.Uint64, math.Ldexp(1, 64) - 2048, ""},
		{"the first float past the uint64 range", dtype.Uint64, math.Ldexp(1, 64),
			"does not fit in uint64"},

		// The pairs this check has no answer for, which are the ones another
		// layer has already decided about.
		{"a string in a string", dtype.String, "AAPL", ""},
		{"a string in an int8", dtype.Int8, "300", ""},
		{"nothing in an int8", dtype.Int8, nil, ""},
		{"a bool in a bool", dtype.Bool, true, ""},
		{"an int64 in a timestamp", dtype.Timestamp{Unit: dtype.Second}, int64(1), ""},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := kernel.Fits(c.dt, c.v)
			if c.want == "" {
				if err != nil {
					t.Fatalf("Fits(%s, %v) = %v, want it to fit", c.dt, c.v, err)
				}
				return
			}
			if err == nil {
				t.Fatalf("Fits(%s, %v) = nil, want %q", c.dt, c.v, c.want)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("Fits(%s, %v) = %v, want it to say %q", c.dt, c.v, err, c.want)
			}
		})
	}
}

// TestFitsAgreesWithCast is the reason the check is here rather than in the
// package that asks it. A value the check accepts has to be one the cast
// accepts, or a query that passes at plan time fails on its first row, which is
// the whole thing this is meant to stop. The two share the arithmetic, and this
// is what says they still do.
func TestFitsAgreesWithCast(t *testing.T) {
	types := []dtype.DataType{
		dtype.Int8, dtype.Int16, dtype.Int32, dtype.Int64,
		dtype.Uint8, dtype.Uint16, dtype.Uint32, dtype.Uint64,
		dtype.Float32, dtype.Float64,
	}
	values := []any{
		int64(0), int64(1), int64(-1), int64(127), int64(128), int64(-128), int64(-129),
		int64(255), int64(256), int64(65535), int64(65536), int64(math.MaxInt32),
		int64(math.MinInt32), int64(math.MaxInt64), int64(math.MinInt64),
		uint64(math.MaxUint64), uint64(math.MaxInt64) + 1,
		1.0, -1.0, 1e300, -1e300, float32(1.5),
		math.Ldexp(1, 63), math.Ldexp(1, 63) - 1024,
		math.Ldexp(1, 64), math.Ldexp(1, 64) - 2048,
	}

	for _, dt := range types {
		for _, v := range values {
			from := naturalType(t, v)
			if !dtype.CanCast(from, dt) {
				continue
			}
			_, err := kernel.Cast(col(t, from, []any{v}), dt)
			if fits := kernel.Fits(dt, v); (fits == nil) != (err == nil) {
				t.Errorf("Fits(%s, %v) = %v and the cast said %v, they have to agree",
					dt, v, fits, err)
			}
		}
	}
}

// naturalType is the type a value has on its own, which is the column the cast
// half of the test above has to start from.
func naturalType(t *testing.T, v any) dtype.DataType {
	t.Helper()
	switch v.(type) {
	case int64:
		return dtype.Int64
	case uint64:
		return dtype.Uint64
	case float32:
		return dtype.Float32
	case float64:
		return dtype.Float64
	default:
		t.Fatalf("no natural type for a %T", v)
		return nil
	}
}
