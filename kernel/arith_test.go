package kernel_test

import (
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// results returns the values of a column as a list, with a nil where a value is
// missing.
func results(t *testing.T, c *array.Chunked) []any {
	t.Helper()

	out := make([]any, c.Len())
	for i := range out {
		out[i] = valueAt(t, c, i)
	}
	return out
}

func TestArith(t *testing.T) {
	tests := []struct {
		name string
		dt   dtype.DataType
		a, b []any
		op   kernel.ArithOp
		want []any
	}{
		{
			name: "add",
			dt:   dtype.Int64,
			a:    []any{int64(1), int64(2), int64(3)},
			b:    []any{int64(10), int64(20), int64(30)},
			op:   kernel.OpAdd,
			want: []any{int64(11), int64(22), int64(33)},
		},
		{
			name: "subtract",
			dt:   dtype.Int64,
			a:    []any{int64(1), int64(2)},
			b:    []any{int64(10), int64(1)},
			op:   kernel.OpSub,
			want: []any{int64(-9), int64(1)},
		},
		{
			name: "multiply",
			dt:   dtype.Int64,
			a:    []any{int64(3), int64(-3)},
			b:    []any{int64(4), int64(4)},
			op:   kernel.OpMul,
			want: []any{int64(12), int64(-12)},
		},
		{
			name: "integer division truncates toward zero",
			dt:   dtype.Int64,
			a:    []any{int64(7), int64(-7), int64(6)},
			b:    []any{int64(2), int64(2), int64(2)},
			op:   kernel.OpDiv,
			want: []any{int64(3), int64(-3), int64(3)},
		},
		{
			name: "remainder",
			dt:   dtype.Int64,
			a:    []any{int64(7), int64(-7)},
			b:    []any{int64(3), int64(3)},
			op:   kernel.OpMod,
			want: []any{int64(1), int64(-1)},
		},
		{
			name: "a literal on the right",
			dt:   dtype.Int64,
			a:    []any{int64(1), int64(2), int64(3)},
			b:    []any{int64(10)},
			op:   kernel.OpMul,
			want: []any{int64(10), int64(20), int64(30)},
		},
		{
			name: "a literal on the left",
			dt:   dtype.Int64,
			a:    []any{int64(100)},
			b:    []any{int64(1), int64(2), int64(4)},
			op:   kernel.OpDiv,
			want: []any{int64(100), int64(50), int64(25)},
		},
		{
			name: "a missing value has nothing to add",
			dt:   dtype.Int64,
			a:    []any{int64(1), nil, int64(3)},
			b:    []any{int64(1), int64(2), nil},
			op:   kernel.OpAdd,
			want: []any{int64(2), nil, nil},
		},
		{
			name: "an addition that wraps, the way Go wraps",
			dt:   dtype.Int8,
			a:    []any{int8(127)},
			b:    []any{int8(1)},
			op:   kernel.OpAdd,
			want: []any{int8(-128)},
		},
		{
			name: "unsigned subtraction wraps too",
			dt:   dtype.Uint8,
			a:    []any{uint8(0)},
			b:    []any{uint8(1)},
			op:   kernel.OpSub,
			want: []any{uint8(255)},
		},
		{
			name: "int16",
			dt:   dtype.Int16,
			a:    []any{int16(300)},
			b:    []any{int16(3)},
			op:   kernel.OpDiv,
			want: []any{int16(100)},
		},
		{
			name: "int32",
			dt:   dtype.Int32,
			a:    []any{int32(2)},
			b:    []any{int32(3)},
			op:   kernel.OpMul,
			want: []any{int32(6)},
		},
		{
			name: "uint16",
			dt:   dtype.Uint16,
			a:    []any{uint16(7)},
			b:    []any{uint16(2)},
			op:   kernel.OpMod,
			want: []any{uint16(1)},
		},
		{
			name: "uint32",
			dt:   dtype.Uint32,
			a:    []any{uint32(7)},
			b:    []any{uint32(2)},
			op:   kernel.OpAdd,
			want: []any{uint32(9)},
		},
		{
			name: "uint64",
			dt:   dtype.Uint64,
			a:    []any{uint64(7)},
			b:    []any{uint64(2)},
			op:   kernel.OpSub,
			want: []any{uint64(5)},
		},
		{
			name: "float division keeps the fraction",
			dt:   dtype.Float64,
			a:    []any{7.0, 1.0},
			b:    []any{2.0, 4.0},
			op:   kernel.OpDiv,
			want: []any{3.5, 0.25},
		},
		{
			name: "float remainder",
			dt:   dtype.Float64,
			a:    []any{7.5},
			b:    []any{2.0},
			op:   kernel.OpMod,
			want: []any{1.5},
		},
		{
			name: "float subtraction",
			dt:   dtype.Float64,
			a:    []any{1.5, 0.5},
			b:    []any{0.25, 1.0},
			op:   kernel.OpSub,
			want: []any{1.25, -0.5},
		},
		{
			name: "float multiplication",
			dt:   dtype.Float64,
			a:    []any{1.5},
			b:    []any{2.0},
			op:   kernel.OpMul,
			want: []any{3.0},
		},
		{
			name: "float32",
			dt:   dtype.Float32,
			a:    []any{float32(1.5)},
			b:    []any{float32(2.5)},
			op:   kernel.OpAdd,
			want: []any{float32(4)},
		},
		{
			name: "float32 division",
			dt:   dtype.Float32,
			a:    []any{float32(7)},
			b:    []any{float32(2)},
			op:   kernel.OpDiv,
			want: []any{float32(3.5)},
		},
		{
			name: "float32 remainder",
			dt:   dtype.Float32,
			a:    []any{float32(7.5)},
			b:    []any{float32(2)},
			op:   kernel.OpMod,
			want: []any{float32(1.5)},
		},
		{
			name: "chunks that do not line up",
			dt:   dtype.Int64,
			a:    []any{int64(1), int64(2), int64(3), int64(4)},
			b:    []any{int64(10), int64(10), int64(10), int64(10)},
			op:   kernel.OpAdd,
			want: []any{int64(11), int64(12), int64(13), int64(14)},
		},
		{
			name: "a column of nothing stays nothing",
			dt:   dtype.Null,
			a:    []any{nil, nil},
			b:    []any{nil, nil},
			op:   kernel.OpAdd,
			want: []any{nil, nil},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, b := col(t, tt.dt, tt.a), col(t, tt.dt, tt.b)
			got, err := kernel.Arith(a, b, tt.op)
			if err != nil {
				t.Fatalf("Arith: %v", err)
			}
			if !dtype.Equal(got.DType(), tt.dt) {
				t.Errorf("the result is a %s column, want %s", got.DType(), tt.dt)
			}
			if have := results(t, got); !same(have, tt.want) {
				t.Errorf("a %s b is %v, want %v", tt.op, have, tt.want)
			}
		})
	}
}

// TestArithAcrossChunks is the case the cursor exists for, where the two sides
// are chunked differently and neither boundary lines up with the other.
func TestArithAcrossChunks(t *testing.T) {
	a := col(t, dtype.Int64, []any{int64(1)}, []any{int64(2), int64(3)}, []any{int64(4), int64(5)})
	b := col(t, dtype.Int64, []any{int64(10), int64(20), int64(30)}, nil, []any{int64(40), int64(50)})

	got, err := kernel.Arith(a, b, kernel.OpAdd)
	if err != nil {
		t.Fatalf("Arith: %v", err)
	}
	want := []any{int64(11), int64(22), int64(33), int64(44), int64(55)}
	if have := results(t, got); !same(have, want) {
		t.Errorf("the sum is %v, want %v", have, want)
	}
}

func TestArithDivideByZero(t *testing.T) {
	for _, op := range []kernel.ArithOp{kernel.OpDiv, kernel.OpMod} {
		t.Run(op.String(), func(t *testing.T) {
			a := col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)})
			b := col(t, dtype.Int64, []any{int64(1), int64(0), int64(1)})

			_, err := kernel.Arith(a, b, op)
			if err == nil {
				t.Fatal("dividing by zero was allowed")
			}
			if !errors.Is(err, kernel.ErrDivideByZero) {
				t.Errorf("the error is %v, which is not the sentinel", err)
			}
			if !strings.Contains(err.Error(), "row 1") {
				t.Errorf("the error is %q, which does not say which row it was", err)
			}
		})
	}
}

// TestArithFloatDivideByZero is the other half of that rule. A float64 has a
// value for one divided by zero and it is not an error to ask for it.
func TestArithFloatDivideByZero(t *testing.T) {
	a := col(t, dtype.Float64, []any{1.0, -1.0, 0.0})
	b := col(t, dtype.Float64, []any{0.0, 0.0, 0.0})

	got, err := kernel.Arith(a, b, kernel.OpDiv)
	if err != nil {
		t.Fatalf("Arith: %v", err)
	}
	if v := got.Value[float64](0); !math.IsInf(v, 1) {
		t.Errorf("one over zero is %v, want +Inf", v)
	}
	if v := got.Value[float64](1); !math.IsInf(v, -1) {
		t.Errorf("minus one over zero is %v, want -Inf", v)
	}
	if v := got.Value[float64](2); !math.IsNaN(v) {
		t.Errorf("zero over zero is %v, want NaN", v)
	}
}

// TestArithSkipsTheZeroUnderANull checks that a row nobody is adding is not
// checked for dividing by zero either. The value under a null is usually zero
// and reading it would fail the whole column for a row that has no value.
func TestArithSkipsTheZeroUnderANull(t *testing.T) {
	a := col(t, dtype.Int64, []any{int64(1), int64(2)})
	b := col(t, dtype.Int64, []any{int64(1), nil})

	got, err := kernel.Arith(a, b, kernel.OpDiv)
	if err != nil {
		t.Fatalf("Arith: %v", err)
	}
	if have, want := results(t, got), []any{int64(1), nil}; !same(have, want) {
		t.Errorf("the quotient is %v, want %v", have, want)
	}
}

func TestArithTypesWithNothingInCommon(t *testing.T) {
	a := col(t, dtype.Int64, []any{int64(1)})
	b := col(t, dtype.Float64, []any{1.0})

	_, err := kernel.Arith(a, b, kernel.OpAdd)
	if err == nil {
		t.Fatal("adding an int64 column to a float64 column was allowed")
	}
	if !strings.Contains(err.Error(), "cast one side") {
		t.Errorf("the error is %q, which does not say what to do about it", err)
	}
}

func TestArithTypeWithNoArithmeticYet(t *testing.T) {
	c := col(t, dtype.String, []any{"a"})

	_, err := kernel.Arith(c, c, kernel.OpAdd)
	if err == nil {
		t.Fatal("adding two string columns was allowed")
	}
	if !strings.Contains(err.Error(), "yet") {
		t.Errorf("the error is %q, which does not say that this is unwritten", err)
	}
}

func TestArithUnalignedLengths(t *testing.T) {
	a := col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)})
	b := col(t, dtype.Int64, []any{int64(1), int64(2)})

	defer func() {
		if recover() == nil {
			t.Fatal("adding columns of different lengths did not panic")
		}
	}()
	_, _ = kernel.Arith(a, b, kernel.OpAdd)
}

func TestArithNilColumn(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("adding a nil column did not panic")
		}
	}()
	_, _ = kernel.Arith(nil, nil, kernel.OpAdd)
}

func TestArithUnknownOperator(t *testing.T) {
	c := col(t, dtype.Int64, []any{int64(1)})

	defer func() {
		if recover() == nil {
			t.Fatal("an operator that does not exist did not panic")
		}
	}()
	_, _ = kernel.Arith(c, c, kernel.ArithOp(200))
}

func TestArithOpString(t *testing.T) {
	want := []string{"+", "-", "*", "/", "%"}
	for i, s := range want {
		if got := kernel.ArithOp(i).String(); got != s {
			t.Errorf("operator %d prints as %q, want %q", i, got, s)
		}
	}
	if got := kernel.ArithOp(200).String(); got != "ArithOp(200)" {
		t.Errorf("an unknown operator prints as %q", got)
	}
}
