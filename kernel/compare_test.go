package kernel_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// answers returns the values of a boolean column as a list, with a nil where a
// value is missing, which is the shape the tests write their expectations in.
func answers(t *testing.T, c *array.Chunked) []any {
	t.Helper()

	if c.DType().Kind() != dtype.BoolKind {
		t.Fatalf("the result is a %s column, want bool", c.DType())
	}
	out := make([]any, c.Len())
	for i := range out {
		out[i] = valueAt(t, c, i)
	}
	return out
}

// same reports whether two lists of answers hold the same values.
func same(a, b []any) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestCompare(t *testing.T) {
	tests := []struct {
		name string
		a, b *array.Chunked
		op   kernel.CompareOp
		want []any
	}{
		{
			name: "equal",
			a:    col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)}),
			b:    col(t, dtype.Int64, []any{int64(3), int64(2), int64(1)}),
			op:   kernel.OpEq,
			want: []any{false, true, false},
		},
		{
			name: "not equal",
			a:    col(t, dtype.Int64, []any{int64(1), int64(2)}),
			b:    col(t, dtype.Int64, []any{int64(1), int64(3)}),
			op:   kernel.OpNe,
			want: []any{false, true},
		},
		{
			name: "less than",
			a:    col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)}),
			b:    col(t, dtype.Int64, []any{int64(2), int64(2), int64(2)}),
			op:   kernel.OpLt,
			want: []any{true, false, false},
		},
		{
			name: "less or equal",
			a:    col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)}),
			b:    col(t, dtype.Int64, []any{int64(2), int64(2), int64(2)}),
			op:   kernel.OpLe,
			want: []any{true, true, false},
		},
		{
			name: "greater than",
			a:    col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)}),
			b:    col(t, dtype.Int64, []any{int64(2), int64(2), int64(2)}),
			op:   kernel.OpGt,
			want: []any{false, false, true},
		},
		{
			name: "greater or equal",
			a:    col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)}),
			b:    col(t, dtype.Int64, []any{int64(2), int64(2), int64(2)}),
			op:   kernel.OpGe,
			want: []any{false, true, true},
		},
		{
			name: "a literal on the right",
			a:    col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)}),
			b:    col(t, dtype.Int64, []any{int64(2)}),
			op:   kernel.OpGt,
			want: []any{false, false, true},
		},
		{
			name: "a literal on the left",
			a:    col(t, dtype.Int64, []any{int64(2)}),
			b:    col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)}),
			op:   kernel.OpGt,
			want: []any{true, false, false},
		},
		{
			name: "a null on either side gives a null",
			a:    col(t, dtype.Int64, []any{int64(1), nil, int64(3)}),
			b:    col(t, dtype.Int64, []any{int64(1), int64(2), nil}),
			op:   kernel.OpEq,
			want: []any{true, nil, nil},
		},
		{
			name: "two nulls are not equal, they are unknown",
			a:    col(t, dtype.Int64, []any{nil}),
			b:    col(t, dtype.Int64, []any{nil}),
			op:   kernel.OpEq,
			want: []any{nil},
		},
		{
			name: "a null literal makes every answer unknown",
			a:    col(t, dtype.Int64, []any{int64(1), int64(2)}),
			b:    col(t, dtype.Int64, []any{nil}),
			op:   kernel.OpLt,
			want: []any{nil, nil},
		},
		{
			name: "chunks that do not line up",
			a:    col(t, dtype.Int64, []any{int64(1)}, []any{int64(2), int64(3)}, []any{int64(4)}),
			b:    col(t, dtype.Int64, []any{int64(1), int64(1)}, []any{int64(9), int64(9)}),
			op:   kernel.OpLt,
			want: []any{false, false, true, true},
		},
		{
			name: "an empty chunk in the middle",
			a:    col(t, dtype.Int64, []any{int64(1)}, nil, []any{int64(2)}),
			b:    col(t, dtype.Int64, []any{int64(2), int64(1)}),
			op:   kernel.OpLt,
			want: []any{true, false},
		},
		{
			name: "a literal that is not in the first chunk",
			a:    col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)}),
			b:    col(t, dtype.Int64, nil, []any{int64(2)}),
			op:   kernel.OpGe,
			want: []any{false, true, true},
		},
		{
			name: "nothing at all",
			a:    col(t, dtype.Int64, nil),
			b:    col(t, dtype.Int64, nil),
			op:   kernel.OpEq,
			want: []any{},
		},
		{
			name: "strings order the way Go orders them",
			a:    col(t, dtype.String, []any{"apple", "banana", "Apple"}),
			b:    col(t, dtype.String, []any{"banana", "apple", "apple"}),
			op:   kernel.OpLt,
			want: []any{true, false, true},
		},
		{
			name: "false is below true",
			a:    col(t, dtype.Bool, []any{false, true, true, false}),
			b:    col(t, dtype.Bool, []any{true, false, true, false}),
			op:   kernel.OpLt,
			want: []any{true, false, false, false},
		},
		{
			name: "unsigned values do not go negative",
			a:    col(t, dtype.Uint8, []any{uint8(0), uint8(255)}),
			b:    col(t, dtype.Uint8, []any{uint8(255), uint8(0)}),
			op:   kernel.OpLt,
			want: []any{true, false},
		},
		{
			name: "a timestamp is compared as the count it is stored as",
			a: col(t, dtype.Timestamp{Unit: dtype.Microsecond},
				[]any{int64(1787650200000000), int64(1787650260000000)}),
			b: col(t, dtype.Timestamp{Unit: dtype.Microsecond},
				[]any{int64(1787650260000000)}),
			op:   kernel.OpLt,
			want: []any{true, false},
		},
		{
			name: "a column of nothing knows nothing",
			a:    col(t, dtype.Null, []any{nil, nil}),
			b:    col(t, dtype.Int64, []any{int64(1)}),
			op:   kernel.OpEq,
			want: []any{nil, nil},
		},
		{
			name: "two columns of nothing have nothing to compare",
			a:    col(t, dtype.Null, []any{nil, nil, nil}),
			b:    col(t, dtype.Null, []any{nil, nil, nil}),
			op:   kernel.OpLt,
			want: []any{nil, nil, nil},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := kernel.Compare(tt.a, tt.b, tt.op)
			if err != nil {
				t.Fatalf("Compare: %v", err)
			}
			if have := answers(t, got); !same(have, tt.want) {
				t.Errorf("a %s b is %v, want %v", tt.op, have, tt.want)
			}
		})
	}
}

// TestCompareEveryType walks the comparison of every type there is one for.
// Each case is a low value against a high one, the high one against the low
// one, and a value against itself, so that all three answers a comparison can
// give are asked for from every branch.
func TestCompareEveryType(t *testing.T) {
	tests := []struct {
		name   string
		dt     dtype.DataType
		lo, hi any
	}{
		{"bool", dtype.Bool, false, true},
		{"int8", dtype.Int8, int8(-1), int8(1)},
		{"int16", dtype.Int16, int16(-300), int16(300)},
		{"int32", dtype.Int32, int32(-70000), int32(70000)},
		{"int64", dtype.Int64, int64(-5), int64(5)},
		{"uint8", dtype.Uint8, uint8(1), uint8(200)},
		{"uint16", dtype.Uint16, uint16(1), uint16(60000)},
		{"uint32", dtype.Uint32, uint32(1), uint32(4000000000)},
		{"uint64", dtype.Uint64, uint64(1), uint64(1) << 63},
		{"float32", dtype.Float32, float32(-1.5), float32(1.5)},
		{"float64", dtype.Float64, -1.5, 1.5},
		{"string", dtype.String, "a", "b"},
		{"binary", dtype.Binary, []byte{1}, []byte{2}},
		{"fixed size binary", dtype.FixedSizeBinary{ByteWidth: 2}, []byte{0, 1}, []byte{0, 2}},
		{"date32", dtype.Date32, int32(1), int32(2)},
		{"date64", dtype.Date64, int64(1), int64(2)},
		{"time32", dtype.Time32{Unit: dtype.Second}, int32(1), int32(2)},
		{"time64", dtype.Time64{Unit: dtype.Nanosecond}, int64(1), int64(2)},
		{"timestamp", dtype.Timestamp{Unit: dtype.Microsecond}, int64(1), int64(2)},
		{"duration", dtype.Duration{Unit: dtype.Second}, int64(1), int64(2)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := col(t, tt.dt, []any{tt.lo, tt.hi, tt.hi})
			b := col(t, tt.dt, []any{tt.hi, tt.lo, tt.hi})

			got, err := kernel.Compare(a, b, kernel.OpLe)
			if err != nil {
				t.Fatalf("Compare: %v", err)
			}
			if have, want := answers(t, got), []any{true, false, true}; !same(have, want) {
				t.Errorf("low, high and equal compare as %v, want %v", have, want)
			}
		})
	}
}

// TestCompareNaN is the IEEE rule, which is the one Go itself follows. It is
// worth its own test because a sort has to answer the same question differently
// and it would be easy to reuse the wrong answer.
func TestCompareNaN(t *testing.T) {
	nan := math.NaN()
	a := col(t, dtype.Float64, []any{nan, nan, 1.0})
	b := col(t, dtype.Float64, []any{nan, 1.0, nan})

	tests := []struct {
		op   kernel.CompareOp
		want []any
	}{
		{kernel.OpEq, []any{false, false, false}},
		{kernel.OpNe, []any{true, true, true}},
		{kernel.OpLt, []any{false, false, false}},
		{kernel.OpLe, []any{false, false, false}},
		{kernel.OpGt, []any{false, false, false}},
		{kernel.OpGe, []any{false, false, false}},
	}

	for _, tt := range tests {
		t.Run(tt.op.String(), func(t *testing.T) {
			got, err := kernel.Compare(a, b, tt.op)
			if err != nil {
				t.Fatalf("Compare: %v", err)
			}
			if have := answers(t, got); !same(have, tt.want) {
				t.Errorf("NaN %s: %v, want %v", tt.op, have, tt.want)
			}
		})
	}
}

// TestCompareFloat32NaN checks that the narrower float goes through the same
// branch, since the two are separate instantiations of the same code and only
// one of them would be covered otherwise.
func TestCompareFloat32NaN(t *testing.T) {
	nan := float32(math.NaN())
	a := col(t, dtype.Float32, []any{nan, float32(1)})
	b := col(t, dtype.Float32, []any{nan, float32(2)})

	got, err := kernel.Compare(a, b, kernel.OpLt)
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}
	if have, want := answers(t, got), []any{false, true}; !same(have, want) {
		t.Errorf("float32 comparison is %v, want %v", have, want)
	}
}

func TestCompareTypesWithNothingInCommon(t *testing.T) {
	a := col(t, dtype.Int64, []any{int64(1)})
	b := col(t, dtype.Float64, []any{1.0})

	_, err := kernel.Compare(a, b, kernel.OpEq)
	if err == nil {
		t.Fatal("comparing an int64 column with a float64 column was allowed")
	}
	if !strings.Contains(err.Error(), "cast one side") {
		t.Errorf("the error is %q, which does not say what to do about it", err)
	}
}

func TestCompareTypeWithNoOrderYet(t *testing.T) {
	dt := dtype.Decimal128{Precision: 10, Scale: 2}
	c := col(t, dt, []any{[]byte(strings.Repeat("\x00", 16))})

	_, err := kernel.Compare(c, c, kernel.OpEq)
	if err == nil {
		t.Fatal("comparing decimals was allowed")
	}
	if !strings.Contains(err.Error(), "yet") {
		t.Errorf("the error is %q, which does not say that this is unwritten", err)
	}
}

func TestCompareUnalignedLengths(t *testing.T) {
	a := col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)})
	b := col(t, dtype.Int64, []any{int64(1), int64(2)})

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("comparing columns of different lengths did not panic")
		}
		if msg, ok := r.(string); !ok || !strings.Contains(msg, "3 values") {
			t.Errorf("the panic is %v, which does not say what the lengths were", r)
		}
	}()
	_, _ = kernel.Compare(a, b, kernel.OpEq)
}

func TestCompareNilColumn(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("comparing a nil column did not panic")
		}
	}()
	_, _ = kernel.Compare(nil, nil, kernel.OpEq)
}

func TestCompareUnknownOperator(t *testing.T) {
	c := col(t, dtype.Int64, []any{int64(1)})

	defer func() {
		if recover() == nil {
			t.Fatal("comparing with an operator that does not exist did not panic")
		}
	}()
	_, _ = kernel.Compare(c, c, kernel.CompareOp(200))
}

func TestCompareOpString(t *testing.T) {
	want := []string{"==", "!=", "<", "<=", ">", ">="}
	for i, s := range want {
		if got := kernel.CompareOp(i).String(); got != s {
			t.Errorf("operator %d prints as %q, want %q", i, got, s)
		}
	}
	if got := kernel.CompareOp(200).String(); got != "CompareOp(200)" {
		t.Errorf("an unknown operator prints as %q", got)
	}
}
