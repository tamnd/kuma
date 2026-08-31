package plan_test

// The tests for the column a literal becomes. They are here rather than in the
// kuma package because this is where the rule about what a literal is worth
// against a column lives, and the engine and the optimizer both ask it.

import (
	"strings"
	"testing"
	"time"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/plan"
)

func TestLiteralColumnWithNoColumnToGoBy(t *testing.T) {
	cases := []struct {
		value any
		want  dtype.DataType
	}{
		{true, dtype.Bool},
		{42, dtype.Int64},
		{int8(42), dtype.Int8},
		{int32(42), dtype.Int32},
		{uint(42), dtype.Uint64},
		{uint16(42), dtype.Uint16},
		{float32(1.5), dtype.Float32},
		{1.5, dtype.Float64},
		{"ES", dtype.String},
		{[]byte("ES"), dtype.Binary},
	}

	for _, c := range cases {
		t.Run(c.want.String(), func(t *testing.T) {
			got, err := plan.LiteralColumn(c.value, nil)
			if err != nil {
				t.Fatalf("LiteralColumn(%#v, nil): %v", c.value, err)
			}
			if !dtype.Equal(got.DType(), c.want) {
				t.Errorf("%#v became a %s column, want %s", c.value, got.DType(), c.want)
			}
			if got.Len() != 1 {
				t.Errorf("%#v became %d values, want one", c.value, got.Len())
			}
		})
	}
}

func TestLiteralColumnTakesTheColumnsType(t *testing.T) {
	cases := []struct {
		name  string
		value any
		hint  dtype.DataType
		want  dtype.DataType
	}{
		{
			name:  "an integer against a uint32 column stays a uint32 comparison",
			value: 0,
			hint:  dtype.Uint32,
			want:  dtype.Uint32,
		},
		{
			name:  "and against an int8 column when it fits in one",
			value: 100,
			hint:  dtype.Int8,
			want:  dtype.Int8,
		},
		{
			name:  "an integer against a float column widens to the float",
			value: 3,
			hint:  dtype.Float32,
			want:  dtype.Float32,
		},
		{
			name:  "a value against a column of its own type is left alone",
			value: 1.5,
			hint:  dtype.Float64,
			want:  dtype.Float64,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := plan.LiteralColumn(c.value, c.hint)
			if err != nil {
				t.Fatalf("LiteralColumn(%#v, %s): %v", c.value, c.hint, err)
			}
			if !dtype.Equal(got.DType(), c.want) {
				t.Errorf("%#v against a %s column became a %s column, want %s",
					c.value, c.hint, got.DType(), c.want)
			}
		})
	}
}

func TestLiteralColumnOfATime(t *testing.T) {
	when := time.Date(2026, 8, 31, 12, 0, 0, 0, time.UTC)

	cases := []struct {
		name string
		hint dtype.DataType
		want int64
	}{
		{
			name: "no column to go by, which is nanoseconds",
			hint: nil,
			want: when.UnixNano(),
		},
		{
			name: "a column of seconds",
			hint: dtype.Timestamp{Unit: dtype.Second},
			want: when.Unix(),
		},
		{
			name: "a column of milliseconds",
			hint: dtype.Timestamp{Unit: dtype.Millisecond},
			want: when.UnixMilli(),
		},
		{
			name: "a column of microseconds",
			hint: dtype.Timestamp{Unit: dtype.Microsecond},
			want: when.UnixMicro(),
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, err := plan.LiteralColumn(when, c.hint)
			if err != nil {
				t.Fatalf("LiteralColumn: %v", err)
			}
			if v := got.Value[int64](0); v != c.want {
				t.Errorf("the time came out as %d, want %d", v, c.want)
			}
		})
	}
}

func TestLiteralColumnOfAValueThatWillNotFit(t *testing.T) {
	cases := []struct {
		name  string
		value any
		hint  dtype.DataType
		says  string
	}{
		{
			name:  "a number too big for the column",
			value: 300,
			hint:  dtype.Int8,
			says:  "300",
		},
		{
			name:  "a fraction against an integer column",
			value: 1.5,
			hint:  dtype.Int64,
			says:  "cannot use a float64 literal with a int64 column",
		},
		{
			name:  "a time with nanoseconds in it against a column of seconds",
			value: time.Date(2026, 8, 31, 12, 0, 0, 1, time.UTC),
			hint:  dtype.Timestamp{Unit: dtype.Second},
			says:  "does not fit",
		},
		{
			name:  "a value no column can hold",
			value: struct{}{},
			hint:  nil,
			says:  "is not a value a column can hold",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := plan.LiteralColumn(c.value, c.hint)
			if err == nil {
				t.Fatalf("LiteralColumn(%#v, %v) gave no error", c.value, c.hint)
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Errorf("LiteralColumn said %q, want it to mention %q", err, c.says)
			}
		})
	}
}
