package kernel_test

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// TestTake goes through every family of column type, since the gather has one
// case per family and a family that is missing a case would otherwise be found
// by whoever used it first.
func TestTake(t *testing.T) {
	tests := []struct {
		name   string
		dt     dtype.DataType
		values []any
	}{
		{"bool", dtype.Bool, []any{true, false, nil, true, false}},
		{"int8", dtype.Int8, []any{int8(1), int8(-2), nil, int8(4), int8(5)}},
		{"int16", dtype.Int16, []any{int16(1), int16(-2), nil, int16(4), int16(5)}},
		{"int32", dtype.Int32, []any{int32(1), int32(-2), nil, int32(4), int32(5)}},
		{"int64", dtype.Int64, []any{int64(1), int64(-2), nil, int64(4), int64(5)}},
		{"uint8", dtype.Uint8, []any{uint8(1), uint8(2), nil, uint8(4), uint8(5)}},
		{"uint16", dtype.Uint16, []any{uint16(1), uint16(2), nil, uint16(4), uint16(5)}},
		{"uint32", dtype.Uint32, []any{uint32(1), uint32(2), nil, uint32(4), uint32(5)}},
		{"uint64", dtype.Uint64, []any{uint64(1), uint64(2), nil, uint64(4), uint64(5)}},
		{"float32", dtype.Float32, []any{float32(1.5), float32(-2.5), nil, float32(4), float32(5)}},
		{"float64", dtype.Float64, []any{1.5, -2.5, nil, 4.0, 5.0}},
		{"string", dtype.String, []any{"a", "", nil, "a longer value than fits inline", "z"}},
		{"binary", dtype.Binary, []any{[]byte("a"), []byte(""), nil, []byte("bb"), []byte("z")}},
		{"date32", dtype.Date32, []any{int32(1), int32(2), nil, int32(4), int32(5)}},
		{
			"timestamp",
			dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "UTC"},
			[]any{int64(1), int64(2), nil, int64(4), int64(5)},
		},
		{
			"fixed size binary",
			dtype.FixedSizeBinary{ByteWidth: 2},
			[]any{[]byte("ab"), []byte("cd"), nil, []byte("ef"), []byte("gh")},
		},
		{
			"decimal",
			dtype.Decimal128{Precision: 18, Scale: 2},
			[]any{make([]byte, 16), make([]byte, 16), nil, make([]byte, 16), make([]byte, 16)},
		},
	}

	// One list of positions for all of them: a reordering, a repeat, a value
	// that is missing in the source, and a position below zero.
	idx := []int{4, 0, 0, 2, -1, 3}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := col(t, tt.dt, tt.values)
			checkTake(t, kernel.Take(src, idx), src, idx)
		})
	}
}

// TestTakeNullColumn is on its own because a null column has no values at all,
// so the gather has nothing to read and still has to get the length right.
func TestTakeNullColumn(t *testing.T) {
	src, err := array.NewChunked(dtype.Null, array.NewNull(4))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	idx := []int{3, 0, -1}
	got := kernel.Take(src, idx)
	checkTake(t, got, src, idx)
	if got.NullCount() != len(idx) {
		t.Errorf("a gather from a null column produced %d nulls, want %d", got.NullCount(), len(idx))
	}
}

// TestTakeChunked is the case the finder exists for. The positions are the
// interesting part rather than the values, so they run forwards, backwards, in
// jumps and all in one chunk.
func TestTakeChunked(t *testing.T) {
	src := col(t, dtype.Int64,
		[]any{int64(0), int64(1)},
		[]any{int64(2), nil, int64(4), int64(5)},
		[]any{int64(6)},
	)
	if src.NumChunks() != 3 {
		t.Fatalf("the column has %d chunks, want 3", src.NumChunks())
	}

	tests := []struct {
		name string
		idx  []int
	}{
		{"in order", []int{0, 1, 2, 3, 4, 5, 6}},
		{"backwards", []int{6, 5, 4, 3, 2, 1, 0}},
		{"jumping about", []int{6, 0, 3, 1, 5, 2, 4}},
		{"all in one chunk", []int{4, 5, 4, 5}},
		{"across a boundary", []int{1, 2, 5, 6}},
		{"the same position over and over", []int{3, 3, 3}},
		{"nothing", nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			checkTake(t, kernel.Take(src, tt.idx), src, tt.idx)
		})
	}
}

// TestTakeSliced checks that a gather reads from the column it was given rather
// than from the buffers underneath it, which for a sliced column are not the
// same thing.
func TestTakeSliced(t *testing.T) {
	whole := col(t, dtype.Int64, []any{int64(0), int64(1)}, []any{int64(2), nil, int64(4), int64(5)})
	src := whole.Slice(1, 5)

	idx := []int{0, 1, 2, 3}
	got := kernel.Take(src, idx)
	checkTake(t, got, src, idx)

	want := []any{int64(1), int64(2), nil, int64(4)}
	for k, v := range want {
		if have := valueAt(t, got, k); have != v {
			t.Errorf("value %d is %v, want %v", k, have, v)
		}
	}
}

// TestTakeEmptyColumn is here because a column with no values has no chunks at
// all, which is the one case where the finder has nothing to search.
func TestTakeEmptyColumn(t *testing.T) {
	src := col(t, dtype.Int64, nil)
	if src.Len() != 0 {
		t.Fatalf("the column has %d values, want none", src.Len())
	}

	if got := kernel.Take(src, nil); got.Len() != 0 {
		t.Errorf("a gather of nothing from nothing produced %d values", got.Len())
	}
	if got := kernel.Take(src, []int{-1, -1}); got.Len() != 2 || got.NullCount() != 2 {
		t.Errorf("a gather of two nulls from an empty column produced %s", got)
	}
}

func TestTakePanics(t *testing.T) {
	src := col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)})

	tests := []struct {
		name string
		want string
		call func()
	}{
		{
			name: "a position past the end",
			want: "position 3 out of range",
			call: func() { kernel.Take(src, []int{0, 3}) },
		},
		{
			name: "a position miles past the end",
			want: "position 1000 out of range",
			call: func() { kernel.Take(src, []int{1000}) },
		},
		{
			name: "a nil column",
			want: "nil column",
			call: func() { kernel.Take(nil, []int{0}) },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("it did not panic")
				}
				if msg, ok := r.(string); !ok || !strings.Contains(msg, tt.want) {
					t.Errorf("the panic says %v, want something with %q in it", r, tt.want)
				}
			}()
			tt.call()
		})
	}
}
