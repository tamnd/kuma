package kernel_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

func TestIndices(t *testing.T) {
	tests := []struct {
		name string
		mask [][]any
		want []int
	}{
		{
			name: "some of them",
			mask: [][]any{{true, false, true, false, true}},
			want: []int{0, 2, 4},
		},
		{
			name: "all of them",
			mask: [][]any{{true, true, true}},
			want: []int{0, 1, 2},
		},
		{
			name: "none of them",
			mask: [][]any{{false, false, false}},
			want: nil,
		},
		{
			name: "a null is not a selection",
			mask: [][]any{{true, nil, nil, true}},
			want: []int{0, 3},
		},
		{
			name: "across chunks",
			mask: [][]any{{false, true}, {true, false, nil}, {true}},
			want: []int{1, 2, 5},
		},
		{
			name: "nothing at all",
			mask: [][]any{nil},
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := kernel.Indices(col(t, dtype.Bool, tt.mask...))
			if !slices.Equal(got, tt.want) {
				t.Errorf("the mask selects %v, want %v", got, tt.want)
			}
		})
	}
}

// TestIndicesOfASlicedMask is here because a sliced boolean array starts part
// way into a byte, so a filter that forgot the offset would still pass every
// test above.
func TestIndicesOfASlicedMask(t *testing.T) {
	whole := col(t, dtype.Bool, []any{
		false, false, false, false, false, false, false, false,
		false, true, false, nil, true,
	})

	got := kernel.Indices(whole.Slice(9, 13))
	if want := []int{0, 3}; !slices.Equal(got, want) {
		t.Errorf("the mask selects %v, want %v", got, want)
	}
}

func TestFilter(t *testing.T) {
	src := col(t, dtype.Int64, []any{int64(10), int64(20)}, []any{nil, int64(40), int64(50)})
	mask := col(t, dtype.Bool, []any{true, false, true}, []any{nil, true})

	got := kernel.Filter(src, mask)
	checkTake(t, got, src, []int{0, 2, 4})

	if got.Len() != 3 {
		t.Fatalf("the result has %d values, want 3", got.Len())
	}
	want := []any{int64(10), nil, int64(50)}
	for k, v := range want {
		if have := valueAt(t, got, k); have != v {
			t.Errorf("value %d is %v, want %v", k, have, v)
		}
	}
}

// TestFilterKeepsTheDType is worth a test of its own because the result is
// built from the type of the column rather than from its values, and a column
// of nothing is where that goes wrong.
func TestFilterKeepsTheDType(t *testing.T) {
	dt := dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}
	src := col(t, dt, []any{int64(1), int64(2)})
	mask := col(t, dtype.Bool, []any{false, false})

	got := kernel.Filter(src, mask)
	if got.Len() != 0 {
		t.Errorf("the result has %d values, want none", got.Len())
	}
	if !dtype.Equal(got.DType(), dt) {
		t.Errorf("the result is a %s column, want %s", got.DType(), dt)
	}
}

func TestFilterPanics(t *testing.T) {
	src := col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)})

	tests := []struct {
		name string
		want string
		call func()
	}{
		{
			name: "a mask of the wrong length",
			want: "column of 3 values by a mask of 2",
			call: func() { kernel.Filter(src, col(t, dtype.Bool, []any{true, false})) },
		},
		{
			name: "a mask that is not boolean",
			want: "not a mask",
			call: func() { kernel.Filter(src, src) },
		},
		{
			name: "a nil mask",
			want: "nil mask",
			call: func() { kernel.Filter(src, nil) },
		},
		{
			name: "a nil column",
			want: "nil column",
			call: func() { kernel.Filter(nil, col(t, dtype.Bool, []any{true})) },
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
