package kernel_test

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

func TestGroupBy(t *testing.T) {
	tests := []struct {
		name  string
		col   *array.Chunked
		ids   []int
		sizes []int
		keys  []any
	}{
		{
			"strings",
			col(t, dtype.String, []any{"b", "a", "b", "c", "a"}),
			[]int{0, 1, 0, 2, 1},
			[]int{2, 2, 1},
			[]any{"b", "a", "c"},
		},
		{
			"numbers",
			col(t, dtype.Int64, []any{int64(7), int64(7), int64(3), int64(7)}),
			[]int{0, 0, 1, 0},
			[]int{3, 1},
			[]any{int64(7), int64(3)},
		},
		{
			"a null is its own group",
			col(t, dtype.Int64, []any{int64(1), nil, int64(1), nil}),
			[]int{0, 1, 0, 1},
			[]int{2, 2},
			[]any{int64(1), nil},
		},
		{
			"booleans",
			col(t, dtype.Bool, []any{true, false, true}),
			[]int{0, 1, 0},
			[]int{2, 1},
			[]any{true, false},
		},
		{
			"every row its own group",
			col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)}),
			[]int{0, 1, 2},
			[]int{1, 1, 1},
			[]any{int64(1), int64(2), int64(3)},
		},
		{
			"chunk boundaries do not matter",
			col(t, dtype.Int64, []any{int64(5)}, []any{}, []any{int64(6), int64(5)}),
			[]int{0, 1, 0},
			[]int{2, 1},
			[]any{int64(5), int64(6)},
		},
		{
			"the empty column",
			col(t, dtype.Int64, []any{}),
			[]int{},
			[]int{},
			[]any{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g, err := kernel.GroupBy(tt.col)
			if err != nil {
				t.Fatalf("GroupBy: %v", err)
			}
			if !slices.Equal(g.IDs(), tt.ids) {
				t.Errorf("the groups are %v, want %v", g.IDs(), tt.ids)
			}
			if !slices.Equal(g.Sizes(), tt.sizes) {
				t.Errorf("the sizes are %v, want %v", g.Sizes(), tt.sizes)
			}
			if g.NumGroups() != len(tt.keys) {
				t.Fatalf("there are %d groups, want %d", g.NumGroups(), len(tt.keys))
			}
			if g.Len() != tt.col.Len() {
				t.Errorf("Len is %d, want %d", g.Len(), tt.col.Len())
			}

			keys := g.Keys()
			if len(keys) != 1 {
				t.Fatalf("there are %d key columns, want 1", len(keys))
			}
			for i, want := range tt.keys {
				if got := valueAt(t, keys[0], i); got != want {
					t.Errorf("key %d is %v, want %v", i, got, want)
				}
			}
		})
	}
}

// TestGroupBySeveralKeys is the case the encoding exists for, where two rows
// agree on one key and not on the other.
func TestGroupBySeveralKeys(t *testing.T) {
	symbol := col(t, dtype.String, []any{"a", "a", "b", "b", "a"})
	day := col(t, dtype.Int32, []any{int32(1), int32(2), int32(1), int32(1), int32(1)})

	g, err := kernel.GroupBy(symbol, day)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}

	if want := []int{0, 1, 2, 2, 0}; !slices.Equal(g.IDs(), want) {
		t.Fatalf("the groups are %v, want %v", g.IDs(), want)
	}
	if len(g.Keys()) != 2 {
		t.Fatalf("there are %d key columns, want 2", len(g.Keys()))
	}

	wantSymbol := []any{"a", "a", "b"}
	wantDay := []any{int32(1), int32(2), int32(1)}
	for i := range g.NumGroups() {
		if got := valueAt(t, g.Keys()[0], i); got != wantSymbol[i] {
			t.Errorf("symbol of group %d is %v, want %v", i, got, wantSymbol[i])
		}
		if got := valueAt(t, g.Keys()[1], i); got != wantDay[i] {
			t.Errorf("day of group %d is %v, want %v", i, got, wantDay[i])
		}
	}
}

// TestGroupByKeyBoundary is why the byte encoding carries a length. Without one
// the two rows below would encode the same way and be called equal.
func TestGroupByKeyBoundary(t *testing.T) {
	left := col(t, dtype.String, []any{"a", "ab"})
	right := col(t, dtype.String, []any{"bc", "c"})

	g, err := kernel.GroupBy(left, right)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if g.NumGroups() != 2 {
		t.Fatalf("there are %d groups, want 2", g.NumGroups())
	}
}

// TestGroupByFloats covers the two values that have more than one way of being
// written down.
func TestGroupByFloats(t *testing.T) {
	nan := math.NaN()
	other := math.Float64frombits(math.Float64bits(nan) | 1<<40)
	if !math.IsNaN(other) {
		t.Fatal("the second bit pattern is not a NaN, so this test proves nothing")
	}

	c := col(t, dtype.Float64, []any{nan, other, 0.0, math.Copysign(0, -1), 1.5})

	g, err := kernel.GroupBy(c)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if want := []int{0, 0, 1, 1, 2}; !slices.Equal(g.IDs(), want) {
		t.Errorf("the groups are %v, want the NaNs together and the zeros together, %v",
			g.IDs(), want)
	}
}

// TestGroupByWidening checks that widening a narrow integer to eight bytes does
// not put two different values in the same group, which is the one thing that
// encoding has to get right.
func TestGroupByWidening(t *testing.T) {
	c := col(t, dtype.Int8, []any{int8(-1), int8(1), int8(-1), int8(127), int8(-128)})

	g, err := kernel.GroupBy(c)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if want := []int{0, 1, 0, 2, 3}; !slices.Equal(g.IDs(), want) {
		t.Errorf("the groups are %v, want %v", g.IDs(), want)
	}
}

// TestGroupByNullColumn is the type whose values are all missing, so there is
// exactly one group and it holds everything.
func TestGroupByNullColumn(t *testing.T) {
	c, err := array.NewChunked(dtype.Null, array.NewNull(3))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	g, err := kernel.GroupBy(c)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if g.NumGroups() != 1 {
		t.Fatalf("there are %d groups, want 1", g.NumGroups())
	}
	if want := []int{3}; !slices.Equal(g.Sizes(), want) {
		t.Errorf("the sizes are %v, want %v", g.Sizes(), want)
	}
}

// TestGroupByDecimals is the difference between grouping and sorting. A sort
// refuses a decimal because comparing its bytes would order it by its last
// digit, and a group by takes one because equal values are still equal bytes.
func TestGroupByDecimals(t *testing.T) {
	dt := dtype.Decimal128{Precision: 18, Scale: 2}
	one, two := make([]byte, 16), make([]byte, 16)
	two[0] = 1
	c := col(t, dt, []any{one, two, one})

	g, err := kernel.GroupBy(c)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if want := []int{0, 1, 0}; !slices.Equal(g.IDs(), want) {
		t.Errorf("the groups are %v, want %v", g.IDs(), want)
	}
}

func TestGroupByRefused(t *testing.T) {
	dt := dtype.List{Elem: dtype.Int64}
	c, err := array.NewChunked(dt)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	if _, err := kernel.GroupBy(c); err == nil {
		t.Fatal("grouping by a list column succeeded")
	} else if !strings.Contains(err.Error(), dt.String()) {
		t.Errorf("the message is %q, want it to name the type", err.Error())
	}
}

func TestGroupByPanics(t *testing.T) {
	tests := []struct {
		name string
		call func()
	}{
		{"no keys", func() { _, _ = kernel.GroupBy() }},
		{"a nil key", func() { _, _ = kernel.GroupBy(nil) }},
		{"keys of different lengths", func() {
			_, _ = kernel.GroupBy(
				col(t, dtype.Int64, []any{int64(1)}),
				col(t, dtype.Int64, []any{int64(1), int64(2)}),
			)
		}},
		{"a negative number of rows", func() { kernel.OneGroup(-1) }},
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

func TestOneGroup(t *testing.T) {
	g := kernel.OneGroup(3)

	if want := []int{0, 0, 0}; !slices.Equal(g.IDs(), want) {
		t.Errorf("the groups are %v, want %v", g.IDs(), want)
	}
	if g.NumGroups() != 1 {
		t.Errorf("there are %d groups, want 1", g.NumGroups())
	}
	if want := []int{3}; !slices.Equal(g.Sizes(), want) {
		t.Errorf("the sizes are %v, want %v", g.Sizes(), want)
	}
	if want := []int{0}; !slices.Equal(g.FirstRows(), want) {
		t.Errorf("the first rows are %v, want %v", g.FirstRows(), want)
	}
	if len(g.Keys()) != 0 {
		t.Errorf("there are %d key columns, want none", len(g.Keys()))
	}

	if empty := kernel.OneGroup(0); empty.NumGroups() != 0 {
		t.Errorf("nothing at all is %d groups, want none", empty.NumGroups())
	}
}

// TestGroupByFirstRows checks the positional first, which is the one that does
// not care whether the value is there.
func TestGroupByFirstRows(t *testing.T) {
	c := col(t, dtype.Int64, []any{int64(9), nil, int64(9), int64(4)})

	g, err := kernel.GroupBy(c)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if want := []int{0, 1, 3}; !slices.Equal(g.FirstRows(), want) {
		t.Errorf("the first rows are %v, want %v", g.FirstRows(), want)
	}
}

// TestGroupByEveryType runs one column of every type a key can be through the
// encoding, since each of them is written a different way and a type that gets
// its own arm gets its own chance to be wrong.
func TestGroupByEveryType(t *testing.T) {
	tests := []struct {
		dt     dtype.DataType
		values []any
	}{
		{dtype.Bool, []any{true, false, true}},
		{dtype.Int8, []any{int8(1), int8(2), int8(1)}},
		{dtype.Int16, []any{int16(1), int16(2), int16(1)}},
		{dtype.Int32, []any{int32(1), int32(2), int32(1)}},
		{dtype.Int64, []any{int64(1), int64(2), int64(1)}},
		{dtype.Uint8, []any{uint8(1), uint8(2), uint8(1)}},
		{dtype.Uint16, []any{uint16(1), uint16(2), uint16(1)}},
		{dtype.Uint32, []any{uint32(1), uint32(2), uint32(1)}},
		{dtype.Uint64, []any{uint64(1), uint64(2), uint64(1)}},
		{dtype.Float32, []any{float32(1), float32(2), float32(1)}},
		{dtype.Float64, []any{1.0, 2.0, 1.0}},
		{dtype.String, []any{"x", "y", "x"}},
		{dtype.Binary, []any{[]byte("x"), []byte("y"), []byte("x")}},
		{dtype.Date32, []any{int32(1), int32(2), int32(1)}},
		{dtype.Date64, []any{int64(1), int64(2), int64(1)}},
		{dtype.Time32{Unit: dtype.Second}, []any{int32(1), int32(2), int32(1)}},
		{dtype.Time64{Unit: dtype.Nanosecond}, []any{int64(1), int64(2), int64(1)}},
		{dtype.Timestamp{Unit: dtype.Second}, []any{int64(1), int64(2), int64(1)}},
		{dtype.Duration{Unit: dtype.Second}, []any{int64(1), int64(2), int64(1)}},
	}

	for _, tt := range tests {
		t.Run(tt.dt.String(), func(t *testing.T) {
			g, err := kernel.GroupBy(col(t, tt.dt, tt.values))
			if err != nil {
				t.Fatalf("GroupBy: %v", err)
			}
			if want := []int{0, 1, 0}; !slices.Equal(g.IDs(), want) {
				t.Errorf("the groups are %v, want %v", g.IDs(), want)
			}
		})
	}
}

// TestGroupByNoChunks is a column with no chunks at all, which is a different
// thing from a column with one empty chunk and is what a slice of nothing or a
// reader that read nothing produces.
func TestGroupByNoChunks(t *testing.T) {
	c, err := array.NewChunked(dtype.Int64)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	g, err := kernel.GroupBy(c)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if g.NumGroups() != 0 {
		t.Errorf("there are %d groups, want none", g.NumGroups())
	}
	if g.Len() != 0 {
		t.Errorf("Len is %d, want 0", g.Len())
	}
}

func TestDistinctIndex(t *testing.T) {
	tests := []struct {
		name string
		cols []*array.Chunked
		want []int
	}{
		{
			"strings",
			[]*array.Chunked{col(t, dtype.String, []any{"b", "a", "b", "c", "a"})},
			[]int{0, 1, 3},
		},
		{
			"a null is a value like any other",
			[]*array.Chunked{col(t, dtype.Int64, []any{int64(1), nil, int64(1), nil})},
			[]int{0, 1},
		},
		{
			"two keys, one of them agreeing",
			[]*array.Chunked{
				col(t, dtype.String, []any{"a", "a", "b", "b", "a"}),
				col(t, dtype.Int32, []any{int32(1), int32(2), int32(1), int32(1), int32(1)}),
			},
			[]int{0, 1, 2},
		},
		{
			"every row already distinct",
			[]*array.Chunked{col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)})},
			[]int{0, 1, 2},
		},
		{
			"chunk boundaries do not matter",
			[]*array.Chunked{col(t, dtype.Int64, []any{int64(5)}, []any{}, []any{int64(6), int64(5)})},
			[]int{0, 1},
		},
		{
			"the empty column",
			[]*array.Chunked{col(t, dtype.Int64, []any{})},
			nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := kernel.DistinctIndex(tt.cols...)
			if err != nil {
				t.Fatalf("DistinctIndex: %v", err)
			}
			if !slices.Equal(got, tt.want) {
				t.Errorf("the rows are %v, want %v", got, tt.want)
			}
		})
	}
}

// TestDistinctIndexIsTheFirstRowOfEachGroup is the promise that the two ways of
// dividing the rows up agree, which is what lets a distinct be documented as a
// group by with the aggregations left out.
func TestDistinctIndexIsTheFirstRowOfEachGroup(t *testing.T) {
	symbol := col(t, dtype.String, []any{"a", "b", "a", nil, "b", nil, "c"})
	day := col(t, dtype.Int32, []any{int32(1), int32(1), int32(2), int32(1), int32(1), int32(1), nil})

	g, err := kernel.GroupBy(symbol, day)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	got, err := kernel.DistinctIndex(symbol, day)
	if err != nil {
		t.Fatalf("DistinctIndex: %v", err)
	}

	if !slices.Equal(got, g.FirstRows()) {
		t.Errorf("the rows are %v, want the first rows of the groups, %v", got, g.FirstRows())
	}
}

func TestDistinctIndexRefused(t *testing.T) {
	dt := dtype.List{Elem: dtype.Int64}
	c, err := array.NewChunked(dt)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	if _, err := kernel.DistinctIndex(c); err == nil {
		t.Fatal("the distinct rows of a list column succeeded")
	} else if !strings.Contains(err.Error(), dt.String()) {
		t.Errorf("the message is %q, want it to name the type", err.Error())
	}
}

func TestDistinctIndexPanics(t *testing.T) {
	tests := []struct {
		name string
		call func()
	}{
		{"no keys", func() { _, _ = kernel.DistinctIndex() }},
		{"a nil key", func() { _, _ = kernel.DistinctIndex(nil) }},
		{"keys of different lengths", func() {
			_, _ = kernel.DistinctIndex(
				col(t, dtype.Int64, []any{int64(1)}),
				col(t, dtype.Int64, []any{int64(1), int64(2)}),
			)
		}},
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
