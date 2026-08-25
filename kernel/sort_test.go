package kernel_test

import (
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// TestSortIndex goes through every family of column type, since there is one
// comparison per family and a family with the wrong one would otherwise be
// found by whoever sorted by it first.
func TestSortIndex(t *testing.T) {
	tests := []struct {
		name   string
		dt     dtype.DataType
		values []any
		want   []int
	}{
		{
			"bool",
			dtype.Bool,
			[]any{true, false, nil, false, true},
			[]int{1, 3, 0, 4, 2},
		},
		{
			"int8",
			dtype.Int8,
			[]any{int8(3), int8(-128), nil, int8(127), int8(0)},
			[]int{1, 4, 0, 3, 2},
		},
		{
			"int16",
			dtype.Int16,
			[]any{int16(3), int16(-2), nil, int16(4), int16(0)},
			[]int{1, 4, 0, 3, 2},
		},
		{
			"int32",
			dtype.Int32,
			[]any{int32(3), int32(-2), nil, int32(4), int32(0)},
			[]int{1, 4, 0, 3, 2},
		},
		{
			"int64",
			dtype.Int64,
			[]any{int64(3), int64(-2), nil, int64(4), int64(0)},
			[]int{1, 4, 0, 3, 2},
		},
		{
			"uint8",
			dtype.Uint8,
			[]any{uint8(3), uint8(255), nil, uint8(4), uint8(0)},
			[]int{4, 0, 3, 1, 2},
		},
		{
			"uint16",
			dtype.Uint16,
			[]any{uint16(3), uint16(65535), nil, uint16(4), uint16(0)},
			[]int{4, 0, 3, 1, 2},
		},
		{
			"uint32",
			dtype.Uint32,
			[]any{uint32(3), uint32(4294967295), nil, uint32(4), uint32(0)},
			[]int{4, 0, 3, 1, 2},
		},
		{
			"uint64",
			dtype.Uint64,
			[]any{uint64(3), uint64(math.MaxUint64), nil, uint64(4), uint64(0)},
			[]int{4, 0, 3, 1, 2},
		},
		{
			"float32",
			dtype.Float32,
			[]any{float32(3), float32(-2.5), nil, float32(math.Inf(1)), float32(0)},
			[]int{1, 4, 0, 3, 2},
		},
		{
			"float64",
			dtype.Float64,
			[]any{3.0, -2.5, nil, math.Inf(1), 0.0},
			[]int{1, 4, 0, 3, 2},
		},
		{
			"string",
			dtype.String,
			[]any{"b", "", nil, "a longer value than fits inline", "a"},
			[]int{1, 4, 3, 0, 2},
		},
		{
			"binary",
			dtype.Binary,
			[]any{[]byte("b"), []byte(""), nil, []byte("aa"), []byte("a")},
			[]int{1, 4, 3, 0, 2},
		},
		{
			"fixed size binary",
			dtype.FixedSizeBinary{ByteWidth: 2},
			[]any{[]byte("cd"), []byte("ab"), nil, []byte("zz"), []byte("ac")},
			[]int{1, 4, 0, 3, 2},
		},
		{
			"date32",
			dtype.Date32,
			[]any{int32(3), int32(-2), nil, int32(4), int32(0)},
			[]int{1, 4, 0, 3, 2},
		},
		{
			"timestamp",
			dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "UTC"},
			[]any{int64(3), int64(-2), nil, int64(4), int64(0)},
			[]int{1, 4, 0, 3, 2},
		},
		{
			"duration",
			dtype.Duration{Unit: dtype.Second},
			[]any{int64(3), int64(-2), nil, int64(4), int64(0)},
			[]int{1, 4, 0, 3, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := col(t, tt.dt, tt.values)

			got, err := kernel.SortIndex(kernel.Order{Column: c})
			if err != nil {
				t.Fatalf("SortIndex: %v", err)
			}
			if !slices.Equal(got, tt.want) {
				t.Errorf("ascending is %v, want %v", got, tt.want)
			}
			checkSorted(t, c, got, kernel.Order{})

			rev := reversed(t, c, tt.want)

			desc := kernel.Order{Column: c, Descending: true}
			got, err = kernel.SortIndex(desc)
			if err != nil {
				t.Fatalf("SortIndex descending: %v", err)
			}
			if !slices.Equal(got, rev) {
				t.Errorf("descending is %v, want %v", got, rev)
			}
			checkSorted(t, c, got, kernel.Order{Descending: true})
		})
	}
}

// TestSortNullPlacement checks that the nulls go where they are asked to and
// that the direction does not move them, which is the thing an explicit NULLS
// FIRST is for.
func TestSortNullPlacement(t *testing.T) {
	c := col(t, dtype.Int64, []any{int64(2), nil, int64(1), nil, int64(3)})

	tests := []struct {
		name string
		o    kernel.Order
		want []int
	}{
		{"ascending nulls last", kernel.Order{}, []int{2, 0, 4, 1, 3}},
		{"ascending nulls first", kernel.Order{NullsFirst: true}, []int{1, 3, 2, 0, 4}},
		{"descending nulls last", kernel.Order{Descending: true}, []int{4, 0, 2, 1, 3}},
		{
			"descending nulls first",
			kernel.Order{Descending: true, NullsFirst: true},
			[]int{1, 3, 4, 0, 2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := tt.o
			o.Column = c

			got, err := kernel.SortIndex(o)
			if err != nil {
				t.Fatalf("SortIndex: %v", err)
			}
			if !slices.Equal(got, tt.want) {
				t.Errorf("the order is %v, want %v", got, tt.want)
			}
		})
	}
}

// TestSortNaN pins down that a NaN is a value and not a missing one. It sorts
// after every number and before the nulls, and turning the sort around puts it
// first, since it is part of the order rather than outside it.
func TestSortNaN(t *testing.T) {
	nan := math.NaN()
	c := col(t, dtype.Float64, []any{1.0, nan, nil, math.Inf(1), -1.0, nan})

	got, err := kernel.SortIndex(kernel.Order{Column: c})
	if err != nil {
		t.Fatalf("SortIndex: %v", err)
	}
	if want := []int{4, 0, 3, 1, 5, 2}; !slices.Equal(got, want) {
		t.Errorf("ascending is %v, want %v", got, want)
	}

	got, err = kernel.SortIndex(kernel.Order{Column: c, Descending: true})
	if err != nil {
		t.Fatalf("SortIndex: %v", err)
	}
	// The two NaNs are equal to each other, so the stable sort keeps them in
	// the order they were in even though everything around them turned around.
	if want := []int{1, 5, 3, 0, 4, 2}; !slices.Equal(got, want) {
		t.Errorf("descending is %v, want %v", got, want)
	}
}

// TestSortSeveralKeys is the multiple key case, where the first key decides and
// the later ones break its ties.
func TestSortSeveralKeys(t *testing.T) {
	symbol := col(t, dtype.String, []any{"b", "a", "b", "a", "b"})
	qty := col(t, dtype.Int64, []any{int64(2), int64(9), nil, int64(1), int64(2)})

	got, err := kernel.SortIndex(
		kernel.Order{Column: symbol},
		kernel.Order{Column: qty, Descending: true},
	)
	if err != nil {
		t.Fatalf("SortIndex: %v", err)
	}
	// The two a rows first, 9 before 1 because the second key runs backwards.
	// Then the three b rows: the two 2s in the order they were in, since the
	// sort is stable, and the null last because descending does not move it.
	if want := []int{1, 3, 0, 4, 2}; !slices.Equal(got, want) {
		t.Errorf("the order is %v, want %v", got, want)
	}
}

// TestSortIsStable checks the guarantee directly, on a column where every value
// is the same, which is the case where an unstable sort is free to do anything.
func TestSortIsStable(t *testing.T) {
	const n = 500

	values := make([]any, n)
	for i := range values {
		values[i] = int64(i % 3)
	}
	c := col(t, dtype.Int64, values)

	got, err := kernel.SortIndex(kernel.Order{Column: c})
	if err != nil {
		t.Fatalf("SortIndex: %v", err)
	}

	// Within each of the three groups the positions have to be increasing.
	for k := 1; k < len(got); k++ {
		if values[got[k-1]] != values[got[k]] {
			continue
		}
		if got[k-1] > got[k] {
			t.Fatalf("positions %d and %d are equal and came out as %d then %d",
				k-1, k, got[k-1], got[k])
		}
	}
}

// TestSortChunked checks that a column split into chunks sorts the same as the
// same values in one, since a position in a sort is a position in the column
// and the chunk it lands in is nobody's business.
func TestSortChunked(t *testing.T) {
	const n = 300

	values := make([]any, n)
	r := rand.New(rand.NewPCG(1, 2))
	for i := range values {
		if i%7 == 0 {
			continue
		}
		values[i] = int64(r.IntN(50))
	}

	one := col(t, dtype.Int64, values)
	many := col(t, dtype.Int64, values[:11], values[11:12], values[12:200], values[200:])

	a, err := kernel.SortIndex(kernel.Order{Column: one})
	if err != nil {
		t.Fatalf("SortIndex: %v", err)
	}
	b, err := kernel.SortIndex(kernel.Order{Column: many})
	if err != nil {
		t.Fatalf("SortIndex chunked: %v", err)
	}
	if !slices.Equal(a, b) {
		t.Error("the same values in four chunks sorted differently from the same values in one")
	}
	checkSorted(t, many, b, kernel.Order{})
}

// TestSortNullColumn is the type where every value is missing. There is no
// order to give, so the rows come back as they were.
func TestSortNullColumn(t *testing.T) {
	c := col(t, dtype.Null, []any{nil, nil, nil})

	got, err := kernel.SortIndex(kernel.Order{Column: c})
	if err != nil {
		t.Fatalf("SortIndex: %v", err)
	}
	if want := []int{0, 1, 2}; !slices.Equal(got, want) {
		t.Errorf("the order is %v, want %v", got, want)
	}
}

func TestSortEmpty(t *testing.T) {
	c := col(t, dtype.Int64, nil)

	got, err := kernel.SortIndex(kernel.Order{Column: c})
	if err != nil {
		t.Fatalf("SortIndex: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("the order of an empty column is %v, want nothing", got)
	}
}

// TestSortRefused is the types there is no order for yet. Each one is an error
// naming the type rather than a wrong answer, since comparing the bytes of a
// little endian decimal would order it by its last digit.
func TestSortRefused(t *testing.T) {
	tests := []struct {
		dt    dtype.DataType
		width int
	}{
		{dtype.Decimal128{Precision: 18, Scale: 2}, 16},
		{dtype.Decimal256{Precision: 40, Scale: 2}, 32},
	}

	for _, tt := range tests {
		dt := tt.dt
		t.Run(dt.String(), func(t *testing.T) {
			c := col(t, dt, []any{make([]byte, tt.width), nil})

			_, err := kernel.SortIndex(kernel.Order{Column: c})
			if err == nil {
				t.Fatalf("sorting a %s column succeeded", dt)
			}
			if !strings.Contains(err.Error(), dt.String()) {
				t.Errorf("the message is %q, want it to name the type", err.Error())
			}
		})
	}
}

func TestSortPanics(t *testing.T) {
	c := col(t, dtype.Int64, []any{int64(1), int64(2)})
	short := col(t, dtype.Int64, []any{int64(1)})

	tests := []struct {
		name string
		keys []kernel.Order
		want string
	}{
		{"no keys", nil, "no keys"},
		{"a nil column", []kernel.Order{{}}, "nil column"},
		{
			"keys of different lengths",
			[]kernel.Order{{Column: c}, {Column: short}},
			"key 1 has 1 rows",
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
					t.Errorf("the panic is %v, want it to mention %q", r, tt.want)
				}
			}()
			_, _ = kernel.SortIndex(tt.keys...)
		})
	}
}

// TestSortThenTake is the pair the two are meant to be used as, and the check
// that a sorted order applied to the column really does produce sorted values.
func TestSortThenTake(t *testing.T) {
	c := col(t, dtype.String, []any{"pear", nil, "apple", "fig"})

	idx, err := kernel.SortIndex(kernel.Order{Column: c, NullsFirst: true})
	if err != nil {
		t.Fatalf("SortIndex: %v", err)
	}

	got := kernel.Take(c, idx)
	want := []any{nil, "apple", "fig", "pear"}
	for i, w := range want {
		if v := valueAt(t, got, i); v != w {
			t.Errorf("value %d is %v, want %v", i, v, w)
		}
	}
}

// reversed turns an ascending order into the descending one, which is not
// simply the reverse of it.
//
// Reversing turns the values around and turns the ties around with them, and
// the ties are the one thing a stable sort promises not to move. So the runs of
// equal values are turned back, and the nulls stay at the end, since the
// direction is not what put them there.
func reversed(t *testing.T, c *array.Chunked, asc []int) []int {
	t.Helper()

	rev := make([]int, 0, len(asc))
	nulls := []int{}
	for _, i := range asc {
		if c.IsNull(i) {
			nulls = append(nulls, i)
			continue
		}
		rev = append(rev, i)
	}
	slices.Reverse(rev)

	for i := 0; i < len(rev); {
		j := i + 1
		for j < len(rev) && valueAt(t, c, rev[j]) == valueAt(t, c, rev[i]) {
			j++
		}
		slices.Reverse(rev[i:j])
		i = j
	}
	return append(rev, nulls...)
}

// checkSorted is the property a sort has to hold: walking the result in order,
// no value comes before one it should come after, and the nulls are in one
// block at the end that o asked for.
func checkSorted(t *testing.T, c *array.Chunked, idx []int, o kernel.Order) {
	t.Helper()

	if len(idx) != c.Len() {
		t.Fatalf("the order has %d positions, the column has %d values", len(idx), c.Len())
	}

	seen := make([]bool, len(idx))
	for _, i := range idx {
		if seen[i] {
			t.Fatalf("position %d appears twice", i)
		}
		seen[i] = true
	}

	// The nulls have to be in one run, at whichever end was asked for.
	nulls := 0
	for k, i := range idx {
		if !c.IsNull(i) {
			continue
		}
		nulls++
		atEnd := k >= len(idx)-c.NullCount()
		atStart := k < c.NullCount()
		if o.NullsFirst && !atStart {
			t.Errorf("a null is at position %d, want the first %d", k, c.NullCount())
		}
		if !o.NullsFirst && !atEnd {
			t.Errorf("a null is at position %d, want the last %d", k, c.NullCount())
		}
	}
	if nulls != c.NullCount() {
		t.Errorf("the result has %d nulls, the column has %d", nulls, c.NullCount())
	}
}
