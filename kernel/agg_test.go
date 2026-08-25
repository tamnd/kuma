package kernel_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// groupsOf returns the grouping of a column of keys written as one chunk, which
// is what most of the tests below want.
func groupsOf(t *testing.T, keys ...any) *kernel.Groups {
	t.Helper()

	g, err := kernel.GroupBy(col(t, dtype.String, keys))
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	return g
}

// checkAgg compares a result column against the values wanted, where a nil is a
// missing value.
func checkAgg(t *testing.T, got *array.Chunked, want []any) {
	t.Helper()

	if got.Len() != len(want) {
		t.Fatalf("there are %d results, want %d", got.Len(), len(want))
	}
	for i, w := range want {
		have := valueAt(t, got, i)
		if f, ok := w.(float64); ok && math.IsNaN(f) {
			if g, ok := have.(float64); !ok || !math.IsNaN(g) {
				t.Errorf("group %d is %v, want NaN", i, have)
			}
			continue
		}
		if have != w {
			t.Errorf("group %d is %v, want %v", i, have, w)
		}
	}
}

func TestSum(t *testing.T) {
	g := groupsOf(t, "a", "b", "a", "b", "a")

	tests := []struct {
		name string
		col  *array.Chunked
		dt   dtype.DataType
		want []any
	}{
		{
			"int64",
			col(t, dtype.Int64, []any{int64(1), int64(10), int64(2), int64(20), int64(3)}),
			dtype.Int64,
			[]any{int64(6), int64(30)},
		},
		{
			"int8 widens",
			col(t, dtype.Int8, []any{int8(100), int8(1), int8(100), int8(1), int8(100)}),
			dtype.Int64,
			[]any{int64(300), int64(2)},
		},
		{
			"uint8 widens",
			col(t, dtype.Uint8, []any{uint8(200), uint8(1), uint8(200), uint8(1), uint8(200)}),
			dtype.Uint64,
			[]any{uint64(600), uint64(2)},
		},
		{
			"float32 widens",
			col(t, dtype.Float32, []any{float32(0.5), float32(1), float32(0.5), float32(1), float32(0.5)}),
			dtype.Float64,
			[]any{1.5, 2.0},
		},
		{
			"booleans count the true ones",
			col(t, dtype.Bool, []any{true, false, true, true, false}),
			dtype.Int64,
			[]any{int64(2), int64(1)},
		},
		{
			"the missing values are skipped",
			col(t, dtype.Int64, []any{int64(1), nil, nil, int64(20), int64(3)}),
			dtype.Int64,
			[]any{int64(4), int64(20)},
		},
		{
			"a group with nothing in it sums to zero",
			col(t, dtype.Int64, []any{nil, int64(10), nil, int64(20), nil}),
			dtype.Int64,
			[]any{int64(0), int64(30)},
		},
		{
			"a NaN is a value and it spreads",
			col(t, dtype.Float64, []any{1.0, 10.0, math.NaN(), 20.0, 3.0}),
			dtype.Float64,
			[]any{math.NaN(), 30.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := kernel.Sum(tt.col, g)
			if err != nil {
				t.Fatalf("Sum: %v", err)
			}
			if got.DType() != tt.dt {
				t.Errorf("the result is a %s column, want %s", got.DType(), tt.dt)
			}
			checkAgg(t, got, tt.want)
		})
	}
}

// TestSumDuration is the one temporal type a total means something for, and the
// unit has to come through with it.
func TestSumDuration(t *testing.T) {
	dt := dtype.Duration{Unit: dtype.Millisecond}
	g := groupsOf(t, "a", "a")
	c := col(t, dt, []any{int64(250), int64(750)})

	got, err := kernel.Sum(c, g)
	if err != nil {
		t.Fatalf("Sum: %v", err)
	}
	if got.DType() != dt {
		t.Errorf("the result is a %s column, want %s", got.DType(), dt)
	}
	checkAgg(t, got, []any{int64(1000)})
}

func TestSumOverflowWraps(t *testing.T) {
	g := groupsOf(t, "a", "a")
	c := col(t, dtype.Int64, []any{int64(math.MaxInt64), int64(1)})

	got, err := kernel.Sum(c, g)
	if err != nil {
		t.Fatalf("Sum: %v", err)
	}
	checkAgg(t, got, []any{int64(math.MinInt64)})
}

func TestSumRefused(t *testing.T) {
	g := groupsOf(t, "a", "a")

	for _, c := range []*array.Chunked{
		col(t, dtype.String, []any{"x", "y"}),
		col(t, dtype.Timestamp{Unit: dtype.Second}, []any{int64(1), int64(2)}),
	} {
		t.Run(c.DType().String(), func(t *testing.T) {
			if _, err := kernel.Sum(c, g); err == nil {
				t.Fatalf("summing a %s column succeeded", c.DType())
			} else if !strings.Contains(err.Error(), c.DType().String()) {
				t.Errorf("the message is %q, want it to name the type", err.Error())
			}
		})
	}
}

func TestMean(t *testing.T) {
	g := groupsOf(t, "a", "b", "a", "b", "a")

	tests := []struct {
		name string
		col  *array.Chunked
		want []any
	}{
		{
			"integers average as floats",
			col(t, dtype.Int64, []any{int64(1), int64(10), int64(2), int64(20), int64(3)}),
			[]any{2.0, 15.0},
		},
		{
			"unsigned integers too",
			col(t, dtype.Uint32, []any{uint32(1), uint32(10), uint32(2), uint32(20), uint32(3)}),
			[]any{2.0, 15.0},
		},
		{
			"the divisor is the values and not the rows",
			col(t, dtype.Int64, []any{int64(1), int64(10), nil, int64(20), int64(3)}),
			[]any{2.0, 15.0},
		},
		{
			"a group with nothing in it has no average",
			col(t, dtype.Int64, []any{nil, int64(10), nil, int64(20), nil}),
			[]any{nil, 15.0},
		},
		{
			"booleans average to the fraction that are true",
			col(t, dtype.Bool, []any{true, false, true, true, false}),
			[]any{2.0 / 3.0, 0.5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := kernel.Mean(tt.col, g)
			if err != nil {
				t.Fatalf("Mean: %v", err)
			}
			if got.DType() != dtype.Float64 {
				t.Errorf("the result is a %s column, want float64", got.DType())
			}
			checkAgg(t, got, tt.want)
		})
	}
}

func TestMeanRefused(t *testing.T) {
	g := groupsOf(t, "a", "a")

	for _, c := range []*array.Chunked{
		col(t, dtype.String, []any{"x", "y"}),
		col(t, dtype.Duration{Unit: dtype.Second}, []any{int64(1), int64(2)}),
	} {
		t.Run(c.DType().String(), func(t *testing.T) {
			if _, err := kernel.Mean(c, g); err == nil {
				t.Fatalf("averaging a %s column succeeded", c.DType())
			} else if !strings.Contains(err.Error(), c.DType().String()) {
				t.Errorf("the message is %q, want it to name the type", err.Error())
			}
		})
	}
}

func TestCountAndSize(t *testing.T) {
	g := groupsOf(t, "a", "b", "a", "b", "a")
	c := col(t, dtype.Int64, []any{int64(1), nil, nil, int64(20), int64(3)})

	checkAgg(t, kernel.Count(c, g), []any{int64(2), int64(1)})
	checkAgg(t, kernel.Size(g), []any{int64(3), int64(2)})

	if got := kernel.Count(c, g); got.NullCount() != 0 {
		t.Errorf("Count has %d missing values, want none", got.NullCount())
	}
}

func TestMinAndMax(t *testing.T) {
	g := groupsOf(t, "a", "b", "a", "b", "a")

	tests := []struct {
		name    string
		col     *array.Chunked
		min     []any
		max     []any
		refused bool
	}{
		{
			name: "integers",
			col:  col(t, dtype.Int64, []any{int64(3), int64(10), int64(1), int64(20), int64(2)}),
			min:  []any{int64(1), int64(10)},
			max:  []any{int64(3), int64(20)},
		},
		{
			name: "strings compare by their bytes",
			col:  col(t, dtype.String, []any{"pear", "a", "apple", "b", "fig"}),
			min:  []any{"apple", "a"},
			max:  []any{"pear", "b"},
		},
		{
			name: "the missing values are skipped",
			col:  col(t, dtype.Int64, []any{nil, int64(10), int64(1), nil, int64(2)}),
			min:  []any{int64(1), int64(10)},
			max:  []any{int64(2), int64(10)},
		},
		{
			name: "a group with nothing in it has neither",
			col:  col(t, dtype.Int64, []any{nil, int64(10), nil, int64(20), nil}),
			min:  []any{nil, int64(10)},
			max:  []any{nil, int64(20)},
		},
		{
			name: "NaN is bigger than every number",
			col:  col(t, dtype.Float64, []any{3.0, 10.0, math.NaN(), 20.0, 2.0}),
			min:  []any{2.0, 10.0},
			max:  []any{math.NaN(), 20.0},
		},
		{
			name:    "a decimal has no order",
			col:     col(t, dtype.Decimal128{Precision: 18, Scale: 2}, []any{make([]byte, 16), make([]byte, 16), make([]byte, 16), make([]byte, 16), make([]byte, 16)}),
			refused: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotMin, err := kernel.Min(tt.col, g)
			if tt.refused {
				if err == nil {
					t.Fatalf("the smallest of a %s column succeeded", tt.col.DType())
				}
				if _, other := kernel.Max(tt.col, g); other == nil {
					t.Fatalf("the largest of a %s column succeeded", tt.col.DType())
				}
				return
			}
			if err != nil {
				t.Fatalf("Min: %v", err)
			}
			if gotMin.DType() != tt.col.DType() {
				t.Errorf("Min is a %s column, want %s", gotMin.DType(), tt.col.DType())
			}
			checkAgg(t, gotMin, tt.min)

			gotMax, err := kernel.Max(tt.col, g)
			if err != nil {
				t.Fatalf("Max: %v", err)
			}
			checkAgg(t, gotMax, tt.max)
		})
	}
}

func TestFirstAndLast(t *testing.T) {
	g := groupsOf(t, "a", "b", "a", "b", "a")

	tests := []struct {
		name  string
		col   *array.Chunked
		first []any
		last  []any
	}{
		{
			"every value is there",
			col(t, dtype.String, []any{"x", "p", "y", "q", "z"}),
			[]any{"x", "p"},
			[]any{"z", "q"},
		},
		{
			"the missing values are skipped at both ends",
			col(t, dtype.Int64, []any{nil, int64(10), int64(2), nil, nil}),
			[]any{int64(2), int64(10)},
			[]any{int64(2), int64(10)},
		},
		{
			"a group with nothing in it has neither",
			col(t, dtype.Int64, []any{nil, int64(10), nil, int64(20), nil}),
			[]any{nil, int64(10)},
			[]any{nil, int64(20)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			checkAgg(t, kernel.First(tt.col, g), tt.first)
			checkAgg(t, kernel.Last(tt.col, g), tt.last)
		})
	}
}

// TestAggOverOneGroup is the whole column at once, which is the same code with
// a grouping that says every row is together.
func TestAggOverOneGroup(t *testing.T) {
	c := col(t, dtype.Int64, []any{int64(1), int64(2), nil, int64(4)})
	g := kernel.OneGroup(c.Len())

	total, err := kernel.Sum(c, g)
	if err != nil {
		t.Fatalf("Sum: %v", err)
	}
	checkAgg(t, total, []any{int64(7)})

	mean, err := kernel.Mean(c, g)
	if err != nil {
		t.Fatalf("Mean: %v", err)
	}
	checkAgg(t, mean, []any{7.0 / 3.0})

	smallest, err := kernel.Min(c, g)
	if err != nil {
		t.Fatalf("Min: %v", err)
	}
	checkAgg(t, smallest, []any{int64(1)})

	checkAgg(t, kernel.Count(c, g), []any{int64(3)})
	checkAgg(t, kernel.Size(g), []any{int64(4)})
	checkAgg(t, kernel.First(c, g), []any{int64(1)})
	checkAgg(t, kernel.Last(c, g), []any{int64(4)})
}

// TestAggEmpty is a column with no rows, which has no groups and so no results.
func TestAggEmpty(t *testing.T) {
	c := col(t, dtype.Int64, []any{})
	g, err := kernel.GroupBy(c)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}

	total, err := kernel.Sum(c, g)
	if err != nil {
		t.Fatalf("Sum: %v", err)
	}
	if total.Len() != 0 {
		t.Errorf("the total has %d values, want none", total.Len())
	}
	if kernel.Size(g).Len() != 0 {
		t.Errorf("the sizes have %d values, want none", kernel.Size(g).Len())
	}
}

// TestAggChunked checks that the walk over the chunks lines up with the walk
// over the group of every row, which is the one thing that can go wrong when a
// column is cut up differently from the keys.
func TestAggChunked(t *testing.T) {
	keys, err := kernel.GroupBy(col(t, dtype.String,
		[]any{"a", "b"}, []any{"a"}, []any{"b", "a"}))
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	c := col(t, dtype.Int64,
		[]any{int64(1)}, []any{int64(10), int64(2)}, []any{}, []any{int64(20), int64(3)})

	total, err := kernel.Sum(c, keys)
	if err != nil {
		t.Fatalf("Sum: %v", err)
	}
	checkAgg(t, total, []any{int64(6), int64(30)})
}

func TestAggPanics(t *testing.T) {
	c := col(t, dtype.Int64, []any{int64(1), int64(2)})
	g := kernel.OneGroup(3)

	tests := []struct {
		name string
		call func()
	}{
		{"a nil column", func() { _, _ = kernel.Sum(nil, g) }},
		{"the wrong length", func() { _, _ = kernel.Sum(c, g) }},
		{"the wrong length for a mean", func() { _, _ = kernel.Mean(c, g) }},
		{"the wrong length for a count", func() { kernel.Count(c, g) }},
		{"the wrong length for a first", func() { kernel.First(c, g) }},
		{"the wrong length for a last", func() { kernel.Last(c, g) }},
		{"the wrong length for a min", func() { _, _ = kernel.Min(c, g) }},
		{"the wrong length for a max", func() { _, _ = kernel.Max(c, g) }},
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

// TestSumEveryWidth runs a total of every numeric width, since each of them
// widens through its own arm.
func TestSumEveryWidth(t *testing.T) {
	g := groupsOf(t, "a", "a")

	tests := []struct {
		dt   dtype.DataType
		vals []any
		want []any
	}{
		{dtype.Int8, []any{int8(1), int8(2)}, []any{int64(3)}},
		{dtype.Int16, []any{int16(1), int16(2)}, []any{int64(3)}},
		{dtype.Int32, []any{int32(1), int32(2)}, []any{int64(3)}},
		{dtype.Int64, []any{int64(1), int64(2)}, []any{int64(3)}},
		{dtype.Uint8, []any{uint8(1), uint8(2)}, []any{uint64(3)}},
		{dtype.Uint16, []any{uint16(1), uint16(2)}, []any{uint64(3)}},
		{dtype.Uint32, []any{uint32(1), uint32(2)}, []any{uint64(3)}},
		{dtype.Uint64, []any{uint64(1), uint64(2)}, []any{uint64(3)}},
		{dtype.Float32, []any{float32(1), float32(2)}, []any{3.0}},
		{dtype.Float64, []any{1.0, 2.0}, []any{3.0}},
	}

	for _, tt := range tests {
		t.Run(tt.dt.String(), func(t *testing.T) {
			got, err := kernel.Sum(col(t, tt.dt, tt.vals), g)
			if err != nil {
				t.Fatalf("Sum: %v", err)
			}
			checkAgg(t, got, tt.want)

			mean, err := kernel.Mean(col(t, tt.dt, tt.vals), g)
			if err != nil {
				t.Fatalf("Mean: %v", err)
			}
			checkAgg(t, mean, []any{1.5})
		})
	}
}

func TestVarAndStd(t *testing.T) {
	// The numbers are picked so the answers come out exact in binary. The two
	// groups have the same spread and different means, which is what a
	// variance is meant not to notice.
	g := groupsOf(t, "a", "b", "a", "b", "a", "b")
	c := col(t, dtype.Float64, []any{2.0, 102.0, 4.0, 104.0, 6.0, 106.0})

	sample, err := kernel.Var(c, g, 1)
	if err != nil {
		t.Fatalf("Var: %v", err)
	}
	checkAgg(t, sample, []any{4.0, 4.0})

	population, err := kernel.Var(c, g, 0)
	if err != nil {
		t.Fatalf("Var: %v", err)
	}
	checkAgg(t, population, []any{8.0 / 3, 8.0 / 3})

	sd, err := kernel.Std(c, g, 1)
	if err != nil {
		t.Fatalf("Std: %v", err)
	}
	checkAgg(t, sd, []any{2.0, 2.0})
}

// TestVarTooFewValues is the rule that a group with fewer values than the
// divisor wants has no variance rather than an infinite one.
func TestVarTooFewValues(t *testing.T) {
	g := groupsOf(t, "one", "two", "two")
	c := col(t, dtype.Int64, []any{int64(5), int64(1), int64(3)})

	got, err := kernel.Var(c, g, 1)
	if err != nil {
		t.Fatalf("Var: %v", err)
	}
	checkAgg(t, got, []any{nil, 2.0})

	// With a ddof of zero the single value has a variance, and it is zero.
	got, err = kernel.Var(c, g, 0)
	if err != nil {
		t.Fatalf("Var: %v", err)
	}
	checkAgg(t, got, []any{0.0, 1.0})
}

// TestVarLargeCloseValues is why this is Welford's method and not a sum of
// squares. These are eight digit numbers a hundredth apart, which is a column
// of prices, and totalling their squares in a float64 and subtracting gives
// minus a sixteenth, a negative variance of a number that cannot have one.
func TestVarLargeCloseValues(t *testing.T) {
	g := kernel.OneGroup(3)
	c := col(t, dtype.Float64, []any{12345678.00, 12345678.01, 12345678.02})

	got, err := kernel.Var(c, g, 1)
	if err != nil {
		t.Fatalf("Var: %v", err)
	}
	// The tolerance is relative and not exact because a hundredth is not a
	// float64 to begin with. What matters is the answer being right to seven
	// digits rather than having the wrong sign.
	if v := got.Value[float64](0); math.Abs(v-0.0001) > 1e-10 {
		t.Errorf("the variance is %v, want 0.0001, so the running mean is not doing its job", v)
	}
}

func TestVarRefused(t *testing.T) {
	g := kernel.OneGroup(2)

	if _, err := kernel.Var(col(t, dtype.String, []any{"a", "b"}), g, 1); err == nil {
		t.Error("the variance of a string column succeeded")
	}
	if _, err := kernel.Std(col(t, dtype.Timestamp{Unit: dtype.Nanosecond},
		[]any{int64(1), int64(2)}), g, 1); err == nil {
		t.Error("the standard deviation of a timestamp column succeeded")
	}
	if _, err := kernel.Var(col(t, dtype.Int64, []any{int64(1), int64(2)}), g, -1); err == nil {
		t.Error("a negative ddof succeeded")
	}
}

func TestNUnique(t *testing.T) {
	tests := []struct {
		name string
		keys []any
		dt   dtype.DataType
		vals []any
		want []any
	}{
		{
			name: "repeats count once",
			keys: []any{"a", "a", "a", "b"},
			dt:   dtype.Int64,
			vals: []any{int64(1), int64(1), int64(2), int64(9)},
			want: []any{int64(2), int64(1)},
		},
		{
			name: "the missing are not a value",
			keys: []any{"a", "a", "b", "b"},
			dt:   dtype.Int64,
			vals: []any{int64(1), nil, nil, nil},
			want: []any{int64(1), int64(0)},
		},
		{
			name: "strings",
			keys: []any{"a", "a", "a"},
			dt:   dtype.String,
			vals: []any{"x", "y", "x"},
			want: []any{int64(2)},
		},
		{
			name: "booleans top out at two",
			keys: []any{"a", "a", "a", "a"},
			dt:   dtype.Bool,
			vals: []any{true, false, true, false},
			want: []any{int64(2)},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := kernel.NUnique(col(t, tt.dt, tt.vals), groupsOf(t, tt.keys...))
			if err != nil {
				t.Fatalf("NUnique: %v", err)
			}
			checkAgg(t, got, tt.want)
		})
	}
}

// TestNUniqueFloats is the rule that distinct here means what it means to
// GroupBy, so every NaN is one value and the two zeros are one value.
func TestNUniqueFloats(t *testing.T) {
	nan := math.NaN()
	other := math.Float64frombits(math.Float64bits(nan) | 1)
	c := col(t, dtype.Float64, []any{nan, other, 0.0, math.Copysign(0, -1), 1.0})

	got, err := kernel.NUnique(c, kernel.OneGroup(5))
	if err != nil {
		t.Fatalf("NUnique: %v", err)
	}
	checkAgg(t, got, []any{int64(3)})
}

func TestNUniqueRefused(t *testing.T) {
	c, err := array.NewChunked(dtype.List{Elem: dtype.Int64})
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	if _, err := kernel.NUnique(c, kernel.OneGroup(0)); err == nil {
		t.Error("the distinct count of a list column succeeded")
	}
}

func TestNUniqueChunked(t *testing.T) {
	g := groupsOf(t, "a", "b", "a", "b", "a")
	c := col(t, dtype.Int64,
		[]any{int64(1), int64(7)},
		[]any{int64(1), int64(8), int64(2)})

	got, err := kernel.NUnique(c, g)
	if err != nil {
		t.Fatalf("NUnique: %v", err)
	}
	checkAgg(t, got, []any{int64(2), int64(2)})
}
