package kernel_test

// The tests for the type rules of the aggregations. What they check is that the
// type a rule promises is the type the aggregation gives, over a column of
// every type there is, since the rule and the kernel list the types separately
// and nothing but a test holds the two lists together.

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// aggregation is one of the operations that takes a column and gives a value
// per group, paired with the rule that says what it comes out as.
type aggregation struct {
	name string
	run  func(*array.Chunked, *kernel.Groups) (*array.Chunked, error)
	give func(dtype.DataType) (dtype.DataType, error)
}

// aggregations is every one that has a rule. Count, Size, First and Last have
// none, because a count is an int64 whatever it counted and a first value is
// the type of the column it came out of, and neither of those can be wrong
// about a type it never looked at.
var aggregations = []aggregation{
	{"Sum", kernel.Sum, kernel.SumType},
	{"Mean", kernel.Mean, kernel.MeanType},
	{"Min", kernel.Min, kernel.MinMaxType},
	{"Max", kernel.Max, kernel.MinMaxType},
	{
		"Var",
		func(c *array.Chunked, g *kernel.Groups) (*array.Chunked, error) { return kernel.Var(c, g, 1) },
		kernel.VarType,
	},
	{
		"Std",
		func(c *array.Chunked, g *kernel.Groups) (*array.Chunked, error) { return kernel.Std(c, g, 1) },
		kernel.StdType,
	},
	{"Median", kernel.Median, kernel.MedianType},
	{
		"Quantile",
		func(c *array.Chunked, g *kernel.Groups) (*array.Chunked, error) {
			return kernel.Quantile(c, g, 0.5, kernel.Linear)
		},
		kernel.QuantileType,
	},
	{"NUnique", kernel.NUnique, kernel.NUniqueType},
}

// everyType is a column of each type an aggregation can be handed, including
// the ones every aggregation turns away, since a rule that is wrong about what
// is refused is as wrong as one that is wrong about what comes back.
var everyType = []struct {
	dt   dtype.DataType
	vals []any
}{
	{dtype.Null, []any{nil, nil}},
	{dtype.Bool, []any{true, false}},
	{dtype.Int8, []any{int8(1), int8(2)}},
	{dtype.Int16, []any{int16(1), int16(2)}},
	{dtype.Int32, []any{int32(1), int32(2)}},
	{dtype.Int64, []any{int64(1), int64(2)}},
	{dtype.Uint8, []any{uint8(1), uint8(2)}},
	{dtype.Uint16, []any{uint16(1), uint16(2)}},
	{dtype.Uint32, []any{uint32(1), uint32(2)}},
	{dtype.Uint64, []any{uint64(1), uint64(2)}},
	{dtype.Float32, []any{float32(1), float32(2)}},
	{dtype.Float64, []any{1.0, 2.0}},
	{dtype.String, []any{"a", "b"}},
	{dtype.Binary, []any{[]byte("a"), []byte("b")}},
	{dtype.FixedSizeBinary{ByteWidth: 2}, []any{[]byte("ab"), []byte("cd")}},
	{dtype.Date32, []any{int32(1), int32(2)}},
	{dtype.Date64, []any{int64(1), int64(2)}},
	{dtype.Time32{Unit: dtype.Second}, []any{int32(1), int32(2)}},
	{dtype.Time64{Unit: dtype.Microsecond}, []any{int64(1), int64(2)}},
	{dtype.Timestamp{Unit: dtype.Second, Zone: "UTC"}, []any{int64(1), int64(2)}},
	{dtype.Duration{Unit: dtype.Millisecond}, []any{int64(1), int64(2)}},
	{dtype.Decimal128{Precision: 18, Scale: 2}, []any{make([]byte, 16), make([]byte, 16)}},
	{dtype.Decimal256{Precision: 40, Scale: 2}, []any{make([]byte, 32), make([]byte, 32)}},

	// A list has no builder yet, so it arrives as a column of no rows. What is
	// being checked is which aggregations refuse it, and that is decided before
	// a row is read.
	{dtype.List{Elem: dtype.Int64}, nil},
}

// TestAggTypeIsWhatTheAggregationGives runs every aggregation over a column of
// every type and checks the answer against the rule.
//
// This is the test the type rules rest on. A plan is checked against the rules
// and then run by the kernels, so a rule that promises a type the kernel does
// not give would be a plan that says one thing and a frame that holds another,
// which is worse than an error.
func TestAggTypeIsWhatTheAggregationGives(t *testing.T) {
	for _, tt := range everyType {
		t.Run(tt.dt.String(), func(t *testing.T) {
			c := columnOf(t, tt.dt, tt.vals)
			g := kernel.OneGroup(c.Len())

			for _, a := range aggregations {
				got, err := a.run(c, g)
				want, werr := a.give(tt.dt)

				switch {
				case err != nil && werr == nil:
					t.Errorf("%s of a %s column says %q, and the rule says it is a %s",
						a.name, tt.dt, err, want)
				case err == nil && werr != nil:
					t.Errorf("%s of a %s column gives a %s column, and the rule says %q",
						a.name, tt.dt, got.DType(), werr)
				case err != nil:
					if !strings.Contains(werr.Error(), tt.dt.String()) {
						t.Errorf("the rule for %s says %q, want it to name the type", a.name, werr)
					}
				case !dtype.Equal(got.DType(), want):
					t.Errorf("%s of a %s column gives a %s column, and the rule promised a %s",
						a.name, tt.dt, got.DType(), want)
				}
			}
		})
	}
}

// TestAggTypeOfADictionaryColumn is the encoding a column read out of a file
// often arrives in, and the one type whose rules are decided by what is behind
// it rather than by what it is stored as.
func TestAggTypeOfADictionaryColumn(t *testing.T) {
	c := dictColumn(t, array.OfStrings("GB", "JP", "US"), 0, 1, 2, 1)
	g := kernel.OneGroup(c.Len())

	// The smallest value of a dictionary of strings is a string, and it comes
	// back still encoded, so the type is the column's own.
	got, err := kernel.Min(c, g)
	if err != nil {
		t.Fatalf("Min: %v", err)
	}
	want, err := kernel.MinMaxType(c.DType())
	if err != nil {
		t.Fatalf("MinMaxType: %v", err)
	}
	if !dtype.Equal(got.DType(), want) {
		t.Errorf("the smallest value is a %s, and the rule promised a %s", got.DType(), want)
	}

	// There is no total of a column of text, whichever way it is stored.
	if _, err := kernel.SumType(c.DType()); err == nil {
		t.Error("a dictionary of strings has a sum")
	}
}

// TestHasOrderAgreesWithTheSort is the other half of [kernel.MinMaxType], since
// the order it asks about is the one the sort uses.
func TestHasOrderAgreesWithTheSort(t *testing.T) {
	for _, tt := range everyType {
		t.Run(tt.dt.String(), func(t *testing.T) {
			c := columnOf(t, tt.dt, tt.vals)

			_, err := kernel.SortIndex(kernel.Order{Column: c})
			if got := kernel.HasOrder(tt.dt); got != (err == nil) {
				t.Errorf("HasOrder says %v for a %s column, and sorting one says %v",
					got, tt.dt, err)
			}
		})
	}
}

// columnOf builds a column of the values, or an empty one of a type there is no
// builder for, which is how a list column gets into these tables.
func columnOf(t *testing.T, dt dtype.DataType, vals []any) *array.Chunked {
	t.Helper()

	if vals == nil {
		c, err := array.NewChunked(dt)
		if err != nil {
			t.Fatalf("NewChunked(%s): %v", dt, err)
		}
		return c
	}
	return col(t, dt, vals)
}
