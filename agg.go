package kuma

import (
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// Aggregation is one thing to work out about each group: which column to read,
// what to do to it, and what to call the answer.
//
// The zero value is not usable. Build one with [Sum], [Mean], [Min], [Max],
// [Count], [Size], [First], [Last], [Var], [Std], [Median], [Quantile] or
// [NUnique], and rename it with [Aggregation.As].
//
// The type is a value with unexported fields rather than an interface, because
// the set of aggregations is closed and known here. An interface would let a
// caller write one of their own, and an aggregation that the fast kernels and
// the query planner cannot see inside is an aggregation that neither of them
// can do anything with. When user defined aggregations arrive they will arrive
// as their own thing, with the cost written on the label.
type Aggregation struct {
	op  aggOp
	col string
	as  string

	// ddof is the divisor adjustment of Var and Std, and q and how are the
	// arguments of Quantile. They are here rather than in a variant per
	// aggregation because two ints and a float are cheaper than an interface
	// and a type switch, and this struct is copied rather than allocated.
	ddof int
	q    float64
	how  Interpolation
}

// aggOp is which aggregation an [Aggregation] is. It is [plan.AggFunc] under
// another name, because the list of aggregations belongs with the plan that has
// to be able to reason about them, and one list is better than two that have to
// be kept in step.
type aggOp = plan.AggFunc

const (
	opSum      = plan.AggSum
	opMean     = plan.AggMean
	opMin      = plan.AggMin
	opMax      = plan.AggMax
	opCount    = plan.AggCount
	opSize     = plan.AggSize
	opFirst    = plan.AggFirst
	opLast     = plan.AggLast
	opVar      = plan.AggVar
	opStd      = plan.AggStd
	opMedian   = plan.AggMedian
	opQuantile = plan.AggQuantile
	opNUnique  = plan.AggNUnique
)

// Sum returns the total of the named column in each group.
//
// The result widens: the sums of the signed columns are int64, of the unsigned
// ones uint64, of the floats float64, and of a boolean column the number of
// true values. A duration keeps its unit, because the total of a run of
// durations is a duration. An int64 total that overflows wraps, the way Go
// addition does.
//
// The total of an empty group is zero rather than missing, since that is what
// adding nothing up gives.
func Sum(col string) Aggregation { return Aggregation{op: opSum, col: col} }

// Mean returns the average of the named column in each group, as a float64.
//
// The average of an empty group is missing, since there is no number that is
// the average of nothing.
func Mean(col string) Aggregation { return Aggregation{op: opMean, col: col} }

// Min returns the smallest value of the named column in each group, keeping the
// column's own type.
//
// Missing values are skipped, so the smallest of a group that is all missing is
// missing. NaN is a value rather than a missing one and it sorts after every
// number, so it is only the smallest when it is the only value.
func Min(col string) Aggregation { return Aggregation{op: opMin, col: col} }

// Max returns the largest value of the named column in each group, keeping the
// column's own type.
//
// Missing values are skipped. NaN sorts after every number, so a group with a
// NaN in it has a NaN for its largest value, which is the honest answer to what
// the largest of these is when one of them is not a number.
func Max(col string) Aggregation { return Aggregation{op: opMax, col: col} }

// Count returns how many values of the named column are there in each group,
// not counting the missing ones.
//
// It is the count of values and [Size] is the count of rows. They differ by
// exactly the nulls, and the difference is usually the thing you wanted to know.
func Count(col string) Aggregation { return Aggregation{op: opCount, col: col} }

// Size returns how many rows each group has, missing values included.
//
// It reads no column, so it is the one aggregation that cannot fail on a type,
// and it is called "size" unless [Aggregation.As] says otherwise.
func Size() Aggregation { return Aggregation{op: opSize, as: "size"} }

// First returns the first value of the named column in each group that is
// there, in the order the rows were in.
//
// Missing values are skipped, so this is the first value and not the value of
// the first row. Sort the frame first if the word first is supposed to mean
// something other than the order the rows arrived in.
func First(col string) Aggregation { return Aggregation{op: opFirst, col: col} }

// Last returns the last value of the named column in each group that is there.
func Last(col string) Aggregation { return Aggregation{op: opLast, col: col} }

// Var returns the variance of the named column in each group, as a float64.
//
// The divisor is the number of values less ddof. A ddof of one is the sample
// variance, which is what pandas and Polars both give when nobody says
// otherwise, and a ddof of zero is the population variance, which is what numpy
// gives. A group with fewer values than the divisor wants is missing rather
// than infinite.
func Var(col string, ddof int) Aggregation {
	return Aggregation{op: opVar, col: col, ddof: ddof}
}

// Std returns the standard deviation of the named column in each group, which
// is the square root of [Var] and takes the same ddof.
func Std(col string, ddof int) Aggregation {
	return Aggregation{op: opStd, col: col, ddof: ddof}
}

// Median returns the middle value of the named column in each group, as a
// float64. It is [Quantile] at a half, interpolated linearly.
func Median(col string) Aggregation { return Aggregation{op: opMedian, col: col} }

// Quantile returns the value q of the way through the sorted values of the
// named column in each group, as a float64.
//
// A q of a half is the median, 0.95 is the ninety fifth percentile, zero is the
// smallest value and one is the largest. How says what to do when q lands
// between two values, which on a small group it nearly always does.
//
// It reports an error when the aggregation runs if q is below zero, above one
// or not a number, or if how is not one of the five.
func Quantile(col string, q float64, how Interpolation) Aggregation {
	return Aggregation{op: opQuantile, col: col, q: q, how: how}
}

// NUnique returns how many distinct values of the named column each group has,
// not counting the missing ones. It is what pandas calls nunique and SQL calls
// COUNT DISTINCT.
//
// Distinct means what it means to [Frame.GroupBy], since it is the same
// encoding doing the deciding, so every NaN counts as one value and negative
// zero counts as the same value as zero.
func NUnique(col string) Aggregation { return Aggregation{op: opNUnique, col: col} }

// As renames the result column.
//
// Without it an aggregation is called after the column it reads, so two
// aggregations of one column need at least one of them named. With it the query
// reads like the table it produces:
//
//	f.Agg(kuma.Sum("qty").As("total"), kuma.Mean("price").As("avg"))
func (a Aggregation) As(name string) Aggregation {
	a.as = name
	return a
}

// Name returns what the result column will be called.
func (a Aggregation) Name() string {
	if a.as != "" {
		return a.as
	}
	return a.col
}

// String returns the aggregation as it would be written, which is what an error
// message about it should say.
func (a Aggregation) String() string {
	s := a.op.String() + "(" + a.col
	switch a.op {
	case opVar, opStd:
		s = fmt.Sprintf("%s, %d", s, a.ddof)
	case opQuantile:
		s = fmt.Sprintf("%s, %v, %s", s, a.q, a.how)
	case opSum, opMean, opMin, opMax, opCount, opSize, opFirst, opLast,
		opMedian, opNUnique:
	}
	s += ")"
	if a.as != "" {
		s += " as " + a.as
	}
	return s
}

// run works the aggregation out over a grouping and returns the result column.
func (a Aggregation) run[S any](g *GroupedFrame[S]) (Column, error) {
	if a.op == opSize {
		return Column{name: a.Name(), data: kernel.Size(g.group)}, nil
	}

	k, ok := g.frame.index[a.col]
	if !ok {
		return Column{}, noColumn(a.op.String(), a.col, g.frame.Names())
	}
	c := g.frame.cols[k].data

	var (
		out *array.Chunked
		err error
	)
	switch a.op {
	case opSum:
		out, err = kernel.Sum(c, g.group)
	case opMean:
		out, err = kernel.Mean(c, g.group)
	case opMin:
		out, err = kernel.Min(c, g.group)
	case opMax:
		out, err = kernel.Max(c, g.group)
	case opCount:
		out = kernel.Count(c, g.group)
	case opFirst:
		out = kernel.First(c, g.group)
	case opLast:
		out = kernel.Last(c, g.group)
	case opVar:
		out, err = kernel.Var(c, g.group, a.ddof)
	case opStd:
		out, err = kernel.Std(c, g.group, a.ddof)
	case opMedian:
		out, err = kernel.Median(c, g.group)
	case opQuantile:
		out, err = kernel.Quantile(c, g.group, a.q, a.how)
	case opNUnique:
		out, err = kernel.NUnique(c, g.group)
	default:
		// opSize is handled above and there is nothing else.
		panic(fmt.Sprintf("kuma: aggregation %d has no implementation", a.op))
	}
	if err != nil {
		return Column{}, err
	}
	return Column{name: a.Name(), data: out}, nil
}
