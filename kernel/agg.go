package kernel

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// The aggregations all take the rows of a column and the groups those rows fall
// into, and give back one value per group, in group order. An aggregation over
// a whole column is an aggregation over [OneGroup], so there is one
// implementation of each rather than two.
//
// Missing values are skipped. That is what SQL does, what pandas does with its
// default, and what Polars does, and it is the only answer that lets a column
// with a hole in it be summed at all.
//
// NaN is not missing. It is a value, it is what a computation that went wrong
// produced, and hiding it would be hiding the thing worth knowing. So a sum
// that meets a NaN is NaN and a maximum that meets one is NaN, which is what
// the arithmetic says and what Polars does. pandas skips NaN because NaN is the
// only missing value it has, which is a limitation rather than a decision.
//
// Every one of them panics if the column and the groups are different lengths,
// since that is a mistake in the program rather than something the data did.

// Sum returns the total of each group.
//
// The result is wider than the input, so a column of int8 sums into int64 and a
// column of uint8 into uint64. That is what stops a total from overflowing at
// 127. An int64 column can still overflow, and it wraps rather than reporting
// an error, the same way Go's addition does everywhere else.
//
// A group with nothing to add up sums to zero rather than to nothing, which is
// what pandas and Polars both answer and what the arithmetic says: zero is what
// you get by adding up no numbers. [Mean] of the same group is missing, because
// there is no such number.
//
// Booleans sum as ones and zeros, so the total is how many are true.
//
// It reports an error for a column there is no sensible total of, which is
// everything that is not a number, a boolean or a duration. Adding two dates
// together is not a date and adding two strings is not this operation.
func Sum(c *array.Chunked, g *Groups) (*array.Chunked, error) {
	checkAgg("Sum", c, g)

	n := g.NumGroups()
	switch dt := c.DType(); dt.Kind() {
	case dtype.BoolKind:
		acc := make([]int64, n)
		eachBool(c, g, func(id int, v bool) {
			if v {
				acc[id]++
			}
		})
		return numbers(dtype.Int64, acc, nil), nil
	case dtype.Int8Kind:
		return numbers(dtype.Int64, sumInto[int8, int64](c, g, n), nil), nil
	case dtype.Int16Kind:
		return numbers(dtype.Int64, sumInto[int16, int64](c, g, n), nil), nil
	case dtype.Int32Kind:
		return numbers(dtype.Int64, sumInto[int32, int64](c, g, n), nil), nil
	case dtype.Int64Kind:
		return numbers(dtype.Int64, sumInto[int64, int64](c, g, n), nil), nil
	case dtype.Uint8Kind:
		return numbers(dtype.Uint64, sumInto[uint8, uint64](c, g, n), nil), nil
	case dtype.Uint16Kind:
		return numbers(dtype.Uint64, sumInto[uint16, uint64](c, g, n), nil), nil
	case dtype.Uint32Kind:
		return numbers(dtype.Uint64, sumInto[uint32, uint64](c, g, n), nil), nil
	case dtype.Uint64Kind:
		return numbers(dtype.Uint64, sumInto[uint64, uint64](c, g, n), nil), nil
	case dtype.Float32Kind:
		return numbers(dtype.Float64, sumInto[float32, float64](c, g, n), nil), nil
	case dtype.Float64Kind:
		return numbers(dtype.Float64, sumInto[float64, float64](c, g, n), nil), nil
	case dtype.DurationKind:
		// Two spans of time add up to a span of time, in the same unit, which
		// is the one temporal type this is meaningful for.
		return numbers(dt, sumInto[int64, int64](c, g, n), nil), nil
	default:
		return nil, noAgg("sum", dt)
	}
}

// SumType returns the type a [Sum] comes out as over a column of type dt, and
// an error for a column there is no sum of.
//
// It is exported for the same reason [ArithType] is. A plan has to say what
// each of its columns holds before there is a column to ask, so every rule that
// decides a type has to be readable from the types alone.
//
// The list is [Sum]'s own list written a second time, because the switch that
// picks the type is also the switch that picks the code, and Go has no way to
// read one off the other. What holds the two together is a test that runs every
// aggregation over a column of every type and checks that what came out is what
// was promised.
func SumType(dt dtype.DataType) (dtype.DataType, error) {
	switch dt.Kind() {
	case dtype.BoolKind, dtype.Int8Kind, dtype.Int16Kind, dtype.Int32Kind, dtype.Int64Kind:
		return dtype.Int64, nil
	case dtype.Uint8Kind, dtype.Uint16Kind, dtype.Uint32Kind, dtype.Uint64Kind:
		return dtype.Uint64, nil
	case dtype.Float32Kind, dtype.Float64Kind:
		return dtype.Float64, nil
	case dtype.DurationKind:
		return dt, nil
	default:
		return nil, noAgg("sum", dt)
	}
}

// Mean returns the average of each group, as a float64.
//
// It is the total divided by how many values there were, not by how many rows
// there were, so a column with holes in it averages the values it has. A group
// with no values at all is missing rather than zero, since the average of
// nothing is not a number and saying zero would be inventing one.
//
// Booleans average as ones and zeros, so the result is the fraction that are
// true.
//
// It reports an error for the same columns [Sum] does, and for durations, since
// the average of two spans is a span and this returns a float64.
func Mean(c *array.Chunked, g *Groups) (*array.Chunked, error) {
	checkAgg("Mean", c, g)

	if _, err := MeanType(c.DType()); err != nil {
		return nil, err
	}
	total, err := Sum(c, g)
	if err != nil {
		return nil, err
	}

	n := g.NumGroups()
	counts := countInto(c, g, n)
	out, valid := make([]float64, n), make([]bool, n)
	for i := range n {
		if counts[i] == 0 {
			continue
		}
		valid[i] = true
		switch total.DType().Kind() {
		case dtype.Uint64Kind:
			out[i] = float64(total.Value[uint64](i)) / float64(counts[i])
		case dtype.Float64Kind:
			out[i] = total.Value[float64](i) / float64(counts[i])
		default:
			out[i] = float64(total.Value[int64](i)) / float64(counts[i])
		}
	}
	return numbers(dtype.Float64, out, valid), nil
}

// MeanType returns the type a [Mean] comes out as over a column of type dt,
// which is always a float64, and an error for a column there is no mean of. It
// is exported for the reason [SumType] gives.
func MeanType(dt dtype.DataType) (dtype.DataType, error) {
	if dt.Kind() == dtype.DurationKind {
		// The average of two spans is a span rather than a float64, so this is
		// an operation that is missing rather than one with no meaning.
		return nil, fmt.Errorf("kernel: there is no mean of a %s column yet", dt)
	}
	return floatAggType("mean", dt)
}

// Count returns how many values each group has, not counting the missing ones.
// It is never itself missing, since a group with nothing in it has none of it.
func Count(c *array.Chunked, g *Groups) *array.Chunked {
	checkAgg("Count", c, g)
	return numbers(dtype.Int64, countInto(c, g, g.NumGroups()), nil)
}

// Size returns how many rows each group has, counting the rows whose value is
// missing. It is what pandas calls size next to count.
func Size(g *Groups) *array.Chunked {
	out := make([]int64, g.NumGroups())
	for i, s := range g.Sizes() {
		out[i] = int64(s)
	}
	return numbers(dtype.Int64, out, nil)
}

// Min returns the smallest value of each group, in the column's own type.
//
// The order is the one [SortIndex] uses, so strings compare by their bytes and
// NaN is larger than every number. A group with no values is missing.
//
// It reports an error for a column there is no order for, which today means the
// decimals and the nested types.
func Min(c *array.Chunked, g *Groups) (*array.Chunked, error) {
	checkAgg("Min", c, g)

	// Nulls last, so a null only wins when there is nothing else, and then the
	// answer is that the group has no smallest value.
	return extreme(c, g, Order{Column: c}, -1)
}

// Max returns the largest value of each group, in the column's own type. It is
// [Min] the other way up, with NaN winning over every number.
func Max(c *array.Chunked, g *Groups) (*array.Chunked, error) {
	checkAgg("Max", c, g)

	// Nulls first this time, for the same reason: a null has to be the thing
	// that loses, and here that means being at the small end.
	return extreme(c, g, Order{Column: c, NullsFirst: true}, 1)
}

// MinMaxType returns the type a [Min] or a [Max] comes out as over a column of
// type dt, which is the column's own type since the answer is one of the values
// that was there, and an error for a column there is no order for. It is
// exported for the reason [SumType] gives.
func MinMaxType(dt dtype.DataType) (dtype.DataType, error) {
	if !HasOrder(dt) {
		return nil, noOrder(dt)
	}
	return dt, nil
}

// extreme gathers the row of each group that wins under cmp, where winning
// means comparing want, which is -1 for the smallest and 1 for the largest.
func extreme(c *array.Chunked, g *Groups, o Order, want int) (*array.Chunked, error) {
	cmp, err := comparisonFor(o)
	if err != nil {
		return nil, err
	}

	best := make([]int, g.NumGroups())
	copy(best, g.FirstRows())
	for row, id := range g.IDs() {
		if cmp(row, best[id]) == want {
			best[id] = row
		}
	}
	return Take(c, best), nil
}

// First returns the first value of each group, skipping the missing ones, in
// the column's own type. A group whose values are all missing is missing.
//
// This is pandas' first rather than SQL's ANY_VALUE. The row that is first
// whether or not its value is there is [Groups.FirstRows], which a caller can
// gather at.
func First(c *array.Chunked, g *Groups) *array.Chunked {
	checkAgg("First", c, g)
	return present(c, g, false)
}

// Last returns the last value of each group, skipping the missing ones. It is
// [First] read backwards.
func Last(c *array.Chunked, g *Groups) *array.Chunked {
	checkAgg("Last", c, g)
	return present(c, g, true)
}

// present gathers the first row of each group whose value is there, or the last
// one when last is set.
func present(c *array.Chunked, g *Groups, last bool) *array.Chunked {
	at := make([]int, g.NumGroups())
	for i := range at {
		at[i] = -1
	}

	row := 0
	for _, a := range c.Chunks() {
		for i := range a.Len() {
			if !missing(a, i) {
				id := g.ids[row]
				if last || at[id] < 0 {
					at[id] = row
				}
			}
			row++
		}
	}

	// A group that never found one asks for row -1, which a gather answers with
	// a missing value, so there is nothing else to do about it here.
	return Take(c, at)
}

// sumInto adds the values of c up per group, reading them as T and totalling
// them as A.
func sumInto[T array.Numeric, A array.Numeric](c *array.Chunked, g *Groups, n int) []A {
	acc := make([]A, n)
	row := 0
	for _, a := range c.Chunks() {
		vs := a.Values[T]()
		switch a.NullCount() {
		case 0:
			for i := range a.Len() {
				acc[g.ids[row]] += A(vs[i])
				row++
			}
		default:
			for i := range a.Len() {
				if a.IsValid(i) {
					acc[g.ids[row]] += A(vs[i])
				}
				row++
			}
		}
	}
	return acc
}

// countInto counts the values of c that are there, per group.
func countInto(c *array.Chunked, g *Groups, n int) []int64 {
	acc := make([]int64, n)
	row := 0
	for _, a := range c.Chunks() {
		switch {
		case !anyMissing(a):
			for range a.Len() {
				acc[g.ids[row]]++
				row++
			}
		default:
			for i := range a.Len() {
				if !missing(a, i) {
					acc[g.ids[row]]++
				}
				row++
			}
		}
	}
	return acc
}

// eachBool calls f with the group and the value of every row of c that is
// there.
func eachBool(c *array.Chunked, g *Groups, f func(id int, v bool)) {
	row := 0
	for _, a := range c.Chunks() {
		bits, off := a.Bools(), a.Offset()
		for i := range a.Len() {
			if a.IsValid(i) {
				f(g.ids[row], bits.Get(off+i))
			}
			row++
		}
	}
}

// numbers builds a one chunk column of dt out of vs, where a false in valid
// means the value is missing. A nil valid means none of them are.
func numbers[T array.Numeric](dt dtype.DataType, vs []T, valid []bool) *array.Chunked {
	b, err := array.NewBuilder(dt)
	if err != nil {
		panic(fmt.Sprintf("kernel: building a %s column: %v", dt, err))
	}

	b.Grow(len(vs))
	for i, v := range vs {
		if len(valid) > 0 && !valid[i] {
			b.AppendNull()
			continue
		}
		b.Append(v)
	}

	out, err := array.NewChunked(dt, b.Finish())
	if err != nil {
		panic(fmt.Sprintf("kernel: building a %s column: %v", dt, err))
	}
	return out
}

// checkAgg reports the two ways an aggregation can be asked for something that
// makes no sense.
func checkAgg(name string, c *array.Chunked, g *Groups) {
	if c == nil {
		panic("kernel: " + name + " of a nil column")
	}
	if c.Len() != g.Len() {
		panic(fmt.Sprintf("kernel: %s of a column of %d values over %d rows of groups",
			name, c.Len(), g.Len()))
	}
}

// Var returns the variance of each group, as a float64.
//
// The divisor is the number of values less ddof. A ddof of one is the sample
// variance, which is what pandas and Polars both do when nobody says otherwise,
// and a ddof of zero is the population variance, which is what numpy does. The
// difference matters on small groups and disappears on large ones, and neither
// of them is right often enough to be the only one offered.
//
// A group with fewer values than the divisor needs is missing rather than
// infinite, so with the usual ddof of one a group of a single value has no
// variance.
//
// It reports an error for a column there is no variance of, which is everything
// that is not a number or a boolean.
func Var(c *array.Chunked, g *Groups, ddof int) (*array.Chunked, error) {
	checkAgg("Var", c, g)
	return variance("variance", c, g, ddof, false)
}

// VarType returns the type a [Var] comes out as over a column of type dt, which
// is always a float64, and an error for a column there is no variance of. It is
// exported for the reason [SumType] gives.
func VarType(dt dtype.DataType) (dtype.DataType, error) {
	return floatAggType("variance", dt)
}

// Std returns the standard deviation of each group, which is the square root of
// [Var] and takes the same ddof.
func Std(c *array.Chunked, g *Groups, ddof int) (*array.Chunked, error) {
	checkAgg("Std", c, g)
	return variance("standard deviation", c, g, ddof, true)
}

// StdType returns the type a [Std] comes out as over a column of type dt, which
// is a float64 wherever [VarType] is one. It is exported for the reason
// [SumType] gives.
func StdType(dt dtype.DataType) (dtype.DataType, error) {
	return floatAggType("standard deviation", dt)
}

// variance is the body of both, taking the square root at the end when asked.
func variance(name string, c *array.Chunked, g *Groups, ddof int, root bool) (*array.Chunked, error) {
	if ddof < 0 {
		return nil, fmt.Errorf("kernel: %s with a ddof of %d", name, ddof)
	}

	// Welford's method, which keeps a running mean and a running sum of squared
	// differences from it. The obvious way is to total the values and the
	// squares and subtract, and it loses every digit it has when the values are
	// large and close together, which is exactly what a column of prices or of
	// timestamps is. This costs a divide per value and is worth it.
	n := g.NumGroups()
	count := make([]float64, n)
	mean := make([]float64, n)
	m2 := make([]float64, n)
	if err := eachFloat(name, c, g, func(id int, v float64) {
		count[id]++
		d := v - mean[id]
		mean[id] += d / count[id]
		m2[id] += d * (v - mean[id])
	}); err != nil {
		return nil, err
	}

	out, valid := make([]float64, n), make([]bool, n)
	for i := range n {
		div := count[i] - float64(ddof)
		if div <= 0 {
			continue
		}
		valid[i] = true
		out[i] = m2[i] / div
		if root {
			out[i] = math.Sqrt(out[i])
		}
	}
	return numbers(dtype.Float64, out, valid), nil
}

// NUnique returns how many distinct values each group has, not counting the
// missing ones. It is what pandas calls nunique and SQL calls COUNT DISTINCT.
//
// Distinct means the same thing it means to [GroupBy], since it is the same
// encoding doing the deciding, so all the NaNs count as one value and negative
// zero counts as zero.
//
// It reports an error for the columns GroupBy refuses, which today means the
// nested types.
func NUnique(c *array.Chunked, g *Groups) (*array.Chunked, error) {
	checkAgg("NUnique", c, g)

	k, err := newKey(c)
	if err != nil {
		return nil, err
	}

	// One map for the whole column, keyed by the group and then the value, so
	// that a column of a million rows in one group and a column of a million
	// groups of one row cost the same. A map per group would be a million maps
	// in the second case.
	out := make([]int64, g.NumGroups())
	seen := make(map[string]struct{})
	var scratch []byte
	for row, id := range g.IDs() {
		scratch = binary.LittleEndian.AppendUint64(scratch[:0], uint64(id))
		before := len(scratch)
		scratch = k.appendRow(scratch, row)

		// A missing value is not a distinct value, and appendRow writes it as
		// the one byte that no value starts with.
		if len(scratch) == before+1 && scratch[before] == 0 {
			continue
		}
		if _, ok := seen[string(scratch)]; ok {
			continue
		}
		seen[string(scratch)] = struct{}{}
		out[id]++
	}
	return numbers(dtype.Int64, out, nil), nil
}

// NUniqueType returns the type an [NUnique] comes out as over a column of type
// dt, which is always a count, and an error for a column there is no key
// encoding for. It is exported for the reason [SumType] gives.
//
// Counting the distinct values of a column is grouping by it and asking how
// many groups there were, so what it can read is [GroupKeyType]'s answer rather
// than a list of its own.
func NUniqueType(dt dtype.DataType) (dtype.DataType, error) {
	if _, err := GroupKeyType(dt); err != nil {
		return nil, err
	}
	return dtype.Int64, nil
}

// floatAggType is the type of the aggregations that do their work in float64,
// which is a float64 for every column [eachFloat] can read and an error for the
// rest. What is a column there is no such aggregation of is one list rather
// than four, since the four read the values the same way.
func floatAggType(what string, dt dtype.DataType) (dtype.DataType, error) {
	switch dt.Kind() {
	case dtype.BoolKind,
		dtype.Int8Kind, dtype.Int16Kind, dtype.Int32Kind, dtype.Int64Kind,
		dtype.Uint8Kind, dtype.Uint16Kind, dtype.Uint32Kind, dtype.Uint64Kind,
		dtype.Float32Kind, dtype.Float64Kind:
		return dtype.Float64, nil
	default:
		return nil, noAgg(what, dt)
	}
}

// noAgg is the error for an aggregation over a column it has no meaning for.
// Both the kernel and the type rule report it, so that a mistake caught while a
// plan is being checked reads the same as the same mistake caught while it runs.
func noAgg(what string, dt dtype.DataType) error {
	return fmt.Errorf("kernel: there is no %s of a %s column", what, dt)
}

// eachFloat calls f with the group and the value of every row of c that is
// there, reading whatever the column holds as a float64.
//
// Reading an int64 as a float64 loses digits past the fifty third bit. That is
// what every one of these operations does in pandas and in Polars, since the
// answer is a float64 either way and the loss is in the last place of a number
// that is already an approximation.
func eachFloat(name string, c *array.Chunked, g *Groups, f func(id int, v float64)) error {
	switch c.DType().Kind() {
	case dtype.BoolKind:
		eachBool(c, g, func(id int, v bool) {
			if v {
				f(id, 1)
			} else {
				f(id, 0)
			}
		})
	case dtype.Int8Kind:
		eachNumber[int8](c, g, f)
	case dtype.Int16Kind:
		eachNumber[int16](c, g, f)
	case dtype.Int32Kind:
		eachNumber[int32](c, g, f)
	case dtype.Int64Kind:
		eachNumber[int64](c, g, f)
	case dtype.Uint8Kind:
		eachNumber[uint8](c, g, f)
	case dtype.Uint16Kind:
		eachNumber[uint16](c, g, f)
	case dtype.Uint32Kind:
		eachNumber[uint32](c, g, f)
	case dtype.Uint64Kind:
		eachNumber[uint64](c, g, f)
	case dtype.Float32Kind:
		eachNumber[float32](c, g, f)
	case dtype.Float64Kind:
		eachNumber[float64](c, g, f)
	default:
		return noAgg(name, c.DType())
	}
	return nil
}

// eachNumber is eachFloat once the type is known.
func eachNumber[T array.Numeric](c *array.Chunked, g *Groups, f func(id int, v float64)) {
	row := 0
	for _, a := range c.Chunks() {
		vs := a.Values[T]()
		switch a.NullCount() {
		case 0:
			for i := range a.Len() {
				f(g.ids[row], float64(vs[i]))
				row++
			}
		default:
			for i := range a.Len() {
				if a.IsValid(i) {
					f(g.ids[row], float64(vs[i]))
				}
				row++
			}
		}
	}
}
