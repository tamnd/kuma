package kernel

import (
	"fmt"
	"math"
	"slices"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Interpolation says what a quantile does when it lands between two values.
//
// These are the five pandas has, under the same names numpy gives them, and
// they exist because there is no single right answer. A median of an even
// number of values is the obvious case: [Linear] and [Midpoint] average the two
// in the middle, [Lower] and [Higher] pick one of them, and [Nearest] picks
// whichever the position is closer to.
type Interpolation int

// The ways a quantile can land between two values.
const (
	// Linear walks the fraction of the way from the lower value to the higher
	// one. It is what pandas and numpy do when nobody says otherwise.
	Linear Interpolation = iota

	// Lower takes the value below the position.
	Lower

	// Higher takes the value above the position.
	Higher

	// Nearest takes whichever of the two the position is closer to, and when it
	// is exactly between them takes the one at the even index, which is what
	// numpy's rounding does.
	Nearest

	// Midpoint takes the average of the two, whatever the position is between
	// them.
	Midpoint
)

// String returns the pandas name of the interpolation.
func (i Interpolation) String() string {
	switch i {
	case Linear:
		return "linear"
	case Lower:
		return "lower"
	case Higher:
		return "higher"
	case Nearest:
		return "nearest"
	case Midpoint:
		return "midpoint"
	default:
		return fmt.Sprintf("Interpolation(%d)", int(i))
	}
}

// Median returns the middle value of each group, as a float64.
//
// It is [Quantile] at a half with [Linear] interpolation, which is the pandas
// default and means an even number of values averages the two in the middle.
func Median(c *array.Chunked, g *Groups) (*array.Chunked, error) {
	checkAgg("Median", c, g)
	return quantile("median", c, g, 0.5, Linear)
}

// MedianType returns the type a [Median] comes out as over a column of type dt,
// which is always a float64, and an error for a column there is no median of.
// It is exported for the reason [SumType] gives.
func MedianType(dt dtype.DataType) (dtype.DataType, error) {
	return floatAggType("median", dt)
}

// Quantile returns the value at position q of each group, as a float64, where q
// runs from zero for the smallest to one for the largest.
//
// The values of a group are put in order and then read at q of the way along,
// which for a q that does not land on a value is decided by how. A group with
// no values is missing.
//
// NaN is a value and sorts after every number, the same as everywhere else
// here, so a group with a NaN in it has one at the top end and a quantile near
// one will find it. That is the answer that says the computation went wrong
// rather than the one that hides it.
//
// It reports an error if q is outside zero to one, if how is not one of the
// five, or if the column is not numeric.
func Quantile(c *array.Chunked, g *Groups, q float64, how Interpolation) (*array.Chunked, error) {
	checkAgg("Quantile", c, g)
	return quantile("quantile", c, g, q, how)
}

// QuantileType returns the type a [Quantile] comes out as over a column of type
// dt, which is always a float64, and an error for a column there is no quantile
// of. It says nothing about q, which is a number the caller passes and not a
// type, and which [Quantile] checks when it is given one. It is exported for
// the reason [SumType] gives.
func QuantileType(dt dtype.DataType) (dtype.DataType, error) {
	return floatAggType("quantile", dt)
}

// quantile is the body of both, with name naming whichever the caller asked for
// so that an error message says the operation they wrote.
func quantile(name string, c *array.Chunked, g *Groups, q float64, how Interpolation) (*array.Chunked, error) {
	if math.IsNaN(q) || q < 0 || q > 1 {
		return nil, fmt.Errorf("kernel: %s at %v, which is not between 0 and 1", name, q)
	}
	if how < Linear || how > Midpoint {
		return nil, fmt.Errorf("kernel: %s with %s", name, how)
	}

	// One slice per group, sized by how many rows the group has, which is an
	// upper bound because some of them may be missing. This is the memory a
	// quantile costs and there is no way around it: the values have to be in
	// order and they arrive in the order the rows are in.
	parts := make([][]float64, g.NumGroups())
	for i, n := range g.Sizes() {
		parts[i] = make([]float64, 0, n)
	}
	if err := eachFloat(name, c, g, func(id int, v float64) {
		parts[id] = append(parts[id], v)
	}); err != nil {
		return nil, err
	}

	out, valid := make([]float64, len(parts)), make([]bool, len(parts))
	for i, vs := range parts {
		if len(vs) == 0 {
			continue
		}
		slices.SortFunc(vs, compareFloat)
		valid[i] = true
		out[i] = at(vs, q, how)
	}
	return numbers(dtype.Float64, out, valid), nil
}

// at reads the sorted values at q of the way along.
func at(vs []float64, q float64, how Interpolation) float64 {
	// The position runs from 0 at the first value to len-1 at the last, which
	// is what numpy and pandas mean by a quantile and is why a q of one lands
	// exactly on the largest value rather than off the end.
	pos := q * float64(len(vs)-1)
	lo := int(math.Floor(pos))
	hi := int(math.Ceil(pos))
	if lo == hi {
		return vs[lo]
	}
	frac := pos - float64(lo)

	switch how {
	case Lower:
		return vs[lo]
	case Higher:
		return vs[hi]
	case Midpoint:
		return vs[lo] + (vs[hi]-vs[lo])/2
	case Nearest:
		switch {
		case frac < 0.5:
			return vs[lo]
		case frac > 0.5:
			return vs[hi]
		case lo%2 == 0:
			return vs[lo]
		default:
			return vs[hi]
		}
	default:
		// Linear. Written as a walk from the lower value rather than as a
		// weighted sum of the two, because the walk cannot drift outside the
		// pair it is walking between and the weighted sum can.
		return vs[lo] + frac*(vs[hi]-vs[lo])
	}
}
