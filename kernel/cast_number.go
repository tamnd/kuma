package kernel

import (
	"math"
	"strconv"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// number is one value on its way from one numeric type to another.
//
// Going through a single struct rather than instantiating a conversion for
// every pair of Go types is a hundred functions the reader does not have to
// trust. It costs a branch and a copy per value, which is the trade the package
// comment says this layer is making until the fast version exists to be checked
// against it.
//
// The three fields are not interchangeable. A uint64 above the signed range and
// an int64 below zero both stop being themselves if they go through the other
// field, and a float64 cannot hold every int64 exactly, so the value stays in
// whichever field the source put it in and every check knows which one it is
// reading.
type number struct {
	i int64
	u uint64
	f float64
	k numKind
}

type numKind uint8

const (
	numInt numKind = iota
	numUint
	numFloat
)

// String prints the value the way an error message should show it.
func (n number) String() string {
	switch n.k {
	case numInt:
		return strconv.FormatInt(n.i, 10)
	case numUint:
		return strconv.FormatUint(n.u, 10)
	default:
		return strconv.FormatFloat(n.f, 'g', -1, 64)
	}
}

// readNumber reads value i of a column whose values are numbers, or are stored
// as numbers, or are booleans.
//
// It panics on anything else, which is a bug in the dispatch above rather than
// something the data did.
func readNumber(a *array.Array, i int) number {
	switch a.DType().Kind() {
	case dtype.BoolKind:
		if a.Bool(i) {
			return number{i: 1}
		}
		return number{i: 0}
	case dtype.Int8Kind:
		return number{i: int64(a.Value[int8](i))}
	case dtype.Int16Kind:
		return number{i: int64(a.Value[int16](i))}
	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		return number{i: int64(a.Value[int32](i))}
	case dtype.Int64Kind, dtype.Date64Kind, dtype.Time64Kind,
		dtype.TimestampKind, dtype.DurationKind:
		return number{i: a.Value[int64](i)}
	case dtype.Uint8Kind:
		return number{u: uint64(a.Value[uint8](i)), k: numUint}
	case dtype.Uint16Kind:
		return number{u: uint64(a.Value[uint16](i)), k: numUint}
	case dtype.Uint32Kind:
		return number{u: uint64(a.Value[uint32](i)), k: numUint}
	case dtype.Uint64Kind:
		return number{u: a.Value[uint64](i), k: numUint}
	case dtype.Float32Kind:
		return number{f: float64(a.Value[float32](i)), k: numFloat}
	case dtype.Float64Kind:
		return number{f: a.Value[float64](i), k: numFloat}
	default:
		panic("kernel: readNumber on a " + a.DType().String() + " column")
	}
}

// appendNumber converts n to a column of kind k and appends it, and reports
// whether the value fitted.
//
// A temporal kind is here with the integer of its own width, since the stored
// count is what a cast to or from a number is about.
func appendNumber(b *array.Builder, k dtype.Kind, n number) bool {
	switch k {
	case dtype.Int8Kind:
		return appendSigned[int8](b, n, math.MinInt8, math.MaxInt8)
	case dtype.Int16Kind:
		return appendSigned[int16](b, n, math.MinInt16, math.MaxInt16)
	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		return appendSigned[int32](b, n, math.MinInt32, math.MaxInt32)
	case dtype.Int64Kind, dtype.Date64Kind, dtype.Time64Kind,
		dtype.TimestampKind, dtype.DurationKind:
		return appendSigned[int64](b, n, math.MinInt64, math.MaxInt64)
	case dtype.Uint8Kind:
		return appendUnsigned[uint8](b, n, math.MaxUint8)
	case dtype.Uint16Kind:
		return appendUnsigned[uint16](b, n, math.MaxUint16)
	case dtype.Uint32Kind:
		return appendUnsigned[uint32](b, n, math.MaxUint32)
	case dtype.Uint64Kind:
		return appendUnsigned[uint64](b, n, math.MaxUint64)
	case dtype.Float32Kind:
		v, ok := toFloat32(n)
		if ok {
			b.Append(v)
		}
		return ok
	case dtype.Float64Kind:
		b.Append(toFloat64(n))
		return true
	default:
		panic("kernel: appendNumber to a " + k.String() + " column")
	}
}

// appendSigned converts n to a signed integer between lo and hi and appends it.
//
// The float bound is the power of two just past the range rather than hi
// itself, because float64(math.MaxInt64) rounds up to 2^63 and comparing
// against a limit that rounded the wrong way lets one value through that does
// not fit. The power of two is exact in a float64 for every width up to and
// including 64, so the comparison is exact.
func appendSigned[T array.Numeric](b *array.Builder, n number, lo, hi int64) bool {
	switch n.k {
	case numInt:
		if n.i < lo || n.i > hi {
			return false
		}
		b.Append(T(n.i))
	case numUint:
		if n.u > uint64(hi) {
			return false
		}
		b.Append(T(n.u))
	default:
		t := math.Trunc(n.f)
		limit := float64(hi) + 1
		if math.IsNaN(t) || t < float64(lo) || t >= limit {
			return false
		}
		b.Append(T(t))
	}
	return true
}

// appendUnsigned converts n to an unsigned integer no larger than hi and
// appends it. The float bound is the power of two past the range, for the
// reason given above.
func appendUnsigned[T array.Numeric](b *array.Builder, n number, hi uint64) bool {
	switch n.k {
	case numInt:
		if n.i < 0 || uint64(n.i) > hi {
			return false
		}
		b.Append(T(n.i))
	case numUint:
		if n.u > hi {
			return false
		}
		b.Append(T(n.u))
	default:
		t := math.Trunc(n.f)
		limit := float64(hi) + 1
		if math.IsNaN(t) || t < 0 || t >= limit {
			return false
		}
		b.Append(T(t))
	}
	return true
}

// toFloat32 narrows n, and reports whether the value survived.
//
// Precision is not the question here. Every int64 above 2^24 loses digits on the
// way into a float32 and that is what asking for a float32 means. What does not
// survive is a finite number that comes out infinite, and testing the result
// rather than the input is what makes that exact at the edge of the range.
//
// An infinity or a NaN that was already one goes through as itself, since it
// arrived as an answer about the data and not as an overflow.
func toFloat32(n number) (float32, bool) {
	switch n.k {
	case numInt:
		return float32(n.i), true
	case numUint:
		return float32(n.u), true
	default:
		v := float32(n.f)
		if math.IsInf(float64(v), 0) && !math.IsInf(n.f, 0) {
			return 0, false
		}
		return v, true
	}
}

// toFloat64 widens n, which always works.
func toFloat64(n number) float64 {
	switch n.k {
	case numInt:
		return float64(n.i)
	case numUint:
		return float64(n.u)
	default:
		return n.f
	}
}
