package kernel

import (
	"bytes"
	"cmp"
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// CompareOp is one of the six comparisons.
type CompareOp uint8

// The comparisons.
const (
	OpEq CompareOp = iota // equal
	OpNe                  // not equal
	OpLt                  // less than
	OpLe                  // less than or equal
	OpGt                  // greater than
	OpGe                  // greater than or equal
)

var compareNames = [...]string{
	OpEq: "==",
	OpNe: "!=",
	OpLt: "<",
	OpLe: "<=",
	OpGt: ">",
	OpGe: ">=",
}

// String returns the operator as it is written in Go.
func (o CompareOp) String() string {
	if int(o) >= len(compareNames) {
		return fmt.Sprintf("CompareOp(%d)", uint8(o))
	}
	return compareNames[o]
}

// Compare returns a boolean column saying, for each row, whether the value in a
// stands in the relation op to the value in b.
//
// A column of one value on either side is that value against every row of the
// other, which is how a comparison against a literal is written. Two columns of
// different lengths that are not that panic.
//
// A missing value compares to nothing, so a null on either side gives a null
// rather than a false. That is what SQL does and what Polars does, and it is
// what makes Filter drop the row: a row nobody can say belongs in the result
// does not go in it. The two columns have to have a type in common, which is
// [dtype.Coerce]'s question, and an int64 column against a float64 column is an
// error there rather than a quiet upcast.
//
// NaN is unordered, so every comparison against it is false except !=, which is
// true. That is the IEEE rule and it is what the Go operators do, so a
// comparison here gives the same answer as the same comparison written out over
// the values. Polars decided the other way and calls NaN equal to itself.
// The pandas answer is the IEEE one, by way of numpy.
//
// A dictionary encoded column is compared through its indices, so a column of
// country codes read out of a parquet file answers the same question the same
// way as one that was never encoded, and neither side has to be decoded to ask
// it. A dictionary entry that is itself null is a missing value like any other.
//
// Not yet: the decimals, the intervals and the nested types, which have no
// order here for the same reason [SortIndex] gives.
func Compare(a, b *array.Chunked, op CompareOp) (*array.Chunked, error) {
	if a == nil || b == nil {
		panic("kernel: compare of a nil column")
	}
	if int(op) >= len(compareNames) {
		panic(fmt.Sprintf("kernel: compare with an unknown operator %d", uint8(op)))
	}
	n, fixedA, fixedB := binaryLen("compare", a, b)

	dt, err := dtype.Coerce(a.DType(), b.DType())
	if err != nil {
		return nil, fmt.Errorf("kernel: cannot compare %s with %s: %w", a.DType(), b.DType(), err)
	}

	// Dictionary encoding is storage rather than meaning, so what is compared
	// is the values behind the indices. Coerce keeps the encoding when both
	// sides have one, because a gather and a group by would rather have it, and
	// a comparison is the operation that has to see through it.
	if d, ok := dt.(dtype.Dictionary); ok {
		dt = d.Value
	}

	// A null column has no values to compare, so every answer is that nobody
	// knows. It is the one type that gets this far without a comparison.
	if dt.Kind() == dtype.NullKind {
		return nulls(dtype.Bool, n), nil
	}

	rel, err := comparator(dt)
	if err != nil {
		return nil, err
	}

	out := builder(dtype.Bool)
	out.Grow(n)

	ca, cb := newDictCursor(a, fixedA), newDictCursor(b, fixedB)
	for range n {
		x, i, okx := ca.next()
		y, j, oky := cb.next()
		if !okx || !oky {
			out.AppendNull()
			continue
		}
		r, ordered := rel(x, i, y, j)
		out.AppendBool(holds(op, r, ordered))
	}
	return one(dtype.Bool, out.Finish()), nil
}

// holds says whether a three way comparison satisfies the operator. An
// unordered pair, which is what a NaN makes, satisfies only !=.
func holds(op CompareOp, r int, ordered bool) bool {
	if !ordered {
		return op == OpNe
	}
	switch op {
	case OpEq:
		return r == 0
	case OpNe:
		return r != 0
	case OpLt:
		return r < 0
	case OpLe:
		return r <= 0
	case OpGt:
		return r > 0
	default:
		return r >= 0
	}
}

// relation compares value i of x with value j of y, both of which are known to
// be present. It reports whether the two are ordered at all, which is false
// only for a pair with a NaN in it.
type relation func(x *array.Array, i int, y *array.Array, j int) (r int, ordered bool)

// comparator returns the relation for a type, or an error for a type this
// package has no comparison for yet.
func comparator(dt dtype.DataType) (relation, error) {
	switch dt.Kind() {
	case dtype.BoolKind:
		return func(x *array.Array, i int, y *array.Array, j int) (int, bool) {
			a, b := x.Bool(i), y.Bool(j)
			switch {
			case a == b:
				return 0, true
			case b:
				return -1, true
			default:
				return 1, true
			}
		}, nil
	case dtype.Int8Kind:
		return ordered[int8](), nil
	case dtype.Int16Kind:
		return ordered[int16](), nil
	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		return ordered[int32](), nil
	case dtype.Int64Kind, dtype.Date64Kind, dtype.Time64Kind,
		dtype.TimestampKind, dtype.DurationKind:
		return ordered[int64](), nil
	case dtype.Uint8Kind:
		return ordered[uint8](), nil
	case dtype.Uint16Kind:
		return ordered[uint16](), nil
	case dtype.Uint32Kind:
		return ordered[uint32](), nil
	case dtype.Uint64Kind:
		return ordered[uint64](), nil
	case dtype.Float32Kind:
		return floating[float32](), nil
	case dtype.Float64Kind:
		return floating[float64](), nil
	case dtype.StringKind, dtype.BinaryKind, dtype.LargeStringKind,
		dtype.LargeBinaryKind, dtype.FixedSizeBinaryKind:
		return func(x *array.Array, i int, y *array.Array, j int) (int, bool) {
			return bytes.Compare(x.Bytes(i), y.Bytes(j)), true
		}, nil
	default:
		return nil, fmt.Errorf("kernel: there is no comparison for a %s column yet", dt)
	}
}

// ordered is the relation for the types where every pair of values has an
// order, which is every fixed width type except the two floats.
func ordered[T array.Numeric]() relation {
	return func(x *array.Array, i int, y *array.Array, j int) (int, bool) {
		return cmp.Compare(x.Value[T](i), y.Value[T](j)), true
	}
}

// floating is the relation for the floats, where a NaN is unordered against
// everything including itself.
//
// This is not [cmp.Compare], which puts NaN below every number so that a sort
// has somewhere to put it. A sort needs a total order and a comparison needs
// the truth, and those are two different questions about the same pair.
func floating[T float32 | float64]() relation {
	return func(x *array.Array, i int, y *array.Array, j int) (int, bool) {
		a, b := x.Value[T](i), y.Value[T](j)
		switch {
		case a < b:
			return -1, true
		case a > b:
			return 1, true
		case a == b:
			return 0, true
		default:
			return 0, false
		}
	}
}

// nulls returns a column of n missing values of the given type.
func nulls(dt dtype.DataType, n int) *array.Chunked {
	b := builder(dt)
	b.AppendNulls(n)
	return one(dt, b.Finish())
}

// builder returns a builder for a type this package has already decided it can
// write, so a type it cannot build is a bug here rather than a condition.
func builder(dt dtype.DataType) *array.Builder {
	b, err := array.NewBuilder(dt)
	if err != nil {
		panic("kernel: " + err.Error())
	}
	return b
}

// one returns a column of a single chunk. The builder was made for the type, so
// the two agree and the error cannot happen.
func one(dt dtype.DataType, a *array.Array) *array.Chunked {
	c, err := array.NewChunked(dt, a)
	if err != nil {
		panic("kernel: " + err.Error())
	}
	return c
}
