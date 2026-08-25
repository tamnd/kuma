package kernel

import (
	"errors"
	"fmt"
	"math"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// ArithOp is one of the five arithmetic operators.
type ArithOp uint8

// The operators.
const (
	OpAdd ArithOp = iota // addition
	OpSub                // subtraction
	OpMul                // multiplication
	OpDiv                // division
	OpMod                // remainder
)

var arithNames = [...]string{
	OpAdd: "+",
	OpSub: "-",
	OpMul: "*",
	OpDiv: "/",
	OpMod: "%",
}

// String returns the operator as it is written in Go.
func (o ArithOp) String() string {
	if int(o) >= len(arithNames) {
		return fmt.Sprintf("ArithOp(%d)", uint8(o))
	}
	return arithNames[o]
}

// Arith returns a column holding, for each row, the value in a combined with
// the value in b under the operator op.
//
// The result has the type the two columns have in common, which for arithmetic
// means the same type on both sides. An int64 column plus a float64 column is
// an error naming the cast to write, not a quiet upcast, for the reason
// [dtype.Coerce] gives. A column of one value on either side is that value
// against every row of the other, which is how adding a literal is written.
//
// The answer is the one the Go operator gives on the same two values, which is
// the rule worth stating once because it settles three questions that other
// libraries answer differently. Integer arithmetic wraps rather than widening
// or failing. Integer division truncates toward zero, so 7 / 2 is 3 and not 3.5,
// and a caller who wants the fraction casts first. Float division by zero gives
// an infinity or a NaN, because that is what a float64 means.
//
// Integer division and remainder by zero have no answer at all, so they are an
// error naming the row, in the same shape as the one a cast that does not fit
// gives.
//
// A missing value on either side gives a missing value out, since there is
// nothing to add.
//
// Not yet: the decimals, where the scale of the result is a decision rather
// than a lookup, and the temporal types, where a timestamp plus a duration has
// to reconcile two units.
func Arith(a, b *array.Chunked, op ArithOp) (*array.Chunked, error) {
	if a == nil || b == nil {
		panic("kernel: arithmetic on a nil column")
	}
	if int(op) >= len(arithNames) {
		panic(fmt.Sprintf("kernel: arithmetic with an unknown operator %d", uint8(op)))
	}
	n, fixedA, fixedB := binaryLen("arithmetic", a, b)

	dt, err := dtype.Coerce(a.DType(), b.DType())
	if err != nil {
		return nil, fmt.Errorf("kernel: cannot combine %s with %s: %w", a.DType(), b.DType(), err)
	}
	if dt.Kind() == dtype.NullKind {
		return nulls(dt, n), nil
	}

	apply, err := arithmetic(dt, op)
	if err != nil {
		return nil, err
	}

	out := builder(dt)
	out.Grow(n)

	ca, cb := newCursor(a, fixedA), newCursor(b, fixedB)
	for row := range n {
		x, i := ca.next()
		y, j := cb.next()
		if x.IsNull(i) || y.IsNull(j) {
			out.AppendNull()
			continue
		}
		if err := apply(out, x, i, y, j); err != nil {
			return nil, fmt.Errorf("kernel: %s at row %d: %w", op, row, err)
		}
	}
	return one(dt, out.Finish()), nil
}

// ErrDivideByZero is what dividing by zero in an integer column gives.
var ErrDivideByZero = errors.New("division by zero")

// operate combines value i of x with value j of y, both of which are known to
// be present, and appends the result.
type operate func(out *array.Builder, x *array.Array, i int, y *array.Array, j int) error

// arithmetic returns the operation for a type, or an error for a type there is
// no arithmetic for.
func arithmetic(dt dtype.DataType, op ArithOp) (operate, error) {
	switch dt.Kind() {
	case dtype.Int8Kind:
		return integer[int8](op), nil
	case dtype.Int16Kind:
		return integer[int16](op), nil
	case dtype.Int32Kind:
		return integer[int32](op), nil
	case dtype.Int64Kind:
		return integer[int64](op), nil
	case dtype.Uint8Kind:
		return integer[uint8](op), nil
	case dtype.Uint16Kind:
		return integer[uint16](op), nil
	case dtype.Uint32Kind:
		return integer[uint32](op), nil
	case dtype.Uint64Kind:
		return integer[uint64](op), nil
	case dtype.Float32Kind:
		return floats[float32](op), nil
	case dtype.Float64Kind:
		return floats[float64](op), nil
	default:
		return nil, fmt.Errorf("kernel: there is no arithmetic for a %s column yet", dt)
	}
}

// whole is every integer type a column can hold.
type whole interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
}

// integer is the operation for the integer types, where dividing by zero is
// the one pair with no answer.
func integer[T whole](op ArithOp) operate {
	return func(out *array.Builder, x *array.Array, i int, y *array.Array, j int) error {
		a, b := x.Value[T](i), y.Value[T](j)
		switch op {
		case OpAdd:
			out.Append(a + b)
		case OpSub:
			out.Append(a - b)
		case OpMul:
			out.Append(a * b)
		case OpDiv:
			if b == 0 {
				return ErrDivideByZero
			}
			out.Append(a / b)
		default:
			if b == 0 {
				return ErrDivideByZero
			}
			out.Append(a % b)
		}
		return nil
	}
}

// floats is the operation for the two float types, where every pair has an
// answer and some of those answers are infinities.
func floats[T float32 | float64](op ArithOp) operate {
	return func(out *array.Builder, x *array.Array, i int, y *array.Array, j int) error {
		a, b := x.Value[T](i), y.Value[T](j)
		switch op {
		case OpAdd:
			out.Append(a + b)
		case OpSub:
			out.Append(a - b)
		case OpMul:
			out.Append(a * b)
		case OpDiv:
			out.Append(a / b)
		default:
			out.Append(T(math.Mod(float64(a), float64(b))))
		}
		return nil
	}
}
