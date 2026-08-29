package kernel

import (
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// And returns the logical and of two boolean columns.
//
// The logic is three valued, because a column has a third thing a value can be.
// A missing value is not known rather than false, so false and null is false,
// since nothing the unknown value could turn out to be would make the pair
// true, while true and null is null, since it depends. That is Kleene's rule,
// it is what SQL does, and it is what makes a filter on two conditions agree
// with a filter on each of them in turn.
//
// A column of one value on either side is that value against every row of the
// other, so a predicate can be turned off with a single false without building
// a column of them.
//
// It returns an error unless both columns hold conditions, since the column is
// often one a caller picked out of a file at runtime.
func And(a, b *array.Chunked) (*array.Chunked, error) {
	return connective(a, b, "and", false)
}

// Or returns the logical or of two boolean columns, under the same three valued
// rule that [And] describes. True or null is true, false or null is null.
func Or(a, b *array.Chunked) (*array.Chunked, error) {
	return connective(a, b, "or", true)
}

// connective is And and Or, which differ in one value and in nothing else.
//
// The short value is the one that decides the answer on its own, which is false
// in a conjunction and true in a disjunction. When either side holds it the
// other side is not looked at, and that one rule is what makes the missing
// values behave.
func connective(a, b *array.Chunked, name string, short bool) (*array.Chunked, error) {
	if a == nil || b == nil {
		panic("kernel: " + name + " of a nil column")
	}
	if err := condition(a, name); err != nil {
		return nil, err
	}
	if err := condition(b, name); err != nil {
		return nil, err
	}
	n, fixedA, fixedB := binaryLen(name, a, b)

	out := builder(dtype.Bool)
	out.Grow(n)

	ca, cb := newCursor(a, fixedA), newCursor(b, fixedB)
	for range n {
		x, i := ca.next()
		y, j := cb.next()
		xk, xv := answer(x, i)
		yk, yv := answer(y, j)
		switch {
		case xk && xv == short, yk && yv == short:
			out.AppendBool(short)
		case !xk || !yk:
			out.AppendNull()
		default:
			out.AppendBool(!short)
		}
	}
	return one(dtype.Bool, out.Finish()), nil
}

// answer returns whether value i of a is present and, when it is, what it is.
// The value of a column of nothing is never read, which is what lets a column
// of nulls stand in for a condition nobody has answered.
func answer(a *array.Array, i int) (ok, v bool) {
	if a.IsNull(i) {
		return false, false
	}
	return true, a.Bool(i)
}

// Not returns the negation of a boolean column. A missing value stays missing,
// since the negation of a thing nobody knows is another thing nobody knows.
//
// It returns an error unless the column holds conditions.
func Not(c *array.Chunked) (*array.Chunked, error) {
	if c == nil {
		panic("kernel: not of a nil column")
	}
	if err := condition(c, "not"); err != nil {
		return nil, err
	}

	out := builder(dtype.Bool)
	out.Grow(c.Len())
	for _, a := range c.Chunks() {
		for i := range a.Len() {
			ok, v := answer(a, i)
			if !ok {
				out.AppendNull()
				continue
			}
			out.AppendBool(!v)
		}
	}
	return one(dtype.Bool, out.Finish()), nil
}

// IsCondition reports whether a column of this type is one the logic can read.
//
// That is a boolean column, or a column of nothing, whose values are all
// missing and so are all the answer nobody has. Letting the second one through
// costs a line and saves a caller special casing the column an empty file gave
// them.
//
// It is exported for the same reason [CompareType] is. A plan has to know
// whether an expression can be filtered on before there is a column to ask.
func IsCondition(dt dtype.DataType) bool {
	switch dt.Kind() {
	case dtype.BoolKind, dtype.NullKind:
		return true
	default:
		return false
	}
}

// condition returns an error unless c is a column the logic can read.
func condition(c *array.Chunked, name string) error {
	if !IsCondition(c.DType()) {
		return fmt.Errorf("kernel: cannot take the %s of a %s column, which is not a condition",
			name, c.DType())
	}
	return nil
}
