package kuma

import (
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// Expr is anything that turns into a column when a frame is put behind it,
// which is a column handle or an expression built out of one.
//
// The type parameter is the schema, so an expression written against Trade
// cannot be handed to a frame of orders. That check is the compiler's, and it
// is the reason the typed handles exist. What the handle carries underneath is
// a [plan.Expr], which is the same expression with the schema forgotten.
//
// The interface has an unexported method, so the expression types in this
// package are the whole of it. That is on purpose. An expression is a small
// tree that the frame walks, not an interface a caller implements, and keeping
// it closed is what lets the walk be a switch rather than a virtual call per
// row.
//
// String returns the expression as it would be written, which is what an error
// about it names and what a column built from it is called.
type Expr[S any] interface {
	fmt.Stringer

	expr() *plan.Expr
}

// lookupFunc finds a column of the frame an expression is being evaluated
// against. It is a function rather than the frame itself because the frame is
// generic and the walk below is not, and a schema type parameter would be
// carried through every step of it for nothing.
type lookupFunc func(name string) (*array.Chunked, error)

// lookup returns the column finder for this frame, naming op in the error a
// wrong column name produces.
func (f *Frame[S]) lookup(op string) lookupFunc {
	return func(name string) (*array.Chunked, error) {
		i, ok := f.index[name]
		if !ok {
			return nil, noColumn(op, name, f.Names())
		}
		return f.cols[i].data, nil
	}
}

// eval works out the values of one step of an expression.
//
// The hint is the type the other side of a two sided step turned out to be. It
// is used by a literal and ignored by everything else, since a literal is the
// one thing in an expression with no type of its own: the caller wrote 100 and
// meant the number, so what it becomes is the column's business.
func eval(n *plan.Expr, look lookupFunc, hint dtype.DataType) (*array.Chunked, error) {
	switch n.Kind() {
	case plan.KindColumn:
		return look(n.Name())
	case plan.KindLiteral:
		return plan.LiteralColumn(n.Value(), hint)
	case plan.KindCompare:
		a, b, err := operands(n, look)
		if err != nil {
			return nil, err
		}
		return kernel.Compare(a, b, n.CompareOp())
	case plan.KindArith:
		a, b, err := operands(n, look)
		if err != nil {
			return nil, err
		}
		return kernel.Arith(a, b, n.ArithOp())
	case plan.KindAnd, plan.KindOr:
		a, b, err := operands(n, look)
		if err != nil {
			return nil, err
		}
		if n.Kind() == plan.KindAnd {
			return kernel.And(a, b)
		}
		return kernel.Or(a, b)
	case plan.KindNot:
		a, err := eval(n.Left(), look, nil)
		if err != nil {
			return nil, err
		}
		return kernel.Not(a)
	case plan.KindIsNull:
		a, err := eval(n.Left(), look, nil)
		if err != nil {
			return nil, err
		}
		return kernel.IsNull(a), nil
	case plan.KindCast:
		a, err := eval(n.Left(), look, nil)
		if err != nil {
			return nil, err
		}
		return kernel.Cast(a, n.DType())
	default:
		a, err := eval(n.Left(), look, nil)
		if err != nil {
			return nil, err
		}
		return kernel.IsNotNull(a), nil
	}
}

// operands works out both sides of a two sided step, doing the side that has a
// type first so that the literal on the other side knows what to become.
func operands(n *plan.Expr, look lookupFunc) (a, b *array.Chunked, err error) {
	l, r := n.Left(), n.Right()
	if l.Kind() == plan.KindLiteral && r.Kind() != plan.KindLiteral {
		b, err = eval(r, look, nil)
		if err != nil {
			return nil, nil, err
		}
		a, err = eval(l, look, b.DType())
		if err != nil {
			return nil, nil, err
		}
		return a, b, nil
	}

	a, err = eval(l, look, nil)
	if err != nil {
		return nil, nil, err
	}
	b, err = eval(r, look, a.DType())
	if err != nil {
		return nil, nil, err
	}
	return a, b, nil
}

// Eval works out an expression against the frame and returns the result as a
// column.
//
//	total, err := f.Eval(t.Price.MulExpr(t.Qty))
//
// The column is named after the expression, so the one above is called
// "(price * qty)". Rename it, or use WithColumn, when the name matters.
//
// It reports an error if a column named in the expression is not in the frame,
// if two columns have no type in common, or if a literal cannot be used with
// the column it was written against.
func (f *Frame[S]) Eval(e Expr[S]) (Column, error) {
	n := e.expr()
	data, err := eval(n, f.lookup("Eval"), nil)
	if err != nil {
		return Column{}, err
	}
	if data.Len() != f.rows {
		return Column{}, fmt.Errorf("kuma: %s gives %d values for a frame of %d rows, "+
			"an expression has to read at least one column: %w", n, data.Len(), f.rows, ErrLength)
	}
	return Column{name: n.String(), data: data}, nil
}

// WithExpr returns a frame with the result of an expression added as a column
// called name, or replacing the column of that name when there is one.
//
//	f, err := f.WithExpr("notional", t.Price.MulExpr(t.Qty))
//
// It is [Frame.Eval] and [Frame.WithColumn] in one step, and it is the step a
// query spends most of its time in, so it is worth the one method. The result
// is a Dynamic frame for the reason WithColumn gives.
func (f *Frame[S]) WithExpr(name string, e Expr[S]) (*Frame[Dynamic], error) {
	c, err := f.Eval(e)
	if err != nil {
		return nil, err
	}
	return f.WithColumn(c.Rename(name))
}
