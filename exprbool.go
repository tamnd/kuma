package kuma

import (
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// BoolValue is a boolean piece of an expression, which is a [BoolCol] or a
// [BoolExpr]. It is what [Frame.Filter] takes, so a frame can be filtered by a
// condition that was worked out or by a boolean column that was already there.
type BoolValue[S any] interface {
	Expr[S]
	boolValue()
}

// BoolCol is a handle on a boolean column of a frame with schema S.
type BoolCol[S any] struct{ boolops[S] }

// BoolExpr is a condition: an expression whose value in each row is true, false
// or missing.
//
// It is what a comparison gives, it is what [Frame.Filter] takes, and it is the
// type that carries the schema through a predicate, so a condition written
// against one table cannot be used to filter another.
type BoolExpr[S any] struct{ boolops[S] }

// NewBoolCol returns a handle on the boolean column called name in a frame with
// schema S.
func NewBoolCol[S any](name string) BoolCol[S] { return BoolCol[S]{boolops[S]{plan.Col(name)}} }

// Bool returns a handle on a boolean column of a frame with no schema behind
// it, which is the light version of [NewBoolCol].
func Bool(name string) BoolCol[Dynamic] { return NewBoolCol[Dynamic](name) }

// Name returns the column the handle names.
func (c BoolCol[S]) Name() string { return c.n.Name() }

// Series returns the column as a Series[bool], reporting an error if the frame
// has no such column or holds something else there.
func (c BoolCol[S]) Series(f *Frame[S]) (Series[bool], error) {
	return f.Series[bool](c.n.Name())
}

// boolops is the method set of everything boolean valued, shared by a column
// handle and an expression for the reason [f64ops] gives.
//
// The logic is three valued, because a column has a third thing a value can be.
// A missing value is not known rather than false, so false and missing is
// false, while true and missing is missing. [kernel.And] has the table and the
// reason.
type boolops[S any] struct{ n *plan.Expr }

func (o boolops[S]) expr() *plan.Expr { return o.n }
func (o boolops[S]) boolValue()       {}

// String returns the condition as it would be written.
func (o boolops[S]) String() string { return o.n.String() }

// And returns whether both this and x hold in the same row.
func (o boolops[S]) And(x BoolValue[S]) BoolExpr[S] {
	return boolOf[S](plan.And(o.n, x.expr()))
}

// Or returns whether this or x holds in the same row.
func (o boolops[S]) Or(x BoolValue[S]) BoolExpr[S] {
	return boolOf[S](plan.Or(o.n, x.expr()))
}

// Not returns the negation. A missing value stays missing, since the negation
// of a thing nobody knows is another thing nobody knows.
func (o boolops[S]) Not() BoolExpr[S] { return boolOf[S](plan.Not(o.n)) }

// Eq returns whether the value equals v.
func (o boolops[S]) Eq(v bool) BoolExpr[S] {
	return boolOf[S](plan.Compare(kernel.OpEq, o.n, plan.Lit(v)))
}

// Ne returns whether the value differs from v.
func (o boolops[S]) Ne(v bool) BoolExpr[S] {
	return boolOf[S](plan.Compare(kernel.OpNe, o.n, plan.Lit(v)))
}

// EqExpr returns whether the value equals the value of x in the same row.
func (o boolops[S]) EqExpr(x BoolValue[S]) BoolExpr[S] {
	return boolOf[S](plan.Compare(kernel.OpEq, o.n, x.expr()))
}

// NeExpr returns whether the value differs from the value of x in the same row.
func (o boolops[S]) NeExpr(x BoolValue[S]) BoolExpr[S] {
	return boolOf[S](plan.Compare(kernel.OpNe, o.n, x.expr()))
}

// IsNull returns whether the value is missing, which is the one question about
// a condition that always has a plain true or false answer.
func (o boolops[S]) IsNull() BoolExpr[S] { return boolOf[S](plan.IsNull(o.n)) }

// IsNotNull returns whether the value is there.
func (o boolops[S]) IsNotNull() BoolExpr[S] { return boolOf[S](plan.IsNotNull(o.n)) }

func boolOf[S any](n *plan.Expr) BoolExpr[S] { return BoolExpr[S]{boolops[S]{n}} }
