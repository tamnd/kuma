package kuma

import (
	"time"

	"github.com/tamnd/kuma/kernel"
)

// TimeValue is a timestamp valued piece of an expression, which is a [TimeCol]
// or a [TimeExpr].
type TimeValue[S any] interface {
	Expr[S]
	timeValue()
}

// TimeCol is a handle on a timestamp column of a frame with schema S.
type TimeCol[S any] struct{ timeops[S] }

// TimeExpr is a timestamp valued expression.
type TimeExpr[S any] struct{ timeops[S] }

// NewTimeCol returns a handle on the timestamp column called name in a frame
// with schema S.
func NewTimeCol[S any](name string) TimeCol[S] { return TimeCol[S]{timeops[S]{colNode(name)}} }

// Time returns a handle on a timestamp column of a frame with no schema behind
// it, which is the light version of [NewTimeCol].
func Time(name string) TimeCol[Dynamic] { return NewTimeCol[Dynamic](name) }

// Name returns the column the handle names.
func (c TimeCol[S]) Name() string { return c.n.name }

// Series returns the column as a Series[time.Time], reporting an error if the
// frame has no such column or holds something else there.
func (c TimeCol[S]) Series(f *Frame[S]) (Series[time.Time], error) {
	return f.Series[time.Time](c.n.name)
}

// timeops is the method set of everything timestamp valued, shared by a column
// handle and an expression for the reason [f64ops] gives.
//
// A timestamp column is a count of units since the epoch, and a time.Time
// written in a query becomes that count in the column's own unit. It has to
// land on the unit exactly: a time with nanoseconds in it compared against a
// column of seconds is an error saying to truncate it first, rather than a
// comparison that quietly drops the part that did not fit.
//
// Not yet: Trunc, Year, Month and the rest of the calendar, which need the
// temporal side of the cast kernel first.
type timeops[S any] struct{ n *node }

func (o timeops[S]) expr() *node { return o.n }
func (o timeops[S]) timeValue()  {}

// String returns the expression as it would be written.
func (o timeops[S]) String() string { return o.n.String() }

// Eq returns whether the value is the instant t.
func (o timeops[S]) Eq(t time.Time) BoolExpr[S] { return o.cmp(kernel.OpEq, litNode(t)) }

// Ne returns whether the value is any instant other than t.
func (o timeops[S]) Ne(t time.Time) BoolExpr[S] { return o.cmp(kernel.OpNe, litNode(t)) }

// Before returns whether the value is earlier than t.
func (o timeops[S]) Before(t time.Time) BoolExpr[S] { return o.cmp(kernel.OpLt, litNode(t)) }

// AtOrBefore returns whether the value is t or earlier.
func (o timeops[S]) AtOrBefore(t time.Time) BoolExpr[S] { return o.cmp(kernel.OpLe, litNode(t)) }

// After returns whether the value is later than t.
func (o timeops[S]) After(t time.Time) BoolExpr[S] { return o.cmp(kernel.OpGt, litNode(t)) }

// AtOrAfter returns whether the value is t or later.
func (o timeops[S]) AtOrAfter(t time.Time) BoolExpr[S] { return o.cmp(kernel.OpGe, litNode(t)) }

// EqExpr returns whether the value is the same instant as the value of x in the
// same row.
func (o timeops[S]) EqExpr(x TimeValue[S]) BoolExpr[S] { return o.cmp(kernel.OpEq, x.expr()) }

// NeExpr returns whether the value is a different instant from the value of x
// in the same row.
func (o timeops[S]) NeExpr(x TimeValue[S]) BoolExpr[S] { return o.cmp(kernel.OpNe, x.expr()) }

// BeforeExpr returns whether the value is earlier than the value of x in the
// same row.
func (o timeops[S]) BeforeExpr(x TimeValue[S]) BoolExpr[S] { return o.cmp(kernel.OpLt, x.expr()) }

// AtOrBeforeExpr returns whether the value is no later than the value of x in
// the same row.
func (o timeops[S]) AtOrBeforeExpr(x TimeValue[S]) BoolExpr[S] {
	return o.cmp(kernel.OpLe, x.expr())
}

// AfterExpr returns whether the value is later than the value of x in the same
// row.
func (o timeops[S]) AfterExpr(x TimeValue[S]) BoolExpr[S] { return o.cmp(kernel.OpGt, x.expr()) }

// AtOrAfterExpr returns whether the value is no earlier than the value of x in
// the same row.
func (o timeops[S]) AtOrAfterExpr(x TimeValue[S]) BoolExpr[S] {
	return o.cmp(kernel.OpGe, x.expr())
}

// IsNull returns whether the value is missing.
func (o timeops[S]) IsNull() BoolExpr[S] { return boolOf[S](unaryNode(kindIsNull, o.n)) }

// IsNotNull returns whether the value is there.
func (o timeops[S]) IsNotNull() BoolExpr[S] { return boolOf[S](unaryNode(kindIsNotNull, o.n)) }

func (o timeops[S]) cmp(op kernel.CompareOp, r *node) BoolExpr[S] {
	return boolOf[S](cmpNode(op, o.n, r))
}
