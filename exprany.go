package kuma

import (
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// AnyValue is a piece of an expression whose type is not known until the frame
// is read, which is an [AnyCol] or an [AnyExpr].
type AnyValue[S any] interface {
	Expr[S]
	anyValue()
}

// AnyCol is a handle on a column whose type is only known at runtime.
type AnyCol[S any] struct{ anyops[S] }

// AnyExpr is an expression over columns whose types are only known at runtime.
type AnyExpr[S any] struct{ anyops[S] }

// Dyn returns a handle on a column of any type, which is the way to write a
// query against a file nobody has seen at compile time.
//
//	f, err := f.Filter(kuma.Dyn("price").Gt(100))
//
// The literal takes the column's own type when it can, so the 100 above works
// whether the file turned out to hold that column as an int64, a uint32 or a
// float64, and it is an error rather than a rounding when the two cannot be
// reconciled. That is the whole difference from the typed handles: the check
// that would be the compiler's happens when the frame is read.
//
// Everything here is available on a typed frame as well, since a schema that
// covers most of a file may still leave a column nobody wants to name.
func Dyn(name string) AnyCol[Dynamic] { return NewAnyCol[Dynamic](name) }

// NewAnyCol returns a handle on the column called name in a frame with schema
// S, whatever type that column turns out to hold.
func NewAnyCol[S any](name string) AnyCol[S] { return AnyCol[S]{anyops[S]{plan.Col(name)}} }

// Lit is a value written in a query, for the times one is needed on the left of
// an operator rather than on the right.
//
//	f, err := f.Filter(kuma.Lit(100).Lt(kuma.Dyn("price")))
//
// A literal on its own is not much use, since an expression that reads no
// column has one value where the frame has rows, which [Frame.Eval] reports
// rather than stretching it to fit.
func Lit(v any) AnyExpr[Dynamic] { return NewLit[Dynamic](v) }

// NewLit is [Lit] for a frame with schema S.
func NewLit[S any](v any) AnyExpr[S] { return AnyExpr[S]{anyops[S]{plan.Lit(v)}} }

// Name returns the column the handle names.
func (c AnyCol[S]) Name() string { return c.n.Name() }

// Column returns the column itself, reporting an error if the frame has no such
// column.
func (c AnyCol[S]) Column(f *Frame[S]) (Column, error) { return f.Column(c.n.Name()) }

// anyops is the method set of a value whose type is not known until the frame
// is read.
//
// The literals are plain Go values rather than a type per method, so what a 100
// means is settled against the column it is used with rather than at the point
// it is written. That is the trade the dynamic path makes everywhere: the same
// question is asked, it is just asked later.
type anyops[S any] struct{ n *plan.Expr }

func (o anyops[S]) expr() *plan.Expr { return o.n }
func (o anyops[S]) anyValue()        {}

// String returns the expression as it would be written.
func (o anyops[S]) String() string { return o.n.String() }

// Eq returns whether the value equals v.
func (o anyops[S]) Eq(v any) BoolExpr[S] { return o.cmp(kernel.OpEq, plan.Lit(v)) }

// Ne returns whether the value differs from v.
func (o anyops[S]) Ne(v any) BoolExpr[S] { return o.cmp(kernel.OpNe, plan.Lit(v)) }

// Lt returns whether the value is less than v.
func (o anyops[S]) Lt(v any) BoolExpr[S] { return o.cmp(kernel.OpLt, plan.Lit(v)) }

// Le returns whether the value is less than or equal to v.
func (o anyops[S]) Le(v any) BoolExpr[S] { return o.cmp(kernel.OpLe, plan.Lit(v)) }

// Gt returns whether the value is greater than v.
func (o anyops[S]) Gt(v any) BoolExpr[S] { return o.cmp(kernel.OpGt, plan.Lit(v)) }

// Ge returns whether the value is greater than or equal to v.
func (o anyops[S]) Ge(v any) BoolExpr[S] { return o.cmp(kernel.OpGe, plan.Lit(v)) }

// EqExpr returns whether the value equals the value of x in the same row.
func (o anyops[S]) EqExpr(x AnyValue[S]) BoolExpr[S] { return o.cmp(kernel.OpEq, x.expr()) }

// NeExpr returns whether the value differs from the value of x in the same row.
func (o anyops[S]) NeExpr(x AnyValue[S]) BoolExpr[S] { return o.cmp(kernel.OpNe, x.expr()) }

// LtExpr returns whether the value is less than the value of x in the same row.
func (o anyops[S]) LtExpr(x AnyValue[S]) BoolExpr[S] { return o.cmp(kernel.OpLt, x.expr()) }

// LeExpr returns whether the value is at most the value of x in the same row.
func (o anyops[S]) LeExpr(x AnyValue[S]) BoolExpr[S] { return o.cmp(kernel.OpLe, x.expr()) }

// GtExpr returns whether the value is greater than the value of x in the same
// row.
func (o anyops[S]) GtExpr(x AnyValue[S]) BoolExpr[S] { return o.cmp(kernel.OpGt, x.expr()) }

// GeExpr returns whether the value is at least the value of x in the same row.
func (o anyops[S]) GeExpr(x AnyValue[S]) BoolExpr[S] { return o.cmp(kernel.OpGe, x.expr()) }

// Add returns the value plus v.
func (o anyops[S]) Add(v any) AnyExpr[S] { return o.arith(kernel.OpAdd, plan.Lit(v)) }

// Sub returns the value minus v.
func (o anyops[S]) Sub(v any) AnyExpr[S] { return o.arith(kernel.OpSub, plan.Lit(v)) }

// Mul returns the value times v.
func (o anyops[S]) Mul(v any) AnyExpr[S] { return o.arith(kernel.OpMul, plan.Lit(v)) }

// Div returns the value divided by v, which truncates in an integer column and
// does not in a float one, the same as the Go operator on those two types.
func (o anyops[S]) Div(v any) AnyExpr[S] { return o.arith(kernel.OpDiv, plan.Lit(v)) }

// Mod returns the remainder of the value divided by v.
func (o anyops[S]) Mod(v any) AnyExpr[S] { return o.arith(kernel.OpMod, plan.Lit(v)) }

// AddExpr returns the value plus the value of x in the same row.
func (o anyops[S]) AddExpr(x AnyValue[S]) AnyExpr[S] { return o.arith(kernel.OpAdd, x.expr()) }

// SubExpr returns the value minus the value of x in the same row.
func (o anyops[S]) SubExpr(x AnyValue[S]) AnyExpr[S] { return o.arith(kernel.OpSub, x.expr()) }

// MulExpr returns the value times the value of x in the same row.
func (o anyops[S]) MulExpr(x AnyValue[S]) AnyExpr[S] { return o.arith(kernel.OpMul, x.expr()) }

// DivExpr returns the value divided by the value of x in the same row.
func (o anyops[S]) DivExpr(x AnyValue[S]) AnyExpr[S] { return o.arith(kernel.OpDiv, x.expr()) }

// ModExpr returns the remainder of the value divided by the value of x in the
// same row.
func (o anyops[S]) ModExpr(x AnyValue[S]) AnyExpr[S] { return o.arith(kernel.OpMod, x.expr()) }

// IsNull returns whether the value is missing.
func (o anyops[S]) IsNull() BoolExpr[S] { return boolOf[S](plan.IsNull(o.n)) }

// IsNotNull returns whether the value is there.
func (o anyops[S]) IsNotNull() BoolExpr[S] { return boolOf[S](plan.IsNotNull(o.n)) }

func (o anyops[S]) cmp(op kernel.CompareOp, r *plan.Expr) BoolExpr[S] {
	return boolOf[S](plan.Compare(op, o.n, r))
}

func (o anyops[S]) arith(op kernel.ArithOp, r *plan.Expr) AnyExpr[S] {
	return AnyExpr[S]{anyops[S]{plan.Arith(op, o.n, r)}}
}
