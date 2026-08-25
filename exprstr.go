package kuma

import "github.com/tamnd/kuma/kernel"

// StrValue is a string valued piece of an expression, which is a [StrCol] or a
// [StrExpr].
type StrValue[S any] interface {
	Expr[S]
	stringValue()
}

// StrCol is a handle on a string column of a frame with schema S.
type StrCol[S any] struct{ strops[S] }

// StrExpr is a string valued expression.
type StrExpr[S any] struct{ strops[S] }

// NewStrCol returns a handle on the string column called name in a frame with
// schema S.
func NewStrCol[S any](name string) StrCol[S] { return StrCol[S]{strops[S]{colNode(name)}} }

// Str returns a handle on a string column of a frame with no schema behind it,
// which is the light version of [NewStrCol].
func Str(name string) StrCol[Dynamic] { return NewStrCol[Dynamic](name) }

// Name returns the column the handle names.
func (c StrCol[S]) Name() string { return c.n.name }

// Series returns the column as a Series[string], reporting an error if the
// frame has no such column or holds something else there.
func (c StrCol[S]) Series(f *Frame[S]) (Series[string], error) {
	return f.Series[string](c.n.name)
}

// strops is the method set of everything string valued, shared by a column
// handle and an expression for the reason [f64ops] gives.
//
// Strings are ordered by their bytes, which for text stored as UTF-8 is the
// same order as by code point. It is not a language aware collation and it does
// not try to be: a collation is a table per language and per locale, and the
// place for it is a package of its own rather than a comparison that quietly
// does something different depending on where it runs.
type strops[S any] struct{ n *node }

func (o strops[S]) expr() *node  { return o.n }
func (o strops[S]) stringValue() {}

// String returns the expression as it would be written.
func (o strops[S]) String() string { return o.n.String() }

// Eq returns whether the value equals v.
func (o strops[S]) Eq(v string) BoolExpr[S] { return o.cmp(kernel.OpEq, litNode(v)) }

// Ne returns whether the value differs from v.
func (o strops[S]) Ne(v string) BoolExpr[S] { return o.cmp(kernel.OpNe, litNode(v)) }

// Lt returns whether the value sorts before v.
func (o strops[S]) Lt(v string) BoolExpr[S] { return o.cmp(kernel.OpLt, litNode(v)) }

// Le returns whether the value sorts before v or equals it.
func (o strops[S]) Le(v string) BoolExpr[S] { return o.cmp(kernel.OpLe, litNode(v)) }

// Gt returns whether the value sorts after v.
func (o strops[S]) Gt(v string) BoolExpr[S] { return o.cmp(kernel.OpGt, litNode(v)) }

// Ge returns whether the value sorts after v or equals it.
func (o strops[S]) Ge(v string) BoolExpr[S] { return o.cmp(kernel.OpGe, litNode(v)) }

// EqExpr returns whether the value equals the value of x in the same row.
func (o strops[S]) EqExpr(x StrValue[S]) BoolExpr[S] { return o.cmp(kernel.OpEq, x.expr()) }

// NeExpr returns whether the value differs from the value of x in the same row.
func (o strops[S]) NeExpr(x StrValue[S]) BoolExpr[S] { return o.cmp(kernel.OpNe, x.expr()) }

// LtExpr returns whether the value sorts before the value of x in the same row.
func (o strops[S]) LtExpr(x StrValue[S]) BoolExpr[S] { return o.cmp(kernel.OpLt, x.expr()) }

// LeExpr returns whether the value does not sort after the value of x in the
// same row.
func (o strops[S]) LeExpr(x StrValue[S]) BoolExpr[S] { return o.cmp(kernel.OpLe, x.expr()) }

// GtExpr returns whether the value sorts after the value of x in the same row.
func (o strops[S]) GtExpr(x StrValue[S]) BoolExpr[S] { return o.cmp(kernel.OpGt, x.expr()) }

// GeExpr returns whether the value does not sort before the value of x in the
// same row.
func (o strops[S]) GeExpr(x StrValue[S]) BoolExpr[S] { return o.cmp(kernel.OpGe, x.expr()) }

// IsNull returns whether the value is missing, which is not the same as being
// the empty string.
func (o strops[S]) IsNull() BoolExpr[S] { return boolOf[S](unaryNode(kindIsNull, o.n)) }

// IsNotNull returns whether the value is there.
func (o strops[S]) IsNotNull() BoolExpr[S] { return boolOf[S](unaryNode(kindIsNotNull, o.n)) }

func (o strops[S]) cmp(op kernel.CompareOp, r *node) BoolExpr[S] {
	return boolOf[S](cmpNode(op, o.n, r))
}
