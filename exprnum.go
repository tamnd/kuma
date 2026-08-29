package kuma

import (
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// F64Value is a float64 valued piece of an expression, which is an [F64Col] or
// an [F64Expr]. It is what the methods taking another column rather than a
// literal accept, so that t.Price.GtExpr(t.Limit) and
// t.Price.GtExpr(t.Limit.Add(1)) are both written the same way.
type F64Value[S any] interface {
	Expr[S]
	float64Value()
}

// I64Value is an int64 valued piece of an expression, which is an [I64Col] or
// an [I64Expr]. It is I64's half of what [F64Value] describes.
type I64Value[S any] interface {
	Expr[S]
	int64Value()
}

// F64Col is a handle on a float64 column of a frame with schema S.
//
//	var TradeCols = struct {
//		Price kuma.F64Col[Trade]
//	}{
//		Price: kuma.NewF64Col[Trade]("price"),
//	}
//
// The handle is one word and never escapes, so building the variable above once
// at package level and using it everywhere costs nothing per query.
//
// The schema type is what stops a handle written for one table being used
// against another. A F64Col[Trade] cannot be handed to a Frame[Order], and the
// compiler says so rather than the data being read and the answer being wrong.
// [F64] is the same handle without that check, for a frame whose columns are
// only known at runtime.
type F64Col[S any] struct{ f64ops[S] }

// I64Col is a handle on an int64 column of a frame with schema S. It is the
// int64 half of what [F64Col] describes.
type I64Col[S any] struct{ i64ops[S] }

// F64Expr is a float64 valued expression, which is what doing arithmetic to a
// float64 column gives. It has the same methods a column handle has, so a
// chain of them reads the same the whole way along.
type F64Expr[S any] struct{ f64ops[S] }

// I64Expr is an int64 valued expression, which is what doing arithmetic to an
// int64 column gives.
type I64Expr[S any] struct{ i64ops[S] }

// NewF64Col returns a handle on the float64 column called name in a frame with
// schema S. It is what kumagen writes and what a hand written schema variable
// calls.
func NewF64Col[S any](name string) F64Col[S] { return F64Col[S]{f64ops[S]{plan.Col(name)}} }

// NewI64Col returns a handle on the int64 column called name in a frame with
// schema S.
func NewI64Col[S any](name string) I64Col[S] { return I64Col[S]{i64ops[S]{plan.Col(name)}} }

// F64 returns a handle on a float64 column of a frame with no schema behind it,
// which is the light version of [NewF64Col].
//
//	price := kuma.F64("price")
//	high, err := f.Filter(price.Gt(100))
//
// The name is still written once rather than scattered through the code, and a
// string still cannot be compared to a number. What is given up is the check
// that the column belongs to the frame it is used against, which moves from
// compile time to the moment the frame is read.
func F64(name string) F64Col[Dynamic] { return NewF64Col[Dynamic](name) }

// I64 returns a handle on an int64 column of a frame with no schema behind it.
func I64(name string) I64Col[Dynamic] { return NewI64Col[Dynamic](name) }

// Name returns the column the handle names.
func (c F64Col[S]) Name() string { return c.n.Name() }

// Name returns the column the handle names.
func (c I64Col[S]) Name() string { return c.n.Name() }

// Series returns the column as a Series[float64], reporting an error if the
// frame has no such column or holds something else there.
func (c F64Col[S]) Series(f *Frame[S]) (Series[float64], error) {
	return f.Series[float64](c.n.Name())
}

// Series returns the column as a Series[int64], reporting an error if the frame
// has no such column or holds something else there.
func (c I64Col[S]) Series(f *Frame[S]) (Series[int64], error) {
	return f.Series[int64](c.n.Name())
}

// f64ops is the method set of everything float64 valued.
//
// A column handle and an expression are the same thing at two ends of a chain:
// t.Price is a handle, t.Price.Mul(2) is an expression, and both of them are a
// float64 to whatever comes next. They share these methods by embedding rather
// than by having them written twice, which is a smaller thing than it looks:
// two copies of twenty methods would be two chances for the two of them to
// disagree.
type f64ops[S any] struct{ n *plan.Expr }

func (o f64ops[S]) expr() *plan.Expr { return o.n }
func (o f64ops[S]) float64Value()    {}

// String returns the expression as it would be written.
func (o f64ops[S]) String() string { return o.n.String() }

// Eq returns whether the value equals v. A missing value equals nothing, not
// even another missing value, so it gives a missing answer.
func (o f64ops[S]) Eq(v float64) BoolExpr[S] { return o.cmp(kernel.OpEq, plan.Lit(v)) }

// Ne returns whether the value differs from v.
func (o f64ops[S]) Ne(v float64) BoolExpr[S] { return o.cmp(kernel.OpNe, plan.Lit(v)) }

// Lt returns whether the value is less than v.
func (o f64ops[S]) Lt(v float64) BoolExpr[S] { return o.cmp(kernel.OpLt, plan.Lit(v)) }

// Le returns whether the value is less than or equal to v.
func (o f64ops[S]) Le(v float64) BoolExpr[S] { return o.cmp(kernel.OpLe, plan.Lit(v)) }

// Gt returns whether the value is greater than v.
func (o f64ops[S]) Gt(v float64) BoolExpr[S] { return o.cmp(kernel.OpGt, plan.Lit(v)) }

// Ge returns whether the value is greater than or equal to v.
func (o f64ops[S]) Ge(v float64) BoolExpr[S] { return o.cmp(kernel.OpGe, plan.Lit(v)) }

// EqExpr returns whether the value equals the value of x in the same row.
func (o f64ops[S]) EqExpr(x F64Value[S]) BoolExpr[S] { return o.cmp(kernel.OpEq, x.expr()) }

// NeExpr returns whether the value differs from the value of x in the same row.
func (o f64ops[S]) NeExpr(x F64Value[S]) BoolExpr[S] { return o.cmp(kernel.OpNe, x.expr()) }

// LtExpr returns whether the value is less than the value of x in the same row.
func (o f64ops[S]) LtExpr(x F64Value[S]) BoolExpr[S] { return o.cmp(kernel.OpLt, x.expr()) }

// LeExpr returns whether the value is at most the value of x in the same row.
func (o f64ops[S]) LeExpr(x F64Value[S]) BoolExpr[S] { return o.cmp(kernel.OpLe, x.expr()) }

// GtExpr returns whether the value is greater than the value of x in the same
// row.
func (o f64ops[S]) GtExpr(x F64Value[S]) BoolExpr[S] { return o.cmp(kernel.OpGt, x.expr()) }

// GeExpr returns whether the value is at least the value of x in the same row.
func (o f64ops[S]) GeExpr(x F64Value[S]) BoolExpr[S] { return o.cmp(kernel.OpGe, x.expr()) }

// Add returns the value plus v.
func (o f64ops[S]) Add(v float64) F64Expr[S] { return o.arith(kernel.OpAdd, plan.Lit(v)) }

// Sub returns the value minus v.
func (o f64ops[S]) Sub(v float64) F64Expr[S] { return o.arith(kernel.OpSub, plan.Lit(v)) }

// Mul returns the value times v.
func (o f64ops[S]) Mul(v float64) F64Expr[S] { return o.arith(kernel.OpMul, plan.Lit(v)) }

// Div returns the value divided by v. Dividing by zero gives an infinity or a
// NaN, which is what a float64 division does.
func (o f64ops[S]) Div(v float64) F64Expr[S] { return o.arith(kernel.OpDiv, plan.Lit(v)) }

// Mod returns the remainder of the value divided by v, which is math.Mod.
func (o f64ops[S]) Mod(v float64) F64Expr[S] { return o.arith(kernel.OpMod, plan.Lit(v)) }

// AddExpr returns the value plus the value of x in the same row.
func (o f64ops[S]) AddExpr(x F64Value[S]) F64Expr[S] { return o.arith(kernel.OpAdd, x.expr()) }

// SubExpr returns the value minus the value of x in the same row.
func (o f64ops[S]) SubExpr(x F64Value[S]) F64Expr[S] { return o.arith(kernel.OpSub, x.expr()) }

// MulExpr returns the value times the value of x in the same row.
func (o f64ops[S]) MulExpr(x F64Value[S]) F64Expr[S] { return o.arith(kernel.OpMul, x.expr()) }

// DivExpr returns the value divided by the value of x in the same row.
func (o f64ops[S]) DivExpr(x F64Value[S]) F64Expr[S] { return o.arith(kernel.OpDiv, x.expr()) }

// ModExpr returns the remainder of the value divided by the value of x in the
// same row.
func (o f64ops[S]) ModExpr(x F64Value[S]) F64Expr[S] { return o.arith(kernel.OpMod, x.expr()) }

// AsI64 returns the value as an int64.
//
// The fraction is thrown away, the way a Go conversion does it, so 3.9 becomes
// 3. A value too large for an int64, and NaN or an infinity, fit nowhere and
// are an error naming the row. [kernel.Cast] has the rest of the rule.
func (o f64ops[S]) AsI64() I64Expr[S] { return i64Of[S](plan.Cast(dtype.Int64, o.n)) }

// IsNull returns whether the value is missing.
func (o f64ops[S]) IsNull() BoolExpr[S] { return boolOf[S](plan.IsNull(o.n)) }

// IsNotNull returns whether the value is there.
func (o f64ops[S]) IsNotNull() BoolExpr[S] { return boolOf[S](plan.IsNotNull(o.n)) }

func (o f64ops[S]) cmp(op kernel.CompareOp, r *plan.Expr) BoolExpr[S] {
	return boolOf[S](plan.Compare(op, o.n, r))
}

func (o f64ops[S]) arith(op kernel.ArithOp, r *plan.Expr) F64Expr[S] {
	return f64Of[S](plan.Arith(op, o.n, r))
}

// i64ops is the method set of everything int64 valued, which is [f64ops] over
// whole numbers.
//
// Integer arithmetic here is Go's: it wraps rather than widening, division
// truncates toward zero so that 7 / 2 is 3, and dividing by zero is an error
// naming the row rather than an infinity. [kernel.Arith] says why.
type i64ops[S any] struct{ n *plan.Expr }

func (o i64ops[S]) expr() *plan.Expr { return o.n }
func (o i64ops[S]) int64Value()      {}

// String returns the expression as it would be written.
func (o i64ops[S]) String() string { return o.n.String() }

// Eq returns whether the value equals v.
func (o i64ops[S]) Eq(v int64) BoolExpr[S] { return o.cmp(kernel.OpEq, plan.Lit(v)) }

// Ne returns whether the value differs from v.
func (o i64ops[S]) Ne(v int64) BoolExpr[S] { return o.cmp(kernel.OpNe, plan.Lit(v)) }

// Lt returns whether the value is less than v.
func (o i64ops[S]) Lt(v int64) BoolExpr[S] { return o.cmp(kernel.OpLt, plan.Lit(v)) }

// Le returns whether the value is less than or equal to v.
func (o i64ops[S]) Le(v int64) BoolExpr[S] { return o.cmp(kernel.OpLe, plan.Lit(v)) }

// Gt returns whether the value is greater than v.
func (o i64ops[S]) Gt(v int64) BoolExpr[S] { return o.cmp(kernel.OpGt, plan.Lit(v)) }

// Ge returns whether the value is greater than or equal to v.
func (o i64ops[S]) Ge(v int64) BoolExpr[S] { return o.cmp(kernel.OpGe, plan.Lit(v)) }

// EqExpr returns whether the value equals the value of x in the same row.
func (o i64ops[S]) EqExpr(x I64Value[S]) BoolExpr[S] { return o.cmp(kernel.OpEq, x.expr()) }

// NeExpr returns whether the value differs from the value of x in the same row.
func (o i64ops[S]) NeExpr(x I64Value[S]) BoolExpr[S] { return o.cmp(kernel.OpNe, x.expr()) }

// LtExpr returns whether the value is less than the value of x in the same row.
func (o i64ops[S]) LtExpr(x I64Value[S]) BoolExpr[S] { return o.cmp(kernel.OpLt, x.expr()) }

// LeExpr returns whether the value is at most the value of x in the same row.
func (o i64ops[S]) LeExpr(x I64Value[S]) BoolExpr[S] { return o.cmp(kernel.OpLe, x.expr()) }

// GtExpr returns whether the value is greater than the value of x in the same
// row.
func (o i64ops[S]) GtExpr(x I64Value[S]) BoolExpr[S] { return o.cmp(kernel.OpGt, x.expr()) }

// GeExpr returns whether the value is at least the value of x in the same row.
func (o i64ops[S]) GeExpr(x I64Value[S]) BoolExpr[S] { return o.cmp(kernel.OpGe, x.expr()) }

// Add returns the value plus v, wrapping the way Go wraps.
func (o i64ops[S]) Add(v int64) I64Expr[S] { return o.arith(kernel.OpAdd, plan.Lit(v)) }

// Sub returns the value minus v.
func (o i64ops[S]) Sub(v int64) I64Expr[S] { return o.arith(kernel.OpSub, plan.Lit(v)) }

// Mul returns the value times v.
func (o i64ops[S]) Mul(v int64) I64Expr[S] { return o.arith(kernel.OpMul, plan.Lit(v)) }

// Div returns the value divided by v, truncated toward zero. Dividing by zero
// is an error naming the row.
func (o i64ops[S]) Div(v int64) I64Expr[S] { return o.arith(kernel.OpDiv, plan.Lit(v)) }

// Mod returns the remainder of the value divided by v.
func (o i64ops[S]) Mod(v int64) I64Expr[S] { return o.arith(kernel.OpMod, plan.Lit(v)) }

// AddExpr returns the value plus the value of x in the same row.
func (o i64ops[S]) AddExpr(x I64Value[S]) I64Expr[S] { return o.arith(kernel.OpAdd, x.expr()) }

// SubExpr returns the value minus the value of x in the same row.
func (o i64ops[S]) SubExpr(x I64Value[S]) I64Expr[S] { return o.arith(kernel.OpSub, x.expr()) }

// MulExpr returns the value times the value of x in the same row.
func (o i64ops[S]) MulExpr(x I64Value[S]) I64Expr[S] { return o.arith(kernel.OpMul, x.expr()) }

// DivExpr returns the value divided by the value of x in the same row.
func (o i64ops[S]) DivExpr(x I64Value[S]) I64Expr[S] { return o.arith(kernel.OpDiv, x.expr()) }

// ModExpr returns the remainder of the value divided by the value of x in the
// same row.
func (o i64ops[S]) ModExpr(x I64Value[S]) I64Expr[S] { return o.arith(kernel.OpMod, x.expr()) }

// AsF64 returns the value as a float64, which is how an int64 column is used
// with a float64 one. Every int64 value has a float64 nearest to it and the two
// are the same number up to 2^53, above which the conversion rounds.
func (o i64ops[S]) AsF64() F64Expr[S] { return f64Of[S](plan.Cast(dtype.Float64, o.n)) }

// IsNull returns whether the value is missing.
func (o i64ops[S]) IsNull() BoolExpr[S] { return boolOf[S](plan.IsNull(o.n)) }

// IsNotNull returns whether the value is there.
func (o i64ops[S]) IsNotNull() BoolExpr[S] { return boolOf[S](plan.IsNotNull(o.n)) }

func (o i64ops[S]) cmp(op kernel.CompareOp, r *plan.Expr) BoolExpr[S] {
	return boolOf[S](plan.Compare(op, o.n, r))
}

func (o i64ops[S]) arith(op kernel.ArithOp, r *plan.Expr) I64Expr[S] {
	return i64Of[S](plan.Arith(op, o.n, r))
}

func f64Of[S any](n *plan.Expr) F64Expr[S] { return F64Expr[S]{f64ops[S]{n}} }

func i64Of[S any](n *plan.Expr) I64Expr[S] { return I64Expr[S]{i64ops[S]{n}} }
