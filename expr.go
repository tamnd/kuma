package kuma

import (
	"fmt"
	"time"

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
		return literal(n.Value(), hint)
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

// literal builds the one value column that a value written in a query becomes.
//
// A column of one value is what the kernels take as the other side of an
// operation over a whole column, so there is no scalar path here to keep in
// step with the column path.
//
// The hint is the type of the column the literal is being used with, and it
// decides what the literal becomes. An integer literal takes the column's own
// integer type, so comparing a uint32 column against 0 does not turn it into an
// int64 column. What is refused is refused by [dtype.CoerceLiteral], which is
// where the rule about a float literal against an integer column lives.
func literal(v any, hint dtype.DataType) (*array.Chunked, error) {
	if t, ok := v.(time.Time); ok {
		// A time is the one value with no column type of its own to coerce, so
		// what it becomes is a rule rather than a coercion.
		return timeLiteral(t, plan.TimeLiteralType(hint))
	}

	want, err := plan.LiteralTypeAgainst(v, hint)
	if err != nil {
		return nil, err
	}

	dt, err := plan.LiteralTypeAgainst(v, nil)
	if err != nil {
		return nil, err
	}

	c, err := litColumn(dt, v)
	if err != nil {
		return nil, err
	}
	if dtype.Equal(want, dt) {
		return c, nil
	}
	// The cast is where a literal that does not fit the column is caught, so
	// comparing an int8 column against 300 says so rather than comparing
	// against 44.
	return kernel.Cast(c, want)
}

// timeLiteral builds a wall clock time as a count in the column's own unit.
//
// This is a separate path because a time.Time is not a number and the cast
// kernel does not rescale a timestamp yet. Doing it here also means the literal
// lands in the column's unit exactly or not at all: a time with nanoseconds in
// it against a column of seconds is an error rather than a silent truncation,
// for the same reason 1.5 against an integer column is.
func timeLiteral(t time.Time, ts dtype.Timestamp) (*array.Chunked, error) {
	var count int64
	switch ts.Unit {
	case dtype.Second:
		count = t.Unix()
	case dtype.Millisecond:
		count = t.UnixMilli()
	case dtype.Microsecond:
		count = t.UnixMicro()
	default:
		count = t.UnixNano()
	}
	if !t.Equal(timeOfCount(count, ts.Unit)) {
		return nil, fmt.Errorf("kuma: the time %s does not fit a %s column exactly, truncate it first",
			t.Format(time.RFC3339Nano), ts)
	}
	return litColumn(ts, count)
}

// timeOfCount is timeLiteral's conversion the other way, which is how the
// conversion is checked for having lost anything.
func timeOfCount(count int64, unit dtype.TimeUnit) time.Time {
	switch unit {
	case dtype.Second:
		return time.Unix(count, 0)
	case dtype.Millisecond:
		return time.UnixMilli(count)
	case dtype.Microsecond:
		return time.UnixMicro(count)
	default:
		return time.Unix(0, count)
	}
}

// litColumn builds a column holding the single value v, which is what a literal
// written in a query becomes.
func litColumn(dt dtype.DataType, v any) (*array.Chunked, error) {
	b, err := array.NewBuilder(dt)
	if err != nil {
		return nil, fmt.Errorf("kuma: %w", err)
	}
	appendLiteral(b, v)

	c, err := array.NewChunked(dt, b.Finish())
	if err != nil {
		return nil, fmt.Errorf("kuma: %w", err)
	}
	return c, nil
}

// appendLiteral writes one value of whatever type it turns out to be.
//
// A type that is not one of these has already been refused by
// [plan.LiteralType], which is asked first, or turned into a count by
// [timeLiteral], so getting here with one is a mistake in this file rather than
// something a caller can write.
func appendLiteral(b *array.Builder, v any) {
	switch v := v.(type) {
	case nil:
		b.AppendNull()
	case bool:
		b.AppendBool(v)
	case int:
		b.Append(int64(v))
	case int8:
		b.Append(v)
	case int16:
		b.Append(v)
	case int32:
		b.Append(v)
	case int64:
		b.Append(v)
	case uint:
		b.Append(uint64(v))
	case uint8:
		b.Append(v)
	case uint16:
		b.Append(v)
	case uint32:
		b.Append(v)
	case uint64:
		b.Append(v)
	case float32:
		b.Append(v)
	case float64:
		b.Append(v)
	case string:
		b.AppendString(v)
	case []byte:
		b.AppendBytes(v)
	default:
		panic(fmt.Sprintf("kuma: no way to write a literal of type %T", v))
	}
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
