package kuma

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// Expr is anything that turns into a column when a frame is put behind it,
// which is a column handle or an expression built out of one.
//
// The type parameter is the schema, so an expression written against Trade
// cannot be handed to a frame of orders. That check is the compiler's, and it
// is the reason the typed handles exist.
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

	expr() *node
}

// nodeKind is what one step of an expression does.
type nodeKind uint8

const (
	kindColumn    nodeKind = iota // a column of the frame, by name
	kindLiteral                   // a value written in the query
	kindCompare                   // one of the six comparisons
	kindArith                     // one of the five arithmetic operators
	kindAnd                       // three valued and
	kindOr                        // three valued or
	kindNot                       // three valued not
	kindIsNull                    // whether the value is missing
	kindIsNotNull                 // whether the value is there
	kindCast                      // the same values in another type
)

// node is one step of an expression.
//
// It is one struct with a kind rather than an interface per operation. An
// expression is built once and walked once, the tree is a few nodes deep, and
// the shape of it is fixed by this package, so the tag is cheaper to build and
// easier to read than a family of types would be.
type node struct {
	kind nodeKind
	name string           // the column, when kind is kindColumn
	lit  any              // the value, when kind is kindLiteral
	cmp  kernel.CompareOp // the comparison, when kind is kindCompare
	ari  kernel.ArithOp   // the operator, when kind is kindArith
	dt   dtype.DataType   // the type, when kind is kindCast
	l, r *node            // the operands, none for a leaf
}

func colNode(name string) *node { return &node{kind: kindColumn, name: name} }

func litNode(v any) *node { return &node{kind: kindLiteral, lit: v} }

func cmpNode(op kernel.CompareOp, l, r *node) *node {
	return &node{kind: kindCompare, cmp: op, l: l, r: r}
}

func ariNode(op kernel.ArithOp, l, r *node) *node {
	return &node{kind: kindArith, ari: op, l: l, r: r}
}

func logicNode(kind nodeKind, l, r *node) *node {
	return &node{kind: kind, l: l, r: r}
}

func unaryNode(kind nodeKind, l *node) *node {
	return &node{kind: kind, l: l}
}

func castNode(dt dtype.DataType, l *node) *node {
	return &node{kind: kindCast, dt: dt, l: l}
}

// String returns the expression as it would be written, which is what an error
// about it names and what a column built from it is called.
func (n *node) String() string {
	var sb strings.Builder
	n.write(&sb)
	return sb.String()
}

func (n *node) write(sb *strings.Builder) {
	switch n.kind {
	case kindColumn:
		sb.WriteString(n.name)
	case kindLiteral:
		sb.WriteString(literalText(n.lit))
	case kindCompare:
		n.infix(sb, n.cmp.String())
	case kindArith:
		n.infix(sb, n.ari.String())
	case kindAnd:
		n.infix(sb, "and")
	case kindOr:
		n.infix(sb, "or")
	case kindNot:
		sb.WriteString("(not ")
		n.l.write(sb)
		sb.WriteByte(')')
	case kindIsNull:
		n.suffix(sb, "is null")
	case kindCast:
		n.suffix(sb, "as "+n.dt.String())
	default:
		n.suffix(sb, "is not null")
	}
}

// infix writes a two sided step in brackets, so that reading the text back
// gives the tree that produced it rather than whatever Go's precedence would
// have made of it. Every step but a leaf is written that way, for the same
// reason.
func (n *node) infix(sb *strings.Builder, op string) {
	sb.WriteByte('(')
	n.l.write(sb)
	sb.WriteByte(' ')
	sb.WriteString(op)
	sb.WriteByte(' ')
	n.r.write(sb)
	sb.WriteByte(')')
}

// suffix writes a one sided step whose operator comes after the value.
func (n *node) suffix(sb *strings.Builder, op string) {
	sb.WriteByte('(')
	n.l.write(sb)
	sb.WriteByte(' ')
	sb.WriteString(op)
	sb.WriteByte(')')
}

// literalText is a literal as it would be written in Go, which is why a string
// is quoted and a time is in RFC 3339.
func literalText(v any) string {
	switch v := v.(type) {
	case nil:
		return "null"
	case bool:
		return strconv.FormatBool(v)
	case string:
		return strconv.Quote(v)
	case time.Time:
		return v.Format(time.RFC3339Nano)
	case float32:
		return strconv.FormatFloat(float64(v), 'g', -1, 32)
	case float64:
		return strconv.FormatFloat(v, 'g', -1, 64)
	case []byte:
		return strconv.Quote(string(v))
	default:
		return fmt.Sprint(v)
	}
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
func eval(n *node, look lookupFunc, hint dtype.DataType) (*array.Chunked, error) {
	switch n.kind {
	case kindColumn:
		return look(n.name)
	case kindLiteral:
		return literal(n.lit, hint)
	case kindCompare:
		a, b, err := operands(n, look)
		if err != nil {
			return nil, err
		}
		return kernel.Compare(a, b, n.cmp)
	case kindArith:
		a, b, err := operands(n, look)
		if err != nil {
			return nil, err
		}
		return kernel.Arith(a, b, n.ari)
	case kindAnd, kindOr:
		a, b, err := operands(n, look)
		if err != nil {
			return nil, err
		}
		if n.kind == kindAnd {
			return kernel.And(a, b)
		}
		return kernel.Or(a, b)
	case kindNot:
		a, err := eval(n.l, look, nil)
		if err != nil {
			return nil, err
		}
		return kernel.Not(a)
	case kindIsNull:
		a, err := eval(n.l, look, nil)
		if err != nil {
			return nil, err
		}
		return kernel.IsNull(a), nil
	case kindCast:
		a, err := eval(n.l, look, nil)
		if err != nil {
			return nil, err
		}
		return kernel.Cast(a, n.dt)
	default:
		a, err := eval(n.l, look, nil)
		if err != nil {
			return nil, err
		}
		return kernel.IsNotNull(a), nil
	}
}

// operands works out both sides of a two sided step, doing the side that has a
// type first so that the literal on the other side knows what to become.
func operands(n *node, look lookupFunc) (a, b *array.Chunked, err error) {
	if n.l.kind == kindLiteral && n.r.kind != kindLiteral {
		b, err = eval(n.r, look, nil)
		if err != nil {
			return nil, nil, err
		}
		a, err = eval(n.l, look, b.DType())
		if err != nil {
			return nil, nil, err
		}
		return a, b, nil
	}

	a, err = eval(n.l, look, nil)
	if err != nil {
		return nil, nil, err
	}
	b, err = eval(n.r, look, a.DType())
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
		ts, ok := hint.(dtype.Timestamp)
		if !ok {
			// The literal is on its own, or against a column that is not a
			// timestamp. Nanoseconds is what a time.Time holds, and what the
			// other side makes of that is the other side's business.
			ts = dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "UTC"}
		}
		return timeLiteral(t, ts)
	}

	dt, err := literalType(v)
	if err != nil {
		return nil, err
	}
	want := dt
	if hint != nil {
		want, err = dtype.CoerceLiteral(hint, dt)
		if err != nil {
			return nil, fmt.Errorf("kuma: %w", err)
		}
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

// literalType is the column type a value written in a query has on its own,
// before it meets a column.
func literalType(v any) (dtype.DataType, error) {
	switch v.(type) {
	case nil:
		return dtype.Null, nil
	case bool:
		return dtype.Bool, nil
	case int:
		return dtype.Int64, nil
	case int8:
		return dtype.Int8, nil
	case int16:
		return dtype.Int16, nil
	case int32:
		return dtype.Int32, nil
	case int64:
		return dtype.Int64, nil
	case uint:
		return dtype.Uint64, nil
	case uint8:
		return dtype.Uint8, nil
	case uint16:
		return dtype.Uint16, nil
	case uint32:
		return dtype.Uint32, nil
	case uint64:
		return dtype.Uint64, nil
	case float32:
		return dtype.Float32, nil
	case float64:
		return dtype.Float64, nil
	case string:
		return dtype.String, nil
	case []byte:
		return dtype.Binary, nil
	default:
		return nil, fmt.Errorf("kuma: %T is not a value a column can hold: %w", v, ErrWrongType)
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
// A type that is not one of these has already been refused by [literalType],
// which is asked first and is the same list, so getting here with one is a
// mistake in this file rather than something a caller can write.
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
