package plan

import (
	"bytes"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// Kind is what one step of an expression does.
type Kind uint8

// The kinds of step an expression can be made of.
const (
	KindColumn    Kind = iota // a column of the frame, by name
	KindLiteral               // a value written in the query
	KindCompare               // one of the six comparisons
	KindArith                 // one of the five arithmetic operators
	KindAnd                   // three valued and
	KindOr                    // three valued or
	KindNot                   // three valued not
	KindIsNull                // whether the value is missing
	KindIsNotNull             // whether the value is there
	KindCast                  // the same values in another type
)

// Expr is one step of an expression, and by way of its operands the whole of
// one.
//
// It is one struct with a kind rather than an interface per operation. An
// expression is built once and walked many times, the tree is a few nodes deep,
// and the set of steps is fixed by this package, so the tag is cheaper to build
// and easier to read than a family of types would be. A walk over it is a switch
// on [Expr.Kind] rather than a virtual call.
//
// Every field is read through a method and none of them can be written, because
// an expression that changed after it was built would break both of the rules in
// the package comment at once. The constructors are the only way to make one,
// and two calls that describe the same step return the same *Expr. See intern.go
// for what that buys and what it costs.
type Expr struct {
	kind Kind
	name string           // the column, when kind is KindColumn
	lit  any              // the value, when kind is KindLiteral
	cmp  kernel.CompareOp // the comparison, when kind is KindCompare
	ari  kernel.ArithOp   // the operator, when kind is KindArith
	dt   dtype.DataType   // the type, when kind is KindCast
	l, r *Expr            // the operands, none for a leaf
}

// The constructors. Every one of them goes through [intern], which is what makes
// two equal expressions one expression. They describe the step rather than
// building it, so that writing one the program has written before costs a
// lookup and no allocation at all.

// Col is the column of that name in whatever frame the expression is run
// against. The name is not checked here, since there is no frame yet to check it
// against, and a name no frame has is an error when the plan is bound.
func Col(name string) *Expr {
	return intern(key{kind: KindColumn, name: name}, nil, nil)
}

// Lit is a value written in the query rather than read from a column.
//
// The value keeps the Go type it was written with, and what that becomes is
// decided when the literal meets the column it is used with. Anything a column
// cannot hold is accepted here and refused then, so that the error names the
// expression it came from.
func Lit(v any) *Expr {
	if b, ok := v.([]byte); ok {
		// The one literal that refers to memory the caller still holds. It is
		// copied so that an expression cannot change under whoever built it,
		// which is what everything else here assumes, and so that what is in
		// the table stays equal to the key it went in under.
		v = bytes.Clone(b)
	}

	lit, ok := litKey(v)
	if !ok {
		// A value that cannot be looked up again once it is stored. Leaving it
		// out of the table costs nothing but the sharing, and putting it in
		// would mean an entry that can never be found and never be removed.
		return build(key{kind: KindLiteral}, nil, v)
	}
	return intern(key{kind: KindLiteral, lit: lit}, nil, v)
}

// Compare is one of the six comparisons between two values.
func Compare(op kernel.CompareOp, l, r *Expr) *Expr {
	return intern(key{kind: KindCompare, cmp: op, l: l, r: r}, nil, nil)
}

// Arith is one of the five arithmetic operators over two values.
func Arith(op kernel.ArithOp, l, r *Expr) *Expr {
	return intern(key{kind: KindArith, ari: op, l: l, r: r}, nil, nil)
}

// And is the three valued and of two conditions.
func And(l, r *Expr) *Expr { return intern(key{kind: KindAnd, l: l, r: r}, nil, nil) }

// Or is the three valued or of two conditions.
func Or(l, r *Expr) *Expr { return intern(key{kind: KindOr, l: l, r: r}, nil, nil) }

// Not is the three valued negation of a condition, which leaves a missing value
// missing.
func Not(l *Expr) *Expr { return intern(key{kind: KindNot, l: l}, nil, nil) }

// IsNull is whether the value is missing, which is never itself missing.
func IsNull(l *Expr) *Expr { return intern(key{kind: KindIsNull, l: l}, nil, nil) }

// IsNotNull is whether the value is there, which is never itself missing.
func IsNotNull(l *Expr) *Expr { return intern(key{kind: KindIsNotNull, l: l}, nil, nil) }

// Cast is the same values in another type.
func Cast(dt dtype.DataType, l *Expr) *Expr {
	// A type goes into the key by the name it prints rather than by its value.
	// The dtype package promises that two types print the same name if and
	// only if they are equal, and a struct type holds a slice of fields, which
	// a map key cannot.
	return intern(key{kind: KindCast, dt: dt.String(), l: l}, dt, nil)
}

// Kind is what this step does, and which of the other accessors mean anything.
func (e *Expr) Kind() Kind { return e.kind }

// Name is the column this step reads, for a step of kind [KindColumn] and the
// empty string for any other.
func (e *Expr) Name() string { return e.name }

// Value is the value this step is, for a step of kind [KindLiteral] and nil for
// any other. A nil is also what a literal null is, so ask the kind first.
//
// A binary literal is returned as the slice the expression holds rather than a
// copy of it, so treat it as read only. Writing to it would change an expression
// that other plans may be sharing.
func (e *Expr) Value() any { return e.lit }

// CompareOp is the comparison this step makes, for a step of kind
// [KindCompare].
func (e *Expr) CompareOp() kernel.CompareOp { return e.cmp }

// ArithOp is the operator this step applies, for a step of kind [KindArith].
func (e *Expr) ArithOp() kernel.ArithOp { return e.ari }

// DType is the type this step casts to, for a step of kind [KindCast] and nil
// for any other. It is not the type the step produces, which depends on the
// frame and is not known until the plan is bound.
func (e *Expr) DType() dtype.DataType { return e.dt }

// Left is the first operand, or nil for a leaf.
func (e *Expr) Left() *Expr { return e.l }

// Right is the second operand, or nil for a leaf and for a one sided step.
func (e *Expr) Right() *Expr { return e.r }

// Columns returns the names of the columns the expression reads, in the order
// they are first written and with no name twice.
//
// It is what a projection pushdown asks for, and what an explain shows when it
// says which columns a step of a query depends on. An expression made of
// nothing but literals reads no columns and gets nothing back.
//
// A name appearing here does not say the column exists. Checking the expression
// against a schema is what says that.
func (e *Expr) Columns() []string {
	var names []string
	e.eachColumn(func(name string) {
		if !slices.Contains(names, name) {
			names = append(names, name)
		}
	})
	return names
}

// eachColumn calls yield with the name of every column the expression reads, in
// order and with the repeats left in. It is the walk both [Expr.Columns] and
// the passes are written on, since one of them wants a list and the others want
// a set and none of them wants to write the walk again.
func (e *Expr) eachColumn(yield func(string)) {
	switch {
	case e == nil:
		return
	case e.kind == KindColumn:
		yield(e.name)
		return
	}
	e.l.eachColumn(yield)
	e.r.eachColumn(yield)
}

// conjuncts returns the parts of an expression that are joined by and, which
// are the parts a predicate pushdown moves one at a time.
//
// An expression that is not an and is one conjunct, which is itself. The parts
// come back in the order they are written, so rebuilding them with and in that
// order gives back an expression that says the same thing, and gives back this
// very expression when it was written left to right to begin with.
func conjuncts(e *Expr) []*Expr {
	if e == nil || e.kind != KindAnd {
		return []*Expr{e}
	}
	return append(conjuncts(e.l), conjuncts(e.r)...)
}

// substitute returns the expression with every column named in by replaced by
// what it stands for.
//
// It is what pushing a predicate under a projection means: a filter written
// over the columns a step produces has to be rewritten over the columns that
// step reads before it can run underneath it. An expression with none of those
// names in it comes back as itself, since a rebuilt expression that says the
// same thing is the same expression.
func substitute(e *Expr, by map[string]*Expr) *Expr {
	switch e.kind {
	case KindColumn:
		if r, ok := by[e.name]; ok {
			return r
		}
		return e
	case KindLiteral:
		return e
	case KindCompare:
		return Compare(e.cmp, substitute(e.l, by), substitute(e.r, by))
	case KindArith:
		return Arith(e.ari, substitute(e.l, by), substitute(e.r, by))
	case KindAnd:
		return And(substitute(e.l, by), substitute(e.r, by))
	case KindOr:
		return Or(substitute(e.l, by), substitute(e.r, by))
	case KindNot:
		return Not(substitute(e.l, by))
	case KindIsNull:
		return IsNull(substitute(e.l, by))
	case KindIsNotNull:
		return IsNotNull(substitute(e.l, by))
	default:
		return Cast(e.dt, substitute(e.l, by))
	}
}

// String returns the expression as it would be written, which is what an error
// about it names and what a column built from it is called.
func (e *Expr) String() string {
	var sb strings.Builder
	e.write(&sb)
	return sb.String()
}

func (e *Expr) write(sb *strings.Builder) {
	switch e.kind {
	case KindColumn:
		sb.WriteString(e.name)
	case KindLiteral:
		sb.WriteString(LiteralText(e.lit))
	case KindCompare:
		e.infix(sb, e.cmp.String())
	case KindArith:
		e.infix(sb, e.ari.String())
	case KindAnd:
		e.infix(sb, "and")
	case KindOr:
		e.infix(sb, "or")
	case KindNot:
		sb.WriteString("(not ")
		e.l.write(sb)
		sb.WriteByte(')')
	case KindIsNull:
		e.suffix(sb, "is null")
	case KindCast:
		e.suffix(sb, "as "+e.dt.String())
	default:
		e.suffix(sb, "is not null")
	}
}

// infix writes a two sided step in brackets, so that reading the text back
// gives the tree that produced it rather than whatever Go's precedence would
// have made of it. Every step but a leaf is written that way, for the same
// reason.
func (e *Expr) infix(sb *strings.Builder, op string) {
	sb.WriteByte('(')
	e.l.write(sb)
	sb.WriteByte(' ')
	sb.WriteString(op)
	sb.WriteByte(' ')
	e.r.write(sb)
	sb.WriteByte(')')
}

// suffix writes a one sided step whose operator comes after the value.
func (e *Expr) suffix(sb *strings.Builder, op string) {
	sb.WriteByte('(')
	e.l.write(sb)
	sb.WriteByte(' ')
	sb.WriteString(op)
	sb.WriteByte(')')
}

// LiteralText is a literal as it would be written in Go, which is why a string
// is quoted and a time is in RFC 3339. It is exported because an operator that
// prints a value of its own, such as the bounds of a slice, should print it the
// way an expression does.
func LiteralText(v any) string {
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
