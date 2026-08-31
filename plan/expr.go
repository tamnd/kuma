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
	// A binary literal is the one that refers to memory the caller still holds,
	// and [literal] copies it so that an expression cannot change under whoever
	// built it, which is what everything else here assumes, and so that what is
	// in the table stays equal to the key it went in under.
	return literal(v, nil)
}

// LitAs is a value written in the query, at the type it is used at.
//
// It is what [Coerce] writes in place of the literal a caller wrote, once it
// knows the column that literal is used with: 100 against a float64 column is a
// float64, and saying so in the plan means the engine reads the answer rather
// than working it out again for every batch it runs over. It is also how a plan
// read back from somewhere else says what it meant, since the type a literal
// takes is a fact about the query rather than about the value.
//
// The type is not checked here, since there is no schema yet to check it
// against. A type the value cannot take, or one the column it is used with
// cannot be compared to, is an error when the plan is checked, the same as any
// other.
func LitAs(dt dtype.DataType, v any) *Expr {
	if dt == nil {
		return Lit(v)
	}
	return literal(v, dt)
}

// literal builds a value written in the query, at type dt when it has one.
func literal(v any, dt dtype.DataType) *Expr {
	if b, ok := v.([]byte); ok {
		v = bytes.Clone(b)
	}

	name := ""
	if dt != nil {
		// A type goes into the key by the name it prints, for the reason [Cast]
		// gives.
		name = dt.String()
	}

	lit, ok := litKey(v)
	if !ok {
		// A value that cannot be looked up again once it is stored. Leaving it
		// out of the table costs nothing but the sharing, and putting it in
		// would mean an entry that can never be found and never be removed.
		return build(key{kind: KindLiteral, dt: name}, dt, v)
	}
	return intern(key{kind: KindLiteral, lit: lit, dt: name}, dt, v)
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

// DType is the type this step casts to, for a step of kind [KindCast], and the
// type a value is used at for a literal [Coerce] has been over. It is nil for
// any other step and for a literal that has not been coerced yet, which is one
// whose type is still whatever it turns out to be used with.
//
// For a cast it is not the type the step produces, which depends on the frame
// and is not known until the plan is bound.
func (e *Expr) DType() dtype.DataType { return e.dt }

// LiteralHint returns the type a literal is to be worked out at, which is the
// type [Coerce] wrote onto it when it has one and dt otherwise.
//
// It is what the engine asks before it builds the one value column a literal
// becomes, and dt is the type the other side of the step turned out to be. A
// literal the optimizer has been over says what it is, and one it has not takes
// its type from what it is used with, which is the rule this exists to keep in
// one place rather than in each of the three walks that need it.
func LiteralHint(e *Expr, dt dtype.DataType) dtype.DataType {
	if e.kind == KindLiteral && e.dt != nil {
		return e.dt
	}
	return dt
}

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
	return rebuilt(e, func(x *Expr) (*Expr, bool) {
		if x.kind != KindColumn {
			return nil, false
		}
		r, ok := by[x.name]
		return r, ok
	})
}

// rebuilt returns the expression with every step that by has an answer for
// replaced by that answer, and every other step built again out of what its
// operands came back as.
//
// It is the one walk that rewrites an expression, and the two passes that do
// that want different things from it: one is replacing named columns and the
// other is replacing whole subexpressions by the column that now holds them.
// Building the same expression again gives back the expression that was there,
// since two expressions that say the same thing are one expression, so a walk
// that replaces nothing costs a walk and no allocation.
func rebuilt(e *Expr, by func(*Expr) (*Expr, bool)) *Expr {
	if r, ok := by(e); ok {
		return r
	}
	switch e.kind {
	case KindColumn, KindLiteral:
		return e
	case KindCompare:
		return Compare(e.cmp, rebuilt(e.l, by), rebuilt(e.r, by))
	case KindArith:
		return Arith(e.ari, rebuilt(e.l, by), rebuilt(e.r, by))
	case KindAnd:
		return And(rebuilt(e.l, by), rebuilt(e.r, by))
	case KindOr:
		return Or(rebuilt(e.l, by), rebuilt(e.r, by))
	case KindNot:
		return Not(rebuilt(e.l, by))
	case KindIsNull:
		return IsNull(rebuilt(e.l, by))
	case KindIsNotNull:
		return IsNotNull(rebuilt(e.l, by))
	default:
		return Cast(e.dt, rebuilt(e.l, by))
	}
}

// eachStep calls yield with every step of an expression, the outermost first
// and the repeats left in.
//
// A step that appears twice is yielded twice, which is the whole point of it:
// counting how often a value is written is how the pass that works each one out
// once finds the ones worth working out at all.
func (e *Expr) eachStep(yield func(*Expr)) {
	if e == nil {
		return
	}
	yield(e)
	e.l.eachStep(yield)
	e.r.eachStep(yield)
}

// String returns the expression as it would be written, which is what an error
// about it names and what a column built from it is called.
//
// An expression that is not there reads as a question mark rather than being a
// panic. An operator with a hole in it is a plan the check has something to say
// about, and the message it gives is one that prints the plan, so the printer
// has to survive the plans it is most often asked to print.
func (e *Expr) String() string {
	var sb strings.Builder
	e.write(&sb)
	return sb.String()
}

func (e *Expr) write(sb *strings.Builder) {
	if e == nil {
		sb.WriteByte('?')
		return
	}

	switch e.kind {
	case KindColumn:
		sb.WriteString(e.name)
	case KindLiteral:
		if e.dt == nil {
			sb.WriteString(LiteralText(e.lit))
			break
		}
		// A coerced literal is written the way a cast is, because it is one:
		// the pass has worked out the type this value is used at and the plan
		// now says so. A plan that reads the same as the one it came from would
		// be an explain claiming a change it cannot show.
		sb.WriteByte('(')
		sb.WriteString(LiteralText(e.lit))
		sb.WriteString(" as ")
		sb.WriteString(e.dt.String())
		sb.WriteByte(')')
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
