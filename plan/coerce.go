package plan

import (
	"github.com/tamnd/kuma/dtype"
)

// Coerce writes into the plan the type each value in it is used at.
//
// A value written in a query has no type of its own. Someone who writes 100
// against a float64 column means the number and not the int64, which is why a
// literal takes the type of the column it is used with rather than dragging the
// column up to its own. That rule is [LiteralTypeAgainst] and it is the same
// rule wherever it is asked from, so the answer is the same every time and the
// engine works it out again for every batch it runs over.
//
// This pass asks it once. A literal that turns out to be used at a type other
// than the one its Go value already has is replaced by [LitAs] of that type,
// which is the same value with the answer attached. After it the plan says what
// it does: a filter over a float64 column reads
//
//	Filter (price > (100 as float64))
//
// and the comparison happens in float64 because the plan says float64, rather
// than because of a rule the reader has to know and the engine has to apply.
//
// It is worth saying plainly that this does not make a query faster. The engine
// still builds the one value column the same way, since the value keeps the Go
// type it was written with and only the answer about it is written down. What
// the pass buys is that the answer is in the plan: every other pass can be
// checked by reading the plan it produces, and a coercion that lives only in the
// evaluator is the one piece of what a query does that an explain cannot show
// and a test cannot assert on. It is also what a plan written out to be read
// somewhere else needs, since the type a value is used at is otherwise
// recoverable only by having the schema and this package's rules to hand.
//
// The value keeps its Go type on purpose. Converting it would make [LitAs] of a
// float64 hundred and [Lit] of the same hundred two expressions that say the
// same thing, which is the one thing this package does not allow. A hundred
// written plainly and a hundred used as a float64 are different, and a hundred
// used as a float64 and a hundred written as one are not.
//
// A value is left alone in three cases. One already at the type it is used at
// has nothing to say that the value does not say itself, so an int64 against an
// int64 column stays plain. A missing value is missing in every type and naming
// one would be noise. And a value the pass cannot work a type out for is left
// exactly as written, because the plan check has a better error about it than
// this pass could invent, and it is the expression the caller wrote that the
// error should name.
//
// A type already written down is left alone too, which is the same rule seen
// from the other side. A pass asks the same question the plan it was given
// asks, and a caller who writes [LitAs] by hand has said which type they mean,
// so quietly moving it to another one would answer a different question. It is
// also what makes the pass idempotent, which is what running the passes to
// fixpoint needs.
func Coerce(n *Node) (*Node, error) {
	return rewriteExprs(n, coerceExpr)
}

// coerceExpr returns the expression with the type of each value in it written
// down, and the expression it was given when there was nothing to write.
//
// Only a two sided step can say anything, since the type a value takes is the
// type of what it is used with and a step with one operand has nothing to use
// it with. Every other step is rebuilt out of what its operands came back as,
// which is [rebuilt]'s job, and an operand that did not move rebuilds into the
// step that was already there.
func coerceExpr(e *Expr, s dtype.Schema) *Expr {
	return rebuilt(e, func(x *Expr) (*Expr, bool) {
		if x.kind != KindCompare && x.kind != KindArith {
			return nil, false
		}
		l, r := coerceOperands(coerceExpr(x.l, s), coerceExpr(x.r, s), s)
		switch {
		case l == x.l && r == x.r:
			return x, true
		case x.kind == KindCompare:
			return Compare(x.cmp, l, r), true
		default:
			return Arith(x.ari, l, r), true
		}
	})
}

// coerceOperands writes the type onto whichever side of a step is a value and
// takes its type from the other one.
//
// It is [operandTypes]' own rule about which side that is. A step with a value
// on each side has no column to take a type from, and one with a column on each
// side has no value to give a type to, so both are left as they are.
func coerceOperands(l, r *Expr, s dtype.Schema) (*Expr, *Expr) {
	switch {
	case l.kind == KindLiteral && r.kind != KindLiteral:
		return coerceLiteral(l, r, s), r
	case r.kind == KindLiteral && l.kind != KindLiteral:
		return l, coerceLiteral(r, l, s)
	default:
		return l, r
	}
}

// coerceLiteral returns the value written down as the type it is used at when
// there is one to write, and the value it was given otherwise. The against
// argument is the other side of the step, which is what decides the type.
func coerceLiteral(lit, against *Expr, s dtype.Schema) *Expr {
	if lit.dt != nil || lit.lit == nil {
		// Written down already, or a missing value, which is missing whatever
		// type it is written as.
		return lit
	}

	dt, err := TypeOf(against, s)
	if err != nil {
		return lit
	}
	want, err := LiteralTypeAgainst(lit.lit, dt)
	if err != nil {
		return lit
	}
	if have, ok := LiteralType(lit.lit); !ok || dtype.Equal(want, have) {
		return lit
	}
	return LitAs(want, lit.lit)
}
