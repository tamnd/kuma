package plan

import (
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// Fold works out at plan time the parts of an expression that the data does not
// decide.
//
// A step whose operands are all values written in the query has one answer for
// every row of every frame it will ever meet, so working it out once here is
// working it out at all. A comparison against 100 times 1.1 becomes a
// comparison against 110, a condition that always holds stops being a condition,
// and a cast of a column to the type it already has goes away.
//
// The answer is worked out by running the same kernel the engine would have
// run, over the same one value column [LiteralColumn] would have built. A pass
// with its own arithmetic would be shorter and would be wrong the first time a
// kernel changed its mind about an overflow or a division by zero, and it would
// be wrong quietly. This one cannot drift, because there is nothing for it to
// drift from.
//
// Three rules keep it honest.
//
// The first is that a folded expression has to have the type the written one
// had. A literal takes its type from what it is used with, so folding one step
// changes what the step above it sees: 2 plus 3 written out is an int64 next to
// an int8 column, and 5 on its own is an int8. That is a different plan, not a
// faster one, so the pass works out the type of the expression as written and
// the type of the one it made, and keeps its answer only when the two agree.
//
// The second is that an expression has to read at least one column, since a
// column of one value is not a column of a frame. That is why the rules that
// would replace a whole condition by a value are not here: false and anything
// is false, but a filter whose condition is false reads nothing and the engine
// has nowhere to get a row count from. Those belong with an operator that says
// the answer is no rows, and the plan has no such operator yet.
//
// The third is about names. A projection and an aggregation with no name of
// their own are called after the expression they hold, so folding one would
// quietly rename the column it produces. Both have somewhere to write the old
// name down and this pass writes it there. A group key has nowhere, so the keys
// of an aggregate are left as they were written.
func Fold(n *Node) (*Node, error) {
	if n.op == OpScan {
		return n, nil
	}

	l, err := Fold(n.l)
	if err != nil {
		return nil, err
	}
	r := n.r
	if r != nil {
		if r, err = Fold(r); err != nil {
			return nil, err
		}
	}

	m := n.withInputs(l, r)
	switch m.op {
	case OpProject, OpFilter, OpAggregate, OpSort, OpDistinct:
	default:
		// A limit and an explode hold no expression, and the keys of a join are
		// typed against a schema each rather than one, which is more machinery
		// than a pair of column names is worth.
		return m, nil
	}

	s, err := m.l.Schema()
	if err != nil {
		return nil, err
	}
	return foldNode(m, s), nil
}

// foldNode folds the expressions of one operator against the schema of its
// input, and returns the operator itself when none of them moved.
func foldNode(n *Node, s dtype.Schema) *Node {
	switch n.op {
	case OpProject:
		cols, moved := n.cols, false
		for i, p := range n.cols {
			e := foldExpr(p.Expr, s)
			if e == p.Expr {
				continue
			}
			if !moved {
				cols, moved = append([]Projection(nil), n.cols...), true
			}
			if cols[i].As == "" {
				// A projection with no name of its own is called after the
				// expression it holds, so folding one would rename the column
				// it produces. Writing down the name it had keeps the fold and
				// the name both.
				cols[i].As = p.Expr.String()
			}
			cols[i].Expr = e
		}
		if !moved {
			return n
		}
		return Project(n.l, cols)

	case OpFilter:
		pred := foldExpr(n.pred, s)
		if pred == n.pred {
			return n
		}
		return Filter(n.l, pred)

	case OpAggregate:
		// The keys are left as they were written. A group key is the one
		// expression with no name to be given, since the column it produces is
		// called after it, so there is nowhere to write down the name a fold
		// would take away.
		aggs, moved := n.aggs, false
		for i, a := range n.aggs {
			if a.Expr == nil {
				// A size counts rows without reading a column, so it holds no
				// expression to fold.
				continue
			}
			e := foldExpr(a.Expr, s)
			if e == a.Expr {
				continue
			}
			if !moved {
				aggs, moved = append([]Agg(nil), n.aggs...), true
			}
			if aggs[i].As == "" {
				aggs[i].As = a.Name()
			}
			aggs[i].Expr = e
		}
		if !moved {
			return n
		}
		return Aggregate(n.l, n.by, aggs)

	case OpSort:
		keys, moved := n.sort, false
		for i, k := range n.sort {
			e := foldExpr(k.Expr, s)
			if e == k.Expr {
				continue
			}
			if !moved {
				keys, moved = append([]SortKey(nil), n.sort...), true
			}
			keys[i].Expr = e
		}
		if !moved {
			return n
		}
		return Sort(n.l, keys)

	default:
		by, moved := foldAll(n.by, s)
		if !moved {
			return n
		}
		return Distinct(n.l, by)
	}
}

// foldAll folds a list of expressions, and says whether any of them moved. It
// returns the list it was given when none did, since a pass that changed nothing
// has to hand back what it was given.
func foldAll(exprs []*Expr, s dtype.Schema) ([]*Expr, bool) {
	out, moved := exprs, false
	for i, e := range exprs {
		f := foldExpr(e, s)
		if f == e {
			continue
		}
		if !moved {
			out, moved = append([]*Expr(nil), exprs...), true
		}
		out[i] = f
	}
	return out, moved
}

// foldExpr folds one expression and keeps the answer only when it is the same
// expression to the type checker, which is the first of the rules [Fold]
// describes.
func foldExpr(e *Expr, s dtype.Schema) *Expr {
	out, _ := folded(e, s)
	if out == e || !sameType(e, out, s) {
		return e
	}
	return out
}

// sameType reports whether two expressions come out as the same type over the
// same schema. An expression that does not type at all is not the same as one
// that does, and two that both fail are left alone rather than swapped, since
// the error a caller is shown should be about what they wrote.
func sameType(a, b *Expr, s dtype.Schema) bool {
	at, err := TypeOf(a, s)
	if err != nil {
		return false
	}
	bt, err := TypeOf(b, s)
	if err != nil {
		return false
	}
	return dtype.Equal(at, bt)
}

// folded returns the expression with what can be worked out worked out, and the
// one value column it comes to when every leaf under it is a value written in
// the query. The column is nil for anything that reads a column, which is what
// tells a step above whether there is anything to fold.
func folded(e *Expr, s dtype.Schema) (*Expr, *array.Chunked) {
	switch e.kind {
	case KindColumn:
		return e, nil

	case KindLiteral:
		c, err := LiteralColumn(e.lit, nil)
		if err != nil {
			return e, nil
		}
		return e, c

	case KindCompare, KindArith:
		l, lc := folded(e.l, s)
		r, rc := folded(e.r, s)
		out := e
		if l != e.l || r != e.r {
			if e.kind == KindCompare {
				out = Compare(e.cmp, l, r)
			} else {
				out = Arith(e.ari, l, r)
			}
		}
		if lc == nil || rc == nil {
			return out, nil
		}
		a, b, err := pair(l, lc, r, rc)
		if err != nil {
			return out, nil
		}
		var c *array.Chunked
		if e.kind == KindCompare {
			c, err = kernel.Compare(a, b, e.cmp)
		} else {
			c, err = kernel.Arith(a, b, e.ari)
		}
		if err != nil {
			return out, nil
		}
		return asLiteral(out, c)

	case KindAnd, KindOr:
		l, lc := folded(e.l, s)
		r, rc := folded(e.r, s)
		out := e
		if l != e.l || r != e.r {
			if e.kind == KindAnd {
				out = And(l, r)
			} else {
				out = Or(l, r)
			}
		}
		if lc != nil && rc != nil {
			var c *array.Chunked
			var err error
			if e.kind == KindAnd {
				c, err = kernel.And(lc, rc)
			} else {
				c, err = kernel.Or(lc, rc)
			}
			if err != nil {
				return out, nil
			}
			return asLiteral(out, c)
		}
		// True and a condition is that condition, and false or one is too. The
		// side that stays has to be a boolean rather than a column of nothing,
		// since three valued and of true with a missing value is missing and
		// not the value.
		//
		// The other pair, false and a condition or true or one, is the answer
		// on its own and is the rule this pass does not have, for the reason
		// [Fold] gives.
		idle := e.kind == KindAnd
		if writtenAs(l, idle) && isBool(r, s) {
			return r, nil
		}
		if writtenAs(r, idle) && isBool(l, s) {
			return l, nil
		}
		return out, nil

	case KindNot:
		l, lc := folded(e.l, s)
		if lc != nil {
			c, err := kernel.Not(lc)
			if err == nil {
				return asLiteral(rebuiltNot(e, l), c)
			}
		}
		// Not of not is the condition itself, for a condition that is a boolean.
		// One over a column of nothing stays where it is, since not of nothing is
		// nothing and the column is not a boolean.
		if l.kind == KindNot && isBool(l.l, s) {
			return l.l, nil
		}
		return rebuiltNot(e, l), nil

	case KindIsNull, KindIsNotNull:
		l, lc := folded(e.l, s)
		out := e
		if l != e.l {
			if e.kind == KindIsNull {
				out = IsNull(l)
			} else {
				out = IsNotNull(l)
			}
		}
		if lc == nil {
			return out, nil
		}
		if e.kind == KindIsNull {
			return asLiteral(out, kernel.IsNull(lc))
		}
		return asLiteral(out, kernel.IsNotNull(lc))

	default:
		l, lc := folded(e.l, s)
		out := e
		if l != e.l {
			out = Cast(e.dt, l)
		}
		if lc != nil {
			if c, err := kernel.Cast(lc, e.dt); err == nil {
				return asLiteral(out, c)
			}
			return out, nil
		}
		// A cast to the type the column already has is a kernel call and a name
		// in the plan and nothing else.
		if dt, err := TypeOf(l, s); err == nil && dtype.Equal(dt, e.dt) {
			return l, nil
		}
		return out, nil
	}
}

// rebuiltNot is the negation of what its operand came back as, and the negation
// itself when the operand did not move.
func rebuiltNot(e, l *Expr) *Expr {
	if l == e.l {
		return e
	}
	return Not(l)
}

// pair works out the two sides of a step as columns, doing the side that has a
// type of its own first so that a literal on the other side knows what to
// become. It is the evaluator's own order, because a fold that used a different
// one would work out a different answer.
func pair(l *Expr, lc *array.Chunked, r *Expr, rc *array.Chunked) (a, b *array.Chunked, err error) {
	if l.kind == KindLiteral && r.kind != KindLiteral {
		a, err = hinted(l, lc, rc.DType())
		if err != nil {
			return nil, nil, err
		}
		return a, rc, nil
	}

	b, err = hinted(r, rc, lc.DType())
	if err != nil {
		return nil, nil, err
	}
	return lc, b, nil
}

// hinted rebuilds one side against the type the other side turned out to be.
// The hint reaches a literal and nothing else, which is the rule the evaluator
// follows and the reason a literal has no type until it is used.
func hinted(e *Expr, c *array.Chunked, dt dtype.DataType) (*array.Chunked, error) {
	if e.kind != KindLiteral {
		return c, nil
	}
	return LiteralColumn(e.lit, dt)
}

// asLiteral returns the step written as the value it came to, when that value is
// one a literal can hold and one that comes back as the type it went in as. It
// returns the step it was given otherwise, along with the column either way, so
// that a step above can still use the answer even when this one cannot be
// written down.
func asLiteral(e *Expr, c *array.Chunked) (*Expr, *array.Chunked) {
	v, ok := valueOf(c)
	if !ok {
		return e, c
	}
	dt, err := LiteralTypeAgainst(v, nil)
	if err != nil || !dtype.Equal(dt, c.DType()) {
		return e, c
	}
	return Lit(v), c
}

// valueOf reads the one value of a column back out as the Go value a literal
// would be written with.
//
// The types it knows are the ones a literal can be written as. A timestamp is
// not one of them, since what comes back is a count and a literal of a count is
// an integer, and a missing value is not one either, since a literal of nothing
// has no type to keep.
func valueOf(c *array.Chunked) (any, bool) {
	if c.Len() != 1 || c.IsNull(0) {
		return nil, false
	}
	switch c.DType().Kind() {
	case dtype.BoolKind:
		return c.Bool(0), true
	case dtype.Int8Kind:
		return c.Value[int8](0), true
	case dtype.Int16Kind:
		return c.Value[int16](0), true
	case dtype.Int32Kind:
		return c.Value[int32](0), true
	case dtype.Int64Kind:
		return c.Value[int64](0), true
	case dtype.Uint8Kind:
		return c.Value[uint8](0), true
	case dtype.Uint16Kind:
		return c.Value[uint16](0), true
	case dtype.Uint32Kind:
		return c.Value[uint32](0), true
	case dtype.Uint64Kind:
		return c.Value[uint64](0), true
	case dtype.Float32Kind:
		return c.Value[float32](0), true
	case dtype.Float64Kind:
		return c.Value[float64](0), true
	case dtype.StringKind:
		return string(c.Bytes(0)), true
	case dtype.BinaryKind:
		return c.Bytes(0), true
	default:
		return nil, false
	}
}

// writtenAs reports whether the expression is the boolean b written in the query.
func writtenAs(e *Expr, b bool) bool {
	if e.kind != KindLiteral {
		return false
	}
	v, ok := e.lit.(bool)
	return ok && v == b
}

// isBool reports whether the expression comes out as a boolean over the schema,
// which is what a condition has to be for the side of an and or an or to stand
// on its own.
func isBool(e *Expr, s dtype.Schema) bool {
	dt, err := TypeOf(e, s)
	return err == nil && dtype.Equal(dt, dtype.Bool)
}
