package plan

import (
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// PushPredicate rewrites a plan so that every filter runs as early as the
// query allows.
//
// It is the pass that makes the rest of a query smaller. A row that is thrown
// away before a join is a row the join does not pair, before a sort is a row
// the sort does not order, and before an aggregation is a row nothing has to
// add up. At the bottom it leaves the filter sitting on the scan, which is
// where a reader that can skip a row group by its statistics will pick it up.
//
// A filter is taken apart into the conditions its ands are made of, and each
// one moves on its own, so a filter over two columns of a join can end up as
// one filter on each side. What is left over is put back together above the
// operator that stopped it, and two filters that meet become one, which is one
// pass over the rows rather than two.
//
// What stops a condition is the operator underneath it:
//
//   - A limit, since which rows the first ten are depends on what was thrown
//     away before it.
//   - A projection, unless every column the condition reads is a column of the
//     input under another name. Pushing under a worked out column would mean
//     working it out twice, once to decide the rows and once to produce them.
//   - An aggregation, unless every column the condition reads is a group key,
//     which is what a having clause over a key really is.
//   - A distinct by named columns, unless the condition reads only those
//     columns, since which of a group of rows is the one kept depends on which
//     rows are there.
//   - An explode, for a condition on a column being taken apart, since that
//     column holds a list until the explode has run.
//   - A join, for a condition that reads both sides, and for the side of an
//     outer join that may be filled with nulls, since a row that was filled
//     with nulls because nothing matched is a different row from one that was
//     never there.
func PushPredicate(n *Node) (*Node, error) {
	return placed(n, nil)
}

// A cond is one condition on its way down, and the filter it was written in.
//
// The filter it came from is what lets the pass hand back the plan it was
// given when there was nothing to do. A condition that has been rewritten on
// the way down has no filter to go back to, and the one that lands where it
// started does.
type cond struct {
	expr *Expr
	from *Node
}

// placed returns the plan that filtering n by all of the conditions would
// give, with each of them as far down as it goes.
//
// The conditions arrive from above and every one of them is put somewhere on
// the way down, so what comes back never has a condition missing from it and
// never has one twice.
func placed(n *Node, cs []cond) (*Node, error) {
	switch n.op {
	case OpScan:
		return filterOver(n, cs), nil

	case OpFilter:
		// The filter itself goes down with everything that arrived. It is
		// built again wherever its conditions come to rest, which is here when
		// none of them moved.
		return placed(n.l, append(written(n), cs...))

	case OpSort:
		// Ordering rows and choosing rows do not depend on each other, so a
		// condition goes straight through and the sort has less to order.
		return through(n, cs)

	case OpLimit:
		return stop(n, cs)

	case OpProject:
		return throughProject(n, cs)

	case OpAggregate:
		return throughAggregate(n, cs)

	case OpDistinct:
		return throughDistinct(n, cs)

	case OpExplode:
		return throughExplode(n, cs)

	default:
		return throughJoin(n, cs)
	}
}

// through sends every condition into the one input of an operator, for the
// operators that stop none of them.
func through(n *Node, cs []cond) (*Node, error) {
	in, err := placed(n.l, cs)
	if err != nil {
		return nil, err
	}
	return n.withInputs(in, n.r), nil
}

// stop keeps every condition above the operator, for the operators that stop
// all of them.
func stop(n *Node, cs []cond) (*Node, error) {
	in, err := placed(n.l, nil)
	if err != nil {
		return nil, err
	}
	return filterOver(n.withInputs(in, n.r), cs), nil
}

// split sends the conditions the operator lets by into its input and keeps the
// rest above it. Each condition is offered to take, which returns it as it has
// to be written underneath, and false for one that cannot go.
func split(n *Node, cs []cond, take func(cond) (cond, bool)) (*Node, error) {
	var down, up []cond
	for _, c := range cs {
		if under, ok := take(c); ok {
			down = append(down, under)
		} else {
			up = append(up, c)
		}
	}

	in, err := placed(n.l, down)
	if err != nil {
		return nil, err
	}
	return filterOver(n.withInputs(in, n.r), up), nil
}

// throughProject sends down the conditions that read nothing but columns the
// projection passes through, under whatever those columns are called
// underneath.
//
// A condition on a column that is worked out stays where it is. Pushing it
// under would mean working the value out once to decide the row and once to
// produce it, which is slower than the filter it saves unless the scan can use
// it to skip something, and no scan can yet.
func throughProject(n *Node, cs []cond) (*Node, error) {
	by := make(map[string]*Expr, len(n.cols))
	for _, p := range n.cols {
		if p.Expr.kind == KindColumn {
			by[p.Name()] = p.Expr
		}
	}

	return split(n, cs, func(c cond) (cond, bool) {
		if !readsOnly(c.expr, by) {
			return cond{}, false
		}
		return rewritten(c, substitute(c.expr, by)), true
	})
}

// throughAggregate sends down the conditions that read nothing but group keys,
// which is a having clause over a key and is the same question asked of the
// rows rather than of the groups.
//
// The keys have to be plain columns for this. A group by a worked out value
// names its column after the expression, and a condition on that column is a
// condition on a value the input does not hold.
func throughAggregate(n *Node, cs []cond) (*Node, error) {
	keys := make(map[string]*Expr, len(n.by))
	for _, e := range n.by {
		if e != nil && e.kind == KindColumn {
			keys[e.name] = e
		}
	}

	return split(n, cs, func(c cond) (cond, bool) {
		return c, readsOnly(c.expr, keys)
	})
}

// throughDistinct sends down the conditions that read nothing but the columns
// the distinct is taken by, and everything when it is taken by whole rows.
//
// A condition on any other column has to stay above. Which row of a group of
// rows with the same key is the one that is kept depends on which rows are
// there, so throwing some away first can keep a different row.
func throughDistinct(n *Node, cs []cond) (*Node, error) {
	if len(n.by) == 0 {
		return through(n, cs)
	}

	keys := make(map[string]*Expr, len(n.by))
	for _, e := range n.by {
		if e != nil && e.kind == KindColumn {
			keys[e.name] = e
		}
	}

	return split(n, cs, func(c cond) (cond, bool) {
		return c, readsOnly(c.expr, keys)
	})
}

// throughExplode sends down the conditions that read none of the columns being
// taken apart, since those columns hold a list until the explode has run and
// the condition was written about an element of one.
func throughExplode(n *Node, cs []cond) (*Node, error) {
	apart := make(map[string]bool, len(n.names))
	for _, name := range n.names {
		apart[name] = true
	}

	return split(n, cs, func(c cond) (cond, bool) {
		ok := true
		c.expr.eachColumn(func(name string) { ok = ok && !apart[name] })
		return c, ok
	})
}

// throughJoin sends each condition into the side it reads, and keeps above the
// ones that read both sides or that the kind of join will not have moved.
//
// A condition on the side of an outer join that can be filled with nulls stays
// where it is. A row that came out null filled because nothing matched it is
// not a row that was in the input, so a condition that turns it away is not a
// condition the input could have answered.
func throughJoin(n *Node, cs []cond) (*Node, error) {
	left, err := n.l.Schema()
	if err != nil {
		return nil, err
	}
	right, err := n.r.Schema()
	if err != nil {
		return nil, err
	}

	// A semi or an anti join keeps no column of its right side, so a condition
	// over the join is a condition over the left side alone.
	takesLeft := n.how != kernel.RightJoin && n.how != kernel.OuterJoin
	takesRight := n.how == kernel.InnerJoin || n.how == kernel.CrossJoin || n.how == kernel.RightJoin

	var down, other, up []cond
	for _, c := range cs {
		switch {
		case takesLeft && covers(left, c.expr):
			down = append(down, c)
		case takesRight && covers(right, c.expr):
			other = append(other, c)
		default:
			up = append(up, c)
		}
	}

	l, err := placed(n.l, down)
	if err != nil {
		return nil, err
	}
	r, err := placed(n.r, other)
	if err != nil {
		return nil, err
	}
	return filterOver(n.withInputs(l, r), up), nil
}

// filterOver returns the operator with the conditions filtering what it
// produces, and the operator itself when there are none.
//
// The conditions are put back together in the order they were written, and
// when they are all the conditions of one filter that is still reading the
// same input, that filter is what comes back. That is the pass saying it found
// nothing, and it has to say so by handing back the node it was given.
func filterOver(in *Node, cs []cond) *Node {
	if len(cs) == 0 {
		return in
	}

	e := cs[0].expr
	for _, c := range cs[1:] {
		e = And(e, c.expr)
	}
	if o := cs[0].from; o != nil && o.l == in && o.pred == e {
		return o
	}
	return Filter(in, e)
}

// written is the conditions a filter is made of, each remembering the filter
// it was written in.
func written(n *Node) []cond {
	parts := conjuncts(n.pred)
	cs := make([]cond, len(parts))
	for i, e := range parts {
		cs[i] = cond{expr: e, from: n}
	}
	return cs
}

// rewritten is the condition as it has to be written further down. One that
// came out unchanged is still the condition that was written where it was, and
// one that was rewritten is not.
func rewritten(c cond, e *Expr) cond {
	if e == c.expr {
		return c
	}
	return cond{expr: e}
}

// readsOnly reports whether an expression reads nothing but the named columns.
func readsOnly(e *Expr, names map[string]*Expr) bool {
	ok := true
	e.eachColumn(func(name string) {
		if _, found := names[name]; !found {
			ok = false
		}
	})
	return ok
}

// covers reports whether a schema holds every column an expression reads,
// which is how a condition over a join is asked which side it belongs to.
func covers(s dtype.Schema, e *Expr) bool {
	ok := true
	e.eachColumn(func(name string) {
		if s.Index(name) < 0 {
			ok = false
		}
	})
	return ok
}
