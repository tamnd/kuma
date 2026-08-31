package plan

import (
	"github.com/tamnd/kuma/dtype"
)

// rewriteExprs returns the plan with every expression in it replaced by what f
// makes of it, and returns the plan it was given when f changed none of them.
//
// It is the walk the passes that rewrite expressions share. Each of them has
// something different to say about one expression and nothing different to say
// about which expressions a plan holds, where the schema to check them against
// comes from, or what has to happen to the name of a column whose expression
// moved. Written once, a new pass over expressions is the rule it applies and
// nothing else.
//
// The rule is handed an expression along with the schema of the input of the
// operator holding it, and hands back the expression it was given when it has
// nothing to say. That is the same contract [Pass] has and for the same reason:
// it is what lets this return the node it was given rather than an equal copy
// of it.
func rewriteExprs(n *Node, f func(*Expr, dtype.Schema) *Expr) (*Node, error) {
	if n.op == OpScan {
		return n, nil
	}

	l, err := rewriteExprs(n.l, f)
	if err != nil {
		return nil, err
	}
	r := n.r
	if r != nil {
		if r, err = rewriteExprs(r, f); err != nil {
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
	return rewriteNode(m, s, f), nil
}

// rewriteNode rewrites the expressions of one operator against the schema of its
// input, and returns the operator itself when none of them moved.
func rewriteNode(n *Node, s dtype.Schema, f func(*Expr, dtype.Schema) *Expr) *Node {
	switch n.op {
	case OpProject:
		cols, moved := n.cols, false
		for i, p := range n.cols {
			e := f(p.Expr, s)
			if e == p.Expr {
				continue
			}
			if !moved {
				cols, moved = append([]Projection(nil), n.cols...), true
			}
			if cols[i].As == "" {
				// A projection with no name of its own is called after the
				// expression it holds, so rewriting one would rename the column
				// it produces. Writing down the name it had keeps the rewrite
				// and the name both.
				cols[i].As = p.Expr.String()
			}
			cols[i].Expr = e
		}
		if !moved {
			return n
		}
		return Project(n.l, cols)

	case OpFilter:
		pred := f(n.pred, s)
		if pred == n.pred {
			return n
		}
		return Filter(n.l, pred)

	case OpAggregate:
		// The keys are left as they were written. A group key is the one
		// expression with no name to be given, since the column it produces is
		// called after it, so there is nowhere to write down the name a rewrite
		// would take away.
		aggs, moved := n.aggs, false
		for i, a := range n.aggs {
			if a.Expr == nil {
				// A size counts rows without reading a column, so it holds no
				// expression to rewrite.
				continue
			}
			e := f(a.Expr, s)
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
			e := f(k.Expr, s)
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
		by, moved := rewriteAll(n.by, s, f)
		if !moved {
			return n
		}
		return Distinct(n.l, by)
	}
}

// rewriteAll rewrites a list of expressions, and says whether any of them moved.
// It returns the list it was given when none did, since a pass that changed
// nothing has to hand back what it was given.
func rewriteAll(exprs []*Expr, s dtype.Schema, f func(*Expr, dtype.Schema) *Expr) ([]*Expr, bool) {
	out, moved := exprs, false
	for i, e := range exprs {
		g := f(e, s)
		if g == e {
			continue
		}
		if !moved {
			out, moved = append([]*Expr(nil), exprs...), true
		}
		out[i] = g
	}
	return out, moved
}
