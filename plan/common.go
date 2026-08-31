package plan

import "github.com/tamnd/kuma/dtype"

// HoistCommon rewrites a plan so that a value written more than once in one
// operator is worked out once.
//
// A query that asks for the amount, the amount with tax on it and the amount
// rounded has written the amount three times, and without this pass the engine
// multiplies the two columns three times over every row. The pass puts the
// repeated value in a projection underneath, where it is worked out once, and
// leaves the operator above reading it by name.
//
// The column it is put in is named after the expression it holds, which is what
// a projection with no name of its own is called anyway, so the operator above
// still produces exactly the columns it produced before and under the same
// names. An expression whose name is already a column of the input is left
// alone rather than shadowing it.
//
// Two operators have expressions worth doing this to. A projection can write a
// value in several of its columns, and an aggregation can write one in several
// of its aggregations, which is what a sum and a maximum of the same product
// look like. A filter has one expression and a sort and a distinct have keys
// that are almost always plain columns, and hoisting out of any of those would
// mean a projection above as well as below to put the columns back, which costs
// more than it saves.
//
// Finding the repeats is a pointer comparison rather than a walk, because two
// expressions that say the same thing are the same expression. That is what the
// interning in intern.go is for and this is the pass that spends it.
//
// A repeat inside a repeat takes more than one round. The first round takes
// both of them out and the outer one still works the inner one out for itself,
// and the round after that takes the inner one out from under it. That is what
// running a pass to fixpoint is for and [Optimize] does it, so the plan a query
// runs is the settled one.
func HoistCommon(n *Node) (*Node, error) {
	if n.op == OpScan {
		return n, nil
	}

	l, err := HoistCommon(n.l)
	if err != nil {
		return nil, err
	}
	r := n.r
	if r != nil {
		if r, err = HoistCommon(r); err != nil {
			return nil, err
		}
	}

	m := n.withInputs(l, r)
	switch m.op {
	case OpProject:
		return hoistProject(m)
	case OpAggregate:
		return hoistAggregate(m)
	default:
		return m, nil
	}
}

// hoistProject puts the values a projection writes more than once into a
// projection underneath it.
func hoistProject(n *Node) (*Node, error) {
	s, err := n.l.Schema()
	if err != nil {
		return nil, err
	}

	exprs := make([]*Expr, len(n.cols))
	for i, p := range n.cols {
		exprs[i] = p.Expr
	}
	shared := repeated(exprs, s)
	if len(shared) == 0 {
		return n, nil
	}

	by := held(shared)
	cols := make([]Projection, len(n.cols))
	for i, p := range n.cols {
		cols[i] = Projection{Expr: rebuilt(p.Expr, by), As: p.As}
		exprs[i] = cols[i].Expr
	}
	return Project(Project(n.l, under(exprs, shared, s)), cols), nil
}

// hoistAggregate puts the values an aggregation writes more than once into a
// projection underneath it.
//
// The group keys go into the counting with the aggregations, since grouping by
// a value and adding it up is writing it twice like any other pair.
func hoistAggregate(n *Node) (*Node, error) {
	s, err := n.l.Schema()
	if err != nil {
		return nil, err
	}

	exprs := make([]*Expr, 0, len(n.by)+len(n.aggs))
	exprs = append(exprs, n.by...)
	for _, a := range n.aggs {
		// A size counts rows without reading a column, so it has no expression
		// to share with anything.
		if a.Expr != nil {
			exprs = append(exprs, a.Expr)
		}
	}
	shared := repeated(exprs, s)
	if len(shared) == 0 {
		return n, nil
	}

	by := held(shared)
	exprs = exprs[:0]

	keys := make([]*Expr, len(n.by))
	for i, e := range n.by {
		keys[i] = rebuilt(e, by)
		exprs = append(exprs, keys[i])
	}

	aggs := make([]Agg, len(n.aggs))
	for i, a := range n.aggs {
		aggs[i] = a
		if a.Expr != nil {
			aggs[i].Expr = rebuilt(a.Expr, by)
			exprs = append(exprs, aggs[i].Expr)
		}
	}
	return Aggregate(Project(n.l, under(exprs, shared, s)), keys, aggs), nil
}

// repeated returns the values the expressions write more than once between
// them, outermost first and in the order they are written.
//
// A value written inside another value that is itself repeated is not in the
// answer, since working out the outer one once works out the inner one once
// with it. A plain column and a literal are never in it either: one is already
// a column and the other costs nothing to write again.
func repeated(exprs []*Expr, s dtype.Schema) []*Expr {
	seen := make(map[*Expr]int)
	for _, e := range exprs {
		e.eachStep(func(x *Expr) { seen[x]++ })
	}

	var shared []*Expr
	taken := make(map[*Expr]bool)

	var pick func(*Expr)
	pick = func(e *Expr) {
		if e == nil {
			return
		}
		if seen[e] > 1 && worthHolding(e, s) {
			if !taken[e] {
				taken[e] = true
				shared = append(shared, e)
			}
			return
		}
		pick(e.l)
		pick(e.r)
	}
	for _, e := range exprs {
		pick(e)
	}
	return shared
}

// worthHolding reports whether a value is one to put in a column of its own.
//
// A column is one already. A literal is written into the query rather than
// worked out from anything. And a value whose name is already a column of the
// input would hide that column from everything above, which is a wrong answer
// rather than a slow one, so it is left where it is.
func worthHolding(e *Expr, s dtype.Schema) bool {
	switch e.kind {
	case KindColumn, KindLiteral:
		return false
	default:
		return s.Index(e.String()) < 0
	}
}

// held returns the rewrite that replaces each of the shared values by the
// column that now holds it.
func held(shared []*Expr) func(*Expr) (*Expr, bool) {
	by := make(map[*Expr]*Expr, len(shared))
	for _, e := range shared {
		by[e] = Col(e.String())
	}
	return func(e *Expr) (*Expr, bool) {
		r, ok := by[e]
		return r, ok
	}
}

// under is the projection that goes underneath: the columns of the input that
// the rewritten expressions still read, in the order the input has them, and
// then the shared values in the order they were written.
//
// The input columns nothing reads any more are left out, so a pass that puts a
// projection in does not undo the one that works out how little a query has to
// read.
func under(exprs, shared []*Expr, s dtype.Schema) []Projection {
	need := make(map[string]bool)
	for _, e := range exprs {
		e.eachColumn(func(name string) { need[name] = true })
	}

	cols := make([]Projection, 0, len(need)+len(shared))
	for _, f := range s.Fields {
		if need[f.Name] {
			cols = append(cols, Projection{Expr: Col(f.Name)})
		}
	}
	for _, e := range shared {
		cols = append(cols, Projection{Expr: e})
	}
	return cols
}
