package plan

// Fuse collapses a projection written over another projection into one.
//
// A query that works out a value and then works something out from it has two
// projections in it, and the engine makes a pass over the data for each of them
// and holds the column in between. A query that asks for the amount with tax on
// it, out of an amount it worked out a step earlier, reads
//
//	Project (notional * 1.2) as gross
//	  Project (price * qty) as notional
//	    Scan trades/*.parquet
//
// and the notional column is built in full, read once and thrown away. Fused it
// reads
//
//	Project ((price * qty) * 1.2) as gross
//	  Scan trades/*.parquet
//
// which is one pass over each chunk and nothing held in between. This is the
// plan half of what the architecture calls fusion. Turning the one expression
// that is left into a single loop over the chunk is the kernel half and is a
// separate thing, but it cannot happen at all until the expressions are in one
// operator, which is what this does.
//
// A value is only worth inlining if inlining it does not mean working it out
// twice. A column of the lower projection that the upper one reads twice stays
// where it is, because working it out once and reading it by name is the whole
// point of it being there. That rule is [HoistCommon]'s rule read backwards.
// That pass takes a value written more than once and puts it in a projection
// underneath, and this one takes a value written once and brings it back up, so
// between them they leave a value worked out exactly once and neither has
// anything to say about what the other did. Two passes that undid each other
// would be caught by [Optimize] failing to settle, and this is why they do not.
//
// A plain column and a value written in the query are inlined however many
// times they are read, since neither is work. That is the same pair [HoistCommon]
// refuses to hoist and for the same reason.
//
// The two projections collapse together or not at all, so one column of the
// lower one that has to stay keeps the rest of them where they are. Taking some
// up and leaving others is a bigger change than it sounds. A value moved up
// reads the columns the lower projection reads rather than the ones it produces,
// so the lower one would have to pass those through as well, under names that
// may already be taken by what it produces. That is worth doing one day and it
// is not worth doing as part of this, so for now a pair like that is left alone
// and the plan says so.
func Fuse(n *Node) (*Node, error) {
	if n.op == OpScan {
		return n, nil
	}

	l, err := Fuse(n.l)
	if err != nil {
		return nil, err
	}
	r := n.r
	if r != nil {
		if r, err = Fuse(r); err != nil {
			return nil, err
		}
	}

	m := n.withInputs(l, r)
	if m.op != OpProject || m.l.op != OpProject {
		return m, nil
	}
	return fused(m), nil
}

// fused returns the two projections as one, and the operator it was given when
// they are not ones to collapse.
func fused(n *Node) *Node {
	in := n.l

	reads := make(map[string]int)
	for _, p := range n.cols {
		p.Expr.eachColumn(func(name string) { reads[name]++ })
	}

	by := make(map[string]*Expr, len(in.cols))
	for _, p := range in.cols {
		name := p.Name()
		if _, ok := by[name]; ok {
			// Two columns of one name, which is a plan the check has something
			// to say about. Rewriting it here would only decide which of them
			// won and hide the question.
			return n
		}
		if !inlinable(p.Expr, reads[name]) {
			return n
		}
		by[name] = p.Expr
	}

	for name := range reads {
		if _, ok := by[name]; !ok {
			// A name that is not one of the columns underneath. The plan is
			// wrong and the check says so about the query as it was written,
			// which is a better error than anything this could leave behind.
			return n
		}
	}

	cols := make([]Projection, len(n.cols))
	for i, p := range n.cols {
		cols[i] = p
		e := substitute(p.Expr, by)
		if e == p.Expr {
			continue
		}
		if cols[i].As == "" {
			// A projection with no name of its own is called after the
			// expression it holds, so inlining into one would rename the column
			// it produces. Writing down the name it had keeps both.
			cols[i].As = p.Expr.String()
		}
		cols[i].Expr = e
	}
	return Project(in.l, cols)
}

// inlinable reports whether a column of the lower projection is one to write
// into the projection above, given the number of times that one reads it.
//
// A column and a value written in the query cost nothing to write again. Any
// other value read more than once would be worked out once for each place it
// was written into, so it stays where it is and stays read by name.
func inlinable(e *Expr, reads int) bool {
	switch e.kind {
	case KindColumn, KindLiteral:
		return true
	default:
		return reads <= 1
	}
}
