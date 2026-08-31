package plan

import "math"

// PushSlice rewrites a plan so that every limit runs as early as the query
// allows, and so that two limits over one another become one.
//
// It is the pass that stops a query building what it is going to throw away. A
// head of twenty over a directory of Parquet files should read the first row
// group and stop, and until the limit is written into the scan there is nothing
// for a reader to stop on. Even with a source that reads everything it pays,
// because every operator between the scan and where the limit used to be works
// on twenty rows rather than on all of them.
//
// Almost nothing lets a limit through, which is the point worth remembering
// about this pass. A limit means the first n rows of what the operator under it
// produces, so moving it down is only right when the operator produces its rows
// one for one and in the order it was given them. A projection does. A filter,
// a sort, a distinct, an explode, an aggregation and a join all do not, so the
// limit stays above them.
//
// Two limits meeting is the other half. A slice of a slice is one slice, and
// working the two out into one is what makes a limit written above a query that
// was already limited, which is what a page of a page looks like, cost what one
// limit costs.
//
// It never reports an error. Where a limit may go is a question about the shape
// of the plan rather than about what any of it holds, so unlike the other two
// pushdowns this one never has to ask a source anything.
func PushSlice(n *Node) (*Node, error) {
	return sliced(n, nil), nil
}

// A lim is a limit on its way down, and the limit it was written as.
//
// The node it came from is what lets the pass hand back the plan it was given
// when there was nothing to do, the same as a condition remembers its filter. A
// limit that was worked out from two of them has no node to go back to.
type lim struct {
	span
	from *Node
}

// sliced returns the plan that limiting n by l would give, with the limit as
// far down as it goes and nil for no limit at all.
func sliced(n *Node, l *lim) *Node {
	switch n.op {
	case OpScan:
		return limitScan(n, l)

	case OpLimit:
		// The limit itself goes down with whatever arrived, and the two become
		// one. It is built again wherever it comes to rest, which is here when
		// it did not move and there was nothing above it to merge with.
		return sliced(n.l, merged(n, l))

	case OpProject:
		// A projection works out a value for each row it is given and hands
		// back that many rows in that order, so the first twenty rows of a
		// projection are the projection of the first twenty rows. Moving the
		// limit under it means the values are worked out for twenty rows rather
		// than worked out for all of them and twenty kept.
		return n.withInputs(sliced(n.l, l), n.r)

	default:
		return stopSlice(n, l)
	}
}

// stopSlice keeps the limit above the operator and carries on into its inputs,
// for the operators that do not produce their rows one for one and in order.
func stopSlice(n *Node, l *lim) *Node {
	right := n.r
	if right != nil {
		right = sliced(right, nil)
	}
	return limitOver(n.withInputs(sliced(n.l, nil), right), l)
}

// merged is what a limit node and a limit already on its way down come to
// between them, which is the second one taken out of the first.
func merged(n *Node, l *lim) *lim {
	if l == nil {
		return &lim{span: span{off: n.off, n: n.n}, from: n}
	}

	// The rows the node keeps are the ones the limit above it counts from, so
	// what the two of them keep together starts that much further in and is at
	// most what is left of the first once the second has skipped its share.
	return &lim{span: span{off: add(n.off, l.off), n: min(rest(n.n, l.off), l.n)}}
}

// limitOver returns the operator with the limit over it, and the operator
// itself when there is none.
//
// A limit that is still the one it was written as, sitting on the input it was
// written over, is handed back as that very node. That is the pass saying it
// found nothing, and it has to say so by handing back what it was given.
func limitOver(in *Node, l *lim) *Node {
	if l == nil {
		return in
	}
	if o := l.from; o != nil && o.l == in {
		return o
	}
	return Limit(in, l.off, l.n)
}

// limitScan is the limit written into the scan, which is where it stops being
// an operator and becomes part of the read.
//
// A scan that was already reading a run of rows is cut down the same way two
// limit nodes are, since a scan reading a run and a limit over it is two slices
// of the same thing.
func limitScan(n *Node, l *lim) *Node {
	if l == nil {
		return n
	}

	off, count := l.off, l.n
	if r := n.rows; r != nil {
		off, count = add(r.off, l.off), min(rest(r.n, l.off), l.n)
		if off == r.off && count == r.n {
			// The scan already reads exactly this run, so there is nothing to
			// write into it. Handing back a new node holding the same numbers
			// would make the pass look like it had found something every time
			// it ran, and the optimizer would run until it gave up.
			return n
		}
	}
	return n.withRows(off, count)
}

// rest is how many of n rows are left once off of them have been skipped, and
// none when the skip runs past the end.
func rest(n, off int64) int64 {
	if n < off {
		return 0
	}
	return n - off
}

// add is the sum of two row counts, held at the largest int64 rather than
// wrapping round to a negative one. Neither number can be negative, since a
// plan with a limit that skips backwards is turned away when it is checked, so
// the only way the sum can be wrong is by being too big, and a query that skips
// more rows than there can ever be reads nothing either way.
func add(a, b int64) int64 {
	if s := a + b; s >= a {
		return s
	}
	return math.MaxInt64
}
