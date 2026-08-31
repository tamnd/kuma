package plan

import (
	"maps"
	"slices"

	"github.com/tamnd/kuma/kernel"
)

// PushProjection rewrites a plan so that every scan in it reads only the
// columns something above it asks for.
//
// It is the first pass worth having and usually the largest single win in the
// whole optimizer. A table of forty columns that a query touches two of is a
// query that decompresses two columns instead of forty, and the work that is
// saved is work no later pass can save, because a column that was never read is
// a column nothing has to carry, filter, sort or join.
//
// Nothing about what the plan produces changes. The pass walks down from the
// root carrying the set of columns whatever is above needs, and the only
// operator it rewrites is the scan at the bottom. A projection and an
// aggregation are where the set gets smaller, since they say outright which
// columns they read and what they produce does not depend on the rest. A
// filter, a sort, a limit, an explode and a distinct by named columns pass the
// set along with their own columns added, since they hand the columns of their
// input straight through. A distinct over whole rows compares every column, so
// it asks for every column.
//
// The one thing it will not do is leave a scan with no columns at all. A frame
// of no columns has no number of rows either, so a query that reads nothing but
// a literal keeps the first column of the source to count the rows with.
func PushProjection(n *Node) (*Node, error) {
	return pushProjection(n, nil)
}

// pushProjection returns n producing what it produces now, reading no more than
// the columns in need on the way down to its scans.
//
// A nil need means every column this operator produces is wanted, which is what
// the root of a plan is given and what an operator hands to an input whose
// columns it cannot account for one by one.
func pushProjection(n *Node, need map[string]bool) (*Node, error) {
	switch n.op {
	case OpScan:
		return narrowScan(n, need)

	case OpProject:
		// What a projection produces is worked out from what its expressions
		// read, and nothing else of its input reaches the rows above it, so
		// what is needed below is those expressions and is not the set that
		// arrived.
		below := make(map[string]bool)
		for _, p := range n.cols {
			p.Expr.eachColumn(func(name string) { below[name] = true })
		}
		return rewriteInput(n, below)

	case OpAggregate:
		// The same, with the group keys and the aggregated expressions
		// standing in for the projected ones. A size counts rows and reads
		// nothing, which is why the expression of an aggregation can be nil.
		below := make(map[string]bool)
		for _, e := range n.by {
			e.eachColumn(func(name string) { below[name] = true })
		}
		for _, a := range n.aggs {
			a.Expr.eachColumn(func(name string) { below[name] = true })
		}
		return rewriteInput(n, below)

	case OpFilter:
		return rewriteInput(n, with(need, n.pred))

	case OpSort:
		below := need
		for _, k := range n.sort {
			below = with(below, k.Expr)
		}
		return rewriteInput(n, below)

	case OpLimit:
		return rewriteInput(n, need)

	case OpExplode:
		// An explode reads the columns it takes apart and hands the rest
		// through, so a name it was given is needed below whether or not
		// anything above reads the column afterwards.
		below := need
		if below != nil {
			below = maps.Clone(below)
			for _, name := range n.names {
				below[name] = true
			}
		}
		return rewriteInput(n, below)

	case OpDistinct:
		if len(n.by) == 0 {
			// Distinct rows are told apart by every column they hold, so
			// there is no column here that nothing reads.
			return rewriteInput(n, nil)
		}
		below := need
		for _, e := range n.by {
			below = with(below, e)
		}
		return rewriteInput(n, below)

	default:
		return pushJoin(n, need)
	}
}

// pushJoin splits the needed columns between the two sides of a join and pushes
// each half into the side it came from.
//
// A join produces the columns of the left side followed by the columns of the
// right side that are not shared key names, and no column is renamed, so a name
// that is needed above belongs to whichever side has a column of that name. A
// key column is needed on both sides whatever else is, since a join with a key
// it cannot read is not a join.
func pushJoin(n *Node, need map[string]bool) (*Node, error) {
	if need == nil {
		l, err := pushProjection(n.l, nil)
		if err != nil {
			return nil, err
		}
		r, err := pushProjection(n.r, nil)
		if err != nil {
			return nil, err
		}
		return n.withInputs(l, r), nil
	}

	lkeys := make([]*Expr, len(n.on))
	rkeys := make([]*Expr, len(n.on))
	for i, k := range n.on {
		lkeys[i], rkeys[i] = k.Left, k.Right
	}

	left, err := sideOfJoin(n.l, need, lkeys)
	if err != nil {
		return nil, err
	}

	// A semi or an anti join answers a question about the left side and keeps
	// no column of the right one, so the right side is read for its keys alone
	// however many of its columns share a name with one on the left.
	above := need
	if n.how == kernel.SemiJoin || n.how == kernel.AntiJoin {
		above = nil
	}
	right, err := sideOfJoin(n.r, above, rkeys)
	if err != nil {
		return nil, err
	}
	return n.withInputs(left, right), nil
}

// sideOfJoin pushes into one side of a join the columns of that side that are
// needed above, together with the columns its half of every key reads. A nil
// need is a side nothing above reads, which is a side read for its keys alone.
func sideOfJoin(in *Node, need map[string]bool, keys []*Expr) (*Node, error) {
	s, err := in.Schema()
	if err != nil {
		return nil, err
	}

	below := make(map[string]bool)
	for _, f := range s.Fields {
		if need[f.Name] {
			below[f.Name] = true
		}
	}
	for _, e := range keys {
		e.eachColumn(func(name string) { below[name] = true })
	}
	return pushProjection(in, below)
}

// rewriteInput pushes into the one input of an operator and returns the
// operator reading what came back, which is the operator itself when nothing
// under it moved.
func rewriteInput(n *Node, need map[string]bool) (*Node, error) {
	in, err := pushProjection(n.l, need)
	if err != nil {
		return nil, err
	}
	return n.withInputs(in, n.r), nil
}

// narrowScan returns the scan reading the columns in need and no others, and
// the scan it was given when that is what it already reads.
func narrowScan(n *Node, need map[string]bool) (*Node, error) {
	if need == nil {
		return n, nil
	}

	s, err := n.Schema()
	if err != nil {
		return nil, err
	}

	read := make([]string, 0, len(s.Fields))
	for _, f := range s.Fields {
		if need[f.Name] {
			read = append(read, f.Name)
		}
	}

	// How many rows a frame has is a fact about its columns, so a frame of none
	// has no rows to speak of and a query that counted them would come back
	// with nothing. Keeping one column is what counting the rows of a table
	// costs, and the cheapest one to keep is the first, which is the one a
	// reader has already found the start of.
	if len(read) == 0 && len(s.Fields) > 0 {
		read = append(read, s.Fields[0].Name)
	}

	// The schema of the scan is what it reads now, so this is the pass saying
	// it found nothing to do, and it has to say so by handing back the node it
	// was given. A scan that keeps one column to count the rows with would
	// otherwise be narrowed to that column over and over, each time to a new
	// node holding the same thing, and the optimizer would run until it gave
	// up.
	if slices.Equal(read, s.Names()) {
		return n, nil
	}
	return ScanOnly(n.src, read), nil
}

// with returns the needed set with the columns of an expression added, and nil
// when it was already nil, since a set that means every column cannot be added
// to.
func with(need map[string]bool, e *Expr) map[string]bool {
	if need == nil {
		return nil
	}
	out := maps.Clone(need)
	e.eachColumn(func(name string) { out[name] = true })
	return out
}
