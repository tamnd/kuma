package plan

import (
	"fmt"
)

// A Pass is one rewrite of a plan into a better plan that asks the same
// question.
//
// It takes a plan and returns a plan rather than changing the one it was given,
// which is what lets a caller keep the plan they wrote and print it beside the
// one that ran. Every pass in this package is written that way and a pass
// written elsewhere has to be too.
//
// The one rule beyond that is what makes running them to fixpoint terminate: a
// pass that changed nothing returns the node it was given, the same pointer.
// Returning an equal but newly built plan is what a pass does when it changed
// something, so a pass that rebuilds the tree whether or not it found anything
// to do is a pass that never stops being applied.
type Pass struct {
	// Name is what to call the pass in an explain and in an error, such as
	// "projection pushdown".
	Name string

	// Rewrite is the pass itself.
	Rewrite func(*Node) (*Node, error)
}

// The number of times the passes are run over a plan before it is called done.
//
// Reaching it means two passes are undoing each other, since a set of passes
// that each move a plan towards the same shape settles in two or three rounds
// and the third is usually the one that finds nothing. It is a bug when it
// happens rather than a plan that needed longer, so it comes back as an error
// naming the passes rather than as the plan as far as it got.
const maxRounds = 8

// Optimize runs the passes over the plan until none of them changes it, and
// returns the plan that came out.
//
// The passes are run in the order given, over and over, because one of them
// finding something usually gives the next one something to find: a projection
// that is pushed into a scan is what lets the columns a filter reads be worked
// out, and a predicate that sinks past a join is what makes the projection
// under it smaller. Running each of them once in a good order gets most of it
// and running them to fixpoint gets the rest, and the rest is the part nobody
// can predict from the query as written.
//
// It returns an error if a pass does, which is the same error checking the plan
// would give, since a pass that has to know what an operator produces asks
// [Node.Schema] for it. A plan that does not check is not optimized either, so
// the mistake a caller is told about is the one they made.
func Optimize(n *Node, passes ...Pass) (*Node, error) {
	if n == nil {
		return nil, errNoPlan
	}

	for range maxRounds {
		done := true
		for _, p := range passes {
			out, err := p.Rewrite(n)
			if err != nil {
				return nil, fmt.Errorf("kuma: %s: %w", p.Name, err)
			}
			if out != n {
				n, done = out, false
			}
		}
		if done {
			return n, nil
		}
	}
	return nil, fmt.Errorf("kuma: the optimizer did not settle after %d rounds over the plan, "+
		"which means two passes are undoing each other", maxRounds)
}

// Passes returns the passes that run over every query, in the order they run
// in.
//
// It is what [github.com/tamnd/kuma.LazyFrame.Collect] optimizes with, and it
// is here so that a caller who wants to see what one pass does can run that one
// on its own and so that a test can name the pass it is about.
func Passes() []Pass {
	return []Pass{
		{Name: "projection pushdown", Rewrite: PushProjection},
	}
}

// withInputs returns this operator reading the given inputs instead of the ones
// it was built over, and the node itself when they are the ones it already has.
//
// It is how a pass rebuilds the spine of a plan: the parts of an operator that
// are not its inputs are copied across as they are, including the slices, which
// nothing writes to. Returning the node itself when nothing moved is what makes
// a pass able to say it found nothing without comparing two trees.
func (n *Node) withInputs(l, r *Node) *Node {
	if n.l == l && n.r == r {
		return n
	}
	m := *n
	m.l, m.r = l, r
	return &m
}
