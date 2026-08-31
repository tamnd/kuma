package plan

// A step of a query that could not be built. It is here so that a mistake made
// early does not stop the rest of the query from being written down, since the
// rest of it is most of what says where in the caller's code the mistake is.

import (
	"errors"
	"fmt"
)

// Poison is a step that could not be built, standing in the plan where the step
// would have been.
//
// Most steps are written down without being understood, and are checked later.
// A few cannot be. Adding a column has to name the columns that are already
// there in order to say what it produces, so it works out the schema of its
// input while the query is being written rather than when it is checked, and
// when that fails there is no operator to build.
//
// The choice then is between throwing the rest of the query away and writing
// down that the step was asked for and could not be had. Throwing it away is
// what this replaces. A query is built up over several calls and has no line
// numbers, so the plan is the only thing that says which call the mistake is
// near, and cutting the plan off at the mistake removes exactly the part that
// would have said so. Someone who wrote five steps and gets a two line plan
// back cannot tell which of the five it is about.
//
// So the step becomes an operator holding the mistake, keeping its input, and
// the steps after it build on top of it as though nothing had happened. The
// mistake surfaces once, when the query is checked or collected, and what it
// prints is the whole query with a mark on the operator the mistake is really
// in, which is somewhere below this one:
//
//	kuma: column "nope" not found in Project
//	  available: symbol, price
//
//	in the plan
//	  Limit 3
//	    Sort by x
//	      With x
//	>       Project nope
//	          Scan frame
//
// The step is written as the caller wrote it rather than as the operator it
// would have become. There is no operator, and inventing a name for one that
// was never worked out would be describing something that is not there.
//
// A query can hold more than one of these, because a step after a poisoned one
// may fail to build for the same reason. That is not a problem to be tidied
// away: each of them is a step the caller wrote, and all of them report the
// first mistake, which is the only one worth reporting since the steps after it
// were written against something that did not happen.
//
// A step poisoned with nothing said about why is still a step that could not be
// built and is still refused, in the same way an operator with a piece missing
// prints and is refused rather than being taken for a whole one. Saying nothing
// about why is a mistake in whatever wrote the plan and not in the query, so
// what comes back names the step and stops there.
func Poison(input *Node, step string, err error) *Node {
	if err == nil {
		err = fmt.Errorf("kuma: the step %s could not be built and did not say why", step)
	}
	p := &poison{step: step, err: err}
	n := &Node{op: OpPoison, l: input, bad: p}

	// A mistake that already names an operator keeps the one it names. It came
	// from checking the input, so the operator it names is down there, and that
	// is the one to point at. This step is only where it was noticed.
	var oe *OperatorError
	if errors.As(err, &oe) {
		p.err, p.at = oe.Err, oe.At
	} else {
		p.at = n
	}
	return n
}

// A poison is what a step that could not be built is holding, which is the step
// as the caller wrote it and the mistake it could not be built around.
//
// It hangs off the node behind a pointer rather than sitting in it. Every
// operator of every plan pays for a field on the node whether or not it uses
// one, the passes copy a node for every rewrite they make, and a step that
// could not be built is the rarest operator there is, so three fields of it on
// the node would be paid for by every query that is right in order to be used
// by the ones that are not.
type poison struct {
	step string // the step as the caller wrote it
	err  error  // the mistake it could not be built around
	at   *Node  // the operator that mistake is in
}

// Step is the step a poisoned operator stands for, as the caller wrote it, and
// the empty string for every other operator.
func (n *Node) Step() string {
	if n == nil || n.op != OpPoison {
		return ""
	}
	return n.bad.step
}

// carried returns the mistake a poisoned step is holding.
//
// The error is built fresh every time rather than kept, because it says which
// plan it is about and one poisoned query can be the bottom of several. Two
// queries built on one poisoned step would otherwise share an error and write
// their plans into each other's.
func (n *Node) carried() error {
	return &OperatorError{Err: n.bad.err, At: n.bad.at, Plan: n}
}

// poisoned returns the first step of a plan that could not be built, and nil
// for a plan that was built all the way.
//
// It looks below an operator before looking at the operator itself, so what
// comes back is the earliest step that failed rather than the last one to
// notice.
func poisoned(n *Node) *Node {
	if n == nil {
		return nil
	}
	if bad := poisoned(n.l); bad != nil {
		return bad
	}
	if bad := poisoned(n.r); bad != nil {
		return bad
	}
	if n.op == OpPoison {
		return n
	}
	return nil
}
