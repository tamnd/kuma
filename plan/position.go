package plan

// Saying where in a plan a mistake is. A message that names the operator is
// most of the answer when a query is four lines long, and none of it when a
// query is forty and three of the operators are projections.

import (
	"errors"
	"strings"
)

// OperatorError says which operator of a plan a mistake is in.
//
// A message on its own is enough to fix a short query. It stops being enough as
// soon as a plan has more than one operator of a kind in it, because "column
// "sym" not found in Filter" does not say which filter, and a query built up
// over a few calls has no line numbers to fall back on. So the plan goes in the
// message with a mark against the operator that found the mistake:
//
//	kuma: column "sym" not found in Filter
//	  available: symbol, price, qty
//	  did you mean: symbol?
//
//	in the plan
//	    Project symbol, price
//	>     Filter (sym > 100)
//	        Scan trades/*.parquet
//
// The mark is in a gutter rather than in the line, so the operators stay lined
// up under each other and the plan reads the same as it does everywhere else it
// is printed.
//
// The operator written down is the one that found the mistake, not the ones it
// came back up through. A filter reading a name that is not there is a mistake
// in the filter, even though the projection above it is what the caller asked
// to check, and pointing at the projection would send them to the wrong line.
//
// A plan of one operator gets the message and no plan, since drawing a tree of
// one line to point at the only line in it is not telling anybody anything.
type OperatorError struct {
	// Err is what went wrong, which is the error that would have been returned
	// on its own before there was anywhere to put it.
	Err error

	// At is the operator that found the mistake.
	At *Node

	// Plan is the plan it was found in, meaning the operator the check was
	// asked about rather than the root of anything larger it may hang under.
	Plan *Node
}

// Error returns the message with the plan under it, in the form described on
// [OperatorError].
func (e *OperatorError) Error() string {
	msg := e.Err.Error()
	if e.Plan == nil || e.At == nil || e.Plan.l == nil {
		return msg
	}

	var sb strings.Builder
	sb.WriteString(msg)
	sb.WriteString("\n\nin the plan\n")
	writeMarked(&sb, e.Plan, e.At, 0)
	return sb.String()
}

// Unwrap returns the error underneath, so that asking whether a name was wrong
// with errors.Is works the same whether or not the plan was there to point at.
func (e *OperatorError) Unwrap() error { return e.Err }

// writeMarked writes the plan as text with a mark against one operator.
//
// It is [Node.Tree] with a gutter of two columns on the front of every line,
// holding the mark on the one line that gets it. The newline goes before the
// line rather than after it for the reason it does there, which is that the
// text is easier to put something after when it does not end in one.
func writeMarked(sb *strings.Builder, n, at *Node, depth int) {
	if n == nil {
		return
	}

	if depth > 0 {
		sb.WriteByte('\n')
	}
	if n == at {
		sb.WriteString("> ")
	} else {
		sb.WriteString("  ")
	}
	sb.WriteString(strings.Repeat("  ", depth))
	sb.WriteString(n.String())

	writeMarked(sb, n.l, at, depth+1)
	writeMarked(sb, n.r, at, depth+1)
}

// found returns the error with the operator it came from written into it, and
// the plan to print set to n.
//
// An error that already names an operator keeps the one it names, since that is
// the operator that found the mistake and n is only one it came back up
// through. What n does become is the plan to print, so that a check ends up
// showing the whole query it was asked about rather than the part of it below
// the mistake.
//
// Widening the plan is a write to the error rather than another error built
// around it. That keeps whatever a caller wrapped the first one in, and it
// means an operator reported from the bottom of a deep plan costs one error
// rather than one per level on the way up. Nothing else can be holding it: the
// walk builds it on the way back and hands it straight to the caller, and a
// second walk over the same plan builds its own.
func found(n *Node, err error) error {
	var oe *OperatorError
	if errors.As(err, &oe) {
		oe.Plan = n
		return err
	}
	return &OperatorError{Err: err, At: n, Plan: n}
}
