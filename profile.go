package kuma

// Timing a query while it runs. The clock is read once per operator rather than
// once per row or once per batch, so a profiled query is the same query with a
// handful of nanoseconds on it, and the profile is a fact about the run rather
// than an estimate of one.

import (
	"context"
	"time"

	"github.com/tamnd/kuma/plan"
)

// A recorder collects what each operator of a query cost while the query runs.
//
// It builds a tree rather than a map from operator to cost, because the same
// node can stand in a plan twice, which is what a self join over one query is,
// and the two runs of it are two different costs. The tree is built the way the
// engine nests: whatever is recorded while an operator runs is what that
// operator read, so the shape falls out of the calls rather than having to be
// worked out afterwards.
//
// There is no lock on it. Every operator works its input out in full before it
// starts, so one query is one goroutine walking its plan, and the day that
// changes is the day an operator level clock stops meaning what it says anyway.
type recorder struct {
	// done is what has been recorded at the depth being filled in now, which is
	// the inputs of whichever operator is running. It is swapped out and back by
	// around, and at the end it holds the one measure of the whole query.
	done []plan.Measure
}

// around runs one operator and writes down what it cost.
//
// The clock is read either side of the operator, so the time it gets is the
// operator and everything under it, which is what a clock read around a call
// can honestly report. Taking the inputs back off is [plan.Profile]'s job and
// is done once, at the end, rather than here where it would be done again for
// every operator above this one.
func (r *recorder) around(n *plan.Node, op func() (*Frame[Dynamic], error)) (*Frame[Dynamic], error) {
	// Whatever this operator records is its inputs, so the ones already
	// collected are put aside and given back with this one on the end.
	outer := r.done
	r.done = nil

	start := time.Now()
	f, err := op()
	took := time.Since(start)

	rows := int64(0)
	if f != nil {
		rows = int64(f.NumRows())
	}
	r.done = append(outer, plan.Measure{Node: n, Took: took, Rows: rows, Input: r.done})
	return f, err
}

// whole returns the measure of the query, which is the one thing left after the
// last operator has finished and put itself on the end of nothing.
func (r *recorder) whole() plan.Measure {
	if len(r.done) != 1 {
		return plan.Measure{}
	}
	return r.done[0]
}

// recorderKey is the type the recorder is kept under in a context. It is its
// own type so that nothing else in any package can be reading or writing the
// same key.
type recorderKey struct{}

// withRecorder returns a context carrying the recorder, and the context it was
// given when there is nothing to record, so that a query that is only being run
// does not pay for a context it has no use for.
func withRecorder(ctx context.Context, r *recorder) context.Context {
	if r == nil {
		return ctx
	}
	return context.WithValue(ctx, recorderKey{}, r)
}

// recorderFrom returns the recorder a query is being timed into, or nil when it
// is not being timed.
func recorderFrom(ctx context.Context) *recorder {
	r, _ := ctx.Value(recorderKey{}).(*recorder)
	return r
}
