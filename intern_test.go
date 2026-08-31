package kuma

// The tests for the parts of the engine the exported API cannot reach yet are
// in the package rather than beside it. A plan holding an operator the engine
// does not run is not something the lazy frame can build, and a source it
// cannot read is not something a caller can hand it, so the only way to check
// that both say so rather than answering wrongly is from in here. They go away
// as the operators arrive.

import (
	"context"
	"errors"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// internFrame is a frame of two columns, built the way a caller would.
func internFrame(t *testing.T) *Frame[Dynamic] {
	t.Helper()

	f, err := NewFrame(
		NewSeries("symbol", "AAPL", "MSFT", "AAPL").Column(),
		NewSeries("qty", int64(100), 50, 25).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	return f
}

func TestRunAnOperatorTheEngineDoesNotHaveYet(t *testing.T) {
	scan := plan.Scan(frameSource{frame: internFrame(t)})

	cases := []struct {
		name string
		node *plan.Node
	}{
		{name: "distinct", node: plan.Distinct(scan, nil)},
		{
			// An aggregate of the whole input is a plan the lazy frame has no
			// way to write, since GroupBy takes at least one column, and the
			// engine has nowhere to put the answer for an input with no rows.
			name: "aggregate with no keys",
			node: plan.Aggregate(scan, nil, []plan.Agg{{Func: plan.AggSum, Expr: plan.Col("qty")}}),
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := run(t.Context(), c.node)
			if !errors.Is(err, ErrNotSupported) {
				t.Fatalf("run = %v, want ErrNotSupported", err)
			}
		})
	}
}

// otherSource is a source that is not the engine's, which is what a caller
// building a plan against their own reader would have.
type otherSource struct{}

func (otherSource) Name() string { return "elsewhere" }

func (otherSource) Schema() (dtype.Schema, error) {
	return dtype.Schema{Fields: []dtype.Field{{Name: "qty", Type: dtype.Int64}}}, nil
}

func TestRunASourceTheEngineCannotRead(t *testing.T) {
	_, err := run(t.Context(), plan.Scan(otherSource{}))
	if !errors.Is(err, ErrNotSupported) {
		t.Fatalf("run = %v, want ErrNotSupported", err)
	}
}

// TestRunChecksThePlanBeforeReading is what makes a mistake anywhere in a query
// cost nothing to find out about. The source counts the reads, so a plan that
// was never going to work must leave it at zero.
func TestRunChecksThePlanBeforeReading(t *testing.T) {
	src := &countingSource{frame: internFrame(t)}
	n := plan.Filter(plan.Scan(src), plan.Compare(kernel.OpGt, plan.Col("prcie"), plan.Lit(int64(10))))

	if _, err := run(t.Context(), n); !errors.Is(err, ErrNoColumn) {
		t.Fatalf("run = %v, want ErrNoColumn", err)
	}
	if src.reads != 0 {
		t.Errorf("the source was read %d times, want none of it read", src.reads)
	}
}

// countingSource is a frame source that says how many times it was read.
type countingSource struct {
	frame *Frame[Dynamic]
	reads int
}

var _ source = (*countingSource)(nil)

func (s *countingSource) Name() string { return "counting" }

func (s *countingSource) Schema() (dtype.Schema, error) { return s.frame.schema, nil }

func (s *countingSource) read(context.Context) (*Frame[Dynamic], error) {
	s.reads++
	return s.frame, nil
}

// TestRunStopsBetweenOperators is the context check, from the inside, where the
// number of operators that ran can be seen.
func TestRunStopsBetweenOperators(t *testing.T) {
	src := &countingSource{frame: internFrame(t)}
	n := plan.Limit(plan.Scan(src), 0, 1)

	ctx, cancel := context.WithCancel(t.Context())
	cancel()

	if _, err := run(ctx, n); !errors.Is(err, context.Canceled) {
		t.Fatalf("run = %v, want context.Canceled", err)
	}
	if src.reads != 0 {
		t.Errorf("the source was read %d times, want the query given up on first", src.reads)
	}
}
