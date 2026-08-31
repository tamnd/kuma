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
	"slices"
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

// TestRunAnAggregateOfTheWholeInput is the one plan left that the engine turns
// away. The lazy frame has no way to write it, since GroupBy takes at least one
// column, and the engine has nowhere to put the answer for an input with no
// rows, so this is the only place the refusal can be checked.
func TestRunAnAggregateOfTheWholeInput(t *testing.T) {
	scan := plan.Scan(frameSource{frame: internFrame(t)})
	n := plan.Aggregate(scan, nil, []plan.Agg{{Func: plan.AggSum, Expr: plan.Col("qty")}})

	if _, err := run(t.Context(), n); !errors.Is(err, ErrNotSupported) {
		t.Fatalf("run = %v, want ErrNotSupported", err)
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

// TestRunANarrowedScan is the engine half of the projection pushdown. The pass
// writes down which columns a scan is for, and this is what happens when the
// query runs: a source that cannot leave a column unread has the ones nothing
// asked for dropped as soon as it hands them over, so nothing above carries
// them.
func TestRunANarrowedScan(t *testing.T) {
	n := plan.ScanOnly(frameSource{frame: internFrame(t)}, []string{"qty"})

	got, err := run(t.Context(), n)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if want := []string{"qty"}; !slices.Equal(got.Names(), want) {
		t.Errorf("the scan gave %v, want %v", got.Names(), want)
	}
	if got.NumRows() != 3 {
		t.Errorf("the scan gave %d rows, want the 3 the frame has", got.NumRows())
	}
}

// TestAQueryNarrowsItsOwnScan is the pass running where it runs for real, which
// is inside run and without the caller asking for it.
func TestAQueryNarrowsItsOwnScan(t *testing.T) {
	src := &countingSource{frame: internFrame(t)}
	n := plan.Project(plan.Scan(src), []plan.Projection{{Expr: plan.Col("qty")}})

	got, err := run(t.Context(), n)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if want := []string{"qty"}; !slices.Equal(got.Names(), want) {
		t.Errorf("the query gave %v, want %v", got.Names(), want)
	}
	if src.reads != 1 {
		t.Errorf("the source was read %d times, want once", src.reads)
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
