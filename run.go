package kuma

import (
	"context"
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// source is somewhere a plan reads rows from. It is a [plan.Source] with the
// reading added, since a plan on its own only has to say what a source holds
// and the engine is the part that wants what is in it.
//
// It is not exported. What a source has to do is decided by what the engine
// needs of it, and that is still moving, so a reader outside this package would
// be a promise nobody is ready to keep. The two there are today are a frame
// that is already in memory and, through the readers, a file.
type source interface {
	plan.Source

	read(ctx context.Context) (*Frame[Dynamic], error)
}

// frameSource is a frame that a plan reads from, which is what [Frame.Lazy]
// puts at the leaf of the plan it builds.
type frameSource struct {
	frame *Frame[Dynamic]
}

var _ source = frameSource{}

// Name says what the source is in an explain, which for a frame that is
// already in memory is nothing more than that it is one.
func (s frameSource) Name() string { return "frame" }

// Schema returns what the frame holds, which it knows without reading it.
func (s frameSource) Schema() (dtype.Schema, error) { return s.frame.schema, nil }

// read returns the frame. Nothing is copied and nothing can go wrong, which is
// what makes a frame the cheapest leaf a plan can have.
func (s frameSource) read(context.Context) (*Frame[Dynamic], error) { return s.frame, nil }

// run works a plan out and returns what it produced.
//
// The whole plan is checked before any of it runs, so a column that is not
// there costs nothing to find out about however far down the tree it was
// written. What runs after that cannot meet a wrong name or a type with no
// operation for it, and the errors it does report are the ones the data
// decides, such as a file that is not there or a value that will not cast.
//
// Each operator works its input out in full before it starts. That is the
// simplest thing that is correct, and it is what the eager frame already does,
// so a query through the lazy API gives the same answer as the same query
// written out by hand. It is also why the plan is worth optimizing: a pushdown
// that reads two columns instead of forty saves the whole of that column, not a
// fraction of it. The morsel scheduler that runs the operators against each
// other rather than one after another is a later change, and it does not change
// what any of this returns.
//
// The plan is optimized between the check and the run. It is checked first so
// that a mistake in the query is reported as it was written rather than as some
// pass left it, and so that the passes themselves can rely on every name in the
// plan being a column and every expression having a type.
func run(ctx context.Context, n *plan.Node) (*Frame[Dynamic], error) {
	f, _, err := runMeasured(ctx, n, nil)
	return f, err
}

// runMeasured is run with somewhere to write down what each operator cost, and
// the measure of the whole query when there was somewhere.
//
// A nil recorder is a query that is only being run, which is the usual one, and
// it costs that query a nil check per operator. Building the recorder is what
// [LazyFrame.Profile] does and running the query is otherwise the same, so
// there is one path through the engine rather than two that have to agree.
func runMeasured(ctx context.Context, n *plan.Node, rec *recorder) (*Frame[Dynamic], plan.Measure, error) {
	if _, err := n.Schema(); err != nil {
		return nil, plan.Measure{}, err
	}

	n, err := plan.Optimize(n, plan.Passes()...)
	if err != nil {
		return nil, plan.Measure{}, err
	}

	f, err := runNode(withRecorder(ctx, rec), n)
	if err != nil || rec == nil {
		return f, plan.Measure{}, err
	}
	return f, rec.whole(), nil
}

// runNode is run without the check, which the operators below call for their
// inputs so that a plan is checked once rather than once per operator.
func runNode(ctx context.Context, n *plan.Node) (*Frame[Dynamic], error) {
	// A query over a large file is a long call, and the caller who gave up on
	// it should not wait for the rest of it. Between operators is where that
	// can be noticed without a check per row.
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if rec := recorderFrom(ctx); rec != nil {
		return rec.around(n, func() (*Frame[Dynamic], error) { return runOp(ctx, n) })
	}
	return runOp(ctx, n)
}

// runOp is the operator itself, which is runNode without the check and without
// the clock, so that a query being profiled and a query being run go through
// exactly the same code.
func runOp(ctx context.Context, n *plan.Node) (*Frame[Dynamic], error) {
	switch n.Op() {
	case plan.OpScan:
		return runScan(ctx, n)
	case plan.OpFilter:
		return runFilter(ctx, n)
	case plan.OpProject:
		return runProject(ctx, n)
	case plan.OpAggregate:
		return runAggregate(ctx, n)
	case plan.OpJoin:
		return runJoin(ctx, n)
	case plan.OpSort:
		return runSort(ctx, n)
	case plan.OpLimit:
		return runLimit(ctx, n)
	case plan.OpDistinct:
		return runDistinct(ctx, n)
	case plan.OpExplode:
		return runExplode(ctx, n)
	default:
		// Every operator the plan has is above. The next one the plan grows
		// arrives here until the engine is taught it, and saying so is better
		// than a wrong answer.
		return nil, fmt.Errorf("kuma: %s is not something the engine runs yet: %w", n.Op(), ErrNotSupported)
	}
}

func runScan(ctx context.Context, n *plan.Node) (*Frame[Dynamic], error) {
	src, ok := n.Source().(source)
	if !ok {
		return nil, fmt.Errorf("kuma: %s is a source the engine cannot read: %w", n.Source().Name(), ErrNotSupported)
	}

	f, err := src.read(ctx)
	if err != nil {
		return nil, err
	}

	if n.ScanColumns() != nil {
		// A source that can leave the columns it was not asked for unread has
		// done that already and hands back what it read. One that cannot, such
		// as a frame that is in memory in full, hands back everything and the
		// columns are dropped here, which is worth doing even so: every
		// operator above this one carries the columns it is given, and the ones
		// nothing asked for are carried for nothing.
		//
		// The names came from the source's own schema when the plan was
		// checked, so the only way this can fail is a source that answers two
		// different ways.
		f, err = f.Select(n.ScanColumns()...)
		if err != nil {
			return nil, fmt.Errorf("kuma: %s read columns other than the ones it said it had: %w", n.Source().Name(), err)
		}
	}

	// The same again for the rows. A source that can stop reading has stopped,
	// and one that cannot is cut down to the run the query asked for, so that
	// what everything above works on is that run rather than the lot.
	if off, count, ok := n.ScanRows(); ok {
		f = f.Slice(rowsIn(off, count, int64(f.rows)))
	}
	return f, nil
}

// rowsIn turns the run of rows a scan asks for into the two positions
// [Frame.Slice] wants, held inside a frame that may have fewer rows than the
// query asked for. A head of twenty over a frame of three is three rows and not
// an error, the same as it is for a limit.
//
// The counts are int64 because a plan says how many rows it wants without
// knowing what is going to answer, and a frame in memory is held in an int. The
// clamp to the frame's own row count is what makes the narrowing safe on a
// machine where an int is 32 bits.
func rowsIn(off, count, rows int64) (start, end int) {
	if off > rows {
		off = rows
	}
	if count > rows-off {
		count = rows - off
	}
	return int(off), int(off + count)
}

func runFilter(ctx context.Context, n *plan.Node) (*Frame[Dynamic], error) {
	f, err := runNode(ctx, n.Input())
	if err != nil {
		return nil, err
	}

	cond, err := f.evalNode(n.Cond(), "Filter")
	if err != nil {
		return nil, err
	}

	// A row the condition is missing an answer for is not kept, which is the
	// rule the eager filter follows and the rule the plan documents.
	return f.Take(kernel.Indices(cond)), nil
}

func runProject(ctx context.Context, n *plan.Node) (*Frame[Dynamic], error) {
	f, err := runNode(ctx, n.Input())
	if err != nil {
		return nil, err
	}

	cols := make([]Column, len(n.Columns()))
	for i, p := range n.Columns() {
		data, err := f.evalNode(p.Expr, "Select")
		if err != nil {
			return nil, err
		}
		cols[i] = Column{name: p.Name(), data: data}
	}
	return NewFrame(cols...)
}

// runAggregate divides the rows up by the keys and works out one column for
// every aggregation.
//
// The keys come first and the aggregations after, in the order they were
// written, which is what the plan says the result holds and what the eager
// [GroupedFrame.Agg] produces. The grouping is worked out once and handed to
// every aggregation, so a query asking for a sum, a mean and a count over the
// same keys divides the rows up once.
func runAggregate(ctx context.Context, n *plan.Node) (*Frame[Dynamic], error) {
	f, err := runNode(ctx, n.Input())
	if err != nil {
		return nil, err
	}

	// An aggregate with no keys is one row about the whole input, which is what
	// SQL gives for a select of a sum with no group by. It is not the same
	// operation: the answer for an empty input is one row of nothing rather
	// than no rows, so it needs a grouping the kernels cannot make yet.
	by := n.By()
	if len(by) == 0 {
		return nil, fmt.Errorf("kuma: an aggregate of the whole input, with nothing to group by, is not something the engine runs yet: %w",
			ErrNotSupported)
	}

	keys, err := evalKeys(f, by, "GroupBy")
	if err != nil {
		return nil, err
	}

	g, err := kernel.GroupBy(keys...)
	if err != nil {
		return nil, err
	}

	cols := make([]Column, 0, len(by)+len(n.Aggs()))
	for i, k := range g.Keys() {
		// A key is called what it was written as, so grouping by a column is a
		// column of that name and grouping by an expression is a column of the
		// expression's own text. That is the name the plan promised.
		cols = append(cols, Column{name: by[i].String(), data: k})
	}
	for _, a := range n.Aggs() {
		c, err := runAgg(f, g, a)
		if err != nil {
			return nil, err
		}
		cols = append(cols, c)
	}
	return NewFrame(cols...)
}

// evalKeys works the keys of a group by or a distinct out as columns of their
// own, so that grouping by something that is not a column, such as the month of
// a date, costs one column rather than a projection the caller has to write.
// The op is the step to name in an error.
func evalKeys(f *Frame[Dynamic], by []*plan.Expr, op string) ([]*array.Chunked, error) {
	keys := make([]*array.Chunked, len(by))
	for i, e := range by {
		data, err := f.evalNode(e, op)
		if err != nil {
			return nil, err
		}
		keys[i] = data
	}
	return keys, nil
}

// runAgg works one aggregation out, having first worked out the column it
// reads. A size reads no column, which is why it is not one line.
func runAgg(f *Frame[Dynamic], g *kernel.Groups, a plan.Agg) (Column, error) {
	if a.Func == plan.AggSize {
		return Column{name: a.Name(), data: kernel.Size(g)}, nil
	}

	data, err := f.evalNode(a.Expr, a.Func.String())
	if err != nil {
		return Column{}, err
	}

	out, err := aggregate(a, data, g)
	if err != nil {
		return Column{}, err
	}
	return Column{name: a.Name(), data: out}, nil
}

// runJoin puts the rows of the two inputs together where their keys match.
//
// Both sides are read in full before anything is paired, which is the same
// thing the eager join does. What the plan buys here is that a filter over
// either side has already run by the time the join sees it, so the join is over
// the rows that survived rather than over the file.
func runJoin(ctx context.Context, n *plan.Node) (*Frame[Dynamic], error) {
	l, err := runNode(ctx, n.Input())
	if err != nil {
		return nil, err
	}
	r, err := runNode(ctx, n.Right())
	if err != nil {
		return nil, err
	}

	left, right, err := joinSides(l, r, n.JoinKeys())
	if err != nil {
		return nil, err
	}

	p, err := kernel.Join(left, right, n.JoinType())
	if err != nil {
		return nil, err
	}

	shared := make(map[string]bool, len(n.JoinKeys()))
	for _, name := range n.SharedKeys() {
		shared[name] = true
	}
	return joinFrame(l, r, shared, n.JoinType(), p)
}

// joinSides works the keys of a join out over both inputs, so that joining on
// something that is worked out rather than read, such as a trimmed string,
// costs one column per side rather than a projection on each of them.
func joinSides(l, r *Frame[Dynamic], on []plan.JoinKey) (left, right kernel.Side, err error) {
	left = kernel.Side{Rows: l.rows}
	right = kernel.Side{Rows: r.rows}
	if len(on) == 0 {
		// A cross join reads no keys, and the plan has already turned away
		// every other join that arrived without any.
		return left, right, nil
	}

	left.Keys = make([]*array.Chunked, len(on))
	right.Keys = make([]*array.Chunked, len(on))
	for i, k := range on {
		if left.Keys[i], err = l.evalNode(k.Left, "Join"); err != nil {
			return left, right, err
		}
		if right.Keys[i], err = r.evalNode(k.Right, "Join"); err != nil {
			return left, right, err
		}
	}
	return left, right, nil
}

func runSort(ctx context.Context, n *plan.Node) (*Frame[Dynamic], error) {
	f, err := runNode(ctx, n.Input())
	if err != nil {
		return nil, err
	}

	keys, err := sortKeys(f, n.SortKeys())
	if err != nil {
		return nil, err
	}

	idx, err := kernel.SortIndex(keys...)
	if err != nil {
		return nil, err
	}
	return f.Take(idx), nil
}

// sortKeys works the keys of a sort out as columns of their own, so that
// sorting by something that is not a column, such as the number of days since a
// date, costs one column rather than a place in the result.
func sortKeys(f *Frame[Dynamic], sort []plan.SortKey) ([]kernel.Order, error) {
	keys := make([]kernel.Order, len(sort))
	for i, k := range sort {
		data, err := f.evalNode(k.Expr, "Sort")
		if err != nil {
			return nil, err
		}
		keys[i] = kernel.Order{Column: data, Descending: k.Descending, NullsFirst: k.NullsFirst}
	}
	return keys, nil
}

func runLimit(ctx context.Context, n *plan.Node) (*Frame[Dynamic], error) {
	f, err := runNode(ctx, n.Input())
	if err != nil {
		return nil, err
	}

	// Asking for more rows than there are gives what there is, since a head of
	// twenty over a frame of three is three rows and not an error.
	off := min(int(n.Offset()), f.rows)
	end := f.rows
	if n.Limit() >= 0 && off+int(n.Limit()) < end {
		end = off + int(n.Limit())
	}
	return f.Slice(off, end), nil
}

// runDistinct keeps the first row of each set of rows the keys agree on.
//
// It is the eager [Frame.Distinct] over keys that are worked out rather than
// read, so a distinct by the day of a timestamp is a step of a query rather than
// a column somebody has to add and then drop again.
func runDistinct(ctx context.Context, n *plan.Node) (*Frame[Dynamic], error) {
	f, err := runNode(ctx, n.Input())
	if err != nil {
		return nil, err
	}

	// With nothing to compare by, every column is compared, which is what the
	// eager Distinct does with no names and what a select distinct means.
	by := n.By()
	if len(by) == 0 {
		return distinct(f, columnData(f.cols))
	}

	keys, err := evalKeys(f, by, "Distinct")
	if err != nil {
		return nil, err
	}
	return distinct(f, keys)
}

// runExplode turns each element of the named list columns into a row of its
// own, repeating the other columns of the row it came from.
//
// It is the eager [Frame.Explode] over columns the plan has already checked, so
// the only thing that can go wrong here is the one thing the data decides: two
// columns being taken apart together that turn out to disagree about how many
// elements a row holds.
func runExplode(ctx context.Context, n *plan.Node) (*Frame[Dynamic], error) {
	f, err := runNode(ctx, n.Input())
	if err != nil {
		return nil, err
	}

	// The names are known to be columns of the input and known to be lists,
	// since the plan was checked before any of this ran, so the lookup here
	// cannot fail and the positions are all it is for.
	cols := make([]int, len(n.ExplodeNames()))
	for i, name := range n.ExplodeNames() {
		cols[i] = f.index[name]
	}
	return explode(f, cols)
}

// evalNode works an expression out over the frame, and checks that what came
// back is a column of the frame's own length.
//
// The length check is the same one the eager Eval does. An expression made of
// nothing but literals produces one value rather than one per row, and a column
// of one value in a frame of a million is not something to go on with.
func (f *Frame[S]) evalNode(e *plan.Expr, op string) (*array.Chunked, error) {
	data, err := eval(e, f.lookup(op), nil)
	if err != nil {
		return nil, err
	}
	if data.Len() != f.rows {
		return nil, fmt.Errorf("kuma: %s gives %d values for a frame of %d rows, "+
			"an expression has to read at least one column: %w", e, data.Len(), f.rows, ErrLength)
	}
	return data, nil
}
