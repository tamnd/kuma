package kuma

import (
	"context"
	"fmt"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/plan"
)

// LazyFrame is a query that has been written down and not run.
//
//	out, err := f.Lazy().
//		Filter(t.Price.Gt(100)).
//		SortDesc("price").
//		Head(20).
//		Collect(ctx)
//
// Every step returns a new LazyFrame, so a query is built up the way it reads.
// Nothing is worked out until [LazyFrame.Collect], which is what lets the whole
// query be looked at before any of it runs: a column that is not there is an
// error before the first file is opened, and the optimizer passes get to see
// what the last step wanted before deciding what the first one has to read.
//
// The type parameter is the schema, the same one [Frame] carries, so a handle
// written for a table of trades cannot be used against a query over orders and
// the compiler is what says so. A step that changes the columns gives back a
// Frame[Dynamic] for the reason [Frame.WithColumn] does, and [Bind] is the way
// back to a typed one. [LazyFrame.SelectAs] is the way to stay typed through a
// projection, being a select whose column list is a struct.
//
// A mistake in a step is kept until Collect rather than returned there and
// then. Writing a query is one expression and Go has no way to break out of the
// middle of one, so the alternative is an error check between every step, which
// nobody writes and everybody skips. What is kept is the first mistake, since
// the steps after it were written against something that did not happen.
//
// The operators it can build today are a scan, a filter, a projection, a group
// by, a join, a distinct, a sort, a limit and an explode, which is every
// operator the plan has and every one the engine runs. A union, a pivot and a
// window are the ones the three of them grow next, together.
//
// The zero LazyFrame is not usable. Use [Frame.Lazy].
type LazyFrame[S any] struct {
	node *plan.Node

	// err is the first thing that went wrong while the query was being written,
	// which is reported by Collect. A frame that is carrying one builds no
	// further, so the mistake a caller is told about is theirs rather than the
	// one it led to.
	err error
}

// Lazy returns a query that starts from this frame.
//
// The frame is the leaf of the plan, so nothing is copied and nothing is read
// again. What it buys over working on the frame directly is that the whole
// query is known before any of it runs, which is what the optimizer passes and
// the plan time check need.
func (f *Frame[S]) Lazy() *LazyFrame[S] {
	return &LazyFrame[S]{node: plan.Scan(frameSource{frame: f.dynamic()})}
}

// Plan returns the plan the query has been built into.
//
// It is the tree an explain prints and the tree the optimizer passes rewrite,
// and it is here so that a program can look at what it built. The plan is
// immutable, so reading it cannot disturb the query it came from.
func (lf *LazyFrame[S]) Plan() *plan.Node { return lf.node }

// Schema returns the columns the query would produce, without running it.
//
// It is [plan.Node.Schema], which works the answer out from the source up. A
// field is nullable when a column might have missing values in it rather than
// when it does, since which rows arrive is what the data decides.
func (lf *LazyFrame[S]) Schema() (dtype.Schema, error) {
	if lf.err != nil {
		return dtype.Schema{}, lf.err
	}
	return lf.node.Schema()
}

// Validate returns the first thing wrong with the query, and nil for a query
// that will run.
//
// It is what Collect checks before it reads anything, on its own, for a program
// that wants to say a query is wrong at the point it was built.
func (lf *LazyFrame[S]) Validate() error {
	if lf.err != nil {
		return lf.err
	}
	return lf.node.Validate()
}

// Filter keeps the rows the condition holds for.
//
//	q := f.Lazy().Filter(t.Price.Gt(100).And(t.Side.Eq("BUY")))
//
// A row the condition is missing an answer for is not kept, which is the rule
// [Frame.Filter] follows and the reason filtering on a condition and then on
// its negation does not always give every row back.
func (lf *LazyFrame[S]) Filter(cond BoolValue[S]) *LazyFrame[S] {
	if lf.err != nil {
		return lf
	}
	return &LazyFrame[S]{node: plan.Filter(lf.node, cond.expr())}
}

// Select keeps the named columns, in the order named.
//
// It is a projection, so a name that is not there is an error at Collect, and
// naming the same column twice is a frame with two columns of one name, which
// is an error for the same reason.
//
// What comes back is a Dynamic query, since the columns are no longer the ones
// the schema type describes. [LazyFrame.SelectAs] is the same step told by a
// struct instead of by a list of names, and it keeps the query typed.
func (lf *LazyFrame[S]) Select(names ...string) *LazyFrame[Dynamic] {
	if lf.err != nil {
		return &LazyFrame[Dynamic]{err: lf.err}
	}

	cols := make([]plan.Projection, len(names))
	for i, name := range names {
		cols[i] = plan.Projection{Expr: plan.Col(name)}
	}
	return &LazyFrame[Dynamic]{node: plan.Project(lf.node, cols)}
}

// With adds the result of an expression as a column called name, or replaces
// the column of that name when there is one.
//
//	q := f.Lazy().With("notional", t.Price.MulExpr(t.Qty.AsF64()))
//
// It is [Frame.WithExpr] written as a step of a query. The result is a Dynamic
// frame because the columns are no longer the ones the schema type describes.
func (lf *LazyFrame[S]) With(name string, e Expr[S]) *LazyFrame[Dynamic] {
	if lf.err != nil {
		return &LazyFrame[Dynamic]{err: lf.err}
	}

	// A projection says which columns it produces, so the ones that are already
	// there have to be named. That means asking the plan what it has, which is
	// the schema check the source up to here, and it is where a mistake written
	// earlier in the query is usually found.
	s, err := lf.node.Schema()
	if err != nil {
		return &LazyFrame[Dynamic]{err: err}
	}

	cols := make([]plan.Projection, 0, len(s.Fields)+1)
	replaced := false
	for _, f := range s.Fields {
		if f.Name == name {
			cols = append(cols, plan.Projection{Expr: e.expr(), As: name})
			replaced = true
			continue
		}
		cols = append(cols, plan.Projection{Expr: plan.Col(f.Name)})
	}
	if !replaced {
		cols = append(cols, plan.Projection{Expr: e.expr(), As: name})
	}
	return &LazyFrame[Dynamic]{node: plan.Project(lf.node, cols)}
}

// Drop leaves out the named columns and keeps the rest in the order they were
// in. A name that is not there is an error at Collect, since dropping a column
// that does not exist is a mistake rather than nothing to do.
func (lf *LazyFrame[S]) Drop(names ...string) *LazyFrame[Dynamic] {
	if lf.err != nil {
		return &LazyFrame[Dynamic]{err: lf.err}
	}

	s, err := lf.node.Schema()
	if err != nil {
		return &LazyFrame[Dynamic]{err: err}
	}

	drop := make(map[string]bool, len(names))
	for _, name := range names {
		if _, ok := s.Field(name); !ok {
			return &LazyFrame[Dynamic]{err: noColumn("Drop", name, s.Names())}
		}
		drop[name] = true
	}

	cols := make([]plan.Projection, 0, len(s.Fields))
	for _, f := range s.Fields {
		if !drop[f.Name] {
			cols = append(cols, plan.Projection{Expr: plan.Col(f.Name)})
		}
	}
	return &LazyFrame[Dynamic]{node: plan.Project(lf.node, cols)}
}

// GroupBy divides the rows up by the values of the named columns.
//
//	out, err := f.Lazy().GroupBy("symbol").Agg(kuma.Sum("qty")).Collect(ctx)
//
// Nothing is divided up here. What comes back is a query waiting for the
// aggregations, because the plan holds the keys and the aggregations as one
// operator: knowing both at once is what lets a pass read only the columns the
// aggregations touch. That is also why there is no lazy [GroupedFrame], which
// exists in the eager API so that several questions cost one grouping. Here the
// same query asked twice is the same plan twice, and the answer is to write the
// aggregations together.
//
// Two rows are in the same group when every key agrees, and a missing value
// agrees with a missing value, which is the rule [Frame.GroupBy] follows. A
// name that is not there is an error at [LazyFrame.Collect].
func (lf *LazyFrame[S]) GroupBy(names ...string) *LazyGroupBy[S] {
	if lf.err != nil {
		return &LazyGroupBy[S]{err: lf.err}
	}
	if len(names) == 0 {
		return &LazyGroupBy[S]{err: fmt.Errorf("kuma: GroupBy with no columns to group by: %w", ErrLength)}
	}

	by := make([]*plan.Expr, len(names))
	for i, name := range names {
		by[i] = plan.Col(name)
	}
	return &LazyGroupBy[S]{node: lf.node, by: by}
}

// LazyGroupBy is a query whose rows are to be divided up, waiting for the
// aggregations that say what to work out about each group.
//
// It is what [LazyFrame.GroupBy] returns and the only things to do with it are
// [LazyGroupBy.Agg] and [LazyGroupBy.Count]. It holds no data and does no work:
// the grouping happens when the query it builds is collected.
//
// The zero value is not usable.
type LazyGroupBy[S any] struct {
	node *plan.Node
	by   []*plan.Expr

	// err is the mistake the query was already carrying, or the one made in the
	// group by itself, kept for the frame Agg gives back to report.
	err error
}

// Agg works out the given aggregations for every group.
//
//	q := f.Lazy().GroupBy("symbol").Agg(
//		kuma.Sum("qty").As("total"),
//		kuma.Mean("price").As("avg"),
//		kuma.Size(),
//	)
//
// The result has the key columns first, one row per group, followed by one
// column per aggregation in the order they were given, which is what
// [GroupedFrame.Agg] produces for the same query written eagerly.
//
// An aggregation is named after the column it reads unless [Aggregation.As]
// says otherwise, so two aggregations of one column need at least one of them
// named, and asking for both without naming either is an error about duplicate
// column names at [LazyFrame.Collect].
//
// [LazyGroupBy.AggAs] is the same step with a struct saying what the result
// holds, which is how a query stays typed across a group by.
func (lg *LazyGroupBy[S]) Agg(aggs ...Aggregation) *LazyFrame[Dynamic] {
	if lg.err != nil {
		return &LazyFrame[Dynamic]{err: lg.err}
	}
	if len(aggs) == 0 {
		return &LazyFrame[Dynamic]{err: fmt.Errorf("kuma: Agg with nothing to aggregate: %w", ErrLength)}
	}

	as := make([]plan.Agg, len(aggs))
	for i, a := range aggs {
		as[i] = a.plan()
	}
	return &LazyFrame[Dynamic]{node: plan.Aggregate(lg.node, lg.by, as)}
}

// Count works out how many rows each group has, which is the group by anybody
// writes first. It counts rows and not values, so it is [Size] rather than
// [Count].
func (lg *LazyGroupBy[S]) Count() *LazyFrame[Dynamic] { return lg.Agg(Size()) }

// Join puts the rows of two queries together where their keys match.
//
//	q := trades.Lazy().Join(sectors.Lazy(), kuma.Using("symbol"), kuma.InnerJoin)
//
// It is [Frame.Join] written as a step of a query and it keeps every rule that
// one has: a missing key matches nothing, including another missing key, the
// result holds one column for a key that is called the same thing on both
// sides, and the rows come out in the left query's order. [Using] is the way to
// write the keys when both sides call them the same thing.
//
// The other query is collected as part of this one, so a filter on the right
// side runs before the join sees it and the join is over the rows that survived
// rather than over the file they came from.
//
// The result is a Dynamic query, since the columns are no longer the ones either
// schema describes. [LazyFrame.JoinAs] is the same step with a struct saying what
// the result holds, which is how a query stays typed across a join. Either side
// may have any schema, since a join reads columns by name and neither side's
// type says anything about the other's.
func (lf *LazyFrame[S]) Join[R any](other *LazyFrame[R], on []On, how JoinType) *LazyFrame[Dynamic] {
	if lf.err != nil {
		return &LazyFrame[Dynamic]{err: lf.err}
	}
	if other == nil {
		return &LazyFrame[Dynamic]{err: fmt.Errorf("kuma: Join with no query to join to: %w", ErrNoValues)}
	}
	if other.err != nil {
		return &LazyFrame[Dynamic]{err: other.err}
	}

	keys := make([]plan.JoinKey, len(on))
	for i, o := range on {
		keys[i] = plan.JoinKey{Left: plan.Col(o.Left), Right: plan.Col(o.Right)}
	}
	return &LazyFrame[Dynamic]{node: plan.Join(lf.node, other.node, keys, how)}
}

// InnerJoin keeps only the pairs of rows that matched, on the columns both
// sides call by the named names. It is [LazyFrame.Join] for the common case.
func (lf *LazyFrame[S]) InnerJoin[R any](other *LazyFrame[R], names ...string) *LazyFrame[Dynamic] {
	return lf.Join(other, Using(names...), InnerJoin)
}

// LeftJoin keeps every row of this query, with the other query's columns
// missing where nothing matched.
func (lf *LazyFrame[S]) LeftJoin[R any](other *LazyFrame[R], names ...string) *LazyFrame[Dynamic] {
	return lf.Join(other, Using(names...), LeftJoin)
}

// CrossJoin pairs every row of this query with every row of the other one.
//
// The result has as many rows as the two multiplied together, which is why it
// is a step a caller has to name rather than something a forgotten key falls
// into.
func (lf *LazyFrame[S]) CrossJoin[R any](other *LazyFrame[R]) *LazyFrame[Dynamic] {
	return lf.Join(other, nil, CrossJoin)
}

// Distinct keeps the first row of each set of rows that agree on the named
// columns, and with no names it compares every column.
//
//	q := f.Lazy().Select("symbol", "sector").Distinct()
//
// It is [Frame.Distinct] written as a step of a query, so the rows come out in
// the order they were already in and the one kept out of a set of equal rows is
// the first of them. A name that is not there is an error at
// [LazyFrame.Collect].
//
// Naming the columns is worth doing when there are any that make a row unique,
// since it is what a projection pushdown will have to read. Selecting the
// columns first and then asking for the distinct rows of what is left, which is
// the query above, is the other way to say the same thing and the one that says
// what the answer holds as well.
func (lf *LazyFrame[S]) Distinct(names ...string) *LazyFrame[S] {
	if lf.err != nil {
		return lf
	}

	by := make([]*plan.Expr, len(names))
	for i, name := range names {
		by[i] = plan.Col(name)
	}
	return &LazyFrame[S]{node: plan.Distinct(lf.node, by)}
}

// Explode turns each element of the named list columns into a row of its own,
// repeating the other columns of the row it came from.
//
//	q := f.Lazy().Explode("tags").GroupBy("tags").Count()
//
// It is [Frame.Explode] written as a step of a query and it keeps every rule
// that one has: a row holding nothing becomes one row holding a missing value,
// several columns taken apart together have to agree about how many elements
// each row holds, and the columns stay where they were, holding the element
// type rather than a list of it.
//
// The result is a Dynamic query, since a column that held a list of strings
// holds a string afterwards and the schema type says otherwise.
// [LazyFrame.ExplodeAs] is the same step with a struct saying what the result
// holds, which is how a query stays typed across it.
//
// A name that is not there, a name that is not a list column, and no name at
// all are all errors at [LazyFrame.Collect]. Which column to take apart is not
// something this can work out on its own, since a query with two list columns
// in it has two answers and they are different frames.
func (lf *LazyFrame[S]) Explode(names ...string) *LazyFrame[Dynamic] {
	if lf.err != nil {
		return &LazyFrame[Dynamic]{err: lf.err}
	}
	return &LazyFrame[Dynamic]{node: plan.Explode(lf.node, names)}
}

// Sort puts the rows in order, the first key deciding and each later one
// breaking the ties of the one before.
//
// The sort is stable, for the reasons [Frame.Sort] gives, and a key can be an
// expression rather than a column name once [LazyFrame.SortByExpr] is used.
func (lf *LazyFrame[S]) Sort(by ...By) *LazyFrame[S] {
	if lf.err != nil {
		return lf
	}

	keys := make([]plan.SortKey, len(by))
	for i, b := range by {
		keys[i] = plan.SortKey{
			Expr:       plan.Col(b.Name),
			Descending: b.Descending,
			NullsFirst: b.NullsFirst,
		}
	}
	return &LazyFrame[S]{node: plan.Sort(lf.node, keys)}
}

// SortBy puts the rows in ascending order of the named columns, with the nulls
// at the end.
func (lf *LazyFrame[S]) SortBy(names ...string) *LazyFrame[S] {
	return lf.Sort(names2By(names, Order{})...)
}

// SortDesc puts the rows in descending order of the named columns, with the
// nulls at the end.
func (lf *LazyFrame[S]) SortDesc(names ...string) *LazyFrame[S] {
	return lf.Sort(names2By(names, Order{Descending: true})...)
}

// SortByExpr puts the rows in order of something that is worked out rather than
// read, such as the length of a string or the day of a timestamp.
//
// The column it sorts by is not in the result. Sorting by an expression and
// keeping it is [LazyFrame.With] and then a sort by its name.
func (lf *LazyFrame[S]) SortByExpr(e Expr[S], o Order) *LazyFrame[S] {
	if lf.err != nil {
		return lf
	}
	return &LazyFrame[S]{node: plan.Sort(lf.node, []plan.SortKey{{
		Expr:       e.expr(),
		Descending: o.Descending,
		NullsFirst: o.NullsFirst,
	}})}
}

// Head keeps the first n rows, and every row of a frame with fewer than n of
// them.
func (lf *LazyFrame[S]) Head(n int) *LazyFrame[S] { return lf.Slice(0, n) }

// Slice keeps at most n rows, having skipped the first off of them.
//
// It is the operator a slice pushdown sinks into a scan, so a head of twenty
// over a directory of Parquet files reads the first row group and stops rather
// than reading the lot and throwing it away. Neither number may be negative,
// since there is no such thing as skipping backwards, and [Frame.Tail] has no
// lazy spelling because how far back to start is something only the row count
// knows.
func (lf *LazyFrame[S]) Slice(off, n int) *LazyFrame[S] {
	if lf.err != nil {
		return lf
	}
	return &LazyFrame[S]{node: plan.Limit(lf.node, int64(off), int64(n))}
}

// Collect runs the query and returns what it produced.
//
// The plan is checked in full before anything is read, so a mistake anywhere in
// the query comes back here having cost nothing. What runs after that can still
// fail on what the data turns out to be, such as a file that is not there or a
// value that will not fit the type it is being cast to.
//
// The plan is optimized after it is checked and before it is run, which is what
// makes a query worth writing down rather than calling. A constant fold, a
// predicate pushdown, a slice pushdown, a common subexpression pass and a
// projection pushdown are the passes that are there now, so a step no data
// decides is worked out once at plan time, a filter runs before the join that
// would have paired the rows it throws away, a head of twenty is worked out for
// twenty rows rather than for all of them, a value written in three columns is
// worked out once, and a query over two columns of a frame of forty reads the
// two.
//
// The context is checked between operators. A query that is given up on stops
// at the next one rather than at the next row, which is a check that costs
// nothing to make and is close enough for reads that take seconds.
//
// The frame that comes back is checked against the schema type the way [Bind]
// checks one, so a typed query that produced the wrong columns is an error here
// rather than a handle that reads the wrong data later.
func (lf *LazyFrame[S]) Collect(ctx context.Context) (*Frame[S], error) {
	if lf.err != nil {
		return nil, lf.err
	}

	f, err := run(ctx, lf.node)
	if err != nil {
		return nil, err
	}
	return Bind[S](f)
}

// String returns the plan the query has been built into, one operator per line
// with the inputs indented under it.
//
// It is not [LazyFrame.Explain], which runs the passes and puts the query that
// will run next to the query that was written. This is the plan as it stands,
// before any of that, for reading in a test failure and at a prompt.
func (lf *LazyFrame[S]) String() string {
	if lf.err != nil {
		return "invalid query: " + lf.err.Error()
	}
	return lf.node.Tree()
}

// Explain returns the query as written, the query that will run, and the passes
// that made the difference between the two, in the format [plan.Explain]
// documents.
//
// It is the answer to the question a query engine is otherwise no help with,
// which is whether the thing you wrote is the thing that happens. A filter
// written over a join runs under it, a head of twenty reads twenty rows, a
// query over two columns of a frame of forty reads the two: all of that is
// worth having and none of it is visible from the query as written, so it is
// printable rather than something to be taken on trust.
//
// The passes it runs are the ones [LazyFrame.Collect] runs, so what it prints
// is what will happen and not an idea of what might. It does not read anything
// and it does not run the query. An error is what Collect would report about
// the same query, which is a query that does not check, and it is reported as
// the query was written rather than as some pass left it.
func (lf *LazyFrame[S]) Explain() (string, error) {
	if lf.err != nil {
		return "", lf.err
	}
	return plan.Explain(lf.node, plan.Passes()...)
}

// Profile runs the query and returns the answer along with what it cost, in the
// format [plan.Profile] documents.
//
// It is [LazyFrame.Explain] with a wall time and a row count on each operator,
// which is the other half of the same question. The explain says a filter runs
// before the join and the profile says whether that was where the time went,
// and neither is much use without the other when a query is slower than it
// looks like it should be.
//
// The answer comes back too, rather than being worked out and thrown away. A
// query that is worth timing is usually a query somebody wanted the answer to,
// and running it twice to have both would be a profile of a run that is not the
// one the answer came from.
//
// The clock is read once per operator. That costs a few nanoseconds against
// work measured in microseconds at the very least, so this is the same query
// and not a slower one put up to be measured, and the numbers are a fact about
// the run rather than an estimate of it.
func (lf *LazyFrame[S]) Profile(ctx context.Context) (*Frame[S], string, error) {
	if lf.err != nil {
		return nil, "", lf.err
	}

	f, ran, err := runMeasured(ctx, lf.node, &recorder{})
	if err != nil {
		return nil, "", err
	}

	text, err := plan.Profile(lf.node, ran, plan.Passes()...)
	if err != nil {
		return nil, "", err
	}

	out, err := Bind[S](f)
	if err != nil {
		return nil, "", err
	}
	return out, text, nil
}
