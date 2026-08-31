package kuma

import (
	"fmt"
	"reflect"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/plan"
)

// The steps that take the schema of their result as a type parameter live here.
//
// Most steps either keep the columns they were given, in which case they keep
// the schema type with them, or change them, in which case what comes back is
// Dynamic and [Bind] is the way back to a typed frame. The ones here are the
// middle case: the caller already knows what the result should hold, has
// written the struct for it, and would rather say so than take a Dynamic frame
// and bind it. Saying so is worth something beyond the line it saves, since the
// struct also says which columns to read and in which order, and a query that
// says that is a query a projection pushdown can do something with.
//
// They are spelled with an As because a method cannot be written twice with
// different type parameters. Select and SelectAs are the same step told two
// ways, the first by naming the columns and the second by naming the struct.
//
// There are four of them. Three change which columns there are, being a
// projection, an aggregation and a join, and the fourth is an explode, which
// keeps every column and changes what the ones it takes apart hold. Every
// other step either keeps the columns as they are or takes rows out, and those
// keep the schema type without being told anything.

// SelectAs keeps the columns the struct R names, in the order R names them, and
// returns a frame whose schema type is R.
//
//	type Quote struct {
//		Symbol string  `kuma:"symbol"`
//		Price  float64 `kuma:"price"`
//	}
//
//	q, err := f.SelectAs[Quote]()
//
// It is [Frame.Select] and [Bind] in one step, and the struct is what says which
// columns and in what order rather than a list of names that has to agree with
// the struct by hand. After it returns, a handle written for Quote works on the
// frame and a handle written for anything else does not compile.
//
// The column for a field is worked out the way [Bind] works it out, which is the
// kuma tag when there is one and the field name in snake case when there is not.
// A field tagged "-" is skipped and so is an unexported one, and a struct that
// names no columns at all is an error rather than an empty frame.
//
// It is an error if a column R names is not there, or is there and holds
// something the field cannot be read out of. Unlike [Bind] the columns R does
// not name are left out rather than kept, which is the difference between the
// two: a bind says what a frame holds and a select as says what to keep of it.
//
// Nothing is copied. The frame that comes back shares the columns it kept.
func (f *Frame[S]) SelectAs[R any]() (*Frame[R], error) {
	return selectAs[R](f, "SelectAs")
}

// AggAs works out the given aggregations for every group and returns the columns
// the struct R names, as a frame whose schema type is R.
//
//	type Total struct {
//		Symbol string  `kuma:"symbol"`
//		Qty    int64   `kuma:"qty"`
//		Price  float64 `kuma:"price"`
//	}
//
//	totals, err := g.AggAs[Total](kuma.Sum("qty"), kuma.Mean("price"))
//
// It is [GroupedFrame.Agg] and [Frame.SelectAs] in one step, so the struct names
// the key columns it wants as well as the aggregations, and the columns come out
// in the struct's order rather than in the keys first order the untyped one
// produces.
//
// An aggregation is named after the column it reads unless [Aggregation.As] says
// otherwise, so a struct that names a column no aggregation produced is an error
// saying which columns there were, and the fix is usually an As. Asking for an
// aggregation the struct does not name is not a mistake: it is worked out and
// then left out, the same as a column a select does not keep.
func (g *GroupedFrame[S]) AggAs[R any](aggs ...Aggregation) (*Frame[R], error) {
	f, err := g.Agg(aggs...)
	if err != nil {
		return nil, err
	}
	return selectAs[R](f, "AggAs")
}

// selectAs keeps the columns the struct R names, in the order R names them,
// which is what the eager typed steps all end in. The who is the step to name in
// an error, since a caller who wrote an AggAs is not helped by being told about
// a SelectAs they did not write.
func selectAs[R, S any](f *Frame[S], who string) (*Frame[R], error) {
	fields, err := schemaOf[R]()
	if err != nil {
		return nil, err
	}
	if len(fields) == 0 {
		return nil, noFields[R](who)
	}

	cols := make([]Column, len(fields))
	for i, sf := range fields {
		j, ok := f.index[sf.column]
		if !ok {
			return nil, noColumn(who, sf.column, f.Names())
		}
		c := f.cols[j]
		if !canReadType(sf.typ, c.DType()) {
			return nil, wrongField(who, sf, c.DType())
		}
		cols[i] = c
	}
	return newFrame[R](cols)
}

// ExplodeAs turns each element of the named list columns into a row of its own
// and keeps the columns the struct R names, as a frame whose schema type is R.
//
//	type Tagged struct {
//		Symbol string `kuma:"symbol"`
//		Tag    string `kuma:"tag"`
//	}
//
//	out, err := f.ExplodeAs[Tagged]("tag")
//
// It is [Frame.Explode] and [Frame.SelectAs] in one step, and it is the way an
// explode over a typed frame stays typed. It is worth more here than anywhere
// else, because a struct cannot name a list column today, so without this the
// frame on either side of an explode is Dynamic and the struct that describes
// the result is the only one there is to write.
//
// The exploded columns are nullable whatever the lists held, since a row
// holding nothing becomes a row holding a missing value, and a Go field that
// reads one of those reads the zero value. That is a fact about the data rather
// than about the type, which is why it is not an error, and it is the reason
// worth filtering the missing rows out before this step rather than after.
func (f *Frame[S]) ExplodeAs[R any](names ...string) (*Frame[R], error) {
	out, err := f.Explode(names...)
	if err != nil {
		return nil, err
	}
	return selectAs[R](out, "ExplodeAs")
}

// JoinAs puts the rows of two frames together where their keys match and keeps
// the columns the struct Out names, as a frame whose schema type is Out.
//
//	type Enriched struct {
//		Symbol string  `kuma:"symbol"`
//		Price  float64 `kuma:"price"`
//		Sector string  `kuma:"sector"`
//	}
//
//	both, err := trades.JoinAs[Enriched](sectors, kuma.Using("symbol"), kuma.InnerJoin)
//
// It is [Frame.Join] and [Frame.SelectAs] in one step, and it is the only one of
// the typed steps that has to name two schemas, since a join reads a second
// frame. The result comes first so that the type of the frame being joined to is
// worked out from the argument, which leaves one type to write rather than two.
//
// Every rule [Frame.Join] follows is kept, since it is that method underneath. A
// column the struct does not name is left out, so a wide join whose result is a
// handful of columns is one step here rather than a join and a select.
//
// The columns of a join that could not match are nullable, so a field of Out
// that reads one of them will read a null. That is a fact about the data rather
// than about the type, which is why it is not an error: a left join says the
// right side may be missing and a Go field cannot.
func (f *Frame[S]) JoinAs[Out, R any](other *Frame[R], on []On, how JoinType) (*Frame[Out], error) {
	joined, err := f.Join(other, on, how)
	if err != nil {
		return nil, err
	}
	return selectAs[Out](joined, "JoinAs")
}

// SelectAs keeps the columns the struct R names, in the order R names them, and
// gives back a query whose schema type is R.
//
//	out, err := f.Lazy().Filter(t.Price.Gt(100)).SelectAs[Quote]().Collect(ctx)
//
// It is [Frame.SelectAs] written as a step of a query, and it is the way a query
// over a typed frame stays typed through a projection. Without it a select gives
// back a Dynamic query and the schema has to be put back on at the end, by which
// point the compiler has spent the middle of the query unable to help.
//
// This is the one step that is checked where it is written rather than at
// [LazyFrame.Collect]. The struct says what the result should hold and the plan
// already knows what reaches this point, so the two can be compared there and
// then, and a column that is not there or holds the wrong type is an error the
// query carries from here. Every other step has nothing to compare against until
// the plan is walked.
func (lf *LazyFrame[S]) SelectAs[R any]() *LazyFrame[R] {
	n, err := projectAs[R](lf.node, "SelectAs")
	if err != nil {
		return &LazyFrame[R]{node: plan.Poison(lf.node, asStep[R]("SelectAs"), err)}
	}
	return &LazyFrame[R]{node: n}
}

// AggAs works out the given aggregations for every group and keeps the columns
// the struct R names, giving back a query whose schema type is R.
//
//	q := f.Lazy().GroupBy("symbol").AggAs[Total](kuma.Sum("qty"))
//
// It is [GroupedFrame.AggAs] written as a step of a query, and it is what keeps
// a query typed across the one step that changes every column it has. The struct
// names the keys it wants along with the aggregations, and what it does not name
// is left out.
//
// Like [LazyFrame.SelectAs] this is checked where it is written, so an
// aggregation that produced a column under a name the struct does not use is an
// error from that line rather than at [LazyFrame.Collect].
func (lg *LazyGroupBy[S]) AggAs[R any](aggs ...Aggregation) *LazyFrame[R] {
	q := lg.Agg(aggs...)
	n, err := projectAs[R](q.node, "AggAs")
	if err != nil {
		return &LazyFrame[R]{node: plan.Poison(q.node, asStep[R]("AggAs"), err)}
	}
	return &LazyFrame[R]{node: n}
}

// JoinAs puts the rows of two queries together where their keys match and keeps
// the columns the struct Out names, giving back a query whose schema type is
// Out.
//
//	q := trades.Lazy().JoinAs[Enriched](sectors.Lazy(), kuma.Using("symbol"), kuma.InnerJoin)
//
// It is [Frame.JoinAs] written as a step of a query, and it is the last of the
// steps that lets a typed query stay typed. The result type is written and the
// type of the query being joined to is worked out from the argument.
//
// Like [LazyFrame.SelectAs] this is checked where it is written, so a struct
// naming a column neither side has is an error from that line rather than at
// [LazyFrame.Collect]. That check is worth more here than anywhere else, since
// which columns a join produces is a rule with three parts to it and getting it
// wrong is easy.
func (lf *LazyFrame[S]) JoinAs[Out, R any](other *LazyFrame[R], on []On, how JoinType) *LazyFrame[Out] {
	q := lf.Join(other, on, how)
	n, err := projectAs[Out](q.node, "JoinAs")
	if err != nil {
		return &LazyFrame[Out]{node: plan.Poison(q.node, asStep[Out]("JoinAs"), err)}
	}
	return &LazyFrame[Out]{node: n}
}

// asStep is how a step that takes its result type as a type parameter is
// written in a plan that could not be built, which is the way it is written in
// the caller's code with the type spelled out in full. The package is part of
// it because two structs called Quote in two packages is the sort of thing that
// happens in the code a query is written in.
func asStep[R any](who string) string {
	return fmt.Sprintf("%s[%s]", who, reflect.TypeFor[R]())
}

// projectAs returns the plan that keeps the columns the struct R names, in the
// order R names them, checked against what the plan at n produces.
//
// The check is the point. The struct says what the result should hold and n
// already knows what reaches it, so the two are compared where the step was
// written rather than at Collect, which is what the untyped steps have to wait
// for. The who is the step to name in an error.
func projectAs[R any](n *plan.Node, who string) (*plan.Node, error) {
	fields, err := schemaOf[R]()
	if err != nil {
		return nil, err
	}
	if len(fields) == 0 {
		return nil, noFields[R](who)
	}

	s, err := n.Schema()
	if err != nil {
		return nil, err
	}

	cols := make([]plan.Projection, len(fields))
	for i, sf := range fields {
		fd, ok := s.Field(sf.column)
		if !ok {
			return nil, noColumn(who, sf.column, s.Names())
		}
		if !canReadType(sf.typ, fd.Type) {
			return nil, wrongField(who, sf, fd.Type)
		}
		cols[i] = plan.Projection{Expr: plan.Col(sf.column)}
	}
	return plan.Project(n, cols), nil
}

// ExplodeAs turns each element of the named list columns into a row of its own
// and keeps the columns the struct R names, giving back a query whose schema
// type is R.
//
//	q := f.Lazy().ExplodeAs[Tagged]("tag")
//
// It is [Frame.ExplodeAs] written as a step of a query, and it is the last of
// the steps that lets a typed query stay typed.
//
// Like [LazyFrame.SelectAs] this is checked where it is written, so a struct
// that reads an exploded column as the list it used to be is an error from that
// line rather than at [LazyFrame.Collect]. The plan works out what an explode
// produces without reading anything, which is what makes that possible.
func (lf *LazyFrame[S]) ExplodeAs[R any](names ...string) *LazyFrame[R] {
	q := lf.Explode(names...)
	n, err := projectAs[R](q.node, "ExplodeAs")
	if err != nil {
		return &LazyFrame[R]{node: plan.Poison(q.node, asStep[R]("ExplodeAs"), err)}
	}
	return &LazyFrame[R]{node: n}
}

// noFields is the error for a schema struct that names no columns, which is one
// with no exported fields or with every field tagged out.
//
// It is a mistake in the program rather than something the data did, but a
// struct is written once and read by a step that returns an error anyway, so
// there is nowhere useful for a panic to go.
func noFields[R any](who string) error {
	return fmt.Errorf("kuma: %s to %s, which names no columns: %w",
		who, reflect.TypeFor[R](), ErrNoValues)
}

// wrongField is the error for a column that is there under the name a field
// asked for and holds something the field cannot be read out of.
func wrongField(who string, sf schemaField, dt dtype.DataType) error {
	return fmt.Errorf("kuma: %s field %s wants column %q as a %s and it is a %s: %w",
		who, sf.field, sf.column, sf.typ, dt, ErrWrongType)
}
