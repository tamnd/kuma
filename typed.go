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
	fields, err := schemaOf[R]()
	if err != nil {
		return nil, err
	}
	if len(fields) == 0 {
		return nil, noFields[R]("SelectAs")
	}

	cols := make([]Column, len(fields))
	for i, sf := range fields {
		j, ok := f.index[sf.column]
		if !ok {
			return nil, noColumn("SelectAs", sf.column, f.Names())
		}
		c := f.cols[j]
		if !canReadType(sf.typ, c.DType()) {
			return nil, wrongField("SelectAs", sf, c.DType())
		}
		cols[i] = c
	}
	return newFrame[R](cols)
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
	if lf.err != nil {
		return &LazyFrame[R]{err: lf.err}
	}

	fields, err := schemaOf[R]()
	if err != nil {
		return &LazyFrame[R]{err: err}
	}
	if len(fields) == 0 {
		return &LazyFrame[R]{err: noFields[R]("SelectAs")}
	}

	s, err := lf.node.Schema()
	if err != nil {
		return &LazyFrame[R]{err: err}
	}

	cols := make([]plan.Projection, len(fields))
	for i, sf := range fields {
		fd, ok := s.Field(sf.column)
		if !ok {
			return &LazyFrame[R]{err: noColumn("SelectAs", sf.column, s.Names())}
		}
		if !canReadType(sf.typ, fd.Type) {
			return &LazyFrame[R]{err: wrongField("SelectAs", sf, fd.Type)}
		}
		cols[i] = plan.Projection{Expr: plan.Col(sf.column)}
	}
	return &LazyFrame[R]{node: plan.Project(lf.node, cols)}
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
