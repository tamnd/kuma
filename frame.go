package kuma

import (
	"fmt"
	"strings"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// Dynamic is the schema type for a frame whose columns are not known at compile
// time, which mostly means one read from a file nobody has seen yet.
//
// Everything on a Frame[Dynamic] takes column names as strings and reports an
// error when a name is wrong. The name is a little clumsy on purpose, so that
// reaching for the dynamic path is a visible decision rather than the thing
// that happens by default.
type Dynamic struct{}

// Frame is a table: an ordered list of named columns, all of the same length.
//
// The type parameter is the schema. It is a marker rather than something the
// frame stores, and it is what lets a generated column handle refuse to be used
// against the wrong table. A frame read from a file with no struct behind it is
// a Frame[Dynamic].
//
// There is no index. The one pandas has is the source of most of the surprising
// behavior in that library, where two frames silently align themselves by label
// in the middle of an arithmetic expression. Joins here take explicit keys and
// nothing aligns itself behind your back.
//
// A Frame is immutable. Every operation returns a new frame sharing the columns
// it did not change, which is what makes Select and Drop cost a slice header
// rather than a copy of the data. That is also what lets the same frame be
// handed to several goroutines.
//
// The zero Frame is not usable. Use NewFrame.
type Frame[S any] struct {
	cols   []Column
	schema dtype.Schema
	rows   int

	// index is the name to position map. A frame with a thousand columns is a
	// real thing, and a linear scan of the schema on every lookup would make a
	// query over a wide table quadratic in the number of columns.
	index map[string]int
}

// NewFrame returns a frame of the given columns, in the order given.
//
// Every column has to be the same length, since a table with a column of three
// values and a column of four is not a table. Names have to be there and have
// to be unique, which is what rejects the two columns called "id" that a CSV
// happily contains. Rename one of them first.
//
// A frame with no columns is fine and has no rows.
func NewFrame(cols ...Column) (*Frame[Dynamic], error) {
	return newFrame[Dynamic](cols)
}

// newFrame is NewFrame for any schema type. Binding a dynamic frame to a struct
// is what produces the other schema types, and it goes through here once it has
// checked the columns against the struct.
func newFrame[S any](cols []Column) (*Frame[S], error) {
	f := &Frame[S]{
		cols:  cols,
		index: make(map[string]int, len(cols)),
	}
	f.schema.Fields = make([]dtype.Field, len(cols))

	for i, c := range cols {
		if c.data == nil {
			return nil, fmt.Errorf("kuma: column %d has no values: %w", i, ErrNoValues)
		}
		if i == 0 {
			f.rows = c.Len()
		} else if c.Len() != f.rows {
			return nil, fmt.Errorf("kuma: column %q has %d rows, %q has %d: %w",
				c.Name(), c.Len(), cols[0].Name(), f.rows, ErrLength)
		}
		if _, dup := f.index[c.Name()]; dup {
			return nil, fmt.Errorf("kuma: two columns are called %q: %w", c.Name(), ErrDuplicateColumn)
		}

		f.index[c.Name()] = i
		f.schema.Fields[i] = c.Field()
	}

	if err := f.schema.Validate(); err != nil {
		return nil, fmt.Errorf("kuma: %w", err)
	}
	return f, nil
}

// Schema returns the fields of the frame, in column order.
//
// It describes the data rather than a declaration about it, so a field is
// nullable when the column has nulls in it. The result is a copy and the caller
// may keep it.
func (f *Frame[S]) Schema() dtype.Schema { return f.schema.Clone() }

// Names returns the column names in order.
func (f *Frame[S]) Names() []string { return f.schema.Names() }

// NumRows returns how many rows the frame has.
func (f *Frame[S]) NumRows() int { return f.rows }

// NumCols returns how many columns it has.
func (f *Frame[S]) NumCols() int { return len(f.cols) }

// Shape returns the number of rows and the number of columns, which is the pair
// people print first when they want to know whether a query did what they meant.
func (f *Frame[S]) Shape() (rows, cols int) { return f.rows, len(f.cols) }

// Index returns the position of the column with the given name, or -1 if the
// frame has no such column.
func (f *Frame[S]) Index(name string) int {
	if i, ok := f.index[name]; ok {
		return i
	}
	return -1
}

// Columns returns the columns in order.
//
// The result shares the frame's own slice and the caller must not modify it.
func (f *Frame[S]) Columns() []Column { return f.cols }

// Column returns the column with the given name.
func (f *Frame[S]) Column(name string) (Column, error) {
	i, ok := f.index[name]
	if !ok {
		return Column{}, noColumn("Column", name, f.Names())
	}
	return f.cols[i], nil
}

// ColumnAt returns column i. It panics if i is out of range, the way indexing a
// slice does, since a position out of range is a bug in the program rather than
// something the data did.
func (f *Frame[S]) ColumnAt(i int) Column {
	if uint(i) >= uint(len(f.cols)) {
		panic(fmt.Sprintf("kuma: column index %d out of range, the frame has %d columns", i, len(f.cols)))
	}
	return f.cols[i]
}

// Series returns the named column read as a Go type.
//
//	prices, err := f.Series[float64]("price")
//
// It reports an error if there is no such column, or if the column is not
// stored as something a T can be read out of. Nothing is copied.
func (f *Frame[S]) Series[T Value](name string) (Series[T], error) {
	c, err := f.Column(name)
	if err != nil {
		return Series[T]{}, err
	}
	return c.As[T]()
}

// Select returns a frame holding the named columns, in the order given.
//
// Naming the same column twice is a duplicate column and is rejected, since the
// result would be a frame with two columns of one name. Selecting no columns
// gives a frame with no columns.
//
// The result is dynamic whatever the frame it came from, because a subset of
// the columns of a Trade is not a Trade. Bind is the way back to a typed frame.
func (f *Frame[S]) Select(names ...string) (*Frame[Dynamic], error) {
	cols := make([]Column, 0, len(names))
	for _, name := range names {
		i, ok := f.index[name]
		if !ok {
			return nil, noColumn("Select", name, f.Names())
		}
		cols = append(cols, f.cols[i])
	}
	return newFrame[Dynamic](cols)
}

// Drop returns a frame with the named columns left out. The others keep their
// order.
//
// Dropping a column that is not there is an error rather than a shrug. A drop
// list that has gone stale is a bug worth hearing about, and the caller who
// genuinely does not know can ask Index first.
func (f *Frame[S]) Drop(names ...string) (*Frame[Dynamic], error) {
	drop := make(map[string]bool, len(names))
	for _, name := range names {
		if _, ok := f.index[name]; !ok {
			return nil, noColumn("Drop", name, f.Names())
		}
		drop[name] = true
	}

	cols := make([]Column, 0, len(f.cols)-len(drop))
	for _, c := range f.cols {
		if !drop[c.Name()] {
			cols = append(cols, c)
		}
	}
	return newFrame[Dynamic](cols)
}

// Rename returns a frame with the column called from called to instead. The
// column keeps its position.
func (f *Frame[S]) Rename(from, to string) (*Frame[Dynamic], error) {
	i, ok := f.index[from]
	if !ok {
		return nil, noColumn("Rename", from, f.Names())
	}

	cols := make([]Column, len(f.cols))
	copy(cols, f.cols)
	cols[i] = cols[i].Rename(to)
	return newFrame[Dynamic](cols)
}

// WithColumn returns a frame with the given column added at the end, or in
// place of the column of the same name if there is one.
//
// This is the assignment that pandas spells df["x"] = value, and it is a method
// returning a new frame rather than a statement mutating an old one, which is
// what stops a column appearing in a frame that some other goroutine is reading.
func (f *Frame[S]) WithColumn(c Column) (*Frame[Dynamic], error) {
	if len(f.cols) > 0 && c.Len() != f.rows {
		return nil, fmt.Errorf("kuma: column %q has %d rows, the frame has %d: %w",
			c.Name(), c.Len(), f.rows, ErrLength)
	}

	cols := make([]Column, len(f.cols), len(f.cols)+1)
	copy(cols, f.cols)
	if i, ok := f.index[c.Name()]; ok {
		cols[i] = c
	} else {
		cols = append(cols, c)
	}
	return newFrame[Dynamic](cols)
}

// Slice returns the rows from i up to but not including j. It panics unless
// 0 <= i <= j <= NumRows.
//
// Every column is sliced, which is constant time each, so this costs one slice
// per column whatever the number of rows. The result shares the memory it came
// from.
func (f *Frame[S]) Slice(i, j int) *Frame[S] {
	if i < 0 || j < i || j > f.rows {
		panic(fmt.Sprintf("kuma: Slice(%d, %d) of a frame of %d rows", i, j, f.rows))
	}

	cols := make([]Column, len(f.cols))
	for k, c := range f.cols {
		cols[k] = c.Slice(i, j)
	}
	return f.rebuild(cols, j-i)
}

// Take returns the rows at the given positions, in the order given.
//
// This is how every reordering of a frame is done. Sorting works out the order
// and takes it, a join works out which row of the left goes with which row of
// the right and takes both, and the values move here.
//
// A position below zero gives a row of nulls, which is what an outer join does
// with a row that matched nothing. A position at or past the end panics, the
// same way indexing a slice does.
//
// Unlike Slice this copies every column, since the rows it wants are scattered
// through the frame.
func (f *Frame[S]) Take(idx []int) *Frame[S] {
	checkPositions(idx, f.rows)

	cols := make([]Column, len(f.cols))
	for k, c := range f.cols {
		cols[k] = Column{name: c.name, data: kernel.Take(c.data, idx)}
	}
	return f.rebuild(cols, len(idx))
}

// Filter returns the rows that mask selects, in the order they were in.
//
// A null in the mask selects nothing. A row that nobody can say belongs in the
// result does not go in the result, which is the same rule a null gets
// everywhere else here.
//
// It reports an error if the mask is not as long as the frame.
func (f *Frame[S]) Filter(mask Series[bool]) (*Frame[S], error) {
	if mask.Len() != f.rows {
		return nil, fmt.Errorf("kuma: a mask of %d values for a frame of %d rows: %w",
			mask.Len(), f.rows, ErrLength)
	}
	// The positions are worked out once and every column is gathered with them,
	// rather than every column walking the mask for itself.
	return f.Take(kernel.Indices(mask.data)), nil
}

// rebuild returns a frame of the given columns, which have to be the columns of
// f in the same order under the same names and are allowed to hold different
// rows.
//
// The schema is built again rather than shared, because nullability is read off
// the data and the nulls in some of the rows are not the nulls in all of them.
// The name to position map is shared, since that is what did not change.
func (f *Frame[S]) rebuild(cols []Column, rows int) *Frame[S] {
	out := &Frame[S]{cols: cols, rows: rows, index: f.index}
	out.schema.Fields = make([]dtype.Field, len(cols))
	for k, c := range cols {
		out.schema.Fields[k] = c.Field()
	}
	return out
}

// Head returns the first n rows, or all of them if the frame is shorter than n.
// A negative n means all but the last n.
func (f *Frame[S]) Head(n int) *Frame[S] { return f.Slice(0, headEnd(n, f.rows)) }

// Tail returns the last n rows, or all of them if the frame is shorter than n.
// A negative n means all but the first n.
func (f *Frame[S]) Tail(n int) *Frame[S] { return f.Slice(tailStart(n, f.rows), f.rows) }

// String returns the shape of the frame and its columns, one per line.
//
// It is not the table from document 04 yet. The pretty printer that prints the
// rows is its own piece of work, and this is what a fmt.Println of a frame does
// until that lands.
func (f *Frame[S]) String() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "kuma.Frame[%T] %d rows x %d cols", *new(S), f.rows, len(f.cols))
	for _, c := range f.cols {
		fmt.Fprintf(&sb, "\n  %s: %s", c.Name(), c.DType())
		if n := c.NullCount(); n > 0 {
			fmt.Fprintf(&sb, ", %d null", n)
		}
	}
	return sb.String()
}
