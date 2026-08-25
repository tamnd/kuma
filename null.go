package kuma

import (
	"fmt"
	"time"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// HasNulls reports whether anything in the column is missing.
func (c Column) HasNulls() bool { return c.data.NullCount() > 0 }

// NullMask returns a boolean column that is true where this one has no value.
//
// It keeps the name of the column it came from, so a mask over a frame of them
// reads like the frame it describes. The result has no nulls of its own,
// because whether a value is missing is always known even when the value is
// not.
//
// The per row question is [Column.IsNull]. This is the whole column at once,
// which is what a filter wants.
func (c Column) NullMask() Column {
	return Column{name: c.name, data: kernel.IsNull(c.data)}
}

// ValidMask returns a boolean column that is true where this one has a value.
// It is [Column.NullMask] the other way round, and it is the mask a filter
// usually wants.
func (c Column) ValidMask() Column {
	return Column{name: c.name, data: kernel.IsNotNull(c.data)}
}

// FillNull returns the column with every missing value replaced by v.
//
// The type argument is what v is written as and the column keeps the type it
// already had, so a timestamp column takes a [time.Time] and stays a timestamp
// column of the unit it was. A T the column cannot be read as is an error, the
// same one [Column.As] gives.
//
// A column with nothing missing is handed straight back rather than copied.
func (c Column) FillNull[T Value](v T) (Column, error) {
	if !c.HasNulls() {
		return c, nil
	}

	fill, err := oneValue(c.DType(), v)
	if err != nil {
		return Column{}, fmt.Errorf("kuma: column %q: %w", c.name, err)
	}

	filled, err := kernel.FillNull(c.data, fill)
	if err != nil {
		return Column{}, fmt.Errorf("kuma: %w", err)
	}
	return Column{name: c.name, data: filled}, nil
}

// DropNulls returns the column with the missing values taken out. It is shorter
// than the column it came from by however many there were.
func (c Column) DropNulls() Column {
	return Column{name: c.name, data: kernel.Filter(c.data, kernel.IsNotNull(c.data))}
}

// HasNulls reports whether anything in the series is missing.
func (s Series[T]) HasNulls() bool { return s.data.NullCount() > 0 }

// NullMask returns a boolean series that is true where this one has no value.
func (s Series[T]) NullMask() Series[bool] {
	return s.Column().NullMask().MustAs[bool]()
}

// ValidMask returns a boolean series that is true where this one has a value.
func (s Series[T]) ValidMask() Series[bool] {
	return s.Column().ValidMask().MustAs[bool]()
}

// FillNull returns the series with every missing value replaced by v.
//
// There is no error to return here, which is the difference between this and
// [Column.FillNull]. A series is a column that has already been read as a T, so
// a T is by construction a type the column takes and the only thing the column
// version can complain about cannot happen.
func (s Series[T]) FillNull(v T) Series[T] {
	filled, err := s.Column().FillNull(v)
	if err != nil {
		// The series is already read as a T, so a T is a type the column takes.
		panic("kuma: " + err.Error())
	}
	return filled.MustAs[T]()
}

// DropNulls returns the series with the missing values taken out.
func (s Series[T]) DropNulls() Series[T] {
	return s.Column().DropNulls().MustAs[T]()
}

// HasNulls reports whether anything in the frame is missing.
func (f *Frame[S]) HasNulls() bool {
	for _, c := range f.cols {
		if c.HasNulls() {
			return true
		}
	}
	return false
}

// NullCounts returns how many values are missing in each column, in column
// order.
//
// It is a slice rather than a map because the columns of a frame are ordered
// and a map would throw that away. Names() lines up with it.
func (f *Frame[S]) NullCounts() []int {
	out := make([]int, len(f.cols))
	for i, c := range f.cols {
		out[i] = c.NullCount()
	}
	return out
}

// IsNull returns a frame of boolean columns, one for each column of this frame
// and with the same name, true where the value is missing.
//
// This is the pandas isna, and it is a frame rather than a mask because a frame
// can hold a column for each of the columns asked about. Counting what is
// missing in each column is what [Frame.NullCounts] is for and does not need
// this.
func (f *Frame[S]) IsNull() *Frame[Dynamic] {
	return f.maskFrame(Column.NullMask)
}

// IsNotNull returns a frame of boolean columns that are true where the value is
// there. It is [Frame.IsNull] the other way round.
func (f *Frame[S]) IsNotNull() *Frame[Dynamic] {
	return f.maskFrame(Column.ValidMask)
}

// maskFrame builds the frame of boolean columns both of the two above want.
func (f *Frame[S]) maskFrame(mask func(Column) Column) *Frame[Dynamic] {
	cols := make([]Column, len(f.cols))
	for i, c := range f.cols {
		cols[i] = mask(c)
	}

	out, err := newFrame[Dynamic](cols)
	if err != nil {
		// The columns are the ones that were already in a frame, renamed to
		// nothing and of the same lengths.
		panic("kuma: " + err.Error())
	}
	return out
}

// FillNull returns a frame with every missing value of the named column
// replaced by v. The column keeps its position and everything else is left
// alone.
//
// The result is a Dynamic frame because the schema changed: a column with
// nothing missing is not nullable and one with something missing is, and that
// is part of what a typed frame promises.
func (f *Frame[S]) FillNull[T Value](name string, v T) (*Frame[Dynamic], error) {
	i, ok := f.index[name]
	if !ok {
		return nil, noColumn("FillNull", name, f.Names())
	}

	filled, err := f.cols[i].FillNull(v)
	if err != nil {
		return nil, err
	}

	cols := make([]Column, len(f.cols))
	copy(cols, f.cols)
	cols[i] = filled
	return newFrame[Dynamic](cols)
}

// DropNulls returns the frame with the rows that have a missing value in any of
// the named columns taken out. With no names it looks at every column, which is
// the pandas dropna default.
//
// [Frame.KeepAtLeast] is the same thing with the rule relaxed, for the caller
// who wants the rows that are mostly there rather than the rows that are
// entirely there.
func (f *Frame[S]) DropNulls(names ...string) (*Frame[S], error) {
	cols, err := f.nullCheckColumns("DropNulls", names)
	if err != nil {
		return nil, err
	}
	return f.keep(len(cols), cols), nil
}

// KeepAtLeast returns the frame with only the rows that have at least present
// values among the named columns. With no names it looks at every column.
//
// This is the pandas thresh, and the pandas how falls out of it: how="any" is
// [Frame.DropNulls], which is this with present set to all of the columns, and
// how="all" is this with present set to one.
//
// A present of zero or below keeps every row, since every row has at least
// nothing.
func (f *Frame[S]) KeepAtLeast(present int, names ...string) (*Frame[S], error) {
	cols, err := f.nullCheckColumns("KeepAtLeast", names)
	if err != nil {
		return nil, err
	}
	return f.keep(present, cols), nil
}

// nullCheckColumns returns the columns the named ones are, or all of them when
// no names were given.
func (f *Frame[S]) nullCheckColumns(who string, names []string) ([]Column, error) {
	if len(names) == 0 {
		return f.cols, nil
	}

	cols := make([]Column, len(names))
	for i, name := range names {
		j, ok := f.index[name]
		if !ok {
			return nil, noColumn(who, name, f.Names())
		}
		cols[i] = f.cols[j]
	}
	return cols, nil
}

// keep returns the frame with only the rows that have at least present values
// among the given columns.
func (f *Frame[S]) keep(present int, cols []Column) *Frame[S] {
	// A row can only fail on a column that has something missing in it, so a
	// frame of complete data is answered here, by counting the columns, and
	// hands back the frame it was given rather than a copy of it.
	complete := 0
	for _, c := range cols {
		if !c.HasNulls() {
			complete++
		}
	}
	if present <= 0 || complete >= present {
		return f
	}

	data := make([]*array.Chunked, len(cols))
	for i, c := range cols {
		data[i] = c.Data()
	}
	return f.Take(kernel.KeepIndex(data, f.rows, present))
}

// oneValue returns a one value array of type dt holding v.
//
// This is where a Go value becomes a column value, and the only interesting
// case is a timestamp: a time.Time is nanoseconds and the column may be in
// seconds, so the value is scaled into the unit the column is stored in rather
// than the column being converted into the unit the caller happened to write.
func oneValue[T Value](dt dtype.DataType, v T) (*array.Array, error) {
	if !CanRead[T](dt) {
		var zero T
		return nil, fmt.Errorf("a %s column does not take a %T: %w", dt, zero, ErrWrongType)
	}

	b, err := array.NewBuilder(dt)
	if err != nil {
		return nil, err
	}

	if t, ok := any(v).(time.Time); ok {
		unit := int64(1)
		if ts, isTS := dt.(dtype.Timestamp); isTS {
			unit = nanosPerUnit(ts.Unit)
		}
		b.Append(t.UnixNano() / unit)
		return b.Finish(), nil
	}

	appendValues(b, []T{v})
	return b.Finish(), nil
}
