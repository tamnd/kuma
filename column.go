package kuma

import (
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// Column is one named column of a frame, with no Go type attached.
//
// A frame holds columns of different types, so what it holds cannot be a
// Series[T] for one particular T. A Column is the same thing with the Go type
// dropped: the name and the values, and the column type to say what the values
// are. Putting the type back on is As, which checks it once.
//
// A Column is immutable and cheap to copy.
//
// The zero Column is not usable. Use NewColumn, or Series.Column.
type Column struct {
	name string
	data *array.Chunked
}

// NewColumn returns a column of the given name over the given values.
func NewColumn(name string, data *array.Chunked) (Column, error) {
	if data == nil {
		return Column{}, fmt.Errorf("kuma: no values for column %q: %w", name, ErrNoValues)
	}
	return Column{name: name, data: data}, nil
}

// Name returns the name of the column.
func (c Column) Name() string { return c.name }

// DType returns the type of the values.
func (c Column) DType() dtype.DataType { return c.data.DType() }

// Len returns how many values the column holds.
func (c Column) Len() int { return c.data.Len() }

// NullCount returns how many of them are missing.
func (c Column) NullCount() int { return c.data.NullCount() }

// IsNull reports whether value i is missing. It panics if i is out of range.
func (c Column) IsNull(i int) bool { return c.data.IsNull(i) }

// IsValid reports whether value i is present. It panics if i is out of range.
func (c Column) IsValid(i int) bool { return c.data.IsValid(i) }

// Data returns the values underneath.
func (c Column) Data() *array.Chunked { return c.data }

// Field returns the column as a schema field.
//
// Nullable is whether the column has any nulls in it, rather than whether it is
// allowed to. A frame holds data rather than a declaration, so its schema
// describes what is there: a column that has been filtered down to the rows
// with a value in them is not nullable any more, and saying so is what lets a
// writer pick the narrower encoding.
func (c Column) Field() dtype.Field {
	return dtype.Field{Name: c.name, Type: c.data.DType(), Nullable: c.data.NullCount() > 0}
}

// Rename returns the same column under a different name.
func (c Column) Rename(name string) Column {
	c.name = name
	return c
}

// Slice returns the values from i up to but not including j, as a column. It
// panics unless 0 <= i <= j <= Len.
func (c Column) Slice(i, j int) Column {
	c.data = c.data.Slice(i, j)
	return c
}

// Take returns the values at the given positions, in the order given, as a
// column.
//
// A position below zero gives a null, which is what an outer join does with a
// row that matched nothing. A position at or past the length panics.
//
// Unlike Slice this copies, because the values it wants are scattered through
// the column and there is no way to point at a scattering.
func (c Column) Take(idx []int) Column {
	checkPositions(idx, c.Len())
	c.data = kernel.Take(c.data, idx)
	return c
}

// String returns a short description of the column, for a log line or a test
// failure. It is not the values.
func (c Column) String() string {
	return fmt.Sprintf("kuma.Column{%q, %s, len %d, nulls %d}",
		c.name, c.data.DType(), c.data.Len(), c.data.NullCount())
}

// As returns the column as a Series[T], which is how the values are read as a
// Go type.
//
// It reports an error unless the column can be read as a T, which is what
// CanRead answers. Nothing is copied.
func (c Column) As[T Value]() (Series[T], error) {
	return SeriesFrom[T](c.name, c.data)
}

// MustAs is As where a wrong type is a bug rather than a condition, which is
// the case in a test, in an example, and anywhere the column was just built a
// few lines above. It panics if the column does not read as a T.
func (c Column) MustAs[T Value]() Series[T] {
	s, err := c.As[T]()
	if err != nil {
		panic(err.Error())
	}
	return s
}
