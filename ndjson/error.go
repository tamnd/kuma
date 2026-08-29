package ndjson

import (
	"errors"
	"fmt"
)

// The errors this package returns, all comparable with [errors.Is].
var (
	// ErrNoData is returned when there is nothing to read at all, or nothing
	// that says what the columns are. A file of empty objects is the second of
	// those: there are lines in it and no members anywhere, so there is no
	// schema to be had. A file whose objects have members and no rows after the
	// sample is a table with rows and is not an error.
	ErrNoData = errors.New("no data")

	// ErrNames is returned when Options.Columns names the same column twice, and
	// when WriteOptions.Names has a different number of names from the number of
	// columns in the table.
	ErrNames = errors.New("bad column names")

	// ErrNoColumn is returned when Options.Columns or Options.Types names a
	// member that no line in the sample had.
	ErrNoColumn = errors.New("no such column")

	// ErrUnsupportedType is returned when Options.Types names a type this reader
	// cannot read JSON into, such as a list or a struct, and when a column being
	// written holds a type that has no JSON form at all.
	ErrUnsupportedType = errors.New("unsupported column type")

	// ErrSyntax is what a line that is not one JSON object unwraps to, which
	// covers a line that is JSON but not an object, a line with more than one
	// value on it, and a line that is not JSON. The error itself is a
	// *LineError, which says which line, and the last of those wraps a
	// *jsontext.SyntacticError with the detail.
	ErrSyntax = errors.New("malformed JSON")

	// ErrUnknownField is what a member the schema has not got unwraps to. The
	// error itself is a *LineError. Options.IgnoreUnknownFields turns it off.
	ErrUnknownField = errors.New("unknown member")

	// ErrTable is returned when a table cannot be written because it does not
	// hold together: a schema and a list of columns of different lengths, or
	// columns with different numbers of rows.
	ErrTable = errors.New("malformed table")

	// ErrValue is what a value that will not go into its column unwraps to. The
	// error itself is a *ValueError, which says which line and which column.
	ErrValue = errors.New("bad value")
)

// LineError says that one line could not be read.
//
// It carries the line so that a caller can go and look at the file, which is
// the whole difference between an error worth having and one that says the
// input is wrong somewhere.
//
//	ndjson: line 12: more than one value on the line: malformed JSON
type LineError struct {
	// Line is the line in the input, counting from one and counting the blank
	// lines that were passed over.
	Line int

	// Err is what was wrong with it, which unwraps to [ErrSyntax] or to
	// [ErrUnknownField].
	Err error
}

// Error returns the message described on LineError.
func (e *LineError) Error() string {
	return fmt.Sprintf("ndjson: line %d: %s", e.Line, e.Err)
}

// Unwrap returns what was wrong with the line, so that asking about ErrSyntax
// and asking about the *jsontext.SyntacticError underneath it both work.
func (e *LineError) Unwrap() error { return e.Err }

// ValueError says that one value could not be read as the type of its column.
//
// It carries the line, the column name and the value, because that is what it
// takes to go and look at the file. A parse error with only the message says
// the file is wrong somewhere, which is not much help in a million rows.
//
//	ndjson: line 4823, column "qty": cannot read 3.9 as int64: invalid syntax
type ValueError struct {
	// Line is the line in the input the value came from, counting from one.
	Line int

	// Column is the name of the column, which is the name of the member.
	Column string

	// Type is the type the value was being read as.
	Type string

	// Value is the JSON text of the value as it appeared in the file, so a
	// string still has its quotes and an object is the whole object.
	Value string

	// Err is what the parse said, usually [strconv.ErrSyntax], [strconv.ErrRange]
	// or one of the errors below about the kind of the value.
	Err error
}

// Error returns the message described on ValueError.
func (e *ValueError) Error() string {
	msg := fmt.Sprintf("ndjson: line %d, column %q: cannot read %s as %s",
		e.Line, e.Column, e.Value, e.Type)
	if e.Err != nil {
		msg += ": " + e.Err.Error()
	}
	return msg
}

// Unwrap returns the errors this wraps, so that both errors.Is(err,
// ndjson.ErrValue) and errors.Is(err, strconv.ErrRange) answer about the same
// error.
func (e *ValueError) Unwrap() []error {
	if e.Err == nil {
		return []error{ErrValue}
	}
	return []error{ErrValue, e.Err}
}
