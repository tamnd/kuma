package csv

import (
	stdcsv "encoding/csv"
	"errors"
	"fmt"
	"strconv"
)

// The errors this package returns, all comparable with [errors.Is].
var (
	// ErrNoData is returned when there is nothing to read at all, not even a
	// header. An input with a header and no rows is a table with no rows and
	// is not an error.
	ErrNoData = errors.New("no data")

	// ErrNames is returned when Options.Names has a different number of names
	// from the number of fields in the file, or when two columns end up with
	// the same name.
	ErrNames = errors.New("bad column names")

	// ErrNoColumn is returned when Options.Types names a column the file does
	// not have.
	ErrNoColumn = errors.New("no such column")

	// ErrUnsupportedType is returned when Options.Types names a type this
	// reader cannot parse text into, such as a list or a struct, and when a
	// column being written holds a type that has no text form at all.
	ErrUnsupportedType = errors.New("unsupported column type")

	// ErrDelimiter is returned when the delimiter or the comment character
	// cannot do the job, which means a quote, a line ending, a rune that is
	// not one, or the same character for both.
	ErrDelimiter = errors.New("invalid delimiter")

	// ErrTable is returned when a table cannot be written because it does not
	// hold together: a schema and a list of columns of different lengths, or
	// columns with different numbers of rows.
	ErrTable = errors.New("malformed table")

	// ErrValue is what a value that will not parse unwraps to. The error
	// itself is a *ValueError, which says which line and which column.
	ErrValue = errors.New("bad value")

	// ErrFieldCount is returned when a row has a different number of fields
	// from the one before it. ErrQuote is returned for a quote where the
	// format does not allow one, which Options.LazyQuotes turns off.
	//
	// Both come from [encoding/csv] and are named here so that a caller has
	// one package to ask about an error rather than two. The error itself is
	// an *encoding/csv.ParseError, which says which line.
	ErrFieldCount = stdcsv.ErrFieldCount
	ErrQuote      = stdcsv.ErrQuote
)

// ValueError says that one field could not be read as the type of its column.
//
// It carries the line, the column name and the value, because that is what it
// takes to go and look at the file. A parse error with only the message says
// the file is wrong somewhere, which is not much help in a million rows.
//
//	csv: line 4823, column "qty": cannot read "n/a" as int64: invalid syntax
type ValueError struct {
	// Line is the line in the input the value came from, counting from one and
	// counting the header.
	Line int

	// Column is the name of the column.
	Column string

	// Type is the type the value was being read as.
	Type string

	// Value is the field as it appeared in the file.
	Value string

	// Err is what the parse said, usually [strconv.ErrSyntax] or
	// [strconv.ErrRange].
	Err error
}

// Error returns the message described on ValueError.
func (e *ValueError) Error() string {
	msg := fmt.Sprintf("csv: line %d, column %q: cannot read %s as %s",
		e.Line, e.Column, strconv.Quote(e.Value), e.Type)
	if e.Err != nil {
		msg += ": " + e.Err.Error()
	}
	return msg
}

// Unwrap returns the errors this wraps, so that both errors.Is(err,
// csv.ErrValue) and errors.Is(err, strconv.ErrRange) answer about the same
// error.
func (e *ValueError) Unwrap() []error {
	if e.Err == nil {
		return []error{ErrValue}
	}
	return []error{ErrValue, e.Err}
}
