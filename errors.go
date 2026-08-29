package kuma

import (
	"errors"

	"github.com/tamnd/kuma/plan"
)

// The errors this package returns. They are comparable with errors.Is, and the
// error values themselves carry the detail: which column, which operation, and
// what the frame actually holds.
//
// Nothing here panics across an API boundary. The exceptions are the ones Go
// itself makes: an index out of range panics, the same way indexing a slice
// does, because a program that reads past the end of a column has a bug in it
// rather than a condition to handle.
var (
	// ErrNoColumn is returned when a column name is not in the frame. The error
	// lists the names that are, and suggests one if the name looks like a typo.
	// It is the same error a plan gives for a name that is not in the schema,
	// so one check covers a query however it went wrong.
	ErrNoColumn = plan.ErrNoColumn

	// ErrDuplicateColumn is returned when two columns in the same frame have
	// the same name. It is the same error a plan gives for an operator that
	// would produce two of them, for the reason ErrNoColumn is.
	ErrDuplicateColumn = plan.ErrDuplicateColumn

	// ErrWrongType is returned when a column is read as a Go type it is not
	// stored as, such as reading a float64 column as an int64, and when a value
	// is written into a query that no column can hold.
	ErrWrongType = plan.ErrWrongType

	// ErrLength is returned when the columns of a frame are not all the same
	// length.
	ErrLength = errors.New("columns of different length")

	// ErrNoValues is returned when a column is built with nothing underneath
	// it.
	ErrNoValues = errors.New("no values")
)

// ColumnError says that a column was asked for and is not there.
//
// It prints on several lines on purpose. A missing column name is the most
// common thing that goes wrong in day to day work, and the fastest way to fix
// it is to see what the frame does hold and to be told which of those names is
// one letter away from the one that was typed.
//
//	kuma: column "sym" not found in Select
//	  available: symbol, price, qty, side
//	  did you mean: symbol?
//
// It is [plan.ColumnError] under this name. The plan has to report a name that
// is not in the schema before anything is read, and there is no reason for a
// caller to meet two errors that say the same thing.
type ColumnError = plan.ColumnError

// noColumn returns the error for a name that is not in names.
func noColumn(op, name string, names []string) error {
	return &ColumnError{Op: op, Name: name, Have: names}
}
