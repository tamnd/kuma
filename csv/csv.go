// Package csv reads and writes comma separated values.
//
// The reader returns a [Table], which is a schema and a column for each field,
// rather than a frame. That is what keeps this package underneath kuma in the
// import graph, so kuma.ReadCSV can be a few lines over the top of it, and so
// anything that wants Arrow columns out of a text file can use this on its own
// without pulling in a query engine.
//
// # Schema inference
//
// With no types given, the reader looks at the first [Options.InferRows] rows
// and works out what each column holds. The types it will infer are int64,
// float64, bool and string, and nothing else. A value that reads as an integer
// is an integer, one that reads as a number but not an integer is a float, and
// the words true and false are a boolean. A column that holds integers in some
// rows and floats in others is a float column. Anything else mixed together is
// a string column, which is the type that can always hold what turned up.
//
// Inference is deliberately narrower than what a cast will accept. The single
// letters t and f are a boolean to [strconv.ParseBool] and are a string here,
// because a column of chemical symbols should not become a boolean column on
// the strength of the row that says F. Naming the type in [Options.Types] gets
// the permissive reading, since a caller who writes the type has said what the
// column is.
//
// Dates and timestamps are not inferred. Text to timestamp parsing is a
// milestone 8 job, so for now a date column reads as a string and is cast
// afterwards.
//
// # Missing values
//
// An empty field is a missing value in every column, including a string
// column, because a CSV file cannot tell an empty string apart from an absent
// one. [Options.NullValues] replaces that rule with a list of the caller's own,
// so a file that writes NA can say so, and a file that really does mean the
// empty string can pass a list that does not contain it.
//
// # Reading part of a file
//
// [Options.Columns] names the columns to read and drops the rest. A column that
// is not named is not inferred, not parsed and not held, which on a wide file
// is most of the work, and the columns come out in the order they were asked
// for rather than in the order the file has them.
//
// # Writing
//
// [Write] takes a table back out to a file. Values go out as they are stored,
// so a column of numbers is formatted into the output buffer and never becomes
// a column of strings along the way, and a field is quoted when it holds the
// delimiter, a quote or a line ending and left alone when it does not.
//
// Reading back what was written gives what went in, with one thing that cannot
// work: an empty string comes back as a missing value, because the file has no
// way to say which one it meant. [WriteOptions.NullValue] is the way out when
// the difference matters.
//
// # Speed
//
// This is the reference reader and it is not the fast one. It is
// [encoding/csv], which is correct about quotes and embedded newlines and both
// line endings, with one value parsed per field. The vectorized reader from
// document 05 is checked against what is here, and where the two disagree this
// one is right by definition.
//
// The writer is not encoding/csv's. That one takes a []string, which would
// mean a string allocated for every value in the table on the way past, so
// this one formats into a buffer of its own and hands whole blocks to the
// writer underneath.
//
// Stability: tier 1, stable.
package csv

import (
	"fmt"
	"unicode/utf8"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Default values for the options that have one.
const (
	// DefaultInferRows is how many rows the reader looks at before deciding
	// what the columns hold. It is large enough that a column of integers with
	// a float somewhere down the file is usually caught, and small enough that
	// deciding costs nothing next to reading.
	DefaultInferRows = 1024

	// DefaultChunkSize is how many rows go into one chunk of a column. A column
	// arrives as a list of chunks rather than one buffer, so reading a file
	// larger than memory allows never asks for one allocation the size of the
	// file.
	DefaultChunkSize = 65536
)

// Options controls how a file is read. The zero Options is the useful default:
// comma separated, a header row, types inferred, and an empty field meaning a
// missing value.
type Options struct {
	// Delimiter is what separates fields. Zero means a comma.
	Delimiter rune

	// Comment, if not zero, starts a comment line. A line whose first
	// non-space character is this is skipped entirely. It cannot be the same
	// as Delimiter.
	Comment rune

	// NoHeader says the first line is data rather than names. The names then
	// come from Names, or are generated as column_1, column_2 and so on.
	NoHeader bool

	// Names, if given, is the column names to use. It overrides a header line,
	// which is still read and thrown away, so a file with names that are not
	// valid identifiers can be renamed on the way in. It must have one name
	// for every field.
	Names []string

	// Columns, if given, is the columns to read, named as they are after Names
	// has been applied. A name that is not a column in the file is an error, and
	// so is the same name twice.
	//
	// A column that is not named has no type inferred, no value parsed and no
	// memory held. Its fields are still pulled out of each line, because a
	// delimited file cannot be read past a field without reading it, but that is
	// the cheap half of the work.
	//
	// The columns come out in the order they are named here, which need not be
	// the order the file has them.
	Columns []string

	// Types names the type of a column instead of inferring it. A column that
	// is not in here is inferred as usual. A name that is not a column in the
	// file is an error, since it is almost always a typo.
	//
	// The types that can be named are bool, the signed and unsigned integers,
	// the floats, string and binary. The parse is the same one [kernel.Cast]
	// does from text, which is more permissive than inference.
	Types map[string]dtype.DataType

	// InferRows is how many rows to look at before deciding what the columns
	// hold. Zero means [DefaultInferRows]. A negative value reads the whole
	// input before deciding, which is exact and holds the file in memory
	// twice.
	InferRows int

	// NullValues is the list of field values that mean nothing is there. A nil
	// list means the empty field and nothing else. A list that is not nil
	// replaces that rule rather than adding to it, so an empty but not nil
	// list means no field is missing and an empty field is an empty string.
	NullValues []string

	// ChunkSize is how many rows go into one chunk of a column. Zero means
	// [DefaultChunkSize].
	ChunkSize int

	// Skip is how many lines to throw away before reading anything, for the
	// files that start with a banner. The header, if there is one, is the
	// first line after these.
	Skip int

	// TrimLeadingSpace drops the space at the start of a field, even inside
	// quotes.
	TrimLeadingSpace bool

	// LazyQuotes accepts a quote in an unquoted field and an unescaped quote
	// in a quoted one, rather than reporting them.
	LazyQuotes bool

	// IgnoreParseErrors turns a value that will not parse into a missing value
	// instead of stopping the read. It is for the file that is nearly clean
	// and has to be loaded today.
	IgnoreParseErrors bool
}

// withDefaults returns the options with the zero values filled in, so the rest
// of the package can read the fields without asking what zero meant each time.
func (o *Options) withDefaults() Options {
	out := Options{}
	if o != nil {
		out = *o
	}
	if out.Delimiter == 0 {
		out.Delimiter = ','
	}
	if out.InferRows == 0 {
		out.InferRows = DefaultInferRows
	}
	if out.ChunkSize <= 0 {
		out.ChunkSize = DefaultChunkSize
	}
	if out.NullValues == nil {
		out.NullValues = []string{""}
	}
	return out
}

// checkDelim returns an error when the delimiter or the comment character
// cannot do the job it has been given. A comment of zero means there is none.
//
// The rule is [encoding/csv]'s. A character that is part of the format cannot
// also be a delimiter, and the two given here cannot be the same character,
// since a line has to be either a comment or a row of values.
func checkDelim(delim, comment rune) error {
	switch {
	case !validDelim(delim):
		return fmt.Errorf("csv: %q cannot separate fields: %w", delim, ErrDelimiter)
	case comment != 0 && !validDelim(comment):
		return fmt.Errorf("csv: %q cannot start a comment: %w", comment, ErrDelimiter)
	case comment != 0 && delim == comment:
		return fmt.Errorf("csv: %q cannot be both the delimiter and the comment: %w",
			delim, ErrDelimiter)
	}
	return nil
}

// validDelim reports whether r is a character the format leaves free.
func validDelim(r rune) bool {
	return r != 0 && r != '"' && r != '\r' && r != '\n' &&
		r != utf8.RuneError && utf8.ValidRune(r)
}

// isNull reports whether a field means that nothing is there.
func (o *Options) isNull(s string) bool {
	for _, v := range o.NullValues {
		if s == v {
			return true
		}
	}
	return false
}

// Table is what the reader returns and what the writer takes, which is a schema
// and the columns that go with it.
//
// It is [array.Table] because a table out of a CSV file and a table out of a
// parquet file are the same thing, and a caller holding one should not have to
// convert it to hand it to the other. A caller who wants rows and names and a
// query engine wants kuma.ReadCSV, which turns one of these into a frame.
type Table = array.Table
