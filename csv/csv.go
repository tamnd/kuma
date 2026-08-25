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
// # Speed
//
// This is the reference reader and it is not the fast one. It is
// [encoding/csv], which is correct about quotes and embedded newlines and both
// line endings, with one value parsed per field. The vectorized reader from
// document 05 is checked against what is here, and where the two disagree this
// one is right by definition.
//
// Stability: tier 1, stable.
package csv

import (
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

// isNull reports whether a field means that nothing is there.
func (o *Options) isNull(s string) bool {
	for _, v := range o.NullValues {
		if s == v {
			return true
		}
	}
	return false
}

// Table is a schema and the columns that go with it, which is a frame with the
// frame taken off.
//
// It is what the reader returns and what the writer takes. A caller who wants
// rows and names and a query engine wants kuma.ReadCSV, which turns one of
// these into a frame; a caller who wants the columns wants this.
type Table struct {
	// Schema is the name, type and nullability of each column, in order.
	Schema dtype.Schema

	// Columns holds one column per field of the schema, in the same order.
	Columns []*array.Chunked
}

// NumRows returns how many rows the table has.
func (t *Table) NumRows() int {
	if len(t.Columns) == 0 {
		return 0
	}
	return t.Columns[0].Len()
}

// NumCols returns how many columns the table has.
func (t *Table) NumCols() int { return len(t.Columns) }
