// Package ndjson reads and writes newline delimited JSON.
//
// One JSON object to a line and one line to a row, which is the shape a log
// file, an export out of a document store and most of what an API streams come
// in. There is no header and no schema in the file, so the columns are the
// members the objects turned out to have.
//
// The reader returns a [Table], which is a schema and a column for each member,
// rather than a frame. That is what keeps this package underneath kuma in the
// import graph, so kuma.ReadNDJSON can be a few lines over the top of it, and so
// anything that wants Arrow columns out of a text file can use this on its own
// without pulling in a query engine.
//
// # Schema inference
//
// With no types given, the reader looks at the first [Options.InferRows] lines
// and works out what the columns are and what each one holds. The columns are
// the members those lines had, in the order they were first seen, and the types
// it will infer are int64, float64, bool and string.
//
// JSON says what a value is, which is the one thing a CSV file cannot do, so
// inference here is reading rather than guessing. A number that fits in an int64
// is an int64, a number with a point or an exponent in it or one too large to
// fit is a float64, and a column with both is a float64. The literals true and
// false are a boolean. A quoted value is a string however much it looks like a
// number, and a column whose values come in two kinds is a string column, which
// is the type that holds whatever turned up.
//
// Naming a column in [Options.Types] gets a more permissive reading than
// inference does. A quoted value in a column that has been declared a number is
// parsed as one, since a caller who writes the type has said what the column is,
// and files that quote their numbers are common enough for that to be worth
// having.
//
// Dates and timestamps are not inferred. Text to timestamp parsing is a
// milestone 8 job, so for now a date column reads as a string and is cast
// afterwards.
//
// # Missing values
//
// A member whose value is null is missing, and so is a member a line does not
// have at all. That is the whole rule. There is nothing here like the CSV
// reader's list of null values, which a delimited file needs because it cannot
// say the difference between an empty field and an absent one. JSON can, so null
// is missing and a pair of quotes is a string of no bytes.
//
// # Nested objects and arrays
//
// A member holding an object or an array infers as a string column, and the
// value goes in as the JSON text it appeared as. That is enough to read the
// file and keep everything in it, and it is not a list column: writing such a
// table back out writes the text as a quoted string rather than as the object it
// came from, so that round trip is not exact. Real list and struct columns
// arrive with milestone 8 and this is what there is until then.
//
// # Members the schema has not got
//
// A line with a member that was not in the sample is an error. The alternative
// is dropping data that is in the file without saying so, which is the worse
// answer every time it matters. [Options.IgnoreUnknownFields] asks for it to be
// dropped, and a negative [Options.InferRows] reads the whole input before
// deciding, which is exact and holds the file in memory twice.
//
// A line with the same member on it twice is an error too, since one row cannot
// hold two values for one column and there is nothing to say which of them the
// file meant. The check is on the members being read, so a repeated member that
// nothing is reading is passed over like any other.
//
// # Reading part of a file
//
// [Options.Columns] names the members to read and drops the rest. A member that
// is not named is not inferred, not parsed and not held, and the columns come
// out in the order they were asked for rather than in the order the file has
// them.
//
// # Writing
//
// [Write] takes a table back out to a file, one object to a line, with the
// members in the order the schema has them. Values go out as they are stored, so
// a column of numbers is formatted into the output buffer and never becomes a
// column of strings along the way.
//
// Reading back what was written gives what went in, with two things that cannot
// work. A nested value is a string, as above. A float that is a NaN or an
// infinity goes out as null, because JSON has no way to write those values and a
// file that says NaN is not JSON at all.
//
// # Speed
//
// This is the reference reader and it is not the fast one. It is
// [encoding/json/jsontext] a value at a time, which is correct about escapes and
// UTF-8 and about what is and is not JSON. The vectorized reader from document
// 05 is checked against what is here, and where the two disagree this one is
// right by definition.
//
// It is still careful about what would cost a great deal for nothing. A line is
// read without a copy, a member name is looked up without one, and a value only
// becomes a Go string when it will not parse and has to go into an error.
//
// Stability: tier 1, stable.
package ndjson

import (
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Default values for the options that have one.
const (
	// DefaultInferRows is how many lines the reader looks at before deciding
	// what the columns are and what they hold. It is large enough that a member
	// only some of the objects have is usually caught, and small enough that
	// deciding costs nothing next to reading.
	DefaultInferRows = 1024

	// DefaultChunkSize is how many rows go into one chunk of a column. A column
	// arrives as a list of chunks rather than one buffer, so reading a file
	// larger than memory allows never asks for one allocation the size of the
	// file.
	DefaultChunkSize = 65536
)

// Options controls how a file is read. The zero Options is the useful default:
// the columns and their types worked out from the first thousand lines, and a
// null or an absent member meaning a missing value.
type Options struct {
	// Columns, if given, is the members to read. A name that no line in the
	// sample had is an error, and so is the same name twice.
	//
	// A member that is not named has no type inferred, no value parsed and no
	// memory held. Its value is still walked over, because a line cannot be read
	// past a member without reading it, but that is the cheap half of the work.
	//
	// The columns come out in the order they are named here, which need not be
	// the order the file has them.
	Columns []string

	// Types names the type of a column instead of inferring it. A column that is
	// not in here is inferred as usual. A name that no line in the sample had is
	// an error, since it is almost always a typo.
	//
	// The types that can be named are bool, the signed and unsigned integers,
	// the floats, string and binary. A declared column also takes a quoted value
	// and reads it as the type asked for, which inference never does.
	Types map[string]dtype.DataType

	// InferRows is how many lines to look at before deciding what the columns
	// are. Zero means [DefaultInferRows]. A negative value reads the whole input
	// before deciding, which is exact and holds the file in memory twice.
	InferRows int

	// ChunkSize is how many rows go into one chunk of a column. Zero means
	// [DefaultChunkSize].
	ChunkSize int

	// IgnoreUnknownFields drops a member that was not in the sample rather than
	// reporting the line. It is for the file whose last thousandth has a member
	// the first thousand lines did not.
	IgnoreUnknownFields bool

	// IgnoreParseErrors turns a value that will not go into its column into a
	// missing value instead of stopping the read. It is for the file that is
	// nearly clean and has to be loaded today. A line that is not JSON at all is
	// still an error, since there is no telling what was meant by it.
	IgnoreParseErrors bool
}

// withDefaults returns the options with the zero values filled in, so the rest
// of the package can read the fields without asking what zero meant each time.
func (o *Options) withDefaults() Options {
	out := Options{}
	if o != nil {
		out = *o
	}
	if out.InferRows == 0 {
		out.InferRows = DefaultInferRows
	}
	if out.ChunkSize <= 0 {
		out.ChunkSize = DefaultChunkSize
	}
	return out
}

// Table is what the reader returns and what the writer takes, which is a schema
// and the columns that go with it.
//
// It is [array.Table] because a table out of a JSON file and a table out of a
// parquet file are the same thing, and a caller holding one should not have to
// convert it to hand it to the other. A caller who wants rows and names and a
// query engine wants kuma.ReadNDJSON, which turns one of these into a frame.
type Table = array.Table
