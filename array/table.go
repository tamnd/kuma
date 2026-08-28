package array

import "github.com/tamnd/kuma/dtype"

// Table is a schema and the columns that go with it, which is a frame with the
// frame taken off.
//
// It is what a file reader returns and what a file writer takes. A caller who
// wants rows and names and a query engine wants the frame the root package
// builds out of one of these; a caller who wants the columns wants this.
//
// It is here rather than in one of the format packages because every one of
// them means the same thing by it, and a table that came out of a CSV file and
// a table that came out of a parquet file should be the same type rather than
// two types a caller has to convert between.
type Table struct {
	// Schema is the name, type and nullability of each column, in order.
	Schema dtype.Schema

	// Columns holds one column per field of the schema, in the same order.
	Columns []*Chunked
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
