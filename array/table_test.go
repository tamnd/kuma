package array_test

import (
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// table builds a table of one column of n rows.
func table(t *testing.T, n int) *array.Table {
	t.Helper()

	b, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		t.Fatal(err)
	}
	for i := range n {
		b.Append(int64(i))
	}
	c, err := array.NewChunked(dtype.Int64, b.Finish())
	if err != nil {
		t.Fatal(err)
	}
	return &array.Table{
		Schema:  dtype.Schema{Fields: []dtype.Field{{Name: "n", Type: dtype.Int64}}},
		Columns: []*array.Chunked{c},
	}
}

// TestTable checks what a table says about its own shape.
//
// The rows are the length of a column rather than a number the table keeps, so
// there is nothing to get out of step with the columns and nothing to update
// when one of them is replaced.
func TestTable(t *testing.T) {
	tab := table(t, 3)

	if got, want := tab.NumRows(), 3; got != want {
		t.Errorf("the table holds %d rows, want %d", got, want)
	}
	if got, want := tab.NumCols(), 1; got != want {
		t.Errorf("the table holds %d columns, want %d", got, want)
	}
}

// TestTableEmpty checks the two ways a table holds nothing.
//
// A table of no columns has no rows to speak of, since the rows are what the
// columns hold, and a table of columns of no rows says nought as well.
func TestTableEmpty(t *testing.T) {
	var none array.Table
	if got := none.NumRows(); got != 0 {
		t.Errorf("a table of no columns holds %d rows, want none", got)
	}
	if got := none.NumCols(); got != 0 {
		t.Errorf("a table of no columns holds %d columns, want none", got)
	}

	tab := table(t, 0)
	if got := tab.NumRows(); got != 0 {
		t.Errorf("a table of empty columns holds %d rows, want none", got)
	}
	if got := tab.NumCols(); got != 1 {
		t.Errorf("a table of empty columns holds %d columns, want 1", got)
	}
}
