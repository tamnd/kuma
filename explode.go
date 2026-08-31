package kuma

import (
	"fmt"
	"slices"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// Explode turns each element of a list column into a row of its own, repeating
// the other columns of the row it came from.
//
//	orders, err := f.Explode("tags")
//
// It is what pandas calls explode and what SQL writes as an unnest. A row
// holding three elements becomes three rows, everything else in it is repeated
// three times, and the exploded column comes back holding the element type
// rather than a list of it.
//
// A row that is missing, and a row that is present and holds nothing, both
// become one row holding a missing value. That is what pandas does and it is
// the rule worth having, because the alternative is a row that disappears, and
// a frame where taking one column apart can lose a row is a frame whose row
// count depends on which column was chosen. Dropping those rows is a filter on
// the exploded column afterwards, which says so at the line that does it.
//
// Several columns can be taken apart in one step, and then the rows have to
// agree: a row with two elements in one column and three in another is two
// different answers about how many rows it becomes, and no answer to that is
// better than the others. It reports an error saying which row disagreed.
// Exploding them one after the other is a different thing, being every element
// of one against every element of the other, and is written as two calls.
//
// The result is dynamic whatever the frame it came from, because a Trade whose
// tags are a list is not a Trade once the list is gone. [Bind] is the way back
// to a typed frame.
//
// It reports an error if a name is not a column of the frame, if a named column
// is not a list column, or if no name is given at all. Which column to take
// apart is not something this can work out on its own: a frame with two list
// columns has two answers and they are different frames.
func (f *Frame[S]) Explode(names ...string) (*Frame[Dynamic], error) {
	if len(names) == 0 {
		return nil, fmt.Errorf("kuma: Explode needs the name of a column to take apart: %w", ErrNoColumn)
	}

	cols := make([]int, len(names))
	for i, name := range names {
		k, ok := f.index[name]
		if !ok {
			return nil, noColumn("Explode", name, f.Names())
		}
		if _, ok := f.cols[k].DType().(dtype.List); !ok {
			return nil, fmt.Errorf("kuma: Explode on column %q, which is %s and holds one value per row: %w",
				name, f.cols[k].DType(), ErrWrongType)
		}
		cols[i] = k
	}
	return explode(f, cols)
}

// explode is the eager [Frame.Explode] and the engine's explode operator, which
// differ in where the columns come from and in nothing else. Every column named
// here is a list column, which is what the callers check.
func explode[S any](f *Frame[S], cols []int) (*Frame[Dynamic], error) {
	flat := make([]*array.Chunked, len(cols))

	var rows []int
	for i, k := range cols {
		var came []int
		flat[i], came = kernel.Explode(f.cols[k].data)

		if i == 0 {
			rows = came
			continue
		}
		if err := sameRows(f.cols[cols[0]].Name(), f.cols[k].Name(), rows, came, f.rows); err != nil {
			return nil, err
		}
	}

	out := make([]Column, len(f.cols))
	for k, c := range f.cols {
		if i := slices.Index(cols, k); i >= 0 {
			out[k] = Column{name: c.name, data: flat[i]}
			continue
		}
		out[k] = Column{name: c.name, data: kernel.Take(c.data, rows)}
	}
	return newFrame[Dynamic](out)
}

// sameRows reports whether two columns came apart into the same rows, and says
// which row they disagreed about when they did not.
//
// The two lists say which row each of the new rows came from, so the columns
// agree exactly when every row of the frame held the same number of elements in
// both. Comparing the counts rather than the lists is the same answer and it is
// the number the message needs, so the check is written the way the error is
// rather than as a comparison of two slices with a second pass behind it.
func sameRows(a, b string, left, right []int, n int) error {
	lc, rc := counts(left, n), counts(right, n)

	for row := range lc {
		if lc[row] == rc[row] {
			continue
		}
		return fmt.Errorf("kuma: Explode cannot take row %d apart two ways, "+
			"since it holds %s in %q and %s in %q: %w",
			row, values(lc[row]), a, values(rc[row]), b, ErrLength)
	}
	return nil
}

// values is a count of values with the word after it, so that a message about
// one of them does not say one values.
func values(n int) string {
	if n == 1 {
		return "1 value"
	}
	return fmt.Sprintf("%d values", n)
}

// counts returns how many new rows each of the n rows came apart into.
func counts(rows []int, n int) []int {
	c := make([]int, n)
	for _, r := range rows {
		c[r]++
	}
	return c
}
