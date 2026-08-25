package kuma

import (
	"fmt"

	"github.com/tamnd/kuma/kernel"
)

// Order says which way a sort runs and where the missing values go.
//
// The zero value is ascending with the nulls at the end, which is what pandas
// does when nobody says otherwise and what most databases do.
//
// Null placement is not part of the direction. Asking for descending order does
// not move the nulls, which is what a database does with an explicit NULLS LAST
// and what makes a query that says both mean what it says.
type Order struct {
	Descending bool
	NullsFirst bool
}

// By names a column to sort by, and how.
//
// The [Asc] and [Desc] functions cover the two common cases. The struct is
// there for the third one, which is null placement.
type By struct {
	Name string
	Order
}

// Asc returns a By that sorts the named column in ascending order, with the
// nulls at the end.
func Asc(name string) By { return By{Name: name} }

// Desc returns a By that sorts the named column in descending order, with the
// nulls at the end.
func Desc(name string) By { return By{Name: name, Order: Order{Descending: true}} }

// Sort returns the rows ordered by the given columns.
//
// The first key decides and each later one breaks the ties of the one before,
// so sorting by symbol and then by time gives the trades of each symbol in
// order. The sort is stable, so rows that every key calls equal come out in the
// order they went in.
//
// Stability is not optional here. Both pandas and Polars let a caller give the
// guarantee up for speed, with a kind argument in one and a maintain_order
// argument in the other. The guarantee is worth more than the speed: a stable
// sort makes the output of a query reproducible, which is what a test needs and
// what a diff of two runs needs, and it lets a caller build a sort out of
// several passes. The cost is a scratch buffer, since the comparison is a
// closure over columns and dwarfs everything around it.
//
// NaN is a value and not a missing one, so it sorts after every number and
// descending order puts it first.
//
// It reports an error if a name is not a column of the frame, or if a column is
// of a type there is no order for yet, which today means the decimals and the
// nested types.
func (f *Frame[S]) Sort(by ...By) (*Frame[S], error) {
	idx, err := f.SortIndex(by...)
	if err != nil {
		return nil, err
	}
	return f.Take(idx), nil
}

// SortBy returns the rows in ascending order of the named columns, with the
// nulls at the end. It is [Frame.Sort] for the common case.
func (f *Frame[S]) SortBy(names ...string) (*Frame[S], error) {
	return f.Sort(names2By(names, Order{})...)
}

// SortDesc returns the rows in descending order of the named columns, with the
// nulls at the end.
func (f *Frame[S]) SortDesc(names ...string) (*Frame[S], error) {
	return f.Sort(names2By(names, Order{Descending: true})...)
}

// SortIndex returns the positions that [Frame.Sort] would put the rows in,
// without moving anything.
//
// This is the operation to reach for when the order matters more than the
// sorted frame does: applying one frame's order to another, checking whether a
// frame is already sorted, or taking the first ten of a million rows without
// paying to move the other 999990.
func (f *Frame[S]) SortIndex(by ...By) ([]int, error) {
	if len(by) == 0 {
		return nil, fmt.Errorf("kuma: Sort with no columns to sort by: %w", ErrLength)
	}

	keys := make([]kernel.Order, len(by))
	for i, b := range by {
		k, ok := f.index[b.Name]
		if !ok {
			return nil, noColumn("Sort", b.Name, f.Names())
		}
		keys[i] = kernel.Order{
			Column:     f.cols[k].data,
			Descending: b.Descending,
			NullsFirst: b.NullsFirst,
		}
	}

	idx, err := kernel.SortIndex(keys...)
	if err != nil {
		return nil, err
	}
	return idx, nil
}

// names2By turns a list of names into a list of keys that share one order.
func names2By(names []string, o Order) []By {
	by := make([]By, len(names))
	for i, name := range names {
		by[i] = By{Name: name, Order: o}
	}
	return by
}

// Sort returns the values in the order o describes.
//
// A series is one column, so there are no ties to break and the stability of
// the sort is not something anyone can observe. Everything else [Frame.Sort]
// says applies: the nulls go where o puts them whichever way the values run,
// and NaN sorts after every number.
//
// It reports an error if the column is of a type there is no order for yet.
func (s Series[T]) Sort(o Order) (Series[T], error) {
	idx, err := kernel.SortIndex(kernel.Order{
		Column:     s.data,
		Descending: o.Descending,
		NullsFirst: o.NullsFirst,
	})
	if err != nil {
		return Series[T]{}, err
	}
	return s.Take(idx), nil
}

// SortIndex returns the positions that [Series.Sort] would put the values in,
// without moving them. It is what pandas calls argsort.
func (s Series[T]) SortIndex(o Order) ([]int, error) {
	return kernel.SortIndex(kernel.Order{
		Column:     s.data,
		Descending: o.Descending,
		NullsFirst: o.NullsFirst,
	})
}

// Sort returns the values of the column in the order o describes.
func (c Column) Sort(o Order) (Column, error) {
	idx, err := kernel.SortIndex(kernel.Order{
		Column:     c.data,
		Descending: o.Descending,
		NullsFirst: o.NullsFirst,
	})
	if err != nil {
		return Column{}, err
	}
	c.data = kernel.Take(c.data, idx)
	return c, nil
}
