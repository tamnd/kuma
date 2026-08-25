package kuma

import (
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/kernel"
)

// Interpolation is how [Quantile] fills the gap between two values.
//
// It is [kernel.Interpolation] under another name, so the constants below are
// the same constants and either name may be used.
type Interpolation = kernel.Interpolation

// The five ways a quantile that falls between two values can be answered. They
// are the five pandas and numpy offer, and they mean the same things here.
const (
	// Linear splits the two neighbors by the fraction the position landed at.
	Linear = kernel.Linear

	// Lower takes the smaller neighbor and Higher the larger one, so a quantile
	// is always a value that was in the data.
	Lower  = kernel.Lower
	Higher = kernel.Higher

	// Nearest takes the closer neighbor, and the even index when they are
	// equally close.
	Nearest = kernel.Nearest

	// Midpoint splits the two neighbors evenly whatever the fraction was.
	Midpoint = kernel.Midpoint
)

// GroupedFrame is a frame with its rows divided into groups, waiting for
// somebody to say what to work out about each one.
//
// The division is done once, when [Frame.GroupBy] is called, and every
// aggregation after that reads it. That is why this is a value the caller holds
// rather than an argument to a method that does everything at once: asking for
// the total, the average and the count of a grouping costs one pass to work out
// the groups and three cheap passes over the values, not three groupings.
//
// A GroupedFrame is immutable, like the frame it came from.
type GroupedFrame[S any] struct {
	frame *Frame[S]
	names []string
	group *kernel.Groups
}

// GroupBy divides the rows into groups by the values of the named columns.
//
// Two rows are in the same group when they agree on every key. Missing counts
// as a value, so the rows whose key is missing form a group of their own, which
// is what SQL and Polars do. The pandas default is to drop those rows, and a
// row disappearing out of a total because a field was blank is the kind of
// thing that is noticed a quarter later.
//
// The groups come out in the order they first appear in the frame. That is
// deterministic without being sorted, so a caller who wants them sorted can
// sort the result and one who does not pay for it does not. The pandas default
// is to sort, and turning it off is a keyword argument.
//
// It reports an error if a name is not a column of the frame, or if a column is
// of a type there is no key encoding for yet, which today means the nested
// types.
func (f *Frame[S]) GroupBy(names ...string) (*GroupedFrame[S], error) {
	if len(names) == 0 {
		return nil, fmt.Errorf("kuma: GroupBy with no columns to group by: %w", ErrLength)
	}

	keys := make([]*array.Chunked, len(names))
	for i, name := range names {
		k, ok := f.index[name]
		if !ok {
			return nil, noColumn("GroupBy", name, f.Names())
		}
		keys[i] = f.cols[k].data
	}

	g, err := kernel.GroupBy(keys...)
	if err != nil {
		return nil, err
	}
	return &GroupedFrame[S]{frame: f, names: names, group: g}, nil
}

// NumGroups returns how many groups there are.
func (g *GroupedFrame[S]) NumGroups() int { return g.group.NumGroups() }

// Names returns the names of the columns the rows were grouped by.
func (g *GroupedFrame[S]) Names() []string { return append([]string(nil), g.names...) }

// Frame returns the frame the groups were worked out over.
func (g *GroupedFrame[S]) Frame() *Frame[S] { return g.frame }

// Groups returns the grouping underneath, for a caller who wants to run a
// kernel over it that there is no method for here.
func (g *GroupedFrame[S]) Groups() *kernel.Groups { return g.group }

// Keys returns the key columns, one row per group, in the order the groups came
// out in.
func (g *GroupedFrame[S]) Keys() []Column {
	keys := g.group.Keys()
	cols := make([]Column, len(keys))
	for i, k := range keys {
		cols[i] = Column{name: g.names[i], data: k}
	}
	return cols
}

// Agg works out the given aggregations for every group and returns them as a
// frame.
//
// The result has the key columns first, one row per group, followed by one
// column per aggregation in the order they were given. So a group by symbol
// with a sum of qty and an average of price comes back as three columns and as
// many rows as there are symbols, which is the table you would have written by
// hand.
//
// An aggregation is named after the column it reads unless [Aggregation.As]
// says otherwise, which is what Polars does. That means asking for two
// aggregations of the same column without naming them is an error about
// duplicate column names, and the fix is to say what you want them called.
//
// It reports an error if an aggregation names a column that is not there, or if
// a column is of a type that aggregation has no answer for, such as the sum of
// a string.
func (g *GroupedFrame[S]) Agg(aggs ...Aggregation) (*Frame[Dynamic], error) {
	if len(aggs) == 0 {
		return nil, fmt.Errorf("kuma: Agg with nothing to aggregate: %w", ErrLength)
	}

	cols := make([]Column, 0, len(g.names)+len(aggs))
	cols = append(cols, g.Keys()...)

	for _, a := range aggs {
		c, err := a.run(g)
		if err != nil {
			return nil, err
		}
		cols = append(cols, c)
	}
	return NewFrame(cols...)
}

// Count returns a frame of the keys and how many rows each group has, which is
// the group by anybody writes first.
//
// It counts rows and not values, so it is [Size] rather than [Count], and it
// cannot fail because there is no column to be the wrong type.
func (g *GroupedFrame[S]) Count() (*Frame[Dynamic], error) {
	f, err := g.Agg(Size())
	if err != nil {
		// Size reads no column, so the only way Agg fails here is a key column
		// already called "size", which is the caller's to fix.
		return nil, err
	}
	return f, nil
}
