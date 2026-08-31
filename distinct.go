package kuma

import (
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/kernel"
)

// Distinct returns the frame with only the first row of each set of rows that
// agree on the named columns, and with no names it compares every column.
//
// It is what pandas calls drop_duplicates and what SQL writes as a select
// distinct. The rows that are kept come out in the order they were already in,
// and the one kept out of a set of equal rows is the first of them, which is
// the pandas default. Keeping the last instead, and throwing away every row
// that has a duplicate rather than all but one of them, are the other two
// things the pandas keep argument can say and are a later change.
//
// Two rows agree on a column when they hold the same value, and a missing value
// agrees with a missing value, which is the rule [Frame.GroupBy] follows.
// Naming the columns that make a row unique is worth doing when there are any,
// since comparing three key columns reads three columns and comparing nothing in
// particular reads all forty.
//
// It reports an error if a name is not a column of the frame, or if a column
// being compared is of a type there is no way to compare, which today means the
// nested types.
func (f *Frame[S]) Distinct(names ...string) (*Frame[S], error) {
	cols, err := f.namedColumns("Distinct", names)
	if err != nil {
		return nil, err
	}
	return distinct(f, columnData(cols))
}

// distinct returns the frame with only the first row of each set of rows the
// given key columns agree on. It is the eager [Frame.Distinct] and the engine's
// distinct operator, which differ in where the key columns come from and in
// nothing else.
func distinct[S any](f *Frame[S], keys []*array.Chunked) (*Frame[S], error) {
	// A frame with no columns has no rows either, so there is nothing to
	// compare and nothing to take out.
	if len(keys) == 0 {
		return f, nil
	}

	idx, err := kernel.DistinctIndex(keys...)
	if err != nil {
		return nil, err
	}

	// A frame that was already distinct is its own answer. The positions come
	// out in order and there is one of them per row, so a count that matches
	// means every row was the first of its own set, and gathering them would
	// copy the whole frame to produce a copy of it.
	if len(idx) == f.rows {
		return f, nil
	}
	return f.Take(idx), nil
}

// columnData is the data of some columns, which is what the kernels take.
func columnData(cols []Column) []*array.Chunked {
	data := make([]*array.Chunked, len(cols))
	for i, c := range cols {
		data[i] = c.data
	}
	return data
}
