package kernel

import (
	"bytes"
	"cmp"
	"fmt"
	"math"
	"slices"
	"sort"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Order is one column of a sort, and how it takes part.
//
// The zero value of the two flags is ascending with the nulls at the end, which
// is what pandas does when nobody says otherwise and what most databases do.
type Order struct {
	Column     *array.Chunked
	Descending bool
	NullsFirst bool
}

// SortIndex returns the positions of the rows in sorted order, ready to hand to
// [Take].
//
// It returns positions rather than a sorted column because a sort of a table is
// one order applied to every column, and because the caller who wants the order
// itself, to apply to a second table or to look at, would otherwise have no way
// to ask for it.
//
// The first key decides, and each later one breaks the ties of the one before.
// The sort is stable, so rows that every key calls equal come out in the order
// they went in.
//
// Null placement is a property of the key and not of the direction. Asking for
// descending order does not move the nulls, which is what a database does with
// an explicit NULLS LAST and what makes a query that says both mean what it
// says. NaN is a value rather than a missing one, and it sorts after every
// number, so descending order puts it first.
//
// A dictionary encoded key is ordered by the values behind its indices, since
// the indices are the order a writer happened to meet the values in and mean
// nothing about the order they belong in. A row with no index and a row
// pointing at a dictionary entry that is itself null are both missing, and the
// placement puts them in the same block.
//
// It panics if there are no keys, if a key column is nil, or if two of them are
// different lengths, all of which are mistakes in the program rather than in
// the data. It returns an error for a column of a type there is no order for,
// since the key column is usually one a user picked at runtime.
func SortIndex(keys ...Order) ([]int, error) {
	if len(keys) == 0 {
		panic("kernel: sort with no keys")
	}

	n := -1
	cmps := make([]comparison, len(keys))
	for i, k := range keys {
		if k.Column == nil {
			panic(fmt.Sprintf("kernel: sort key %d is a nil column", i))
		}
		if n < 0 {
			n = k.Column.Len()
		} else if k.Column.Len() != n {
			panic(fmt.Sprintf("kernel: sort key %d has %d rows, key 0 has %d",
				i, k.Column.Len(), n))
		}

		c, err := comparisonFor(k)
		if err != nil {
			return nil, err
		}
		cmps[i] = c
	}

	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}

	// The stability does not come from a stable algorithm, it comes from the
	// comparison. Two rows that every key calls equal are ordered by the
	// positions they came in at, which makes the comparison a total order, and
	// a total order has one answer that every correct sort has to produce. That
	// buys the pattern defeating sort in slices.SortFunc, which is a good deal
	// faster than the merge in SortStableFunc and does not want the scratch
	// buffer that one allocates.
	//
	// One key is the common case, and the loop over keys is worth skipping in
	// it, since the body of a sort is the comparison and nothing else.
	if len(cmps) == 1 {
		slices.SortStableFunc(idx, cmps[0])
		return idx, nil
	}
	slices.SortStableFunc(idx, func(i, j int) int {
		for _, c := range cmps {
			if r := c(i, j); r != 0 {
				return r
			}
		}
		return 0
	})
	return idx, nil
}

// comparison orders two rows by their positions in the column, the way the
// function [slices.SortStableFunc] wants.
type comparison func(i, j int) int

// comparisonFor returns the comparison for one key, or an error if the column
// holds values there is no order for.
func comparisonFor(k Order) (comparison, error) {
	r := newRows(k.Column)

	// A dictionary encoded column is ordered by the values behind its indices
	// rather than by the indices, which are the order a writer happened to meet
	// the values in. Which comparison that wants is the values' business, and
	// reaching them is the rows' business.
	dt := k.Column.DType()
	if d, ok := dt.(dtype.Dictionary); ok {
		dt = d.Value
	}

	switch dt.Kind() {
	case dtype.NullKind:
		// Every value is missing, so every pair is equal and the placement has
		// nothing to place. A stable sort leaves the rows alone, which is the
		// only answer that does not invent an order.
		return func(_, _ int) int { return 0 }, nil
	case dtype.BoolKind:
		return compareBools(&r, k), nil
	case dtype.Int8Kind:
		return compareNumbers[int8](&r, k, cmp.Compare), nil
	case dtype.Int16Kind:
		return compareNumbers[int16](&r, k, cmp.Compare), nil
	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		return compareNumbers[int32](&r, k, cmp.Compare), nil
	case dtype.Int64Kind, dtype.Date64Kind, dtype.Time64Kind,
		dtype.TimestampKind, dtype.DurationKind:
		return compareNumbers[int64](&r, k, cmp.Compare), nil
	case dtype.Uint8Kind:
		return compareNumbers[uint8](&r, k, cmp.Compare), nil
	case dtype.Uint16Kind:
		return compareNumbers[uint16](&r, k, cmp.Compare), nil
	case dtype.Uint32Kind:
		return compareNumbers[uint32](&r, k, cmp.Compare), nil
	case dtype.Uint64Kind:
		return compareNumbers[uint64](&r, k, cmp.Compare), nil
	case dtype.Float32Kind:
		return compareNumbers[float32](&r, k, compareFloat), nil
	case dtype.Float64Kind:
		return compareNumbers[float64](&r, k, compareFloat), nil
	case dtype.StringKind, dtype.BinaryKind, dtype.FixedSizeBinaryKind:
		return compareBytes(&r, k), nil
	default:
		// The decimals are stored little endian, so comparing their bytes would
		// order them by their last digit, and the nested types have no single
		// order to give. Both want writing rather than guessing at.
		return nil, fmt.Errorf("kernel: there is no order for a %s column yet", k.Column.DType())
	}
}

// compareNumbers is the fixed width case, where the values are numbers or are
// stored as numbers.
//
// The typed slices are taken once for the column rather than once per
// comparison, which matters more here than in a gather: a sort reads every row
// log n times, and taking a typed slice checks the layout of the chunk and
// builds a slice header every time it is asked.
func compareNumbers[T array.Numeric](r *rows, k Order, less func(a, b T) int) comparison {
	vals := make([][]T, len(r.values))
	for c, a := range r.values {
		vals[c] = a.Values[T]()
	}

	desc, nullsFirst := k.Descending, k.NullsFirst
	return func(i, j int) int {
		ci, xi, oki := r.value(i)
		cj, xj, okj := r.value(j)
		if !oki || !okj {
			return compareNulls(!oki, !okj, nullsFirst)
		}

		d := less(vals[ci][xi], vals[cj][xj])
		if desc {
			return -d
		}
		return d
	}
}

// compareBools is the boolean case, whose values are bits and so are not
// reachable through a typed slice. False sorts before true.
func compareBools(r *rows, k Order) comparison {
	desc, nullsFirst := k.Descending, k.NullsFirst
	return func(i, j int) int {
		ci, xi, oki := r.value(i)
		cj, xj, okj := r.value(j)
		if !oki || !okj {
			return compareNulls(!oki, !okj, nullsFirst)
		}

		d := 0
		switch vi, vj := r.values[ci].Bool(xi), r.values[cj].Bool(xj); {
		case vi == vj:
		case vi:
			d = 1
		default:
			d = -1
		}
		if desc {
			return -d
		}
		return d
	}
}

// compareBytes is the case where a value is a run of bytes, which is the two
// string types and the fixed width one.
//
// The order is the order of the bytes. For a string that is the order of the
// code points, since UTF-8 was designed so that those two are the same thing,
// and it is what every database means by an ordering that nobody configured.
// What it is not is a collation: a Swedish speaker sorting Swedish names wants
// something else, and that is a table lookup and a whole other piece of work.
func compareBytes(r *rows, k Order) comparison {
	desc, nullsFirst := k.Descending, k.NullsFirst
	return func(i, j int) int {
		ci, xi, oki := r.value(i)
		cj, xj, okj := r.value(j)
		if !oki || !okj {
			return compareNulls(!oki, !okj, nullsFirst)
		}

		d := bytes.Compare(r.values[ci].Bytes(xi), r.values[cj].Bytes(xj))
		if desc {
			return -d
		}
		return d
	}
}

// compareNulls orders a pair where at least one value is missing.
//
// Two missing values are equal, which is the one place in this library where a
// null is treated as equal to another null. Sorting has to put them somewhere
// and putting them in a block is the only thing that makes NULLS FIRST mean
// anything, so this is the ordering answer to the question and not the
// arithmetic one.
func compareNulls(i, j, nullsFirst bool) int {
	if i && j {
		return 0
	}
	first := 1
	if nullsFirst {
		first = -1
	}
	if i {
		return first
	}
	return -first
}

// compareFloat orders two floats with NaN after every number.
//
// [cmp.Compare] puts NaN first, which is the answer that makes a NaN sort like
// a missing value. A NaN is not missing. It is the answer a computation gave,
// it survives arithmetic, and a column of prices with one NaN in it should not
// have that row appear where the smallest price belongs. Putting it after the
// numbers is what Polars does and what pandas does with its own missing values
// last, and it means ascending order ends with the values worth looking at.
func compareFloat[T float32 | float64](a, b T) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	case a == b:
		return 0
	}

	// At least one of them is NaN, since nothing else is unordered against
	// another float.
	an, bn := math.IsNaN(float64(a)), math.IsNaN(float64(b))
	switch {
	case an && bn:
		return 0
	case an:
		return 1
	default:
		return -1
	}
}

// rows turns a position in a column into a chunk and a position within it.
//
// This is the same job the finder in take.go does and it is a different type
// because the access pattern is the opposite one. A gather walks forwards, so a
// finder remembers where it was and usually guesses right. A sort jumps, so
// remembering the last answer would be a branch that is wrong nearly every
// time, and the thing worth doing instead is the single chunk case, which is
// most columns and which turns the lookup into nothing at all.
type rows struct {
	chunks []*array.Array

	// values[c] is where the values of chunk c are, which is the chunk itself
	// unless it is dictionary encoded, in which case it is the dictionary it
	// reads from. A comparison reads values out of these and nulls out of the
	// chunks, since which rows are missing is the chunk's to say.
	values []*array.Array

	// starts[c] is the position in the column that chunk c begins at.
	starts []int

	// one is true when there is a single chunk, so a position is already a
	// position in it.
	one bool
}

// newRows returns a rows over the chunks of c.
func newRows(c *array.Chunked) rows {
	chunks := c.Chunks()
	r := rows{
		chunks: chunks,
		values: make([]*array.Array, len(chunks)),
		starts: make([]int, len(chunks)),
		one:    len(chunks) == 1,
	}

	n := 0
	for k, a := range chunks {
		r.values[k] = a
		if d := a.Dictionary(); d != nil {
			r.values[k] = d
		}
		r.starts[k] = n
		n += a.Len()
	}
	return r
}

// value returns the chunk holding row i, where in that chunk's values the value
// of the row is, and whether there is a value there at all.
//
// The position is a position in values[chunk] rather than in the chunk, and the
// two are the same thing unless the chunk is dictionary encoded, in which case
// it is where in the dictionary the row's index points. A row with no index and
// a row pointing at a dictionary entry that is itself null are both missing.
func (r *rows) value(i int) (chunk, index int, ok bool) {
	c, x := r.at(i)
	a := r.chunks[c]
	if a.IsNull(x) {
		return c, x, false
	}
	if a.Dictionary() == nil {
		return c, x, true
	}

	x = a.Index(x)
	return c, x, !r.values[c].IsNull(x)
}

// at returns the chunk holding position i and the position within it.
func (r *rows) at(i int) (chunk, index int) {
	if r.one {
		return 0, i
	}
	c := sort.Search(len(r.chunks), func(k int) bool {
		return r.starts[k]+r.chunks[k].Len() > i
	})
	return c, i - r.starts[c]
}
