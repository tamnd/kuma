package kernel

import (
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Filter returns a column holding the values of c that mask selects, in the
// order they were in.
//
// A null in the mask selects nothing. It is not a value that happens to be
// false, it is the absence of an answer, and a row nobody can say belongs in
// the result does not go in the result. That is what Polars does. The pandas
// answer was a warning and is now an error, which is the same behavior with
// more noise around it.
//
// It panics if the two columns are not the same length, or if mask is not a
// boolean column.
func Filter(c, mask *array.Chunked) *array.Chunked {
	if c == nil {
		panic("kernel: filter of a nil column")
	}
	checkMask(mask)
	if c.Len() != mask.Len() {
		panic(fmt.Sprintf("kernel: filter of a column of %d values by a mask of %d",
			c.Len(), mask.Len()))
	}
	return Take(c, Indices(mask))
}

// Indices returns the positions mask selects, in order.
//
// This is the half of a filter that does not depend on the column being
// filtered, and it is exported because a frame filters every one of its columns
// with the same mask and should only answer this question once.
func Indices(mask *array.Chunked) []int {
	checkMask(mask)

	// The count includes the positions under a null, which cannot be selected,
	// so it is an upper bound rather than the answer. It is here to size the
	// slice once, and a filter that keeps most of a column is the case worth
	// sizing for.
	out := make([]int, 0, countTrue(mask))

	base := 0
	for _, a := range mask.Chunks() {
		bits, off := a.Bools(), a.Offset()
		switch a.NullCount() {
		case 0:
			for i := range a.Len() {
				if bits.Get(off + i) {
					out = append(out, base+i)
				}
			}
		default:
			for i := range a.Len() {
				if a.IsValid(i) && bits.Get(off+i) {
					out = append(out, base+i)
				}
			}
		}
		base += a.Len()
	}
	return out
}

// countTrue returns how many bits of the mask are set, nulls included.
func countTrue(mask *array.Chunked) int {
	n := 0
	for _, a := range mask.Chunks() {
		n += a.Bools().CountOnesRange(a.Offset(), a.Offset()+a.Len())
	}
	return n
}

// checkMask panics unless mask is a boolean column.
func checkMask(mask *array.Chunked) {
	if mask == nil {
		panic("kernel: filter by a nil mask")
	}
	if mask.DType().Kind() != dtype.BoolKind {
		panic(fmt.Sprintf("kernel: filter by a %s column, which is not a mask", mask.DType()))
	}
}
