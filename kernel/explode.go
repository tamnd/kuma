package kernel

import (
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Explode turns every element of a list column into a value of its own.
//
// It returns the flattened column, whose type is the element type of the one it
// was given, and the row of the input each of its values came from. The second
// of those is what the columns beside this one are gathered by, so a row that
// held three elements becomes three rows and everything else in it is repeated
// three times. It is what pandas calls explode and what SQL writes as an
// unnest.
//
// A row that is missing, and a row that is present and holds nothing, both
// become one row holding a missing value. They are two different things and this
// is the one place they are treated the same, because the alternative is a row
// that disappears, and a frame where exploding a column can lose a row is a
// frame where the count of the rows depends on which column was taken apart,
// which is what pandas decided as well. Dropping those rows afterwards is a
// filter on the column, which says so at the line that does it.
//
// The elements go through [Take] rather than being copied out a row at a time,
// so a list of lists comes back as a column of lists and a list of dictionary
// encoded strings stays dictionary encoded.
//
// It panics if c is nil or is not a list column, the same way indexing a slice
// panics, since a column that holds one value per row has nothing to take apart
// and asking it to is a bug rather than something the data did.
func Explode(c *array.Chunked) (*array.Chunked, []int) {
	if c == nil {
		panic("kernel: explode a nil column")
	}
	dt, ok := c.DType().(dtype.List)
	if !ok {
		panic(fmt.Sprintf("kernel: explode a %s column", c.DType()))
	}

	elems, base, held := listElems(c, dt)

	// The result is one value per element, plus one for each row that has no
	// elements to give, so what the rows hold is the guess and the rows with
	// nothing in them are what it is short by.
	take := make([]int, 0, held)
	rows := make([]int, 0, held)

	row := 0
	for k, a := range c.Chunks() {
		for i := range a.Len() {
			lo, hi := 0, 0
			if !a.IsNull(i) {
				lo, hi = a.ListRange(i)
			}

			if lo == hi {
				// A missing row and an empty one, which are the two rows with
				// no element to give. The position below zero is what [Take]
				// turns into the missing value.
				take = append(take, -1)
				rows = append(rows, row)
			}
			for e := lo; e < hi; e++ {
				take = append(take, base[k]+e)
				rows = append(rows, row)
			}
			row++
		}
	}
	return Take(elems, take), rows
}
