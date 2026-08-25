package kernel

import (
	"fmt"

	"github.com/tamnd/kuma/array"
)

// A binary kernel is one that reads two columns and writes a third, which is
// every comparison, every arithmetic operator and the logical connectives. What
// they share is in this file: how the two sides line up, and how the values are
// reached without a search per row.

// binaryLen returns how many rows a kernel over a and b produces, and whether
// each side is a single value being used for every row.
//
// A column of one value against a column of many is that value against each of
// them, which is what makes Price.Gt(100) work: the literal is a column of one.
// Two columns of the same length go value by value. Anything else is a mistake
// in the program rather than in the data, since the columns of a frame are all
// the same length by construction, so it panics.
//
// One row against one row is value by value rather than a broadcast, which is
// the same answer either way and saves deciding which side is the literal.
func binaryLen(name string, a, b *array.Chunked) (n int, fixedA, fixedB bool) {
	switch {
	case a.Len() == b.Len():
		return a.Len(), false, false
	case a.Len() == 1:
		return b.Len(), true, false
	case b.Len() == 1:
		return a.Len(), false, true
	default:
		panic(fmt.Sprintf("kernel: %s of a column of %d values and a column of %d",
			name, a.Len(), b.Len()))
	}
}

// cursor walks the values of a column one at a time.
//
// A kernel over two columns cannot take a typed slice per chunk the way a sort
// does, because the two sides are chunked however they were built and the
// boundaries need not line up. What it can do is remember where it got to,
// which is what this is: the cost per value is an increment and a comparison
// rather than the binary search that reading through the column would cost.
type cursor struct {
	chunks []*array.Array

	// c is the chunk the next value is in and i is where in that chunk it is.
	c, i int

	// fixed is a column of one value used for every row, which is what a
	// literal is. The chunk and the index then never move.
	fixed bool
}

// newCursor returns a cursor over the values of c. When fixed is true the
// column holds one value and the cursor hands it out for every row.
func newCursor(c *array.Chunked, fixed bool) cursor {
	cur := cursor{chunks: c.Chunks(), fixed: fixed}
	if fixed {
		// The one value need not be in the first chunk, since a column of one
		// value can have been built by appending an empty chunk to it, so the
		// position is found once here rather than assumed.
		cur.fixed = false
		cur.skip()
		cur.fixed = true
	}
	return cur
}

// next returns the chunk holding the next value and where in that chunk it is.
// It must not be called more times than the column has rows.
func (c *cursor) next() (*array.Array, int) {
	if c.fixed {
		return c.chunks[c.c], c.i
	}
	c.skip()
	a, i := c.chunks[c.c], c.i
	c.i++
	return a, i
}

// skip moves past any empty chunks, so that the position is a real value.
func (c *cursor) skip() {
	for c.chunks[c.c].Len() == c.i {
		c.c++
		c.i = 0
	}
}
