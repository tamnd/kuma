package kernel

import (
	"fmt"
	"math"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/dtype"
)

// takeList gathers rows out of a list column.
//
// A list column is offsets into a child, and the two halves are gathered
// differently. The offsets cannot survive being reordered, since they say where
// a row is and the rows are moving, so they are worked out again. The elements
// can: every element of every kept row is a position in a child, so the whole
// thing comes down to one ordinary gather over the elements. That is the point
// of doing it this way rather than a row at a time, which would mean a builder
// call per element with the type switched on every one of them.
//
// The children of the chunks are laid end to end as one column first, so that an
// element has one position however many chunks the column arrived in. Laying
// them out copies nothing, being a list of arrays.
//
// The elements go through Take rather than through the switch inside it, so a
// list of lists works and so does a list of dictionary encoded strings. Each of
// those does whatever it does one level down and this does not have to know.
func takeList(c *array.Chunked, dt dtype.List, idx []int) *array.Chunked {
	elems, base, held := listElems(c, dt)

	off, take, valid := listRows(c, elemHint(held, c.Len(), len(idx)), base, idx)
	out, err := array.NewListFrom(dt, off, gatherElems(elems, dt.Elem, take), valid)
	if err != nil {
		panic("kernel: " + err.Error())
	}

	col, err := array.NewChunked(dt, out)
	if err != nil {
		panic("kernel: " + err.Error())
	}
	return col
}

// listElems lays the children of the chunks of a list column end to end as one
// column, so that an element has one position however many chunks the column
// arrived in.
//
// It returns that column, the position each chunk's child begins at, and how
// many elements the rows of the column hold between them. The last of those is
// what the rows hold rather than what the children are long, which are different
// numbers for a column sliced out of a longer one.
//
// Laying the children out copies nothing, being a list of arrays.
func listElems(c *array.Chunked, dt dtype.List) (elems *array.Chunked, base []int, held int) {
	chunks := c.Chunks()

	kids := make([]*array.Array, len(chunks))
	base = make([]int, len(chunks))
	n := 0
	for k, a := range chunks {
		kids[k] = a.Child()
		base[k] = n
		n += a.Child().Len()
		held += chunkElems(a)
	}

	elems, err := array.NewChunked(dt.Elem, kids...)
	if err != nil {
		// Every chunk of a list column of this type has a child of the element
		// type, which is what NewList makes true.
		panic("kernel: " + err.Error())
	}
	return elems, base, held
}

// gatherElems gathers the elements at the given positions into the one array a
// list column keeps them in.
//
// The gather hands back a chunked column, and a chunked column leaves out a
// chunk that holds nothing, so a gather that kept no elements comes back with no
// chunks at all. That is the one case worth writing down: a list column still
// needs a child for its offsets to point at when every row it kept is empty or
// missing.
func gatherElems(elems *array.Chunked, elem dtype.DataType, take []int) *array.Array {
	if chunks := Take(elems, take).Chunks(); len(chunks) > 0 {
		return chunks[0]
	}

	out, err := array.Empty(elem)
	if err != nil {
		panic("kernel: " + err.Error())
	}
	return out
}

// chunkElems returns how many elements the rows of one chunk hold, which is not
// the length of its child. A chunk that was sliced out of a longer one keeps the
// whole child and points at a range of it, so the child is what the column it
// came from held and this is what this chunk holds.
func chunkElems(a *array.Array) int {
	if a.Len() == 0 {
		return 0
	}
	start, _ := a.ListRange(0)
	_, end := a.ListRange(a.Len() - 1)
	return end - start
}

// elemHint guesses how many elements are kept by a gather that asks for rows
// many of them out of a column whose length rows hold n elements between them.
//
// The count is not known until the rows have been walked, and growing the list
// of positions as they are walked is where a gather of lists spends what a
// gather of numbers does not: doubling from nothing to a million positions
// allocates two million of them. A column whose rows are all the same length
// makes the guess exact, and one whose rows vary makes it close, which is enough
// to turn a run of allocations into one. The rounding up is so that a column
// averaging under one element per row still gets a position per row.
//
// The multiplication is checked rather than trusted, since both sides of it come
// from the caller and a capacity that wrapped is worse than a capacity that was
// never guessed at.
func elemHint(n, length, rows int) int {
	if n <= 0 || length <= 0 {
		return rows
	}
	if avg := n/length + 1; rows <= math.MaxInt/avg {
		return rows * avg
	}
	return rows
}

// listRows works out what the gathered column looks like: where each of its rows
// begins, which elements it holds and which of its rows are missing. The valid
// it returns is nil when nothing was missing.
//
// It is one pass over the positions, and all it reads per row is the two
// offsets, which is why the row wise half of a list gather costs about what
// gathering a column of numbers costs.
func listRows(c *array.Chunked, hint int, base, idx []int) (off []int32, take []int, valid *bitmap.Bitmap) {
	off = make([]int32, len(idx)+1)
	take = make([]int, 0, hint)
	f := newFinder(c)

	for m, i := range idx {
		a, pos, chunk := f.atChunk(i)
		if a == nil || a.IsNull(pos) {
			// A position below zero is a null the caller asked for and a null
			// row is one the column already had. Either way the row is missing
			// and holds nothing, which is the same offset twice.
			if valid == nil {
				valid = bitmap.NewSet(len(idx))
			}
			valid.Set(m, false)
			off[m+1] = off[m]
			continue
		}

		start, end := a.ListRange(pos)
		for e := start; e < end; e++ {
			take = append(take, base[chunk]+e)
		}
		if len(take) > math.MaxInt32 {
			panic(fmt.Sprintf("kernel: gathering %d elements into a list column, which holds at most %d",
				len(take), math.MaxInt32))
		}
		off[m+1] = int32(len(take))
	}
	return off, take, valid
}
