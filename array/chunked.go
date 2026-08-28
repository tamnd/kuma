package array

import (
	"fmt"
	"slices"

	"github.com/tamnd/kuma/dtype"
)

// Chunked is a column held as a sequence of arrays rather than as one.
//
// This is what Arrow calls a ChunkedArray. The name is shorter here because the
// package is already called array, and array.Chunked is what the thing is.
//
// A column is chunked for two reasons. A file arrives in record batches and
// joining them into one array would mean copying every value to gain nothing.
// And a column longer than one allocation can hold has to be more than one
// allocation, which for a string column is any column with more than two
// gigabytes of text in it.
//
// The chunks may be of any length and need not be of the same length. What they
// must be is the same type, since a column is one type by definition.
//
// A Chunked is immutable, the same way an Array is. Append returns a new column
// and Slice returns a new column, and neither touches the one it was called on,
// which is what lets the executor hand the same column to several goroutines.
//
// The zero Chunked is not usable. Use NewChunked.
type Chunked struct {
	dt     dtype.DataType
	chunks []*Array
	length int
	nulls  int

	// starts holds the index of the first value of each chunk, followed by the
	// total length. Looking up value i is a binary search here, which is why
	// it is worth keeping rather than adding up chunk lengths every time.
	starts []int
}

// NewChunked returns a column of type dt holding the given chunks in order.
//
// Every chunk has to be of type dt, compared with dtype.Equal rather than by
// Kind, since a timestamp in microseconds and a timestamp in nanoseconds are
// the same Kind and are not the same column.
//
// A chunk with no values is dropped. It holds nothing, and keeping it would
// mean every lookup had to step over a chunk that can never be the answer.
//
// The chunks are not copied. The column shares them, and since an Array is
// immutable there is nothing to share badly.
func NewChunked(dt dtype.DataType, chunks ...*Array) (*Chunked, error) {
	if dt == nil {
		return nil, errNilDType
	}

	c := &Chunked{dt: dt, starts: []int{0}}
	for i, a := range chunks {
		if a == nil {
			return nil, fmt.Errorf("array: chunk %d is nil", i)
		}
		if !dtype.Equal(a.DType(), dt) {
			return nil, fmt.Errorf("array: chunk %d is a %s column, want %s", i, a.DType(), dt)
		}
		if a.Len() == 0 {
			continue
		}

		c.chunks = append(c.chunks, a)
		c.length += a.Len()
		c.nulls += a.NullCount()
		c.starts = append(c.starts, c.length)
	}
	return c, nil
}

// DType returns the type of the column.
func (c *Chunked) DType() dtype.DataType { return c.dt }

// Len returns how many values the column holds, across all of its chunks.
func (c *Chunked) Len() int { return c.length }

// NullCount returns how many of them are missing.
func (c *Chunked) NullCount() int { return c.nulls }

// NumChunks returns how many chunks the column is held in. It is not the same
// as the number of chunks NewChunked was handed, since the empty ones are
// dropped.
func (c *Chunked) NumChunks() int { return len(c.chunks) }

// Chunk returns chunk i. It panics if i is out of range.
func (c *Chunked) Chunk(i int) *Array {
	if uint(i) >= uint(len(c.chunks)) {
		panic("array: chunk index out of range")
	}
	return c.chunks[i]
}

// Chunks returns the chunks in order.
//
// The result shares the column's own slice and the caller must not modify it.
// This is the loop a kernel runs: read each chunk as a slice of values through
// Values and do the work there, rather than asking this column for one value at
// a time.
func (c *Chunked) Chunks() []*Array { return c.chunks }

// String returns a short description of the column, for a log line or a test
// failure. It is not the values.
func (c *Chunked) String() string {
	return fmt.Sprintf("array.Chunked{%s, len %d, nulls %d, chunks %d}",
		c.dt, c.length, c.nulls, len(c.chunks))
}

// Append returns a column with the given chunks added to the end. The column it
// is called on is unchanged.
//
// This is how a reader builds a column: finish an array for each batch it
// reads, and append it. The chunks already in the column are shared with the
// new one rather than copied.
func (c *Chunked) Append(chunks ...*Array) (*Chunked, error) {
	all := make([]*Array, 0, len(c.chunks)+len(chunks))
	all = append(all, c.chunks...)
	all = append(all, chunks...)
	return NewChunked(c.dt, all...)
}

// Slice returns the values from i up to but not including j, as a column. It
// panics unless 0 <= i <= j <= Len.
//
// The chunks the range covers whole are shared as they are, and the one or two
// at the ends are sliced, which is constant time each. So the cost is a binary
// search and a null count over the two partial chunks, whatever the length of
// the range in between.
func (c *Chunked) Slice(i, j int) *Chunked {
	if i < 0 || j < i || j > c.length {
		panic(fmt.Sprintf("array: Slice(%d, %d) of a column of %d values", i, j, c.length))
	}

	out := &Chunked{dt: c.dt, starts: []int{0}}
	if i == j {
		return out
	}

	first, from := c.locate(i)
	last, to := c.locate(j - 1)
	for k := first; k <= last; k++ {
		a := c.chunks[k]
		lo, hi := 0, a.Len()
		if k == first {
			lo = from
		}
		if k == last {
			hi = to + 1
		}
		if lo != 0 || hi != a.Len() {
			// Only a partial chunk is worth slicing. A whole one is already the
			// array it would return, and slicing it would recount its nulls for
			// an answer it is holding.
			a = a.Slice(lo, hi)
		}

		out.chunks = append(out.chunks, a)
		out.length += a.Len()
		out.nulls += a.NullCount()
		out.starts = append(out.starts, out.length)
	}
	return out
}

// IsValid reports whether value i is present. It panics if i is out of range.
func (c *Chunked) IsValid(i int) bool {
	k, n := c.locate(i)
	return c.chunks[k].IsValid(n)
}

// IsNull reports whether value i is missing. It panics if i is out of range.
func (c *Chunked) IsNull(i int) bool { return !c.IsValid(i) }

// Value returns value i. It panics if T is not the type the column stores or if
// i is out of range.
//
// It is for asking about one value, and it costs a binary search over the
// chunks to find which one holds it. A kernel reads Chunks instead and works on
// each of them as a slice.
func (c *Chunked) Value[T Numeric](i int) T {
	k, n := c.locate(i)
	return c.chunks[k].Value[T](n)
}

// Bool returns value i of a Bool column. It panics if the column is not Bool or
// if i is out of range.
func (c *Chunked) Bool(i int) bool {
	k, n := c.locate(i)
	return c.chunks[k].Bool(n)
}

// Bytes returns value i of a column whose values are bytes rather than numbers.
// It panics if the column is something else or if i is out of range.
//
// The result aliases the chunk it came from and the caller must not modify it.
func (c *Chunked) Bytes(i int) []byte {
	k, n := c.locate(i)
	return c.chunks[k].Bytes(n)
}

// At returns the chunk holding value i and where in that chunk it is. It panics
// if i is out of range.
//
// It is for a caller that wants something an Array can answer and a Chunked
// cannot, such as the dictionary a dictionary encoded column points at, which
// each chunk carries its own of.
func (c *Chunked) At(i int) (chunk *Array, index int) {
	k, n := c.locate(i)
	return c.chunks[k], n
}

// locate returns which chunk holds value i and where in that chunk it is. It
// panics if i is out of range.
func (c *Chunked) locate(i int) (chunk, index int) {
	if uint(i) >= uint(c.length) {
		panic("array: index out of range")
	}

	// starts ends with the total length, and i is less than that, so the search
	// never lands on the last entry and k-1 is always a real chunk.
	k, found := slices.BinarySearch(c.starts, i)
	if !found {
		k--
	}
	return k, i - c.starts[k]
}
