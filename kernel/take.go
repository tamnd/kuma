package kernel

import (
	"fmt"
	"sort"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Take returns a column holding the values of c at the given positions, in the
// order given.
//
// A position below zero produces a null. That is not a courtesy to sloppy
// callers, it is what a left join does with a row that matched nothing, and
// having one rule for it here means the join does not need a second pass to put
// the nulls in. A position at or past the length of the column panics, the same
// way indexing a slice does.
//
// The result is a new column that shares nothing with c, since the values it
// wants are scattered through the old one and there is no way to point at them.
// That makes this the expensive operation it looks like: gathering a million
// rows out of a column writes a million values.
//
// A dictionary encoded column is the exception and the reason the encoding is
// worth having. Only the indices are gathered and the result points at the same
// values as c, so taking a million rows out of a column of country codes writes
// a million int32s rather than a million strings.
//
// A list column is the other exception, and it is halfway between the two. The
// offsets are rebuilt, since the rows are in a new order and were never going to
// survive that, but the elements are gathered rather than copied a row at a
// time, so a column of lists costs what a column of its elements costs.
//
// It panics if c is nil or is a type this package cannot read yet, which today
// means the struct and map types.
func Take(c *array.Chunked, idx []int) *array.Chunked {
	if c == nil {
		panic("kernel: take from a nil column")
	}
	switch dt := c.DType().(type) {
	case dtype.Dictionary:
		return takeDictionary(c, dt, idx)
	case dtype.List:
		return takeList(c, dt, idx)
	}

	b, err := array.NewBuilder(c.DType())
	if err != nil {
		panic("kernel: " + err.Error())
	}
	b.Grow(len(idx))
	gather(b, c, idx)

	out, err := array.NewChunked(c.DType(), b.Finish())
	if err != nil {
		// The builder was made for this dtype, so what it built is of this
		// dtype.
		panic("kernel: " + err.Error())
	}
	return out
}

// gather appends the values of c at the given positions to b.
//
// The switch is on the kind rather than on the dtype, and it is done once for
// the whole gather rather than once per value, which is the only thing in here
// that counts as a decision. The layouts that share a case share it because
// they are the same bytes: a timestamp column is int64 and gathering one is
// gathering int64s.
func gather(b *array.Builder, c *array.Chunked, idx []int) {
	f := newFinder(c)

	switch c.DType().Kind() {
	case dtype.NullKind:
		for _, i := range idx {
			// Nothing to read, but the position is still checked, since a
			// gather that runs off the end of a null column is as wrong as one
			// that runs off the end of any other.
			f.at(i)
			b.AppendNull()
		}
	case dtype.BoolKind:
		gatherBools(b, &f, idx)
	case dtype.Int8Kind:
		gatherValues[int8](b, &f, idx)
	case dtype.Int16Kind:
		gatherValues[int16](b, &f, idx)
	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		gatherValues[int32](b, &f, idx)
	case dtype.Int64Kind, dtype.Date64Kind, dtype.Time64Kind,
		dtype.TimestampKind, dtype.DurationKind:
		gatherValues[int64](b, &f, idx)
	case dtype.Uint8Kind:
		gatherValues[uint8](b, &f, idx)
	case dtype.Uint16Kind:
		gatherValues[uint16](b, &f, idx)
	case dtype.Uint32Kind:
		gatherValues[uint32](b, &f, idx)
	case dtype.Uint64Kind:
		gatherValues[uint64](b, &f, idx)
	case dtype.Float32Kind:
		gatherValues[float32](b, &f, idx)
	case dtype.Float64Kind:
		gatherValues[float64](b, &f, idx)
	case dtype.StringKind, dtype.BinaryKind, dtype.FixedSizeBinaryKind,
		dtype.Decimal128Kind, dtype.Decimal256Kind, dtype.IntervalKind:
		gatherBytes(b, &f, idx)
	default:
		panic(fmt.Sprintf("kernel: take from a %s column", c.DType()))
	}
}

// gatherValues is the fixed width case, where the values are numbers or are
// stored as numbers.
//
// The typed slice is taken once per chunk rather than once per value, which
// matters because taking it checks the layout of the column and builds a slice
// header, and a gather over a column in one chunk would otherwise pay for that
// a million times to read a million values out of the same place.
func gatherValues[T array.Numeric](b *array.Builder, f *finder, idx []int) {
	var cur *array.Array
	var vals []T

	for _, i := range idx {
		a, k := f.at(i)
		if a == nil {
			b.AppendNull()
			continue
		}
		if a != cur {
			cur, vals = a, a.Values[T]()
		}
		if a.IsNull(k) {
			b.AppendNull()
			continue
		}
		b.Append(vals[k])
	}
}

// gatherBools is the boolean case, whose values are bits rather than bytes and
// so are not reachable through a typed slice.
func gatherBools(b *array.Builder, f *finder, idx []int) {
	for _, i := range idx {
		a, k := f.at(i)
		if a == nil || a.IsNull(k) {
			b.AppendNull()
			continue
		}
		b.AppendBool(a.Bool(k))
	}
}

// gatherBytes is the case where a value is a run of bytes: the strings, and the
// fixed width types that are not numbers Go has, such as the decimals.
func gatherBytes(b *array.Builder, f *finder, idx []int) {
	for _, i := range idx {
		a, k := f.at(i)
		if a == nil || a.IsNull(k) {
			b.AppendNull()
			continue
		}
		b.AppendBytes(a.Bytes(k))
	}
}

// finder turns a position in a column into a position in one of its chunks.
//
// It remembers the chunk the last position landed in, because the positions a
// gather is given are usually sorted, or nearly so: a filter produces them in
// order and a sort produces long runs of them. When the guess is wrong it
// searches, so the worst case is a binary search per position and the common
// case is a comparison.
type finder struct {
	chunks []*array.Array

	// starts[k] is the position in the column that chunk k begins at. The
	// column knows this and keeps it private, and working it out again here
	// costs one pass over a handful of chunks.
	starts []int

	length int
	last   int
}

// newFinder returns a finder over the chunks of c.
func newFinder(c *array.Chunked) finder {
	chunks := c.Chunks()
	f := finder{chunks: chunks, starts: make([]int, len(chunks)), length: c.Len()}
	n := 0
	for k, a := range chunks {
		f.starts[k] = n
		n += a.Len()
	}
	return f
}

// at returns the chunk holding position i and the position within it. A
// position below zero gives a nil chunk, which is the caller's signal to
// produce a null. It panics if i is past the end of the column.
func (f *finder) at(i int) (*array.Array, int) {
	if i < 0 {
		return nil, 0
	}
	if uint(i) >= uint(f.length) {
		panic(fmt.Sprintf("kernel: take position %d out of range, the column has %d values", i, f.length))
	}

	// The chunk the last position landed in, and then the one after it, which
	// between them are every position of a gather that runs forwards.
	if k := f.last; f.holds(k, i) {
		return f.chunks[k], i - f.starts[k]
	}
	if k := f.last + 1; k < len(f.chunks) && f.holds(k, i) {
		f.last = k
		return f.chunks[k], i - f.starts[k]
	}

	f.last = sort.Search(len(f.chunks), func(k int) bool {
		return f.starts[k]+f.chunks[k].Len() > i
	})
	return f.chunks[f.last], i - f.starts[f.last]
}

// atChunk is at with the number of the chunk as well.
//
// The list gather needs it because a chunked list column has one child per
// chunk, so an element is a position in a particular one of them and finding
// the row is only half of finding the value.
func (f *finder) atChunk(i int) (a *array.Array, pos, chunk int) {
	a, pos = f.at(i)
	return a, pos, f.last
}

// holds reports whether chunk k is where position i lives.
func (f *finder) holds(k, i int) bool {
	return i >= f.starts[k] && i < f.starts[k]+f.chunks[k].Len()
}
