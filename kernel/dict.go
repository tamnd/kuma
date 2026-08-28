package kernel

import (
	"fmt"
	"math"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Gathering a dictionary encoded column.
//
// A gather over a dictionary encoded column is a gather over its indices. The
// values are not touched and not copied: taking a million rows out of a column
// of country codes writes a million int32s and leaves the two hundred and fifty
// strings where they are. That is the whole reason the encoding is worth having
// and the reason this is not one more case in gather, which would have written
// a million strings.
//
// The result points at the same values as the column it came from, so a filter
// over a dictionary column costs the indices it keeps and nothing else, and the
// two columns share the values the way two batches out of one file do.
//
// A chunked column may hold a different dictionary in every chunk, since two
// files are two dictionaries and putting the columns end to end does not merge
// them. When the chunks a gather reads do not agree, the result gets their
// dictionaries laid end to end and the indices of each chunk shifted to where
// its values landed. Nothing is deduplicated: two chunks holding the same two
// hundred and fifty strings produce five hundred, because finding out that they
// hold the same values means comparing all of them against all of them, and the
// case this path exists for is chunks that hold different ones.

// takeDictionary gathers a dictionary encoded column at the positions given.
//
// The values come first, since where a chunk's values land in them is what its
// indices have to be shifted by, and in the ordinary case where every chunk
// reads from the one dictionary the shift is zero and the values are the ones
// that were already there.
func takeDictionary(c *array.Chunked, dt dtype.Dictionary, idx []int) *array.Chunked {
	values, bases := dictValues(c, dt)

	b, err := array.NewBuilder(dt.Index)
	if err != nil {
		panic("kernel: " + err.Error())
	}
	b.Grow(len(idx))

	f := newFinder(c)
	switch dt.Index.Kind() {
	case dtype.Int8Kind:
		takeIndices[int8](b, &f, bases, idx)
	case dtype.Int16Kind:
		takeIndices[int16](b, &f, bases, idx)
	case dtype.Int32Kind:
		takeIndices[int32](b, &f, bases, idx)
	case dtype.Int64Kind:
		takeIndices[int64](b, &f, bases, idx)
	case dtype.Uint8Kind:
		takeIndices[uint8](b, &f, bases, idx)
	case dtype.Uint16Kind:
		takeIndices[uint16](b, &f, bases, idx)
	case dtype.Uint32Kind:
		takeIndices[uint32](b, &f, bases, idx)
	case dtype.Uint64Kind:
		takeIndices[uint64](b, &f, bases, idx)
	default:
		panic(fmt.Sprintf("kernel: take from a dictionary indexed by %s", dt.Index))
	}

	out, err := array.NewDictionary(b.Finish(), values)
	if err != nil {
		// The indices were read out of a column whose indices had already been
		// checked, and the shift puts each of them where its own values went,
		// so this is a mistake in this file rather than in the data.
		panic("kernel: " + err.Error())
	}
	col, err := array.NewChunked(dt, out)
	if err != nil {
		panic("kernel: " + err.Error())
	}
	return col
}

// takeIndices gathers the indices of a dictionary encoded column, shifting each
// chunk's by where that chunk's values landed.
//
// The typed slice and the shift are read once per chunk rather than once per
// value, the same way the plain gather reads its values, because a gather that
// runs forwards changes chunk a handful of times and reads a million indices.
func takeIndices[T array.Numeric](b *array.Builder, f *finder, bases map[*array.Array]int, idx []int) {
	var cur *array.Array
	var vals []T
	var base T

	for _, i := range idx {
		a, k := f.at(i)
		if a == nil {
			b.AppendNull()
			continue
		}
		if a != cur {
			cur, vals, base = a, a.Indices().Values[T](), T(bases[a.Dictionary()])
		}
		if a.IsNull(k) {
			b.AppendNull()
			continue
		}
		b.Append(vals[k] + base)
	}
}

// dictValues returns the values the result of a gather over c points into, and
// where each dictionary c reads from starts in them.
//
// The map is nil when every chunk reads from the one dictionary, since a lookup
// in a nil map is zero and zero is the shift. That is the case worth being fast
// in, and it is the case a column read out of one file is always in.
func dictValues(c *array.Chunked, dt dtype.Dictionary) (*array.Array, map[*array.Array]int) {
	chunks := c.Chunks()
	if len(chunks) == 0 {
		return emptyOf(dt.Value), nil
	}
	shared := true
	for _, a := range chunks[1:] {
		if a.Dictionary() != chunks[0].Dictionary() {
			shared = false
			break
		}
	}
	if shared {
		return chunks[0].Dictionary(), nil
	}

	// The dictionaries are compared by identity rather than by their values,
	// which is what makes two chunks out of one file count once and two chunks
	// holding equal values count twice. The first is the case this is here to
	// keep cheap and the second is a copy of some strings.
	bases := make(map[*array.Array]int, len(chunks))
	parts := make([]*array.Array, 0, len(chunks))
	n := 0
	for _, a := range chunks {
		d := a.Dictionary()
		if _, ok := bases[d]; ok {
			continue
		}
		bases[d] = n
		n += d.Len()
		parts = append(parts, d)
	}
	if limit := dictLimit(dt.Index); n > limit {
		panic(fmt.Sprintf("kernel: take from a dictionary column whose chunks hold %d values between them, more than the %d an index of %s can name", n, limit, dt.Index))
	}

	all, err := array.NewChunked(dt.Value, parts...)
	if err != nil {
		panic("kernel: " + err.Error())
	}
	whole := make([]int, n)
	for i := range whole {
		whole[i] = i
	}
	b, err := array.NewBuilder(dt.Value)
	if err != nil {
		panic("kernel: " + err.Error())
	}
	b.Grow(n)
	gather(b, all, whole)
	return b.Finish(), bases
}

// dictLimit is how many values a dictionary indexed by t can hold.
//
// An index is a position rather than a number, so what matters is the largest
// one that fits and not whether the type is signed. The wide ones are capped at
// the largest slice this machine could address, since a dictionary is an array
// and an array cannot be longer than that whatever its indices could name.
func dictLimit(t dtype.DataType) int {
	switch t.Kind() {
	case dtype.Int8Kind:
		return math.MaxInt8 + 1
	case dtype.Int16Kind:
		return math.MaxInt16 + 1
	case dtype.Int32Kind:
		return min(math.MaxInt32+1, math.MaxInt)
	case dtype.Uint8Kind:
		return math.MaxUint8 + 1
	case dtype.Uint16Kind:
		return math.MaxUint16 + 1
	case dtype.Uint32Kind:
		return min(math.MaxUint32+1, math.MaxInt)
	case dtype.Int64Kind, dtype.Uint64Kind:
		return math.MaxInt
	default:
		panic(fmt.Sprintf("kernel: a dictionary indexed by %s", t))
	}
}

// emptyOf returns an array of no values of the given type, which is what the
// values of a gather over a column with no chunks in it point at.
func emptyOf(dt dtype.DataType) *array.Array {
	b, err := array.NewBuilder(dt)
	if err != nil {
		panic("kernel: " + err.Error())
	}
	return b.Finish()
}
