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
//
// Grouping one is the other half of this file. The key of a row is the index
// when the column allows it, which is a group by over an int32 no matter what
// the values are, and is the value behind the index when it does not. What
// decides is whether every chunk reads from the one dictionary and whether its
// values are distinct, since the same index is only the same value when both
// hold. Either way the values are read where they are and the column is never
// decoded into the million strings it stands for.
//
// Comparing and ordering one is the third thing in here. A comparison of a
// dictionary encoded column is the comparison of the values behind its indices,
// so the work is following the index, which is what dictIndex is for. The
// encoding is storage rather than meaning, so a column of country codes answers
// the same question the same way whether it was read out of a parquet file that
// wrote a dictionary or one that did not.

// dictIndex returns where in its dictionary each row of a points, or nil when a
// is not encoded at all. A row with no value gives minus one, whether it holds
// no index or points at a dictionary entry that is itself null.
//
// The index type is looked at once per chunk rather than once per row, which is
// the whole reason this exists. [array.Array.Index] switches on the type of the
// indices and rebuilds their slice header every time it is asked, which is
// nothing next to a lookup and is most of the work in a sort that asks twice per
// comparison and reads every row log n times.
func dictIndex(a *array.Array) func(int) int {
	d := a.Dictionary()
	if d == nil {
		return nil
	}

	idx := a.Indices()
	switch idx.DType().Kind() {
	case dtype.Int8Kind:
		return indexReader[int8](a, d, idx)
	case dtype.Int16Kind:
		return indexReader[int16](a, d, idx)
	case dtype.Int32Kind:
		return indexReader[int32](a, d, idx)
	case dtype.Int64Kind:
		return indexReader[int64](a, d, idx)
	case dtype.Uint8Kind:
		return indexReader[uint8](a, d, idx)
	case dtype.Uint16Kind:
		return indexReader[uint16](a, d, idx)
	case dtype.Uint32Kind:
		return indexReader[uint32](a, d, idx)
	case dtype.Uint64Kind:
		return indexReader[uint64](a, d, idx)
	default:
		panic(fmt.Sprintf("kernel: a dictionary indexed by %s", idx.DType()))
	}
}

// indexReader reads the indices of one chunk as T.
//
// A chunk with no missing rows reading from a dictionary with no missing values
// is the ordinary case and it gets a reader that is a slice index and nothing
// else. The other one has both questions to ask and asks them in the order that
// finds the answer soonest.
func indexReader[T array.Numeric](a, d, idx *array.Array) func(int) int {
	vs := idx.Values[T]()
	if a.NullCount() == 0 && d.NullCount() == 0 {
		return func(i int) int { return int(vs[i]) }
	}

	holes := d.NullCount() > 0
	return func(i int) int {
		if a.IsNull(i) {
			return -1
		}
		j := int(vs[i])
		if holes && d.IsNull(j) {
			return -1
		}
		return j
	}
}

// dictCursor is a cursor that reads through the encoding, so that a kernel over
// two columns sees values whether or not either side was encoded.
//
// The reader is bound when the chunk changes rather than per value, which for a
// column of one value being compared against every row of another means once.
type dictCursor struct {
	cur cursor

	// chunk is the chunk the reader below was bound to, so that a walk that
	// stays in one chunk binds once and a comparison against a literal binds
	// once for the whole column.
	chunk *array.Array

	// values is where the values of that chunk are and at is where in them a row
	// of it points, which is nil when the chunk is not encoded.
	values *array.Array
	at     func(int) int
}

// newDictCursor returns a cursor over c, which may or may not be encoded.
func newDictCursor(c *array.Chunked, fixed bool) dictCursor {
	return dictCursor{cur: newCursor(c, fixed)}
}

// next returns where the next value is and whether there is one there.
func (c *dictCursor) next() (*array.Array, int, bool) {
	a, i := c.cur.next()
	if a != c.chunk {
		c.chunk, c.values, c.at = a, a, dictIndex(a)
		if d := a.Dictionary(); d != nil {
			c.values = d
		}
	}
	if c.at == nil {
		return a, i, !a.IsNull(i)
	}

	i = c.at(i)
	return c.values, i, i >= 0
}

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

// bindDictionary writes the value behind an index, using the writer the value
// type would have used had the column not been encoded at all.
//
// Both halves are bound once per chunk, the values so that the string data is
// found once and the indices so that reading one is a slice index rather than a
// switch on the index type. A row whose value is missing never arrives here,
// since the key has already written it as one.
func bindDictionary(value func(*array.Array) writer) func(*array.Array) writer {
	return func(a *array.Array) writer {
		write, idx := value(a.Dictionary()), a.Indices()
		switch idx.DType().Kind() {
		case dtype.Int8Kind:
			return bindIndices[int8](write, idx)
		case dtype.Int16Kind:
			return bindIndices[int16](write, idx)
		case dtype.Int32Kind:
			return bindIndices[int32](write, idx)
		case dtype.Int64Kind:
			return bindIndices[int64](write, idx)
		case dtype.Uint8Kind:
			return bindIndices[uint8](write, idx)
		case dtype.Uint16Kind:
			return bindIndices[uint16](write, idx)
		case dtype.Uint32Kind:
			return bindIndices[uint32](write, idx)
		case dtype.Uint64Kind:
			return bindIndices[uint64](write, idx)
		default:
			panic(fmt.Sprintf("kernel: group by a dictionary indexed by %s", idx.DType()))
		}
	}
}

// indexKey returns a binder that writes the index of a row rather than the
// value behind it, and reports whether this column allows one.
//
// It is allowed when every chunk reads from the one dictionary and no two of
// its values are equal, because then two rows hold the same value exactly when
// they hold the same index, and the key of a column of country codes is an
// int32 rather than a string. That is the whole point of the encoding as far as
// a group by is concerned, and it is what makes grouping a dictionary column
// beat grouping the strings it stands for rather than merely tie with them.
//
// The dictionary is not allowed to hold a null, since a row pointing at one is
// a row with no value and has to group with the rows that have none, which an
// index cannot say. Finding out whether the values are distinct costs a pass
// over them, so the question is only asked when there are fewer of them than
// there are rows to group, which is the shape the encoding is used for.
func indexKey(c *array.Chunked, dt dtype.Dictionary) (func(*array.Array) writer, bool) {
	chunks := c.Chunks()
	if len(chunks) == 0 {
		return nil, false
	}
	d := chunks[0].Dictionary()
	if d.NullCount() > 0 || d.Len() > c.Len() {
		return nil, false
	}
	for _, a := range chunks[1:] {
		if a.Dictionary() != d {
			return nil, false
		}
	}
	if !distinct(d) {
		return nil, false
	}

	bind, err := binderFor(dt.Index)
	if err != nil {
		return nil, false
	}
	return func(a *array.Array) writer { return bind(a.Indices()) }, true
}

// distinct reports whether no two values of d are equal.
//
// The comparison is the encoding a group by would have used on the values
// anyway, so two values are equal here exactly when they would have landed in
// one group there. A dictionary written by anything that reads Parquet or Arrow
// has distinct values already, and this is here because nothing says it has to.
func distinct(d *array.Array) bool {
	bind, err := binderFor(d.DType())
	if err != nil {
		return false
	}

	write := bind(d)
	seen := make(map[string]struct{}, d.Len())
	var scratch []byte
	for i := range d.Len() {
		scratch = write(scratch[:0], i)
		if _, ok := seen[string(scratch)]; ok {
			return false
		}
		seen[string(scratch)] = struct{}{}
	}
	return true
}

// bindIndices reads the indices of one chunk as T and hands each of them to the
// writer over the values.
func bindIndices[T array.Numeric](write writer, idx *array.Array) writer {
	vs := idx.Values[T]()
	return func(dst []byte, i int) []byte {
		return write(dst, int(vs[i]))
	}
}

// missing reports whether row j of a has no value.
//
// For a dictionary encoded column that is two questions rather than one: the
// row itself can be missing, and the row can be there and point at a value in
// the dictionary that is missing. Both of them are a missing value, they have
// to group together and a count has to skip both, so every part of this package
// that asks the question asks it here.
func missing(a *array.Array, j int) bool {
	if a.IsNull(j) {
		return true
	}
	d := a.Dictionary()
	return d != nil && d.NullCount() > 0 && d.IsNull(a.Index(j))
}

// anyMissing reports whether a has a row with no value in it, which is the
// question a loop asks once so that it does not have to ask about every row.
func anyMissing(a *array.Array) bool {
	if a.NullCount() > 0 {
		return true
	}
	d := a.Dictionary()
	return d != nil && d.NullCount() > 0
}
