package parquet

import (
	"bytes"
	"encoding/binary"
	"math"

	"github.com/tamnd/kuma/array"
)

// Writing what a chunk holds.
//
// This is stats.go the other way round. A reader skips a row group by asking
// what its columns hold and finding the answer cannot match the filter, which
// costs a look at the footer and saves reading the group, and every one of those
// answers came from a writer that wrote them down. So a file written without
// them is a file that has to be read whole however small the question, and all
// the skipping next door does nothing on it.
//
// What goes down is the smallest and largest value of each column chunk and how
// many of its values are missing. The bounds are the pair the format defined an
// order for, and the file says which order that is, because the older pair meant
// whatever the writer of the day thought and a reader that acts on the wrong
// comparison skips a group holding rows somebody wanted. The bounds are exact
// values out of the chunk rather than truncations of them, since truncating is
// for keeping a footer small on columns of long strings and this writer has no
// long strings to keep it small for yet.
//
// The comparison is Go's on the type kuma stores, which is the one the format
// defines for the type the file says it is. That is the whole reason the
// unsigned integers are annotated in the schema: a uint64 travels in an int64
// and compares as a uint64, and a reader that took the signed order would put
// every large value first.
//
// A NaN is left out. It compares false against everything, itself included, so a
// chunk whose largest value was one would be a chunk that nothing could be
// filtered out of, and the format tells a writer to leave it out for that
// reason.
//
// The values go past as the pages are written rather than in a pass of their
// own. A pass of its own would be the tidier thing and it is why the first cut
// of this was twice as slow: a page writer has already walked the nulls and
// already pulled every value out of the column, and on a column of strings
// pulling them out a second time costs about what writing them costs. So what is
// here is state and the two lines that fold a value into it, and the walking
// stays where the walking was.

// tracker is what a page writer hands over when a column chunk is finished: the
// smallest and largest value that went past, written the way a statistic writes
// one, and whether there was anything at all.
//
// The bounds of a chunk are the chunk's own, so taking them is what starts the
// next one. That way there is nothing to reset and nothing to forget to reset,
// which on a file of a thousand row groups would be a footer of bounds that grew
// wider and wider and belonged to no chunk at all. A column of nothing has no
// tracker, having no values to be the smallest and largest of.
type tracker func() (lo, hi []byte, ok bool)

// bounds is the smallest and largest value of a chunk of numbers.
type bounds[T array.Numeric] struct {
	lo, hi T
	seen   bool
}

// add folds one value in.
func (b *bounds[T]) add(v T) {
	// The NaN test is free on the integers, none of which can be one, and the
	// conversion is only ever asked whether the value is a number rather than
	// what number it is.
	if math.IsNaN(float64(v)) {
		return
	}

	switch {
	case !b.seen:
		b.lo, b.hi, b.seen = v, v, true
	case v < b.lo:
		b.lo = v
	case v > b.hi:
		b.hi = v
	}
}

// fold folds in a run of values that are all there, which is a page of a column
// with nothing missing and the gathered values of one without.
func (b *bounds[T]) fold(vals []T) {
	for _, v := range vals {
		b.add(v)
	}
}

// taker is the tracker for a column whose bounds are written by put.
func (b *bounds[T]) taker(put func(T) []byte) tracker {
	return func() ([]byte, []byte, bool) {
		if !b.seen {
			return nil, nil, false
		}
		b.seen = false
		return put(b.lo), put(b.hi), true
	}
}

// boolBounds is the same for a chunk of booleans, where false is the smaller of the
// two and there are only the two.
type boolBounds struct {
	lo, hi, seen bool
}

// add folds one value in.
func (f *boolBounds) add(v bool) {
	switch {
	case !f.seen:
		f.lo, f.hi, f.seen = v, v, true
	case v:
		f.hi = true
	default:
		f.lo = false
	}
}

// taker is the tracker for a column of booleans.
func (f *boolBounds) taker() tracker {
	return func() ([]byte, []byte, bool) {
		if !f.seen {
			return nil, nil, false
		}
		f.seen = false
		return boolBound(f.lo), boolBound(f.hi), true
	}
}

// blobBounds is the same for a chunk whose values are bytes, which compare as
// unsigned bytes from the left, the shorter first when one is the front of the
// other.
//
// That is the order the format defines and it is the reason the format had to
// define one, since the writers that came before it compared those same bytes as
// signed numbers and put every value with a high bit set before every value
// without one.
type blobBounds struct {
	lo, hi []byte
	seen   bool
}

// add folds one value in. What is kept is the value where it sits in the column
// rather than a copy of it, the column outliving the chunk being written out of
// it by some way.
func (b *blobBounds) add(v []byte) {
	switch {
	case !b.seen:
		b.lo, b.hi, b.seen = v, v, true
	case bytes.Compare(v, b.lo) < 0:
		b.lo = v
	case bytes.Compare(v, b.hi) > 0:
		b.hi = v
	}
}

// taker is the tracker for a column of bytes, whose bounds are the bytes.
func (b *blobBounds) taker() tracker {
	return func() ([]byte, []byte, bool) {
		if !b.seen {
			return nil, nil, false
		}
		b.seen = false
		return b.lo, b.hi, true
	}
}

// A bound is one value written the way a page writes it, except that a byte
// array is written as itself rather than behind four bytes of its length,
// because a statistic is one value and there is nothing behind it to find.

// int32Bound writes a bound of every integer parquet keeps in four bytes, which
// is every one of kuma's up to that width and the unsigned ones that fit in it
// once the annotation on the column is undone.
func int32Bound[T array.Numeric](v T) []byte {
	return binary.LittleEndian.AppendUint32(nil, uint32(v))
}

// int64Bound writes a bound of the wider integers, a time and a timestamp.
func int64Bound[T array.Numeric](v T) []byte {
	return binary.LittleEndian.AppendUint64(nil, uint64(v))
}

// floatBound writes a bound of a float.
func floatBound(v float32) []byte {
	return binary.LittleEndian.AppendUint32(nil, math.Float32bits(v))
}

// doubleBound writes a bound of a double.
func doubleBound(v float64) []byte {
	return binary.LittleEndian.AppendUint64(nil, math.Float64bits(v))
}

// boolBound writes a bound of a boolean, which is the one bit a page packs eight
// to a byte and a statistic writes on its own.
func boolBound(v bool) []byte {
	if v {
		return []byte{1}
	}
	return []byte{0}
}
