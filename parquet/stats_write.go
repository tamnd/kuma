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

// tracker is the smallest and largest value of a column chunk, kept as its rows
// go past.
//
// The bounds of a chunk are the chunk's own, so taking them is what starts the
// next one. That way there is nothing to reset and nothing to forget to reset,
// which on a file of a thousand row groups would be a footer of bounds that grew
// wider and wider and belonged to no chunk at all.
type tracker struct {
	// track folds rows [i, j) of a into what has been seen.
	track func(a *array.Array, i, j int)

	// take is what has been seen, written as a statistic writes it, and whether
	// anything was. It forgets what it returns.
	take func() (lo, hi []byte, ok bool)
}

// numberBounds tracks a column of numbers.
func numberBounds[T array.Numeric](put func(T) []byte) tracker {
	var lo, hi T
	var seen bool

	return tracker{
		track: func(a *array.Array, i, j int) {
			vals := a.Values[T]()
			for k := i; k < j; k++ {
				// The NaN test is free on the integers, none of which can be
				// one, and the conversion is only ever asked whether the value
				// is a number rather than what number it is.
				v := vals[k]
				if !a.IsValid(k) || math.IsNaN(float64(v)) {
					continue
				}

				switch {
				case !seen:
					lo, hi, seen = v, v, true
				case v < lo:
					lo = v
				case v > hi:
					hi = v
				}
			}
		},

		take: func() ([]byte, []byte, bool) {
			if !seen {
				return nil, nil, false
			}
			seen = false
			return put(lo), put(hi), true
		},
	}
}

// boolBounds tracks a column of booleans, where false is the smaller of the two
// and there are only the two.
func boolBounds() tracker {
	var lo, hi, seen bool

	return tracker{
		track: func(a *array.Array, i, j int) {
			for k := i; k < j; k++ {
				if !a.IsValid(k) {
					continue
				}

				v := a.Bool(k)
				switch {
				case !seen:
					lo, hi, seen = v, v, true
				case v:
					hi = true
				default:
					lo = false
				}
			}
		},

		take: func() ([]byte, []byte, bool) {
			if !seen {
				return nil, nil, false
			}
			seen = false
			return boolBound(lo), boolBound(hi), true
		},
	}
}

// bytesBounds tracks a column whose values are bytes, which compare as unsigned
// bytes from the left, the shorter first when one is the front of the other.
//
// That is the order the format defines and it is the reason the format had to
// define one, since the writers that came before it compared those same bytes as
// signed numbers and put every value with a high bit set before every value
// without one.
func bytesBounds() tracker {
	var lo, hi []byte
	var seen bool

	return tracker{
		track: func(a *array.Array, i, j int) {
			for k := i; k < j; k++ {
				if !a.IsValid(k) {
					continue
				}

				v := a.Bytes(k)
				switch {
				case !seen:
					lo, hi, seen = v, v, true
				case bytes.Compare(v, lo) < 0:
					lo = v
				case bytes.Compare(v, hi) > 0:
					hi = v
				}
			}
		},

		take: func() ([]byte, []byte, bool) {
			if !seen {
				return nil, nil, false
			}
			seen = false
			return lo, hi, true
		},
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
