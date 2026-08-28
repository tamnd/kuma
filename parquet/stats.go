package parquet

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// What a writer said about a column chunk, and how much of it can be believed.
//
// A writer usually writes down the smallest and largest value of every chunk it
// writes. That is what makes a scan cheap: a filter looking for the orders of
// one customer reads the footer, finds that the customer's identifier is outside
// the range of nine row groups out of ten, and reads the tenth. The rest of the
// file is never opened, and the bigger the file the more of it that is.
//
// The catch is that a bound is a claim about an order, and parquet took a while
// to say what its orders are. The first files wrote a min and a max compared
// however the writer felt like comparing them, which for a string meant its
// bytes read as signed numbers, so a file written in 2014 says a string starting
// with a byte above 127 is smaller than one starting with an a. The format then
// added a second pair of bounds with an order defined per type and a list on the
// footer saying which columns that order applies to. Both pairs are still in the
// files and the new one is only worth reading when the file said the order,
// which is what ColumnOrder is and what Column carries.
//
// So ReadBounds is two questions rather than one. Which pair to read is settled
// by the order the file gave the column and by what its type is, and then the
// bytes are decoded the way the column's values are decoded, since a bound is
// one of the values. What comes back is an array of two, and a caller that can
// compare two values of a type can compare its filter against them.
//
// Everything here errs towards reading a row group. A bound that is missing, a
// bound that is a NaN and a bound out of a writer nobody can name all come back
// as no bounds at all, and no bounds means read the group. Skipping a group that
// holds a row the filter wanted is the one mistake that changes an answer.

// Bounds is what a writer said about the values of one column chunk.
type Bounds struct {
	// Values holds the smallest value of the chunk and then the largest, so it
	// is an array of two of whatever the column holds. It is nil when the chunk
	// said nothing about its range, or said something this package will not act
	// on, and a caller that finds it nil has to read the chunk.
	Values *array.Array

	// MinExact and MaxExact say the bounds are values out of the chunk rather
	// than values either side of it. A writer that cut a long string down to
	// keep the footer small says so this way, and the bound is still a bound:
	// an inexact minimum is below every value in the chunk rather than equal to
	// one of them, which is all a skip needs. What it cannot do is answer a
	// question about the value itself, so a scan reading a minimum out of the
	// footer instead of the column wants an exact one.
	MinExact bool
	MaxExact bool

	// Count is how many values the chunk holds, which for a flat column is how
	// many rows the row group has. Nulls is how many of them are missing and
	// means nothing unless HasNulls, since a writer that said nothing and a
	// writer that said none are not saying the same thing.
	Count    int64
	Nulls    int64
	HasNulls bool
}

// AllNull says every value of the chunk is missing.
//
// It is worth its own question because it is the one thing a filter can settle
// without comparing anything: a chunk of nothing but nulls has no value that
// matches anything, so a filter of any kind skips the row group. A chunk of no
// values at all counts as one, since it has nothing to match either, and it says
// so whether or not the writer bothered to count the nulls of a chunk it wrote
// nothing in.
func (b Bounds) AllNull() bool { return b.Count == 0 || (b.HasNulls && b.Nulls == b.Count) }

// ReadBounds decodes what a writer said about the values of one column chunk.
//
// The chunk has to be the one holding the column c, which is one of the leaves
// Metadata.Columns returned, and the order the file gave that column comes along
// on it. Nothing is read out of the file: the bounds are in the footer, which is
// the whole point of them.
//
// The bounds come back decoded into an array of two values, the smallest first,
// or as no array at all when the chunk said nothing worth acting on. That covers
// a chunk with no statistics, one bounded only by the pair the format deprecated
// on a type whose old order was the wrong one, a column whose type has no order
// at all, and a float chunk bounded by a NaN.
//
// A bound this package cannot decode is an error rather than an absence, the
// same way a column it cannot assemble is. A chunk whose bounds contradict what
// it says about itself is an error too, since a footer that disagrees with
// itself is not one to skip anything on.
func ReadBounds(c Column, m *ColumnMeta) (Bounds, error) {
	s := &m.Stats
	b := Bounds{Count: m.NumValues, Nulls: s.NullCount, HasNulls: s.HasNullCount}
	if b.HasNulls && (b.Nulls < 0 || b.Nulls > b.Count) {
		return Bounds{}, fmt.Errorf("parquet: %w: the chunk for %s holds %d values and says %d are missing",
			ErrFormat, c.Name(), b.Count, b.Nulls)
	}

	lo, hi, minExact, maxExact := believable(&c, s)
	if lo == nil || hi == nil {
		return b, nil
	}
	if nan(c.Element.Type, lo) || nan(c.Element.Type, hi) {
		return b, nil
	}

	values, err := decodeBounds(&c, lo, hi)
	if err != nil {
		return Bounds{}, err
	}
	b.Values, b.MinExact, b.MaxExact = values, minExact, maxExact
	return b, nil
}

// believable picks the pair of bounds a reader can act on, if the chunk has one.
//
// The pair to read is MinValue and MaxValue, which is the one the format defined
// an order for. It says nothing unless the file said which order, so a file that
// left the orders out leaves a reader where it was before the format settled the
// question: the deprecated pair, and only on the types nobody ever disagreed
// about. A boolean, a signed integer, a float and a double compare now the way
// they compared then. A string does not, since the writers of the day compared
// its bytes as signed numbers, and neither does an integer wide enough to be
// unsigned in the file and negative when it is read back signed.
//
// The two exactness flags belong to the new pair. The old one was never
// truncated, because truncating a bound is what the flags were added to say.
func believable(c *Column, s *Statistics) (lo, hi []byte, minExact, maxExact bool) {
	if !ordered(c) {
		return nil, nil, false, false
	}
	if c.Order == TypeDefinedOrder && s.MinValue != nil && s.MaxValue != nil {
		return s.MinValue, s.MaxValue, s.MinExact, s.MaxExact
	}
	if !oldOrder(c) {
		return nil, nil, false, false
	}
	return s.Min, s.Max, true, true
}

// ordered says whether the format defines what smaller means for the column at
// all.
//
// Two types have no order. An int96 is the timestamp parquet had before it had
// one, and its twelve bytes are two numbers that the writers of the day compared
// as bytes, which puts every morning before every evening whatever day it is. An
// interval is a width of time in months, days and milliseconds, and those cannot
// be put on one line, since a month is longer than thirty days in seven months
// of the year and shorter in four. The null type is not unordered so much as not
// worth ordering, every value of it being missing. Neither is a column with no
// type at all, which is not a column, and which is worth a word here because a
// Column is a struct a caller can fill in.
func ordered(c *Column) bool {
	if c.Element.Type == Int96 || c.Element.Converted == ConvertedInterval {
		return false
	}
	return c.Type != nil && c.Type.Kind() != dtype.NullKind
}

// oldOrder says whether the deprecated pair of bounds is worth reading, which is
// a question about which comparison the writers that wrote them used.
//
// They used a signed one on everything. That is the right comparison for a
// boolean, for the integers parquet writes signed, and for the two floating
// point types, so those are the ones to read. It is the wrong one for a string,
// whose bytes are unsigned and which is the reason the format settled the
// question at all, and the wrong one for an unsigned integer wide enough to have
// the top bit set, which is a uint32 in an int32 and a uint64 in an int64. A
// uint8 and a uint16 are written in an int32 that is never negative, so a signed
// comparison puts them in the order an unsigned one would and they are read like
// any other integer.
//
// This is only asked about a column that ordered let through, so the type is one
// there is something to ask about.
func oldOrder(c *Column) bool {
	if k := c.Type.Kind(); k == dtype.Uint32Kind || k == dtype.Uint64Kind {
		return false
	}
	switch c.Element.Type {
	case Boolean, Int32, Int64, Float, Double:
		return true
	default:
		return false
	}
}

// nan says a bound is a NaN, which is a bound that means nothing.
//
// NaN compares false against everything, itself included, so a chunk whose
// smallest or largest value is one has no range for a filter to be kept out of.
// The format tells a writer to leave NaN out of its bounds and the writers do,
// but a reader that acted on one anyway would skip a row group holding rows it
// wanted, which is the one kind of mistake a skip must never make. The two zeros
// need no such care: nothing can tell them apart by comparing, so a bound of
// either is a bound of both.
func nan(t Type, v []byte) bool {
	switch t {
	case Float:
		return len(v) == 4 && math.IsNaN(float64(math.Float32frombits(binary.LittleEndian.Uint32(v))))
	case Double:
		return len(v) == 8 && math.IsNaN(math.Float64frombits(binary.LittleEndian.Uint64(v)))
	default:
		return false
	}
}

// decodeBounds turns a pair of bounds into an array holding the two of them.
//
// They are read the way the column's own values are read, with the same builder
// and the same decoder and the same narrowing of what parquet wrote to what kuma
// stores, because a bound is one of the values. The one difference is a byte
// array, which a page writes behind four bytes of its length and a statistic
// writes as itself, so the length goes back in front of it here.
//
// A bound that is not exactly one value of the column is a footer that
// contradicts its own schema. Whatever it holds is not the smallest value of the
// chunk, and a skip is the last thing to guess at.
func decodeBounds(c *Column, lo, hi []byte) (*array.Array, error) {
	b, err := array.NewBuilder(c.Type)
	if err != nil {
		return nil, fmt.Errorf("parquet: %s: %w", c.Name(), err)
	}
	values, err := valuesFor(*c, b)
	if err != nil {
		return nil, err
	}

	var d PlainDecoder
	for _, v := range [2][]byte{lo, hi} {
		if c.Element.Type == ByteArray {
			v = append(binary.LittleEndian.AppendUint32(nil, uint32(len(v))), v...)
		}

		d.Reset(v)
		if err := values.decode(&d, 1); err != nil {
			return nil, fmt.Errorf("parquet: the bounds of %s: %w", c.Name(), err)
		}
		if d.Left() != 0 {
			return nil, fmt.Errorf("parquet: %w: a bound of %s with %d bytes left over",
				ErrFormat, c.Name(), d.Left())
		}
		values.run(0, 1)
	}
	return b.Finish(), nil
}
