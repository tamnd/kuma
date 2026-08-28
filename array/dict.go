package array

import (
	"errors"
	"fmt"

	"github.com/tamnd/kuma/dtype"
)

// Dictionary encoding is a column stored as small integers into a shared array
// of values. A column of ten million country codes holds two hundred and fifty
// strings and ten million int32s rather than ten million strings, which is less
// memory, fewer cache misses and a comparison that is an integer comparison.
// pandas calls it a Categorical and Parquet uses it as its main encoding, so
// data arrives already in this shape and expanding it on the way in would be
// throwing the work away.
//
// The indices and the values are both ordinary arrays. That is the whole idea:
// the index array is an integer column like any other and the value array is a
// string column like any other, so slicing, null handling and the accessors are
// the ones already here rather than a second set written for this case.
//
// A null is in the indices. The values are the distinct values of the column
// and there is only one way to be missing, so a dictionary that holds a null of
// its own is legal and means a value that is missing rather than a row that is.

// integer is the index types a dictionary can be encoded with. Arrow allows
// unsigned ones and other writers produce them, though a signed int32 is what
// nearly everything writes.
type integer interface {
	int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64
}

// NewDictionary returns a dictionary encoded array holding the values of dict
// at the positions in indices.
//
// The result borrows both, so slicing it is still constant time and two columns
// read out of the same file share the one copy of the values. The nulls are the
// indices' nulls.
//
// Every index is checked against the length of dict, since the alternative is a
// read out of range in whatever kernel touches the column next, a long way from
// whoever built it. An index in a null slot is not checked, because a producer
// is allowed to leave anything there and the value behind it is never read.
func NewDictionary(indices, dict *Array) (*Array, error) {
	if indices == nil || dict == nil {
		return nil, errors.New("array: a dictionary encoded column needs both indices and values")
	}
	if indices.dict != nil {
		return nil, fmt.Errorf("array: the indices of a dictionary are %s, which is itself dictionary encoded", indices.dt)
	}
	if dict.dict != nil {
		return nil, fmt.Errorf("array: a dictionary of %s, which is itself dictionary encoded", dict.dt)
	}

	dt := dtype.Dictionary{Index: indices.dt, Value: dict.dt}
	if err := dtype.Validate(dt); err != nil {
		return nil, fmt.Errorf("array: %w", err)
	}
	if err := checkIndices(indices, dict.length); err != nil {
		return nil, err
	}

	out := *indices
	out.dt = dt
	out.dict = dict
	return &out, nil
}

// Dictionary returns the values a dictionary encoded column indexes into, or
// nil for a column that is not dictionary encoded.
//
// It is the array this column was built with rather than a copy, which is what
// makes it shared, so it has its own length and its own nulls and neither is
// this column's.
func (a *Array) Dictionary() *Array { return a.dict }

// Indices returns the index values of a dictionary encoded column as an array
// of the index type, or nil for a column that is not dictionary encoded.
//
// It shares this column's memory and carries this column's nulls and offset, so
// a slice of a dictionary column has the indices of that slice. Reading them is
// then the ordinary Values and Value, since what comes back is an integer
// column like any other.
func (a *Array) Indices() *Array {
	// The two conditions are the same condition. Only NewDictionary sets either
	// of them and it sets both, so asking twice costs a comparison and saves a
	// panic in the case that cannot happen.
	d, ok := a.dt.(dtype.Dictionary)
	if !ok || a.dict == nil {
		return nil
	}

	out := *a
	out.dt = d.Index
	out.dict = nil
	return &out
}

// Index returns where in the dictionary value i of a dictionary encoded column
// is, or -1 when the value is missing. It panics if the column is not
// dictionary encoded or if i is out of range.
//
// A missing value is -1 rather than whatever the producer happened to leave in
// the index, which is the same convention kernel.Take reads as a null, so a
// caller that forgets to ask IsNull gets an index out of range rather than a
// value that was never there.
func (a *Array) Index(i int) int {
	if a.dict == nil {
		panic(fmt.Sprintf("array: Index on a %s column", a.dt))
	}
	if uint(i) >= uint(a.length) {
		panic("array: index out of range")
	}
	if a.IsNull(i) {
		return -1
	}

	// The index has already been checked against the dictionary, so it is a
	// position in it and the conversion to int is the one that fits.
	idx := a.Indices()
	switch lay, _ := dtypeLayout(idx.dt); lay {
	case layoutInt8:
		return int(idx.Value[int8](i))
	case layoutInt16:
		return int(idx.Value[int16](i))
	case layoutInt32:
		return int(idx.Value[int32](i))
	case layoutInt64:
		return int(idx.Value[int64](i))
	case layoutUint8:
		return int(idx.Value[uint8](i))
	case layoutUint16:
		return int(idx.Value[uint16](i))
	case layoutUint32:
		return int(idx.Value[uint32](i))
	case layoutUint64:
		return int(idx.Value[uint64](i))
	default:
		panic(fmt.Sprintf("array: a dictionary indexed by %s", idx.dt))
	}
}

// checkIndices reports whether every index that will be read is a position in a
// dictionary of n values.
//
// The null check is inside the branch that has already found a bad index rather
// than in the loop, so a column with no nulls and nothing wrong with it costs
// one comparison per value and no bitmap reads at all.
func checkIndices(a *Array, n int) error {
	switch lay, _ := dtypeLayout(a.dt); lay {
	case layoutInt8:
		return checkRange[int8](a, n)
	case layoutInt16:
		return checkRange[int16](a, n)
	case layoutInt32:
		return checkRange[int32](a, n)
	case layoutInt64:
		return checkRange[int64](a, n)
	case layoutUint8:
		return checkRange[uint8](a, n)
	case layoutUint16:
		return checkRange[uint16](a, n)
	case layoutUint32:
		return checkRange[uint32](a, n)
	case layoutUint64:
		return checkRange[uint64](a, n)
	default:
		// dtype.Validate has already refused a non integer index type, so this
		// is an integer type that this package cannot read, which is none of
		// them today.
		return fmt.Errorf("array: a dictionary indexed by %s", a.dt)
	}
}

// checkRange is checkIndices once the index type is known.
//
// The comparison is unsigned so that a negative index and one past the end are
// the same test. A negative int32 converts to a very large uint64, which is out
// of range for every dictionary there could be.
func checkRange[T integer](a *Array, n int) error {
	limit := uint64(n)
	for i, v := range a.Values[T]() {
		if uint64(v) >= limit && a.IsValid(i) {
			return fmt.Errorf("array: index %d of the column is %d, which is not a value of a dictionary of %d",
				i, v, n)
		}
	}
	return nil
}
