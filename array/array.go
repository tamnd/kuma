// Package array holds the values of one column.
//
// An Array is a dtype, a length, a validity bitmap and the values themselves.
// It is one struct for every type rather than one type per dtype, because the
// engine dispatches on a dtype at runtime and a hierarchy of concrete types
// would mean an interface call per element or a type switch per batch. The
// dtype says which of the value fields is the one in use, and the constructors
// are what make sure it is.
//
// An Array is immutable. Nothing here modifies one after it is built, which is
// what lets the executor hand the same chunk to several goroutines without
// copying it or locking it. Builders are how values get in, and Slice hands
// back a new Array that shares the same memory.
//
// # Nulls
//
// A null is a clear bit in the validity bitmap. It is not a NaN and it is not a
// sentinel value, so a column of integers with a missing value is still a
// column of integers. NullCount is kept up to date rather than recomputed,
// because the branch that matters in every kernel is the one that asks whether
// there are any nulls at all, and a column with none is the common case.
//
// A nil validity bitmap means no nulls. It is not the same as a bitmap with
// every bit set, in that it costs nothing and reads faster, and it is what a
// column that came from a file with no missing values gets.
//
// # Slicing
//
// Slice is constant time. An Array carries an offset into the buffers it
// shares with the array it was sliced from, so slicing a chunk of a million
// rows into a morsel of eight thousand copies nothing. The one thing it has to
// do is count the nulls in the new range, which is a popcount over a few
// hundred bytes, because the whole point of NullCount is that reading it is
// free.
//
// This is the layer the bitmap package's doc comment refers to when it says the
// layers above keep their own offset.
//
// # Building
//
// A Builder is how values get in. It is for one dtype, decided when it is made,
// and it hands its memory over to the Array rather than copying it. A column
// with no nulls never allocates a validity bitmap, since the builder counts the
// values until the first null arrives and only fills in the bits before it if
// one ever does.
//
// The constructors are the other way in, for a reader that already has the
// bytes laid out the way the column wants them, such as one reading a mapped
// file.
//
// # What is not here yet
//
// The nested types, meaning List, Struct and Map, and ChunkedArray, which is a
// column too long to be one Array. Both are coming and neither changes the
// shape of what is here.
//
// Stability: tier 1, stable.
package array

import (
	"fmt"

	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/strview"
)

// Array is the values of one column, or of one chunk of one column.
//
// The zero Array is not usable. Use one of the constructors.
type Array struct {
	dt     dtype.DataType
	offset int
	length int
	nulls  int

	// validity is nil when there are no nulls. It is shared with whatever this
	// was sliced from, so bit i of this array is bit offset+i of the bitmap and
	// its length is not this array's length.
	validity *bitmap.Bitmap

	// values holds fixed width values, or the packed bits of a Bool column. It
	// is nil for String, Binary and Null.
	values *buffer.Buffer

	// strings holds the values of a String or Binary column, and is nil for
	// everything else.
	strings *strview.Data
}

// DType returns the type of the values.
func (a *Array) DType() dtype.DataType { return a.dt }

// Len returns the number of values.
func (a *Array) Len() int { return a.length }

// NullCount returns how many of the values are missing. Reading it is free,
// which is why every operation that produces an Array pays to keep it right.
func (a *Array) NullCount() int { return a.nulls }

// Offset returns where in the shared buffers this array starts, in elements. It
// is zero for an array that was not sliced.
func (a *Array) Offset() int { return a.offset }

// IsValid reports whether value i is present. It panics if i is out of range,
// matching the behavior of an ordinary slice index.
func (a *Array) IsValid(i int) bool {
	if uint(i) >= uint(a.length) {
		panic("array: index out of range")
	}
	if a.nulls == 0 {
		return true
	}
	if a.validity == nil {
		// A Null column, where the type itself says every value is missing.
		return false
	}
	return a.validity.Get(a.offset + i)
}

// IsNull reports whether value i is missing.
func (a *Array) IsNull(i int) bool { return !a.IsValid(i) }

// Validity returns the validity bitmap, or nil when there are no nulls.
//
// It is the bitmap this array shares with whatever it was sliced from, so bit i
// of this array is bit Offset()+i of the result and its length is not this
// array's length. A kernel walking one array wants IsValid. A kernel combining
// two of them wants this, and has to line up the offsets itself.
func (a *Array) Validity() *bitmap.Bitmap { return a.validity }

// Buffer returns the buffer holding the fixed width values, or nil for a
// String, Binary or Null column. It is shared the same way Validity is.
func (a *Array) Buffer() *buffer.Buffer { return a.values }

// Strings returns the values of a String or Binary column, or nil for anything
// else. It is shared the same way Validity is, so value i of this array is
// value Offset()+i of the result.
func (a *Array) Strings() *strview.Data { return a.strings }

// String returns a description of the array for debugging. It does not print
// the values, since an array is as long as a column.
func (a *Array) String() string {
	return fmt.Sprintf("array.Array{%s, len %d, nulls %d, offset %d}",
		a.dt, a.length, a.nulls, a.offset)
}

// Slice returns values i through j-1 as a new array sharing this one's memory.
// It panics if the range is out of bounds.
//
// It is constant time except for counting the nulls in the new range, which is
// a popcount over one byte per eight values and only happens at all when there
// are nulls to count.
func (a *Array) Slice(i, j int) *Array {
	if i < 0 || j < i || j > a.length {
		panic("array: slice out of range")
	}

	out := *a
	out.offset = a.offset + i
	out.length = j - i

	switch {
	case a.nulls == 0:
		out.nulls = 0
	case a.validity == nil:
		out.nulls = out.length // a Null column, all of it missing
	default:
		out.nulls = out.length - a.validity.CountOnesRange(out.offset, out.offset+out.length)
	}
	return &out
}

// Clone returns a copy that shares no memory with a, holding only the values in
// range so that slicing a chunk out of a column and cloning it does not carry
// the rest of the column along with it.
func (a *Array) Clone() *Array {
	out := &Array{dt: a.dt, length: a.length, nulls: a.nulls}

	if a.validity != nil {
		out.validity = a.validity.Slice(a.offset, a.offset+a.length)
	}
	switch {
	case a.values == nil:
	case a.dt.Kind() == dtype.BoolKind:
		// The values are bits, so the copy goes through the bitmap, where the
		// shifting of a range that does not begin on a byte boundary is already
		// written and already tested.
		bits := a.Bools().Slice(a.offset, a.offset+a.length)
		out.values = buffer.New(len(bits.Bytes()))
		copy(out.values.Bytes(), bits.Bytes())
	default:
		w := byteWidth(a.dt)
		out.values = buffer.New(a.length * w)
		copy(out.values.Bytes(), a.values.Bytes()[a.offset*w:(a.offset+a.length)*w])
	}
	if a.strings != nil {
		var b strview.Builder
		b.Grow(a.length)
		for i := range a.length {
			b.Append(a.strings.At(a.offset + i))
		}
		out.strings = b.Finish()
	}
	return out
}
