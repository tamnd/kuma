package array

import (
	"errors"
	"fmt"
	"math"
	"unsafe"

	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/strview"
)

// errNilDType is the one error every constructor here can return, since a
// column with no type is the one mistake that is not about the values.
var errNilDType = errors.New("array: nil dtype")

// New returns a fixed width array of length values of type dt, holding the
// values in values and the nulls in valid.
//
// The array takes over both, meaning the caller must not modify them
// afterwards. Nothing is copied, which is what lets a column be read straight
// out of a mapped file.
//
// A nil valid says every value is present. It may also be longer than the
// array, since a bitmap sized in whole bytes usually is. A bitmap with
// every bit set is dropped rather than kept, because a column with no nulls
// reads faster without one and the two say the same thing.
//
// The buffer has to be long enough for length values, and may be longer. For a
// Bool column the values are bits, so it needs one byte per eight values.
func New(dt dtype.DataType, length int, values *buffer.Buffer, valid *bitmap.Bitmap) (*Array, error) {
	if dt == nil {
		return nil, errNilDType
	}
	if length < 0 {
		return nil, fmt.Errorf("array: negative length %d", length)
	}
	if values == nil {
		return nil, fmt.Errorf("array: nil value buffer for a %s column", dt)
	}

	need, err := valueBytes(dt, length)
	if err != nil {
		return nil, err
	}
	if got := values.Len(); got < need {
		return nil, fmt.Errorf("array: a %s column of %d values needs %d bytes, the buffer has %d",
			dt, length, need, got)
	}

	a := &Array{dt: dt, length: length, values: values}
	if err := a.setValidity(valid); err != nil {
		return nil, err
	}
	return a, nil
}

// NewStrings returns a String or Binary array over d, with the nulls in valid.
// The length is the number of values in d.
//
// LargeString and LargeBinary are not here. They are the 64 bit offset layout
// that arrives over Arrow IPC and they are converted at that boundary, which
// they can be without losing anything, since a view has no global offset to
// overflow. A value is found by a block number and an offset inside that block,
// so a column of any size is a column with more blocks, and the only thing that
// does not fit is a single value longer than two gigabytes.
//
// The array takes over d and valid, the same way New does.
func NewStrings(dt dtype.DataType, d *strview.Data, valid *bitmap.Bitmap) (*Array, error) {
	if dt == nil {
		return nil, errNilDType
	}
	switch dt.Kind() {
	case dtype.StringKind, dtype.BinaryKind:
	default:
		return nil, fmt.Errorf("array: NewStrings on a %s column", dt)
	}
	if d == nil {
		return nil, fmt.Errorf("array: nil values for a %s column", dt)
	}

	a := &Array{dt: dt, length: d.Len(), strings: d}
	if err := a.setValidity(valid); err != nil {
		return nil, err
	}
	return a, nil
}

// NewNull returns an array of length values, all of them missing, of the type
// that has no values. It panics if length is negative.
//
// It carries no bitmap. A Null column is one where the type itself says every
// value is missing, so there is nothing to record per value and nothing to
// allocate, however long the column is.
func NewNull(length int) *Array {
	if length < 0 {
		panic("array: negative length")
	}
	return &Array{dt: dtype.Null, length: length, nulls: length}
}

// Empty returns an array of type dt holding no values.
//
// It is for the operation that produced nothing and still has a type to hand
// back, such as a gather that kept no rows out of a column that is the same type
// either way. The nested types have one too, and their children are empty in the
// same way, since a list of no rows still points at a child.
func Empty(dt dtype.DataType) (*Array, error) {
	if dt == nil {
		return nil, errNilDType
	}

	switch t := dt.(type) {
	case dtype.List:
		child, err := Empty(t.Elem)
		if err != nil {
			return nil, err
		}
		// One offset, being where the first row would begin if there were one.
		return NewListFrom(dt, []int32{0}, child, nil)
	case dtype.Dictionary:
		indices, err := Empty(t.Index)
		if err != nil {
			return nil, err
		}
		values, err := Empty(t.Value)
		if err != nil {
			return nil, err
		}
		return NewDictionary(indices, values)
	}

	if dt.Kind() == dtype.NullKind {
		return NewNull(0), nil
	}

	b, err := NewBuilder(dt)
	if err != nil {
		return nil, err
	}
	return b.Finish(), nil
}

// Of returns an array of the given values, with no nulls, of the dtype that
// matches T. It is for tests, examples and the odd literal column, not for
// loading data, which goes through a builder.
//
// This and the two below build the struct rather than going through New. There
// is nothing for New to check that is not already decided here, and an error
// return that cannot happen is a branch nobody can test.
func Of[T Numeric](values ...T) *Array {
	var zero T
	w := int(unsafe.Sizeof(zero))

	buf := buffer.New(len(values) * w)
	if len(values) > 0 {
		// The values are copied as bytes in the machine's own order, which is
		// the order everything else in this package reads them back in.
		copy(buf.Bytes(), unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*w))
	}
	return &Array{dt: layoutDType(goLayout[T]()), length: len(values), values: buf}
}

// OfBools returns a Bool array of the given values, with no nulls.
func OfBools(values ...bool) *Array {
	bits := bitmap.New(len(values))
	for i, v := range values {
		bits.Set(i, v)
	}

	buf := buffer.New(len(bits.Bytes()))
	copy(buf.Bytes(), bits.Bytes())

	return &Array{dt: dtype.Bool, length: len(values), values: buf}
}

// OfStrings returns a String array of the given values, with no nulls.
func OfStrings(values ...string) *Array {
	var b strview.Builder
	b.Grow(len(values))
	for _, v := range values {
		b.AppendString(v)
	}

	d := b.Finish()
	return &Array{dt: dtype.String, length: d.Len(), strings: d}
}

// setValidity attaches valid to a and counts the nulls, or reports why it does
// not fit.
func (a *Array) setValidity(valid *bitmap.Bitmap) error {
	if valid == nil {
		return nil
	}
	if valid.Len() < a.length {
		return fmt.Errorf("array: a validity bitmap of %d bits for %d values",
			valid.Len(), a.length)
	}
	nulls := a.length - valid.CountOnesRange(0, a.length)
	if nulls == 0 {
		return nil
	}
	a.validity = valid
	a.nulls = nulls
	return nil
}

// valueBytes returns how many bytes length values of type dt take, or reports
// why dt is not a type New can hold.
func valueBytes(dt dtype.DataType, length int) (int, error) {
	switch dt.Kind() {
	case dtype.NullKind:
		return 0, errors.New("array: a null column has no values, use NewNull")
	case dtype.StringKind, dtype.BinaryKind:
		return 0, fmt.Errorf("array: a %s column has no fixed width, use NewStrings", dt)
	case dtype.LargeStringKind:
		return 0, errors.New("array: a large_string column is converted at the IPC boundary, store it as a string")
	case dtype.LargeBinaryKind:
		return 0, errors.New("array: a large_binary column is converted at the IPC boundary, store it as a binary")
	case dtype.ListKind:
		return 0, fmt.Errorf("array: a %s column is offsets into a child, use NewList or NewListBuilder", dt)
	case dtype.LargeListKind:
		return 0, errors.New("array: a large_list column is converted at the IPC boundary, store it as a list")
	case dtype.BoolKind:
		if length > math.MaxInt-7 {
			return 0, tooWide(dt, length)
		}
		return (length + 7) / 8, nil
	default:
		bits, ok := dtype.Bits(dt)
		if !ok {
			return 0, fmt.Errorf("array: a %s column is not supported yet", dt)
		}
		// A negative width is only reachable through a type someone built by
		// hand and never validated, and it has to be caught here because a
		// negative number of bytes is a panic several layers down where nothing
		// remembers what the type was.
		if bits < 0 {
			return 0, fmt.Errorf("array: a %s column is %d bits wide", dt, bits)
		}
		if bits%8 != 0 {
			return 0, fmt.Errorf("array: a %s column is %d bits wide, which is not a whole number of bytes", dt, bits)
		}
		// The length is a number a producer sent, so the multiplication is
		// checked rather than trusted. A length near the top of an int times
		// eight comes out small, and a small answer here is a buffer that looks
		// big enough for a column nothing could hold.
		if width := bits / 8; width == 0 || length <= math.MaxInt/width {
			return length * width, nil
		}
		return 0, tooWide(dt, length)
	}
}

// tooWide is the error for a column of more values than there are bytes to
// count them in.
func tooWide(dt dtype.DataType, length int) error {
	return fmt.Errorf("array: a %s column of %d values is more bytes than an int can count", dt, length)
}

// layoutDType returns the dtype a Go type maps to, which is the plain one
// rather than any of the types that share its layout. A caller who wants
// timestamps builds the array with New and says so.
func layoutDType(p layout) dtype.DataType {
	switch p {
	case layoutInt8:
		return dtype.Int8
	case layoutInt16:
		return dtype.Int16
	case layoutInt32:
		return dtype.Int32
	case layoutInt64:
		return dtype.Int64
	case layoutUint8:
		return dtype.Uint8
	case layoutUint16:
		return dtype.Uint16
	case layoutUint32:
		return dtype.Uint32
	case layoutUint64:
		return dtype.Uint64
	case layoutFloat32:
		return dtype.Float32
	case layoutFloat64:
		return dtype.Float64
	default:
		// goLayout returns one of the above for every type Numeric allows.
		panic("array: no dtype for this layout")
	}
}
