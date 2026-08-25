package array

import (
	"fmt"
	"unsafe"

	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/dtype"
)

// Numeric is every Go type a fixed width column can be read as.
//
// The types are exact rather than approximate. A column holds machine numbers,
// and a named type whose underlying type is int64 is a question for the layer
// that knows what the name means, not for the layer that owns the bytes.
type Numeric interface {
	int8 | int16 | int32 | int64 |
		uint8 | uint16 | uint32 | uint64 |
		float32 | float64
}

// Values returns the values of a fixed width column as a Go slice, without
// copying them. It panics if T is not the type the column stores.
//
// This is a method with its own type parameter, which Go 1.27 allows on a
// concrete type. Before that it had to be a function, and a function cannot be
// the thing a caller reaches for after checking a dtype, because
// array.Values[int64](a) reads as a conversion of a and a.Values[int64]() reads
// as a question about a.
//
// The result aliases the column and the caller must not modify it. It is the
// bytes the file was read into, reinterpreted, so writing to it writes to every
// other array sliced from the same buffer.
//
// T has to match how the column is stored, not what it means. A timestamp
// column is int64, a date32 column is int32, and both are read as those. What
// this refuses is a reinterpretation that changes the width or the meaning of
// the bits, since reading a float64 column as an int64 gives numbers that are
// not wrong so much as unrelated.
func (a *Array) Values[T Numeric]() []T {
	want := goLayout[T]()
	got, ok := dtypeLayout(a.dt)
	if !ok || got != want {
		panic(fmt.Sprintf("array: cannot read a %s column as %s", a.dt, want))
	}
	if a.length == 0 {
		return nil
	}
	p := (*T)(unsafe.Pointer(&a.values.Bytes()[0]))
	return unsafe.Slice(p, a.offset+a.length)[a.offset:]
}

// Value returns value i of a fixed width column. It panics if T is not the type
// the column stores or if i is out of range.
//
// It does not report whether the value is present. A null still has bytes
// behind it, usually zero, and reading them as though they were a value is the
// mistake this whole layout exists to prevent, so ask IsValid first.
func (a *Array) Value[T Numeric](i int) T {
	if uint(i) >= uint(a.length) {
		panic("array: index out of range")
	}
	return a.Values[T]()[i]
}

// Bool returns value i of a Bool column. It panics if the column is not Bool or
// if i is out of range.
//
// Bool is the one fixed width type whose values are bits rather than bytes, so
// it is not reachable through Values and gets a method of its own.
func (a *Array) Bool(i int) bool {
	if a.dt.Kind() != dtype.BoolKind {
		panic(fmt.Sprintf("array: Bool on a %s column", a.dt))
	}
	if uint(i) >= uint(a.length) {
		panic("array: index out of range")
	}
	k := a.offset + i
	return a.values.Bytes()[k>>3]&(1<<(uint(k)&7)) != 0
}

// Bools returns the values of a Bool column as a bitmap over the shared buffer,
// where value i of this array is bit Offset()+i. It panics if the column is not
// Bool.
//
// This is for a kernel that wants to work a word at a time, which is the reason
// booleans are packed in the first place. Reading one value is Bool.
func (a *Array) Bools() *bitmap.Bitmap {
	if a.dt.Kind() != dtype.BoolKind {
		panic(fmt.Sprintf("array: Bools on a %s column", a.dt))
	}
	return bitmap.FromBytes(a.values.Bytes(), a.offset+a.length)
}

// Bytes returns value i of a column whose values are bytes rather than numbers,
// meaning String, Binary, FixedSizeBinary, the decimals and the intervals. It
// panics if the column is something else or if i is out of range.
//
// The result aliases the column and the caller must not modify it. Converting
// it to a string copies, which is why that is left to the caller rather than
// done here for every value on the way past.
func (a *Array) Bytes(i int) []byte {
	if uint(i) >= uint(a.length) {
		panic("array: index out of range")
	}
	if a.strings != nil {
		return a.strings.At(a.offset + i)
	}
	w, ok := rawWidth(a.dt)
	if !ok {
		panic(fmt.Sprintf("array: Bytes on a %s column", a.dt))
	}
	k := (a.offset + i) * w
	return a.values.Bytes()[k : k+w : k+w]
}

// layout is how a column's values are laid out, with the meaning taken off.
// A timestamp and an int64 are the same eight bytes and this is what says so.
type layout uint8

const (
	layoutNone layout = iota
	layoutInt8
	layoutInt16
	layoutInt32
	layoutInt64
	layoutUint8
	layoutUint16
	layoutUint32
	layoutUint64
	layoutFloat32
	layoutFloat64
)

// String returns the Go type this layout corresponds to, so that the panic
// message from Values names something the caller can type.
func (p layout) String() string {
	switch p {
	case layoutInt8:
		return "int8"
	case layoutInt16:
		return "int16"
	case layoutInt32:
		return "int32"
	case layoutInt64:
		return "int64"
	case layoutUint8:
		return "uint8"
	case layoutUint16:
		return "uint16"
	case layoutUint32:
		return "uint32"
	case layoutUint64:
		return "uint64"
	case layoutFloat32:
		return "float32"
	case layoutFloat64:
		return "float64"
	default:
		return "no Go type"
	}
}

// goLayout returns the layout of a Go type.
func goLayout[T Numeric]() layout {
	var zero T
	switch any(zero).(type) {
	case int8:
		return layoutInt8
	case int16:
		return layoutInt16
	case int32:
		return layoutInt32
	case int64:
		return layoutInt64
	case uint8:
		return layoutUint8
	case uint16:
		return layoutUint16
	case uint32:
		return layoutUint32
	case uint64:
		return layoutUint64
	case float32:
		return layoutFloat32
	case float64:
		return layoutFloat64
	}
	// Numeric lists exactly the cases above, so this is only reachable by
	// adding a type to the constraint and forgetting this switch.
	panic("array: no layout for this type")
}

// dtypeLayout returns the layout of a dtype, and whether it has one.
//
// The temporal types are here because they are integers with a calendar
// attached. A timestamp column is int64 microseconds and a kernel that adds a
// day to every value is an integer kernel, so refusing to hand it an []int64
// would mean a copy for nothing.
func dtypeLayout(t dtype.DataType) (layout, bool) {
	switch t.Kind() {
	case dtype.Int8Kind:
		return layoutInt8, true
	case dtype.Int16Kind:
		return layoutInt16, true
	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		return layoutInt32, true
	case dtype.Int64Kind, dtype.Date64Kind, dtype.Time64Kind,
		dtype.TimestampKind, dtype.DurationKind:
		return layoutInt64, true
	case dtype.Uint8Kind:
		return layoutUint8, true
	case dtype.Uint16Kind:
		return layoutUint16, true
	case dtype.Uint32Kind:
		return layoutUint32, true
	case dtype.Uint64Kind:
		return layoutUint64, true
	case dtype.Float32Kind:
		return layoutFloat32, true
	case dtype.Float64Kind:
		return layoutFloat64, true
	default:
		return layoutNone, false
	}
}

// rawWidth returns how many bytes one value of a byte valued fixed width type
// takes, and whether t is one of those. The decimals are here because a decimal
// is a wide integer that Go has no type for, and the intervals because a day
// time interval is two int32s and a month day nano interval is two int32s and
// an int64, neither of which is one number either.
func rawWidth(t dtype.DataType) (int, bool) {
	switch t.Kind() {
	case dtype.FixedSizeBinaryKind, dtype.Decimal128Kind, dtype.Decimal256Kind, dtype.IntervalKind:
		return byteWidth(t), true
	default:
		return 0, false
	}
}

// byteWidth returns how many bytes one value takes. It panics for Bool, whose
// values are one bit, and for the variable width types, which have no answer.
func byteWidth(t dtype.DataType) int {
	bits, ok := dtype.Bits(t)
	if !ok {
		panic(fmt.Sprintf("array: %s has no fixed width", t))
	}
	if bits < 8 {
		panic(fmt.Sprintf("array: %s is %d bits, not a whole number of bytes", t, bits))
	}
	return bits / 8
}
