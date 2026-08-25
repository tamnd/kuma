package array

import (
	"fmt"
	"unsafe"

	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/strview"
)

// Builder accumulates values into an Array.
//
// A builder is for one dtype, decided when it is made, and the append methods
// panic if they are handed something else. That is the same rule the read side
// follows: a column knows what it holds, and code that does not know is code
// with a bug in it rather than code that should be handed a conversion.
//
// A column with no nulls never allocates a validity bitmap. The builder counts
// values until the first null arrives and only then fills in the bits for
// everything before it, so the common case of a column read from a file with
// nothing missing costs nothing to track.
//
// The zero Builder is not usable. Use NewBuilder.
type Builder struct {
	dt     dtype.DataType
	lay    layout // layoutNone unless the values are numbers
	width  int    // bytes per value, zero for Bool, the byte types and Null
	length int
	nulls  int

	values *buffer.Buffer  // fixed width values
	bools  bitmap.Builder  // Bool values
	strs   strview.Builder // String and Binary values

	// valid is nil until the first null, since a column with none needs no
	// bitmap at all and that is the case worth not paying for.
	valid *bitmap.Builder
}

// NewBuilder returns a builder for a column of type dt.
func NewBuilder(dt dtype.DataType) (*Builder, error) {
	if dt == nil {
		return nil, errNilDType
	}

	// The layout is worked out once here rather than per value, since it is a
	// question about the type and the type does not change. Append is the
	// hottest loop in a row oriented reader and an interface call per value in
	// it is one that buys nothing.
	lay, _ := dtypeLayout(dt)

	b := &Builder{dt: dt, lay: lay}
	switch dt.Kind() {
	case dtype.NullKind, dtype.BoolKind, dtype.StringKind, dtype.BinaryKind:
	default:
		n, err := valueBytes(dt, 1)
		if err != nil {
			return nil, err
		}
		b.width = n
		b.values = buffer.New(0)
	}
	return b, nil
}

// DType returns the type of the column being built.
func (b *Builder) DType() dtype.DataType { return b.dt }

// Len returns how many values have been appended, nulls included.
func (b *Builder) Len() int { return b.length }

// NullCount returns how many of them are missing.
func (b *Builder) NullCount() int { return b.nulls }

// Grow makes room for n more values. It panics if n is negative.
//
// It is worth calling when the length is known, which for a reader it usually
// is, since a chunk is read into a buffer of a size the reader chose.
func (b *Builder) Grow(n int) {
	if n < 0 {
		panic("array: negative grow")
	}
	switch {
	case b.values != nil:
		b.values.Grow(n * b.width)
	case b.dt.Kind() == dtype.BoolKind:
		b.bools.Grow(n)
	case isStringKind(b.dt):
		b.strs.Grow(n)
	}
	if b.valid != nil {
		b.valid.Grow(n)
	}
}

// Append adds one value. It panics if T is not the type the column stores.
//
// The type is checked against the layout of the column rather than its dtype,
// the same way Values is, so a timestamp column takes an int64 and a date32
// column takes an int32.
func (b *Builder) Append[T Numeric](v T) {
	b.checkLayout(goLayout[T]())
	b.values.Append(unsafe.Slice((*byte)(unsafe.Pointer(&v)), b.width))
	b.appendValid()
}

// AppendValues adds every value in vs. It panics if T is not the type the
// column stores.
//
// This is the one to reach for in a loop over a batch. The type is checked once
// instead of once per value, and the values go into the buffer as one copy.
func (b *Builder) AppendValues[T Numeric](vs []T) {
	b.checkLayout(goLayout[T]())
	if len(vs) == 0 {
		return
	}

	b.values.Append(unsafe.Slice((*byte)(unsafe.Pointer(&vs[0])), len(vs)*b.width))
	b.appendValidMany(len(vs))
}

// AppendBool adds one value to a Bool column. It panics if the column is not
// Bool.
func (b *Builder) AppendBool(v bool) {
	b.checkKind(dtype.BoolKind, "AppendBool")
	b.bools.Append(v)
	b.appendValid()
}

// AppendBools adds every value in vs to a Bool column. It panics if the column
// is not Bool.
func (b *Builder) AppendBools(vs []bool) {
	b.checkKind(dtype.BoolKind, "AppendBools")
	b.bools.AppendBools(vs)
	b.appendValidMany(len(vs))
}

// AppendBytes adds one value to a String, Binary or FixedSizeBinary column, or
// to one of the decimals or intervals. It copies p, so the caller may reuse it,
// which is what makes it safe to build a column out of a reader's own read
// buffer.
//
// It panics if the column is something else, or if the column is fixed width
// and p is not exactly that wide.
func (b *Builder) AppendBytes(p []byte) {
	if isStringKind(b.dt) {
		b.strs.Append(p)
		b.appendValid()
		return
	}
	if _, ok := rawWidth(b.dt); !ok {
		panic(fmt.Sprintf("array: AppendBytes on a %s column", b.dt))
	}
	if len(p) != b.width {
		panic(fmt.Sprintf("array: a %s value is %d bytes, got %d", b.dt, b.width, len(p)))
	}

	b.values.Append(p)
	b.appendValid()
}

// AppendString adds one value to a String or Binary column. It panics if the
// column is something else.
func (b *Builder) AppendString(s string) {
	if !isStringKind(b.dt) {
		panic(fmt.Sprintf("array: AppendString on a %s column", b.dt))
	}
	b.strs.AppendString(s)
	b.appendValid()
}

// AppendNull adds one missing value.
//
// A null still takes up its place in the values, since everything downstream
// indexes the values and the bitmap with the same number. What goes there is
// zero, and reading it as though it were a value is the mistake the bitmap
// exists to prevent.
func (b *Builder) AppendNull() { b.AppendNulls(1) }

// AppendNulls adds n missing values. It panics if n is negative.
func (b *Builder) AppendNulls(n int) {
	if n < 0 {
		panic("array: negative count")
	}
	if n == 0 {
		return
	}

	switch {
	case b.values != nil:
		b.values.Grow(n * b.width)
		b.values.Resize(b.values.Len() + n*b.width)
	case b.dt.Kind() == dtype.BoolKind:
		b.bools.AppendMany(false, n)
	case isStringKind(b.dt):
		for range n {
			b.strs.Append(nil)
		}
	}

	if b.dt.Kind() != dtype.NullKind {
		// A null column keeps no bitmap, since its type already says that every
		// value is missing. Anything else records which ones are.
		b.startValidity()
		b.valid.AppendMany(false, n)
	}
	b.length += n
	b.nulls += n
}

// Reset drops everything and leaves a builder for the same dtype ready to use
// again.
//
// It gives up the memory rather than keeping it, because the memory may have
// been handed to an Array by Finish and writing into it again would change a
// column somebody else is reading.
func (b *Builder) Reset() {
	b.length, b.nulls = 0, 0
	b.valid = nil
	b.bools.Reset()
	b.strs.Reset()
	if b.values != nil {
		b.values = buffer.New(0)
	}
}

// Finish returns the values appended so far as an Array and resets the builder.
//
// The array takes the builder's memory rather than a copy of it, except for a
// Bool column, whose bits are copied once into an aligned buffer. Finish is
// what makes that safe: the builder comes back empty, so there is no way to
// write through it into a column that has already been handed out.
func (b *Builder) Finish() *Array {
	a := &Array{dt: b.dt, length: b.length, nulls: b.nulls}
	if b.valid != nil {
		a.validity = b.valid.Finish()
	}

	switch {
	case b.values != nil:
		a.values = b.values
	case b.dt.Kind() == dtype.BoolKind:
		// The bits come out of a plain byte slice, and a column's values are
		// meant to start at an aligned address, so this is the one copy in the
		// whole path. It is one byte per eight values.
		bits := b.bools.Finish().Bytes()
		a.values = buffer.New(len(bits))
		copy(a.values.Bytes(), bits)
	case isStringKind(b.dt):
		a.strings = b.strs.Finish()
	}

	b.Reset()
	return a
}

// appendValid records one present value.
func (b *Builder) appendValid() {
	if b.valid != nil {
		b.valid.Append(true)
	}
	b.length++
}

// appendValidMany records n present values, which is one call into the bitmap
// rather than n of them.
func (b *Builder) appendValidMany(n int) {
	if b.valid != nil {
		b.valid.AppendMany(true, n)
	}
	b.length += n
}

// startValidity makes sure there is a bitmap, filling in the values appended
// before the first null, all of which were present.
func (b *Builder) startValidity() {
	if b.valid != nil {
		return
	}
	b.valid = &bitmap.Builder{}
	b.valid.AppendMany(true, b.length)
}

// checkLayout panics unless the column stores values laid out as want. A column
// whose values are not numbers has a layout of layoutNone, which no Go type
// maps to, so it fails here the same way a mismatch does.
func (b *Builder) checkLayout(want layout) {
	if b.lay != want {
		panic(fmt.Sprintf("array: cannot append %s to a %s column", want, b.dt))
	}
}

// checkKind panics unless the column is of kind want, naming the method that
// was called so that the message says what to call instead.
func (b *Builder) checkKind(want dtype.Kind, method string) {
	if b.dt.Kind() != want {
		panic(fmt.Sprintf("array: %s on a %s column", method, b.dt))
	}
}

// isStringKind reports whether values of type dt live in a strview.Data.
func isStringKind(dt dtype.DataType) bool {
	switch dt.Kind() {
	case dtype.StringKind, dtype.BinaryKind:
		return true
	default:
		return false
	}
}
