package array

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
)

// A list column is offsets into one child array.
//
// Every element of every row lives in the same child, one row after another, and
// the offsets say where each row begins. Row i is child[offsets[i]:offsets[i+1]],
// so there are length+1 offsets and the last one is the number of elements the
// column holds in total. A row of no elements is two offsets that are the same
// number, which is not the same thing as a null row and does not read as one.
//
// That layout is what makes a list column cheap. Reading one row is two integers
// and a slice of the child, which shares memory rather than copying, and a kernel
// that does not care where the rows begin runs over the child as though it were
// an ordinary column. An explode is the second of those: the rows come apart and
// the values do not move at all.
//
// The offsets live in the values buffer, since a list column has no fixed width
// values of its own and the buffer is already there, aligned and shared by
// slicing the same way everything else is.

// NewList returns a List array of length rows over child, with the rows marked
// out by offsets and the nulls in valid.
//
// The offsets are length+1 int32 values, where row i is the elements of child
// from offsets[i] up to offsets[i+1]. They have to start at or after zero, never
// go backwards, and end at or before the length of child. A file that says
// otherwise is a file that would have this package index out of range later, so
// it is checked here, once, rather than trusted and paid for in a panic
// somewhere with no idea what the type was.
//
// The array takes over offsets, child and valid, meaning the caller must not
// modify them afterwards. Nothing is copied.
//
// LargeList is not here, for the reason NewStrings gives about LargeString: it
// is the 64 bit offset layout that arrives over Arrow IPC and it is converted at
// that boundary. A child with more elements than an int32 can count becomes more
// chunks rather than a wider offset, and the only thing that does not fit is a
// single row with more than two billion elements in it.
func NewList(dt dtype.DataType, length int, offsets *buffer.Buffer, child *Array,
	valid *bitmap.Bitmap) (*Array, error) {
	if dt == nil {
		return nil, errNilDType
	}
	lt, ok := dt.(dtype.List)
	if !ok {
		return nil, fmt.Errorf("array: NewList on a %s column", dt)
	}
	if length < 0 {
		return nil, fmt.Errorf("array: negative length %d", length)
	}
	if child == nil {
		return nil, fmt.Errorf("array: nil child for a %s column", dt)
	}
	if !dtype.Equal(child.DType(), lt.Elem) {
		return nil, fmt.Errorf("array: a %s column over a %s child", dt, child.DType())
	}
	if offsets == nil {
		return nil, fmt.Errorf("array: nil offsets for a %s column", dt)
	}

	need, err := offsetBytes(dt, length)
	if err != nil {
		return nil, err
	}
	if got := offsets.Len(); got < need {
		return nil, fmt.Errorf("array: a %s column of %d rows needs %d bytes of offsets, the buffer has %d",
			dt, length, need, got)
	}
	if err := checkOffsets(dt, offsetsOf(offsets, 0, length), child.Len()); err != nil {
		return nil, err
	}

	a := &Array{dt: dt, length: length, values: offsets, child: child}
	if err := a.setValidity(valid); err != nil {
		return nil, err
	}
	return a, nil
}

// Child returns the array every row of a list column draws its elements from,
// or nil for anything else.
//
// It is the whole child rather than the part this array's rows reach, since an
// array that was sliced shares it with the one it was sliced from. Use Offsets
// to find a row in it, or List to get one row already sliced out.
//
// This is what a kernel that does not care where the rows begin wants. Summing
// every number in a list column is a sum over the child, and it is only the ones
// that have to stay inside a row that pay for the offsets.
func (a *Array) Child() *Array { return a.child }

// Offsets returns where each row of a list column begins in the child, or nil
// for anything else.
//
// There are Len()+1 of them, so the last one is where the last row ends. They
// are absolute positions in the shared child rather than positions relative to
// this array, which is why a sliced list column does not have to rewrite them.
//
// The result aliases the column and the caller must not modify it.
func (a *Array) Offsets() []int32 {
	if a.child == nil {
		return nil
	}
	return offsetsOf(a.values, a.offset, a.length)
}

// List returns the elements of row i of a list column, as an array sharing the
// child's memory. It panics if the column is not a list or if i is out of range.
//
// A null row comes back empty, which is what its offsets say. Ask IsValid to
// tell that from a row that is present and has nothing in it, the same as for
// any other type.
func (a *Array) List(i int) *Array {
	if a.child == nil {
		panic(fmt.Sprintf("array: List on a %s column", a.dt))
	}
	if uint(i) >= uint(a.length) {
		panic("array: index out of range")
	}
	off := a.Offsets()
	return a.child.Slice(int(off[i]), int(off[i+1]))
}

// ListBuilder accumulates rows into a List array.
//
// A row is built by appending its elements to the builder Elem returns and then
// calling Append, which closes the row. Appending nothing and then calling
// Append is a row of no elements, and AppendNull is a row that is missing.
//
//	b, err := array.NewListBuilder(dtype.List{Elem: dtype.Int64})
//	b.Elem().AppendValues([]int64{1, 2, 3})
//	b.Append()
//	b.AppendNull()
//	a := b.Finish()
//
// The elements go through an ordinary Builder because that is what they are: a
// list column's child is a column, and everything that is true of appending to
// one is true here. What this adds is the offsets.
//
// The zero ListBuilder is not usable. Use NewListBuilder.
type ListBuilder struct {
	dt    dtype.DataType
	elem  *Builder
	off   []int32
	nulls int

	// valid is nil until the first null, the same as in Builder and for the same
	// reason.
	valid *bitmap.Builder
}

// NewListBuilder returns a builder for a list column of type dt.
func NewListBuilder(dt dtype.DataType) (*ListBuilder, error) {
	if dt == nil {
		return nil, errNilDType
	}
	lt, ok := dt.(dtype.List)
	if !ok {
		return nil, fmt.Errorf("array: NewListBuilder on a %s column", dt)
	}

	elem, err := NewBuilder(lt.Elem)
	if err != nil {
		return nil, err
	}
	return &ListBuilder{dt: dt, elem: elem, off: []int32{0}}, nil
}

// DType returns the type of the column being built.
func (b *ListBuilder) DType() dtype.DataType { return b.dt }

// Len returns how many rows have been closed, nulls included. Elements appended
// since the last Append are not a row yet and are not counted.
func (b *ListBuilder) Len() int { return len(b.off) - 1 }

// NullCount returns how many of those rows are missing.
func (b *ListBuilder) NullCount() int { return b.nulls }

// Elem returns the builder the elements of the row being built go into.
//
// It is the child column's builder and it stays the same one for the life of
// this builder, so a caller may hold on to it rather than asking again per row.
func (b *ListBuilder) Elem() *Builder { return b.elem }

// Append closes the row made of the elements appended since the last Append or
// AppendNull. With no elements appended it is a row that is present and empty,
// which is not a null.
func (b *ListBuilder) Append() {
	b.off = append(b.off, int32(b.elem.Len()))
	if b.valid != nil {
		b.valid.Append(true)
	}
}

// AppendNull adds a missing row.
//
// A null row has no elements, so it takes up no room in the child and its two
// offsets are the same number. That is also what an empty row looks like, which
// is why the bitmap is what tells them apart rather than the offsets.
func (b *ListBuilder) AppendNull() {
	if b.valid == nil {
		b.valid = &bitmap.Builder{}
		b.valid.AppendMany(true, b.Len())
	}
	b.off = append(b.off, int32(b.elem.Len()))
	b.valid.Append(false)
	b.nulls++
}

// Grow makes room for n more rows. It panics if n is negative.
//
// It grows the offsets and the bitmap, which are per row. The child is grown
// through Elem, since how many elements n rows hold is something only the caller
// knows.
func (b *ListBuilder) Grow(n int) {
	if n < 0 {
		panic("array: negative grow")
	}
	b.off = append(b.off, make([]int32, n)...)[:len(b.off)]
	if b.valid != nil {
		b.valid.Grow(n)
	}
}

// Reset drops everything and leaves a builder for the same dtype ready to use
// again. It gives up the memory rather than keeping it, for the reason
// [Builder.Reset] gives.
func (b *ListBuilder) Reset() {
	b.off = []int32{0}
	b.nulls = 0
	b.valid = nil
	b.elem.Reset()
}

// Finish returns the rows closed so far as an Array and resets the builder.
//
// Elements appended without a closing Append are dropped, since they are half a
// row and there is nothing else to do with them. The child comes from the
// element builder's own Finish, so it is that builder's memory being handed over
// rather than a copy.
func (b *ListBuilder) Finish() *Array {
	length := b.Len()

	buf := buffer.New((length + 1) * offsetWidth)
	copy(buf.Bytes(), unsafe.Slice((*byte)(unsafe.Pointer(&b.off[0])), (length+1)*offsetWidth))

	a := &Array{dt: b.dt, length: length, nulls: b.nulls, values: buf}
	if b.valid != nil {
		a.validity = b.valid.Finish()
	}

	// The child is trimmed to what the closed rows reach, so elements appended
	// after the last Append do not come along in a child nothing points into.
	a.child = b.elem.Finish().Slice(0, int(b.off[length]))

	b.Reset()
	return a
}

// offsetWidth is the size of one offset in bytes, which is four because the
// offsets of a List are int32 and LargeList is converted at the IPC boundary.
const offsetWidth = 4

// offsetsOf reads length+1 offsets out of buf, starting at row offset.
func offsetsOf(buf *buffer.Buffer, offset, length int) []int32 {
	p := (*int32)(unsafe.Pointer(&buf.Bytes()[0]))
	return unsafe.Slice(p, offset+length+1)[offset:]
}

// offsetBytes returns how many bytes the offsets of a list column of length rows
// take.
func offsetBytes(dt dtype.DataType, length int) (int, error) {
	if length > math.MaxInt/offsetWidth-1 {
		return 0, tooWide(dt, length)
	}
	return (length + 1) * offsetWidth, nil
}

// checkOffsets reports why off is not a run of rows over a child of n elements.
//
// It is O(n) over the rows, and it happens once when a column is built rather
// than per read. The alternative is a bounds check in List, which would be per
// row and would find the same mistake later and with less to say about it.
func checkOffsets(dt dtype.DataType, off []int32, n int) error {
	if off[0] < 0 {
		return fmt.Errorf("array: a %s column starts at offset %d", dt, off[0])
	}
	for i := 1; i < len(off); i++ {
		if off[i] < off[i-1] {
			return fmt.Errorf("array: a %s column has row %d running from %d back to %d",
				dt, i-1, off[i-1], off[i])
		}
	}
	if last := off[len(off)-1]; int(last) > n {
		return fmt.Errorf("array: a %s column ends at offset %d over a child of %d elements",
			dt, last, n)
	}
	return nil
}

// cloneList is Clone for a list column.
//
// The offsets are absolute positions in the shared child, so a copy that holds
// only the rows in range has to rebase them: the child is cut down to the
// elements those rows reach and every offset moves back by where that cut began.
// Copying the offsets as they stand would leave them pointing past the end of a
// child that no longer has the rest of the column in it.
func (a *Array) cloneList() *Array {
	off := a.Offsets()
	start, end := int(off[0]), int(off[a.length])

	buf := buffer.New((a.length + 1) * offsetWidth)
	rebased := offsetsOf(buf, 0, a.length)
	for i, o := range off {
		rebased[i] = o - int32(start)
	}

	out := &Array{
		dt:     a.dt,
		length: a.length,
		nulls:  a.nulls,
		values: buf,
		child:  a.child.Slice(start, end).Clone(),
	}
	if a.validity != nil {
		out.validity = a.validity.Slice(a.offset, a.offset+a.length)
	}
	return out
}
