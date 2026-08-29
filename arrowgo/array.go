package arrowgo

import (
	"fmt"
	"unsafe"

	"github.com/apache/arrow-go/v18/arrow"
	arrowarray "github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/strview"
)

// ImportArray returns a kuma column holding the values of an arrow-go one.
//
// The bytes are shared rather than copied for every layout the two sides agree
// on, which is all of the fixed width types, the validity bitmap and a column
// already in the view layout. A view column is walked once to check that every
// view points inside a block it names, since a view that does not is a read of
// memory that is not ours, but nothing is allocated per value and nothing is
// moved.
//
// An offset layout string or binary column is the one that copies, because kuma
// has no offset layout to put it in. So does a large_string or large_binary
// one, for the same reason.
//
// The arrow-go array has to stay alive for as long as the column returned here
// is used. See the note on memory in the package comment.
func ImportArray(a arrow.Array) (*array.Array, error) {
	if a == nil {
		return nil, fmt.Errorf("arrowgo: nil arrow array")
	}

	if d, ok := a.(*arrowarray.Dictionary); ok {
		return importDict(d)
	}

	dt, err := ImportType(a.DataType())
	if err != nil {
		return nil, err
	}

	data := a.Data()
	off, n := data.Offset(), data.Len()
	valid, err := importValidity(data, off+n)
	if err != nil {
		return nil, err
	}

	switch a.DataType().ID() {
	case arrow.NULL:
		return array.NewNull(n), nil
	case arrow.STRING, arrow.BINARY, arrow.LARGE_STRING, arrow.LARGE_BINARY:
		return importOffsets(a, dt, valid, off, n)
	case arrow.STRING_VIEW, arrow.BINARY_VIEW:
		return importViews(data, dt, valid, off, n)
	}
	return importFixed(data, dt, valid, off, n)
}

// importFixed wraps the one value buffer of a fixed width column, which is
// every remaining type including bool, where the values are bits rather than
// bytes and the array's offset counts them.
func importFixed(data arrow.ArrayData, dt dtype.DataType,
	valid *bitmap.Bitmap, off, n int) (*array.Array, error) {
	values := bufferBytes(data, 1)
	if values == nil && off+n > 0 {
		return nil, fmt.Errorf("arrowgo: a %s column of %d values arrived with no value buffer", dt, n)
	}

	whole, err := array.New(dt, off+n, buffer.Wrap(values), valid)
	if err != nil {
		return nil, fmt.Errorf("arrowgo: %w", err)
	}
	return whole.Slice(off, off+n), nil
}

// bufferBytes reads one buffer of an arrow-go array, and gives nil when it is
// not there. A column of no values has no buffers at all, which is a column
// rather than a mistake, so the absence is answered rather than reported and
// the caller decides whether a length went with it.
func bufferBytes(data arrow.ArrayData, i int) []byte {
	bufs := data.Buffers()
	if len(bufs) <= i || bufs[i] == nil {
		return nil
	}
	return bufs[i].Bytes()
}

// importViews wraps a column that is already views and blocks, which is the
// layout kuma stores and so the one that crosses whole.
//
// The views are sliced to the range the array covers rather than left whole
// with an offset on the array, since a Go slice header costs nothing and it
// means the blocks are the only thing being shared.
func importViews(data arrow.ArrayData, dt dtype.DataType,
	valid *bitmap.Bitmap, off, n int) (*array.Array, error) {
	p := bufferBytes(data, 1)
	if p == nil && off+n > 0 {
		return nil, fmt.Errorf("arrowgo: a %s column of %d values arrived with no view buffer", dt, n)
	}

	views, err := viewsOf(p, off, n)
	if err != nil {
		return nil, err
	}
	bufs := data.Buffers()
	blocks := make([]*buffer.Buffer, 0, max(len(bufs)-2, 0))
	for _, b := range bufs[min(2, len(bufs)):] {
		if b == nil {
			blocks = append(blocks, buffer.New(0))
			continue
		}
		blocks = append(blocks, buffer.Wrap(b.Bytes()))
	}

	d, err := strview.NewData(views, blocks)
	if err != nil {
		return nil, fmt.Errorf("arrowgo: %w", err)
	}
	out, err := array.NewStrings(dt, d, sliceValidity(valid, off, n))
	if err != nil {
		return nil, fmt.Errorf("arrowgo: %w", err)
	}
	return out, nil
}

// viewsOf reads the view buffer as views without copying it. The buffer is a
// run of sixteen byte headers and [strview.View] is those sixteen bytes, so the
// two are the same memory read two ways.
func viewsOf(p []byte, off, n int) ([]strview.View, error) {
	if len(p) < (off+n)*strview.Size {
		return nil, fmt.Errorf("arrowgo: %d values at offset %d need %d bytes of views, the buffer has %d",
			n, off, (off+n)*strview.Size, len(p))
	}
	if n == 0 {
		return nil, nil
	}
	all := unsafe.Slice((*strview.View)(unsafe.Pointer(&p[0])), off+n)
	return all[off : off+n : off+n], nil
}

// importOffsets converts an offset layout column into the view layout. This is
// the one path that copies, and it copies because there is nowhere else for the
// bytes to go: kuma has no column shape that reads an offsets buffer.
//
// It goes through the arrow-go accessor rather than through the buffers so that
// the four offset layouts are one function, since a thirty two bit offset and a
// sixty four bit one differ only in the width being copied out of. The values
// are read once each and appended, which is what a builder does anyway when the
// data is loaded from a file.
func importOffsets(a arrow.Array, dt dtype.DataType,
	valid *bitmap.Bitmap, off, n int) (*array.Array, error) {
	bytes, ok := a.(interface{ Value(int) []byte })
	if !ok {
		if s, sok := a.(interface{ Value(int) string }); sok {
			bytes = stringValues{s}
		} else {
			return nil, fmt.Errorf("arrowgo: a %s column has no way to read a value out of it", a.DataType())
		}
	}

	var b strview.Builder
	b.Grow(n)
	for i := range n {
		if a.IsNull(i) {
			b.Append(nil)
			continue
		}
		b.Append(bytes.Value(i))
	}

	out, err := array.NewStrings(dt, b.Finish(), sliceValidity(valid, off, n))
	if err != nil {
		return nil, fmt.Errorf("arrowgo: %w", err)
	}
	return out, nil
}

// stringValues reads a utf8 column, whose accessor gives a string where the
// binary one gives bytes. The conversion does not copy, since the bytes are
// appended and never kept.
type stringValues struct {
	a interface{ Value(int) string }
}

func (s stringValues) Value(i int) []byte {
	v := s.a.Value(i)
	if v == "" {
		return nil
	}
	return unsafe.Slice(unsafe.StringData(v), len(v))
}

// importDict carries a dictionary encoded column across as its two halves,
// which is how kuma holds one as well.
func importDict(d *arrowarray.Dictionary) (*array.Array, error) {
	indices, err := ImportArray(d.Indices())
	if err != nil {
		return nil, fmt.Errorf("the indices of a %s: %w", d.DataType(), err)
	}
	values, err := ImportArray(d.Dictionary())
	if err != nil {
		return nil, fmt.Errorf("the values of a %s: %w", d.DataType(), err)
	}

	out, err := array.NewDictionary(indices, values)
	if err != nil {
		return nil, fmt.Errorf("arrowgo: %w", err)
	}
	return out, nil
}

// importValidity wraps the validity bitmap of an arrow-go array, which is its
// first buffer and is absent when the column has no nulls.
//
// The length asked for is the offset plus the length, because the bits before
// the offset belong to the array this one was sliced out of and the bitmap has
// to be long enough to reach past them.
func importValidity(data arrow.ArrayData, bits int) (*bitmap.Bitmap, error) {
	p := bufferBytes(data, 0)
	if p == nil {
		return nil, nil
	}
	if len(p)*8 < bits {
		return nil, fmt.Errorf("arrowgo: a validity bitmap of %d bytes for %d values", len(p), bits)
	}
	return bitmap.FromBytes(p, bits), nil
}

// sliceValidity narrows a bitmap to the range an array covers, for the columns
// that carry no offset of their own because their values were sliced instead.
func sliceValidity(b *bitmap.Bitmap, off, n int) *bitmap.Bitmap {
	if b == nil {
		return nil
	}
	return b.Slice(off, off+n)
}

// ExportArray returns an arrow-go column over the values of a kuma one.
//
// Nothing is copied. The value buffer, the validity bitmap and the blocks
// behind a string column are all handed over as they are, and the array offset
// goes along with them, so exporting a morsel sliced out of a chunk of a
// million rows hands over the same bytes the chunk is using.
//
// The result is backed by memory the Go collector owns, so releasing it frees
// nothing and forgetting to release it leaks nothing.
func ExportArray(a *array.Array) (arrow.Array, error) {
	if a == nil {
		return nil, fmt.Errorf("arrowgo: nil kuma array")
	}

	if a.Dictionary() != nil {
		return exportDict(a)
	}

	dt, err := ExportType(a.DType())
	if err != nil {
		return nil, err
	}
	if a.DType().Kind() == dtype.NullKind {
		return arrowarray.NewNull(a.Len()), nil
	}

	bufs, err := exportBuffers(a)
	if err != nil {
		return nil, err
	}

	data := arrowarray.NewData(dt, a.Len(), bufs, nil, a.NullCount(), a.Offset())
	defer data.Release()
	return arrowarray.MakeFromData(data), nil
}

// exportBuffers lays a kuma column out the way arrow-go reads one: the validity
// bitmap first, then either the values or the views and the blocks behind them.
func exportBuffers(a *array.Array) ([]*memory.Buffer, error) {
	bufs := []*memory.Buffer{nil}
	if v := a.Validity(); v != nil {
		bufs[0] = memory.NewBufferBytes(v.Bytes())
	}

	if d := a.Strings(); d != nil {
		bufs = append(bufs, memory.NewBufferBytes(viewBytes(d.Views())))
		for _, b := range d.Blocks() {
			bufs = append(bufs, memory.NewBufferBytes(b.Bytes()))
		}
		return bufs, nil
	}

	values := a.Buffer()
	if values == nil {
		return nil, fmt.Errorf("arrowgo: a %s column of %d values has no value buffer", a.DType(), a.Len())
	}
	return append(bufs, memory.NewBufferBytes(values.Bytes())), nil
}

// viewBytes reads the views as the bytes arrow-go wants them in, which is the
// same memory the other way round from [viewsOf].
func viewBytes(views []strview.View) []byte {
	if len(views) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&views[0])), len(views)*strview.Size)
}

// exportDict hands over both halves of a dictionary encoded column.
func exportDict(a *array.Array) (arrow.Array, error) {
	dt, err := ExportType(a.DType())
	if err != nil {
		return nil, err
	}
	indices, err := ExportArray(a.Indices())
	if err != nil {
		return nil, fmt.Errorf("the indices of a %s: %w", a.DType(), err)
	}
	values, err := ExportArray(a.Dictionary())
	if err != nil {
		return nil, fmt.Errorf("the values of a %s: %w", a.DType(), err)
	}

	out := arrowarray.NewDictionaryArray(dt, indices, values)
	indices.Release()
	values.Release()
	return out, nil
}
