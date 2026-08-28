package ipc

import (
	"errors"
	"fmt"
	"strings"
	"unsafe"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/strview"
)

// Layout is one array the way the C data interface describes it, with the
// pointers turned into slices. The type is not in here because the type travels
// separately, in a schema that usually covers a whole table.
//
// Buffers is in the order the format string implies, which is the order the C
// struct has them in: the validity bitmap first, then the values. A validity
// buffer of no bytes means every value is present, which is how a null pointer
// arrives. A Null array has no buffers at all. The three text and byte layouts
// have a third buffer, and the view layout has one buffer per data block after
// the views, so a column of strings can be any number of buffers long.
//
// The buffer of buffer sizes that the C ABI appends to the view layout is not
// here. It is there so that a consumer reading the C struct knows how long each
// data block is, and a Go slice already knows, so the cgo layer builds it on the
// way out and drops it on the way in.
type Layout struct {
	// Length is the number of values.
	Length int

	// Offset is how many values at the front of the buffers are not part of
	// this array. It is in values rather than bytes, and for a Bool column
	// that means bits.
	Offset int

	// NullCount is how many of the values are missing. Export fills it in.
	// Import ignores it and counts for itself, since kuma keeps the count on
	// the array and a producer is allowed to say that it does not know.
	NullCount int

	// Buffers are the raw bytes, borrowed rather than owned.
	Buffers [][]byte
}

// Export describes a kuma array in C data interface terms.
//
// Nothing is copied. The slices in the result point into the array's own
// buffers, so they stay valid for as long as the array does and no longer, and
// a caller that hands them to another library has to keep the array alive until
// that library is finished with them.
//
// The nested types are not here, because the array package has no nested arrays
// to export yet.
func Export(a *array.Array) (Layout, error) {
	if a == nil {
		return Layout{}, errors.New("ipc: nil array")
	}

	l := Layout{Length: a.Len(), Offset: a.Offset(), NullCount: a.NullCount()}
	t := a.DType()
	switch t.Kind() {
	case dtype.NullKind:
		// A Null array is all of its information: every value is missing and
		// there is nothing to point at.
		return l, nil

	case dtype.StringKind, dtype.BinaryKind:
		d := a.Strings()
		if d == nil {
			return Layout{}, fmt.Errorf("ipc: %w: a %s array with no values", ErrBuffers, t)
		}
		l.Buffers = make([][]byte, 0, 2+len(d.Blocks()))
		l.Buffers = append(l.Buffers, validityBytes(a), viewBytes(d.Views()))
		for _, b := range d.Blocks() {
			l.Buffers = append(l.Buffers, b.Bytes())
		}
		return l, nil
	}

	if _, ok := dtype.Bits(t); !ok {
		return Layout{}, fmt.Errorf("ipc: %w: exporting a %s array", ErrUnsupported, t)
	}
	if a.Buffer() == nil {
		return Layout{}, fmt.Errorf("ipc: %w: a %s array with no values", ErrBuffers, t)
	}
	l.Buffers = [][]byte{validityBytes(a), a.Buffer().Bytes()}
	return l, nil
}

// Import builds a kuma array out of the buffers of an incoming one.
//
// It takes the format string rather than the type, because the format string is
// what says which layout the buffers are in. The three text layouts all become
// one kuma type and only the format string tells them apart, so a caller that
// has already called Type has thrown that away.
//
// Nothing is copied except the views of a column that arrived in one of the
// offset layouts, which are sixteen bytes per value and have to be built
// because kuma has no offsets. The values themselves are still borrowed, in
// that case and in every other. That means the array shares the producer's
// memory, so whatever release callback comes with those buffers must not run
// until the array is gone.
//
// The null count in the layout is ignored and the nulls are counted from the
// bitmap, since a producer is allowed to send a count of minus one for a count
// it has not worked out.
//
// A dictionary encoded column does not arrive here as one. Its format string
// names the index type, so this builds the indices, and the caller that read
// the dictionary member of the schema is the one that knows better.
func Import(format string, l Layout) (*array.Array, error) {
	if l.Length < 0 || l.Offset < 0 {
		return nil, fmt.Errorf("ipc: %w: length %d and offset %d", ErrBuffers, l.Length, l.Offset)
	}

	switch format {
	case "u":
		return importOffsets(format, dtype.String, l, 4)
	case "z":
		return importOffsets(format, dtype.Binary, l, 4)
	case "U":
		return importOffsets(format, dtype.String, l, 8)
	case "Z":
		return importOffsets(format, dtype.Binary, l, 8)
	case "vu":
		return importViews(format, dtype.String, l)
	case "vz":
		return importViews(format, dtype.Binary, l)
	}

	if strings.HasPrefix(format, "+") {
		// Type knows which of the nested format strings name something kuma has
		// no type for at all, such as a union, and says so in its own words.
		// The rest are types kuma has and cannot hold values of yet.
		if _, err := Type(format, nil); errors.Is(err, ErrFormat) {
			return nil, err
		}
		return nil, fmt.Errorf("ipc: %w: %q is a nested array, which this package cannot build yet",
			ErrUnsupported, format)
	}

	t, err := Type(format, nil)
	if err != nil {
		return nil, err
	}
	if t.Kind() == dtype.NullKind {
		// Implementations disagree about whether a Null array carries a
		// validity buffer nobody reads, so anything it arrives with is fine and
		// none of it is looked at.
		return array.NewNull(l.Length), nil
	}
	return importFixed(t, l)
}

// importFixed builds an array of a type whose values are a fixed number of
// bytes each, which is everything except the text and byte layouts.
func importFixed(t dtype.DataType, l Layout) (*array.Array, error) {
	if len(l.Buffers) != 2 {
		return nil, fmt.Errorf("ipc: %w: a %s array has 2 buffers, have %d", ErrBuffers, t, len(l.Buffers))
	}
	total := l.Offset + l.Length
	valid, err := validityOf(l.Buffers[0], total)
	if err != nil {
		return nil, err
	}
	a, err := array.New(t, total, buffer.Wrap(l.Buffers[1]), valid)
	if err != nil {
		return nil, fmt.Errorf("ipc: %w", err)
	}
	return a.Slice(l.Offset, total), nil
}

// importViews builds a String or Binary array from the layout kuma already
// stores, which is the one case where the values need no conversion at all.
func importViews(format string, t dtype.DataType, l Layout) (*array.Array, error) {
	if len(l.Buffers) < 2 {
		return nil, fmt.Errorf("ipc: %w: the %q layout has at least 2 buffers, have %d",
			ErrBuffers, format, len(l.Buffers))
	}
	total := l.Offset + l.Length
	if need := total * strview.Size; len(l.Buffers[1]) < need {
		return nil, fmt.Errorf("ipc: %w: %d views need %d bytes, the buffer has %d",
			ErrBuffers, total, need, len(l.Buffers[1]))
	}

	var views []strview.View
	if total > 0 {
		views = unsafe.Slice((*strview.View)(unsafe.Pointer(unsafe.SliceData(l.Buffers[1]))), total)
	}
	blocks := make([]*buffer.Buffer, 0, len(l.Buffers)-2)
	for _, b := range l.Buffers[2:] {
		blocks = append(blocks, buffer.Wrap(b))
	}

	// NewData checks every view against the blocks, which is the whole reason
	// this is cheap enough to do at the boundary: after it, no read of a value
	// has to wonder whether the offset in front of it is real.
	d, err := strview.NewData(views, blocks)
	if err != nil {
		return nil, fmt.Errorf("ipc: %w", err)
	}
	return finishStrings(t, l, d)
}

// importOffsets builds a String or Binary array from one of the two offset
// layouts, which is what almost every other implementation sends. The width is
// 4 for "u" and "z" and 8 for "U" and "Z".
func importOffsets(format string, t dtype.DataType, l Layout, width int) (*array.Array, error) {
	if len(l.Buffers) != 3 {
		return nil, fmt.Errorf("ipc: %w: the %q layout has 3 buffers, have %d",
			ErrBuffers, format, len(l.Buffers))
	}
	total := l.Offset + l.Length

	// The values in front of the offset are converted along with the rest. The
	// array keeps the offset instead of dropping it, so that the validity
	// bitmap can be shared rather than shifted, and a bitmap that starts part
	// way through a byte cannot be shared.
	d, err := viewsFromOffsets(l.Buffers[1], l.Buffers[2], total, width)
	if err != nil {
		return nil, err
	}
	return finishStrings(t, l, d)
}

// finishStrings puts the validity on a String or Binary array and cuts it down
// to the range the layout asks for.
func finishStrings(t dtype.DataType, l Layout, d *strview.Data) (*array.Array, error) {
	total := l.Offset + l.Length
	valid, err := validityOf(l.Buffers[0], total)
	if err != nil {
		return nil, err
	}
	a, err := array.NewStrings(t, d, valid)
	if err != nil {
		return nil, fmt.Errorf("ipc: %w", err)
	}
	return a.Slice(l.Offset, total), nil
}

// validityOf wraps an incoming validity buffer. No bytes means no nulls, which
// is what a null pointer arrives as.
func validityOf(b []byte, n int) (*bitmap.Bitmap, error) {
	if len(b) == 0 {
		return nil, nil
	}
	if len(b)*8 < n {
		return nil, fmt.Errorf("ipc: %w: a validity bitmap of %d bytes covers %d values, need %d",
			ErrBuffers, len(b), len(b)*8, n)
	}
	return bitmap.FromBytes(b, n), nil
}

// validityBytes is the bitmap of a to export, or no bytes when it has no nulls.
func validityBytes(a *array.Array) []byte {
	if v := a.Validity(); v != nil {
		return v.Bytes()
	}
	return nil
}

// viewBytes reads a slice of views as the bytes it already is. A view is a byte
// array of a fixed size and not a struct of fields, exactly so that this is a
// reinterpretation and not a rewrite.
func viewBytes(views []strview.View) []byte {
	if len(views) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(views))), len(views)*strview.Size)
}
