package ipc_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"testing"
	"unsafe"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
	"github.com/tamnd/kuma/strview"
)

// TestRoundTripArrays exports an array and imports it back. The values have to
// survive and the buffers have to be the same buffers, since an export that
// copies is a correct answer to the wrong question.
func TestRoundTripArrays(t *testing.T) {
	long := "a string too long to live inside a view"

	tests := []struct {
		name string
		a    *array.Array
	}{
		{"null", array.NewNull(5)},
		{"bool", array.OfBools(true, false, true, true)},
		{"int8", array.Of[int8](1, -2, 3)},
		{"int32", array.Of[int32](1, 2, 3, 4, 5)},
		{"int64", array.Of[int64](1, 2, 3)},
		{"uint16", array.Of[uint16](7, 8)},
		{"float64", array.Of(1.5, 2.5, math.Inf(-1))},
		{"empty", array.Of[int64]()},
		{"strings", array.OfStrings("a", "bb", long, "")},
		{"inline strings", array.OfStrings("a", "bb", "ccc")},
		{"empty strings", array.OfStrings()},
		{"binary", buildBinary(t, [][]byte{{1, 2}, bytes.Repeat([]byte{3}, 40)})},
		{"nulls", buildInts(t, []int64{1, 2, 3, 4}, 1, 3)},
		{"string nulls", buildStrings(t, []string{"a", long, "c"}, 0, 2)},
		{"timestamp", fixed(t, dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}, 3)},
		{"decimal", fixed(t, dtype.Decimal128{Precision: 18, Scale: 2}, 2)},
		{"fixed size binary", fixed(t, dtype.FixedSizeBinary{ByteWidth: 4}, 3)},
		{"interval", fixed(t, dtype.Interval{Unit: dtype.MonthDayNano}, 2)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			format, err := ipc.Format(tt.a.DType())
			if err != nil {
				t.Fatalf("Format = %v", err)
			}
			l, err := ipc.Export(tt.a)
			if err != nil {
				t.Fatalf("Export = %v", err)
			}
			if l.Length != tt.a.Len() || l.NullCount != tt.a.NullCount() {
				t.Errorf("Export = length %d and %d nulls, want %d and %d",
					l.Length, l.NullCount, tt.a.Len(), tt.a.NullCount())
			}

			got, err := ipc.Import(format, l)
			if err != nil {
				t.Fatalf("Import = %v", err)
			}
			equalArrays(t, got, tt.a)
		})
	}
}

// TestRoundTripSlice checks that a slice of an array exports as an offset into
// the same buffers rather than as a copy of the part in range, and comes back
// as the same values.
func TestRoundTripSlice(t *testing.T) {
	whole := buildInts(t, []int64{10, 20, 30, 40, 50, 60, 70, 80, 90}, 2, 5)
	a := whole.Slice(3, 7)

	l, err := ipc.Export(a)
	if err != nil {
		t.Fatalf("Export = %v", err)
	}
	if l.Offset != 3 || l.Length != 4 {
		t.Fatalf("Export = offset %d and length %d, want 3 and 4", l.Offset, l.Length)
	}
	if !sameBytes(l.Buffers[1], whole.Buffer().Bytes()) {
		t.Error("Export copied the values of a sliced array")
	}

	got, err := ipc.Import("l", l)
	if err != nil {
		t.Fatalf("Import = %v", err)
	}
	if got.Offset() != 3 {
		t.Errorf("Import = offset %d, want 3", got.Offset())
	}
	equalArrays(t, got, a)
}

// TestImportZeroCopy is the point of the whole package. Nothing an import
// returns may be a copy of anything it was given, apart from the views it has
// to build for a column that arrived with offsets.
func TestImportZeroCopy(t *testing.T) {
	values := make([]byte, 8*4)
	binary.NativeEndian.PutUint64(values, 42)
	valid := []byte{0b00001001}

	got, err := ipc.Import("l", ipc.Layout{Length: 4, Buffers: [][]byte{valid, values}})
	if err != nil {
		t.Fatalf("Import = %v", err)
	}
	if !sameBytes(got.Buffer().Bytes(), values) {
		t.Error("the values were copied")
	}
	if !sameBytes(got.Validity().Bytes(), valid) {
		t.Error("the validity bitmap was copied")
	}
	if got.NullCount() != 2 {
		t.Errorf("NullCount = %d, want 2", got.NullCount())
	}

	// The view layout is what kuma already stores, so the views themselves are
	// borrowed as well.
	src := array.OfStrings("a", "a string too long to live inside a view")
	l, err := ipc.Export(src)
	if err != nil {
		t.Fatalf("Export = %v", err)
	}
	str, err := ipc.Import("vu", l)
	if err != nil {
		t.Fatalf("Import = %v", err)
	}
	if !sameBytes(viewBytes(str.Strings().Views()), l.Buffers[1]) {
		t.Error("the views were copied")
	}
	if !sameBytes(str.Strings().Blocks()[0].Bytes(), l.Buffers[2]) {
		t.Error("the data block was copied")
	}
}

// TestImportOffsets covers the layout kuma does not store and nearly every
// other implementation sends. The values are converted into views and the bytes
// they are made of stay where they are.
func TestImportOffsets(t *testing.T) {
	vals := []string{"", "a", "twelve chars", "a string too long to live inside a view", "z"}

	for _, format := range []string{"u", "U", "z", "Z"} {
		t.Run(format, func(t *testing.T) {
			l, data := offsetLayout(vals, format == "U" || format == "Z")
			got, err := ipc.Import(format, l)
			if err != nil {
				t.Fatalf("Import = %v", err)
			}

			want := dtype.String
			if format == "z" || format == "Z" {
				want = dtype.Binary
			}
			if !dtype.Equal(got.DType(), want) {
				t.Errorf("Import = %s, want %s", got.DType(), want)
			}
			if got.Len() != len(vals) {
				t.Fatalf("Len = %d, want %d", got.Len(), len(vals))
			}
			for i, v := range vals {
				if string(got.Bytes(i)) != v {
					t.Errorf("value %d = %q, want %q", i, got.Bytes(i), v)
				}
			}
			if !sameBytes(got.Strings().Blocks()[0].Bytes(), data) {
				t.Error("the data buffer was copied")
			}
		})
	}
}

// TestImportOffsetsBlocks covers the one thing the 64 bit layout does that the
// 32 bit one never has to, which is splitting the data into more than one block
// because an offset inside a block is a signed 32 bit number and the data can
// be larger than that. Two gigabytes of test data would be absurd, so the size
// a block splits at is lowered instead and the arithmetic is the same.
func TestImportOffsetsBlocks(t *testing.T) {
	defer func(old int64) { *ipc.MaxBlock = old }(*ipc.MaxBlock)
	*ipc.MaxBlock = 16

	vals := []string{"a value of twenty by", "another twenty bytes", "and one more of the "}
	l, data := offsetLayout(vals, true)

	got, err := ipc.Import("U", l)
	if err != nil {
		t.Fatalf("Import = %v", err)
	}
	for i, v := range vals {
		if string(got.Bytes(i)) != v {
			t.Errorf("value %d = %q, want %q", i, got.Bytes(i), v)
		}
	}
	blocks := got.Strings().Blocks()
	if len(blocks) != 3 {
		t.Fatalf("the data went into %d blocks, want 3", len(blocks))
	}
	if !sameBytes(blocks[0].Bytes(), data) {
		t.Error("the data was copied into its blocks rather than sliced")
	}
}

func TestImportErrors(t *testing.T) {
	values := make([]byte, 128)
	offsets := make([]byte, 4*4)

	// Nine values with a validity bitmap of one byte, which covers eight.
	shortValidity, _ := offsetLayout([]string{"a", "b", "c", "d", "e", "f", "g", "h", "i"}, false)
	shortValidity.Buffers[0] = []byte{0xFF}

	tests := []struct {
		name   string
		format string
		layout ipc.Layout
		want   error
	}{
		{name: "negative length", format: "l", layout: ipc.Layout{Length: -1}, want: ipc.ErrBuffers},
		{name: "negative offset", format: "l", layout: ipc.Layout{Offset: -1}, want: ipc.ErrBuffers},
		{name: "no buffers", format: "l", want: ipc.ErrBuffers},
		{
			name: "too many buffers", format: "l",
			layout: ipc.Layout{Length: 1, Buffers: [][]byte{nil, values, values}},
			want:   ipc.ErrBuffers,
		},
		{
			name: "values too short", format: "l",
			layout: ipc.Layout{Length: 100, Buffers: [][]byte{nil, values}},
		},
		{
			name: "validity too short", format: "l",
			layout: ipc.Layout{Length: 9, Buffers: [][]byte{{0xFF}, values}},
			want:   ipc.ErrBuffers,
		},
		{
			name: "offset past the values", format: "l",
			layout: ipc.Layout{Length: 4, Offset: 60, Buffers: [][]byte{nil, values}},
		},
		{
			name: "views too short", format: "vu",
			layout: ipc.Layout{Length: 5, Buffers: [][]byte{nil, values[:16]}},
			want:   ipc.ErrBuffers,
		},
		{
			name: "string validity too short", format: "u", layout: shortValidity,
			want: ipc.ErrBuffers,
		},
		{
			name: "offsets too short", format: "u",
			layout: ipc.Layout{Length: 4, Buffers: [][]byte{nil, offsets, nil}},
			want:   ipc.ErrBuffers,
		},
		{
			name: "offsets with no data buffer", format: "u",
			layout: ipc.Layout{Length: 1, Buffers: [][]byte{nil, offsets}},
			want:   ipc.ErrBuffers,
		},

		// A producer that disagrees with itself about where a value is has to
		// be stopped here. Past this point the offset is a slice index and a
		// wrong one reads somebody else's memory.
		{
			name: "offsets past the data", format: "u",
			layout: ipc.Layout{Length: 1, Buffers: [][]byte{nil, badOffsets(0, 99), nil}},
			want:   ipc.ErrBuffers,
		},
		{
			name: "offsets going backwards", format: "u",
			layout: ipc.Layout{Length: 1, Buffers: [][]byte{nil, badOffsets(8, 4), values}},
			want:   ipc.ErrBuffers,
		},
		{
			name: "negative offset in the offsets", format: "u",
			layout: ipc.Layout{Length: 1, Buffers: [][]byte{nil, badOffsets(-4, 4), values}},
			want:   ipc.ErrBuffers,
		},
		{
			name: "a view naming a block that is not there", format: "vu",
			layout: ipc.Layout{Length: 1, Buffers: [][]byte{nil, viewBytes([]strview.View{
				strview.MakeRef(bytes.Repeat([]byte("x"), 20), 3, 0),
			})}},
		},

		// The types this package understands and cannot build a value of yet
		// say so in those words, and the ones with no kuma type at all say that
		// instead, because the two mean different things to whoever reads it.
		{name: "list", format: "+l", want: ipc.ErrUnsupported},
		{name: "struct", format: "+s", want: ipc.ErrUnsupported},
		{name: "map", format: "+m", want: ipc.ErrUnsupported},
		{name: "fixed size list", format: "+w:3", want: ipc.ErrUnsupported},
		{name: "union", format: "+ud:0,1", want: ipc.ErrFormat},
		{name: "list view", format: "+vl", want: ipc.ErrFormat},
		{name: "float16", format: "e", want: ipc.ErrFormat},
		{name: "not a format string", format: "int64", want: ipc.ErrFormat},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ipc.Import(tt.format, tt.layout)
			if err == nil {
				t.Fatalf("Import(%q) = %s, want an error", tt.format, got)
			}
			if got != nil {
				t.Errorf("Import(%q) returned an array with an error", tt.format)
			}
			if tt.want != nil && !errors.Is(err, tt.want) {
				t.Errorf("Import(%q) = %v, want %v", tt.format, err, tt.want)
			}
		})
	}
}

// TestImportOverflow checks the lengths that come out small when they are
// multiplied by the width of a value.
//
// A length arrives from another process, and every layout here turns one into a
// number of bytes before comparing it with a buffer. Two to the sixty first
// values of eight bytes each is exactly eight bytes once the multiplication has
// wrapped, so a buffer of any size at all looks big enough, and what follows is
// a slice header claiming most of the address space. None of these is a
// plausible column and all of them are two bytes of a message changed.
func TestImportOverflow(t *testing.T) {
	buf := make([]byte, 128)

	for _, tt := range []struct {
		name   string
		format string
		layout ipc.Layout
	}{
		{
			name: "a total past an int", format: "l",
			layout: ipc.Layout{Length: math.MaxInt, Offset: 1, Buffers: [][]byte{nil, buf}},
		},
		{
			name: "values of eight bytes", format: "l",
			layout: ipc.Layout{Length: 1<<61 + 1, Buffers: [][]byte{nil, buf}},
		},
		{
			name: "bits of a bool", format: "b",
			layout: ipc.Layout{Length: math.MaxInt, Buffers: [][]byte{nil, buf}},
		},
		{
			name: "views of sixteen bytes", format: "vu",
			layout: ipc.Layout{Length: 1<<60 + 3, Buffers: [][]byte{nil, buf}},
		},
		{
			name: "offsets of four bytes", format: "u",
			layout: ipc.Layout{Length: 1<<62 + 1, Buffers: [][]byte{nil, buf, nil}},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ipc.Import(tt.format, tt.layout)
			if err == nil {
				t.Fatalf("Import(%q) = %d values, want an error", tt.format, got.Len())
			}
		})
	}
}

// TestImportNullBuffers pins down the one place implementations disagree. A
// Null array carries no values, and whether it also carries a validity buffer
// nobody reads is not worth refusing a table over.
func TestImportNullBuffers(t *testing.T) {
	for _, l := range []ipc.Layout{
		{Length: 3},
		{Length: 3, Buffers: [][]byte{nil}},
		{Length: 3, Buffers: [][]byte{{0xFF}}},
	} {
		got, err := ipc.Import("n", l)
		if err != nil {
			t.Fatalf("Import = %v", err)
		}
		if got.Len() != 3 || got.NullCount() != 3 {
			t.Errorf("Import = %d values and %d nulls, want 3 and 3", got.Len(), got.NullCount())
		}
	}
}

// TestImportIgnoresNullCount checks that a count of minus one, which is what a
// producer sends when it has not worked one out, does not become the answer.
func TestImportIgnoresNullCount(t *testing.T) {
	values := make([]byte, 32)
	l := ipc.Layout{Length: 4, NullCount: -1, Buffers: [][]byte{{0b00000101}, values}}

	got, err := ipc.Import("l", l)
	if err != nil {
		t.Fatalf("Import = %v", err)
	}
	if got.NullCount() != 2 {
		t.Errorf("NullCount = %d, want 2", got.NullCount())
	}
}

func TestExportErrors(t *testing.T) {
	if _, err := ipc.Export(nil); err == nil {
		t.Error("Export(nil) = no error")
	}
}

// TestExportEmptyValidity checks that an array with no nulls exports a validity
// buffer of no bytes, which is the null pointer the C data interface expects,
// rather than a bitmap of ones that a consumer would have to read.
func TestExportEmptyValidity(t *testing.T) {
	l, err := ipc.Export(array.Of[int64](1, 2, 3))
	if err != nil {
		t.Fatalf("Export = %v", err)
	}
	if l.Buffers[0] != nil {
		t.Errorf("Export = a validity buffer of %d bytes, want none", len(l.Buffers[0]))
	}
}

func buildInts(t *testing.T, vals []int64, nulls ...int) *array.Array {
	t.Helper()
	b, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		t.Fatalf("NewBuilder = %v", err)
	}
	for i, v := range vals {
		if isNull(i, nulls) {
			b.AppendNull()
			continue
		}
		b.Append(v)
	}
	return b.Finish()
}

func buildStrings(t *testing.T, vals []string, nulls ...int) *array.Array {
	t.Helper()
	b, err := array.NewBuilder(dtype.String)
	if err != nil {
		t.Fatalf("NewBuilder = %v", err)
	}
	for i, v := range vals {
		if isNull(i, nulls) {
			b.AppendNull()
			continue
		}
		b.AppendString(v)
	}
	return b.Finish()
}

func buildBinary(t *testing.T, vals [][]byte) *array.Array {
	t.Helper()
	b, err := array.NewBuilder(dtype.Binary)
	if err != nil {
		t.Fatalf("NewBuilder = %v", err)
	}
	for _, v := range vals {
		b.AppendBytes(v)
	}
	return b.Finish()
}

func isNull(i int, nulls []int) bool {
	for _, n := range nulls {
		if n == i {
			return true
		}
	}
	return false
}

// fixed builds an array of a type whose values are bytes nobody in this file
// interprets, so the bytes are a counter and the test only asks that they come
// back.
func fixed(t *testing.T, dt dtype.DataType, n int) *array.Array {
	t.Helper()
	bits, ok := dtype.Bits(dt)
	if !ok {
		t.Fatalf("%s has no width", dt)
	}
	buf := buffer.New(n * bits / 8)
	for i := range buf.Bytes() {
		buf.Bytes()[i] = byte(i + 1)
	}
	a, err := array.New(dt, n, buf, nil)
	if err != nil {
		t.Fatalf("New = %v", err)
	}
	return a
}

// offsetLayout writes vals in the offset layout, in the 32 bit form or the 64
// bit one, and returns the layout along with the data buffer so that a test can
// check the import did not copy it.
func offsetLayout(vals []string, wide bool) (ipc.Layout, []byte) {
	width := 4
	if wide {
		width = 8
	}
	offsets := make([]byte, 0, (len(vals)+1)*width)
	var data []byte
	put := func(n int) {
		if wide {
			offsets = binary.NativeEndian.AppendUint64(offsets, uint64(n))
			return
		}
		offsets = binary.NativeEndian.AppendUint32(offsets, uint32(n))
	}

	put(0)
	for _, v := range vals {
		data = append(data, v...)
		put(len(data))
	}
	return ipc.Layout{Length: len(vals), Buffers: [][]byte{nil, offsets, data}}, data
}

// badOffsets writes one pair of 32 bit offsets, whatever they say.
func badOffsets(lo, hi int32) []byte {
	b := binary.NativeEndian.AppendUint32(nil, uint32(lo))
	return binary.NativeEndian.AppendUint32(b, uint32(hi))
}

// viewBytes is the export side reinterpretation, written again here so that a
// test can build a buffer of views to hand to Import.
func viewBytes(views []strview.View) []byte {
	if len(views) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(views))), len(views)*strview.Size)
}

// sameBytes reports whether two slices are the same memory rather than the same
// contents, which is what proves nothing was copied.
func sameBytes(a, b []byte) bool {
	if len(a) == 0 || len(b) == 0 {
		return len(a) == len(b)
	}
	return unsafe.SliceData(a) == unsafe.SliceData(b)
}

func equalArrays(t *testing.T, got, want *array.Array) {
	t.Helper()
	if !dtype.Equal(got.DType(), want.DType()) {
		t.Fatalf("type = %s, want %s", got.DType(), want.DType())
	}
	if got.Len() != want.Len() {
		t.Fatalf("length = %d, want %d", got.Len(), want.Len())
	}
	if got.NullCount() != want.NullCount() {
		t.Errorf("null count = %d, want %d", got.NullCount(), want.NullCount())
	}
	for i := range want.Len() {
		if got.IsNull(i) != want.IsNull(i) {
			t.Errorf("value %d is null %v, want %v", i, got.IsNull(i), want.IsNull(i))
			continue
		}
		if want.IsNull(i) {
			continue
		}
		if a, b := valueAt(got, i), valueAt(want, i); !bytes.Equal(a, b) {
			t.Errorf("value %d = %x, want %x", i, a, b)
		}
	}
}

// valueAt returns the bytes of one value, whatever the type means by them.
func valueAt(a *array.Array, i int) []byte {
	switch a.DType().Kind() {
	case dtype.BoolKind:
		if a.Bool(i) {
			return []byte{1}
		}
		return []byte{0}
	case dtype.StringKind, dtype.BinaryKind:
		return a.Bytes(i)
	default:
		bits, _ := dtype.Bits(a.DType())
		w := bits / 8
		k := (a.Offset() + i) * w
		return a.Buffer().Bytes()[k : k+w]
	}
}
