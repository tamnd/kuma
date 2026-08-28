//go:build cgo

package ipc_test

import (
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"testing"
	"unsafe"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
	"github.com/tamnd/kuma/ipc/ipctest"
)

// TestCRoundTrip exports an array into the C structs and imports it back
// through them. The values have to survive and every buffer has to be the same
// buffer it started as, which is the whole reason this interface exists.
func TestCRoundTrip(t *testing.T) {
	long := "a string too long to live inside a view"

	tests := []struct {
		name string
		a    *array.Array
	}{
		{"int64", array.Of[int64](1, 2, 3)},
		{"bool", array.OfBools(true, false, true)},
		{"float64", array.Of(1.5, 2.5)},
		{"nulls", buildInts(t, []int64{1, 2, 3, 4}, 1, 3)},
		{"empty", array.Of[int64]()},
		{"null", array.NewNull(5)},
		{"strings", array.OfStrings("a", "bb", long)},
		{"inline strings", array.OfStrings("a", "bb")},
		{"empty strings", array.OfStrings()},
		{"string nulls", buildStrings(t, []string{"a", long, "c"}, 0)},
		{"sliced", buildInts(t, []int64{10, 20, 30, 40, 50}, 2).Slice(1, 4)},
		{"timestamp", fixed(t, dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}, 3)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			schema, values := ipctest.Pair(t)
			field := dtype.Field{Name: "col", Type: tt.a.DType(), Nullable: true}
			if err := ipc.ExportField(field, schema); err != nil {
				t.Fatalf("ExportField = %v", err)
			}
			if err := ipc.ExportArray(tt.a, values); err != nil {
				t.Fatalf("ExportArray = %v", err)
			}

			if ipctest.Length(values) != tt.a.Len() || ipctest.Offset(values) != tt.a.Offset() {
				t.Errorf("exported length %d and offset %d, want %d and %d",
					ipctest.Length(values), ipctest.Offset(values), tt.a.Len(), tt.a.Offset())
			}
			want := arrayPointers(tt.a)
			exported := ipctest.Buffers(values)
			if len(exported) < len(want) {
				t.Fatalf("exported %d buffers, want at least %d", len(exported), len(want))
			}
			for i, p := range want {
				if exported[i] != p {
					t.Errorf("exported buffer %d is not the buffer it came from", i)
				}
			}

			imported, err := ipc.ImportArray(schema, values)
			if err != nil {
				t.Fatalf("ImportArray = %v", err)
			}
			if !imported.Field.Equal(field) {
				t.Errorf("imported field = %v, want %v", imported.Field, field)
			}
			equalArrays(t, imported.Array, tt.a)
			for i, p := range arrayPointers(imported.Array) {
				if p != want[i] {
					t.Errorf("imported buffer %d is a copy", i)
				}
			}

			// The import releases the schema and Release releases the array.
			// Releasing twice is allowed and does nothing the second time,
			// which is what a deferred call after an early return looks like.
			if !ipctest.SchemaReleased(schema) {
				t.Error("the schema was not released by the import")
			}
			imported.Release()
			if !ipctest.ArrayReleased(values) {
				t.Error("the array was not released")
			}
			imported.Release()
		})
	}
}

// TestCRoundTripFields covers the types whose schema says more than a format
// string does, which is the nested ones and the dictionary. None of them have
// values that can be exported yet, and all of them have a schema that has to
// travel before a reader can decide what to do about that.
func TestCRoundTripFields(t *testing.T) {
	tests := []dtype.Field{
		{Name: "n", Type: dtype.Int64},
		{Name: "nullable", Type: dtype.Float64, Nullable: true},
		{Name: "", Type: dtype.String, Nullable: true},
		{Name: "when", Type: dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "Europe/Berlin"}},
		{Name: "price", Type: dtype.Decimal128{Precision: 18, Scale: 2}},
		{Name: "tags", Type: dtype.List{Elem: dtype.String}, Nullable: true},
		{Name: "big", Type: dtype.LargeList{Elem: dtype.Int32}},
		{Name: "point", Type: dtype.FixedSizeList{Elem: dtype.Float32, Len: 3}},
		{Name: "meta", Type: dtype.Map{Key: dtype.String, Value: dtype.Int64}},
		{Name: "country", Type: dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String}},
		{
			Name: "row",
			Type: dtype.Struct{Fields: []dtype.Field{
				{Name: "id", Type: dtype.Int64},
				{Name: "name", Type: dtype.String, Nullable: true},
				{Name: "inner", Type: dtype.List{Elem: dtype.Struct{Fields: []dtype.Field{
					{Name: "x", Type: dtype.Float64, Nullable: true},
				}}}, Nullable: true},
			}},
		},
		{
			Name:     "documented",
			Type:     dtype.Int32,
			Metadata: dtype.Metadata{{Key: "unit", Value: "cm"}, {Key: "note", Value: ""}},
		},
	}

	for _, want := range tests {
		t.Run(want.Type.String(), func(t *testing.T) {
			schema, _ := ipctest.Pair(t)
			if err := ipc.ExportField(want, schema); err != nil {
				t.Fatalf("ExportField = %v", err)
			}
			format, err := ipc.Format(want.Type)
			if err != nil {
				t.Fatalf("Format = %v", err)
			}
			if got := ipctest.Format(schema); got != format {
				t.Errorf("exported format = %q, want %q", got, format)
			}

			got, err := ipc.ImportField(schema)
			if err != nil {
				t.Fatalf("ImportField = %v", err)
			}
			if !got.Equal(want) {
				t.Errorf("ImportField = %v, want %v", got, want)
			}
			if !ipctest.SchemaReleased(schema) {
				t.Error("the schema was not released")
			}
		})
	}
}

// TestCImportOffsets imports a column in the layout kuma does not store, out of
// memory a producer that is not kuma allocated. The length of the data buffer
// is nowhere in the C struct, so this is also the test that the last offset is
// read and believed.
func TestCImportOffsets(t *testing.T) {
	vals := []string{"", "one", "a string too long to live inside a view", "four"}

	for _, format := range []string{"u", "U", "z", "Z"} {
		t.Run(format, func(t *testing.T) {
			before := ipctest.Releases()
			offsets, data := offsetBuffers(vals, format == "U" || format == "Z")

			schema := ipctest.Schema(t, format)
			values := ipctest.Array(t, len(vals), [][]byte{nil, offsets, data})
			imported, err := ipc.ImportArray(schema, values)
			if err != nil {
				t.Fatalf("ImportArray = %v", err)
			}
			for i, want := range vals {
				if got := string(imported.Array.Bytes(i)); got != want {
					t.Errorf("value %d = %q, want %q", i, got, want)
				}
			}

			imported.Release()
			if got := ipctest.Releases() - before; got != 2 {
				t.Errorf("the producer saw %d releases, want 2", got)
			}
		})
	}
}

// TestCImportErrors checks that a struct this package refuses is still released
// on the way out. A consumer that takes a struct owns it, and one that says no
// and keeps it is a leak inside a library that never sees the allocation.
//
// Everything here is something the C struct is wrong about on its own terms. A
// producer that writes a buffer shorter than its own length is not in the list,
// because the struct has no buffer sizes in it and there is nothing to compare
// against. See importLayout.
func TestCImportErrors(t *testing.T) {
	tests := []struct {
		name     string
		format   string
		length   int
		children int
		buffers  [][]byte
		want     error
	}{
		{name: "unknown format", format: "nonsense", length: 1,
			buffers: [][]byte{nil, {1}}, want: ipc.ErrFormat},
		{name: "union", format: "+us:0,1", length: 1,
			buffers: [][]byte{nil, {1}}, want: ipc.ErrFormat},
		{name: "list with no child", format: "+l", length: 1,
			buffers: [][]byte{nil, make([]byte, 8)}, want: ipc.ErrChildren},
		{name: "children", format: "l", length: 1, children: 1,
			buffers: [][]byte{nil, make([]byte, 8)}, want: ipc.ErrUnsupported},
		{name: "too few buffers", format: "l", length: 1,
			buffers: [][]byte{{1}}, want: ipc.ErrBuffers},
		{name: "too many buffers", format: "l", length: 1,
			buffers: [][]byte{nil, make([]byte, 8), {2}}, want: ipc.ErrBuffers},
		{name: "no sizes buffer", format: "vu", length: 1,
			buffers: [][]byte{nil, make([]byte, 16)}, want: ipc.ErrBuffers},
		{name: "negative offset", format: "u", length: 1,
			buffers: [][]byte{nil, negativeOffsets(), nil}, want: ipc.ErrBuffers},
		{name: "negative block size", format: "vu", length: 0,
			buffers: [][]byte{nil, nil, {1, 2, 3, 4}, negativeSize()}, want: ipc.ErrBuffers},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			before := ipctest.Releases()
			schema := ipctest.Schema(t, tt.format)
			values := ipctest.Array(t, tt.length, tt.buffers)
			ipctest.SetChildren(values, tt.children)

			if _, err := ipc.ImportArray(schema, values); !errors.Is(err, tt.want) {
				t.Fatalf("ImportArray = %v, want %v", err, tt.want)
			}
			if got := ipctest.Releases() - before; got != 2 {
				t.Errorf("the producer saw %d releases, want 2", got)
			}
		})
	}
}

// TestCNegativeLength covers the numbers a producer should never write. They
// arrive as an int64 and everything downstream is an int, so they are checked
// where they arrive rather than where they overflow.
func TestCNegativeLength(t *testing.T) {
	schema := ipctest.Schema(t, "l")
	values := ipctest.Array(t, 1, [][]byte{nil, make([]byte, 8)})
	ipctest.SetLength(values, -1)

	if _, err := ipc.ImportArray(schema, values); !errors.Is(err, ipc.ErrBuffers) {
		t.Errorf("ImportArray = %v, want %v", err, ipc.ErrBuffers)
	}
}

// TestCReleased covers a struct that has already been given back. It is the
// shape of a double import, which is a use after free in whichever library owns
// the memory, so it has to be an error and not a read.
func TestCReleased(t *testing.T) {
	schema, values := ipctest.Pair(t)
	if _, err := ipc.ImportField(schema); err == nil {
		t.Error("ImportField on a released schema = nil error")
	}
	if _, err := ipc.ImportArray(schema, values); err == nil {
		t.Error("ImportArray on a released schema = nil error")
	}
	if _, err := ipc.ImportField(nil); err == nil {
		t.Error("ImportField(nil) = nil error")
	}
	if _, err := ipc.ImportArray(nil, nil); err == nil {
		t.Error("ImportArray(nil, nil) = nil error")
	}

	var empty *ipc.Imported
	empty.Release()
}

// TestCExportErrors covers what cannot be exported, which is a type with no
// format string and, one level down, a nested type whose child has none. The
// struct is left released either way, so a caller that ignores the error does
// not hand a half built schema to somebody else.
func TestCExportErrors(t *testing.T) {
	schema, values := ipctest.Pair(t)

	bad := dtype.Field{Name: "col", Type: dtype.Time32{Unit: dtype.Nanosecond}}
	if err := ipc.ExportField(bad, schema); !errors.Is(err, ipc.ErrType) {
		t.Errorf("ExportField = %v, want %v", err, ipc.ErrType)
	}
	if !ipctest.SchemaReleased(schema) {
		t.Error("a failed ExportField left the schema allocated")
	}

	nested := dtype.Field{Name: "col", Type: dtype.List{Elem: dtype.Time32{Unit: dtype.Nanosecond}}}
	if err := ipc.ExportField(nested, schema); !errors.Is(err, ipc.ErrType) {
		t.Errorf("ExportField = %v, want %v", err, ipc.ErrType)
	}
	if !ipctest.SchemaReleased(schema) {
		t.Error("a failed ExportField left the children allocated")
	}

	if err := ipc.ExportArray(nil, values); err == nil {
		t.Error("ExportArray(nil) = nil error")
	}
	if err := ipc.ExportField(dtype.Field{Type: dtype.Int64}, nil); err == nil {
		t.Error("ExportField into nil = nil error")
	}
	if err := ipc.ExportArray(array.Of[int64](1), nil); err == nil {
		t.Error("ExportArray into nil = nil error")
	}
}

// TestCExportLifetime exports one array into two sets of structs and releases
// them one at a time. Two exports share one set of buffers, so a release that
// freed or unpinned anything the array still owns would show up here as the
// second consumer reading rubbish.
func TestCExportLifetime(t *testing.T) {
	a := array.OfStrings("a", "a string too long to live inside a view", "c")
	field := dtype.Field{Name: "col", Type: a.DType(), Nullable: true}

	firstSchema, firstValues := ipctest.Pair(t)
	secondSchema, secondValues := ipctest.Pair(t)
	for _, pair := range []struct {
		schema *ipc.CSchema
		values *ipc.CArray
	}{{firstSchema, firstValues}, {secondSchema, secondValues}} {
		if err := ipc.ExportField(field, pair.schema); err != nil {
			t.Fatalf("ExportField = %v", err)
		}
		if err := ipc.ExportArray(a, pair.values); err != nil {
			t.Fatalf("ExportArray = %v", err)
		}
	}

	one, err := ipc.ImportArray(firstSchema, firstValues)
	if err != nil {
		t.Fatalf("ImportArray = %v", err)
	}
	one.Release()

	two, err := ipc.ImportArray(secondSchema, secondValues)
	if err != nil {
		t.Fatalf("ImportArray = %v", err)
	}
	defer two.Release()
	equalArrays(t, two.Array, a)
}

// TestCMetadataLength covers the one thing in a schema that has no length. The
// metadata is a pointer, and the pairs inside it are what say how far it goes.
func TestCMetadataLength(t *testing.T) {
	want := dtype.Metadata{
		{Key: "unit", Value: "cm"},
		{Key: "long", Value: strings.Repeat("x", 300)},
		{Key: "", Value: ""},
	}
	schema, _ := ipctest.Pair(t)
	field := dtype.Field{Name: "col", Type: dtype.Int64, Metadata: want}
	if err := ipc.ExportField(field, schema); err != nil {
		t.Fatalf("ExportField = %v", err)
	}

	got, err := ipc.ImportField(schema)
	if err != nil {
		t.Fatalf("ImportField = %v", err)
	}
	if !got.Metadata.Equal(want) {
		t.Errorf("metadata = %v, want %v", got.Metadata, want)
	}
}

// TestCRelease hands back a pair of structs without importing them, which is
// what a caller does when it exported something it then could not send.
func TestCRelease(t *testing.T) {
	schema, values := ipctest.Pair(t)
	if err := ipc.ExportField(dtype.Field{Name: "col", Type: dtype.Int64}, schema); err != nil {
		t.Fatalf("ExportField = %v", err)
	}
	if err := ipc.ExportArray(array.Of[int64](1, 2, 3), values); err != nil {
		t.Fatalf("ExportArray = %v", err)
	}

	ipc.ReleaseSchema(schema)
	ipc.ReleaseArray(values)
	if !ipctest.SchemaReleased(schema) || !ipctest.ArrayReleased(values) {
		t.Fatal("the structs were not released")
	}

	// Twice is allowed and a null pointer is allowed, so that a caller can
	// defer the call and return early without thinking about it.
	ipc.ReleaseSchema(schema)
	ipc.ReleaseArray(values)
	ipc.ReleaseSchema(nil)
	ipc.ReleaseArray(nil)
}

// TestCReexport imports an array from another library and exports it straight
// back out, which is what a bridge between two libraries that are not kuma
// does. The buffers are C memory on the way in and have to survive being
// written into a C struct on the way out, where the ones a kuma array is
// usually made of are Go memory and are pinned first.
func TestCReexport(t *testing.T) {
	vals := []string{"one", "a string too long to live inside a view", "three"}
	offsets, data := offsetBuffers(vals, false)

	schema := ipctest.Schema(t, "u")
	values := ipctest.Array(t, len(vals), [][]byte{nil, offsets, data})
	imported, err := ipc.ImportArray(schema, values)
	if err != nil {
		t.Fatalf("ImportArray = %v", err)
	}
	defer imported.Release()

	outSchema, outValues := ipctest.Pair(t)
	field := dtype.Field{Name: "col", Type: imported.Array.DType(), Nullable: true}
	if err := ipc.ExportField(field, outSchema); err != nil {
		t.Fatalf("ExportField = %v", err)
	}
	if err := ipc.ExportArray(imported.Array, outValues); err != nil {
		t.Fatalf("ExportArray = %v", err)
	}

	// The data block is the producer's own buffer, so it has to come out as the
	// same address it went in as. The views are the one thing kuma had to build.
	blocks := imported.Array.Strings().Blocks()
	if len(blocks) != 1 {
		t.Fatalf("the import made %d data blocks, want 1", len(blocks))
	}
	if got := ipctest.Buffers(outValues)[2]; got != start(blocks[0].Bytes()) {
		t.Error("the data block was copied on the way out")
	}

	back, err := ipc.ImportArray(outSchema, outValues)
	if err != nil {
		t.Fatalf("ImportArray = %v", err)
	}
	defer back.Release()
	equalArrays(t, back.Array, imported.Array)
}

// TestCMetadataErrors covers a metadata blob that describes something that is
// not there. The lengths inside it are what say how long it is, so a blob that
// writes a number no encoder would is the one shape of nonsense that can be
// caught before anything is read.
func TestCMetadataErrors(t *testing.T) {
	tests := []struct {
		name string
		blob []byte
	}{
		{"negative pair count", binary.NativeEndian.AppendUint32(nil, ^uint32(0))},
		{"impossible key length", binary.NativeEndian.AppendUint32(
			binary.NativeEndian.AppendUint32(nil, 1), math.MaxInt32)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			schema := ipctest.Schema(t, "l")
			ipctest.SetMetadata(schema, tt.blob)

			if _, err := ipc.ImportField(schema); !errors.Is(err, ipc.ErrMetadata) {
				t.Errorf("ImportField = %v, want %v", err, ipc.ErrMetadata)
			}
			if !ipctest.SchemaReleased(schema) {
				t.Error("the schema was not released")
			}
		})
	}
}

// arrayPointers is where each buffer of a kuma array starts, in the order the C
// data interface puts them in. A validity bitmap that is not there is a null
// pointer, which is what an array with no nulls exports.
func arrayPointers(a *array.Array) []unsafe.Pointer {
	if a.DType().Kind() == dtype.NullKind {
		return nil
	}
	ptrs := []unsafe.Pointer{nil}
	if v := a.Validity(); v != nil {
		ptrs[0] = start(v.Bytes())
	}
	switch a.DType().Kind() {
	case dtype.StringKind, dtype.BinaryKind:
		d := a.Strings()
		ptrs = append(ptrs, start(viewBytes(d.Views())))
		for _, b := range d.Blocks() {
			ptrs = append(ptrs, start(b.Bytes()))
		}
	default:
		ptrs = append(ptrs, start(a.Buffer().Bytes()))
	}
	return ptrs
}

func start(b []byte) unsafe.Pointer {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Pointer(unsafe.SliceData(b))
}

// negativeOffsets and negativeSize are numbers a producer cannot mean. They are
// checked because the alternative to checking them is a slice of that length.
func negativeOffsets() []byte {
	return binary.NativeEndian.AppendUint32(binary.NativeEndian.AppendUint32(nil, 0), ^uint32(0))
}

func negativeSize() []byte {
	return binary.NativeEndian.AppendUint64(nil, ^uint64(0))
}

// offsetBuffers writes values in one of the two offset layouts, which is what
// nearly every other implementation sends and what kuma has to convert.
func offsetBuffers(vals []string, wide bool) (offsets, data []byte) {
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
	return offsets, data
}
