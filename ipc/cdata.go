//go:build cgo

package ipc

/*
#include <stdlib.h>
#include "abi.h"

// The release callbacks, which are Go functions with an //export comment in
// cdata_export.go. An exported Go function is a real C function, so it can be
// stored in a struct member the way any other one can, but Go cannot write a C
// function pointer into a struct and C cannot call one it was never given, so
// the four helpers below are the two directions of that.
void kumaReleaseSchema(struct ArrowSchema*);
void kumaReleaseArray(struct ArrowArray*);

static void kuma_set_schema_release(struct ArrowSchema* schema) {
	schema->release = kumaReleaseSchema;
}

static void kuma_set_array_release(struct ArrowArray* array) {
	array->release = kumaReleaseArray;
}

static void kuma_call_schema_release(struct ArrowSchema* schema) {
	if (schema != NULL && schema->release != NULL) {
		schema->release(schema);
	}
}

static void kuma_call_array_release(struct ArrowArray* array) {
	if (array != NULL && array->release != NULL) {
		array->release(array);
	}
}
*/
import "C"

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"runtime"
	"runtime/cgo"
	"unsafe"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/strview"
)

// CSchema is the ArrowSchema struct of the C data interface, which describes
// one field: its name, its type, whether it is nullable and its metadata.
//
// It is an alias for the cgo type, so a caller with a struct of its own can
// convert a pointer to it through unsafe.Pointer. Two packages that both run
// cgo get two Go types for the same C struct, and that conversion is how they
// meet. The struct itself is the same memory either way.
type CSchema = C.struct_ArrowSchema

// CArray is the ArrowArray struct of the C data interface, which holds the
// values: a length, an offset, a null count and the buffer pointers.
type CArray = C.struct_ArrowArray

// exportedArray is what an exported array needs to hold on to until its release
// callback runs. It hangs off the private_data member through a cgo.Handle,
// since a C struct cannot hold a Go pointer and a handle is an integer.
type exportedArray struct {
	// pin keeps the Go buffers where they are. Storing a Go pointer in C
	// memory is only allowed for a pinned one, and the buffers are the whole
	// point of the exercise, so they are pinned rather than copied.
	pin runtime.Pinner

	// free is the C memory allocated for this array other than the buffer list
	// and the private data, which is the buffer of buffer sizes and nothing
	// else so far.
	free []unsafe.Pointer
}

// ExportField fills out with the C schema for f.
//
// The strings and the child structs are allocated with malloc and are owned by
// the release callback, which the consumer calls when it is finished. The
// consumer owns the struct out points at, both before and after: this fills it
// in and never allocates or frees it.
//
// Nested types export their children, so a struct of three fields or a map of
// string to int64 comes out whole. That is worth having even though the values
// of those types cannot be exported yet, because a schema travels on its own:
// it is what a reader asks for before it decides which columns to read.
//
// A dictionary type exports the index type as the format string and the value
// type as the dictionary member, which is how the C data interface splits it.
//
// On error nothing is left allocated and out is a released schema.
func ExportField(f dtype.Field, out *CSchema) error {
	if out == nil {
		return errors.New("ipc: nil schema struct")
	}
	*out = CSchema{}

	format, err := Format(f.Type)
	if err != nil {
		return err
	}
	children, err := childFields(f.Type)
	if err != nil {
		return err
	}
	metadata, err := EncodeMetadata(f.Metadata)
	if err != nil {
		return err
	}

	// The release callback goes on before the children do, so that a failure
	// part of the way through the children can hand the whole half built thing
	// to the same code that frees a finished one.
	C.kuma_set_schema_release(out)
	out.format = C.CString(format)
	out.name = C.CString(f.Name)
	if metadata != nil {
		out.metadata = (*C.char)(C.CBytes(metadata))
	}
	if f.Nullable {
		out.flags = C.ARROW_FLAG_NULLABLE
	}

	if len(children) > 0 {
		out.n_children = C.int64_t(len(children))
		out.children = (**C.struct_ArrowSchema)(callocPointers(len(children)))
		slots := unsafe.Slice(out.children, len(children))
		for i, child := range children {
			slots[i] = (*CSchema)(C.calloc(1, C.sizeof_struct_ArrowSchema))
			if err := ExportField(child, slots[i]); err != nil {
				C.kuma_call_schema_release(out)
				return err
			}
		}
	}

	if d, ok := f.Type.(dtype.Dictionary); ok {
		out.dictionary = (*CSchema)(C.calloc(1, C.sizeof_struct_ArrowSchema))
		value := dtype.Field{Name: f.Name, Type: d.Value, Nullable: true}
		if err := ExportField(value, out.dictionary); err != nil {
			C.kuma_call_schema_release(out)
			return err
		}
	}
	return nil
}

// ExportArray fills out with the C array for a.
//
// No value is copied. The buffer pointers point into the array's own memory,
// which is pinned until the consumer calls the release callback, so the array
// has to stay reachable in Go until then as well. Pinning stops the collector
// moving the buffers, not the caller dropping the last reference to the frame
// they came from.
//
// The view layout gets the buffer of buffer sizes the C ABI asks for, built
// here, since a Go slice knows its own length and a C pointer does not.
//
// The type does not travel with the values. A consumer needs the schema too,
// which is what ExportField is for.
func ExportArray(a *array.Array, out *CArray) error {
	if out == nil {
		return errors.New("ipc: nil array struct")
	}
	l, err := Export(a)
	if err != nil {
		return err
	}
	*out = CArray{}

	// The view layout carries one more buffer in C than it does in Go: the
	// sizes of the data blocks, which a consumer reading raw pointers has no
	// other way to learn.
	sizes := -1
	switch a.DType().Kind() {
	case dtype.StringKind, dtype.BinaryKind:
		sizes = len(l.Buffers)
	default:
	}

	n := len(l.Buffers)
	if sizes >= 0 {
		n++
	}

	state := &exportedArray{}
	handle := C.malloc(C.size_t(unsafe.Sizeof(C.uintptr_t(0))))
	*(*C.uintptr_t)(handle) = C.uintptr_t(cgo.NewHandle(state))
	out.private_data = handle
	C.kuma_set_array_release(out)

	out.length = C.int64_t(l.Length)
	out.null_count = C.int64_t(l.NullCount)
	out.offset = C.int64_t(l.Offset)
	out.n_buffers = C.int64_t(n)

	if n > 0 {
		out.buffers = (*unsafe.Pointer)(callocPointers(n))
		slots := unsafe.Slice(out.buffers, n)
		for i, b := range l.Buffers {
			if len(b) == 0 {
				// An empty buffer is a null pointer, which is what the
				// interface says a validity bitmap with no nulls looks like and
				// what every consumer already handles for an empty column.
				continue
			}
			p := unsafe.SliceData(b)
			state.pin.Pin(p)
			slots[i] = unsafe.Pointer(p)
		}
		if sizes >= 0 {
			slots[sizes] = exportSizes(state, l.Buffers[2:])
		}
	}
	return nil
}

// Imported is a kuma array over memory that belongs to whoever produced it.
//
// The values are not copied, so the array is only valid until Release is
// called, and Release must be called or the producer never learns that it can
// free anything. The usual shape is a call and a defer on the line after it.
//
// Release sets Array to nil rather than leaving it pointing at memory that has
// been handed back, so that code holding it past the release fails on the spot
// instead of reading whatever moved in.
type Imported struct {
	// Field is what the schema said: the name, the type, the nullability and
	// the metadata.
	Field dtype.Field

	// Array is the values.
	Array *array.Array

	c *CArray
}

// Release hands the memory back to the library that produced it. It is safe to
// call more than once and does nothing the second time.
func (i *Imported) Release() {
	if i == nil || i.c == nil {
		return
	}
	c := i.c
	i.c = nil
	i.Array = nil
	C.kuma_call_array_release(c)
}

// ReleaseSchema and ReleaseArray hand a struct back to whoever produced it by
// calling its own release callback. They do nothing for a null pointer or for a
// struct that has already been released, which is what makes them safe to
// defer.
//
// A Go caller needs them for two things: a struct this package exported and
// then did not manage to send, and a struct another library handed over that
// this package is finished with. The memory the struct itself sits in belongs
// to whoever allocated it either way, and neither of these frees that.
func ReleaseSchema(s *CSchema) { C.kuma_call_schema_release(s) }

// ReleaseArray releases an array struct, whoever produced it.
func ReleaseArray(a *CArray) { C.kuma_call_array_release(a) }

// ImportField reads a C schema and returns the field it describes.
//
// The schema is released before this returns, whether or not it succeeded,
// because everything in it has been copied into Go by then. That is the
// contract of the interface: a consumer that is handed a struct owns it.
//
// The struct itself is not freed. Whoever allocated it frees it.
func ImportField(s *CSchema) (dtype.Field, error) {
	if s == nil {
		return dtype.Field{}, errors.New("ipc: nil schema struct")
	}
	defer C.kuma_call_schema_release(s)
	if s.release == nil {
		return dtype.Field{}, errors.New("ipc: the schema has already been released")
	}
	return importField(s)
}

// ImportArray reads a C schema and a C array and returns a kuma array over the
// producer's own memory.
//
// Both structs are taken over. The schema is released before this returns,
// since the type is copied into Go. The array is released by Imported.Release,
// which the caller has to get to, and is released here if the import fails.
//
// Nothing is copied except the views of a column that arrived in one of the
// offset layouts. See Import for what that means and why.
//
// Nested and dictionary encoded arrays are not supported yet, because the array
// package has nowhere to put them. Their schemas import fine, so a reader can
// still see what a file holds before it decides that it cannot read one column
// of it.
func ImportArray(s *CSchema, a *CArray) (*Imported, error) {
	if s == nil || a == nil {
		return nil, errors.New("ipc: nil schema or array struct")
	}
	defer C.kuma_call_schema_release(s)
	if s.release == nil || a.release == nil {
		C.kuma_call_array_release(a)
		return nil, errors.New("ipc: the schema or the array has already been released")
	}

	f, err := importField(s)
	if err != nil {
		C.kuma_call_array_release(a)
		return nil, err
	}
	if s.dictionary != nil {
		C.kuma_call_array_release(a)
		return nil, fmt.Errorf("ipc: %w: %s is a dictionary encoded array", ErrUnsupported, f.Type)
	}

	l, err := importLayout(C.GoString(s.format), a)
	if err != nil {
		C.kuma_call_array_release(a)
		return nil, err
	}
	values, err := Import(C.GoString(s.format), l)
	if err != nil {
		C.kuma_call_array_release(a)
		return nil, err
	}
	return &Imported{Field: f, Array: values, c: a}, nil
}

// importField is ImportField without the release, so that a child, which its
// parent releases, can go through the same code.
func importField(s *CSchema) (dtype.Field, error) {
	if s.format == nil {
		return dtype.Field{}, fmt.Errorf("ipc: %w: the schema has no format string", ErrFormat)
	}
	format := C.GoString(s.format)

	children, err := importChildren(s)
	if err != nil {
		return dtype.Field{}, err
	}
	t, err := Type(format, children)
	if err != nil {
		return dtype.Field{}, err
	}

	// A dictionary is the one type the format string does not name. It says
	// what the indices are and the values hang off their own schema.
	if s.dictionary != nil {
		var value dtype.Field
		value, err = importField(s.dictionary)
		if err != nil {
			return dtype.Field{}, err
		}
		t = dtype.Dictionary{Index: t, Value: value.Type}
		if err = dtype.Validate(t); err != nil {
			return dtype.Field{}, fmt.Errorf("ipc: format %q: %w", format, err)
		}
	}

	metadata, err := importMetadata(s.metadata)
	if err != nil {
		return dtype.Field{}, err
	}
	var name string
	if s.name != nil {
		name = C.GoString(s.name)
	}
	return dtype.Field{
		Name:     name,
		Type:     t,
		Nullable: s.flags&C.ARROW_FLAG_NULLABLE != 0,
		Metadata: metadata,
	}, nil
}

// importChildren reads the child schemas of a nested type. The ordered flag of
// a dictionary and the sorted flag of a map are read and dropped, since kuma
// has no type that is different for having them set.
func importChildren(s *CSchema) ([]dtype.Field, error) {
	n := int64(s.n_children)
	if n < 0 {
		return nil, fmt.Errorf("ipc: %w: negative child count %d", ErrChildren, n)
	}
	if n == 0 {
		return nil, nil
	}
	if s.children == nil {
		return nil, fmt.Errorf("ipc: %w: %d children and no child pointers", ErrChildren, n)
	}

	children := make([]dtype.Field, 0, n)
	for _, c := range unsafe.Slice(s.children, n) {
		if c == nil {
			return nil, fmt.Errorf("ipc: %w: child %d is a null pointer", ErrChildren, len(children))
		}
		f, err := importField(c)
		if err != nil {
			return nil, err
		}
		children = append(children, f)
	}
	return children, nil
}

// importMetadata copies the metadata blob out of C memory. The blob has no
// length of its own, so the length is what walking the pairs adds up to, and a
// blob that lies about its own lengths reads past its end the same way it would
// in any other language that speaks this interface.
func importMetadata(p *C.char) (dtype.Metadata, error) {
	if p == nil {
		return nil, nil
	}
	size, err := metadataSize(unsafe.Pointer(p))
	if err != nil {
		return nil, err
	}
	return DecodeMetadata(C.GoBytes(unsafe.Pointer(p), C.int(size)))
}

// metadataSize is the length in bytes of the metadata blob at p.
func metadataSize(p unsafe.Pointer) (int, error) {
	pairs := int64(int32At(p, 0))
	if pairs < 0 {
		return 0, fmt.Errorf("ipc: %w: negative pair count %d", ErrMetadata, pairs)
	}
	size := int64(metadataHeader)
	for range pairs {
		for range 2 {
			n := int64(int32At(p, size))
			if n < 0 {
				return 0, fmt.Errorf("ipc: %w: negative string length %d", ErrMetadata, n)
			}
			size += metadataLength + n
			if size > math.MaxInt32 {
				return 0, fmt.Errorf("ipc: %w: the blob is longer than %d bytes", ErrMetadata, math.MaxInt32)
			}
		}
	}
	return int(size), nil
}

// importLayout works out how long each buffer of a C array is.
//
// The C struct does not say. It has the pointers and nothing else, so the
// lengths come from the type and the length for most of the layouts, from the
// last offset for the two offset layouts, and from the trailing buffer of sizes
// for the view layout. Getting this wrong is how a consumer reads past the end
// of somebody else's allocation, which is why it is one function and not a line
// in each of five.
//
// A producer that allocated less than its own length says is not caught here
// and cannot be. There is no size in the struct to compare against, so the
// length is a promise. All this can do is work out the same size the
// specification says the producer wrote. What is checked is everything
// the struct is wrong about on its own terms: a buffer count that does not
// match the layout, a negative length, offset, block size or offset value.
func importLayout(format string, a *CArray) (Layout, error) {
	if a.n_children != 0 || a.dictionary != nil {
		return Layout{}, fmt.Errorf("ipc: %w: %q is a nested or dictionary encoded array", ErrUnsupported, format)
	}
	l := Layout{Length: int(a.length), Offset: int(a.offset), NullCount: int(a.null_count)}
	if l.Length < 0 || l.Offset < 0 {
		return Layout{}, fmt.Errorf("ipc: %w: length %d and offset %d", ErrBuffers, l.Length, l.Offset)
	}
	total := int64(l.Offset) + int64(l.Length)

	n := int64(a.n_buffers)
	if n < 0 {
		return Layout{}, fmt.Errorf("ipc: %w: negative buffer count %d", ErrBuffers, n)
	}
	if n > 0 && a.buffers == nil {
		return Layout{}, fmt.Errorf("ipc: %w: %d buffers and no buffer pointers", ErrBuffers, n)
	}
	var buffers []unsafe.Pointer
	if n > 0 {
		buffers = unsafe.Slice(a.buffers, n)
	}

	if format == "n" {
		// A Null array is its length. Whatever buffers it came with, if any,
		// hold nothing anybody reads.
		return l, nil
	}
	if len(buffers) < 2 {
		return Layout{}, fmt.Errorf("ipc: %w: a %q array has at least 2 buffers, have %d",
			ErrBuffers, format, len(buffers))
	}

	validity, err := bufferBytes(buffers[0], (total+7)/8)
	if err != nil {
		return Layout{}, err
	}
	l.Buffers = [][]byte{validity}

	switch format {
	case "u", "z":
		err = appendOffsets(&l, format, buffers, total, 4)
	case "U", "Z":
		err = appendOffsets(&l, format, buffers, total, 8)
	case "vu", "vz":
		err = appendViews(&l, format, buffers, total)
	default:
		err = appendFixed(&l, format, buffers, total)
	}
	if err != nil {
		return Layout{}, err
	}
	return l, nil
}

// appendFixed adds the values buffer of a type with a fixed number of bits per
// value. Bool is one of those, at one bit each.
func appendFixed(l *Layout, format string, buffers []unsafe.Pointer, total int64) error {
	t, err := Type(format, nil)
	if err != nil {
		return err
	}
	bits, ok := dtype.Bits(t)
	if !ok {
		return fmt.Errorf("ipc: %w: importing a %s array", ErrUnsupported, t)
	}
	if len(buffers) != 2 {
		return fmt.Errorf("ipc: %w: a %s array has 2 buffers, have %d", ErrBuffers, t, len(buffers))
	}
	values, err := bufferBytes(buffers[1], (total*int64(bits)+7)/8)
	if err != nil {
		return err
	}
	l.Buffers = append(l.Buffers, values)
	return nil
}

// appendOffsets adds the offsets and the data of one of the two offset layouts.
// The size of the data buffer is the last offset, which is the only place it is
// written down.
func appendOffsets(l *Layout, format string, buffers []unsafe.Pointer, total int64, width int) error {
	if len(buffers) != 3 {
		return fmt.Errorf("ipc: %w: the %q layout has 3 buffers, have %d", ErrBuffers, format, len(buffers))
	}
	offsets, err := bufferBytes(buffers[1], (total+1)*int64(width))
	if err != nil {
		return err
	}

	var size int64
	if len(offsets) > 0 {
		last := total * int64(width)
		if width == 4 {
			size = int64(int32At(unsafe.Pointer(unsafe.SliceData(offsets)), last))
		} else {
			size = int64At(unsafe.Pointer(unsafe.SliceData(offsets)), last)
		}
		if size < 0 {
			return fmt.Errorf("ipc: %w: the last offset of a %q array is %d", ErrBuffers, format, size)
		}
	}
	data, err := bufferBytes(buffers[2], size)
	if err != nil {
		return err
	}
	l.Buffers = append(l.Buffers, offsets, data)
	return nil
}

// appendViews adds the views and the data blocks of the view layout.
//
// The C ABI puts a buffer of int64 sizes at the end, one per data block, so
// this is the one layout whose buffer count is not fixed by its format string
// and the one whose data lengths are written down rather than worked out.
func appendViews(l *Layout, format string, buffers []unsafe.Pointer, total int64) error {
	if len(buffers) < 3 {
		return fmt.Errorf("ipc: %w: the %q layout has at least 3 buffers, have %d",
			ErrBuffers, format, len(buffers))
	}
	views, err := bufferBytes(buffers[1], total*strview.Size)
	if err != nil {
		return err
	}
	l.Buffers = append(l.Buffers, views)

	blocks := buffers[2 : len(buffers)-1]
	sizes := buffers[len(buffers)-1]
	if len(blocks) > 0 && sizes == nil {
		return fmt.Errorf("ipc: %w: the %q layout has %d data blocks and no buffer of sizes",
			ErrBuffers, format, len(blocks))
	}
	for i, b := range blocks {
		size := int64At(sizes, int64(i)*8)
		if size < 0 {
			return fmt.Errorf("ipc: %w: data block %d of a %q array is %d bytes",
				ErrBuffers, i, format, size)
		}
		block, err := bufferBytes(b, size)
		if err != nil {
			return err
		}
		l.Buffers = append(l.Buffers, block)
	}
	return nil
}

// bufferBytes reads C memory as a Go slice, borrowing rather than copying. A
// null pointer is no bytes, which is what an absent validity bitmap and an
// empty column both look like.
func bufferBytes(p unsafe.Pointer, size int64) ([]byte, error) {
	if p == nil || size == 0 {
		return nil, nil
	}
	if size < 0 || size > math.MaxInt {
		return nil, fmt.Errorf("ipc: %w: a buffer of %d bytes", ErrBuffers, size)
	}
	return unsafe.Slice((*byte)(p), size), nil
}

// exportSizes builds the buffer of data block sizes the view layout needs in C.
func exportSizes(state *exportedArray, blocks [][]byte) unsafe.Pointer {
	// One entry is allocated for a column with no data blocks, so that the
	// pointer is a real one. A consumer that reads no entries from it does not
	// care, and a consumer that checks for a null pointer first is happier.
	n := max(len(blocks), 1)
	sizes := C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.int64_t(0))))
	state.free = append(state.free, sizes)
	slots := unsafe.Slice((*C.int64_t)(sizes), n)
	slots[0] = 0
	for i, b := range blocks {
		slots[i] = C.int64_t(len(b))
	}
	return sizes
}

// childFields is the children a nested type exports, with the names Arrow uses
// for the ones that have no name of their own.
func childFields(t dtype.DataType) ([]dtype.Field, error) {
	switch x := t.(type) {
	case dtype.List:
		return itemField(x.Elem), nil
	case dtype.LargeList:
		return itemField(x.Elem), nil
	case dtype.FixedSizeList:
		return itemField(x.Elem), nil
	case dtype.Struct:
		return x.Fields, nil
	case dtype.Map:
		// A map is a list of entries and an entry is a struct of two, so the
		// key and the value are a level further down than they look. Keys are
		// not nullable, which the interface states rather than implies.
		entries := dtype.Struct{Fields: []dtype.Field{
			{Name: "key", Type: x.Key},
			{Name: "value", Type: x.Value, Nullable: true},
		}}
		return []dtype.Field{{Name: "entries", Type: entries}}, nil
	default:
		return nil, nil
	}
}

// itemField is the single child of a list, under the name Arrow gives it. The
// values of a list are nullable whatever the list itself is, since a list of
// two that holds one value has to say which one is missing.
func itemField(t dtype.DataType) []dtype.Field {
	return []dtype.Field{{Name: "item", Type: t, Nullable: true}}
}

// callocPointers allocates a zeroed array of n pointers.
func callocPointers(n int) unsafe.Pointer {
	return C.calloc(C.size_t(n), C.size_t(unsafe.Sizeof(unsafe.Pointer(nil))))
}

// int32At and int64At read a number out of C memory at a byte offset. The blob
// and the offsets are written in the byte order of the machine and are not
// aligned to anything, since a string in front of them can be any length.
func int32At(p unsafe.Pointer, off int64) int32 {
	return int32(binary.NativeEndian.Uint32(unsafe.Slice((*byte)(unsafe.Add(p, off)), 4)))
}

func int64At(p unsafe.Pointer, off int64) int64 {
	return int64(binary.NativeEndian.Uint64(unsafe.Slice((*byte)(unsafe.Add(p, off)), 8)))
}
