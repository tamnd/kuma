//go:build cgo

package ipctest

/*
#cgo CFLAGS: -I${SRCDIR}/..

#include <stdlib.h>
#include "abi.h"

// The producer. Its release callbacks free what Array and Schema allocated and
// count themselves, so that a test can say that a consumer handed the memory
// back rather than that it did not crash.
static int kuma_test_releases;

static void kuma_test_release_array(struct ArrowArray* array) {
	int64_t i;
	for (i = 0; i < array->n_buffers; i++) {
		free((void*)array->buffers[i]);
	}
	free(array->buffers);
	array->buffers = NULL;
	array->release = NULL;
	kuma_test_releases++;
}

static void kuma_test_release_schema(struct ArrowSchema* schema) {
	free((void*)schema->format);
	free((void*)schema->metadata);
	schema->format = NULL;
	schema->metadata = NULL;
	schema->release = NULL;
	kuma_test_releases++;
}

static void kuma_test_set_array_release(struct ArrowArray* array) {
	array->release = kuma_test_release_array;
}

static void kuma_test_set_schema_release(struct ArrowSchema* schema) {
	schema->release = kuma_test_release_schema;
}

static int kuma_test_release_count(void) { return kuma_test_releases; }
*/
import "C"

import (
	"testing"
	"unsafe"

	"github.com/tamnd/kuma/ipc"
)

// Pair allocates a zeroed schema and array in C memory, which is where a
// producer would put them, and frees the two structs when the test ends.
//
// Nothing is filled in. This is what a test exports into.
func Pair(t testing.TB) (*ipc.CSchema, *ipc.CArray) {
	t.Helper()
	schema := (*C.struct_ArrowSchema)(C.calloc(1, C.sizeof_struct_ArrowSchema))
	values := (*C.struct_ArrowArray)(C.calloc(1, C.sizeof_struct_ArrowArray))
	t.Cleanup(func() {
		C.free(unsafe.Pointer(schema))
		C.free(unsafe.Pointer(values))
	})
	return schemaOf(schema), arrayOf(values)
}

// Schema returns a schema holding the given format string, with a release
// callback of this package's own. The format string is not checked, so a test
// can hand over one that no implementation would write.
func Schema(t testing.TB, format string) *ipc.CSchema {
	t.Helper()
	schema, _ := Pair(t)
	c := cSchema(schema)
	c.format = C.CString(format)
	C.kuma_test_set_schema_release(c)
	return schema
}

// Array returns an array of the given length whose buffers are copies of the
// given bytes in C memory, with a release callback of this package's own. An
// empty buffer becomes a null pointer, which is how the interface writes a
// validity bitmap for a column with no nulls.
//
// The copies are what makes this worth having. An import that reaches past the
// end of a buffer reaches into memory the allocator knows the size of, so a
// test under a sanitizer or a leak checker sees it.
func Array(t testing.TB, length int, buffers [][]byte) *ipc.CArray {
	t.Helper()
	_, values := Pair(t)
	c := cArray(values)
	c.length = C.int64_t(length)
	c.n_buffers = C.int64_t(len(buffers))
	if len(buffers) > 0 {
		c.buffers = (*unsafe.Pointer)(C.calloc(C.size_t(len(buffers)),
			C.size_t(unsafe.Sizeof(unsafe.Pointer(nil)))))
		slots := unsafe.Slice(c.buffers, len(buffers))
		for i, b := range buffers {
			if len(b) == 0 {
				continue
			}
			p := C.malloc(C.size_t(len(b)))
			copy(unsafe.Slice((*byte)(p), len(b)), b)
			slots[i] = p
		}
	}
	C.kuma_test_set_array_release(c)
	return values
}

// Releases is how many structs this package has had handed back since the
// program started. A test reads it before and after and compares the two, since
// the tests of one package share the count.
func Releases() int { return int(C.kuma_test_release_count()) }

// SchemaReleased reports whether a schema has been released, which the
// interface says is a release callback that is now a null pointer.
func SchemaReleased(s *ipc.CSchema) bool { return cSchema(s).release == nil }

// ArrayReleased reports whether an array has been released.
func ArrayReleased(a *ipc.CArray) bool { return cArray(a).release == nil }

// Format is the format string of a schema, or the empty string if it has none.
func Format(s *ipc.CSchema) string {
	c := cSchema(s)
	if c.format == nil {
		return ""
	}
	return C.GoString(c.format)
}

// Length is the length an array struct says it has, as written rather than as
// corrected, so a test can see a producer's own number.
func Length(a *ipc.CArray) int { return int(cArray(a).length) }

// Offset is where an array struct says its values start.
func Offset(a *ipc.CArray) int { return int(cArray(a).offset) }

// SetLength writes the length of an array struct. It is here so that a test can
// write a number no producer should, such as a negative one.
func SetLength(a *ipc.CArray, n int) { cArray(a).length = C.int64_t(n) }

// SetMetadata writes the metadata blob of a schema. The blob is a pointer with
// no length of its own, so this is how a test hands over one that says it is
// longer than it is.
func SetMetadata(s *ipc.CSchema, blob []byte) {
	c := cSchema(s)
	C.free(unsafe.Pointer(c.metadata))
	c.metadata = nil
	if len(blob) == 0 {
		return
	}
	p := C.malloc(C.size_t(len(blob)))
	copy(unsafe.Slice((*byte)(p), len(blob)), blob)
	c.metadata = (*C.char)(p)
}

// SetChildren writes the child count of an array struct without allocating any
// children, which is enough for a consumer that stops as soon as it sees that
// there are some.
func SetChildren(a *ipc.CArray, n int) { cArray(a).n_children = C.int64_t(n) }

// Buffers is where each buffer of an array struct starts. The pointers are what
// a test compares against the memory a kuma array is made of, since two buffers
// holding the same bytes are not the same buffer and only the address says so.
func Buffers(a *ipc.CArray) []unsafe.Pointer {
	c := cArray(a)
	if c.buffers == nil {
		return nil
	}
	return unsafe.Slice(c.buffers, int(c.n_buffers))
}

// The four conversions between the C types of this package and the ones the ipc
// package has. They are the same two structs. Two packages that both run cgo
// get their own Go type for each, and a pointer conversion is how they meet,
// which is what a caller with structs of its own has to do as well.
func cSchema(s *ipc.CSchema) *C.struct_ArrowSchema {
	return (*C.struct_ArrowSchema)(unsafe.Pointer(s))
}

func cArray(a *ipc.CArray) *C.struct_ArrowArray {
	return (*C.struct_ArrowArray)(unsafe.Pointer(a))
}

func schemaOf(s *C.struct_ArrowSchema) *ipc.CSchema {
	return (*ipc.CSchema)(unsafe.Pointer(s))
}

func arrayOf(a *C.struct_ArrowArray) *ipc.CArray {
	return (*ipc.CArray)(unsafe.Pointer(a))
}
