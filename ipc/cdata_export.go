//go:build cgo

package ipc

/*
#include <stdlib.h>
#include "abi.h"
*/
import "C"

import (
	"runtime/cgo"
	"unsafe"
)

// This file is separate from cdata.go because a file that exports a Go function
// to C may only declare things in its preamble, never define them, and cdata.go
// defines the four helpers that set and call the release callbacks.

//export kumaReleaseSchema
func kumaReleaseSchema(s *C.struct_ArrowSchema) {
	if s == nil {
		return
	}

	// Every child of a schema this package exported was allocated by this
	// package, so the children go through this function rather than through
	// their own release callback. A child that was allocated and not yet filled
	// in is all zeroes, and freeing a null pointer is allowed, so a schema that
	// failed part of the way through frees the same way a whole one does.
	if s.children != nil {
		for _, child := range unsafe.Slice(s.children, int(s.n_children)) {
			if child == nil {
				continue
			}
			kumaReleaseSchema(child)
			C.free(unsafe.Pointer(child))
		}
		C.free(unsafe.Pointer(s.children))
	}
	if s.dictionary != nil {
		kumaReleaseSchema(s.dictionary)
		C.free(unsafe.Pointer(s.dictionary))
	}

	C.free(unsafe.Pointer(s.format))
	C.free(unsafe.Pointer(s.name))
	C.free(unsafe.Pointer(s.metadata))

	*s = C.struct_ArrowSchema{}
}

//export kumaReleaseArray
func kumaReleaseArray(a *C.struct_ArrowArray) {
	if a == nil {
		return
	}
	if a.private_data != nil {
		handle := cgo.Handle(*(*C.uintptr_t)(a.private_data))
		if state, ok := handle.Value().(*exportedArray); ok {
			// Unpinning is what lets the collector move or free the buffers
			// again. Nothing points at them from C after this line.
			state.pin.Unpin()
			for _, p := range state.free {
				C.free(p)
			}
		}
		handle.Delete()
		C.free(a.private_data)
	}
	if a.buffers != nil {
		C.free(unsafe.Pointer(a.buffers))
	}

	*a = C.struct_ArrowArray{}
}
