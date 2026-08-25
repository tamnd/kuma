package buffer

import (
	"testing"
	"unsafe"
)

// The two allocation paths are tested from inside the package because only one
// of them normally runs. alignedBytes asks the allocator for exactly what it
// wants and keeps the answer when it is already on a boundary, which on every
// Go release so far it is, so the padded fallback would otherwise never be
// covered and would rot until the day it was needed.

func TestPaddedBytes(t *testing.T) {
	for _, size := range []int{64, 128, 192, 320, 4096, 1 << 16} {
		p := paddedBytes(size)
		if len(p) != size {
			t.Errorf("paddedBytes(%d) is %d bytes", size, len(p))
		}
		if cap(p) != size {
			t.Errorf("paddedBytes(%d) has cap %d, want the padding out of reach", size, cap(p))
		}
		if !isAligned(p) {
			t.Errorf("paddedBytes(%d) is not aligned", size)
		}
		for i, c := range p {
			if c != 0 {
				t.Fatalf("paddedBytes(%d) is not zeroed at %d", size, i)
			}
		}
	}
}

// TestPaddedBytesEveryOffset covers the case the fallback exists for, which is
// an allocation that did not land on a boundary. Asking for one byte more than
// a size class is the reliable way to get an odd address, and doing it many
// times covers every offset the loop has to correct for.
func TestPaddedBytesEveryOffset(t *testing.T) {
	for range 1000 {
		if p := paddedBytes(65); !isAligned(p) || len(p) != 65 {
			t.Fatalf("paddedBytes(65) gave %d bytes, aligned %v", len(p), isAligned(p))
		}
	}
}

// TestAlignOrPad covers both answers, which is the reason that decision is its
// own function. In an ordinary run only the first one happens.
func TestAlignOrPad(t *testing.T) {
	const size = 256

	// Something known to be on a boundary, rather than a plain make, because a
	// slice that does not escape can live on the stack and a stack frame is
	// only 8 byte aligned. That is not a problem in alignedBytes, where the
	// size is not a constant and the slice therefore has to be on the heap, but
	// it makes for a test that skips itself for no reason.
	direct := alignedBytes(size)
	if got := alignOrPad(direct, size); &got[0] != &direct[0] {
		t.Error("alignOrPad reallocated a slice that was already aligned")
	}

	// One byte past a boundary, which is what the fallback exists for.
	raw := make([]byte, size+2*Alignment)
	off := int(-uintptr(unsafe.Pointer(&raw[0])) & (Alignment - 1))
	askew := raw[off+1 : off+1+size]

	got := alignOrPad(askew, size)
	if !isAligned(got) {
		t.Error("alignOrPad returned an unaligned slice")
	}
	if len(got) != size {
		t.Errorf("alignOrPad returned %d bytes, want %d", len(got), size)
	}
	if &got[0] == &askew[0] {
		t.Error("alignOrPad kept a slice that was off a boundary")
	}
}

func TestAlignedBytesEmpty(t *testing.T) {
	if p := alignedBytes(0); p != nil {
		t.Errorf("alignedBytes(0) = %v, want nil", p)
	}
	if !isAligned(nil) {
		t.Error("isAligned(nil) is false")
	}
}

// TestAlignedBytesTakesTheDirectPath is the measurement that justifies asking
// the allocator first. It is not a correctness test, because the fallback is
// correct too, and it does not fail if the allocator changes. It reports,
// because a silent return to over allocating by 19 percent is the kind of thing
// that goes unnoticed for a year.
func TestAlignedBytesTakesTheDirectPath(t *testing.T) {
	padded := 0
	for _, size := range []int{64, 512, 4096, 1 << 16, 1 << 20} {
		if !isAligned(make([]byte, size)) {
			padded++
		}
	}
	if padded > 0 {
		t.Logf("%d of 5 sizes needed the padded path, so buffers are over allocating again", padded)
	}
}
