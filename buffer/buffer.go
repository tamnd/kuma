// Package buffer provides the byte buffer that columns store their values in.
//
// A Buffer is a byte slice with two properties an ordinary slice does not
// have. Its first byte sits at an address that is a multiple of 64, and its
// capacity is rounded up to a multiple of 64.
//
// Both of those are for the kernels. An aligned load is the cheaper
// instruction on every vector unit that offers the choice, and an aligned
// start means a run over a column does not straddle a cache line at every
// boundary for no reason. The rounded up capacity is the more valuable half:
// it means a kernel handling the last few elements of a column can issue a
// full width load that reads past the end of the data, because the bytes it
// reads are inside the allocation and belong to nobody else. Without that,
// every kernel needs a scalar tail loop, and the tail loop is where the bugs
// are.
//
// Alignment is not free. Go's allocator promises 8 byte alignment and nothing
// stronger, so an aligned buffer asks for 63 bytes more than it needs and
// starts partway in. On the chunk sizes this engine works in, tens of
// kilobytes at a time, that is a rounding error against what it buys.
//
// There is no reference counting. Buffers are ordinary Go memory and the
// garbage collector already knows how to free them. Arrow implementations in
// other languages need Retain and Release because their host languages have no
// collector, and porting that discipline to Go would be a tax that everybody
// forgets to pay exactly once, in the place it matters. Pool is the escape
// hatch for the executor, where a scratch buffer has a lifetime short enough
// to prove by reading the code.
//
// The zero Buffer is empty and ready to use.
//
// Stability: tier 1, stable.
package buffer

import "unsafe"

// Alignment is the address boundary every buffer starts on, and the multiple
// every capacity is rounded up to.
//
// 64 is the AVX-512 vector width, and also the cache line size on every x86
// part since the Pentium 4 and on arm64. Arrow recommends the same number for
// the same reasons, so a buffer that satisfies this also satisfies anything
// that wants to read it over the C data interface.
const Alignment = 64

// Buffer is a growable, aligned sequence of bytes.
type Buffer struct {
	// buf is the whole allocation. Its length is the capacity, since the
	// aligned region is handed out with its cap trimmed to its len so that
	// nothing can append into the padding by accident.
	buf []byte
	// n is how much of buf the caller has asked for.
	n int
}

// New returns a buffer of n zeroed bytes.
func New(n int) *Buffer {
	if n < 0 {
		panic("buffer: negative length")
	}
	return &Buffer{buf: alignedBytes(roundUp(n)), n: n}
}

// Wrap returns a buffer backed by p, without copying it, and takes ownership of
// it. The caller must not use p afterwards.
//
// This is the path for bytes that came from somewhere that already laid them
// out, meaning a memory mapped file or an Arrow IPC message. Such bytes are
// usually aligned, because the format asks for it, but nothing here checks and
// Aligned is how a caller finds out. A wrapped buffer that has to grow stops
// being wrapped, since growing means a fresh aligned allocation and a copy.
func Wrap(p []byte) *Buffer {
	return &Buffer{buf: p, n: len(p)}
}

// Len returns the number of bytes in use.
func (b *Buffer) Len() int { return b.n }

// Cap returns how many bytes the buffer holds before it has to grow.
func (b *Buffer) Cap() int { return len(b.buf) }

// Bytes returns the bytes in use. Modifying the result modifies the buffer.
//
// The result stops at the length, so the padding past it is not visible here.
// A kernel that wants to read the padding on purpose, which is the reason the
// padding exists, has to go through Padded.
func (b *Buffer) Bytes() []byte { return b.buf[:b.n] }

// Padded returns the bytes in use followed by the padding after them, which is
// the whole allocation. It is what a kernel reads when it wants to process the
// final partial vector without a scalar tail loop.
//
// The padding bytes are not part of the data and their contents mean nothing.
// A kernel may read them and must not let them change its answer.
//
// The result is a whole number of Alignment sized blocks for every buffer this
// package allocates. It is not for a wrapped one, which is exactly as long as
// the bytes it was handed, so a kernel that reads whole vectors has to check
// Aligned or work on a Clone.
func (b *Buffer) Padded() []byte { return b.buf }

// Aligned reports whether the buffer starts on an Alignment boundary. It is
// true for every buffer this package allocates, and it is worth asking about a
// wrapped one.
func (b *Buffer) Aligned() bool {
	if len(b.buf) == 0 {
		return true
	}
	return uintptr(unsafe.Pointer(&b.buf[0]))&(Alignment-1) == 0
}

// Grow makes room for n more bytes without adding them, reallocating if it has
// to. It is worth calling when the final size is known.
func (b *Buffer) Grow(n int) {
	if n < 0 {
		panic("buffer: negative count")
	}
	if b.n+n > len(b.buf) {
		b.realloc(b.n + n)
	}
}

// Resize sets the length to n. Growing zeroes the new bytes, so a buffer that
// was shrunk and then grown again never shows what used to be there.
func (b *Buffer) Resize(n int) {
	if n < 0 {
		panic("buffer: negative length")
	}
	if n > b.n {
		b.Grow(n - b.n)
		clear(b.buf[b.n:n])
	}
	b.n = n
}

// Append adds p to the end.
func (b *Buffer) Append(p []byte) {
	b.Grow(len(p))
	b.n += copy(b.buf[b.n:], p)
}

// Reset sets the length to zero and keeps the memory. The bytes that were
// there are still there until something writes over them, which is what makes
// this cheap and why Resize zeroes on the way back up.
func (b *Buffer) Reset() { b.n = 0 }

// Zero sets every byte to zero, including the padding past the length.
func (b *Buffer) Zero() { clear(b.buf) }

// Clone returns a copy that shares no memory with b. The copy is aligned even
// if b was wrapped around something that was not.
func (b *Buffer) Clone() *Buffer {
	out := New(b.n)
	copy(out.buf, b.buf[:b.n])
	return out
}

// realloc moves the contents into a fresh allocation big enough for want bytes.
//
// The new size is the smallest power of two that fits, rather than the
// smallest multiple of Alignment, for two reasons. Doubling is what makes
// repeated appends amortize to constant time. And a power of two is what Pool
// hands out and takes back, so a buffer that grew its way to a class size can
// still be recycled.
func (b *Buffer) realloc(want int) {
	size := Alignment
	for size < want {
		size *= 2
	}
	next := alignedBytes(size)
	copy(next, b.buf[:b.n])
	b.buf = next
}

// roundUp returns n rounded up to a multiple of Alignment.
func roundUp(n int) int { return (n + Alignment - 1) &^ (Alignment - 1) }

// alignedBytes returns size zeroed bytes starting on an Alignment boundary.
//
// The only portable way to beat the allocator's 8 byte promise is to ask for
// Alignment-1 extra bytes and start at the first aligned address inside them.
// The address is read once, to work out how far in that is, and everything
// after that is ordinary slicing, so nothing here depends on the memory staying
// where it is. Go's collector does not move heap objects, and if that ever
// changes then Aligned is the thing that starts returning false and this is the
// function to fix.
//
// The result has its capacity trimmed to its length, so appending to it
// allocates rather than quietly running into the padding at the end.
func alignedBytes(size int) []byte {
	if size == 0 {
		return nil
	}
	raw := make([]byte, size+Alignment-1)
	off := int(-uintptr(unsafe.Pointer(&raw[0])) & (Alignment - 1))
	return raw[off : off+size : off+size]
}
