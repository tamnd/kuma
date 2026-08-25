// Package bitmap implements the Arrow validity bitmap.
//
// A bitmap holds one bit per value, packed least significant bit first within
// each byte, which is the layout the Arrow columnar specification requires. A
// set bit means the value at that position is valid, and a clear bit means it
// is null.
//
// The zero Bitmap is empty and ready to use. Append grows it.
//
// Bits past the length live in the last byte and are always zero. Every
// operation that could leave them set clears them again before returning, so
// CountOnes and Bytes never have to think about it. This is not merely tidy:
// Arrow requires the padding to be zero for a buffer to round trip through IPC
// correctly.
//
// Stability: tier 1, stable.
package bitmap

import "math/bits"

// Bitmap is a growable sequence of bits.
type Bitmap struct {
	bits []byte
	n    int
}

// New returns a bitmap of n bits, all clear.
func New(n int) *Bitmap {
	if n < 0 {
		panic("bitmap: negative length")
	}
	return &Bitmap{bits: make([]byte, sizeOf(n)), n: n}
}

// NewSet returns a bitmap of n bits, all set. This is the common case for a
// column with no nulls, so it is worth having rather than looping over New.
func NewSet(n int) *Bitmap {
	b := New(n)
	for i := range b.bits {
		b.bits[i] = 0xFF
	}
	b.clearPadding()
	return b
}

// FromBytes wraps an existing byte slice as a bitmap of n bits without copying.
// The caller must not modify buf afterwards. It panics if buf is too short to
// hold n bits.
func FromBytes(buf []byte, n int) *Bitmap {
	if n < 0 {
		panic("bitmap: negative length")
	}
	if len(buf) < sizeOf(n) {
		panic("bitmap: buffer too short for length")
	}
	return &Bitmap{bits: buf, n: n}
}

// sizeOf returns the number of bytes needed to hold n bits.
func sizeOf(n int) int { return (n + 7) / 8 }

// Len returns the number of bits.
func (b *Bitmap) Len() int { return b.n }

// Bytes returns the underlying storage. The padding bits in the final byte are
// zero. Modifying the result modifies the bitmap.
func (b *Bitmap) Bytes() []byte { return b.bits }

// Get reports whether bit i is set. It panics if i is out of range, matching
// the behavior of an ordinary slice index.
func (b *Bitmap) Get(i int) bool {
	if uint(i) >= uint(b.n) {
		panic("bitmap: index out of range")
	}
	return b.bits[i>>3]&(1<<(uint(i)&7)) != 0
}

// Set sets bit i to v. It panics if i is out of range.
func (b *Bitmap) Set(i int, v bool) {
	if uint(i) >= uint(b.n) {
		panic("bitmap: index out of range")
	}
	mask := byte(1) << (uint(i) & 7)
	if v {
		b.bits[i>>3] |= mask
	} else {
		b.bits[i>>3] &^= mask
	}
}

// Append adds one bit to the end.
func (b *Bitmap) Append(v bool) {
	if need := sizeOf(b.n + 1); need > len(b.bits) {
		b.bits = append(b.bits, 0)
	}
	b.n++
	b.Set(b.n-1, v)
}

// CountOnes returns the number of set bits, which for a validity bitmap is the
// number of valid values.
func (b *Bitmap) CountOnes() int {
	n := 0
	for _, byt := range b.bits {
		n += bits.OnesCount8(byt)
	}
	return n
}

// Clone returns a copy that shares no memory with b.
func (b *Bitmap) Clone() *Bitmap {
	out := &Bitmap{bits: make([]byte, len(b.bits)), n: b.n}
	copy(out.bits, b.bits)
	return out
}

// Slice returns bits i through j-1 as a new bitmap, renumbered so that bit i
// becomes bit 0. It panics if the range is out of bounds.
//
// It copies rather than aliasing. A bitmap could carry a starting bit offset
// and slice in constant time, which is what an Arrow array does, but then Bytes
// would no longer return a buffer that begins at bit zero with zeroed padding,
// and every operation and every caller would have to carry an offset that is
// almost always zero. Copying costs one byte per eight bits, so slicing a chunk
// of eight thousand rows moves a kilobyte. The layers above this one keep their
// own offset for the places where constant time slicing is worth the
// complexity.
func (b *Bitmap) Slice(i, j int) *Bitmap {
	if i < 0 || j < i || j > b.n {
		panic("bitmap: slice out of range")
	}

	out := &Bitmap{bits: make([]byte, sizeOf(j-i)), n: j - i}
	if out.n == 0 {
		return out
	}

	// A byte aligned start is a plain copy, and it is the common case because
	// chunk boundaries are chosen to be round numbers.
	if i&7 == 0 {
		copy(out.bits, b.bits[i>>3:])
		out.clearPadding()
		return out
	}

	// Otherwise each output byte is the tail of one input byte and the head of
	// the next.
	shift := uint(i & 7)
	src := b.bits[i>>3:]
	for k := range out.bits {
		out.bits[k] = src[k] >> shift
		if k+1 < len(src) {
			out.bits[k] |= src[k+1] << (8 - shift)
		}
	}
	out.clearPadding()
	return out
}

// And sets b to the intersection of b and other. Both must have the same
// length. This is the operation that combines two validity bitmaps in a binary
// kernel, where a result is valid only if both inputs were.
func (b *Bitmap) And(other *Bitmap) {
	b.checkSameLen(other)
	for i := range b.bits {
		b.bits[i] &= other.bits[i]
	}
}

// Or sets b to the union of b and other. Both must have the same length.
func (b *Bitmap) Or(other *Bitmap) {
	b.checkSameLen(other)
	for i := range b.bits {
		b.bits[i] |= other.bits[i]
	}
}

// AndNot clears in b every bit that is set in other. Both must have the same
// length.
func (b *Bitmap) AndNot(other *Bitmap) {
	b.checkSameLen(other)
	for i := range b.bits {
		b.bits[i] &^= other.bits[i]
	}
}

// Xor sets b to the symmetric difference of b and other, meaning the bits that
// are set in exactly one of them. Both must have the same length.
func (b *Bitmap) Xor(other *Bitmap) {
	b.checkSameLen(other)
	for i := range b.bits {
		b.bits[i] ^= other.bits[i]
	}
}

// Not inverts every bit.
func (b *Bitmap) Not() {
	for i := range b.bits {
		b.bits[i] = ^b.bits[i]
	}
	b.clearPadding()
}

func (b *Bitmap) checkSameLen(other *Bitmap) {
	if b.n != other.n {
		panic("bitmap: length mismatch")
	}
}

// clearPadding zeroes the bits past the length in the final byte.
func (b *Bitmap) clearPadding() {
	if rem := b.n & 7; rem != 0 && len(b.bits) > 0 {
		b.bits[len(b.bits)-1] &= byte(1)<<uint(rem) - 1
	}
}
