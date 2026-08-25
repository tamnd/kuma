package bitmap

// Builder accumulates bits and hands over the finished Bitmap.
//
// Bitmap.Append works one bit at a time, which is the right shape when the
// values arrive one at a time. A builder exists for the case that dominates
// when a column is being read, which is a long run of the same bit: a chunk
// with no nulls is n set bits in a row, and AppendMany writes the whole bytes
// of that run as a fill instead of n calls that each recompute the same byte
// index and mask.
//
// The zero Builder is empty and ready to use.
type Builder struct {
	bits []byte
	n    int
}

// Len returns the number of bits appended so far.
func (b *Builder) Len() int { return b.n }

// Grow makes room for n more bits without appending them. It is worth calling
// when the final length is known, and harmless when it is only a guess.
func (b *Builder) Grow(n int) {
	if n < 0 {
		panic("bitmap: negative count")
	}
	b.grow(n)
}

// Append adds one bit to the end.
func (b *Builder) Append(v bool) {
	b.grow(1)
	b.setBit(b.n, v)
	b.n++
}

// AppendMany adds n copies of v to the end. AppendMany(true, n) is how a column
// with no nulls builds its validity bitmap, so it is the path worth keeping
// fast.
func (b *Builder) AppendMany(v bool, n int) {
	if n < 0 {
		panic("bitmap: negative count")
	}
	b.grow(n)
	end := b.n + n

	// The partial byte at the front, one bit at a time.
	for b.n < end && b.n&7 != 0 {
		b.setBit(b.n, v)
		b.n++
	}

	// Then whole bytes, which the compiler turns into a fill.
	if lo, hi := b.n>>3, end>>3; lo < hi {
		fill := byte(0)
		if v {
			fill = 0xFF
		}
		seg := b.bits[lo:hi]
		for k := range seg {
			seg[k] = fill
		}
		b.n = hi << 3
	}

	// Then whatever is left over at the end.
	for b.n < end {
		b.setBit(b.n, v)
		b.n++
	}
}

// AppendBools adds one bit per element of vals.
func (b *Builder) AppendBools(vals []bool) {
	b.grow(len(vals))
	for _, v := range vals {
		b.setBit(b.n, v)
		b.n++
	}
}

// Reset drops the accumulated bits and keeps the buffer for reuse.
func (b *Builder) Reset() {
	clear(b.bits)
	b.n = 0
}

// Finish returns the accumulated bits and resets the builder to empty.
//
// The returned bitmap takes over the buffer rather than copying it, so a
// builder cannot be used to observe or modify a bitmap it has already handed
// over. Call Finish once per bitmap.
func (b *Builder) Finish() *Bitmap {
	size := sizeOf(b.n)
	out := &Bitmap{bits: b.bits[:size:size], n: b.n}
	out.clearPadding()
	b.bits, b.n = nil, 0
	return out
}

func (b *Builder) setBit(i int, v bool) {
	mask := byte(1) << (uint(i) & 7)
	if v {
		b.bits[i>>3] |= mask
	} else {
		b.bits[i>>3] &^= mask
	}
}

// grow makes room for n more bits.
//
// Newly available bytes are cleared, which together with Reset clearing the
// buffer it keeps means unwritten storage is always zero. Every bit below the
// length gets written explicitly anyway, so this is not load bearing today. It
// costs one fill per growth and it removes the question from every method that
// touches b.bits.
func (b *Builder) grow(n int) {
	need := sizeOf(b.n + n)
	if need <= len(b.bits) {
		return
	}
	if need <= cap(b.bits) {
		b.bits = b.bits[:need]
	} else {
		grown := make([]byte, need, max(need, 2*cap(b.bits)))
		copy(grown, b.bits)
		b.bits = grown
	}
	clear(b.bits[sizeOf(b.n):])
}
