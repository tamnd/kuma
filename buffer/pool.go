package buffer

import (
	"math/bits"
	"sync"
)

// Size classes are powers of two from 64 bytes to 16 megabytes. Below 64 there
// is nothing to save, since that is the alignment and therefore the smallest
// allocation this package makes. Above 16 megabytes a buffer is large enough
// that the allocator is no longer the expensive part, and holding one alive in
// a pool costs more than making a new one.
const (
	minShift   = 6
	maxShift   = 24
	numClasses = maxShift - minShift + 1
)

// Pool recycles scratch buffers.
//
// The engine allocates and drops buffers in a very particular pattern: an
// operator takes one for the duration of a morsel, writes it, reads it, and is
// done with it before the next morsel starts. Those are all the same handful of
// sizes, they are all short lived, and there are a great many of them. That is
// the shape a free list is good at and the shape a garbage collector has to
// work hardest at, because every one of them survives long enough to be scanned
// and none of them survive long enough to be worth having been scanned.
//
// This is the only place in kuma where memory has a lifetime that somebody has
// to reason about, and it is deliberately kept to the one place where the
// reasoning is easy. Use it when the buffer cannot outlive the function that
// asked for it and the code makes that obvious. Anywhere else, call New and let
// the collector do its job.
//
// The zero Pool is empty and ready to use. It is safe for concurrent use, and
// it must not be copied, which go vet enforces because the sync.Pool inside it
// must not be copied either. It is worth having one per executor rather than
// one per process, so that two queries running at once do not fight over the
// same free lists.
type Pool struct {
	classes [numClasses]sync.Pool
}

// Get returns a buffer of n bytes.
//
// The bytes are whatever the previous user of that memory left in them. This
// is the difference between Get and New and it is the whole point: a scratch
// buffer that is about to be overwritten does not need to be zeroed first, and
// zeroing it would give back most of what the pool saves. Call Zero when the
// contents matter, or call New.
func (p *Pool) Get(n int) *Buffer {
	if n < 0 {
		panic("buffer: negative length")
	}
	class, ok := classFor(n)
	if !ok {
		return New(n)
	}
	if v := p.classes[class].Get(); v != nil {
		if b, ok := v.(*Buffer); ok {
			b.n = n
			return b
		}
	}
	return &Buffer{buf: alignedBytes(1 << (minShift + class)), n: n}
}

// Put offers a buffer back. The caller must not use it afterwards, and must not
// put the same buffer twice.
//
// A buffer whose capacity is not one of the size classes is dropped rather than
// kept, because a free list of odd sizes is a free list nothing can be served
// from. That includes wrapped buffers, whose memory belongs to whatever handed
// it over. Dropping is not a failure and there is nothing to report: the
// buffer is ordinary Go memory and the collector takes it from here.
func (p *Pool) Put(b *Buffer) {
	class, ok := classOf(len(b.buf))
	if !ok || !b.Aligned() {
		return
	}
	b.n = 0
	p.classes[class].Put(b)
}

// classFor returns the smallest class that holds n bytes.
func classFor(n int) (int, bool) {
	switch {
	case n > 1<<maxShift:
		return 0, false
	case n <= 1<<minShift:
		return 0, true
	}
	return bits.Len(uint(n-1)) - minShift, true
}

// classOf returns the class a buffer of exactly size bytes belongs to, and
// reports whether there is one.
func classOf(size int) (int, bool) {
	if size < 1<<minShift || size > 1<<maxShift || size&(size-1) != 0 {
		return 0, false
	}
	return bits.TrailingZeros(uint(size)) - minShift, true
}
