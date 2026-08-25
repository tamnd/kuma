package buffer_test

import (
	"sync"
	"testing"
	"unsafe"

	"github.com/tamnd/kuma/buffer"
)

func TestPoolGet(t *testing.T) {
	var p buffer.Pool
	for _, n := range []int{0, 1, 64, 65, 4096, 1 << 24} {
		b := p.Get(n)
		if b.Len() != n {
			t.Errorf("Get(%d).Len() = %d", n, b.Len())
		}
		if b.Cap() < n {
			t.Errorf("Get(%d).Cap() = %d, too small", n, b.Cap())
		}
		if !b.Aligned() {
			t.Errorf("Get(%d) is not aligned", n)
		}
	}
}

func TestPoolReuses(t *testing.T) {
	var p buffer.Pool
	first := p.Get(1000)
	p.Put(first)

	// sync.Pool is allowed to drop anything at any time, so this asks for the
	// same size until the memory comes back rather than demanding it on the
	// first try. What is being tested is that Put and Get agree on the size
	// class, not that sync.Pool has a memory.
	for range 100 {
		if got := p.Get(1000); got == first {
			return
		}
	}
	t.Error("a buffer that was put back never came out again")
}

func TestPoolRoundsUpToAClass(t *testing.T) {
	var p buffer.Pool
	b := p.Get(65)
	if b.Cap() != 128 {
		t.Errorf("Get(65).Cap() = %d, want the 128 byte class", b.Cap())
	}
	p.Put(b)

	// Anything up to the class size has to be servable from the same free list,
	// otherwise the pool holds memory it can never hand out.
	if got := p.Get(100); got.Cap() != 128 {
		t.Errorf("Get(100).Cap() = %d, want the same 128 byte class", got.Cap())
	}
}

func TestPoolPastTheLargestClass(t *testing.T) {
	var p buffer.Pool
	const huge = 1<<24 + 1

	b := p.Get(huge)
	if b.Len() != huge || !b.Aligned() {
		t.Errorf("Get(%d) gave Len %d Aligned %v", huge, b.Len(), b.Aligned())
	}

	// Putting it back is allowed and does nothing, which is the contract: a
	// caller should not have to know where the classes stop.
	p.Put(b)
	if got := p.Get(huge); got == b {
		t.Error("a buffer past the largest class was pooled after all")
	}
}

func TestPoolPutIgnoresOddSizes(t *testing.T) {
	var p buffer.Pool

	// Three buffers that must not end up on a free list. One has a capacity
	// that is not a size class, so nothing could ever be served from it. One is
	// shorter than the smallest class. And one is wrapped around memory that
	// starts off a boundary, which means it belongs to somebody else and is not
	// aligned either.
	raw := make([]byte, 4*buffer.Alignment)
	off := int(-uintptr(unsafe.Pointer(&raw[0])) & (buffer.Alignment - 1))

	p.Put(buffer.New(300))
	p.Put(buffer.Wrap(make([]byte, 32)))
	p.Put(buffer.Wrap(raw[off+1 : off+1+128]))

	for _, n := range []int{32, 128, 256, 300} {
		b := p.Get(n)
		if b.Cap() < n {
			t.Errorf("Get(%d) came back with Cap %d", n, b.Cap())
		}
		if !b.Aligned() {
			t.Errorf("Get(%d) came back unaligned, so a wrapped buffer got pooled", n)
		}
	}
}

func TestPoolPutResetsLength(t *testing.T) {
	var p buffer.Pool
	b := p.Get(64)
	p.Put(b)
	if b.Len() != 0 {
		t.Errorf("Put left the length at %d", b.Len())
	}
}

func TestPoolGetNegative(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Get(-1) did not panic")
		}
	}()
	var p buffer.Pool
	p.Get(-1)
}

// TestPoolConcurrent is the test that matters under the race detector. The
// executor calls this from every worker at once and a free list that is only
// correct on one goroutine is worse than no free list.
func TestPoolConcurrent(t *testing.T) {
	var p buffer.Pool
	var wg sync.WaitGroup
	for w := range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 500 {
				n := 64 << (i % 6)
				b := p.Get(n)
				if b.Len() != n {
					t.Errorf("worker %d got Len %d, want %d", w, b.Len(), n)
					return
				}
				// Write the whole thing, since the contents of a pooled buffer
				// are undefined and this is where two workers holding the same
				// memory would show up.
				for j := range b.Bytes() {
					b.Bytes()[j] = byte(w)
				}
				for j, c := range b.Bytes() {
					if c != byte(w) {
						t.Errorf("worker %d found %d at %d, so two workers have the same buffer", w, c, j)
						return
					}
				}
				p.Put(b)
			}
		}()
	}
	wg.Wait()
}

func BenchmarkPool(b *testing.B) {
	var p buffer.Pool
	for b.Loop() {
		buf := p.Get(4096)
		p.Put(buf)
	}
}

// BenchmarkPoolParallel is the shape the executor actually has, which is every
// worker asking at once.
func BenchmarkPoolParallel(b *testing.B) {
	var p buffer.Pool
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			buf := p.Get(4096)
			p.Put(buf)
		}
	})
}

// BenchmarkNoPool is the number the pool has to beat to be worth its
// complexity. If these two are close, delete the pool.
func BenchmarkNoPool(b *testing.B) {
	for b.Loop() {
		sink = buffer.New(4096)
	}
}
