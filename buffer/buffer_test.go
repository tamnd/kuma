package buffer_test

import (
	"bytes"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tamnd/kuma/buffer"
)

// aligned reports whether p starts on an Alignment boundary, computed here
// rather than through Buffer.Aligned so that the tests are not checking the
// implementation against itself.
func aligned(p []byte) bool {
	if len(p) == 0 {
		return true
	}
	return uintptr(unsafe.Pointer(&p[0]))%buffer.Alignment == 0
}

func TestNew(t *testing.T) {
	for _, n := range []int{0, 1, 8, 63, 64, 65, 1000, 1 << 16} {
		b := buffer.New(n)
		if b.Len() != n {
			t.Errorf("New(%d).Len() = %d", n, b.Len())
		}
		if b.Cap() < n || b.Cap()%buffer.Alignment != 0 {
			t.Errorf("New(%d).Cap() = %d, want a multiple of %d at least %d",
				n, b.Cap(), buffer.Alignment, n)
		}
		if !b.Aligned() || !aligned(b.Bytes()) {
			t.Errorf("New(%d) is not aligned", n)
		}
		if got := b.Bytes(); !bytes.Equal(got, make([]byte, n)) {
			t.Errorf("New(%d) is not zeroed: %v", n, got)
		}
	}
}

// TestNewIsAlignedEveryTime allocates a lot of sizes in a row, because a single
// allocation being aligned proves nothing. Go hands out consecutive addresses
// from a span, so a bug that only starts the region at the right place half the
// time would still pass a test that allocated once.
func TestNewIsAlignedEveryTime(t *testing.T) {
	for n := range 512 {
		b := buffer.New(n)
		if !aligned(b.Padded()) {
			t.Fatalf("New(%d) is not aligned", n)
		}
	}
}

func TestNewNegative(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("New(-1) did not panic")
		}
	}()
	buffer.New(-1)
}

func TestZeroValue(t *testing.T) {
	var b buffer.Buffer
	if b.Len() != 0 || b.Cap() != 0 {
		t.Fatalf("zero Buffer has Len %d Cap %d, want 0 and 0", b.Len(), b.Cap())
	}
	if !b.Aligned() {
		t.Error("zero Buffer is not aligned")
	}
	b.Append([]byte("hello"))
	if got := string(b.Bytes()); got != "hello" {
		t.Errorf("after Append the zero Buffer holds %q", got)
	}
	if !b.Aligned() {
		t.Error("the zero Buffer stopped being aligned once it grew")
	}
}

func TestAppend(t *testing.T) {
	var b buffer.Buffer
	var want []byte
	for i := range 300 {
		p := bytes.Repeat([]byte{byte(i)}, i%7)
		b.Append(p)
		want = append(want, p...)
		if !bytes.Equal(b.Bytes(), want) {
			t.Fatalf("after %d appends the contents diverged", i)
		}
		if !b.Aligned() {
			t.Fatalf("after %d appends the buffer is not aligned", i)
		}
	}
}

// TestAppendGrowsGeometrically pins down the cost of the thing callers do most.
// Appending a byte at a time to a megabyte has to reallocate on the order of
// twenty times, not on the order of sixteen thousand.
func TestAppendGrowsGeometrically(t *testing.T) {
	var b buffer.Buffer
	reallocs, last := 0, 0
	for range 1 << 20 {
		b.Append([]byte{1})
		if b.Cap() != last {
			reallocs++
			last = b.Cap()
		}
	}
	if reallocs > 20 {
		t.Errorf("appending a megabyte one byte at a time reallocated %d times", reallocs)
	}
}

func TestGrow(t *testing.T) {
	b := buffer.New(10)
	b.Grow(1000)
	if b.Len() != 10 {
		t.Errorf("Grow changed the length to %d", b.Len())
	}
	if b.Cap() < 1010 {
		t.Errorf("Grow(1000) on a buffer of 10 left Cap at %d", b.Cap())
	}

	// Growing into room that is already there must not reallocate, because the
	// executor calls Grow on every morsel and a copy per morsel is the whole
	// cost the pool exists to avoid.
	before := b.Cap()
	b.Grow(1)
	if b.Cap() != before {
		t.Errorf("Grow reallocated when there was room: Cap went %d to %d", before, b.Cap())
	}
}

func TestGrowNegative(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Grow(-1) did not panic")
		}
	}()
	buffer.New(0).Grow(-1)
}

func TestGrowKeepsContents(t *testing.T) {
	b := buffer.New(0)
	b.Append([]byte("kuma"))
	b.Grow(1 << 20)
	if got := string(b.Bytes()); got != "kuma" {
		t.Errorf("after Grow the buffer holds %q", got)
	}
}

func TestResize(t *testing.T) {
	b := buffer.New(4)
	copy(b.Bytes(), "abcd")

	b.Resize(2)
	if got := string(b.Bytes()); got != "ab" {
		t.Errorf("after shrinking to 2 the buffer holds %q", got)
	}

	// The bytes that were dropped must not come back. Reset and Resize are the
	// only two ways a buffer changes length, and the day one of them forgets to
	// zero is the day a column reports a value nobody wrote.
	b.Resize(4)
	if got := string(b.Bytes()); got != "ab\x00\x00" {
		t.Errorf("after growing back to 4 the buffer holds %q, want the tail zeroed", got)
	}

	b.Resize(0)
	if b.Len() != 0 {
		t.Errorf("Resize(0) left Len at %d", b.Len())
	}
}

func TestResizeNegative(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Resize(-1) did not panic")
		}
	}()
	buffer.New(0).Resize(-1)
}

func TestReset(t *testing.T) {
	b := buffer.New(64)
	before := b.Cap()
	b.Reset()
	if b.Len() != 0 {
		t.Errorf("after Reset Len is %d", b.Len())
	}
	if b.Cap() != before {
		t.Errorf("Reset gave up the memory: Cap went %d to %d", before, b.Cap())
	}
}

func TestZero(t *testing.T) {
	b := buffer.New(3)
	copy(b.Bytes(), "xyz")
	b.Zero()
	if got := b.Bytes(); !bytes.Equal(got, []byte{0, 0, 0}) {
		t.Errorf("after Zero the buffer holds %v", got)
	}
	if got := b.Padded(); !bytes.Equal(got, make([]byte, len(got))) {
		t.Error("Zero left the padding set")
	}
}

func TestPadded(t *testing.T) {
	b := buffer.New(65)
	if len(b.Padded()) != 128 {
		t.Errorf("New(65).Padded() is %d bytes, want 128", len(b.Padded()))
	}
	if len(b.Bytes()) != 65 {
		t.Errorf("New(65).Bytes() is %d bytes, want 65", len(b.Bytes()))
	}

	// This is the property the padding exists for: a kernel may read a whole
	// Alignment sized block past the last byte of data without leaving the
	// allocation.
	for _, n := range []int{1, 63, 64, 65, 127, 128, 129} {
		b := buffer.New(n)
		if len(b.Padded())%buffer.Alignment != 0 {
			t.Errorf("New(%d).Padded() is %d bytes, not a whole number of blocks",
				n, len(b.Padded()))
		}
		if len(b.Padded()) < n {
			t.Errorf("New(%d).Padded() is %d bytes, shorter than the data", n, len(b.Padded()))
		}
	}
}

func TestClone(t *testing.T) {
	b := buffer.New(4)
	copy(b.Bytes(), "abcd")

	c := b.Clone()
	if !bytes.Equal(c.Bytes(), b.Bytes()) {
		t.Fatalf("Clone holds %q, want %q", c.Bytes(), b.Bytes())
	}

	copy(c.Bytes(), "wxyz")
	if got := string(b.Bytes()); got != "abcd" {
		t.Errorf("writing to the clone changed the original to %q", got)
	}
	if !c.Aligned() {
		t.Error("the clone is not aligned")
	}
}

func TestWrap(t *testing.T) {
	// Deliberately offset by one, since the point of Wrap is bytes that came
	// from somewhere with its own ideas about layout.
	raw := make([]byte, 9)
	p := raw[1:]
	copy(p, "12345678")

	b := buffer.Wrap(p)
	if b.Len() != 8 || b.Cap() != 8 {
		t.Errorf("Wrap gave Len %d Cap %d, want 8 and 8", b.Len(), b.Cap())
	}
	if got := string(b.Bytes()); got != "12345678" {
		t.Errorf("Wrap holds %q", got)
	}

	// Writing through the buffer writes through to what was wrapped, since
	// nothing was copied.
	b.Bytes()[0] = 'x'
	if p[0] != 'x' {
		t.Error("Wrap copied when it should have aliased")
	}

	// Growing has to leave the caller's memory alone.
	b.Append([]byte("9"))
	if got := string(p); got != "x2345678" {
		t.Errorf("growing a wrapped buffer wrote back into the original: %q", got)
	}
	if !b.Aligned() {
		t.Error("a wrapped buffer that grew did not become aligned")
	}
	if got := string(b.Bytes()); got != "x23456789" {
		t.Errorf("after Append the buffer holds %q", got)
	}
}

func TestWrapReportsUnaligned(t *testing.T) {
	// Find an aligned address inside an ordinary allocation, then take one
	// slice starting there and one starting a byte later. Aligned has to
	// disagree about them, and doing it this way does not depend on where the
	// allocator happened to put anything.
	raw := make([]byte, 2*buffer.Alignment)
	off := int(-uintptr(unsafe.Pointer(&raw[0])) & (buffer.Alignment - 1))

	if !buffer.Wrap(raw[off:]).Aligned() {
		t.Error("Aligned said no to a slice starting on a boundary")
	}
	if buffer.Wrap(raw[off+1:]).Aligned() {
		t.Error("Aligned said yes to a slice starting one byte past a boundary")
	}
}

func TestWrapEmpty(t *testing.T) {
	b := buffer.Wrap(nil)
	if b.Len() != 0 || !b.Aligned() {
		t.Errorf("Wrap(nil) gave Len %d Aligned %v", b.Len(), b.Aligned())
	}
}

// TestNewDoesNotOverAllocate is the regression test for the memory a buffer
// costs beyond the bytes it hands out. Asking the allocator for 63 bytes of
// slack pushed a 4096 byte request into the 4864 byte size class, which is 19
// percent of the buffer rather than the one percent the 63 bytes suggest, and
// an engine holding gigabytes of columns cannot pay that quietly.
//
// The threshold is halfway between the two, so this fails when the padded path
// comes back for every allocation and not when something allocated a little
// extra somewhere.
func TestNewDoesNotOverAllocate(t *testing.T) {
	const (
		runs = 1000
		size = 4096
	)

	var before, after runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&before)
	for range runs {
		sink = buffer.New(size)
	}
	runtime.ReadMemStats(&after)

	per := float64(after.TotalAlloc-before.TotalAlloc) / runs
	if per > 4500 {
		t.Errorf("New(%d) costs %.0f bytes of allocator memory, want close to %d", size, per, size)
	}
}

func BenchmarkNew(b *testing.B) {
	for b.Loop() {
		sink = buffer.New(4096)
	}
}

func BenchmarkAppend(b *testing.B) {
	const n = 1 << 16
	chunk := make([]byte, 64)
	var buf buffer.Buffer
	b.SetBytes(n)
	for b.Loop() {
		buf.Reset()
		for range n / len(chunk) {
			buf.Append(chunk)
		}
	}
}

// BenchmarkAppendCold measures the same work from an empty buffer every time,
// which is the case that pays for every reallocation on the way up rather than
// reusing the memory from the previous round.
func BenchmarkAppendCold(b *testing.B) {
	const n = 1 << 16
	chunk := make([]byte, 64)
	b.SetBytes(n)
	for b.Loop() {
		var buf buffer.Buffer
		for range n / len(chunk) {
			buf.Append(chunk)
		}
		sink = &buf
	}
}

func BenchmarkGrow(b *testing.B) {
	for b.Loop() {
		buf := buffer.New(0)
		buf.Grow(1 << 16)
		sink = buf
	}
}

// sink keeps the benchmarks from being optimized away.
var sink *buffer.Buffer
