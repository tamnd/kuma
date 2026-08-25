package strview_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tamnd/kuma/strview"
)

func TestBuilder(t *testing.T) {
	values := []string{"", "kuma", "exactly12345", "thirteen char", long}

	var b strview.Builder
	for i, v := range values {
		if b.Len() != i {
			t.Fatalf("Len() = %d after %d appends", b.Len(), i)
		}
		b.AppendString(v)
	}

	d := b.Finish()
	if d.Len() != len(values) {
		t.Fatalf("Len() = %d, want %d", d.Len(), len(values))
	}
	for i, want := range values {
		if got := string(d.At(i)); got != want {
			t.Errorf("At(%d) = %q, want %q", i, got, want)
		}
	}

	// Short values need no block at all, so a column of nothing but short
	// values allocates nothing beyond its views.
	if len(d.Blocks()) != 1 {
		t.Errorf("a column with one long value has %d blocks, want 1", len(d.Blocks()))
	}
}

func TestBuilderShortValuesUseNoBlocks(t *testing.T) {
	d := build("", "a", "kuma", "exactly12345")
	if got := len(d.Blocks()); got != 0 {
		t.Errorf("a column of short values allocated %d blocks", got)
	}
}

func TestBuilderCopiesWhatItIsGiven(t *testing.T) {
	// A caller reading records into one reused slice has to be able to build a
	// column out of it, so Append copies rather than aliasing.
	p := []byte(long)
	var b strview.Builder
	b.Append(p)
	copy(p, "OVERWRITTEN")

	if got := string(b.Finish().At(0)); got != long {
		t.Errorf("At(0) = %q, want %q", got, long)
	}
}

func TestBuilderZeroValue(t *testing.T) {
	var b strview.Builder
	if b.Len() != 0 {
		t.Fatalf("the zero Builder has Len %d", b.Len())
	}
	if d := b.Finish(); d.Len() != 0 {
		t.Fatalf("building the zero Builder gave %d values", d.Len())
	}
}

func TestBuilderGrow(t *testing.T) {
	var b strview.Builder
	b.Grow(100)
	if b.Len() != 0 {
		t.Fatalf("Grow changed the length to %d", b.Len())
	}
	for range 100 {
		b.AppendString("kuma")
	}
	d := b.Finish()
	if d.Len() != 100 {
		t.Fatalf("Len() = %d, want 100", d.Len())
	}
	if got := string(d.At(99)); got != "kuma" {
		t.Errorf("At(99) = %q", got)
	}

	// Growing twice must not lose what is already there.
	b.AppendString("first")
	b.Grow(10)
	b.AppendString("second")
	d = b.Finish()
	if got, want := string(d.At(0)), "first"; got != want {
		t.Errorf("At(0) = %q, want %q", got, want)
	}
}

func TestBuilderGrowNegative(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Grow(-1) did not panic")
		}
	}()
	var b strview.Builder
	b.Grow(-1)
}

func TestBuilderReset(t *testing.T) {
	var b strview.Builder
	b.AppendString(long)
	b.AppendString("kuma")
	b.Reset()

	if b.Len() != 0 {
		t.Fatalf("Len() = %d after Reset", b.Len())
	}
	d := b.Finish()
	if d.Len() != 0 || len(d.Blocks()) != 0 {
		t.Errorf("after Reset the builder still holds %d values and %d blocks",
			d.Len(), len(d.Blocks()))
	}
}

// TestBuilderBuildHandsOverItsMemory is the property that makes building a
// column cheap and the property that makes reusing a builder safe. Build gives
// its views and blocks to the Data and comes back empty, so the next Append
// cannot write into a column that has already been handed out.
func TestBuilderBuildHandsOverItsMemory(t *testing.T) {
	var b strview.Builder
	b.AppendString(long)
	first := b.Finish()

	if b.Len() != 0 {
		t.Fatalf("Len() = %d after Build", b.Len())
	}

	for range 100 {
		b.AppendString(strings.Repeat("z", 100))
	}
	second := b.Finish()

	if got := string(first.At(0)); got != long {
		t.Errorf("the first column now reads %q, want %q", got, long)
	}
	if second.Blocks()[0] == first.Blocks()[0] {
		t.Error("the second column is writing into the first column's block")
	}
}

func TestBuilderLongValues(t *testing.T) {
	// Values larger than a whole block get a block to themselves rather than
	// being split, since a view names one block and one offset.
	huge := strings.Repeat("kuma", 1<<16)
	var b strview.Builder
	b.AppendString("kuma")
	b.AppendString(huge)
	b.AppendString(huge)

	d := b.Finish()
	for i := 1; i <= 2; i++ {
		if got := d.At(i); string(got) != huge {
			t.Errorf("At(%d) returned %d bytes, want %d", i, len(got), len(huge))
		}
	}
	if d.View(1).Block() == d.View(2).Block() {
		t.Error("two values larger than a block landed in the same block")
	}
}

func TestBuilderManyValues(t *testing.T) {
	// Enough long values to run well past one block, checked by reading every
	// one of them back. This is the multi block path that a column of real
	// strings takes and a small test does not.
	const n = 4000
	var b strview.Builder
	for i := range n {
		b.AppendString(fmt.Sprintf("value number %d, padded out so it cannot be inlined", i))
	}

	d := b.Finish()
	if len(d.Blocks()) < 2 {
		t.Fatalf("%d long values fit in %d block, so this test proves nothing", n, len(d.Blocks()))
	}
	for i := range n {
		want := fmt.Sprintf("value number %d, padded out so it cannot be inlined", i)
		if got := string(d.At(i)); got != want {
			t.Fatalf("At(%d) = %q, want %q", i, got, want)
		}
	}
	if _, err := strview.NewData(d.Views(), d.Blocks()); err != nil {
		t.Errorf("NewData rejected a multi block column the builder made: %v", err)
	}
}

// buildBench returns a column of n values of the kind of string data this
// engine actually sees, which is mostly short with a long one every so often.
func buildBench(n int) *strview.Data {
	var b strview.Builder
	b.Grow(n)
	for i := range n {
		if i%8 == 0 {
			b.AppendString(fmt.Sprintf("a long value, number %d, well past twelve bytes", i))
		} else {
			b.AppendString(fmt.Sprintf("id%08d", i))
		}
	}
	return b.Finish()
}

func BenchmarkBuilderShort(b *testing.B) {
	const n = 1 << 14
	values := make([]string, n)
	for i := range values {
		values[i] = fmt.Sprintf("id%08d", i)
	}

	b.ReportAllocs()
	b.SetBytes(int64(n))
	for b.Loop() {
		var bld strview.Builder
		bld.Grow(n)
		for _, v := range values {
			bld.AppendString(v)
		}
		dataSink = bld.Finish()
	}
}

func BenchmarkBuilderLong(b *testing.B) {
	const n = 1 << 14
	values := make([]string, n)
	for i := range values {
		values[i] = fmt.Sprintf("a long value, number %d, well past twelve bytes", i)
	}

	b.ReportAllocs()
	b.SetBytes(int64(n))
	for b.Loop() {
		var bld strview.Builder
		bld.Grow(n)
		for _, v := range values {
			bld.AppendString(v)
		}
		dataSink = bld.Finish()
	}
}

// BenchmarkScan walks a whole column the way an aggregate does, which is the
// access pattern the fixed width view array exists to make fast.
func BenchmarkScan(b *testing.B) {
	d := buildBench(1 << 14)
	b.ReportAllocs()
	b.SetBytes(int64(d.Len()))
	for b.Loop() {
		total := 0
		for i := range d.Len() {
			total += len(d.At(i))
		}
		intSink = total
	}
}

// BenchmarkScanEqual is the probe side of a hash join or a group by: one value
// in hand, compared against every value in the column.
func BenchmarkScanEqual(b *testing.B) {
	d := buildBench(1 << 14)
	needle := []byte("id00004096")
	b.ReportAllocs()
	b.SetBytes(int64(d.Len()))
	for b.Loop() {
		hits := 0
		for i := range d.Len() {
			if d.EqualValue(i, needle) {
				hits++
			}
		}
		intSink = hits
	}
}

var dataSink *strview.Data
