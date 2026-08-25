package strview

import "testing"

// These tests are inside the package so they can shrink the block size. The
// multi block path is worth testing at every boundary, and doing that through
// the exported API means building columns of megabytes to move a boundary that
// a sixty four byte block moves in three appends.

func TestBlockBoundary(t *testing.T) {
	b := Builder{block: 64}
	value := "a value of twenty!!!" // twenty bytes, so three fit in a block
	for range 7 {
		b.AppendString(value)
	}

	d := b.Build()
	if len(d.blocks) != 3 {
		t.Fatalf("seven twenty byte values in sixty four byte blocks used %d blocks, want 3",
			len(d.blocks))
	}

	// Three per block until the block cannot take another, which is the rule
	// blockFor implements, spelled out here as the answer rather than as the
	// same arithmetic a second time.
	want := []struct{ block, offset int }{
		{0, 0}, {0, 20}, {0, 40},
		{1, 0}, {1, 20}, {1, 40},
		{2, 0},
	}
	for i, w := range want {
		v := d.View(i)
		if v.Block() != w.block || v.Offset() != w.offset {
			t.Errorf("View(%d) is at block %d offset %d, want block %d offset %d",
				i, v.Block(), v.Offset(), w.block, w.offset)
		}
		if got := string(d.At(i)); got != value {
			t.Errorf("At(%d) = %q", i, got)
		}
	}

	if _, err := NewData(d.Views(), d.Blocks()); err != nil {
		t.Errorf("NewData rejected a multi block column the builder made: %v", err)
	}
}

func TestBlockTakesAValueLargerThanItself(t *testing.T) {
	// A view names one block and one offset, so a value that does not fit in a
	// block gets a block of its own rather than being split. Without the empty
	// block check in blockFor this is an allocation loop that never ends.
	b := Builder{block: 16}
	long := "a value of one hundred bytes, near enough, which is a good deal more than any block here holds at all"

	b.AppendString(long)
	b.AppendString(long)
	d := b.Build()

	if len(d.blocks) != 2 {
		t.Fatalf("two oversized values used %d blocks, want 2", len(d.blocks))
	}
	for i := range 2 {
		if got := string(d.At(i)); got != long {
			t.Errorf("At(%d) is %d bytes, want %d", i, len(got), len(long))
		}
	}
}

func TestBlockDefaultsWhenUnset(t *testing.T) {
	var b Builder
	if got := b.block; got != 0 {
		t.Fatalf("the zero Builder has a block size of %d", got)
	}
	b.AppendString("a value that does not fit inline")
	if got := b.blocks[0].Cap(); got < blockSize {
		t.Errorf("the zero Builder allocated a %d byte block, want at least %d", got, blockSize)
	}
}
