package strview

import "github.com/tamnd/kuma/buffer"

// blockSize is how much of a data block a builder fills before starting the
// next one.
//
// 32 kilobytes is small enough that a block stays inside a modern L1 or L2 for
// the whole time it is being written, and large enough that the per block
// bookkeeping disappears. It also keeps every offset far inside the 32 bit
// field the layout gives it, so a column has to be enormous before block
// numbering is even a question.
const blockSize = 32 << 10

// Builder accumulates values into a Data.
//
// Short values go straight into their views. Long values are copied into a data
// block, so the builder owns its bytes and the caller may reuse the slice it
// passed in, which is what makes it safe to build a column out of a scanner's
// reused read buffer.
//
// The zero Builder is empty and ready to use.
type Builder struct {
	views  []View
	blocks []*buffer.Buffer
	// block is the size of a data block, so that a test can walk the multi
	// block path without building a column of megabytes. Zero means blockSize.
	block int
}

// Len returns how many values have been appended.
func (b *Builder) Len() int { return len(b.views) }

// Grow makes room for n more values without adding them.
func (b *Builder) Grow(n int) {
	if n < 0 {
		panic("strview: negative count")
	}
	if cap(b.views)-len(b.views) < n {
		views := make([]View, len(b.views), len(b.views)+n)
		copy(views, b.views)
		b.views = views
	}
}

// Append adds p, copying it if it is too long to live in its view.
//
// It panics if p is longer than MaxValue, or if the column has grown past what
// the block and offset fields can name. MakeRef is where that is enforced,
// rather than here as well, because a limit checked in two places is a limit
// that gets changed in one.
func (b *Builder) Append(p []byte) {
	if len(p) <= MaxInline {
		b.views = append(b.views, MakeInline(p))
		return
	}
	blk := b.blockFor(len(p))
	b.views = append(b.views, MakeRef(p, len(b.blocks)-1, blk.Len()))
	blk.Append(p)
}

// AppendString adds s. It is Append without the conversion, since a string to
// byte slice conversion of a long value would be a copy this package is about
// to make anyway.
func (b *Builder) AppendString(s string) {
	if len(s) <= MaxInline {
		b.views = append(b.views, MakeInline([]byte(s)))
		return
	}
	b.Append([]byte(s))
}

// blockFor returns a block with room for n bytes, starting a new one if the
// current block is full.
//
// A block that has nothing in it takes the value whatever its size, which is
// how a value larger than a whole block gets a block of its own rather than an
// infinite loop.
func (b *Builder) blockFor(n int) *buffer.Buffer {
	size := b.block
	if size <= 0 {
		size = blockSize
	}
	if len(b.blocks) > 0 {
		cur := b.blocks[len(b.blocks)-1]
		if cur.Len() == 0 || cur.Len()+n <= size {
			return cur
		}
	}
	blk := buffer.New(0)
	blk.Grow(max(size, n))
	b.blocks = append(b.blocks, blk)
	return blk
}

// Reset drops everything and leaves a builder ready to use again.
//
// It gives up the memory rather than keeping it, because the memory may have
// been handed to a Data by Finish and writing into it again would change a
// column somebody else is reading.
func (b *Builder) Reset() {
	b.views, b.blocks = nil, nil
}

// Finish returns the values appended so far and resets the builder.
//
// The Data takes the builder's memory rather than a copy of it, which is the
// only reason building a column is one allocation per block instead of two.
// Finish is what makes that safe: the builder comes back empty, so there is no
// way to write through it into a column that has already been handed out.
func (b *Builder) Finish() *Data {
	d := &Data{views: b.views, blocks: b.blocks}
	b.Reset()
	return d
}
