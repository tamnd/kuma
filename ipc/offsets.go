package ipc

import (
	"encoding/binary"
	"fmt"

	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/strview"
)

// maxBlock is how much of the data buffer one block may cover. It is a
// variable rather than a constant only so that the tests can lower it and see
// the split happen without two gigabytes of anything.
var maxBlock int64 = strview.MaxValue

// viewsFromOffsets turns the offset layout into the view layout without moving
// any of the values.
//
// The offsets say where each value starts and ends in one long data buffer. A
// view says the same thing in a different shape: a length, the first four bytes
// of the value, and either the rest of the value inline or the block it lives
// in and where it starts inside that block. So the views are built, which is
// sixteen bytes per value, and the data buffer becomes a block that everything
// points into, which is where all the bytes are.
//
// The 64 bit layout needs more than one block when the data is larger than two
// gigabytes, since the offset inside a block is a signed 32 bit field. A new
// block starts at the first value that would not fit in the current one, and
// because a block boundary is always a value boundary, no value is ever split.
// Every block is a piece of the same buffer and none of it is copied.
//
// The numbers are read one at a time rather than by reinterpreting the buffer
// as a slice of integers. The offsets are in the byte order of the machine, so
// reinterpreting them would be correct, but it would also require the buffer to
// be aligned, and the work here is a pass over the values either way.
func viewsFromOffsets(offsets, data []byte, n, width int) (*strview.Data, error) {
	// One offset per value and one more after the last, compared by dividing
	// the buffer rather than multiplying the count, since a count near the top
	// of an int times four is a small number and the check would pass on the
	// way to allocating a view for every one of them.
	if len(offsets)/width-1 < n {
		return nil, fmt.Errorf("ipc: %w: %d values need %d bytes of offsets each and one more offset, the buffer has %d",
			ErrBuffers, n, width, len(offsets))
	}

	at := func(i int) int64 { return int64(int32(binary.NativeEndian.Uint32(offsets[i*4:]))) }
	if width == 8 {
		at = func(i int) int64 { return int64(binary.NativeEndian.Uint64(offsets[i*8:])) }
	}

	views := make([]strview.View, n)
	var blocks []*buffer.Buffer
	start := int64(0) // where the block being filled begins in data
	used := false     // whether any value points into the block being filled
	for i := range n {
		lo, hi := at(i), at(i+1)
		if lo < 0 || hi < lo || hi > int64(len(data)) {
			return nil, fmt.Errorf("ipc: %w: value %d runs from %d to %d of a data buffer of %d bytes",
				ErrBuffers, i, lo, hi, len(data))
		}
		v := data[lo:hi]
		if len(v) <= strview.MaxInline {
			views[i] = strview.MakeInline(v)
			continue
		}
		if len(v) > strview.MaxValue {
			return nil, fmt.Errorf("ipc: %w: value %d is %d bytes, which is more than a view can name",
				ErrBuffers, i, len(v))
		}
		if hi-start > maxBlock && lo > start {
			// The bytes in front of this value belong to the block being
			// filled only if something points at them. Otherwise they are the
			// inline values, which need no block, and the block simply starts
			// later.
			if used {
				blocks = append(blocks, buffer.Wrap(data[start:lo]))
			}
			start = lo
		}
		views[i] = strview.MakeRef(v, len(blocks), int(lo-start))
		used = true
	}

	// A column whose values all fit inside their views keeps no blocks at all,
	// and so does not hold the data buffer alive for nothing. That is also what
	// kuma's own builder produces for the same values, which keeps the two
	// spellings of one column from looking different.
	if used {
		blocks = append(blocks, buffer.Wrap(data[start:]))
	}

	return strview.NewData(views, blocks)
}
