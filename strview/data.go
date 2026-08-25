package strview

import (
	"bytes"
	"fmt"

	"github.com/tamnd/kuma/buffer"
)

// Data is the value part of a String or Binary column: the views, and the
// blocks the long ones point into.
//
// It carries no validity bitmap and no dtype. Those belong to the array that
// holds this, because a null is not a value and this type is only about values.
type Data struct {
	views  []View
	blocks []*buffer.Buffer
}

// NewData returns a column over views and blocks after checking that every view
// describes something that is actually there. It does not copy either
// argument, and the caller must not modify them afterwards.
//
// The check is the point. Views that came from this process were built by
// Builder and are correct by construction, but views that arrived in an Arrow
// IPC message were written by something else, and a view that points past the
// end of a block is a read of somebody else's memory rather than a wrong
// answer. Validating once at the boundary is what lets every read after it be a
// slice with no bounds thinking in it.
func NewData(views []View, blocks []*buffer.Buffer) (*Data, error) {
	for i := range views {
		if err := validate(&views[i], blocks); err != nil {
			return nil, fmt.Errorf("strview: view %d: %w", i, err)
		}
	}
	return &Data{views: views, blocks: blocks}, nil
}

// validate reports what is wrong with one view, if anything.
func validate(v *View, blocks []*buffer.Buffer) error {
	n := v.Len()
	if n < 0 {
		return fmt.Errorf("negative length %d", n)
	}

	if v.IsInline() {
		// The bytes past the value have to be zero. Arrow requires it, and this
		// package relies on it every time it decides that two identical views
		// hold identical values without looking at either one.
		for _, c := range v[4+n:] {
			if c != 0 {
				return fmt.Errorf("inline value of %d bytes has a nonzero pad", n)
			}
		}
		return nil
	}

	block, off := v.Block(), v.Offset()
	if block < 0 || block >= len(blocks) {
		return fmt.Errorf("names block %d of %d", block, len(blocks))
	}
	b := blocks[block].Bytes()
	if off < 0 || off > len(b) || off+n > len(b) {
		return fmt.Errorf("wants bytes %d to %d of a block that is %d long", off, off+n, len(b))
	}
	if v.Prefix() != prefixOf(b[off:off+n]) {
		return fmt.Errorf("prefix does not match the first %d bytes of its value", PrefixLen)
	}
	return nil
}

// Len returns the number of values.
func (d *Data) Len() int { return len(d.views) }

// View returns the view for value i.
func (d *Data) View(i int) View { return d.views[i] }

// Views returns the views, which alias the column. Modifying them modifies it.
func (d *Data) Views() []View { return d.views }

// Blocks returns the data blocks, which alias the column.
func (d *Data) Blocks() []*buffer.Buffer { return d.blocks }

// At returns value i.
//
// The result aliases the column, either the view itself for a short value or a
// data block for a long one, and the caller must not modify it. Copying here
// would be a copy per element on the hottest path there is.
func (d *Data) At(i int) []byte {
	v := &d.views[i]
	if v.IsInline() {
		return v.Inline()
	}
	off, n := v.Offset(), v.Len()
	return d.blocks[v.Block()].Bytes()[off : off+n]
}

// Equal reports whether values i and j are the same bytes.
//
// Most calls are answered by the first two lines. Two views that are equal as
// sixteen raw bytes hold the same value, whether that is because both carry it
// inline or because both point at the same place. Two views with different
// lengths or different prefixes hold different values. Only values that are the
// same length and start the same way reach a byte comparison.
func (d *Data) Equal(i, j int) bool {
	vi, vj := &d.views[i], &d.views[j]
	if *vi == *vj {
		return true
	}
	if vi.Len() != vj.Len() || vi.Prefix() != vj.Prefix() {
		return false
	}
	return bytes.Equal(d.At(i), d.At(j))
}

// EqualValue reports whether value i is p. It is the probe side of a hash join
// or a group by lookup, where one side is a value already in hand.
func (d *Data) EqualValue(i int, p []byte) bool {
	v := &d.views[i]
	if v.Len() != len(p) || v.Prefix() != prefixOf(p) {
		return false
	}
	return bytes.Equal(d.At(i), p)
}

// Compare returns a negative number, zero or a positive number as value i sorts
// before, with or after value j, ordering by bytes the way bytes.Compare does.
//
// Prefixes settle it whenever they differ, which for real string data is nearly
// always, and they are four bytes sitting in a record the caller already had to
// load. Comparing zero padded prefixes gives the same answer as comparing the
// values: if the two prefixes first differ at a position past the end of the
// shorter value, then the shorter value is a prefix of the longer one and sorts
// first, which is what its pad byte of zero says.
func (d *Data) Compare(i, j int) int {
	vi, vj := &d.views[i], &d.views[j]
	if pi, pj := vi.Prefix(), vj.Prefix(); pi != pj {
		return bytes.Compare(pi[:], pj[:])
	}
	if *vi == *vj {
		return 0
	}
	return bytes.Compare(d.At(i), d.At(j))
}
