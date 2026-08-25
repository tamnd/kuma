// Package strview implements the Arrow variable size binary view layout, which
// is how kuma stores String and Binary columns.
//
// The classic Arrow layout for a string column is a buffer of offsets and a
// buffer of bytes, where element i runs from offsets[i] to offsets[i+1]. That
// is compact and it is a pointer chase for every single comparison, which is
// most of what an engine does to a string column.
//
// The view layout replaces the offsets with a fixed sixteen byte record per
// element:
//
//	a value of 12 bytes or fewer, kept inline
//	  bytes 0 to 3    length
//	  bytes 4 to 15   the value, zero padded
//
//	a value that is longer, kept in a data block
//	  bytes 0 to 3    length
//	  bytes 4 to 7    the first four bytes of the value
//	  bytes 8 to 11   which data block the value is in
//	  bytes 12 to 15  where in that block it starts
//
// Three things fall out of that. A value of twelve bytes or fewer is entirely
// inside its own view, and most real string data is short, so most values are
// read with no second memory access at all. Every view carries the first four
// bytes of its value in the same place whether it is short or long, so two
// values that differ in the first four bytes are ordered without touching the
// data at all. And a scan over a column becomes a dense walk over fixed width
// records, which vectorizes, where a walk over offsets does not.
//
// The cost is memory. Sixteen bytes per element against four or eight for an
// offset, and long values are never deduplicated the way a shared offsets
// buffer can leave them. Dictionary encoding is the answer for the columns
// where that matters, and it is a bigger win than either layout.
//
// Arrow calls the data blocks the data buffers and calls the block number the
// buffer index. They are blocks here because this repository already has a
// buffer package, and a buffer index into a list of buffers held in a buffer is
// one word too many.
//
// Everything is little endian, on every machine, because that is what Arrow
// writes on the wire and converting at the boundary is cheaper than converting
// on every access. On the machines anyone runs this on, that is a plain load.
//
// Stability: tier 1, stable.
package strview

import (
	"encoding/binary"
	"fmt"
	"math"
)

const (
	// Size is the width of one view in bytes.
	Size = 16
	// MaxInline is the longest value that fits inside a view. A value of this
	// length or shorter needs no data block.
	MaxInline = 12
	// PrefixLen is how many leading bytes of a value every view carries,
	// whether the value is inline or not.
	PrefixLen = 4
	// MaxValue is the longest value the layout can describe, and the highest
	// block number and offset it can name. The length, the block number and the
	// offset are all signed 32 bit fields, which is Arrow's choice rather than
	// this package's.
	MaxValue = math.MaxInt32
)

// View describes one value in a String or Binary column.
//
// It is a byte array rather than a struct of fields, so that a slice of views
// is exactly the bytes Arrow puts on the wire and can be handed across the C
// data interface without being rewritten. The accessors read little endian
// integers out of it.
//
// The zero View is a valid empty value, which means a column of views that has
// been sized but not filled reads as empty strings rather than as anything
// surprising.
type View [Size]byte

// MakeInline returns a view holding value inside itself. It panics if the value
// is longer than MaxInline.
//
// The bytes past the value are left zero, and they have to be: it is what makes
// two views of the same short value identical as raw bytes, so equality can be
// a sixteen byte comparison rather than a length check and a loop.
func MakeInline(value []byte) View {
	if len(value) > MaxInline {
		panic("strview: value too long to inline")
	}
	var v View
	binary.LittleEndian.PutUint32(v[0:4], uint32(len(value)))
	copy(v[4:], value)
	return v
}

// MakeRef returns a view for a value that lives at offset in block. It panics
// if the value is short enough to inline, since a column with two ways to spell
// the same value is a column where equality has to know about both. It also
// panics if the value, the block number or the offset is out of range, rather
// than truncating to 32 bits and describing a value that is not there.
func MakeRef(value []byte, block, offset int) View {
	if len(value) <= MaxInline {
		panic("strview: value short enough to inline")
	}
	if block < 0 || offset < 0 {
		panic("strview: negative block or offset")
	}
	if len(value) > MaxValue || block > MaxValue || offset > MaxValue {
		panic("strview: value, block or offset out of range")
	}
	var v View
	binary.LittleEndian.PutUint32(v[0:4], uint32(len(value)))
	copy(v[4:8], value)
	binary.LittleEndian.PutUint32(v[8:12], uint32(block))
	binary.LittleEndian.PutUint32(v[12:16], uint32(offset))
	return v
}

// Len returns the length of the value in bytes.
//
// The length is a signed 32 bit field in the Arrow layout, so a view that came
// from somewhere else can claim a negative length. This returns what it claims.
// Validate is what refuses it.
func (v View) Len() int { return int(int32(binary.LittleEndian.Uint32(v[0:4]))) }

// IsInline reports whether the value is inside the view rather than in a data
// block.
func (v View) IsInline() bool { return v.Len() <= MaxInline }

// Prefix returns the first PrefixLen bytes of the value, zero padded if the
// value is shorter than that.
//
// This is the reason the layout exists. Two values whose prefixes differ are
// ordered by their prefixes alone, with no access to any data block, and for
// real string data that settles the overwhelming majority of comparisons.
func (v View) Prefix() [PrefixLen]byte {
	return [PrefixLen]byte{v[4], v[5], v[6], v[7]}
}

// Block returns which data block holds the value. It is meaningless for an
// inline view.
func (v View) Block() int { return int(int32(binary.LittleEndian.Uint32(v[8:12]))) }

// Offset returns where in its block the value starts. It is meaningless for an
// inline view.
func (v View) Offset() int { return int(int32(binary.LittleEndian.Uint32(v[12:16]))) }

// Inline returns the bytes of an inline value, aliasing the view itself. The
// caller must not modify the result. It panics if the view is not inline.
//
// The receiver is a pointer so that the result aliases the caller's view rather
// than a copy of it, which is what keeps this from allocating.
func (v *View) Inline() []byte {
	if !v.IsInline() {
		panic("strview: view is not inline")
	}
	return v[4 : 4+v.Len()]
}

// String returns a description of the view for debugging. It does not return
// the value, since a long view cannot reach its own data.
func (v View) String() string {
	if v.IsInline() {
		return fmt.Sprintf("inline(%q)", v[4:4+max(v.Len(), 0)])
	}
	return fmt.Sprintf("ref(len %d, block %d, offset %d, prefix %q)",
		v.Len(), v.Block(), v.Offset(), v[4:8])
}

// prefixOf returns the prefix a value would have, which is the first PrefixLen
// bytes zero padded.
func prefixOf(value []byte) [PrefixLen]byte {
	var p [PrefixLen]byte
	copy(p[:], value)
	return p
}
