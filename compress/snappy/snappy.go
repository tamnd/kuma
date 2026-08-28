// Package snappy decompresses the snappy block format.
//
// Snappy is what most parquet files in the wild are compressed with, and it is
// the codec the parquet reader next door reaches for first. It is not in the
// standard library, so it is here, written against the format description
// rather than pulled in as a dependency. The whole of it is a few hundred lines
// and the format has not changed since 2011, which makes owning it cheaper than
// owning a dependency on it.
//
// The block format is the one this package reads, which is the compressed bytes
// on their own with nothing around them. That is what parquet stores in a page
// and what every other columnar format stores too. The framed format, which is
// the one snappy files on disk are in, wraps blocks in chunks with a stream
// header and checksums, and nothing here needs it.
//
// A block is its uncompressed length and then a run of tags. A tag either
// carries bytes of its own, which is a literal, or points back into what has
// been decompressed so far, which is a copy. That is the whole of it: snappy
// has no entropy coding and no dictionary, which is why it decompresses at
// something close to the speed of memcpy and why it is worth having when the
// alternative is reading four times as many bytes off a disk.
//
// A copy is allowed to reach back fewer bytes than it is long, which is how
// snappy writes a run: a thousand noughts are a literal of one and a copy of
// nine hundred and ninety nine reaching back one. So the bytes a copy reads are
// bytes that copy is writing, and it has to be done one at a time when that
// happens.
//
// Only decompression is here. Compression comes with the parquet writer, which
// is the first thing in this repository that will need it.
package snappy

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
)

// ErrCorrupt is a block that is not a snappy block, or is one that has been
// truncated or damaged. Use errors.Is rather than comparing, since it arrives
// wrapped in what was wrong with the block.
var ErrCorrupt = errors.New("bad snappy block")

// The bottom two bits of a tag say which of the four kinds it is. The other six
// hold a length, and for the first of the copies part of an offset as well.
const (
	tagLiteral = 0x00
	tagCopy1   = 0x01
	tagCopy2   = 0x02
	tagCopy4   = 0x03
)

// literalInline is the largest length less one that fits in the tag byte of a
// literal. The six bits go up to sixty three, and the four values above this
// one mean the length is written in that many bytes less fifty nine behind the
// tag instead.
const literalInline = 59

// shortRun is where a copy that overlaps itself stops being written a byte at a
// time and starts being written by doubling what is already there. It is a
// measured number rather than a derived one: below it the copies are shorter
// than the calls that make them, and a compressed page is full of short ones.
const shortRun = 16

// maxBlock is the largest block this can decompress. The length in front of a
// block is a varint of a thirty two bit number, so no block anywhere says more
// than this, and on a machine where an int is thirty two bits wide it is what
// that machine can address rather than what the format allows.
const maxBlock = min(math.MaxUint32, math.MaxInt)

// DecodedLen returns how many bytes the block decompresses to, which is written
// in front of it.
//
// It is worth having on its own because it is what a caller sizes a buffer
// with, and because it is a cheap way of asking whether a block is plausible
// before any of it is decompressed.
func DecodedLen(src []byte) (int, error) {
	n, _, err := decodedLen(src)
	return n, err
}

// decodedLen reads the length in front of a block and also returns how many
// bytes it took, which is where the tags start.
func decodedLen(src []byte) (length, header int, err error) {
	v, n := binary.Uvarint(src)
	switch {
	case n == 0:
		return 0, 0, fmt.Errorf("snappy: %w: the length of the block runs off the end of it", ErrCorrupt)
	case n < 0:
		return 0, 0, fmt.Errorf("snappy: %w: the length of the block is written in more than ten bytes", ErrCorrupt)
	case v > maxBlock:
		return 0, 0, fmt.Errorf("snappy: %w: a block of %d bytes", ErrCorrupt, v)
	}
	return int(v), n, nil
}

// Decode decompresses a block into dst and returns it.
//
// The dst is reused when it is large enough and a new one is allocated when it
// is not, in the way append works, so a caller reading page after page hands
// back what the last call returned and stops allocating after the first of
// them. Passing nil is fine and allocates every time.
//
// Whatever dst held is gone either way. A block says how long it is before it
// says anything else, so there is nothing to decompress into the end of.
func Decode(dst, src []byte) ([]byte, error) {
	size, header, err := decodedLen(src)
	if err != nil {
		return nil, err
	}

	if cap(dst) < size {
		dst = make([]byte, size)
	}
	dst = dst[:size]

	if err := decode(dst, src[header:]); err != nil {
		return nil, err
	}
	return dst, nil
}

// decode is the loop over the tags of a block, writing into a dst that is
// already the length the block said it would fill.
//
// Every tag is checked against what is left of both sides before it is acted
// on, since a block is somebody else's bytes and a length in one is a claim. A
// block that runs out of either is refused rather than truncated, because a
// short read of a compressed page is not a shorter page but a wrong one.
func decode(dst, src []byte) error {
	var d, s int
	for s < len(src) {
		var length, offset int

		switch tag := src[s]; tag & 0x03 {
		case tagLiteral:
			n := int(tag >> 2)
			s++
			if n > literalInline {
				// The length is behind the tag in one to four bytes, little
				// endian, and it is one less than the length as it is in the
				// tag. Read as a thirty two bit number rather than an int so
				// that the four byte case cannot wrap on a machine where an int
				// is thirty two bits wide.
				count := n - literalInline
				if count > len(src)-s {
					return fmt.Errorf("snappy: %w: the length of a literal in %d bytes with %d left",
						ErrCorrupt, count, len(src)-s)
				}
				var v uint32
				for i := range count {
					v |= uint32(src[s+i]) << (8 * i)
				}
				if uint64(v) >= uint64(len(dst)-d) {
					return fmt.Errorf("snappy: %w: a literal of %d bytes with room for %d",
						ErrCorrupt, uint64(v)+1, len(dst)-d)
				}
				n, s = int(v), s+count
			}
			n++

			if n > len(src)-s || n > len(dst)-d {
				return fmt.Errorf("snappy: %w: a literal of %d bytes with %d left and room for %d",
					ErrCorrupt, n, len(src)-s, len(dst)-d)
			}
			copy(dst[d:d+n], src[s:])
			d, s = d+n, s+n
			continue

		case tagCopy1:
			// Three bits of length, which is four to eleven, and eleven bits of
			// offset with the top three of them in the tag. It is the short one
			// and most of a compressed block is made of it.
			if len(src)-s < 2 {
				return fmt.Errorf("snappy: %w: a copy in 2 bytes with %d left", ErrCorrupt, len(src)-s)
			}
			length = 4 + int(tag>>2)&0x07
			offset = int(tag&0xe0)<<3 | int(src[s+1])
			s += 2

		case tagCopy2:
			if len(src)-s < 3 {
				return fmt.Errorf("snappy: %w: a copy in 3 bytes with %d left", ErrCorrupt, len(src)-s)
			}
			length = 1 + int(tag>>2)
			offset = int(binary.LittleEndian.Uint16(src[s+1:]))
			s += 3

		case tagCopy4:
			// A four byte offset, which the reference compressor never writes
			// because it works in blocks of sixty four kilobytes and cannot
			// reach further back than one of them. Another one may, and the
			// format says it is allowed to.
			if len(src)-s < 5 {
				return fmt.Errorf("snappy: %w: a copy in 5 bytes with %d left", ErrCorrupt, len(src)-s)
			}
			length = 1 + int(tag>>2)
			v := binary.LittleEndian.Uint32(src[s+1:])
			if uint64(v) > uint64(d) {
				return fmt.Errorf("snappy: %w: a copy reaching %d bytes back into %d",
					ErrCorrupt, v, d)
			}
			offset = int(v)
			s += 5
		}

		if offset == 0 || offset > d || length > len(dst)-d {
			return fmt.Errorf("snappy: %w: a copy of %d bytes reaching %d back into %d with room for %d",
				ErrCorrupt, length, offset, d, len(dst)-d)
		}
		copyBack(dst, d, offset, length)
		d += length
	}

	if d != len(dst) {
		return fmt.Errorf("snappy: %w: a block that says %d bytes and holds %d", ErrCorrupt, len(dst), d)
	}
	return nil
}

// copyBack writes the length bytes that start offset back from d.
//
// A copy that reaches back further than it is long is bytes that are all there
// already and moves in one go, which is most of them. A copy that does not is a
// run being written out, and the bytes it reads are bytes it is writing. That
// is not a corner case to be tidied away: it is how snappy says that the next
// nine hundred bytes are all the same, and a page of nulls or of one repeated
// string is mostly made of it.
//
// A long run takes as much as is behind it each time round rather than a byte
// at a time. What is behind it doubles every time, since it is everything from
// where the copy reaches back to as far as the copy has got, and the two never
// overlap because the source stops where the destination starts. A run of a
// thousand noughts reaching one byte back is ten copies rather than a thousand
// assignments, and one copy is what the rest of it costs once the pattern is as
// long as what is left.
//
// A short one is written a byte at a time anyway, because the copies of a run
// that short are shorter than the call that makes them.
func copyBack(dst []byte, d, offset, length int) {
	if length <= offset {
		copy(dst[d:d+length], dst[d-offset:d])
		return
	}

	end := d + length
	if length < shortRun {
		for ; d < end; d++ {
			dst[d] = dst[d-offset]
		}
		return
	}
	for start := d - offset; d < end; {
		d += copy(dst[d:end], dst[start:d])
	}
}
