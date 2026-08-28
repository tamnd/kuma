package parquet

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// The two ways parquet writes a run of small integers.
//
// Definition levels, repetition levels and dictionary indices are all lists of
// numbers far smaller than the word they would sit in. A column nested three
// deep has levels of nought to three, and a column of a thousand distinct
// strings has indices of ten bits, so writing either of them as int32 would
// spend four bytes saying something two bits can say. Both are written packed
// into as many bits as the largest value needs, and the width is worked out
// once for the whole page: from the schema for levels, and from a byte at the
// front of the values for indices.
//
// The encoding the format calls RLE is really two encodings taking turns. A run
// of values is either a count and one value repeated that many times, or a
// count of groups of eight values packed end to end, and a one bit tag on the
// front of each run says which. That covers both of the shapes this data comes
// in: the definition levels of a column with no nulls in it are a hundred
// thousand ones, which is a repeat of two bytes, and dictionary indices that
// jump about are packed. A writer switches between them as it goes.
//
// The other one is what parquet did before that and is only ever levels. It is
// the same packing with no runs and no counts, so a reader has to know how many
// values to expect, and it packs them the other way up. Nothing writes it any
// more and the files that have it are not going anywhere. A reader that assumed
// RLE would read one of them as noise rather than refusing it, which is the
// worse of the two ways to be wrong.

// maxWidth is the widest value either decoder will read. The format says a
// dictionary index is an int32 and writes its width in one byte, so a width
// above this is a byte that means something else.
const maxWidth = 32

// maxRun is the longest run this package will believe. A page holds at most an
// int32 of values, so a run longer than that is longer than any page it could
// have come out of.
const maxRun = math.MaxInt32

// RLEDecoder reads values written in the encoding parquet calls RLE, which is
// the hybrid of repeated runs and bit packed runs.
//
// The width is how many bits each value takes and is not in the data, so it has
// to be worked out from the schema or read from the front of the page by
// whoever is calling. A width of nought is a column that only has one possible
// value, which happens to every level of a required column, and it is a real
// width rather than an error: the runs are still counted, so the decoder still
// knows how many values there are.
//
// The zero value is a decoder of no values. Use NewRLEDecoder or Reset.
type RLEDecoder struct {
	buf []byte

	// pos is the byte the next run header starts at, which is also the end of
	// the run being read.
	pos int

	width uint
	mask  uint64

	// left is how many values of the current run have not come back yet, and
	// packed says which of the two kinds of run it is. value is what a repeated
	// run repeats, and bit is the bit a packed run has got to.
	left   int
	packed bool
	value  int32
	bit    int
}

// NewRLEDecoder returns a decoder reading width bit values out of data.
//
// The data is the run of bytes and nothing around it. Neither of the two things
// the format puts in front of these bytes is read here: a data page written the
// first way puts four bytes of length in front of each of its two level runs,
// and dictionary indices have their width in the byte in front of them, and
// both of those belong to whoever is taking the page apart.
func NewRLEDecoder(data []byte, width int) (*RLEDecoder, error) {
	d := &RLEDecoder{}
	if err := d.Reset(data, width); err != nil {
		return nil, err
	}
	return d, nil
}

// Reset points the decoder at other bytes, so that a scan reading a thousand
// pages of one column does not allocate a decoder for each of them.
func (d *RLEDecoder) Reset(data []byte, width int) error {
	if width < 0 || width > maxWidth {
		return fmt.Errorf("parquet: %w: values of %d bits", ErrFormat, width)
	}
	*d = RLEDecoder{buf: data, width: uint(width), mask: 1<<uint(width) - 1}
	return nil
}

// Read decodes values into dst and returns how many it wrote. It returns io.EOF
// once the data has been read to the end, in the way an io.Reader does: the
// call that reads the last values returns them with a nil error and the call
// after it returns io.EOF.
//
// A run that the data does not hold is an error, and values decoded before the
// bad run are returned along with it, since a caller that wanted the levels of
// a page has nothing to do with a prefix of them but a caller counting what it
// got is entitled to see the count.
func (d *RLEDecoder) Read(dst []int32) (int, error) {
	n := 0
	for n < len(dst) {
		if d.left == 0 {
			// A run of no values is legal and says nothing, so this goes round
			// again rather than returning. It cannot go round for ever because
			// every run header is at least one byte.
			if err := d.run(); err != nil {
				if errors.Is(err, io.EOF) && n > 0 {
					return n, nil
				}
				return n, err
			}
			continue
		}

		m := min(d.left, len(dst)-n)
		if d.packed {
			for i := range m {
				dst[n+i] = d.unpack()
			}
		} else {
			for i := range m {
				dst[n+i] = d.value
			}
		}
		d.left -= m
		n += m
	}
	return n, nil
}

// run reads the header of the next run and gets the decoder ready to hand back
// the values in it.
func (d *RLEDecoder) run() error {
	if d.pos >= len(d.buf) {
		return io.EOF
	}

	// The header is a varint holding the length of the run and, in its lowest
	// bit, which kind of run it is. A repeat counts values and a packed run
	// counts groups of eight, because packing is always done eight at a time so
	// that a group of any width is a whole number of bytes.
	h, n := binary.Uvarint(d.buf[d.pos:])
	switch {
	case n == 0:
		return fmt.Errorf("parquet: %w: a run header that runs off the end of the data", ErrFormat)
	case n < 0:
		return fmt.Errorf("parquet: %w: a run header of more than ten bytes", ErrFormat)
	}
	d.pos += n

	if h&1 == 0 {
		return d.repeated(h >> 1)
	}
	return d.bitPacked(h >> 1)
}

// repeated reads a run of one value written once and repeated count times.
func (d *RLEDecoder) repeated(count uint64) error {
	if count > maxRun {
		return fmt.Errorf("parquet: %w: a run of %d values", ErrFormat, count)
	}

	// The value takes as many whole bytes as its width needs, which is not the
	// same as how many bits it uses: a width of ten is two bytes.
	n := int(d.width+7) / 8
	if n > len(d.buf)-d.pos {
		return fmt.Errorf("parquet: %w: a repeated value of %d bytes with %d left",
			ErrFormat, n, len(d.buf)-d.pos)
	}

	var v uint64
	for i := range n {
		v |= uint64(d.buf[d.pos+i]) << (8 * uint(i))
	}
	if v > d.mask {
		return fmt.Errorf("parquet: %w: a repeated value of %d in %d bits", ErrFormat, v, d.width)
	}

	d.pos += n
	d.packed = false
	d.value = int32(v)
	d.left = int(count)
	return nil
}

// bitPacked reads the header of a run of groups of eight packed values. The
// values themselves are left where they are and unpacked as they are asked for.
func (d *RLEDecoder) bitPacked(groups uint64) error {
	if groups > maxRun/8 {
		return fmt.Errorf("parquet: %w: a packed run of %d groups", ErrFormat, groups)
	}

	// Eight values of width bits are width bytes, which is the whole reason the
	// format packs in eights.
	n := groups * uint64(d.width)
	if n > uint64(len(d.buf)-d.pos) {
		return fmt.Errorf("parquet: %w: a packed run of %d bytes with %d left",
			ErrFormat, n, len(d.buf)-d.pos)
	}

	d.packed = true
	d.bit = d.pos * 8
	d.pos += int(n)
	d.left = int(groups) * 8
	return nil
}

// unpack reads the next packed value.
//
// The bits of a run are one stream rather than a value per byte, so a value of
// five bits is the top three bits of one byte and the bottom two of the next,
// and a value can start anywhere. Eight bytes are read at once and the value
// shifted out of them, which is why the width stops at thirty two: a value of
// thirty two bits starting seven bits into a byte needs thirty nine, and there
// are sixty four to take them from.
//
// The bytes of a run have been checked against the buffer before this is
// called, so the only reason to read fewer than eight is that the run ends near
// the end of the buffer, and the bits past the end are zero either way.
func (d *RLEDecoder) unpack() int32 {
	i, shift := d.bit>>3, uint(d.bit&7)
	d.bit += int(d.width)

	var v uint64
	if i+8 <= len(d.buf) {
		v = binary.LittleEndian.Uint64(d.buf[i:])
	} else {
		var tail [8]byte
		copy(tail[:], d.buf[i:])
		v = binary.LittleEndian.Uint64(tail[:])
	}
	return int32((v >> shift) & d.mask)
}

// BitPackedDecoder reads levels written in the encoding parquet calls
// BIT_PACKED, which is the one it deprecated and which old files still have.
//
// It is packed values and nothing else: no runs, no counts, and no length. How
// many values there are is not in the data, so the decoder reads to the end of
// what it was given and the caller keeps the ones it asked for. That is what
// the format intends, since a page says how many values it holds and the levels
// are as long as they need to be to hold that many.
//
// The bits go the other way up here than they do in a packed run of the
// encoding that replaced this one. A value is read from the top of a byte down
// rather than from the bottom up, which is the only difference between the two
// and is enough to turn every value into a different one.
//
// A width of nought reads no values, since a value of no bits leaves nothing to
// count them with. Levels that wide belong to a required column and are not
// written down at all.
//
// The zero value is a decoder of no values. Use NewBitPackedDecoder or Reset.
type BitPackedDecoder struct {
	buf   []byte
	width uint

	// bit is the bit the next value starts at, and end is where the last whole
	// value ends. The format pads a run up to a byte, so the bits between the
	// two are padding rather than a value.
	bit int
	end int
}

// NewBitPackedDecoder returns a decoder reading width bit values out of data.
func NewBitPackedDecoder(data []byte, width int) (*BitPackedDecoder, error) {
	d := &BitPackedDecoder{}
	if err := d.Reset(data, width); err != nil {
		return nil, err
	}
	return d, nil
}

// Reset points the decoder at other bytes.
func (d *BitPackedDecoder) Reset(data []byte, width int) error {
	if width < 0 || width > maxWidth {
		return fmt.Errorf("parquet: %w: values of %d bits", ErrFormat, width)
	}

	*d = BitPackedDecoder{buf: data, width: uint(width)}
	if width > 0 {
		d.end = len(data) * 8 / width * width
	}
	return nil
}

// Read decodes values into dst and returns how many it wrote. It returns io.EOF
// once the data has been read to the end.
func (d *BitPackedDecoder) Read(dst []int32) (int, error) {
	n := 0
	for n < len(dst) && d.bit < d.end {
		i, shift := d.bit>>3, uint(d.bit&7)
		d.bit += int(d.width)

		var v uint64
		if i+8 <= len(d.buf) {
			v = binary.BigEndian.Uint64(d.buf[i:])
		} else {
			var tail [8]byte
			copy(tail[:], d.buf[i:])
			v = binary.BigEndian.Uint64(tail[:])
		}

		// The eight bytes are read most significant first because that is the
		// order the values are packed in, so the value wanted is the width bits
		// under the ones already read.
		dst[n] = int32(v << shift >> (64 - d.width))
		n++
	}
	if n == 0 {
		return 0, io.EOF
	}
	return n, nil
}
