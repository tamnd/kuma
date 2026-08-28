package parquet

import (
	"encoding/binary"
	"fmt"
	"io"
)

// The encoding parquet calls DELTA_BINARY_PACKED, which writes the differences
// between values rather than the values.
//
// It is what a column of row numbers, of identifiers handed out in order, or of
// timestamps a second apart is written as, and those are most of the integer
// columns in a real file. A million row numbers are a million differences of
// one, which is a million bits and a header, where the values themselves are
// four megabytes. A dictionary cannot do that, since a million distinct numbers
// are a million distinct entries.
//
// The values are cut into blocks and each block into miniblocks. A block writes
// down the smallest difference in it and then one width per miniblock, and each
// miniblock packs its differences at its own width with the smallest one already
// taken off them, so a run that climbs by one and a run that climbs by a
// thousand and one both cost nothing but the header. Taking the smallest
// difference off is also what lets a column go down as well as up: the
// difference is signed and the packed number is not, and the two together are.
//
// The whole of a miniblock is written whether or not the values in it are all
// wanted, so the last one of a page holds padding past the end. The padding is
// read and thrown away rather than trusted, because a writer is free to leave
// anything in it.

// maxDeltaBlock is the largest block this decoder will believe. Everything
// writes blocks of a hundred and twenty eight values, the format says a block
// is a multiple of that, and a bound is what keeps the width of a miniblock in
// bits from running off the end of an int on a thirty two bit machine.
const maxDeltaBlock = 1 << 20

// maxDeltaWidth is the widest difference there can be, which is the width of
// the widest value the encoding is defined for.
const maxDeltaWidth = 64

// deltaValue is what a delta encoded page can hold. The format defines the
// encoding for the two integer widths and nothing else, which between them are
// every integer column parquet has, since it writes the narrower ones as int32.
type deltaValue interface {
	int32 | int64
}

// DeltaDecoder reads values written in the encoding parquet calls
// DELTA_BINARY_PACKED.
//
// The header of a page says how the blocks in it are cut up, so the decoder has
// to read that before it knows anything, which is why Reset returns an error
// where the other decoders here do not.
//
// The zero value is a decoder of no values. Use NewDeltaDecoder or Reset.
type DeltaDecoder struct {
	buf []byte
	pos int

	// miniblocks is how many miniblocks a block is cut into and perMini is how
	// many values are in each of them. Both are in the header of the page and
	// hold for every block in it.
	miniblocks int
	perMini    int

	// left is how many values have not been handed back, the first one
	// included, and value is what the next difference is added to. first says
	// whether value is a value that has not been handed back yet, which it is
	// once at the start of the page and never again.
	left  int
	value int64
	first bool

	// minDelta is the difference taken off every value of the block being read
	// and widths is the width of each of its miniblocks. mini is how many of
	// those widths have been used, so a block is done when it reaches
	// miniblocks and the next header is read.
	minDelta int64
	widths   []byte
	mini     int

	// miniLeft is how many values of the miniblock being read have not been
	// unpacked, width is how many bits each of them takes, and bit is the bit
	// the next one starts at.
	miniLeft int
	width    uint
	bit      int
}

// NewDeltaDecoder returns a decoder reading the values in data.
//
// The data is the values of a page and nothing else, the way PlainDecoder wants
// them, with the levels in front of them already taken off.
func NewDeltaDecoder(data []byte) (*DeltaDecoder, error) {
	d := &DeltaDecoder{}
	if err := d.Reset(data); err != nil {
		return nil, err
	}
	return d, nil
}

// Reset points the decoder at other bytes and reads the header in front of
// them, so that a scan reading a thousand pages of one column does not allocate
// a decoder for each of them.
//
// The header is how big a block is, how many miniblocks are in one, how many
// values the page holds and the first of those values. Everything after it is
// differences, and the first value is the only one written down as it is.
func (d *DeltaDecoder) Reset(data []byte) error {
	*d = DeltaDecoder{buf: data}

	block, err := d.count("the values of a delta block")
	if err != nil {
		return err
	}
	minis, err := d.count("the miniblocks of a delta block")
	if err != nil {
		return err
	}

	// A block is a multiple of a hundred and twenty eight values and a
	// miniblock is a multiple of thirty two, which is what makes a miniblock a
	// whole number of bytes at every width. A file that says otherwise has not
	// been written by anything that knows the format, and this is worth asking
	// before the counts behind it are read rather than after.
	switch {
	case block == 0 || block%128 != 0 || block > maxDeltaBlock:
		return fmt.Errorf("parquet: %w: a delta block of %d values", ErrFormat, block)
	case minis == 0 || block%minis != 0 || (block/minis)%32 != 0:
		return fmt.Errorf("parquet: %w: a delta block of %d values in %d miniblocks",
			ErrFormat, block, minis)
	}

	total, err := d.count("the values of a delta page")
	if err != nil {
		return err
	}
	first, err := d.varint("the first value of a delta page")
	if err != nil {
		return err
	}

	d.miniblocks, d.perMini = minis, block/minis
	d.left, d.value, d.first = total, first, total > 0

	// No block has been read yet, and a block is read when the miniblocks of
	// the one before it have all been used up.
	d.mini = minis
	return nil
}

// Read decodes values into dst and returns how many it wrote. It returns io.EOF
// once the page has been read to the end, in the way an io.Reader does.
//
// How many values a page holds is in its header rather than in how many bytes
// it takes, so this stops at that count and never at the end of the data. The
// bytes after it are the padding of the last miniblock.
func (d *DeltaDecoder) Read[T deltaValue](dst []T) (int, error) {
	n := 0
	for n < len(dst) && d.left > 0 {
		if d.first {
			dst[n] = T(d.value)
			d.first, d.left, n = false, d.left-1, n+1
			continue
		}
		if d.miniLeft == 0 {
			if err := d.miniblock(); err != nil {
				return n, err
			}
			continue
		}

		m := min(d.miniLeft, d.left, len(dst)-n)
		for i := range m {
			d.value += d.minDelta + int64(d.next())
			dst[n+i] = T(d.value)
		}
		d.miniLeft, d.left, n = d.miniLeft-m, d.left-m, n+m
	}
	if n == 0 && len(dst) > 0 {
		return 0, io.EOF
	}
	return n, nil
}

// miniblock gets the next miniblock ready to be unpacked, reading the header of
// the next block first when the one being read has no miniblocks left.
func (d *DeltaDecoder) miniblock() error {
	if d.mini == d.miniblocks {
		if err := d.block(); err != nil {
			return err
		}
	}

	d.width = uint(d.widths[d.mini])
	if d.width > maxDeltaWidth {
		return fmt.Errorf("parquet: %w: a delta miniblock of %d bit differences",
			ErrFormat, d.width)
	}
	d.mini++

	// A miniblock is written whole and the last one of a page is mostly
	// padding, and a writer is allowed to stop once it has written the last
	// value rather than the last byte. So what has to be in the page is the
	// bits of the values still wanted, and what the cursor moves on by is
	// whatever of the whole miniblock is there.
	whole := d.perMini * int(d.width) / 8
	want := (min(d.perMini, d.left)*int(d.width) + 7) / 8
	if want > len(d.buf)-d.pos {
		return fmt.Errorf("parquet: %w: a delta miniblock of %d bytes with %d left",
			ErrFormat, want, len(d.buf)-d.pos)
	}

	d.bit = d.pos * 8
	d.pos += min(whole, len(d.buf)-d.pos)
	d.miniLeft = d.perMini
	return nil
}

// block reads the header of the next block, which is the smallest difference in
// it and then the width of each of its miniblocks.
//
// The widths are left where they are rather than copied. There is one byte of
// them per miniblock however few of them the block ends up using, so a page
// that ends half way through a block still has the widths of the miniblocks it
// never wrote.
func (d *DeltaDecoder) block() error {
	minDelta, err := d.varint("the smallest difference of a delta block")
	if err != nil {
		return err
	}
	if d.miniblocks > len(d.buf)-d.pos {
		return fmt.Errorf("parquet: %w: %d miniblock widths with %d bytes left",
			ErrFormat, d.miniblocks, len(d.buf)-d.pos)
	}

	d.minDelta = minDelta
	d.widths = d.buf[d.pos : d.pos+d.miniblocks]
	d.pos += d.miniblocks
	d.mini = 0
	return nil
}

// next unpacks the next difference.
//
// The differences of a miniblock are one stream of bits rather than a value
// each, the same way a packed run of the RLE encoding is, so a value can start
// anywhere and one of sixty four bits can be spread over nine bytes. That is
// why this reads two words where the RLE decoder reads one: there its widths
// stop at thirty two and eight bytes always hold a value, and here a difference
// is as wide as the values are.
func (d *DeltaDecoder) next() uint64 {
	i, shift := d.bit>>3, uint(d.bit&7)
	d.bit += int(d.width)

	v := le64(d.buf, i) >> shift
	if shift > 0 && shift+d.width > 64 {
		v |= le64(d.buf, i+8) << (64 - shift)
	}
	if d.width < 64 {
		v &= 1<<d.width - 1
	}
	return v
}

// le64 reads the eight bytes at i, taking the ones past the end of b as nought.
//
// The bytes holding a value are checked against the page before any of them are
// read, so the bytes past the end that turn up here are the ones after the last
// value of the last miniblock rather than any part of a value.
func le64(b []byte, i int) uint64 {
	if i+8 <= len(b) {
		return binary.LittleEndian.Uint64(b[i:])
	}
	var tail [8]byte
	copy(tail[:], b[i:])
	return binary.LittleEndian.Uint64(tail[:])
}

// count reads one of the counts in the header of a page.
func (d *DeltaDecoder) count(what string) (int, error) {
	v, n := binary.Uvarint(d.buf[d.pos:])
	switch {
	case n == 0:
		return 0, fmt.Errorf("parquet: %w: %s runs off the end of the page", ErrFormat, what)
	case n < 0:
		return 0, fmt.Errorf("parquet: %w: %s is written in more than ten bytes", ErrFormat, what)
	case v > maxRun:
		return 0, fmt.Errorf("parquet: %w: %s is %d", ErrFormat, what, v)
	}
	d.pos += n
	return int(v), nil
}

// varint reads one of the signed numbers of the encoding, which are written
// zigzag so that a small negative number is as short as a small positive one.
func (d *DeltaDecoder) varint(what string) (int64, error) {
	v, n := binary.Varint(d.buf[d.pos:])
	switch {
	case n == 0:
		return 0, fmt.Errorf("parquet: %w: %s runs off the end of the page", ErrFormat, what)
	case n < 0:
		return 0, fmt.Errorf("parquet: %w: %s is written in more than ten bytes", ErrFormat, what)
	}
	d.pos += n
	return v, nil
}
