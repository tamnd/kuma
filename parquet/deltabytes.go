package parquet

import (
	"fmt"
	"io"
)

// The two encodings parquet writes byte arrays as differences in, which are
// DELTA_LENGTH_BYTE_ARRAY and DELTA_BYTE_ARRAY.
//
// A plain page of byte arrays is a length and then the bytes, over and over, so
// a column of country codes costs six bytes a row to say two. The first of
// these encodings takes the lengths out and writes them as one block of
// differences, then puts all the bytes end to end behind that block. Nothing
// about the values changes, only where their lengths live.
//
// The second goes further and is for a column that is sorted, or grouped, or
// built out of a template, which is most of the columns anybody keys a table by.
// It writes how many bytes each value shares with the one in front of it and
// then the rest of that value, so a thousand paths under the same directory cost
// the directory once. A column of one repeated value is the far end of it: what
// is shared is the whole value every time and the rest of it is nothing.
//
// So the second is the first with a block of shared lengths in front of it, and
// that is how they are read here. The one thing they do not share is where a
// value lives. The first hands back slices of the page, the way a plain page
// does, and the second cannot, since a value of it is made of bytes from two
// places, so those are stitched into a buffer of the decoder's own.

// DeltaLengthDecoder reads values written in the encoding parquet calls
// DELTA_LENGTH_BYTE_ARRAY, which is the lengths of the values written as
// differences and then the bytes of all of them end to end.
//
// The zero value is a decoder of no values. Use Reset.
type DeltaLengthDecoder struct {
	// lengths reads the block in front of the page and size is what it holds,
	// which is the length of every value rather than of the ones still wanted:
	// the whole block has to be read to find where the bytes behind it start.
	lengths DeltaDecoder
	size    []int32

	// buf is the bytes of the values, count is how many values the page turned
	// out to hold, pos is how many of them have been handed back and at is
	// where in buf the next one starts.
	buf   []byte
	count int
	pos   int
	at    int
}

// Reset points the decoder at the values of a page and reads the lengths in
// front of them.
//
// The lengths are checked against the bytes here rather than as each value is
// handed back, since a page whose bytes run out short of what its lengths ask
// for is one thing to say once and not once per value. A page this refuses
// leaves a decoder of no values rather than of the ones it had got to, which is
// the only answer that means anything for half a page.
func (d *DeltaLengthDecoder) Reset(data []byte) error {
	d.count, d.pos, d.at = 0, 0, 0

	if err := d.lengths.Reset(data); err != nil {
		return err
	}
	n := d.lengths.Len()
	d.size = grow(d.size, n)
	if _, err := d.lengths.Read(d.size); err != nil {
		return err
	}
	d.buf = data[d.lengths.Offset():]

	total := 0
	for _, size := range d.size {
		if size < 0 || int(size) > len(d.buf)-total {
			return fmt.Errorf("parquet: %w: a value of %d bytes with %d left",
				ErrFormat, size, len(d.buf)-total)
		}
		total += int(size)
	}

	d.count = n
	return nil
}

// Len returns how many values the decoder has not handed back yet.
func (d *DeltaLengthDecoder) Len() int { return d.count - d.pos }

// Read hands back values into dst and returns how many it wrote. It returns
// io.EOF once the page has been read to the end, in the way an io.Reader does.
//
// The values point into the page rather than into a copy, the same way a plain
// page's values do, so whatever appends them takes its own copy.
func (d *DeltaLengthDecoder) Read(dst [][]byte) (int, error) {
	n := min(len(dst), d.count-d.pos)
	d.take(dst[:n])
	if n == 0 && len(dst) > 0 {
		return 0, io.EOF
	}
	return n, nil
}

// take fills dst with that many of the values the decoder has left.
//
// It is Read without the answer, because Reset checked the lengths against the
// bytes and what is left is slicing. The prefixed encoding reads a whole page
// of this one in one go and has nothing to say about an error that cannot
// happen, and a caller reading a buffer at a time wants the io.EOF that Read
// puts back on top.
func (d *DeltaLengthDecoder) take(dst [][]byte) {
	for i := range dst {
		size := int(d.size[d.pos])
		dst[i] = d.buf[d.at : d.at+size]
		d.at, d.pos = d.at+size, d.pos+1
	}
}

// DeltaByteArrayDecoder reads values written in the encoding parquet calls
// DELTA_BYTE_ARRAY, which is how much of each value the one in front of it
// already said and then the rest of it.
//
// The zero value is a decoder of no values. Use Reset.
type DeltaByteArrayDecoder struct {
	// prefixes reads the block of shared lengths and shared is what it holds.
	// suffixes reads everything behind that block, which is a page of the other
	// encoding, and rest is the values it hands back.
	prefixes DeltaDecoder
	shared   []int32
	suffixes DeltaLengthDecoder
	rest     [][]byte

	// buf is every value of the page stitched end to end and offset is where
	// each of them starts, with one more on the end for where the last one
	// stops. count and pos are what they are for the other decoder.
	buf    []byte
	offset []int
	count  int
	pos    int
}

// Reset points the decoder at the values of a page and puts all of them
// together.
//
// A value is made of bytes from the one in front of it and bytes of its own, so
// it cannot be a slice of the page the way the other decoders hand values back.
// It is built into a buffer the decoder keeps and reuses from page to page, and
// building the lot here rather than a value at a time is what lets the values be
// slices of that buffer: appending to it moves it, and a value handed back
// before the next one was built would be left pointing at where it used to be.
func (d *DeltaByteArrayDecoder) Reset(data []byte) error {
	d.count, d.pos = 0, 0

	if err := d.prefixes.Reset(data); err != nil {
		return err
	}
	n := d.prefixes.Len()
	d.shared = grow(d.shared, n)
	if _, err := d.prefixes.Read(d.shared); err != nil {
		return err
	}

	if err := d.suffixes.Reset(data[d.prefixes.Offset():]); err != nil {
		return err
	}
	if got := d.suffixes.Len(); got != n {
		return fmt.Errorf("parquet: %w: a delta page of %d values behind %d shared lengths",
			ErrFormat, got, n)
	}
	d.rest = grow(d.rest, n)
	d.suffixes.take(d.rest)

	d.buf, d.offset = d.buf[:0], grow(d.offset, n+1)
	d.offset[0] = 0
	for i, share := range d.shared {
		// How long the value in front of this one was, which for the first
		// value of a page is nothing at all. A value that shares more than
		// there is to share is a page that cannot be read, and taking as much
		// as there was would be making the value up.
		start, last := 0, 0
		if i > 0 {
			start, last = d.offset[i-1], d.offset[i]-d.offset[i-1]
		}
		if share < 0 || int(share) > last {
			return fmt.Errorf("parquet: %w: a value sharing %d bytes with one of %d",
				ErrFormat, share, last)
		}

		d.buf = append(d.buf, d.buf[start:start+int(share)]...)
		d.buf = append(d.buf, d.rest[i]...)
		d.offset[i+1] = len(d.buf)
	}

	d.count = n
	return nil
}

// Len returns how many values the decoder has not handed back yet.
func (d *DeltaByteArrayDecoder) Len() int { return d.count - d.pos }

// Read hands back values into dst and returns how many it wrote. It returns
// io.EOF once the page has been read to the end, in the way an io.Reader does.
//
// The values point into the decoder's own buffer and hold until the next Reset,
// so whatever appends them takes its own copy, which is what it would have to do
// for a plain page anyway.
func (d *DeltaByteArrayDecoder) Read(dst [][]byte) (int, error) {
	n := 0
	for n < len(dst) && d.pos < d.count {
		dst[n] = d.buf[d.offset[d.pos]:d.offset[d.pos+1]]
		d.pos, n = d.pos+1, n+1
	}
	if n == 0 && len(dst) > 0 {
		return 0, io.EOF
	}
	return n, nil
}
