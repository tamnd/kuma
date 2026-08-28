package parquet

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// The plain encoding, which is the values written down as they are.
//
// Every other encoding in the format is a way of not doing this, and every one
// of them still ends up here: a dictionary page is plain values, and a page of
// dictionary indices is the indices in front of the plain values they point at.
// So this is the bottom of the reader, and it is the one part of it that a file
// of a hundred million numbers spends most of its time in.
//
// A value is little endian and takes the same number of bytes as the physical
// type it was written as, with three exceptions. A boolean is one bit rather
// than one byte, packed eight to a byte from the bottom up. A byte array writes
// four bytes of length in front of every value, which is why reading one is the
// only thing here that cannot skip ahead. And an int96 is a timestamp nobody
// writes any more and everybody still reads, twelve bytes holding a count of
// nanoseconds into a day and then the day.
//
// Nothing is copied. A byte array comes back pointing into the page it was
// read from, so a column of a million strings costs the page and the slice
// headers, and whoever keeps a value for longer than the page keeps a copy.

// julianEpoch is the Julian day the first of January 1970 fell on, which is
// what an int96 timestamp has to be moved by to count from the epoch everything
// else in kuma counts from.
const julianEpoch = 2440588

// nanosPerDay is the width of a day in the unit an int96 counts in.
const nanosPerDay = 24 * 60 * 60 * 1000 * 1000 * 1000

// maxJulianDays is how far from the epoch a day can be and still be a day an
// int64 of nanoseconds can hold, which is a little under two hundred and ninety
// two years either way.
const maxJulianDays = math.MaxInt64 / nanosPerDay

// PlainDecoder reads values written in the plain encoding.
//
// It does not know what type it is reading. A page says what its column's
// physical type is and the caller reads the values with the method for it,
// which is one method per type and no conversions: a column of int32 is read
// with Int32 and nothing else, because a value read at the wrong width is not a
// wrong value but a different value at every position after it.
//
// The zero value is a decoder of no values. Use NewPlainDecoder or Reset.
type PlainDecoder struct {
	buf []byte
	pos int

	// bit is how far into the byte at pos a run of booleans has got. Nothing
	// else in the encoding is smaller than a byte, so every other method starts
	// by moving past whatever byte a run of booleans left half read.
	bit uint8
}

// NewPlainDecoder returns a decoder reading the values in data.
//
// The data is the values of a page and nothing else. The levels in front of
// them belong to whoever is taking the page apart, since how long they are is
// in the page header for one version of the page and in front of them for the
// other.
func NewPlainDecoder(data []byte) *PlainDecoder {
	return &PlainDecoder{buf: data}
}

// Reset points the decoder at other bytes, so that a scan reading a thousand
// pages of one column does not allocate a decoder for each of them.
func (d *PlainDecoder) Reset(data []byte) { *d = PlainDecoder{buf: data} }

// Left is how many bytes have not been read. It is what a caller reading byte
// arrays has instead of a count, since the only way to know how many values are
// in a run of them is to walk it.
func (d *PlainDecoder) Left() int {
	if d.bit != 0 {
		return len(d.buf) - d.pos - 1
	}
	return len(d.buf) - d.pos
}

// Int32 reads values written as int32, which is every integer parquet stores in
// four bytes or fewer, a date, and a time of day in milliseconds.
func (d *PlainDecoder) Int32(dst []int32) (int, error) {
	n, b, err := d.take(4, len(dst))
	if err != nil {
		return 0, err
	}
	for i := range n {
		dst[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return n, nil
}

// Int64 reads values written as int64, which is every wider integer and every
// timestamp the format has a logical type for.
func (d *PlainDecoder) Int64(dst []int64) (int, error) {
	n, b, err := d.take(8, len(dst))
	if err != nil {
		return 0, err
	}
	for i := range n {
		dst[i] = int64(binary.LittleEndian.Uint64(b[8*i:]))
	}
	return n, nil
}

// Float reads values written as float, which is four bytes in the layout every
// machine that matters agrees on.
func (d *PlainDecoder) Float(dst []float32) (int, error) {
	n, b, err := d.take(4, len(dst))
	if err != nil {
		return 0, err
	}
	for i := range n {
		dst[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return n, nil
}

// Double reads values written as double.
func (d *PlainDecoder) Double(dst []float64) (int, error) {
	n, b, err := d.take(8, len(dst))
	if err != nil {
		return 0, err
	}
	for i := range n {
		dst[i] = math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:]))
	}
	return n, nil
}

// Int96 reads values written as int96 and returns them as nanoseconds since the
// epoch.
//
// The twelve bytes are a count of nanoseconds into a day and then the Julian
// day it falls in, which is a day number counted from a morning in 4713 BC. The
// conversion is done here rather than left to the caller because the value is a
// timestamp and nothing else: no writer ever put anything but a timestamp in
// one, and the format has no way of saying what else it might be.
//
// What the timestamp is in is another matter. Some writers wrote UTC and some
// wrote whatever the machine's clock said, and there is nothing in the file to
// tell them apart, which is why the schema gives an int96 column no zone.
func (d *PlainDecoder) Int96(dst []int64) (int, error) {
	n, b, err := d.take(12, len(dst))
	if err != nil {
		return 0, err
	}
	for i := range n {
		v, err := int96Nanos(b[12*i:])
		if err != nil {
			return i, err
		}
		dst[i] = v
	}
	return n, nil
}

// int96Nanos turns the twelve bytes of an int96 into nanoseconds since the
// epoch, refusing the ones that will not fit in an int64.
func int96Nanos(b []byte) (int64, error) {
	nanos := int64(binary.LittleEndian.Uint64(b))
	day := int64(binary.LittleEndian.Uint32(b[8:])) - julianEpoch
	if day > maxJulianDays || day < -maxJulianDays {
		return 0, fmt.Errorf("parquet: %w: a timestamp %d days from the epoch", ErrFormat, day)
	}

	// The day fits and the nanoseconds into it should be less than a day, but
	// nothing in the format says so, and a writer that wrote a whole timestamp
	// into the first eight bytes would land here rather than a hundred years
	// out.
	whole := day * nanosPerDay
	if (nanos > 0 && whole > math.MaxInt64-nanos) || (nanos < 0 && whole < math.MinInt64-nanos) {
		return 0, fmt.Errorf("parquet: %w: a timestamp %d days and %d nanoseconds from the epoch",
			ErrFormat, day, nanos)
	}
	return whole + nanos, nil
}

// Boolean reads values written as boolean, which are one bit each and packed
// eight to a byte from the bottom up.
//
// The bits run on across the whole page rather than starting again per call, so
// a page of three booleans is one byte with five bits in it that are not
// values.
func (d *PlainDecoder) Boolean(dst []bool) (int, error) {
	n := 0
	for n < len(dst) && d.pos < len(d.buf) {
		dst[n] = d.buf[d.pos]>>d.bit&1 == 1
		n++
		d.bit++
		if d.bit == 8 {
			d.bit, d.pos = 0, d.pos+1
		}
	}
	if n == 0 && len(dst) > 0 {
		return 0, io.EOF
	}
	return n, nil
}

// ByteArray reads values written as byte_array, which is four bytes of length
// and then that many bytes.
//
// The values point into the data rather than copying out of it, so they are
// good for as long as the page they were read from is.
func (d *PlainDecoder) ByteArray(dst [][]byte) (int, error) {
	d.align()

	n := 0
	for n < len(dst) && d.pos < len(d.buf) {
		if len(d.buf)-d.pos < 4 {
			return n, fmt.Errorf("parquet: %w: a length of four bytes with %d left",
				ErrFormat, len(d.buf)-d.pos)
		}
		size := int64(binary.LittleEndian.Uint32(d.buf[d.pos:]))
		if size > int64(len(d.buf)-d.pos-4) {
			return n, fmt.Errorf("parquet: %w: a value of %d bytes with %d left",
				ErrFormat, size, len(d.buf)-d.pos-4)
		}

		d.pos += 4
		dst[n] = d.buf[d.pos : d.pos+int(size)]
		d.pos += int(size)
		n++
	}
	if n == 0 && len(dst) > 0 {
		return 0, io.EOF
	}
	return n, nil
}

// Fixed reads values written as fixed_len_byte_array, which are width bytes
// each with no length in front of them. The width is in the schema, since every
// value of the column is the same size.
//
// Like ByteArray, the values point into the data.
func (d *PlainDecoder) Fixed(dst [][]byte, width int) (int, error) {
	if width <= 0 {
		return 0, fmt.Errorf("parquet: %w: a fixed length column of %d bytes", ErrFormat, width)
	}

	n, b, err := d.take(width, len(dst))
	if err != nil {
		return 0, err
	}
	for i := range n {
		dst[i] = b[i*width : (i+1)*width]
	}
	return n, nil
}

// take hands back the bytes of up to want values of size bytes each and moves
// the cursor past them.
//
// The bytes left over when a page ends in the middle of a value are the reason
// this is one function. A page holds as many values as its header says and no
// part of one, so a tail too short to be a value is a page written by something
// that lost count, and a reader that stops quietly on it hands back a column
// with a hole in it and says nothing.
func (d *PlainDecoder) take(size, want int) (int, []byte, error) {
	if want == 0 {
		return 0, nil, nil
	}
	d.align()

	left := len(d.buf) - d.pos
	if left == 0 {
		return 0, nil, io.EOF
	}
	if left < size {
		return 0, nil, fmt.Errorf("parquet: %w: %d bytes left where a value of %d was written",
			ErrFormat, left, size)
	}

	n := min(left/size, want)
	at := d.pos
	d.pos += n * size
	return n, d.buf[at:d.pos], nil
}

// align moves to the start of the next byte. Booleans are the only thing in the
// encoding that can leave the cursor in the middle of one, and a page is all one
// type, so this only ever does anything to a caller that mixed its methods up.
func (d *PlainDecoder) align() {
	if d.bit != 0 {
		d.bit, d.pos = 0, d.pos+1
	}
}
