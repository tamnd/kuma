package parquet

import (
	"encoding/binary"
	"math"
)

// Writing the plain encoding.
//
// This is [PlainDecoder] the other way round and it is the bottom of the writer
// the way that one is the bottom of the reader. A dictionary page is plain
// values, the values behind a page of indices are plain values, and a column
// nothing else fits is written plain, so a file of a hundred million numbers
// spends most of its time here whatever else it does.
//
// There is nothing here that writes an int96. The reader reads them because
// files full of them are still around, and no writer has produced one for years
// because a timestamp in twelve bytes with no zone and no unit was a mistake the
// format has since replaced twice over.
//
// The values go into a buffer rather than to an io.Writer. A page has to be
// finished before anything about it can be written down, since the header in
// front of it says how long it is and what its statistics are, and a page is a
// megabyte at the outside. So the encoder holds the bytes, the caller takes them
// when the page is full, and Reset puts the buffer back to the start for the
// next one.

// PlainEncoder writes values in the plain encoding.
//
// It does not know what type it is writing, the same way [PlainDecoder] does not
// know what it is reading. The caller writes the values with the method for the
// physical type of its column and there are no conversions, because a value
// written at the wrong width is not a wrong value but a different value at every
// position after it.
//
// The zero value is an encoder with nothing in it and is ready to use.
type PlainEncoder struct {
	buf []byte

	// bit is how far into the last byte a run of booleans has got, and is zero
	// when the last byte is finished with. Nothing else in the encoding is
	// smaller than a byte, so every other method starts by closing whatever
	// byte a run of booleans left half written.
	bit uint8
}

// Reset empties the encoder and keeps the buffer, so that a writer putting down
// a thousand pages of one column allocates for the largest of them rather than
// for each of them.
func (e *PlainEncoder) Reset() {
	e.buf, e.bit = e.buf[:0], 0
}

// Bytes returns the values written so far, which is the data of a page.
//
// The bytes are the encoder's own and are good until the next write to it, so a
// caller holding on to a page after calling Reset holds a copy.
func (e *PlainEncoder) Bytes() []byte { return e.buf }

// Len is how many bytes have been written, which is what a caller watching for
// a page to reach its size looks at.
func (e *PlainEncoder) Len() int { return len(e.buf) }

// Int32 writes values as int32, which is every integer parquet stores in four
// bytes or fewer, a date, and a time of day in milliseconds.
func (e *PlainEncoder) Int32(vals []int32) {
	b := e.grow(4 * len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(v))
	}
}

// Int64 writes values as int64, which is every wider integer and every timestamp
// the format has a logical type for.
func (e *PlainEncoder) Int64(vals []int64) {
	b := e.grow(8 * len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint64(b[8*i:], uint64(v))
	}
}

// Float writes values as float.
func (e *PlainEncoder) Float(vals []float32) {
	b := e.grow(4 * len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
	}
}

// Double writes values as double.
func (e *PlainEncoder) Double(vals []float64) {
	b := e.grow(8 * len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint64(b[8*i:], math.Float64bits(v))
	}
}

// Boolean writes values as boolean, which are one bit each and packed eight to a
// byte from the bottom up.
//
// The bits run on across the page rather than starting again per call, the way
// the decoder reads them, so a page of three booleans is one byte with five bits
// in it that are not values. Those bits are zero, which matters because a page
// is compared against another page in a test and not because anything reads
// them.
func (e *PlainEncoder) Boolean(vals []bool) {
	// Finish the byte the last call left part way through, since a run of
	// booleans carries on where it stopped rather than starting again.
	for len(vals) > 0 && e.bit != 0 {
		if vals[0] {
			e.buf[len(e.buf)-1] |= 1 << e.bit
		}
		e.bit, vals = (e.bit+1)&7, vals[1:]
	}

	// Whole bytes, built in a register and appended once. Doing this a value at
	// a time costs an append and a bounds check per bit, which on a column of a
	// hundred million rows is most of what writing it costs.
	for len(vals) >= 8 {
		v := vals[:8]
		var b byte
		for i := range 8 {
			if v[i] {
				b |= 1 << i
			}
		}
		e.buf, vals = append(e.buf, b), vals[8:]
	}

	if len(vals) > 0 {
		e.buf = append(e.buf, 0)
		for i, v := range vals {
			if v {
				e.buf[len(e.buf)-1] |= 1 << i
			}
		}
		e.bit = uint8(len(vals))
	}
}

// ByteArray writes values as byte_array, which is four bytes of length and then
// that many bytes.
//
// The length is four bytes because the format says it is, which puts a value of
// more than four gigabytes outside what this encoding can hold. Nothing that
// reads parquet would read one either.
func (e *PlainEncoder) ByteArray(vals [][]byte) {
	e.align()
	for _, v := range vals {
		e.buf = binary.LittleEndian.AppendUint32(e.buf, uint32(len(v)))
		e.buf = append(e.buf, v...)
	}
}

// ByteArrayString writes strings as byte_array, which is what a column of them
// is written as and saves turning every value into bytes on the way past.
func (e *PlainEncoder) ByteArrayString(vals []string) {
	e.align()
	for _, v := range vals {
		e.buf = binary.LittleEndian.AppendUint32(e.buf, uint32(len(v)))
		e.buf = append(e.buf, v...)
	}
}

// Fixed writes values as fixed_len_byte_array, which is the bytes of each value
// with no length in front of them.
//
// The width is in the schema rather than in the page, so every value has to be
// the width the column said and nothing here can check that: a value of the
// wrong length would be read back as the tail of one value and the head of the
// next. The caller writing the schema is the one that knows.
func (e *PlainEncoder) Fixed(vals [][]byte) {
	e.align()
	for _, v := range vals {
		e.buf = append(e.buf, v...)
	}
}

// grow makes room for n more bytes and returns them to be written into.
func (e *PlainEncoder) grow(n int) []byte {
	e.align()
	at := len(e.buf)
	e.buf = append(e.buf, make([]byte, n)...)
	return e.buf[at:]
}

// align finishes the byte a run of booleans was part way through, so that what
// comes next starts on a byte of its own.
//
// A page is all one type, so this only ever does anything to a caller that mixed
// its methods up. The decoder has the same guard for the same reason and the two
// of them agree about where the next value starts.
func (e *PlainEncoder) align() { e.bit = 0 }
