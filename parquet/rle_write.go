package parquet

import (
	"encoding/binary"
	"fmt"
)

// Writing the encoding parquet calls RLE.
//
// This is [RLEDecoder] the other way round, and it is what levels and dictionary
// indices are written in. The reader takes the runs as they come. The writer has
// to decide where they go, which is the whole of the work here: the same values
// can be written as one run or as fifty and every one of those is a legal file,
// so what makes this worth anything is picking the runs that make it small.
//
// The rule is the one every writer of this format uses. A value that repeats
// eight times or more is written as a repeat, since a repeat costs a count and
// one value however long it runs. Anything else goes into groups of eight packed
// end to end, since a packed group of eight values of two bits is two bytes and
// the shortest repeat is also two. So a column with no nulls in it, whose levels
// are a hundred thousand of the same number, comes out as three bytes, and a
// column of indices that jump about comes out packed with a header per sixty
// three groups.
//
// Sixty three is not a rounded down number. It is the largest count that leaves
// the header of a packed run inside one byte, and one byte is what lets the
// header be written before the groups it counts: the byte is put down empty, the
// groups go in behind it, and it is filled in when the run ends. A run that
// needed two bytes of header could not be, because nothing would know how wide
// to leave the gap.
//
// There is nothing here that writes the encoding this one replaced. The reader
// reads it because files that have it are not going anywhere, and writing it
// would be writing a file for a reader that stopped existing years ago.

// maxPackedGroups is how many groups of eight go into one packed run, which is
// as many as fit in the one byte of header the run is written behind.
const maxPackedGroups = 63

// noRun is the header position of a packed run that is not open.
const noRun = -1

// RLEEncoder writes values in the encoding parquet calls RLE, which is the
// hybrid of repeated runs and bit packed runs.
//
// The width is how many bits each value takes and it does not go into the data,
// the same way the decoder does not read it from there. Levels get it from the
// schema and dictionary indices get it from a byte the page writer puts in front
// of these bytes, so both of those belong to whoever is putting the page
// together.
//
// A width of nought is a column with one possible value, which is what the
// levels of a required column would be and what the indices of a dictionary of
// one value are. It is a real width and not an error: the runs are still
// counted, so the values are still there to be counted back out, and every one
// of them has to be nought because nought is all that fits.
//
// The zero value writes no values. Use NewRLEEncoder or Reset.
type RLEEncoder struct {
	buf   []byte
	width uint
	mask  uint64

	// value is the value the run being watched repeats and repeat is how many
	// times it has, which is what decides between the two kinds of run.
	value  int32
	repeat int

	// group is the values that have not been packed yet and n is how many of
	// them there are. They are held rather than written because packing is done
	// eight at a time, which is what makes a group of any width a whole number
	// of bytes.
	group [8]int32
	n     int

	// header is where the byte in front of the packed run being written sits,
	// or noRun when there is no such run, and groups is how many groups have
	// gone in behind it.
	header int
	groups int
}

// NewRLEEncoder returns an encoder writing width bit values.
func NewRLEEncoder(width int) (*RLEEncoder, error) {
	e := &RLEEncoder{}
	if err := e.Reset(width); err != nil {
		return nil, err
	}
	return e, nil
}

// Reset empties the encoder and sets the width, keeping the buffer so that a
// column of a thousand pages allocates for the largest of them rather than for
// each of them.
func (e *RLEEncoder) Reset(width int) error {
	if width < 0 || width > maxWidth {
		return fmt.Errorf("parquet: values of %d bits", width)
	}
	*e = RLEEncoder{buf: e.buf[:0], width: uint(width), mask: 1<<uint(width) - 1, header: noRun}
	return nil
}

// Write adds values to the run being built.
//
// A value wider than the width is an error and nothing of the call is written,
// since a value that does not fit would be read back as a different value and a
// page of levels that lost one is a page of nulls in the wrong places.
func (e *RLEEncoder) Write(vals []int32) error {
	for i, v := range vals {
		if uint64(uint32(v))&^e.mask != 0 {
			return fmt.Errorf("parquet: a value of %d at %d in %d bits", v, i, e.width)
		}
		e.put(v)
	}
	return nil
}

// put adds one value.
//
// A value the same as the one before it lengthens the run being watched, and one
// that is not ends that run. The value goes into the group either way until the
// run is long enough to be worth writing as a repeat, because a run that stops
// at seven has to be written as part of a packed group and there would be
// nowhere to write it from.
func (e *RLEEncoder) put(v int32) {
	if v == e.value {
		e.repeat++
		if e.repeat >= 8 {
			return
		}
	} else {
		if e.repeat >= 8 {
			e.repeated()
		}
		e.value, e.repeat = v, 1
	}

	e.group[e.n] = v
	e.n++
	if e.n == 8 {
		e.packed()
	}
}

// Len is how many bytes have been written down.
//
// The values of the run being built are not among them, which is at most a group
// of eight and the count of a repeat, so this is what a page writer watching for
// its page to fill looks at and is a few bytes short of what the page will be.
func (e *RLEEncoder) Len() int { return len(e.buf) }

// Finish closes the run being built and returns everything written.
//
// The bytes are the encoder's own and are good until the next write to it. The
// last group is padded out to eight with noughts when it has to be, which is
// what every writer of this format does: a page says how many values it holds
// and a reader stops when it has them, so the values past the end are read and
// thrown away rather than mistaken for anything.
func (e *RLEEncoder) Finish() []byte {
	switch {
	case e.repeat >= 8:
		e.repeated()
	case e.n > 0:
		for i := e.n; i < 8; i++ {
			e.group[i] = 0
		}
		e.packed()
	}
	e.close()
	return e.buf
}

// repeated writes the run being watched as a count and one value.
func (e *RLEEncoder) repeated() {
	// A packed run that was being filled ends here, since the two kinds of run
	// take turns and this one is about to be the current one.
	e.close()

	e.buf = binary.AppendUvarint(e.buf, uint64(e.repeat)<<1)

	// The value takes as many whole bytes as the width needs rather than as
	// many as it uses, so a width of ten is two bytes whatever the value is.
	v := uint64(uint32(e.value))
	for range (e.width + 7) / 8 {
		e.buf = append(e.buf, byte(v))
		v >>= 8
	}

	e.repeat, e.n = 0, 0
}

// packed writes the group of eight, opening a packed run to put it in or adding
// it to the one that is open.
func (e *RLEEncoder) packed() {
	if e.groups >= maxPackedGroups {
		e.close()
	}
	if e.header == noRun {
		// The header cannot be written until the run ends, since what it says
		// is how many groups went into it, so the byte is left empty and filled
		// in by close.
		e.buf = append(e.buf, 0)
		e.header = len(e.buf) - 1
	}

	e.pack()
	e.groups++

	// The values of the group have been written, so whatever run of repeats was
	// being watched has gone into them and there is nothing left to repeat.
	e.n, e.repeat = 0, 0
}

// pack writes the eight values of the group end to end, width bits each.
//
// The bits are one stream rather than a value per byte, so a value of five bits
// is the bottom five of one byte and the run carries on into the next. Eight
// values of width bits are width bytes, which is the whole reason the format
// packs in eights and is why nothing here has to pad.
func (e *RLEEncoder) pack() {
	var acc uint64
	var bits uint
	for _, v := range e.group {
		acc |= (uint64(uint32(v)) & e.mask) << bits
		bits += e.width

		// The accumulator holds at most seven bits over from the value before
		// plus the thirty two of this one, so it never runs out of room.
		for bits >= 8 {
			e.buf = append(e.buf, byte(acc))
			acc, bits = acc>>8, bits-8
		}
	}
}

// close ends the packed run that is open, filling in the header that was left
// for it. It does nothing when no such run is open, which is what lets it be
// called wherever a run might have to end.
func (e *RLEEncoder) close() {
	if e.header == noRun {
		return
	}
	e.buf[e.header] = byte(e.groups)<<1 | 1
	e.header, e.groups = noRun, 0
}
