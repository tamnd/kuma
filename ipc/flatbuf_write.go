package ipc

import (
	"bytes"
	"encoding/binary"
)

// Writing the FlatBuffers that Arrow reads its metadata from.
//
// A FlatBuffer is written back to front. Nothing can be pointed at until it
// has been written, so the leaves go down first and the root goes down last,
// and every offset counts backwards from where the offset itself ends up. That
// is why the code below reads inside out compared to the message it produces:
// the strings of a schema exist before the fields that name them, and the
// fields before the schema that holds them.
//
// The buffer fills from the end of a slice towards the start, so an offset is
// the distance from the end and never moves when the slice grows. Positions
// from the start would all have to be rewritten every time it does.

// fbBuilder writes one buffer. The zero value is ready to use.
type fbBuilder struct {
	buf  []byte // the buffer, filled from the end towards the start
	head int    // the first byte in use, so buf[head:] is what has been written

	slots    []fbSlot // the fields of the table being written
	tableEnd int      // where the table being written ends, as a distance from the end
	vtables  []int    // the vtables written so far, so an identical one is shared
	minalign int      // the widest thing written, which the whole buffer is aligned to
}

// fbSlot is one field of the table being written: which field it is and where
// its value ended up.
type fbSlot struct {
	id  int
	off int
}

// fbOffset is a distance back from the end of the buffer. Everything already
// written is named by one, and the arithmetic that turns it into an offset in
// the finished buffer is the same subtraction every time.
type fbOffset = int

// pos is where the next byte will go, as a distance from the end.
func (w *fbBuilder) pos() fbOffset { return len(w.buf) - w.head }

// prep makes room for a value of the given width plus any bytes that go in
// front of it, and pads so that the value itself lands on a multiple of its
// own width.
//
// Alignment is the whole reason this is not just an append. A reader of a
// FlatBuffer reads a 64 bit number straight out of the bytes, so it has to be
// on an 8 byte boundary, and the boundary is measured in the finished buffer
// rather than in the slice this is building it in.
func (w *fbBuilder) prep(width, extra int) {
	if width > w.minalign {
		w.minalign = width
	}
	pad := (^(w.pos() + extra) + 1) & (width - 1)
	for w.head <= pad+width+extra {
		w.grow()
	}
	for range pad {
		w.head--
		w.buf[w.head] = 0
	}
}

// grow doubles the buffer, keeping what is written at the end of it.
func (w *fbBuilder) grow() {
	if len(w.buf) == 0 {
		w.buf = make([]byte, 64)
		w.head = 64
		return
	}
	next := make([]byte, len(w.buf)*2)
	copy(next[len(w.buf):], w.buf)
	w.head += len(w.buf)
	w.buf = next
}

// The unchecked writes. Each one is preceded by a prep for the same width, so
// the room is there and the position is aligned.
func (w *fbBuilder) putUint8(v uint8) {
	w.head--
	w.buf[w.head] = v
}

func (w *fbBuilder) putUint16(v uint16) {
	w.head -= 2
	binary.LittleEndian.PutUint16(w.buf[w.head:], v)
}

func (w *fbBuilder) putUint32(v uint32) {
	w.head -= 4
	binary.LittleEndian.PutUint32(w.buf[w.head:], v)
}

func (w *fbBuilder) putUint64(v uint64) {
	w.head -= 8
	binary.LittleEndian.PutUint64(w.buf[w.head:], v)
}

// startTable begins a table. Fields go in with the slot calls below, in any
// order, and endTable writes the vtable that says where they went.
func (w *fbBuilder) startTable() {
	w.slots = w.slots[:0]
	w.tableEnd = w.pos()
}

// The slot calls. A field whose value is its default is not written at all,
// which is what keeps a schema of twenty fields from carrying twenty zeroes.
//
// The byte and the bool take no default because in this format they have none
// but zero. The integers take one because they do: the width of a decimal
// defaults to 128 and the unit of a time to milliseconds, and a field left out
// is a field read back as whatever the schema says it should have been.
func (w *fbBuilder) slotUint8(id int, v uint8) {
	if v == 0 {
		return
	}
	w.prep(1, 0)
	w.putUint8(v)
	w.slot(id)
}

func (w *fbBuilder) slotBool(id int, v bool) {
	if !v {
		return
	}
	w.prep(1, 0)
	w.putUint8(1)
	w.slot(id)
}

// slotInt writes an integer field of whatever width the value's type has. The
// three widths are one method because the only difference between them is how
// many bytes go down and how they are read back out.
func (w *fbBuilder) slotInt[T int16 | int32 | int64](id int, v, def T) {
	if v == def {
		return
	}
	switch n := any(v).(type) {
	case int16:
		w.prep(2, 0)
		w.putUint16(uint16(n))
	case int32:
		w.prep(4, 0)
		w.putUint32(uint32(n))
	case int64:
		w.prep(8, 0)
		w.putUint64(uint64(n))
	}
	w.slot(id)
}

// slotOffset points a field at something already written. An offset of zero is
// nothing to point at, which is how an absent string or table is written.
func (w *fbBuilder) slotOffset(id int, off fbOffset) {
	if off == 0 {
		return
	}
	w.prep(4, 0)
	w.putUint32(uint32(w.pos() + 4 - off))
	w.slot(id)
}

// slot records that the field just written belongs to this one.
func (w *fbBuilder) slot(id int) {
	w.slots = append(w.slots, fbSlot{id: id, off: w.pos()})
}

// endTable writes the vtable and returns where the table starts.
//
// The vtable is its own size, then the size of the table, then two bytes per
// field saying how far into the table that field sits, or zero for the ones
// that are not there. An identical vtable that has already been written is
// used again rather than written twice, which is most of why a schema of
// twenty like fields is not twenty copies of the same six bytes.
func (w *fbBuilder) endTable() fbOffset {
	// The table starts with a signed offset back to its vtable, which is not
	// known until the vtable has been written, so the space goes down first.
	w.prep(4, 0)
	w.putUint32(0)
	table := w.pos()

	highest := -1
	for _, s := range w.slots {
		if s.id > highest {
			highest = s.id
		}
	}

	// The vtable, written the same way as everything else: backwards, so the
	// last field first and the two sizes last.
	w.prep(2, (highest+3)*2)
	for id := highest; id >= 0; id-- {
		var off uint16
		for _, s := range w.slots {
			if s.id == id {
				off = uint16(table - s.off)
				break
			}
		}
		w.putUint16(off)
	}
	w.putUint16(uint16(table - w.tableEnd))
	w.putUint16(uint16((highest + 3) * 2))

	if shared := w.sharedVtable(); shared != 0 {
		// Give back the vtable just written and point at the one that is
		// already there.
		w.head = len(w.buf) - table
		binary.LittleEndian.PutUint32(w.buf[w.head:], uint32(shared-table))
	} else {
		w.vtables = append(w.vtables, w.pos())
		binary.LittleEndian.PutUint32(w.buf[len(w.buf)-table:], uint32(w.pos()-table))
	}
	return table
}

// sharedVtable is a vtable already in the buffer that is byte for byte the one
// just written, or zero if this is the first of its shape.
func (w *fbBuilder) sharedVtable() fbOffset {
	start := len(w.buf) - w.pos()
	size := int(binary.LittleEndian.Uint16(w.buf[start:]))
	for _, off := range w.vtables {
		at := len(w.buf) - off
		if size != int(binary.LittleEndian.Uint16(w.buf[at:])) {
			continue
		}
		if bytes.Equal(w.buf[start:start+size], w.buf[at:at+size]) {
			return off
		}
	}
	return 0
}

// str writes a string and returns where it starts. Strings are written with a
// null byte after them, which nothing in this package reads and every C reader
// of a FlatBuffer expects.
func (w *fbBuilder) str(s string) fbOffset {
	w.prep(4, len(s)+1)
	w.putUint8(0)
	w.head -= len(s)
	copy(w.buf[w.head:], s)
	w.putUint32(uint32(len(s)))
	return w.pos()
}

// offsets writes a vector of offsets, which is how a vector of tables or of
// strings is written. The elements go in backwards, since the buffer does.
func (w *fbBuilder) offsets(offs []fbOffset) fbOffset {
	w.startVector(4, len(offs), 4)
	for i := len(offs) - 1; i >= 0; i-- {
		w.prep(4, 0)
		w.putUint32(uint32(w.pos() + 4 - offs[i]))
	}
	return w.endVector(len(offs))
}

// startVector makes room for n elements of the given width, aligned to the
// given alignment, which is wider than the element for a vector of structs.
// The elements are written by the caller and endVector writes the length.
func (w *fbBuilder) startVector(width, n, align int) {
	w.prep(4, width*n)
	w.prep(align, width*n)
}

// endVector writes the length in front of the elements.
func (w *fbBuilder) endVector(n int) fbOffset {
	w.putUint32(uint32(n))
	return w.pos()
}

// finish closes the buffer around a root table and returns the bytes.
//
// The result is aligned to the widest thing in it, so that a reader can look
// at it where it lies. Arrow relies on that: a message is read out of a file
// without being copied anywhere first.
func (w *fbBuilder) finish(root fbOffset) []byte {
	w.prep(w.minalign, 4)
	w.putUint32(uint32(w.pos() + 4 - root))
	return w.buf[w.head:]
}
