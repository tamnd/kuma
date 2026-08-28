package ipc

import (
	"encoding/binary"
	"errors"
	"strings"
	"testing"
)

// The FlatBuffers layer is tested from inside the package, since it is not
// exported and never will be. What is exported is the schema message it is
// underneath, and that is tested against pyarrow. This is the layer below,
// where the reader has to survive bytes that no writer would produce.

// The field numbers the tests below use. They are arbitrary and only have to
// agree between the writer and the reader.
const (
	testName = iota
	testCount
	testFlag
	testBig
	testChild
	testList
	testSmall
)

// buildTest writes a table holding one of everything the reader can read.
func buildTest(t *testing.T, names []string) []byte {
	t.Helper()

	var w fbBuilder
	name := w.str("outer")

	items := make([]fbOffset, len(names))
	for i, s := range names {
		inner := w.str(s)
		w.startTable()
		w.slotOffset(testName, inner)
		items[i] = w.endTable()
	}

	w.startTable()
	w.slotInt(testBig, int64(1)<<40, 0)
	child := w.endTable()

	list := fbOffset(0)
	if len(items) > 0 {
		list = w.offsets(items)
	}

	w.startTable()
	w.slotOffset(testName, name)
	w.slotInt(testCount, int32(-7), 0)
	w.slotBool(testFlag, true)
	w.slotOffset(testChild, child)
	w.slotOffset(testList, list)
	w.slotUint8(testSmall, 200)
	return w.finish(w.endTable())
}

func TestFlatbufRoundTrip(t *testing.T) {
	names := []string{"a", "bb", "ccc"}
	buf := buildTest(t, names)

	root, err := fbRoot(buf)
	if err != nil {
		t.Fatalf("fbRoot: %v", err)
	}

	name, err := root.str(testName)
	if err != nil || name != "outer" {
		t.Errorf("name = %q, %v, want outer", name, err)
	}
	count, err := root.integer(testCount, int32(0))
	if err != nil || count != -7 {
		t.Errorf("count = %d, %v, want -7", count, err)
	}
	flag, err := root.boolean(testFlag, false)
	if err != nil || !flag {
		t.Errorf("flag = %v, %v, want true", flag, err)
	}
	small, err := root.uint8(testSmall, 0)
	if err != nil || small != 200 {
		t.Errorf("small = %d, %v, want 200", small, err)
	}

	child, ok, err := root.table(testChild)
	if err != nil || !ok {
		t.Fatalf("child: %v, %v", ok, err)
	}
	big, err := child.integer(testBig, int64(0))
	if err != nil || big != 1<<40 {
		t.Errorf("big = %d, %v, want %d", big, err, 1<<40)
	}

	list, ok, err := root.vector(testList)
	if err != nil || !ok {
		t.Fatalf("list: %v, %v", ok, err)
	}
	if list.len() != len(names) {
		t.Fatalf("list has %d elements, want %d", list.len(), len(names))
	}
	for i, want := range names {
		item, err := list.table(i)
		if err != nil {
			t.Fatalf("element %d: %v", i, err)
		}
		got, err := item.str(testName)
		if err != nil || got != want {
			t.Errorf("element %d = %q, %v, want %q", i, got, err, want)
		}
	}
}

// TestFlatbufDefaults checks that a field nobody wrote reads as its default
// rather than as an error. That is what lets a reader built against one version
// of a schema read a message written against an older one, and it is the only
// reason the format is worth using.
func TestFlatbufDefaults(t *testing.T) {
	var w fbBuilder
	w.startTable()
	w.slotInt(testCount, int32(5), 0)
	buf := w.finish(w.endTable())

	root, err := fbRoot(buf)
	if err != nil {
		t.Fatalf("fbRoot: %v", err)
	}
	if n, err := root.integer(testCount, int32(0)); err != nil || n != 5 {
		t.Fatalf("count = %d, %v, want 5", n, err)
	}

	// Every kind of read, on fields that are not there and on field numbers
	// past the end of the vtable entirely.
	for _, id := range []int{testName, testFlag, testBig, testList, 99} {
		if v, err := root.uint8(id, 3); err != nil || v != 3 {
			t.Errorf("uint8 %d = %d, %v, want the default", id, v, err)
		}
		if v, err := root.boolean(id, true); err != nil || !v {
			t.Errorf("boolean %d = %v, %v, want the default", id, v, err)
		}
		if v, err := root.integer(id, int16(-1)); err != nil || v != -1 {
			t.Errorf("int16 %d = %d, %v, want the default", id, v, err)
		}
		if v, err := root.integer(id, int64(12)); err != nil || v != 12 {
			t.Errorf("int64 %d = %d, %v, want the default", id, v, err)
		}
		if v, err := root.str(id); err != nil || v != "" {
			t.Errorf("str %d = %q, %v, want the empty string", id, v, err)
		}
		if _, ok, err := root.table(id); err != nil || ok {
			t.Errorf("table %d = %v, %v, want nothing", id, ok, err)
		}
		if _, ok, err := root.vector(id); err != nil || ok {
			t.Errorf("vector %d = %v, %v, want nothing", id, ok, err)
		}
	}
}

// TestFlatbufShared checks that two tables of the same shape share one vtable.
// A schema is mostly the same field over and over, so this is the difference
// between a message that is a few hundred bytes and one that is a few thousand.
func TestFlatbufShared(t *testing.T) {
	var w fbBuilder
	for range 4 {
		w.startTable()
		w.slotInt(testCount, int32(1), 0)
		w.endTable()
	}
	w.startTable()
	w.slotInt(testCount, int32(1), 0)
	w.slotBool(testFlag, true)
	w.endTable()

	if len(w.vtables) != 2 {
		t.Errorf("wrote %d vtables, want 2: four tables of one shape and one of another",
			len(w.vtables))
	}
}

// TestFlatbufShort checks the buffers that are too small to hold anything.
func TestFlatbufShort(t *testing.T) {
	for n := range 12 {
		if _, err := fbRoot(make([]byte, n)); err == nil {
			t.Errorf("%d zero bytes read as a table, want an error", n)
		} else if !errors.Is(err, ErrMessage) {
			t.Errorf("%d zero bytes: %v, want ErrMessage", n, err)
		}
	}
}

// TestFlatbufCorrupt changes one byte at a time and checks that nothing
// panics.
//
// A FlatBuffer is a pile of offsets, so one wrong byte is an offset pointing
// anywhere at all, and the whole reason this package reads the format by hand
// is that the generated readers index on those offsets without looking. Every
// read here goes through a bounds check, and this walks the buffer to say so.
// What comes back does not have to be an error, since plenty of single byte
// changes are still a valid message, but it does have to be ErrMessage when it
// is one.
func TestFlatbufCorrupt(t *testing.T) {
	buf := buildTest(t, []string{"a", "bb"})
	for i := range buf {
		for _, b := range []byte{0x00, 0x01, 0x7F, 0xFF} {
			bad := append([]byte(nil), buf...)
			bad[i] = b
			if err := readAll(bad); err != nil && !errors.Is(err, ErrMessage) {
				t.Fatalf("byte %d set to %#x: %v, want ErrMessage", i, b, err)
			}
		}
	}
}

// readAll reads every field of a table built by buildTest, so that a corrupted
// buffer is walked rather than merely opened.
func readAll(buf []byte) error {
	root, err := fbRoot(buf)
	if err != nil {
		return err
	}
	if _, err = root.str(testName); err != nil {
		return err
	}
	if _, err = root.integer(testCount, int32(0)); err != nil {
		return err
	}
	if _, err = root.integer(testBig, int64(0)); err != nil {
		return err
	}
	if _, err = root.boolean(testFlag, false); err != nil {
		return err
	}
	child, ok, err := root.table(testChild)
	if err != nil {
		return err
	}
	if ok {
		if _, err = child.integer(testBig, int64(0)); err != nil {
			return err
		}
	}
	list, ok, err := root.vector(testList)
	if err != nil || !ok {
		return err
	}
	for i := range list.len() {
		item, err := list.table(i)
		if err != nil {
			return err
		}
		if _, err := item.str(testName); err != nil {
			return err
		}
	}
	return nil
}

// TestFlatbufVectorBounds checks the reads that ask for an element that is not
// there, which is what a length changed on the wire turns into.
func TestFlatbufVectorBounds(t *testing.T) {
	buf := buildTest(t, []string{"a"})
	root, err := fbRoot(buf)
	if err != nil {
		t.Fatalf("fbRoot: %v", err)
	}
	list, ok, err := root.vector(testList)
	if err != nil || !ok {
		t.Fatalf("list: %v, %v", ok, err)
	}
	for _, i := range []int{-1, 1, 1 << 20} {
		if _, err := list.table(i); !errors.Is(err, ErrMessage) {
			t.Errorf("element %d: %v, want ErrMessage", i, err)
		}
	}
}

// TestFlatbufHugeVector checks a length that no buffer could hold. It is the
// one number in the format that is unsigned and wider than the reader's own
// arithmetic, so it is checked where it is read rather than where it is used.
func TestFlatbufHugeVector(t *testing.T) {
	buf := buildTest(t, []string{"a"})
	root, err := fbRoot(buf)
	if err != nil {
		t.Fatalf("fbRoot: %v", err)
	}
	pos, ok, err := root.slot(testList)
	if err != nil || !ok {
		t.Fatalf("slot: %v, %v", ok, err)
	}
	at, err := fbIndirect(buf, pos)
	if err != nil {
		t.Fatalf("fbIndirect: %v", err)
	}
	binary.LittleEndian.PutUint32(buf[at:], 0xFFFFFFF0)

	if _, _, err := root.vector(testList); !errors.Is(err, ErrMessage) {
		t.Errorf("a vector of four billion elements: %v, want ErrMessage", err)
	}
}

// TestFlatbufLongString checks that a string longer than the buffer it grew in
// comes back whole, which is the growth loop rather than the format.
func TestFlatbufLongString(t *testing.T) {
	want := strings.Repeat("kuma", 1000)

	var w fbBuilder
	off := w.str(want)
	w.startTable()
	w.slotOffset(testName, off)
	buf := w.finish(w.endTable())

	root, err := fbRoot(buf)
	if err != nil {
		t.Fatalf("fbRoot: %v", err)
	}
	got, err := root.str(testName)
	if err != nil {
		t.Fatalf("str: %v", err)
	}
	if got != want {
		t.Errorf("read %d bytes back, want %d", len(got), len(want))
	}
}
