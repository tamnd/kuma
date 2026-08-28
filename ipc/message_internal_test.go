package ipc

import (
	"errors"
	"strings"
	"testing"
	"time"
)

// The messages a writer would not write. These are built here, from inside the
// package, because building them means writing FlatBuffers by hand: what is
// being tested is what the reader does with a message that is well formed and
// dishonest, which is the only kind an attacker gets to send.

// TestDecodeSchemaExplosion checks that a small message cannot ask for a large
// amount of work.
//
// Fields hold fields, and nothing stops a hundred of them from pointing at the
// same child. Ten levels of that is a message of a few hundred bytes that
// describes a million columns, and a reader that walks it honestly is a reader
// that can be stopped by anyone who can send it bytes. The budget is what
// refuses it, and the size of the message is what the budget is made of.
func TestDecodeSchemaExplosion(t *testing.T) {
	msg := explodingSchema(t, 10, 4)
	if len(msg) > 4096 {
		t.Fatalf("the test message is %d bytes, which is not the point", len(msg))
	}

	done := make(chan error, 1)
	go func() {
		_, err := DecodeSchema(msg)
		done <- err
	}()

	select {
	case err := <-done:
		if !errors.Is(err, ErrMessage) {
			t.Fatalf("a message of %d bytes describing 4^10 fields: %v, want ErrMessage",
				len(msg), err)
		}
	case <-time.After(30 * time.Second):
		t.Fatalf("a message of %d bytes describing 4^10 fields is still being read",
			len(msg))
	}
}

// explodingSchema writes a schema whose fields all point at the same children,
// so that the tree it describes is fanout to the power of levels while the
// message stays the size of levels times fanout.
func explodingSchema(t *testing.T, levels, fanout int) []byte {
	t.Helper()

	var w fbBuilder
	w.startTable()
	w.slotInt(fbIntWidth, int32(64), 0)
	w.slotBool(fbIntSigned, true)
	intType := w.endTable()

	// The leaf, then one field per level holding fanout copies of the level
	// below it. Every copy is the same offset, which is what makes the message
	// small and the tree it describes enormous.
	field := writeField(&w, intType, 0)
	for range levels {
		children := make([]fbOffset, fanout)
		for i := range children {
			children[i] = field
		}
		field = writeField(&w, intType, w.offsets(children))
	}

	top := make([]fbOffset, fanout)
	for i := range top {
		top[i] = field
	}
	fields := w.offsets(top)

	w.startTable()
	w.slotOffset(fbSchemaFields, fields)
	schema := w.endTable()

	w.startTable()
	w.slotInt(fbMessageVersion, fbVersionV5, 0)
	w.slotUint8(fbMessageHeaderType, fbHeaderSchema)
	w.slotOffset(fbMessageHeader, schema)
	return frame(w.finish(w.endTable()))
}

// writeField writes a Field table holding a type and, if there is one, a vector
// of children that is already in the buffer.
func writeField(w *fbBuilder, typ, children fbOffset) fbOffset {
	name := w.str("f")
	w.startTable()
	w.slotOffset(fbFieldName, name)
	w.slotUint8(fbFieldTypeType, fbTypeInt)
	w.slotOffset(fbFieldType, typ)
	w.slotOffset(fbFieldChildren, children)
	return w.endTable()
}

// TestDecodeSchemaWideVector checks a vector that says it holds more elements
// than the message has bytes.
//
// The reader allocates for a vector before it reads one, so the length is
// checked where it is read. Without that, twelve bytes are a request for two
// billion fields.
func TestDecodeSchemaWideVector(t *testing.T) {
	var w fbBuilder
	w.startVector(4, 1, 4)
	w.putUint32(0)
	fields := w.endVector(1)

	w.startTable()
	w.slotOffset(fbSchemaFields, fields)
	schema := w.endTable()

	w.startTable()
	w.slotInt(fbMessageVersion, fbVersionV5, 0)
	w.slotUint8(fbMessageHeaderType, fbHeaderSchema)
	w.slotOffset(fbMessageHeader, schema)
	msg := frame(w.finish(w.endTable()))

	// The vector now says it holds every element that could ever exist, which
	// is a number no message of this size can be holding.
	at := strings.Index(string(msg), "\x01\x00\x00\x00")
	if at < 0 {
		t.Fatal("cannot find the length of the vector of fields")
	}
	copy(msg[at:], []byte{0xF0, 0xFF, 0xFF, 0x7F})

	if _, err := DecodeSchema(msg); !errors.Is(err, ErrMessage) {
		t.Fatalf("a vector of two billion fields: %v, want ErrMessage", err)
	}
}

// TestDecodeSchemaHeader checks the messages that are not schemas. A stream
// starts with one and then holds record batches, and reading a batch as a
// schema has to say what it actually is.
func TestDecodeSchemaHeader(t *testing.T) {
	for header, want := range map[uint8]string{
		fbHeaderNone:            "message with no header",
		fbHeaderDictionaryBatch: "dictionary batch",
		fbHeaderRecordBatch:     "record batch",
		fbHeaderTensor:          "tensor",
		fbHeaderSparseTensor:    "sparse tensor",
		99:                      "message with no header",
	} {
		var w fbBuilder
		w.startTable()
		body := w.endTable()

		w.startTable()
		w.slotInt(fbMessageVersion, fbVersionV5, 0)
		w.slotUint8(fbMessageHeaderType, header)
		w.slotOffset(fbMessageHeader, body)
		msg := frame(w.finish(w.endTable()))

		_, err := DecodeSchema(msg)
		if !errors.Is(err, ErrMessage) {
			t.Errorf("header %d: %v, want ErrMessage", header, err)
			continue
		}
		if !strings.Contains(err.Error(), want) {
			t.Errorf("header %d: %v, want it to mention a %s", header, err, want)
		}
	}
}

// TestDecodeSchemaVersion checks a message from a future the reader does not
// know about. V5 has been the version since Arrow 1.0, and a message claiming
// a later one is holding something this cannot read.
func TestDecodeSchemaVersion(t *testing.T) {
	var w fbBuilder
	w.startTable()
	schema := w.endTable()

	w.startTable()
	w.slotInt(fbMessageVersion, int16(9), 0)
	w.slotUint8(fbMessageHeaderType, fbHeaderSchema)
	w.slotOffset(fbMessageHeader, schema)
	msg := frame(w.finish(w.endTable()))

	if _, err := DecodeSchema(msg); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("a message from metadata version 9: %v, want ErrUnsupported", err)
	}
}

// TestDecodeSchemaBigEndian checks the one schema that is well formed, readable
// and still refused. Every buffer in it would need its bytes swapped, which is
// a different job from reading it.
func TestDecodeSchemaBigEndian(t *testing.T) {
	var w fbBuilder
	w.startTable()
	w.slotInt(fbSchemaEndianness, int16(1), 0)
	schema := w.endTable()

	w.startTable()
	w.slotInt(fbMessageVersion, fbVersionV5, 0)
	w.slotUint8(fbMessageHeaderType, fbHeaderSchema)
	w.slotOffset(fbMessageHeader, schema)
	msg := frame(w.finish(w.endTable()))

	if _, err := DecodeSchema(msg); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("a big endian schema: %v, want ErrUnsupported", err)
	}
}
