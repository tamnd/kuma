package ipc

import (
	"bytes"
	"errors"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// The streams a kuma writer would not write: the old framing, which files
// written before 2018 are still in, and the messages this cannot read yet.

// legacy takes the continuation off the front of a message, which is how a
// message was framed before the marker was added.
func legacy(msg []byte) []byte { return msg[4:] }

// streamSchema and streamBatch are the one column schema and the one batch the
// tests in here build streams out of.
func streamSchema() dtype.Schema {
	return dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64}}}
}

func streamBatch(vals ...int64) Batch {
	return Batch{Length: len(vals), Columns: []*array.Array{array.Of(vals...)}}
}

// TestStreamLegacyFraming reads a stream written the old way, with the length on
// its own and no continuation in front of it. Both framings appear in one stream
// here, since nothing says a file cannot have been appended to by a newer
// writer, and the reader decides message by message.
func TestStreamLegacyFraming(t *testing.T) {
	s := streamSchema()
	schema, err := EncodeSchema(s)
	if err != nil {
		t.Fatalf("EncodeSchema: %v", err)
	}
	first, err := EncodeBatch(s, streamBatch(1, 2, 3))
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}
	second, err := EncodeBatch(s, streamBatch(4, 5))
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}

	var stream []byte
	stream = append(stream, legacy(schema)...)
	stream = append(stream, legacy(first)...)
	stream = append(stream, second...)
	// The old end of stream is four zero bytes rather than eight, and a reader
	// that took it for a continuation would sit there waiting for a length.
	stream = append(stream, 0, 0, 0, 0)

	r, err := NewReader(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	var got []int64
	for r.Next() {
		col := r.Batch().Columns[0]
		for i := range col.Len() {
			got = append(got, col.Value[int64](i))
		}
	}
	if err := r.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	want := []int64{1, 2, 3, 4, 5}
	if len(got) != len(want) {
		t.Fatalf("read %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("read %v, want %v", got, want)
		}
	}
}

// handHeader writes a message of the given kind with nothing in it, which is
// enough for a reader that decides what to do by the kind alone.
func handHeader(kind uint8) []byte {
	var w fbBuilder
	w.startTable()
	w.slotInt(fbMessageVersion, fbVersionV5, 0)
	w.slotUint8(fbMessageHeaderType, kind)
	w.slotInt(fbMessageBodyLength, int64(0), 0)
	return frame(w.finish(w.endTable()))
}

// TestStreamMessageKinds checks what a reader says about the messages in a
// stream that are not record batches. A dictionary batch belongs in one, and in
// a stream whose schema has no dictionary encoded columns there is nothing it
// could be the values of. The rest do not belong in a stream at all.
func TestStreamMessageKinds(t *testing.T) {
	s := streamSchema()
	schema, err := EncodeSchema(s)
	if err != nil {
		t.Fatalf("EncodeSchema: %v", err)
	}

	for _, tt := range []struct {
		name string
		kind uint8
		want error
	}{
		{"a dictionary batch", fbHeaderDictionaryBatch, ErrMessage},
		{"another schema", fbHeaderSchema, ErrMessage},
		{"a tensor", fbHeaderTensor, ErrMessage},
		{"no header at all", fbHeaderNone, ErrMessage},
	} {
		t.Run(tt.name, func(t *testing.T) {
			stream := append(append([]byte{}, schema...), handHeader(tt.kind)...)
			r, err := NewReader(bytes.NewReader(stream))
			if err != nil {
				t.Fatalf("NewReader: %v", err)
			}
			if r.Next() {
				t.Fatal("Next read a batch out of a message that is not one")
			}
			if err := r.Err(); !errors.Is(err, tt.want) {
				t.Errorf("Err = %v, want %v", err, tt.want)
			}
		})
	}
}

// TestStreamBodyLength checks a message that says its body is a size no reader
// should believe. The bytes are read as they arrive rather than into a buffer
// the message asked for, so a number like this costs an unexpected end of file
// and nothing else.
func TestStreamBodyLength(t *testing.T) {
	s := streamSchema()
	schema, err := EncodeSchema(s)
	if err != nil {
		t.Fatalf("EncodeSchema: %v", err)
	}

	for _, tt := range []struct {
		name string
		body int64
	}{
		{"a body of minus one", -1},
		{"a body of four gigabytes", 1 << 32},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var w fbBuilder
			w.startTable()
			w.slotInt(fbMessageVersion, fbVersionV5, 0)
			w.slotUint8(fbMessageHeaderType, fbHeaderRecordBatch)
			w.slotInt(fbMessageBodyLength, tt.body, 0)
			msg := frame(w.finish(w.endTable()))

			stream := append(append([]byte{}, schema...), msg...)
			r, err := NewReader(bytes.NewReader(stream))
			if err != nil {
				t.Fatalf("NewReader: %v", err)
			}
			if r.Next() {
				t.Fatal("Next read a batch out of a message that lies about its body")
			}
			if err := r.Err(); !errors.Is(err, ErrMessage) {
				t.Errorf("Err = %v, want %v", err, ErrMessage)
			}
		})
	}
}
