package ipc

import (
	"bytes"
	"encoding/binary"
	"errors"
	"testing"
)

// The footers a kuma writer would not write. A footer is the one part of a file
// that nothing else checks: a block in it is three numbers saying where to read
// from and how much, and a reader that believed them would read wherever the
// numbers said.

// handFile builds a file out of a schema, one batch and whatever footer the
// caller wants, and says where the batch landed.
func handFile(t *testing.T, footer []byte) (file []byte, at block) {
	t.Helper()
	s := streamSchema()
	schema, err := EncodeSchema(s)
	if err != nil {
		t.Fatalf("EncodeSchema: %v", err)
	}
	batch, err := EncodeBatch(s, streamBatch(1, 2, 3))
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}

	file = append(file, filePad...)
	file = append(file, schema...)
	at, err = blockOf(int64(len(file)), batch)
	if err != nil {
		t.Fatalf("blockOf: %v", err)
	}
	file = append(file, batch...)
	file = append(file, streamEnd...)
	file = append(file, footer...)
	file = binary.LittleEndian.AppendUint32(file, uint32(len(footer)))
	file = append(file, fileMagic...)
	return file, at
}

// footerOf writes a footer by hand, with every part of it under the test's
// control rather than taken from what was written.
func footerOf(t *testing.T, version int16, schema bool, dicts, batches []block) []byte {
	t.Helper()

	sw := schemaWriter{types: make(map[string]typeRef)}
	at := fbOffset(0)
	if schema {
		off, err := sw.schema(streamSchema())
		if err != nil {
			t.Fatalf("schema: %v", err)
		}
		at = off
	}

	w := &sw.w
	dictsVec := w.blocks(dicts)
	batchesVec := w.blocks(batches)

	w.startTable()
	w.slotInt(fbFooterVersion, version, 0)
	w.slotOffset(fbFooterSchema, at)
	w.slotOffset(fbFooterDictionaries, dictsVec)
	w.slotOffset(fbFooterBatches, batchesVec)
	return w.finish(w.endTable())
}

// TestFooterHandWritten checks what a reader says about the footers that are not
// the one a writer produces.
func TestFooterHandWritten(t *testing.T) {
	_, good := handFile(t, nil)

	for _, tt := range []struct {
		name    string
		footer  []byte
		want    error
		batches int
	}{
		{
			name:    "the footer a writer writes",
			footer:  footerOf(t, fbVersionV5, true, nil, []block{good}),
			batches: 1,
		},
		{
			name:   "a version from the future",
			footer: footerOf(t, fbVersionV5+1, true, nil, []block{good}),
			want:   ErrUnsupported,
		},
		{
			name:   "no schema in it",
			footer: footerOf(t, fbVersionV5, false, nil, []block{good}),
			want:   ErrMessage,
		},
		{
			name:   "a dictionary batch in a file with no dictionary columns",
			footer: footerOf(t, fbVersionV5, true, []block{good}, []block{good}),
			want:   ErrMessage,
		},
		{
			name: "a batch inside the magic",
			footer: footerOf(t, fbVersionV5, true, nil,
				[]block{{offset: 0, meta: good.meta, body: good.body}}),
			want: ErrMessage,
		},
		{
			name: "a batch past the end of the messages",
			footer: footerOf(t, fbVersionV5, true, nil,
				[]block{{offset: 1 << 40, meta: good.meta, body: good.body}}),
			want: ErrMessage,
		},
		{
			name: "a batch of minus one bytes of metadata",
			footer: footerOf(t, fbVersionV5, true, nil,
				[]block{{offset: good.offset, meta: -1, body: good.body}}),
			want: ErrMessage,
		},
		{
			name: "a batch with a body the file cannot hold",
			footer: footerOf(t, fbVersionV5, true, nil,
				[]block{{offset: good.offset, meta: good.meta, body: 1 << 40}}),
			want: ErrMessage,
		},
		{
			name: "a batch whose body runs into the footer",
			footer: footerOf(t, fbVersionV5, true, nil,
				[]block{{offset: good.offset, meta: good.meta, body: good.body + 16}}),
			want: ErrMessage,
		},
		{
			name:   "nothing that is a footer at all",
			footer: []byte{1, 2, 3, 4, 5, 6, 7, 8},
			want:   ErrMessage,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			file, _ := handFile(t, tt.footer)
			r, err := NewFileReader(bytes.NewReader(file), int64(len(file)))
			if tt.want != nil {
				if !errors.Is(err, tt.want) {
					t.Fatalf("NewFileReader = %v, want %v", err, tt.want)
				}
				return
			}
			if err != nil {
				t.Fatalf("NewFileReader: %v", err)
			}
			if r.NumBatches() != tt.batches {
				t.Fatalf("the file holds %d batches, want %d", r.NumBatches(), tt.batches)
			}
			b, err := r.Batch(0)
			if err != nil {
				t.Fatalf("Batch(0): %v", err)
			}
			if b.Length != 3 {
				t.Errorf("the batch has %d rows, want 3", b.Length)
			}
		})
	}
}

// TestFooterPointsAtSchema checks a block that is inside the file and is not a
// record batch. Everything the footer promised holds and the message underneath
// is still the wrong one, which is the reader below saying so rather than this
// one.
func TestFooterPointsAtSchema(t *testing.T) {
	schema, err := EncodeSchema(streamSchema())
	if err != nil {
		t.Fatalf("EncodeSchema: %v", err)
	}
	at, err := blockOf(int64(len(filePad)), schema)
	if err != nil {
		t.Fatalf("blockOf: %v", err)
	}

	file, _ := handFile(t, footerOf(t, fbVersionV5, true, nil, []block{at}))
	r, err := NewFileReader(bytes.NewReader(file), int64(len(file)))
	if err != nil {
		t.Fatalf("NewFileReader: %v", err)
	}
	if _, err := r.Batch(0); !errors.Is(err, ErrMessage) {
		t.Errorf("Batch(0) of a block pointing at the schema = %v, want %v", err, ErrMessage)
	}
}

// TestBlockVector checks the twenty four bytes a block is written in. A Block is
// the one struct in this format with padding in the middle of it, and a writer
// that put the body length where the padding goes would round trip through its
// own reader and produce a file nothing else can open.
func TestBlockVector(t *testing.T) {
	want := []block{
		{offset: 8, meta: 16, body: 0},
		{offset: 1024, meta: 1 << 20, body: 1 << 40},
		{},
	}

	var w fbBuilder
	vec := w.blocks(want)
	w.startTable()
	w.slotOffset(fbFooterBatches, vec)
	buf := w.finish(w.endTable())

	root, err := fbRoot(buf)
	if err != nil {
		t.Fatalf("fbRoot: %v", err)
	}
	got, ok, err := root.vector(fbFooterBatches)
	if err != nil || !ok {
		t.Fatalf("vector = %v, %v", ok, err)
	}
	if got.len() != len(want) {
		t.Fatalf("read %d blocks, want %d", got.len(), len(want))
	}
	for i, b := range want {
		read, err := got.block(i)
		if err != nil {
			t.Fatalf("block(%d): %v", i, err)
		}
		if read != b {
			t.Errorf("block %d read back as %+v, want %+v", i, read, b)
		}
	}
	if _, err := got.block(len(want)); err == nil {
		t.Error("block past the end of the vector was read")
	}
}
