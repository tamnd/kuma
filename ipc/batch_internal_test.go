package ipc

import (
	"encoding/binary"
	"errors"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
)

// The batches kuma would not write. Half of these are what another
// implementation sends, since almost nothing else writes text as views, and the
// other half are the ones that are well formed and dishonest. Both need a
// message built by hand, which is why they are in here rather than beside the
// round trip.

// handBatch writes a record batch message and the body it describes. The
// arguments are the parts a writer would work out for itself, so that a test can
// get any of them wrong on purpose.
func handBatch(rows int64, nodes, buffers []span, variadic []int64, body []byte, compressed bool) []byte {
	var w fbBuilder

	nodesVec := w.spans(nodes)
	buffersVec := w.spans(buffers)
	variadicVec := fbOffset(0)
	if len(variadic) > 0 {
		variadicVec = w.int64s(variadic)
	}
	compression := fbOffset(0)
	if compressed {
		w.startTable()
		compression = w.endTable()
	}

	w.startTable()
	w.slotInt(fbBatchLength, rows, 0)
	w.slotOffset(fbBatchNodes, nodesVec)
	w.slotOffset(fbBatchBuffers, buffersVec)
	w.slotOffset(fbBatchCompression, compression)
	w.slotOffset(fbBatchVariadic, variadicVec)
	batch := w.endTable()

	w.startTable()
	w.slotInt(fbMessageVersion, fbVersionV5, 0)
	w.slotUint8(fbMessageHeaderType, fbHeaderRecordBatch)
	w.slotOffset(fbMessageHeader, batch)
	w.slotInt(fbMessageBodyLength, int64(len(body)), 0)
	return append(frame(w.finish(w.endTable())), body...)
}

// offsetBody writes a column of strings the way every implementation but this
// one does: a validity buffer, one offset per value plus a last one, and the
// bytes they cut up.
func offsetBody(vals []string, wide bool) ([]byte, []span) {
	var offsets, data []byte
	put := func(n int) {
		if wide {
			offsets = binary.LittleEndian.AppendUint64(offsets, uint64(n))
			return
		}
		offsets = binary.LittleEndian.AppendUint32(offsets, uint32(n))
	}

	put(0)
	for _, v := range vals {
		data = append(data, v...)
		put(len(data))
	}

	var b bodyWriter
	b.add(nil)
	b.add(offsets)
	b.add(data)
	b.pad()
	return b.buf, b.descs
}

// TestDecodeBatchOffsets reads the layout another implementation writes.
//
// A schema says a column holds text and not which of the four ways it is laid
// out, since kuma has one type for all of them, so the reader has to work it out
// from the batch. No variadic buffer count means the offset layout, and the two
// widths of that are told apart by how long the offsets are.
func TestDecodeBatchOffsets(t *testing.T) {
	vals := []string{"a", "bb", "ccc"}
	s := dtype.Schema{Fields: []dtype.Field{{Name: "text", Type: dtype.String, Nullable: true}}}

	for _, wide := range []bool{false, true} {
		name := "offsets32"
		if wide {
			name = "offsets64"
		}
		t.Run(name, func(t *testing.T) {
			body, buffers := offsetBody(vals, wide)
			msg := handBatch(int64(len(vals)), []span{{first: int64(len(vals))}}, buffers, nil, body, false)

			b, rest, err := DecodeBatch(s, msg)
			if err != nil {
				t.Fatalf("DecodeBatch: %v", err)
			}
			if len(rest) != 0 {
				t.Errorf("%d bytes left after the batch, want none", len(rest))
			}
			if b.Length != len(vals) {
				t.Fatalf("batch of %d rows, want %d", b.Length, len(vals))
			}
			for i, want := range vals {
				if got := string(b.Columns[0].Bytes(i)); got != want {
					t.Errorf("value %d = %q, want %q", i, got, want)
				}
			}
		})
	}
}

// TestDecodeBatchMixedText checks the one batch this cannot take apart. Two text
// columns and one variadic buffer count means one of them is in the view layout
// and one is not, and nothing in either message says which.
func TestDecodeBatchMixedText(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{
		{Name: "a", Type: dtype.String},
		{Name: "b", Type: dtype.String},
	}}
	body, buffers := offsetBody([]string{"a"}, false)
	msg := handBatch(1, []span{{first: 1}, {first: 1}}, buffers, []int64{0}, body, false)

	err := decodeError(t, s, msg)
	if !errors.Is(err, ErrUnsupported) {
		t.Fatalf("a batch of two text layouts: %v, want ErrUnsupported", err)
	}
	if !strings.Contains(err.Error(), "two different ways") {
		t.Errorf("%v, want it to say what it could not tell apart", err)
	}
}

// TestDecodeBatchCompressed checks that a compressed body is refused rather than
// read as if it were not one. LZ4 and Zstd are both in the format and neither is
// here yet, and a reader that ignores the field reads noise.
func TestDecodeBatchCompressed(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64}}}
	msg := handBatch(0, []span{{}}, []span{{}, {}}, nil, nil, true)

	if err := decodeError(t, s, msg); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("a compressed body: %v, want ErrUnsupported", err)
	}
}

// TestDecodeBatchCounts checks the batches whose metadata does not add up: a
// node per column and a buffer per buffer are what a reader has no way to guess
// at, so a batch that brings the wrong number of either is one written against
// another schema.
func TestDecodeBatchCounts(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64}}}
	values := make([]byte, 8)
	var b bodyWriter
	b.add(nil)
	b.add(values)
	b.pad()

	for _, tt := range []struct {
		name    string
		nodes   []span
		buffers []span
		want    string
	}{
		{"a node too many", []span{{first: 1}, {first: 1}}, b.descs, "field nodes"},
		{"no nodes", nil, b.descs, "field nodes"},
		{"a buffer too few", []span{{first: 1}}, b.descs[:1], "buffers"},
		{"a buffer too many", []span{{first: 1}}, append(b.descs, span{}), "buffers"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			msg := handBatch(1, tt.nodes, tt.buffers, nil, b.buf, false)
			err := decodeError(t, s, msg)
			if !errors.Is(err, ErrMessage) {
				t.Fatalf("%v, want ErrMessage", err)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("%v, want it to mention the %s", err, tt.want)
			}
		})
	}
}

// TestDecodeBatchOutside checks the buffer descriptions that point somewhere
// other than the body. This is the number a reader would otherwise index on, so
// it is the one worth a test of its own.
func TestDecodeBatchOutside(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64}}}
	body := make([]byte, 8)

	for _, tt := range []struct {
		name   string
		buffer span
	}{
		{"past the end", span{first: 8, second: 8}},
		{"longer than the body", span{first: 0, second: 4096}},
		{"a negative offset", span{first: -8, second: 8}},
		{"a negative length", span{first: 0, second: -1}},
		{"an offset that wraps", span{first: 1 << 62, second: 1 << 62}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			msg := handBatch(1, []span{{first: 1}}, []span{{}, tt.buffer}, nil, body, false)
			if err := decodeError(t, s, msg); !errors.Is(err, ErrMessage) {
				t.Fatalf("%v, want ErrMessage", err)
			}
		})
	}
}

// TestDecodeBatchBody checks a message whose body is not all there, which is
// what a stream cut off part way through one looks like.
func TestDecodeBatchBody(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64}}}
	var b bodyWriter
	b.add(nil)
	b.add(make([]byte, 8))
	b.pad()

	msg := handBatch(1, []span{{first: 1}}, b.descs, nil, b.buf, false)
	if err := decodeError(t, s, msg[:len(msg)-8]); !errors.Is(err, ErrMessage) {
		t.Fatalf("a body that is not all there: %v, want ErrMessage", err)
	}
}

// TestDecodeBatchRows checks that every column has to be as long as the batch
// says it is. A node that disagrees is a message that would have a reader
// building a column of one length out of buffers holding another.
func TestDecodeBatchRows(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64}}}
	var b bodyWriter
	b.add(nil)
	b.add(make([]byte, 16))
	b.pad()

	msg := handBatch(2, []span{{first: 1}}, b.descs, nil, b.buf, false)
	err := decodeError(t, s, msg)
	if !errors.Is(err, ErrMessage) {
		t.Fatalf("a column shorter than its batch: %v, want ErrMessage", err)
	}
	if !strings.Contains(err.Error(), `"id"`) {
		t.Errorf("%v, want it to name the column", err)
	}
}

// decodeError is the error of a decode that has to fail.
func decodeError(t *testing.T, s dtype.Schema, msg []byte) error {
	t.Helper()
	b, _, err := DecodeBatch(s, msg)
	if err == nil {
		t.Fatalf("DecodeBatch read a batch of %d rows, want an error", b.Length)
	}
	return err
}
