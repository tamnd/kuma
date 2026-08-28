package ipc_test

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

// The stream is the thin part of this package, so most of what is checked here
// is what happens when the bytes stop early or arrive in the wrong order. The
// values themselves are the batch test's job, and a stream that round trips them
// is only saying that the framing between the messages is right.

// streamOf writes a schema and some batches and gives back the bytes.
func streamOf(t *testing.T, s dtype.Schema, batches ...ipc.Batch) []byte {
	t.Helper()
	var buf bytes.Buffer
	w, err := ipc.NewWriter(&buf, s)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	for i, b := range batches {
		if err := w.Write(b); err != nil {
			t.Fatalf("Write batch %d: %v", i, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	return buf.Bytes()
}

// intBatches is a schema of one column and a batch per slice of values, which is
// the smallest thing that can be several batches long.
func intBatches(t *testing.T, vals ...[]int64) (dtype.Schema, []ipc.Batch) {
	t.Helper()
	s := dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64, Nullable: true}}}
	batches := make([]ipc.Batch, len(vals))
	for i, v := range vals {
		batches[i] = ipc.Batch{Length: len(v), Columns: []*array.Array{buildInts(t, v)}}
	}
	return s, batches
}

// TestStreamRoundTrip writes every batch the encoder can write into a stream of
// its own and reads it back. One stream per case, since a stream carries one
// schema and these are all different ones.
func TestStreamRoundTrip(t *testing.T) {
	for _, c := range batchCases(t) {
		t.Run(c.name, func(t *testing.T) {
			// The same batch three times, so that the framing between messages
			// is exercised rather than just the framing around one.
			stream := streamOf(t, c.schema, c.batch, c.batch, c.batch)

			r, err := ipc.NewReader(bytes.NewReader(stream))
			if err != nil {
				t.Fatalf("NewReader: %v", err)
			}
			if got, want := len(r.Schema().Fields), len(c.schema.Fields); got != want {
				t.Fatalf("the stream carries %d fields, want %d", got, want)
			}

			read := 0
			for r.Next() {
				got := r.Batch()
				if got.Length != c.batch.Length {
					t.Errorf("batch %d has %d rows, want %d", read, got.Length, c.batch.Length)
				}
				for i, col := range c.batch.Columns {
					equalArrays(t, got.Columns[i], col)
				}
				read++
			}
			if err := r.Err(); err != nil {
				t.Fatalf("Err after %d batches: %v", read, err)
			}
			if read != 3 {
				t.Errorf("read %d batches, want 3", read)
			}
		})
	}
}

// TestStreamEmpty checks the stream that is only a schema, which is what a query
// that matched nothing writes.
func TestStreamEmpty(t *testing.T) {
	s, _ := intBatches(t)
	stream := streamOf(t, s)

	r, err := ipc.NewReader(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	if len(r.Schema().Fields) != 1 {
		t.Errorf("the stream carries %d fields, want 1", len(r.Schema().Fields))
	}
	if r.Next() {
		t.Error("Next found a batch in a stream that has none")
	}
	if err := r.Err(); err != nil {
		t.Errorf("Err = %v, want nil at the end of a stream", err)
	}
	// A reader that has finished stays finished rather than reading into
	// whatever comes next on the connection.
	if r.Next() {
		t.Error("Next found a batch after the end of the stream")
	}
}

// TestStreamAll reads a stream with a range loop, and stops half way through
// one, which is the thing an iterator has to get right.
func TestStreamAll(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3}, []int64{4, 5}, []int64{6})
	stream := streamOf(t, s, batches...)

	r, err := ipc.NewReader(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	rows := 0
	for b := range r.All() {
		rows += b.Length
	}
	if err = r.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	if rows != 6 {
		t.Errorf("read %d rows, want 6", rows)
	}

	r, err = ipc.NewReader(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	seen := 0
	for range r.All() {
		seen++
		break
	}
	if seen != 1 {
		t.Errorf("the loop ran %d times, want 1", seen)
	}
	if err := r.Err(); err != nil {
		t.Errorf("Err after a loop that stopped early = %v, want nil", err)
	}
}

// TestStreamNoEndMarker checks the stream that stops after its last batch. The
// marker was optional for long enough that files without it are still around,
// so the end of the file at a message boundary is the end of the stream.
func TestStreamNoEndMarker(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3}, []int64{4, 5})
	stream := streamOf(t, s, batches...)

	r, err := ipc.NewReader(bytes.NewReader(stream[:len(stream)-8]))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	read := 0
	for r.Next() {
		read++
	}
	if err := r.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	if read != 2 {
		t.Errorf("read %d batches, want 2", read)
	}
}

// TestStreamTruncated cuts a stream off in every place a message can be cut and
// checks that the reader says the file is short rather than that it ended.
func TestStreamTruncated(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3})
	stream := streamOf(t, s, batches...)

	// Where the schema message ends, found by reading it back rather than
	// worked out from the bytes.
	head, err := ipc.EncodeSchema(s)
	if err != nil {
		t.Fatalf("EncodeSchema: %v", err)
	}
	schemaEnd := len(head)

	for _, tt := range []struct {
		name string
		at   int
	}{
		{"part of the first prefix", 2},
		{"part of the first length", 6},
		{"part of the schema", schemaEnd - 8},
		{"part of the next prefix", schemaEnd + 2},
		{"part of the next length", schemaEnd + 6},
		{"part of the batch metadata", schemaEnd + 16},
		{"part of the batch body", len(stream) - 16},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cut := stream[:tt.at]
			r, err := ipc.NewReader(bytes.NewReader(cut))
			if err != nil {
				if !errors.Is(err, ipc.ErrMessage) {
					t.Errorf("NewReader = %v, want %v", err, ipc.ErrMessage)
				}
				return
			}
			for r.Next() {
				_ = r.Batch()
			}
			if err := r.Err(); !errors.Is(err, ipc.ErrMessage) {
				t.Errorf("Err = %v, want %v", err, ipc.ErrMessage)
			}
		})
	}
}

// TestStreamHead checks what a reader says about bytes that are not a stream at
// all, which is the first thing it looks at.
func TestStreamHead(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3})
	batch, err := ipc.EncodeBatch(s, batches[0])
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}

	for _, tt := range []struct {
		name string
		head []byte
	}{
		{"nothing at all", nil},
		{"an end of stream marker", []byte{0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0}},
		{"a batch where the schema goes", batch},
		{"a length that is negative", []byte{0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0x80}},
		{"eight bytes of nothing", make([]byte, 8)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := ipc.NewReader(bytes.NewReader(tt.head)); !errors.Is(err, ipc.ErrMessage) {
				t.Errorf("NewReader = %v, want %v", err, ipc.ErrMessage)
			}
		})
	}
}

// TestStreamBatchMismatch checks that a batch is written against the schema the
// stream carries, and that being told about a bad one does not stop the stream.
func TestStreamBatchMismatch(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3})

	var buf bytes.Buffer
	w, err := ipc.NewWriter(&buf, s)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	wrong := ipc.Batch{Length: 2, Columns: []*array.Array{array.Of[float64](1.5, 2.5)}}
	if err = w.Write(wrong); !errors.Is(err, ipc.ErrBuffers) {
		t.Fatalf("Write of a batch of the wrong type = %v, want %v", err, ipc.ErrBuffers)
	}
	// Nothing went out, so the stream is still good and the caller can write
	// the batch it meant to.
	if err = w.Write(batches[0]); err != nil {
		t.Fatalf("Write after a rejected batch: %v", err)
	}
	if err = w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	r, err := ipc.NewReader(&buf)
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	read := 0
	for r.Next() {
		read++
	}
	if err := r.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	if read != 1 {
		t.Errorf("read %d batches, want 1", read)
	}
}

// TestStreamWriterSchema checks that a schema Arrow cannot name is refused
// before anything is written, since the schema is the first message and there is
// no stream without it.
func TestStreamWriterSchema(t *testing.T) {
	bad := dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Time32{Unit: dtype.Nanosecond}}}}
	var buf bytes.Buffer
	if _, err := ipc.NewWriter(&buf, bad); !errors.Is(err, ipc.ErrType) {
		t.Errorf("NewWriter = %v, want %v", err, ipc.ErrType)
	}
	if buf.Len() != 0 {
		t.Errorf("%d bytes went out for a schema that cannot be written, want none", buf.Len())
	}
}

// failWriter takes a fixed number of writes and then stops, which is what a full
// disk or a closed socket looks like from up here.
type failWriter struct {
	ok  int
	err error
}

func (w *failWriter) Write(p []byte) (int, error) {
	if w.ok == 0 {
		return 0, w.err
	}
	w.ok--
	return len(p), nil
}

// TestStreamWriteError checks that an error from underneath sticks. The stream
// is broken at that point, so everything after it would be unreadable and
// reporting the first failure again is more use than reporting a new one.
func TestStreamWriteError(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3})
	boom := errors.New("no room")

	// The schema is the first message, so a writer that takes nothing fails
	// before there is a stream to hand back.
	if _, err := ipc.NewWriter(&failWriter{err: boom}, s); !errors.Is(err, boom) {
		t.Errorf("NewWriter = %v, want %v", err, boom)
	}

	w, err := ipc.NewWriter(&failWriter{ok: 1, err: boom}, s)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.Write(batches[0]); !errors.Is(err, boom) {
		t.Fatalf("Write = %v, want %v", err, boom)
	}
	if err := w.Write(batches[0]); !errors.Is(err, boom) {
		t.Errorf("the second Write = %v, want the first error %v", err, boom)
	}
	if err := w.Close(); !errors.Is(err, boom) {
		t.Errorf("Close = %v, want the first error %v", err, boom)
	}
	if err := w.Close(); !errors.Is(err, boom) {
		t.Errorf("the second Close = %v, want the first error %v", err, boom)
	}
}

// TestStreamClose checks the marker, that closing twice says what closing once
// said, and that a write after the marker is refused rather than appended to a
// stream that has already ended.
func TestStreamClose(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3})

	var buf bytes.Buffer
	w, err := ipc.NewWriter(&buf, s)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err := w.Write(batches[0]); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Errorf("the second Close = %v, want nil", err)
	}

	end := []byte{0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0}
	if got := buf.Bytes(); !bytes.HasSuffix(got, end) {
		t.Errorf("the stream ends with %x, want the end of stream marker %x", got[len(got)-8:], end)
	}
	if err := w.Write(batches[0]); !errors.Is(err, ipc.ErrClosed) {
		t.Errorf("Write after Close = %v, want %v", err, ipc.ErrClosed)
	}
}

// TestStreamKeepsBatches checks that a batch stays readable after the next one
// has been read. The arrays point into the bytes they came out of, so a reader
// that reused one buffer would rewrite a batch somebody was still holding.
func TestStreamKeepsBatches(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3}, []int64{4, 5, 6}, []int64{7, 8, 9})
	stream := streamOf(t, s, batches...)

	r, err := ipc.NewReader(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	var held []ipc.Batch
	for r.Next() {
		held = append(held, r.Batch())
	}
	if err := r.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	if len(held) != len(batches) {
		t.Fatalf("read %d batches, want %d", len(held), len(batches))
	}
	for i, b := range held {
		equalArrays(t, b.Columns[0], batches[i].Columns[0])
	}
}

// TestStreamReaderPipe reads a stream out of something that hands over a few
// bytes at a time, which is what a socket does and what a bytes.Reader never
// does.
func TestStreamReaderPipe(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3}, []int64{4, 5, 6})
	stream := streamOf(t, s, batches...)

	r, err := ipc.NewReader(&dribble{r: bytes.NewReader(stream)})
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	read := 0
	for r.Next() {
		equalArrays(t, r.Batch().Columns[0], batches[read].Columns[0])
		read++
	}
	if err := r.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	if read != 2 {
		t.Errorf("read %d batches, want 2", read)
	}
}

// dribble hands over three bytes at a time, however many were asked for.
type dribble struct{ r io.Reader }

func (d *dribble) Read(p []byte) (int, error) {
	if len(p) > 3 {
		p = p[:3]
	}
	return d.r.Read(p)
}

func ExampleWriter() {
	s := dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64},
		{Name: "symbol", Type: dtype.String},
	}}

	var buf bytes.Buffer
	w, err := ipc.NewWriter(&buf, s)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, ids := range [][]int64{{1, 2}, {3}} {
		symbols := make([]string, len(ids))
		for i, id := range ids {
			symbols[i] = fmt.Sprintf("S%d", id)
		}
		err = w.Write(ipc.Batch{Length: len(ids), Columns: []*array.Array{
			array.Of(ids...), array.OfStrings(symbols...),
		}})
		if err != nil {
			fmt.Println(err)
			return
		}
	}
	if err = w.Close(); err != nil {
		fmt.Println(err)
		return
	}

	r, err := ipc.NewReader(&buf)
	if err != nil {
		fmt.Println(err)
		return
	}
	for b := range r.All() {
		for i := range b.Length {
			fmt.Println(b.Columns[0].Value[int64](i), string(b.Columns[1].Bytes(i)))
		}
	}
	if err := r.Err(); err != nil {
		fmt.Println(err)
	}
	// Output:
	// 1 S1
	// 2 S2
	// 3 S3
}

func BenchmarkStreamWrite(b *testing.B) {
	s, batch := benchBatch()
	var buf bytes.Buffer
	for b.Loop() {
		buf.Reset()
		w, err := ipc.NewWriter(&buf, s)
		if err != nil {
			b.Fatal(err)
		}
		if err := w.Write(batch); err != nil {
			b.Fatal(err)
		}
		if err := w.Close(); err != nil {
			b.Fatal(err)
		}
	}
	b.SetBytes(int64(buf.Len()))
}

func BenchmarkStreamRead(b *testing.B) {
	s, batch := benchBatch()
	var buf bytes.Buffer
	w, err := ipc.NewWriter(&buf, s)
	if err != nil {
		b.Fatal(err)
	}
	if err := w.Write(batch); err != nil {
		b.Fatal(err)
	}
	if err := w.Close(); err != nil {
		b.Fatal(err)
	}
	stream := buf.Bytes()

	b.SetBytes(int64(len(stream)))
	for b.Loop() {
		r, err := ipc.NewReader(bytes.NewReader(stream))
		if err != nil {
			b.Fatal(err)
		}
		for r.Next() {
			_ = r.Batch()
		}
		if err := r.Err(); err != nil {
			b.Fatal(err)
		}
	}
}
