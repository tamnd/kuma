package ipc_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

// A file is a stream with a header and an index around it, so the values going
// round the loop are the stream test's job and most of what is checked here is
// the index: that a batch can be read without the ones in front of it, that the
// footer says where everything is, and that a file which is not one is refused
// before a single number in it is believed.

// fileOf writes a schema and some batches and gives back the bytes.
func fileOf(t *testing.T, s dtype.Schema, batches ...ipc.Batch) []byte {
	t.Helper()
	var buf bytes.Buffer
	w, err := ipc.NewFileWriter(&buf, s)
	if err != nil {
		t.Fatalf("NewFileWriter: %v", err)
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

// openFile reads the footer of a file held in memory.
func openFile(t *testing.T, file []byte) *ipc.FileReader {
	t.Helper()
	r, err := ipc.NewFileReader(bytes.NewReader(file), int64(len(file)))
	if err != nil {
		t.Fatalf("NewFileReader: %v", err)
	}
	return r
}

// TestFileRoundTrip writes every batch the encoder can write into a file of its
// own and reads it back. Three copies of each, since one batch says nothing
// about whether the second one was indexed where it landed.
func TestFileRoundTrip(t *testing.T) {
	for _, c := range batchCases(t) {
		t.Run(c.name, func(t *testing.T) {
			r := openFile(t, fileOf(t, c.schema, c.batch, c.batch, c.batch))
			if got, want := len(r.Schema().Fields), len(c.schema.Fields); got != want {
				t.Fatalf("the file carries %d fields, want %d", got, want)
			}
			if r.NumBatches() != 3 {
				t.Fatalf("the file holds %d batches, want 3", r.NumBatches())
			}

			// Backwards, which is the order a reader of a stream cannot take
			// them in and the reason the format has a footer at all.
			for i := r.NumBatches() - 1; i >= 0; i-- {
				got, err := r.Batch(i)
				if err != nil {
					t.Fatalf("Batch(%d): %v", i, err)
				}
				if got.Length != c.batch.Length {
					t.Errorf("batch %d has %d rows, want %d", i, got.Length, c.batch.Length)
				}
				for k, col := range c.batch.Columns {
					equalArrays(t, got.Columns[k], col)
				}
			}
		})
	}
}

// TestFileRandomAccess reads the batches of a file in no particular order and
// more than once. A batch that came out of the index has to be the batch the
// index named, whatever was read before it.
func TestFileRandomAccess(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3}, []int64{4, 5}, []int64{6}, nil, []int64{7, 8, 9, 10})
	r := openFile(t, fileOf(t, s, batches...))

	if r.NumBatches() != len(batches) {
		t.Fatalf("the file holds %d batches, want %d", r.NumBatches(), len(batches))
	}
	for _, i := range []int{4, 0, 3, 4, 2, 1, 0} {
		got, err := r.Batch(i)
		if err != nil {
			t.Fatalf("Batch(%d): %v", i, err)
		}
		equalArrays(t, got.Columns[0], batches[i].Columns[0])
	}
}

// TestFileKeepsBatches checks that a batch stays readable after another one has
// been read. Every batch is read into a buffer of its own, so holding one costs
// memory and never costs the values in it.
func TestFileKeepsBatches(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3}, []int64{4, 5, 6}, []int64{7, 8, 9})
	r := openFile(t, fileOf(t, s, batches...))

	held := make([]ipc.Batch, r.NumBatches())
	for i := range held {
		b, err := r.Batch(i)
		if err != nil {
			t.Fatalf("Batch(%d): %v", i, err)
		}
		held[i] = b
	}
	for i, b := range held {
		equalArrays(t, b.Columns[0], batches[i].Columns[0])
	}
}

// TestFileEmpty checks the file that is only a schema and a footer, which is
// what a query that matched nothing writes.
func TestFileEmpty(t *testing.T) {
	s, _ := intBatches(t)
	r := openFile(t, fileOf(t, s))

	if len(r.Schema().Fields) != 1 {
		t.Errorf("the file carries %d fields, want 1", len(r.Schema().Fields))
	}
	if r.NumBatches() != 0 {
		t.Errorf("the file holds %d batches, want none", r.NumBatches())
	}
	if _, err := r.Batch(0); err == nil {
		t.Error("Batch(0) of a file with no batches in it returned one")
	}
}

// TestFileMetadata checks that the metadata on a schema survives the footer,
// which is the second place a schema is written and the one nothing else in the
// package reads.
func TestFileMetadata(t *testing.T) {
	s := dtype.Schema{
		Fields: []dtype.Field{{Name: "id", Type: dtype.Int64,
			Metadata: dtype.Metadata{{Key: "unit", Value: "meters"}}}},
		Metadata: dtype.Metadata{{Key: "written by", Value: "kuma"}},
	}
	r := openFile(t, fileOf(t, s))
	if !r.Schema().Equal(s) {
		t.Errorf("the file carries %v, want %v", r.Schema(), s)
	}
}

// TestFileIndex checks the batch numbers that are not in the file. There is
// nothing to read at either end of the index, and a reader that let one through
// would read a batch out of whatever the footer happened to sit next to.
func TestFileIndex(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3})
	r := openFile(t, fileOf(t, s, batches...))

	for _, i := range []int{-1, 1, 1 << 20} {
		if _, err := r.Batch(i); err == nil {
			t.Errorf("Batch(%d) of a file of one batch returned one", i)
		}
	}
}

// TestFileIsAStream reads the middle of a file with the stream reader. The two
// formats are the same bytes with a header and an index around them, and this is
// the check that says so rather than a comment claiming it.
func TestFileIsAStream(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3}, []int64{4, 5})
	file := fileOf(t, s, batches...)

	r, err := ipc.NewReader(bytes.NewReader(file[8:]))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	read := 0
	for b := range r.All() {
		equalArrays(t, b.Columns[0], batches[read].Columns[0])
		read++
	}
	if err := r.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	if read != len(batches) {
		t.Errorf("the stream inside the file has %d batches, want %d", read, len(batches))
	}
}

// TestFileHead checks the bytes a reader looks at before it believes anything
// else: the magic at both ends and the length of the footer between them.
func TestFileHead(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3})
	file := fileOf(t, s, batches...)

	// Where the footer starts, and where its length is written.
	length := len(file) - 10
	footer := int32(binary.LittleEndian.Uint32(file[length:]))

	for _, tt := range []struct {
		name string
		make func([]byte)
	}{
		{"no magic at the front", func(b []byte) { copy(b, "ARROW2") }},
		{"no magic at the end", func(b []byte) { copy(b[len(b)-6:], "ARROW2") }},
		{"a footer of no bytes", func(b []byte) {
			binary.LittleEndian.PutUint32(b[length:], 0)
		}},
		{"a footer of minus one bytes", func(b []byte) {
			binary.LittleEndian.PutUint32(b[length:], 0xFFFFFFFF)
		}},
		{"a footer larger than the file", func(b []byte) {
			binary.LittleEndian.PutUint32(b[length:], uint32(len(b)))
		}},
		{"a footer one byte too long", func(b []byte) {
			binary.LittleEndian.PutUint32(b[length:], uint32(footer)+1)
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			b := bytes.Clone(file)
			tt.make(b)
			_, err := ipc.NewFileReader(bytes.NewReader(b), int64(len(b)))
			if !errors.Is(err, ipc.ErrMessage) {
				t.Errorf("NewFileReader = %v, want %v", err, ipc.ErrMessage)
			}
		})
	}
}

// TestFileTruncated cuts a file off in every place it can be cut and checks that
// the reader says so. A file is read from the back, so a file with its tail
// missing is one nothing can be found in even though every batch is still there.
func TestFileTruncated(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3}, []int64{4, 5})
	file := fileOf(t, s, batches...)

	for _, tt := range []struct {
		name string
		at   int
	}{
		{"nothing at all", 0},
		{"the magic and no more", 8},
		{"part of the first batch", len(file) / 2},
		{"the messages and no footer", len(file) - 40},
		{"all but the magic at the end", len(file) - 6},
		{"all but the last byte", len(file) - 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cut := file[:tt.at]
			if _, err := ipc.NewFileReader(bytes.NewReader(cut), int64(len(cut))); err == nil {
				t.Error("NewFileReader read a footer out of a file that stops before one")
			}
		})
	}
}

// TestFileShortRead checks a reader that is told the file is larger than it is,
// which is what a file still being written looks like to somebody who read the
// size first.
func TestFileShortRead(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3})
	file := fileOf(t, s, batches...)

	_, err := ipc.NewFileReader(bytes.NewReader(file), int64(len(file))+64)
	if !errors.Is(err, ipc.ErrMessage) {
		t.Errorf("NewFileReader = %v, want %v", err, ipc.ErrMessage)
	}
}

// TestFileBatchMismatch checks that a batch is written against the schema the
// file carries, and that being told about a bad one leaves the file alone. A
// batch that went half way out would be indexed at an offset holding something
// else.
func TestFileBatchMismatch(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3})

	var buf bytes.Buffer
	w, err := ipc.NewFileWriter(&buf, s)
	if err != nil {
		t.Fatalf("NewFileWriter: %v", err)
	}
	wrong := ipc.Batch{Length: 2, Columns: []*array.Array{array.Of[float64](1.5, 2.5)}}
	if err = w.Write(wrong); !errors.Is(err, ipc.ErrBuffers) {
		t.Fatalf("Write of a batch of the wrong type = %v, want %v", err, ipc.ErrBuffers)
	}
	if err = w.Write(batches[0]); err != nil {
		t.Fatalf("Write after a rejected batch: %v", err)
	}
	if err = w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	r := openFile(t, buf.Bytes())
	if r.NumBatches() != 1 {
		t.Fatalf("the file holds %d batches, want 1", r.NumBatches())
	}
	got, err := r.Batch(0)
	if err != nil {
		t.Fatalf("Batch(0): %v", err)
	}
	equalArrays(t, got.Columns[0], batches[0].Columns[0])
}

// TestFileWriterSchema checks that a schema Arrow cannot name is refused before
// anything is written, since there is no file without one.
func TestFileWriterSchema(t *testing.T) {
	bad := dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Time32{Unit: dtype.Nanosecond}}}}
	var buf bytes.Buffer
	if _, err := ipc.NewFileWriter(&buf, bad); !errors.Is(err, ipc.ErrType) {
		t.Errorf("NewFileWriter = %v, want %v", err, ipc.ErrType)
	}
	if buf.Len() != 0 {
		t.Errorf("%d bytes went out for a schema that cannot be written, want none", buf.Len())
	}
}

// TestFileWriteError checks that an error from underneath sticks, including the
// one that happens while the footer is going out. A file whose footer was cut
// off is unreadable, so the last thing a writer does is the one that most needs
// to be reported.
func TestFileWriteError(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3})
	boom := errors.New("no room")

	// The magic is the first thing written, so a writer that takes nothing fails
	// before there is a file to hand back.
	if _, err := ipc.NewFileWriter(&failWriter{err: boom}, s); !errors.Is(err, boom) {
		t.Errorf("NewFileWriter = %v, want %v", err, boom)
	}
	if _, err := ipc.NewFileWriter(&failWriter{ok: 1, err: boom}, s); !errors.Is(err, boom) {
		t.Errorf("NewFileWriter with room for the magic = %v, want %v", err, boom)
	}

	// The writers past that point all open, so the interesting error is the one
	// out of a later call rather than the one out of this.
	open := func(ok int) *ipc.FileWriter {
		t.Helper()
		w, err := ipc.NewFileWriter(&failWriter{ok: ok, err: boom}, s)
		if err != nil {
			t.Fatalf("NewFileWriter with room for %d writes: %v", ok, err)
		}
		return w
	}

	w := open(2)
	if err := w.Write(batches[0]); !errors.Is(err, boom) {
		t.Fatalf("Write = %v, want %v", err, boom)
	}
	if err := w.Write(batches[0]); !errors.Is(err, boom) {
		t.Errorf("the second Write = %v, want the first error %v", err, boom)
	}
	if err := w.Close(); !errors.Is(err, boom) {
		t.Errorf("Close = %v, want the first error %v", err, boom)
	}

	// A writer that takes everything but the footer, which is the failure that
	// leaves a directory full of files nothing can open.
	w = open(4)
	if err := w.Write(batches[0]); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err := w.Close(); !errors.Is(err, boom) {
		t.Errorf("Close = %v, want %v", err, boom)
	}
	if err := w.Close(); !errors.Is(err, boom) {
		t.Errorf("the second Close = %v, want the first error %v", err, boom)
	}
}

// TestFileClose checks the magic at the end, that closing twice says what
// closing once said, and that a write after the footer is refused rather than
// appended to a file whose index has already gone out.
func TestFileClose(t *testing.T) {
	s, batches := intBatches(t, []int64{1, 2, 3})

	var buf bytes.Buffer
	w, err := ipc.NewFileWriter(&buf, s)
	if err != nil {
		t.Fatalf("NewFileWriter: %v", err)
	}
	if err := w.Write(batches[0]); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	written := buf.Len()
	if err := w.Close(); err != nil {
		t.Errorf("the second Close = %v, want nil", err)
	}
	if buf.Len() != written {
		t.Errorf("the second Close wrote %d more bytes, want none", buf.Len()-written)
	}

	if got := buf.Bytes(); !bytes.HasSuffix(got, []byte("ARROW1")) {
		t.Errorf("the file ends with %q, want the magic", got[len(got)-6:])
	}
	if err := w.Write(batches[0]); !errors.Is(err, ipc.ErrClosed) {
		t.Errorf("Write after Close = %v, want %v", err, ipc.ErrClosed)
	}
}

func ExampleFileWriter() {
	s := dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64},
		{Name: "symbol", Type: dtype.String},
	}}

	var buf bytes.Buffer
	w, err := ipc.NewFileWriter(&buf, s)
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

	// The last batch of the file, without reading the ones in front of it.
	file := buf.Bytes()
	r, err := ipc.NewFileReader(bytes.NewReader(file), int64(len(file)))
	if err != nil {
		fmt.Println(err)
		return
	}
	b, err := r.Batch(r.NumBatches() - 1)
	if err != nil {
		fmt.Println(err)
		return
	}
	for i := range b.Length {
		fmt.Println(b.Columns[0].Value[int64](i), string(b.Columns[1].Bytes(i)))
	}
	// Output:
	// 3 S3
}

func BenchmarkFileWrite(b *testing.B) {
	s, batch := benchBatch()
	var buf bytes.Buffer
	for b.Loop() {
		buf.Reset()
		w, err := ipc.NewFileWriter(&buf, s)
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

func BenchmarkFileRead(b *testing.B) {
	s, batch := benchBatch()
	var buf bytes.Buffer
	w, err := ipc.NewFileWriter(&buf, s)
	if err != nil {
		b.Fatal(err)
	}
	if err := w.Write(batch); err != nil {
		b.Fatal(err)
	}
	if err := w.Close(); err != nil {
		b.Fatal(err)
	}
	file := buf.Bytes()

	b.SetBytes(int64(len(file)))
	for b.Loop() {
		r, err := ipc.NewFileReader(bytes.NewReader(file), int64(len(file)))
		if err != nil {
			b.Fatal(err)
		}
		for i := range r.NumBatches() {
			if _, err := r.Batch(i); err != nil {
				b.Fatal(err)
			}
		}
	}
}
