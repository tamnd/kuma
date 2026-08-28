package ipc_test

import (
	"bytes"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

// FuzzType checks that reading a format string and writing it back gives the
// same string, for every string that reads as a type at all.
//
// The pair is not quite an inverse, since three text layouts read as one kuma
// type, so the property is stated one step along: whatever Type accepts,
// Format writes, and reading that gives the same type again. A format string
// that survives one trip and not two would be one this package can import and
// then fail to export, which is the bug worth catching.
//
// Format strings are short and structured, so the fuzzer finds the interesting
// ones quickly: a decimal with a comma in the wrong place, a timestamp with a
// zone that is itself a colon, a width that overflows an int32.
func FuzzType(f *testing.F) {
	for _, tt := range mappings {
		f.Add(tt.format)
	}
	f.Add("u")
	f.Add("d:18,2,128")
	f.Add("ts")
	f.Add("+w:")
	f.Add("")

	f.Fuzz(func(t *testing.T, format string) {
		// No children, so the format strings that need one fail here. The
		// nested types are covered by the table, and what is being fuzzed is
		// the parsing of the string rather than the assembly of a tree.
		first, err := ipc.Type(format, nil)
		if err != nil {
			return
		}

		written, err := ipc.Format(first)
		if err != nil {
			t.Fatalf("Type(%q) = %s, which Format refuses: %v", format, first, err)
		}
		second, err := ipc.Type(written, nil)
		if err != nil {
			t.Fatalf("Type(%q) = %s, written as %q, which Type refuses: %v",
				format, first, written, err)
		}
		if !dtype.Equal(first, second) {
			t.Fatalf("Type(%q) = %s, written as %q, which reads as %s",
				format, first, written, second)
		}
		if again, err := ipc.Format(second); err != nil || again != written {
			t.Fatalf("Format(%s) = %q, %v, want %q", second, again, err, written)
		}
	})
}

// FuzzMetadata checks that a blob this package accepts is one it writes back
// byte for byte.
//
// The blob is the one thing here that arrives as untrusted bytes from another
// process, so the first thing being asked is whether a truncated or a hostile
// one is refused rather than read past. The second is that the encoding is
// canonical, since a decoder that accepts two spellings of the same metadata
// would round trip through kuma and come out different.
func FuzzMetadata(f *testing.F) {
	seeds := []dtype.Metadata{
		nil,
		{{Key: "unit", Value: "meters"}},
		{{Key: "", Value: ""}},
		{{Key: "a", Value: "1"}, {Key: "a", Value: "2"}},
	}
	for _, m := range seeds {
		b, err := ipc.EncodeMetadata(m)
		if err != nil {
			f.Fatalf("EncodeMetadata = %v", err)
		}
		f.Add(b)
	}
	f.Add([]byte{1, 0, 0, 0})
	f.Add([]byte{255, 255, 255, 255})

	f.Fuzz(func(t *testing.T, blob []byte) {
		m, err := ipc.DecodeMetadata(blob)
		if err != nil {
			return
		}

		written, err := ipc.EncodeMetadata(m)
		if err != nil {
			t.Fatalf("DecodeMetadata(% x) = %v, which EncodeMetadata refuses: %v", blob, m, err)
		}
		if len(m) > 0 && !bytes.Equal(blob, written) {
			t.Fatalf("DecodeMetadata(% x) = %v, written back as % x", blob, m, written)
		}

		again, err := ipc.DecodeMetadata(written)
		if err != nil {
			t.Fatalf("EncodeMetadata(%v) = % x, which DecodeMetadata refuses: %v", m, written, err)
		}
		if !again.Equal(m) {
			t.Fatalf("round trip of %v gave %v", m, again)
		}
	})
}

// FuzzSchema checks that no arrangement of bytes gets a schema past the reader
// that the writer would then refuse.
//
// A schema message is FlatBuffers, which is offsets all the way down, so a
// single byte changed anywhere in one is an offset pointing at something else.
// The reader is written by hand precisely so that every one of those is bounds
// checked, and this is what says so. The property on the way out matters as
// much: a message that reads as a schema has to be a schema this package would
// write, since anything else is a shape that got past the door and will be
// handed to a builder.
func FuzzSchema(f *testing.F) {
	for _, c := range schemaCases {
		b, err := ipc.EncodeSchema(c.schema)
		if err != nil {
			f.Fatalf("EncodeSchema: %v", err)
		}
		f.Add(b)
	}
	f.Add([]byte(nil))
	f.Add([]byte{0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0})

	f.Fuzz(func(t *testing.T, msg []byte) {
		s, err := ipc.DecodeSchema(msg)
		if err != nil {
			return
		}

		written, err := ipc.EncodeSchema(s)
		if err != nil {
			t.Fatalf("DecodeSchema(% x) = %v, which EncodeSchema refuses: %v", msg, s, err)
		}
		again, err := ipc.DecodeSchema(written)
		if err != nil {
			t.Fatalf("EncodeSchema(%v) = % x, which DecodeSchema refuses: %v", s, written, err)
		}
		if !again.Equal(s) {
			t.Fatalf("round trip of %v gave %v", s, again)
		}
	})
}

// FuzzImport checks that no arrangement of buffers makes an import read past
// the end of one.
//
// This is the call that takes bytes from another process and starts indexing
// with the numbers it finds inside them, so the property being asked for is
// mostly that a bad layout is refused rather than believed. What comes back
// from a layout that is accepted has to hold together: every value readable,
// exportable again, and the same values on the way back in.
func FuzzImport(f *testing.F) {
	f.Add(0, 3, 0, []byte{0xFF}, make([]byte, 24), []byte(nil))
	f.Add(6, 2, 0, []byte(nil), []byte{0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0}, []byte("abcd"))
	f.Add(10, 1, 0, []byte(nil), make([]byte, 16), []byte(nil))
	f.Add(12, 1, 1, []byte(nil), make([]byte, 8), []byte(nil))

	f.Fuzz(func(t *testing.T, which, length, offset int, valid, values, data []byte) {
		format := fuzzFormats[uint(which)%uint(len(fuzzFormats))]
		length = int(uint(length) % 512)
		offset = int(uint(offset) % 512)

		for _, buffers := range [][][]byte{{valid, values}, {valid, values, data}} {
			l := ipc.Layout{Length: length, Offset: offset, NullCount: -1, Buffers: buffers}
			a, err := ipc.Import(format, l)
			if err != nil {
				continue
			}
			if a.Len() != length {
				t.Fatalf("Import(%q) = %d values, want %d", format, a.Len(), length)
			}

			out, err := ipc.Export(a)
			if err != nil {
				t.Fatalf("Import(%q) built an array Export refuses: %v", format, err)
			}
			written, err := ipc.Format(a.DType())
			if err != nil {
				t.Fatalf("Import(%q) built a %s, which Format refuses: %v", format, a.DType(), err)
			}
			back, err := ipc.Import(written, out)
			if err != nil {
				t.Fatalf("Import(%q) exported as %q, which Import refuses: %v", format, written, err)
			}
			equalArrays(t, back, a)
		}
	})
}

// FuzzBatch checks that no arrangement of bytes gets a record batch past the
// reader that then falls apart.
//
// A batch is the message that decides what to index and how far. The metadata
// says how many rows there are, how many values each column has and where in
// the body every buffer starts, and all of those are numbers somebody else
// wrote. The reader has to refuse the ones that do not add up rather than build
// a column out of them, and this is what says so.
//
// What is accepted has to survive being written again. Reading the offset
// layout gives a column kuma writes as views, so the message is not the same
// message the second time, but the values are the same values, and a batch that
// changes on the way through is one this package would corrupt in the middle of
// a copy from one file to another.
func FuzzBatch(f *testing.F) {
	for i, s := range fuzzBatchSchemas {
		msg, err := ipc.EncodeBatch(s, fuzzBatchOf(f, s))
		if err != nil {
			f.Fatalf("EncodeBatch: %v", err)
		}
		f.Add(i, msg)
	}
	f.Add(0, []byte(nil))
	f.Add(0, []byte{0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0})

	f.Fuzz(func(t *testing.T, which int, msg []byte) {
		s := fuzzBatchSchemas[uint(which)%uint(len(fuzzBatchSchemas))]
		b, _, err := ipc.DecodeBatch(s, msg)
		if err != nil {
			return
		}
		if len(b.Columns) != len(s.Fields) {
			t.Fatalf("a batch of %d columns for a schema of %d fields", len(b.Columns), len(s.Fields))
		}
		for i, col := range b.Columns {
			if col.Len() != b.Length {
				t.Fatalf("column %q has %d values in a batch of %d rows",
					s.Fields[i].Name, col.Len(), b.Length)
			}
		}

		// A null column has no buffers, so a batch of nothing else can claim any
		// number of rows without carrying a byte for them. That is not a lie the
		// reader can catch, and it is the one thing here that would take as long
		// as the number says to look at.
		if b.Length > 1<<20 {
			return
		}

		written, err := ipc.EncodeBatch(s, b)
		if err != nil {
			t.Fatalf("DecodeBatch read a batch of %d rows that EncodeBatch refuses: %v", b.Length, err)
		}
		again, rest, err := ipc.DecodeBatch(s, written)
		if err != nil {
			t.Fatalf("EncodeBatch wrote a batch DecodeBatch refuses: %v", err)
		}
		if len(rest) != 0 {
			t.Fatalf("%d bytes left after a batch this package wrote, want none", len(rest))
		}
		if again.Length != b.Length {
			t.Fatalf("round trip of %d rows gave %d", b.Length, again.Length)
		}
		for i, col := range b.Columns {
			equalArrays(t, again.Columns[i], col)
		}
	})
}

// The schemas a fuzzed batch is read against. A batch on the wire says nothing
// about what it holds, so there has to be a schema to read it with, and these
// are the shapes worth having one for: the fixed width column, the text column
// whose layout is inferred rather than declared, the column with no buffers at
// all, and the batch of no columns whose only content is a row count.
var fuzzBatchSchemas = []dtype.Schema{
	{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64, Nullable: true}}},
	{Fields: []dtype.Field{{Name: "text", Type: dtype.String, Nullable: true}}},
	{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64, Nullable: true},
		{Name: "text", Type: dtype.String, Nullable: true},
		{Name: "live", Type: dtype.Bool, Nullable: true},
	}},
	{Fields: []dtype.Field{
		{Name: "nothing", Type: dtype.Null, Nullable: true},
		{Name: "bytes", Type: dtype.Binary, Nullable: true},
	}},
	{},
}

// fuzzBatchOf builds three rows of a schema, which is a seed rather than a test
// of anything. The fuzzer needs one message per schema that is well formed all
// the way through, since a corpus of nothing but rejects never reaches the code
// that reads a column.
func fuzzBatchOf(f *testing.F, s dtype.Schema) ipc.Batch {
	f.Helper()
	b := ipc.Batch{Length: 3}
	if len(s.Fields) == 0 {
		return b
	}
	for _, field := range s.Fields {
		switch field.Type.Kind() {
		case dtype.Int64Kind:
			b.Columns = append(b.Columns, array.Of[int64](1, 2, 3))
		case dtype.StringKind:
			b.Columns = append(b.Columns, array.OfStrings("a", "bb", "ccc"))
		case dtype.BoolKind:
			b.Columns = append(b.Columns, array.OfBools(true, false, true))
		case dtype.NullKind:
			b.Columns = append(b.Columns, array.NewNull(3))
		default:
			builder, err := array.NewBuilder(field.Type)
			if err != nil {
				f.Fatalf("NewBuilder(%s) = %v", field.Type, err)
			}
			for i := range b.Length {
				builder.AppendBytes([]byte{byte(i)})
			}
			b.Columns = append(b.Columns, builder.Finish())
		}
	}
	return b
}

// The format strings FuzzImport picks from, one per layout it has to handle
// plus the two kinds of refusal, since a nested or unknown format arriving with
// plausible buffers is as much a case as a real one.
var fuzzFormats = []string{
	"n", "b", "c", "s", "i", "l", "g", "w:3", "tsu:UTC", "d:18,2",
	"u", "U", "z", "Z", "vu", "vz",
	"+l", "+s", "e", "nonsense",
}

// FuzzStream reads arbitrary bytes as a stream and checks that whatever comes
// out of it is a stream that goes back in.
//
// The framing between messages is the part being fuzzed. A message on its own
// is FuzzBatch's job, and the interesting inputs here are the ones between the
// messages: a length that runs past the end of the file, the old framing with
// no continuation, an end of stream marker in the middle, a file that stops
// after the schema.
//
// A reader is given a lying length on purpose, so what this is really checking
// is that a number in a file cannot make the reader allocate what it says
// rather than what arrived.
func FuzzStream(f *testing.F) {
	for _, s := range fuzzBatchSchemas {
		var buf bytes.Buffer
		w, err := ipc.NewWriter(&buf, s)
		if err != nil {
			f.Fatalf("NewWriter: %v", err)
		}
		if err := w.Write(fuzzBatchOf(f, s)); err != nil {
			f.Fatalf("Write: %v", err)
		}
		if err := w.Close(); err != nil {
			f.Fatalf("Close: %v", err)
		}
		f.Add(buf.Bytes())
	}
	f.Add([]byte(nil))
	f.Add([]byte{0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0})

	f.Fuzz(func(t *testing.T, stream []byte) {
		r, err := ipc.NewReader(bytes.NewReader(stream))
		if err != nil {
			return
		}

		var batches []ipc.Batch
		for b := range r.All() {
			if b.Length > 1<<20 {
				// The same batch of nothing FuzzBatch stops at. A null column
				// carries no bytes, so a row count of its own is free to write
				// and expensive to take seriously.
				return
			}
			batches = append(batches, b)
		}
		if r.Err() != nil {
			return
		}

		var buf bytes.Buffer
		w, err := ipc.NewWriter(&buf, r.Schema())
		if err != nil {
			t.Fatalf("a schema this package read is one it cannot write: %v", err)
		}
		for i, b := range batches {
			if err = w.Write(b); err != nil {
				t.Fatalf("batch %d read out of a stream is one Write refuses: %v", i, err)
			}
		}
		if err = w.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}

		again, err := ipc.NewReader(&buf)
		if err != nil {
			t.Fatalf("NewReader of a stream this package wrote: %v", err)
		}
		read := 0
		for b := range again.All() {
			if read >= len(batches) {
				t.Fatalf("the stream came back with more than the %d batches written", len(batches))
			}
			want := batches[read]
			if b.Length != want.Length {
				t.Fatalf("batch %d came back with %d rows, want %d", read, b.Length, want.Length)
			}
			for i, col := range want.Columns {
				equalArrays(t, b.Columns[i], col)
			}
			read++
		}
		if err := again.Err(); err != nil {
			t.Fatalf("Err on a stream this package wrote: %v", err)
		}
		if read != len(batches) {
			t.Fatalf("the stream came back with %d batches, want %d", read, len(batches))
		}
	})
}
