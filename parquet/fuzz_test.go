package parquet_test

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/tamnd/kuma/parquet"
)

// FuzzReadPages walks arbitrary bytes as the pages of a column chunk.
//
// A chunk is a run of pages with nothing in front of it saying how many, so the
// walk is driven entirely by the sizes in the headers it is reading. That is a
// tighter loop than the footer decoder is in: one size read wrong and the next
// header is read out of the middle of somebody's values, which is a header of
// arbitrary bytes and is exactly what this generates.
//
// What is asked for is that the walk stays inside the chunk. A page may be
// refused, and most of what is generated here will be, but a page that comes
// back has to hold the bytes its header said it would and cannot hold more
// bytes than the chunk has.
func FuzzReadPages(f *testing.F) {
	for _, name := range []string{"pages.parquet", "alltypes.parquet", "chunks.parquet"} {
		b, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			f.Fatalf("read: %v", err)
		}
		m, err := parquet.ReadMetadata(bytes.NewReader(b), int64(len(b)))
		if err != nil {
			f.Fatalf("%s: %v", name, err)
		}
		for _, g := range m.RowGroups {
			for i := range g.Columns {
				at := g.Columns[i].Start()
				f.Add(b[at : at+g.Columns[i].Meta.TotalCompressedSize])
			}
		}
	}

	f.Add([]byte(nil))
	f.Add([]byte{0x15})
	f.Add([]byte("PAR1"))

	f.Fuzz(func(t *testing.T, chunk []byte) {
		n := int64(len(chunk))
		c := &parquet.ColumnChunk{Meta: parquet.ColumnMeta{
			Path:                []string{"x"},
			TotalCompressedSize: n,
		}}

		pages, err := parquet.ReadPages(bytes.NewReader(chunk), n, c)
		if err != nil {
			t.Fatalf("a chunk of %d bytes that is the whole of a file of %d: %v", n, n, err)
		}

		read := 0
		for {
			p, err := pages.Next()
			if err != nil {
				break
			}

			// A page holds what its header said it holds, and its header said
			// something that fits in the chunk. The second is the one a bad
			// file attacks, since the size is what the walk adds to its
			// position to find the next header.
			if len(p.Data) != int(p.CompressedSize) {
				t.Fatalf("a page of %d bytes holds %d", p.CompressedSize, len(p.Data))
			}
			if int64(len(p.Data)) > n {
				t.Fatalf("a page of %d bytes came out of a chunk of %d", len(p.Data), n)
			}

			// The smallest page in the format is a header of one byte and a
			// body of none, so a chunk cannot hold more pages than it has
			// bytes.
			if read++; int64(read) > n {
				t.Fatalf("a chunk of %d bytes read as %d pages", n, read)
			}
		}
	})
}

// FuzzReadMetadata reads arbitrary bytes as a parquet file.
//
// The footer is the part being fuzzed. It is Thrift, which is lengths and
// counts all the way down, and every one of them is a number somebody else
// wrote saying how much to read or how much to allocate. The reader is written
// by hand precisely so that each of those is checked against what is left of
// the buffer, and this is what says so.
//
// What is being asked for is the reader's own promise rather than a well formed
// file. Nothing here is validated against the parquet specification, since a
// footer is decoded first and made sense of afterwards, and a schema that says
// it has four children when it has none is a file to refuse in the schema
// conversion rather than bytes to refuse in the decoder. The promise is
// narrower and is what a hostile file attacks: no read past the end of the
// footer, and nothing allocated by a number in the file that the bytes of the
// file do not pay for.
func FuzzReadMetadata(f *testing.F) {
	for _, name := range []string{
		"alltypes.parquet", "chunks.parquet", "nested.parquet", "empty.parquet",
	} {
		b, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			f.Fatalf("read: %v", err)
		}
		f.Add(b)
	}

	// The smallest thing with the shape of a file, which is where a mutation
	// that is going to reach the decoder at all starts from.
	var empty []byte
	empty = append(empty, "PAR1"...)
	empty = binary.LittleEndian.AppendUint32(empty, 0)
	f.Add(append(empty, "PAR1"...))

	f.Add([]byte(nil))
	f.Add([]byte("PAR1PAR1"))
	f.Add([]byte("PAR1\xff\xff\xff\xffPAR1"))
	f.Add([]byte("PAR1\x04\x00\x00\x00PARE"))

	f.Fuzz(func(t *testing.T, file []byte) {
		m, err := parquet.ReadMetadata(bytes.NewReader(file), int64(len(file)))
		if err != nil {
			return
		}

		// Every list in the footer is bounded by the bytes of the footer, since
		// the shortest element of anything is one byte. This is the allocation
		// bound the reader promises, and a file of thirty bytes that gets a
		// million schema nodes allocated has got past it.
		n := len(file)
		if len(m.Nodes) > n || len(m.RowGroups) > n || len(m.KeyValue) > n {
			t.Fatalf("a file of %d bytes read as %d schema nodes, %d row groups and %d metadata entries",
				n, len(m.Nodes), len(m.RowGroups), len(m.KeyValue))
		}

		// Every string and every byte slice was read out of the footer, so none
		// of them is longer than the file they came out of. A length in a file
		// that is believed rather than checked shows up here.
		within := func(what string, size int) {
			if size > n {
				t.Fatalf("%s is %d bytes long in a file of %d", what, size, n)
			}
		}
		within("the writer", len(m.CreatedBy))
		for _, e := range m.Nodes {
			within("a column name", len(e.Name))
		}
		for _, kv := range m.KeyValue {
			within("a metadata key", len(kv.Key))
			within("a metadata value", len(kv.Value))
		}
		for _, g := range m.RowGroups {
			within("a row group", len(g.Columns))
			for _, c := range g.Columns {
				within("a column path", len(c.Meta.Path))
				within("a file path", len(c.FilePath))
				within("an encoding list", len(c.Meta.Encodings))
				within("a lower bound", len(c.Meta.Stats.MinValue))
				within("an upper bound", len(c.Meta.Stats.MaxValue))
				within("an old lower bound", len(c.Meta.Stats.Min))
				within("an old upper bound", len(c.Meta.Stats.Max))
				for _, p := range c.Meta.Path {
					within("a path element", len(p))
				}
			}
		}

		// The schema conversion is the other half of reading a footer and it is
		// the half that believes the child counts, so it is fuzzed along with
		// it. Either it refuses the schema or it hands one back, and a tree
		// built out of a list cannot have more nodes in it than the list had.
		s, err := m.Schema()
		if err == nil {
			within("the schema", len(s.Fields))
		}
		columns, err := m.Columns()
		if err == nil {
			within("the columns", len(columns))
			for _, c := range columns {
				within("a column path", len(c.Path))
			}
		}

		// Reading the same bytes twice gives the same thing, which is not
		// interesting on its own and is the cheapest way to catch a reader that
		// kept something between calls.
		again, err := parquet.ReadMetadata(bytes.NewReader(file), int64(len(file)))
		if err != nil {
			t.Fatalf("the same file read a second time: %v", err)
		}
		if again.NumRows != m.NumRows || len(again.Nodes) != len(m.Nodes) {
			t.Fatalf("the same file read as %d rows and %d schema nodes, then as %d and %d",
				m.NumRows, len(m.Nodes), again.NumRows, len(again.Nodes))
		}
	})
}

// FuzzRLE decodes arbitrary bytes as a run of packed values.
//
// This is the one thing in the format that is read a bit at a time, and the
// bits say how many more of them to read. A run header holds a count that the
// decoder turns into a position, so a count read wrong is a read somewhere else
// in the buffer, and there is no checksum or magic anywhere near it to say so.
//
// What is asked for is that the decoder stays inside its bytes and inside the
// width it was given. A value wider than the width would be a value read out of
// the bits of the value next to it, which is the shape a bad shift takes.
func FuzzRLE(f *testing.F) {
	f.Add([]byte{0x03, 0x88, 0xc6, 0xfa}, uint8(3))
	f.Add([]byte{0x10, 0x01}, uint8(3))
	f.Add([]byte{0x05, 0x39, 0x77}, uint8(3))
	f.Add([]byte(nil), uint8(0))
	f.Add([]byte{0x80}, uint8(1))
	f.Add(bytes.Repeat([]byte{0xff}, 10), uint8(32))

	f.Fuzz(func(t *testing.T, data []byte, bits uint8) {
		width := int(bits)
		d, err := parquet.NewRLEDecoder(data, width)
		if err != nil {
			if width <= 32 {
				t.Fatalf("values of %d bits: %v", width, err)
			}
			return
		}

		// A run counts its values rather than holding them, so a handful of
		// bytes can honestly say there are two billion zeros behind them and
		// there is nothing wrong with the file that says so. This reads a
		// bounded number of them, since the fuzzer is looking for a decoder
		// that reads the wrong bytes rather than one that reads a lot of them.
		mask := uint64(1)<<uint(width) - 1
		buf := make([]int32, 97)
		var n int
		for read := 0; read < 1<<14; {
			n, err = d.Read(buf)
			for _, v := range buf[:n] {
				if uint64(uint32(v)) > mask {
					t.Fatalf("a value of %d out of %d bit values", v, width)
				}
			}
			read += n
			if err != nil {
				break
			}
		}

		// The deprecated encoding reads the same bytes the other way up. It
		// holds as many values as the bytes have room for, which is the one
		// thing it says about itself and the one thing to check.
		p, err := parquet.NewBitPackedDecoder(data, width)
		if err != nil {
			t.Fatalf("packed values of %d bits: %v", width, err)
		}

		want := 0
		if width > 0 {
			want = len(data) * 8 / width
		}
		got := 0
		for {
			n, err = p.Read(buf)
			for _, v := range buf[:n] {
				if uint64(uint32(v)) > mask {
					t.Fatalf("a packed value of %d out of %d bit values", v, width)
				}
			}
			got += n
			if err != nil {
				break
			}
		}
		if got != want {
			t.Fatalf("%d bytes of %d bit values read as %d values, want %d",
				len(data), width, got, want)
		}
	})
}

// FuzzPlain reads arbitrary bytes as plain values of every type.
//
// The plain encoding has almost nothing in it to get wrong, which is the point
// of this: the one length it does read is the one in front of a byte array, and
// that length is somebody else's number and is the only thing standing between
// a page and a read off the end of it. The rest of what is asked for here is
// that a page of n bytes read as values of m bytes is n over m values and not
// one more, since a decoder that rounded that up would hand back a value made
// half of the page and half of whatever came after it.
func FuzzPlain(f *testing.F) {
	f.Add([]byte{0x01, 0x00, 0x00, 0x00})
	f.Add([]byte{0x03, 0x00, 0x00, 0x00, 'o', 'n', 'e'})
	f.Add(bytes.Repeat([]byte{0xff}, 12))
	f.Add([]byte(nil))
	f.Add([]byte{0x05})

	f.Fuzz(func(t *testing.T, data []byte) {
		var d parquet.PlainDecoder

		// Every fixed width type reads whole values until there are fewer bytes
		// left than one takes, so how many come back is the size of the data
		// divided by the size of a value, whatever the bytes say.
		fixed := []struct {
			name string
			size int
			read func() (int, error)
		}{
			{name: "int32", size: 4, read: func() (int, error) {
				return d.Int32(make([]int32, len(data)/4+1))
			}},
			{name: "int64", size: 8, read: func() (int, error) {
				return d.Int64(make([]int64, len(data)/8+1))
			}},
			{name: "float", size: 4, read: func() (int, error) {
				return d.Float(make([]float32, len(data)/4+1))
			}},
			{name: "double", size: 8, read: func() (int, error) {
				return d.Double(make([]float64, len(data)/8+1))
			}},
			{name: "fixed", size: 3, read: func() (int, error) {
				return d.Fixed(make([][]byte, len(data)/3+1), 3)
			}},
		}
		for _, c := range fixed {
			d.Reset(data)
			n, err := c.read()
			if want := len(data) / c.size; n != want {
				t.Fatalf("%d bytes read as %d values of %d bytes, want %d: %v",
					len(data), n, c.size, want, err)
			}
		}

		// The deprecated timestamp is the one fixed width type that can refuse a
		// value it read cleanly, since twelve bytes hold days an int64 of
		// nanoseconds does not reach.
		d.Reset(data)
		if n, _ := d.Int96(make([]int64, len(data)/12+1)); n > len(data)/12 {
			t.Fatalf("%d bytes read as %d timestamps of twelve bytes", len(data), n)
		}

		// Booleans are one bit each with no padding until the end of the page.
		d.Reset(data)
		if n, _ := d.Boolean(make([]bool, len(data)*8+1)); n != len(data)*8 {
			t.Fatalf("%d bytes read as %d booleans, want %d", len(data), n, len(data)*8)
		}

		// A byte array is the only value here whose length is in the data, so
		// what is asked is that the values fit in the page they came out of.
		d.Reset(data)
		values := make([][]byte, len(data)/4+1)
		n, _ := d.ByteArray(values)
		used := 0
		for _, v := range values[:n] {
			used += 4 + len(v)
		}
		if used > len(data) {
			t.Fatalf("%d values out of %d bytes take %d bytes", n, len(data), used)
		}
		if left := d.Left(); used+left != len(data) {
			t.Fatalf("%d bytes read and %d left out of %d", used, left, len(data))
		}
	})
}
