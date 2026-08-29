package parquet_test

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/parquet"
)

// files is every file in testdata, which is what the round trip runs over. They
// were written by pyarrow with different options on purpose, so between them
// they cover a schema with every logical type in it, chunks with statistics and
// chunks without, a page index, a bloom filter, dictionaries, three codecs and a
// file with no rows at all.
func files(t *testing.T) []string {
	t.Helper()

	entries, err := os.ReadDir("testdata")
	if err != nil {
		t.Fatalf("read testdata: %v", err)
	}

	var out []string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".parquet") {
			out = append(out, e.Name())
		}
	}
	if len(out) < 10 {
		t.Fatalf("testdata holds %d parquet files, which is fewer than it should", len(out))
	}
	return out
}

// footer writes a footer and reads it back, which is the round trip the tests
// below are made of. What comes back has to equal what went in, and the way to
// check that is to read it with the same reader that reads a real file rather
// than with anything the test knows about the bytes.
func footer(t *testing.T, m *parquet.Metadata) *parquet.Metadata {
	t.Helper()

	var buf bytes.Buffer
	buf.WriteString("PAR1")

	n, err := parquet.WriteMetadata(&buf, m)
	if err != nil {
		t.Fatalf("WriteMetadata: %v", err)
	}
	if want := int64(buf.Len() - 4); n != want {
		t.Errorf("WriteMetadata wrote %d bytes and says %d", want, n)
	}

	b := buf.Bytes()
	got, err := parquet.ReadMetadata(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	return got
}

// TestWriteMetadataRoundTrip writes the footer of every file in testdata and
// reads it back. This is the test that says the field numbers are right: a
// number that is wrong here is a footer that reads back with the value in a
// different field, or in no field at all, and there is no way for that to pass.
func TestWriteMetadataRoundTrip(t *testing.T) {
	for _, name := range files(t) {
		t.Run(name, func(t *testing.T) {
			want := read(t, name)
			if got := footer(t, want); !reflect.DeepEqual(got, want) {
				t.Errorf("the footer came back different\n got %+v\nwant %+v", got, want)
			}
		})
	}
}

// TestWriteMetadataFile puts a written footer back on the file it came from and
// reads the whole thing. The round trip above says the structure survives, and
// this says the file does: every offset in a footer points into the pages in
// front of it, so a footer that came out a different length with the same
// offsets in it is still a file that opens and reads the right rows.
func TestWriteMetadataFile(t *testing.T) {
	for _, c := range []struct {
		name string

		// words is the column of strings, which is compared value for value,
		// or minus one for a file that has none.
		words int
	}{
		{"stats.parquet", 1},
		{"dictionary.parquet", 0},
		{"chunks.parquet", 0},
		{"pages.parquet", -1},
	} {
		t.Run(c.name, func(t *testing.T) {
			want := readTable(t, c.name, nil)

			b := bytesOf(t, c.name)
			var buf bytes.Buffer
			buf.Write(b[:footerStart(t, b)])
			if _, err := parquet.WriteMetadata(&buf, read(t, c.name)); err != nil {
				t.Fatalf("WriteMetadata: %v", err)
			}

			rebuilt := buf.Bytes()
			got, err := parquet.Read(bytes.NewReader(rebuilt), int64(len(rebuilt)), nil)
			if err != nil {
				t.Fatalf("Read: %v", err)
			}

			if got.NumRows() != want.NumRows() {
				t.Errorf("the rebuilt file holds %d rows, want %d", got.NumRows(), want.NumRows())
			}
			if got.Schema.String() != want.Schema.String() {
				t.Errorf("the rebuilt file is %s, want %s", got.Schema, want.Schema)
			}
			for i := range want.Columns {
				if g, w := got.Columns[i].String(), want.Columns[i].String(); g != w {
					t.Errorf("column %d came back as %s, want %s", i, g, w)
				}
			}
			if c.words >= 0 {
				if g, w := text(got.Columns[c.words]), text(want.Columns[c.words]); !slices.Equal(g, w) {
					t.Errorf("the strings came back as %v, want %v", g, w)
				}
			}
		})
	}
}

// footerStart is where the footer of a file begins, which is the eight bytes at
// the end read the way the format means them to be read.
func footerStart(t *testing.T, b []byte) int {
	t.Helper()

	if len(b) < 8 {
		t.Fatalf("a file of %d bytes has no footer", len(b))
	}
	n := binary.LittleEndian.Uint32(b[len(b)-8:])
	return len(b) - 8 - int(n)
}

// TestWriteMetadataLogicalTypes writes a schema holding every logical type the
// format has and reads it back.
//
// The files in testdata cover the types pyarrow writes, which is most of them
// and not all: nothing writes a BSON column and nothing has written an interval
// since before the logical types existed. The ones with parameters are the ones
// worth the trouble, since a decimal that comes back with its scale and its
// precision swapped is a column of numbers a thousand times too large.
func TestWriteMetadataLogicalTypes(t *testing.T) {
	kinds := []parquet.LogicalType{
		{Kind: parquet.NoLogical},
		{Kind: parquet.StringLogical},
		{Kind: parquet.MapLogical},
		{Kind: parquet.ListLogical},
		{Kind: parquet.EnumLogical},
		{Kind: parquet.DecimalLogical, Scale: 2, Precision: 9},
		{Kind: parquet.DateLogical},
		{Kind: parquet.TimeLogical, UTC: true, Unit: parquet.Millis},
		{Kind: parquet.TimeLogical, Unit: parquet.Micros},
		{Kind: parquet.TimestampLogical, UTC: true, Unit: parquet.Nanos},
		{Kind: parquet.TimestampLogical, Unit: parquet.NoUnit},
		{Kind: parquet.IntegerLogical, BitWidth: 8, Signed: true},
		{Kind: parquet.IntegerLogical, BitWidth: 64},
		{Kind: parquet.UnknownLogical},
		{Kind: parquet.JSONLogical},
		{Kind: parquet.BSONLogical},
		{Kind: parquet.UUIDLogical},
		{Kind: parquet.Float16Logical},
	}

	want := &parquet.Metadata{Version: 2, NumRows: 7}
	for i, l := range kinds {
		want.Nodes = append(want.Nodes, parquet.SchemaElement{
			Name:       "c" + string(rune('a'+i)),
			Type:       parquet.ByteArray,
			Repetition: parquet.Optional,
			Converted:  parquet.NoConverted,
			Logical:    l,
		})
	}

	got := footer(t, want)
	if len(got.Nodes) != len(want.Nodes) {
		t.Fatalf("the schema came back with %d nodes, want %d", len(got.Nodes), len(want.Nodes))
	}
	for i, node := range got.Nodes {
		if !reflect.DeepEqual(node, want.Nodes[i]) {
			t.Errorf("%s came back as %+v, want %+v", node.Name, node, want.Nodes[i])
		}
	}
}

// TestWriteMetadataAbsent writes the fields where absent and nought are
// different things and checks the difference survives.
//
// A statistic of nil is a writer that said nothing about the values and one of
// empty bytes is a writer that said the smallest of them is the empty string. A
// null count of nought with the flag set is a column with nothing missing and
// one without the flag is a column nobody counted. A footer that flattens either
// of those pairs is a footer that makes a reader skip a row group it should have
// read.
func TestWriteMetadataAbsent(t *testing.T) {
	for _, c := range []struct {
		name  string
		stats parquet.Statistics
	}{
		{"nothing said", parquet.Statistics{}},
		{"an empty bound", parquet.Statistics{MinValue: []byte{}, MaxValue: []byte("z")}},
		{"no nulls", parquet.Statistics{MinValue: []byte("a"), HasNullCount: true}},
		{"nulls uncounted", parquet.Statistics{MinValue: []byte("a")}},
		{"none distinct", parquet.Statistics{MinValue: []byte("a"), HasDistinct: true}},
		{"bounds that are values", parquet.Statistics{
			MinValue: []byte("a"), MaxValue: []byte("z"), MinExact: true, MaxExact: true,
		}},
		{"the older pair", parquet.Statistics{Min: []byte("a"), Max: []byte("z")}},
	} {
		t.Run(c.name, func(t *testing.T) {
			want := &parquet.Metadata{
				Version: 1,
				NumRows: 3,
				RowGroups: []parquet.RowGroup{{
					NumRows: 3,
					Columns: []parquet.ColumnChunk{{Meta: parquet.ColumnMeta{
						Type:      parquet.ByteArray,
						Encodings: []parquet.Encoding{parquet.Plain},
						Path:      []string{"word"},
						NumValues: 3,
						Stats:     c.stats,
					}}},
				}},
			}

			got := footer(t, want).RowGroups[0].Columns[0].Meta.Stats
			if !reflect.DeepEqual(got, c.stats) {
				t.Errorf("the statistics came back as %+v, want %+v", got, c.stats)
			}
		})
	}
}

// TestWriteMetadataRareFields writes the fields no file in testdata holds.
//
// A chunk in another file, a chunk that repeats where its own metadata is, an
// index page, an ordinal on a row group, a field id from a schema language: all
// of them are things the format defines and pyarrow does not write. A footer
// this package rewrites has to carry them anyway, since dropping a field on the
// way through is how a writer turns a file it was asked to copy into a different
// file.
func TestWriteMetadataRareFields(t *testing.T) {
	want := &parquet.Metadata{
		Version: 2,
		NumRows: 9,
		Nodes: []parquet.SchemaElement{
			{Name: "root", Type: parquet.NoType, Repetition: parquet.NoRepetition,
				Converted: parquet.NoConverted, NumChildren: 1},
			{Name: "id", Type: parquet.FixedLenByteArray, TypeLength: 16,
				Repetition: parquet.Required, Converted: parquet.NoConverted, FieldID: 42},
		},
		RowGroups: []parquet.RowGroup{{
			NumRows:             9,
			TotalByteSize:       400,
			TotalCompressedSize: 300,
			FileOffset:          4,
			Ordinal:             3,
			Columns: []parquet.ColumnChunk{{
				FilePath:          "part-1.parquet",
				FileOffset:        1200,
				ColumnIndexOffset: 900,
				ColumnIndexLength: 30,
				OffsetIndexOffset: 930,
				OffsetIndexLength: 20,
				Meta: parquet.ColumnMeta{
					Type:                  parquet.FixedLenByteArray,
					Encodings:             []parquet.Encoding{parquet.Plain, parquet.RLEDictionary},
					Path:                  []string{"id"},
					Codec:                 parquet.Snappy,
					NumValues:             9,
					TotalUncompressedSize: 400,
					TotalCompressedSize:   300,
					DataPageOffset:        100,
					DictionaryPageOffset:  4,
					IndexPageOffset:       80,
					BloomFilterOffset:     700,
					BloomFilterLength:     50,
				},
			}},
		}},
		KeyValue:  []parquet.KeyValue{{Key: "ARROW:schema", Value: "0000"}, {Key: "empty"}},
		CreatedBy: "kuma",
		Orders:    []parquet.ColumnOrder{parquet.TypeDefinedOrder},
	}

	if got := footer(t, want); !reflect.DeepEqual(got, want) {
		t.Errorf("the footer came back different\n got %+v\nwant %+v", got, want)
	}
}

// TestWriteMetadataEmpty writes a footer with nothing in it, which is what a
// file of no rows and no columns has. It is a legal parquet file and the reader
// reads one already, so the writer has to be able to produce one.
func TestWriteMetadataEmpty(t *testing.T) {
	// The schema and the row groups are the two lists the format calls
	// required, so they go out as lists of nothing and come back as lists of
	// nothing rather than as absent.
	got := footer(t, &parquet.Metadata{})
	if got.NumRows != 0 || len(got.Nodes) != 0 || len(got.RowGroups) != 0 {
		t.Errorf("an empty footer came back as %+v", got)
	}
}

// TestWriteMetadataOrders writes the column orders, which are a union with one
// member and an empty struct inside it.
//
// The order is what says whether the bounds on a chunk mean anything, so a file
// that loses it is a file whose statistics have to be ignored. An order this
// package did not recognize goes out as a union with nothing set, which is the
// only honest thing to write for something the reader could not keep.
func TestWriteMetadataOrders(t *testing.T) {
	want := &parquet.Metadata{
		Orders: []parquet.ColumnOrder{
			parquet.TypeDefinedOrder,
			parquet.UndefinedOrder,
			parquet.TypeDefinedOrder,
		},
	}

	if got := footer(t, want).Orders; !slices.Equal(got, want.Orders) {
		t.Errorf("the orders came back as %v, want %v", got, want.Orders)
	}
}

// TestWriteMetadataSize checks a written footer is not much larger than the one
// pyarrow wrote for the same file.
//
// It will not be the same bytes and it is not meant to be. Every field in the
// format is optional and two writers disagree about which of them are worth
// writing, so the test is that leaving a field out where it holds nothing keeps
// this in the same range as another writer rather than at twice the size, which
// is what a writer that wrote every field it had would come out at.
func TestWriteMetadataSize(t *testing.T) {
	for _, name := range files(t) {
		t.Run(name, func(t *testing.T) {
			b := bytesOf(t, name)
			want := len(b) - 8 - footerStart(t, b)

			var buf bytes.Buffer
			if _, err := parquet.WriteMetadata(&buf, read(t, name)); err != nil {
				t.Fatalf("WriteMetadata: %v", err)
			}

			if got := buf.Len() - 8; got > want*3/2 {
				t.Errorf("the footer is %d bytes where pyarrow wrote %d", got, want)
			}
		})
	}
}

// TestWriteMetadataError checks the error from the writer underneath comes back
// rather than being lost, since a file whose footer half went out is a file that
// nothing will open and a caller that was told nothing would not know.
func TestWriteMetadataError(t *testing.T) {
	if _, err := parquet.WriteMetadata(refuser{}, read(t, "stats.parquet")); err == nil {
		t.Error("writing to a broken writer returned no error")
	}
}

// refuser is an io.Writer that refuses everything.
type refuser struct{}

func (refuser) Write([]byte) (int, error) { return 0, os.ErrClosed }

func BenchmarkWriteMetadata(b *testing.B) {
	f, err := os.Open(filepath.Join("testdata", "stats.parquet"))
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		b.Fatal(err)
	}
	m, err := parquet.ReadMetadata(f, info.Size())
	if err != nil {
		b.Fatal(err)
	}

	var buf bytes.Buffer
	if _, err := parquet.WriteMetadata(&buf, m); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(buf.Len()))

	for b.Loop() {
		buf.Reset()
		if _, err := parquet.WriteMetadata(&buf, m); err != nil {
			b.Fatal(err)
		}
	}
}
