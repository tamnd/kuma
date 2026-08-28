package parquet_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/parquet"
)

// read opens one of the files in testdata and reads its footer.
//
// The files are written by pyarrow, by the script next to them, and they are
// checked in rather than written by the test because a reader has to be checked
// against files somebody else wrote and pyarrow is not installed on every
// platform this is tested on.
func read(t *testing.T, name string) *parquet.Metadata {
	t.Helper()

	f, err := os.Open(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		t.Fatalf("stat: %v", err)
	}

	m, err := parquet.ReadMetadata(f, info.Size())
	if err != nil {
		t.Fatalf("ReadMetadata(%s): %v", name, err)
	}
	return m
}

// bytesOf reads one of the files in testdata as bytes, for the tests that
// damage a file before reading it.
func bytesOf(t *testing.T, name string) []byte {
	t.Helper()

	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	return b
}

// TestReadMetadataSchema reads the schema of a file holding a column of every
// type, and checks it against what pyarrow says is in there. The physical type
// says how the values are written and the logical type says what they mean, and
// the pairs below are the whole reason both exist: a date and a count are both
// int32 and a decimal and a blob are both fixed length bytes.
func TestReadMetadataSchema(t *testing.T) {
	m := read(t, "alltypes.parquet")

	if m.NumRows != 3 {
		t.Errorf("the file holds %d rows, want 3", m.NumRows)
	}
	if len(m.RowGroups) != 1 {
		t.Fatalf("the file has %d row groups, want 1", len(m.RowGroups))
	}
	if m.CreatedBy == "" {
		t.Error("the file says nothing about what wrote it")
	}

	// The root of the schema, which is a group with no type of its own and one
	// child per column.
	if len(m.Schema) != 16 {
		t.Fatalf("the schema has %d nodes, want a root and 15 columns", len(m.Schema))
	}
	root := m.Schema[0]
	if root.Type != parquet.NoType {
		t.Errorf("the root has type %s, want none", root.Type)
	}
	if root.NumChildren != 15 {
		t.Errorf("the root has %d children, want 15", root.NumChildren)
	}

	tests := []struct {
		name       string
		typ        parquet.Type
		repetition parquet.Repetition
		logical    parquet.LogicalKind
	}{
		{"flag", parquet.Boolean, parquet.Required, parquet.NoLogical},
		{"small", parquet.Int32, parquet.Optional, parquet.IntegerLogical},
		{"count", parquet.Int32, parquet.Optional, parquet.NoLogical},
		{"total", parquet.Int64, parquet.Optional, parquet.NoLogical},
		{"unsigned", parquet.Int32, parquet.Optional, parquet.IntegerLogical},
		{"ratio", parquet.Float, parquet.Optional, parquet.NoLogical},
		{"weight", parquet.Double, parquet.Optional, parquet.NoLogical},
		{"name", parquet.ByteArray, parquet.Optional, parquet.StringLogical},
		{"blob", parquet.ByteArray, parquet.Optional, parquet.NoLogical},
		{"fixed", parquet.FixedLenByteArray, parquet.Optional, parquet.NoLogical},
		{"price", parquet.FixedLenByteArray, parquet.Optional, parquet.DecimalLogical},
		{"day", parquet.Int32, parquet.Optional, parquet.DateLogical},
		{"clock", parquet.Int64, parquet.Optional, parquet.TimeLogical},
		{"moment", parquet.Int64, parquet.Optional, parquet.TimestampLogical},
		{"local", parquet.Int64, parquet.Optional, parquet.TimestampLogical},
	}

	for i, want := range tests {
		got := m.Schema[i+1]
		if got.Name != want.name {
			t.Fatalf("column %d is %q, want %q", i, got.Name, want.name)
		}
		if got.Type != want.typ {
			t.Errorf("%s is a %s, want %s", want.name, got.Type, want.typ)
		}
		if got.Repetition != want.repetition {
			t.Errorf("%s is %s, want %s", want.name, got.Repetition, want.repetition)
		}
		if got.Logical.Kind != want.logical {
			t.Errorf("%s means %s, want %s", want.name, got.Logical.Kind, want.logical)
		}
		if got.NumChildren != 0 {
			t.Errorf("%s has %d children, want none", want.name, got.NumChildren)
		}
	}
}

// TestReadMetadataParameters checks the logical types that carry more than
// their own name. These are the ones a schema conversion cannot do without: a
// decimal is unreadable without its scale and a timestamp is unreadable without
// its unit.
func TestReadMetadataParameters(t *testing.T) {
	m := read(t, "alltypes.parquet")
	by := map[string]parquet.SchemaElement{}
	for _, e := range m.Schema {
		by[e.Name] = e
	}

	price := by["price"]
	if got := price.Logical; got.Scale != 2 || got.Precision != 9 {
		t.Errorf("price is a decimal of scale %d and precision %d, want 2 and 9", got.Scale, got.Precision)
	}
	if price.Scale != 2 || price.Precision != 9 {
		t.Errorf("the schema element says scale %d and precision %d, want 2 and 9",
			price.Scale, price.Precision)
	}
	if price.TypeLength != 4 {
		t.Errorf("price is %d bytes wide, want 4", price.TypeLength)
	}

	if got := by["small"].Logical; got.BitWidth != 8 || !got.Signed {
		t.Errorf("small is a %d bit integer, signed %v, want 8 and signed", got.BitWidth, got.Signed)
	}
	if got := by["unsigned"].Logical; got.BitWidth != 32 || got.Signed {
		t.Errorf("unsigned is a %d bit integer, signed %v, want 32 and unsigned", got.BitWidth, got.Signed)
	}

	if got := by["clock"].Logical; got.Unit != parquet.Micros || got.UTC {
		t.Errorf("clock is in %s, utc %v, want micros and not utc", got.Unit, got.UTC)
	}
	if got := by["moment"].Logical; got.Unit != parquet.Millis || !got.UTC {
		t.Errorf("moment is in %s, utc %v, want millis and utc", got.Unit, got.UTC)
	}
	if got := by["local"].Logical; got.Unit != parquet.Micros || got.UTC {
		t.Errorf("local is in %s, utc %v, want micros and not utc", got.Unit, got.UTC)
	}

	// The converted type is the same thing said the old way, and a writer that
	// wants to be read by everything says both.
	if got := by["name"].Converted; got != parquet.ConvertedUTF8 {
		t.Errorf("name converts as %s, want utf8", got)
	}
	if got := by["count"].Converted; got != parquet.NoConverted {
		t.Errorf("count converts as %s, want nothing", got)
	}
}

// TestReadMetadataArrowSchema checks the key and value metadata, which is where
// a file written from Arrow keeps the Arrow schema. It is how a type parquet
// cannot describe survives a round trip, and a reader that wants to give back
// what it was given has to know it is there.
func TestReadMetadataArrowSchema(t *testing.T) {
	m := read(t, "alltypes.parquet")

	found := false
	for _, kv := range m.KeyValue {
		if kv.Key == "ARROW:schema" {
			found = true
			if kv.Value == "" {
				t.Error("the arrow schema is there and is empty")
			}
		}
	}
	if !found {
		t.Errorf("the file carries %d metadata entries and none of them is the arrow schema",
			len(m.KeyValue))
	}
}

// TestReadMetadataRowGroups reads a file written in two row groups, which is
// the unit a scan skips. Every column has a chunk in every group, and the
// chunks say where their pages are and what is in them.
func TestReadMetadataRowGroups(t *testing.T) {
	m := read(t, "chunks.parquet")

	if m.NumRows != 6 {
		t.Errorf("the file holds %d rows, want 6", m.NumRows)
	}
	if len(m.RowGroups) != 2 {
		t.Fatalf("the file has %d row groups, want 2", len(m.RowGroups))
	}

	for i, g := range m.RowGroups {
		if g.NumRows != 3 {
			t.Errorf("group %d holds %d rows, want 3", i, g.NumRows)
		}
		if len(g.Columns) != 2 {
			t.Fatalf("group %d has %d columns, want 2", i, len(g.Columns))
		}
		if g.TotalByteSize <= 0 || g.TotalCompressedSize <= 0 {
			t.Errorf("group %d is %d bytes uncompressed and %d compressed, want both above zero",
				i, g.TotalByteSize, g.TotalCompressedSize)
		}

		code := g.Columns[0].Meta
		if len(code.Path) != 1 || code.Path[0] != "code" {
			t.Errorf("the first column of group %d is %v, want code", i, code.Path)
		}
		if code.Type != parquet.ByteArray {
			t.Errorf("code is a %s, want byte_array", code.Type)
		}
		if code.Codec != parquet.Snappy {
			t.Errorf("code is compressed with %s, want snappy", code.Codec)
		}
		if code.NumValues != 3 {
			t.Errorf("code holds %d values in group %d, want 3", code.NumValues, i)
		}
		if code.DictionaryPageOffset == 0 {
			t.Error("code has no dictionary page, want the one pyarrow writes")
		}
		if code.DataPageOffset <= code.DictionaryPageOffset {
			t.Errorf("code's data page is at %d and its dictionary at %d, want the dictionary first",
				code.DataPageOffset, code.DictionaryPageOffset)
		}
		if !slices.Contains(code.Encodings, parquet.RLEDictionary) {
			t.Errorf("code is encoded %v, want the dictionary encoding among them", code.Encodings)
		}
	}

	// The two groups hold different halves of the column, which is the whole
	// point of the statistics: a filter for FR reads the second group only.
	first, second := m.RowGroups[0].Columns[0].Meta.Stats, m.RowGroups[1].Columns[0].Meta.Stats
	if string(first.MinValue) != "GB" || string(first.MaxValue) != "US" {
		t.Errorf("group 0 holds %q to %q, want GB to US", first.MinValue, first.MaxValue)
	}
	if string(second.MinValue) != "DE" || string(second.MaxValue) != "GB" {
		t.Errorf("group 1 holds %q to %q, want DE to GB", second.MinValue, second.MaxValue)
	}
	if !first.HasNullCount || first.NullCount != 0 {
		t.Errorf("group 0 counts %d nulls, said %v, want none and said so", first.NullCount, first.HasNullCount)
	}
}

// TestReadMetadataStatistics checks the statistics of a file with missing
// values in it. The bounds are the raw bytes of the physical type, so an int32
// column's smallest value is four bytes and reading it means knowing that.
func TestReadMetadataStatistics(t *testing.T) {
	m := read(t, "alltypes.parquet")
	by := map[string]parquet.ColumnMeta{}
	for _, c := range m.RowGroups[0].Columns {
		by[c.Meta.Path[0]] = c.Meta
	}

	count := by["count"].Stats
	if !count.HasNullCount || count.NullCount != 1 {
		t.Errorf("count counts %d nulls, said %v, want one and said so", count.NullCount, count.HasNullCount)
	}
	if got := binary.LittleEndian.Uint32(count.MinValue); got != 10 {
		t.Errorf("count starts at %d, want 10", got)
	}
	if got := binary.LittleEndian.Uint32(count.MaxValue); got != 20 {
		t.Errorf("count ends at %d, want 20", got)
	}
	if !count.MinExact || !count.MaxExact {
		t.Error("count's bounds are not exact, want the exact bounds of a column of two values")
	}

	if name := by["name"].Stats; string(name.MinValue) != "GB" || string(name.MaxValue) != "JP" {
		t.Errorf("name runs from %q to %q, want GB to JP", name.MinValue, name.MaxValue)
	}
	if flag := by["flag"].Stats; flag.HasNullCount && flag.NullCount != 0 {
		t.Errorf("flag counts %d nulls, want none in a required column", flag.NullCount)
	}
}

// TestReadMetadataEmpty is a file with a schema and no rows, which still has a
// footer and still says what is in it.
func TestReadMetadataEmpty(t *testing.T) {
	m := read(t, "empty.parquet")

	if m.NumRows != 0 {
		t.Errorf("the file holds %d rows, want none", m.NumRows)
	}
	if len(m.Schema) != 3 {
		t.Fatalf("the schema has %d nodes, want a root and two columns", len(m.Schema))
	}
	if m.Schema[1].Name != "id" || m.Schema[2].Name != "label" {
		t.Errorf("the columns are %q and %q, want id and label", m.Schema[1].Name, m.Schema[2].Name)
	}
	for _, g := range m.RowGroups {
		if g.NumRows != 0 {
			t.Errorf("a row group holds %d rows, want none", g.NumRows)
		}
	}
}

// TestReadMetadataRefused covers the ways a file is not one. Every one of them
// has to say so rather than read something that is not there.
func TestReadMetadataRefused(t *testing.T) {
	good := bytesOf(t, "chunks.parquet")

	// A file whose footer says it is longer than the file is.
	long := bytes.Clone(good)
	binary.LittleEndian.PutUint32(long[len(long)-8:], uint32(len(long)))

	// A file whose footer length is right and whose footer is not a footer.
	damaged := bytes.Clone(good)
	at := len(damaged) - 8 - int(binary.LittleEndian.Uint32(damaged[len(damaged)-8:]))
	for i := at; i < at+16; i++ {
		damaged[i] = 0xff
	}

	// A file that says its footer is encrypted, which this package understands
	// and cannot read.
	encrypted := bytes.Clone(good)
	copy(encrypted[len(encrypted)-4:], "PARE")

	tests := []struct {
		name string
		file []byte
		want error
	}{
		{"empty", nil, parquet.ErrFormat},
		{"too small", []byte("PAR1"), parquet.ErrFormat},
		{"not parquet at all", bytes.Repeat([]byte("x"), 64), parquet.ErrFormat},
		{"no magic at the front", append([]byte("XXXX"), good[4:]...), parquet.ErrFormat},
		{"no magic at the end", append(bytes.Clone(good[:len(good)-4]), "XXXX"...), parquet.ErrFormat},
		{"a footer longer than the file", long, parquet.ErrFormat},
		{"a damaged footer", damaged, parquet.ErrFormat},
		{"an encrypted footer", encrypted, parquet.ErrUnsupported},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parquet.ReadMetadata(bytes.NewReader(tt.file), int64(len(tt.file)))
			if !errors.Is(err, tt.want) {
				t.Fatalf("the error is %v, want %v", err, tt.want)
			}
		})
	}
}

// TestReadMetadataTruncated cuts a real file short at every length there is and
// checks that reading it says so rather than reading past the end of it. A
// footer is somebody else's bytes and half of one is the shape they arrive in
// when a copy went wrong.
func TestReadMetadataTruncated(t *testing.T) {
	good := bytesOf(t, "chunks.parquet")

	for n := range len(good) {
		// The size is the truthful size of what is being read, which is what
		// makes this a truncated file rather than a lie about a whole one.
		if _, err := parquet.ReadMetadata(bytes.NewReader(good[:n]), int64(n)); err == nil {
			t.Fatalf("the first %d bytes of a %d byte file read as a whole one", n, len(good))
		}
	}
}

// failing is a file that stops answering after so many reads, which is what a
// network filesystem gives back in the middle of an outage. ReadMetadata makes
// three reads and every one of them has to hand the error back rather than
// carry on with whatever was in the buffer.
type failing struct {
	at *os.File
	ok int
}

var errGone = errors.New("the file went away")

func (f *failing) ReadAt(p []byte, off int64) (int, error) {
	if f.ok <= 0 {
		return 0, errGone
	}
	f.ok--
	return f.at.ReadAt(p, off)
}

func TestReadMetadataUnreadable(t *testing.T) {
	path := filepath.Join("testdata", "chunks.parquet")
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}

	for _, reads := range []int{0, 1, 2} {
		f, err := os.Open(path)
		if err != nil {
			t.Fatalf("open: %v", err)
		}

		_, err = parquet.ReadMetadata(&failing{at: f, ok: reads}, info.Size())
		f.Close()
		if !errors.Is(err, errGone) {
			t.Errorf("a file that stops answering after %d reads = %v, want the read error",
				reads, err)
		}
	}
}

// TestReadMetadataShortSize is the other half of that: the file is whole and
// the size it is read with is not. A size that is too small cuts the footer off
// and a size that is too large reads past the end of it, and neither may crash.
func TestReadMetadataShortSize(t *testing.T) {
	good := bytesOf(t, "chunks.parquet")

	for _, size := range []int64{-1, 0, 1, 11, int64(len(good)) - 1, int64(len(good)) + 1, 1 << 20} {
		if _, err := parquet.ReadMetadata(bytes.NewReader(good), size); err == nil {
			t.Errorf("a %d byte file read as %d bytes without complaint", len(good), size)
		}
	}
}

// BenchmarkReadMetadata reads a footer out of memory, so what is measured is
// the decoding rather than the three reads in front of it.
//
// This is the cost a scan pays before it has looked at a single value, once per
// file, and a dataset is thousands of files. The footer of a real file is
// bigger than these because it carries a column chunk per column per row group,
// which is why the two shapes here are one row group and two.
func BenchmarkReadMetadata(b *testing.B) {
	for _, name := range []string{"alltypes.parquet", "chunks.parquet"} {
		file, err := os.ReadFile(filepath.Join("testdata", name))
		if err != nil {
			b.Fatalf("read: %v", err)
		}
		r, size := bytes.NewReader(file), int64(len(file))

		b.Run(strings.TrimSuffix(name, ".parquet"), func(b *testing.B) {
			b.SetBytes(size)
			for b.Loop() {
				if _, err := parquet.ReadMetadata(r, size); err != nil {
					b.Fatalf("ReadMetadata: %v", err)
				}
			}
		})
	}
}
