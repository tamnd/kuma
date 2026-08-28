package parquet

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The pages of the files in testdata, walked, plus the pages no writer produces.
//
// A page header is read the same way the footer is, so the interesting inputs
// are the same ones: a length that does not fit in what is left, a count that
// contradicts another count, a struct that is not there at all. Those are built
// by hand with the same builder the Thrift tests use, and the pages that a real
// writer writes come out of the files.

// i32 writes an integer field. Every count and every enumeration in a page
// header is one of these, so it is most of what the pages below are made of.
func (w *builder) i32(id int16, v int32) *builder {
	return w.field(id, thriftInt32).varint(int64(v))
}

// pageOf writes a page header with whatever fields f writes, and the body
// behind it. That is the shape of a page: a header and then bytes, with nothing in
// between saying how long the header was.
func pageOf(body string, f func(*builder)) []byte {
	w := &builder{}
	w.structure(func() { f(w) })
	return append(w.b, body...)
}

// openFile reads a file in testdata and its footer.
//
// The whole file is held in memory because a chunk is read out of it by offset,
// which is what ReadPages wants and what a real caller has in an mmap or a
// range request.
func openFile(t testing.TB, name string) (*bytes.Reader, int64, *Metadata) {
	t.Helper()

	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("reading %s: %v", name, err)
	}
	m, err := ReadMetadata(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatalf("reading the footer of %s: %v", name, err)
	}
	return bytes.NewReader(b), int64(len(b)), m
}

// walk reads a column chunk to the end and returns every page of it.
func walk(t *testing.T, r io.ReaderAt, size int64, c *ColumnChunk) []Page {
	t.Helper()

	name := strings.Join(c.Meta.Path, ".")
	p, err := ReadPages(r, size, c)
	if err != nil {
		t.Fatalf("the chunk for %s: %v", name, err)
	}

	var pages []Page
	for {
		page, err := p.Next()
		if errors.Is(err, io.EOF) {
			return pages
		}
		if err != nil {
			t.Fatalf("page %d of %s: %v", len(pages), name, err)
		}
		pages = append(pages, page)
	}
}

// shape is what a page should say, which is its header without the statistics.
type shape struct {
	kind                     PageKind
	values, nulls, rows      int32
	encoding                 Encoding
	compressed, uncompressed int32
	definition, repetition   int32

	// crc is whether the writer wrote a checksum, and viaCodec is whether the
	// page says its body went through the compression of the chunk. Only the
	// second version of the data page can say no to that, and pyarrow says it
	// on every page of a file written with no compression at all.
	crc, viaCodec bool
}

// check compares a page against what it should be.
//
// The level encodings are not in the table because every page in every file
// here has RLE levels, which is what the second version of the data page fixes
// them at and what every writer picks for the first. A dictionary page has no
// levels and says nothing about them.
func (s shape) check(t *testing.T, at string, p Page) {
	t.Helper()

	got := shape{
		kind: p.Kind, values: p.NumValues, nulls: p.NumNulls, rows: p.NumRows,
		encoding: p.Encoding, compressed: p.CompressedSize, uncompressed: p.UncompressedSize,
		definition: p.DefinitionLength, repetition: p.RepetitionLength,
		crc: p.HasCRC, viaCodec: p.Compressed,
	}
	if got != s {
		t.Errorf("%s is %+v, want %+v", at, got, s)
	}
	if len(p.Data) != int(p.CompressedSize) {
		t.Errorf("%s says %d bytes and holds %d", at, p.CompressedSize, len(p.Data))
	}

	definition, repetition := RLE, RLE
	if p.Kind == DictionaryPage {
		definition, repetition = NoEncoding, NoEncoding
	}
	if p.DefinitionEncoding != definition || p.RepetitionEncoding != repetition {
		t.Errorf("%s has %s definition levels and %s repetition levels, want %s and %s",
			at, p.DefinitionEncoding, p.RepetitionEncoding, definition, repetition)
	}
}

// TestPages walks a file written the second way, with several pages per column,
// a dictionary in front of one of them, and a checksum on every page.
func TestPages(t *testing.T) {
	r, size, m := openFile(t, "pages.parquet")

	tests := []struct {
		column string
		pages  []shape
	}{
		// A plain column of 500 values, written in two pages because the
		// writer was told to flush every hundred rows and to keep a page under
		// a kilobyte. Three bytes of definition levels say all 500 are there,
		// which is what one RLE run of one repeated level costs.
		{"n", []shape{
			{kind: DataPageV2, values: 300, rows: 300, encoding: Plain,
				compressed: 1203, uncompressed: 1203, definition: 3, crc: true},
			{kind: DataPageV2, values: 200, rows: 200, encoding: Plain,
				compressed: 803, uncompressed: 803, definition: 3, crc: true},
		}},

		// Four distinct strings, so the writer wrote them once into a
		// dictionary page and then wrote 500 indices into it. The dictionary
		// page is plain rather than plain_dictionary, which is what the format
		// asks for now and what the older name meant all along.
		{"word", []shape{
			{kind: DictionaryPage, values: 4, encoding: Plain,
				compressed: 35, uncompressed: 35, crc: true, viaCodec: true},
			{kind: DataPageV2, values: 500, rows: 500, encoding: RLEDictionary,
				compressed: 131, uncompressed: 131, definition: 3, crc: true},
		}},

		// A third of the values are null. The null count is the reason to
		// prefer the second version of the data page: it is written down here
		// rather than worked out by decoding the levels.
		{"maybe", []shape{
			{kind: DataPageV2, values: 400, nulls: 134, rows: 400, encoding: Plain,
				compressed: 1115, uncompressed: 1115, definition: 51, crc: true},
			{kind: DataPageV2, values: 100, nulls: 33, rows: 100, encoding: Plain,
				compressed: 282, uncompressed: 282, definition: 14, crc: true},
		}},
	}

	for i, tt := range tests {
		c := &m.RowGroups[0].Columns[i]
		if name := strings.Join(c.Meta.Path, "."); name != tt.column {
			t.Fatalf("column %d is %s, want %s", i, name, tt.column)
		}

		pages := walk(t, r, size, c)
		if len(pages) != len(tt.pages) {
			t.Errorf("%s holds %d pages, want %d", tt.column, len(pages), len(tt.pages))
			continue
		}
		for j, want := range tt.pages {
			want.check(t, fmt.Sprintf("%s page %d", tt.column, j), pages[j])
		}
	}
}

// TestPagesV1 walks a file written the first way, which is what pyarrow writes
// unless it is told otherwise and therefore what most files in the world are.
//
// Every column here is dictionary encoded except the boolean one, which has too
// few distinct values to be worth a dictionary. None of them carry a checksum,
// which is also the default.
func TestPagesV1(t *testing.T) {
	r, size, m := openFile(t, "alltypes.parquet")

	// How many entries each column's dictionary holds. Three rows of values,
	// so a column with a null in it has two, and a column of the same value
	// three times has one.
	entries := []struct {
		column string
		values int32
	}{
		{"flag", 0}, {"small", 2}, {"count", 2}, {"total", 3}, {"unsigned", 3},
		{"ratio", 2}, {"weight", 3}, {"name", 2}, {"blob", 3}, {"fixed", 3},
		{"price", 2}, {"day", 1}, {"clock", 1}, {"moment", 1}, {"local", 1},
	}

	for i, want := range entries {
		c := &m.RowGroups[0].Columns[i]
		if name := strings.Join(c.Meta.Path, "."); name != want.column {
			t.Fatalf("column %d is %s, want %s", i, name, want.column)
		}

		pages := walk(t, r, size, c)
		data := pages[len(pages)-1]
		if want.values == 0 {
			if len(pages) != 1 {
				t.Errorf("%s holds %d pages, want the one data page", want.column, len(pages))
				continue
			}
			if data.Encoding != Plain {
				t.Errorf("%s is encoded as %s, want plain", want.column, data.Encoding)
			}
		} else {
			if len(pages) != 2 {
				t.Errorf("%s holds %d pages, want a dictionary and a data page",
					want.column, len(pages))
				continue
			}
			dictionary := pages[0]
			if dictionary.Kind != DictionaryPage || dictionary.NumValues != want.values {
				t.Errorf("%s begins with a %s of %d values, want a dictionary of %d",
					want.column, dictionary.Kind, dictionary.NumValues, want.values)
			}
			if data.Encoding != RLEDictionary {
				t.Errorf("%s is encoded as %s, want rle_dictionary", want.column, data.Encoding)
			}
		}

		// The first version of the data page is what the levels and the values
		// share a compressed run of bytes in, so it has no lengths for them
		// and nothing to say about being compressed.
		if data.Kind != DataPage || data.NumValues != 3 {
			t.Errorf("%s ends with a %s of %d values, want a data page of 3",
				want.column, data.Kind, data.NumValues)
		}
		if data.DefinitionLength != 0 || data.RepetitionLength != 0 || data.NumRows != 0 {
			t.Errorf("%s has level lengths %d and %d and %d rows, want nothing in any of them",
				want.column, data.RepetitionLength, data.DefinitionLength, data.NumRows)
		}
		for _, p := range pages {
			if p.HasCRC {
				t.Errorf("%s carries a checksum, and this file was written without them",
					want.column)
			}
			if !p.Compressed {
				t.Errorf("%s says its body did not go through the codec, which only the "+
					"second version of the data page may say", want.column)
			}
		}
	}
}

// TestPagesCompressed walks the snappy file, where the sizes in the header are
// two different numbers.
//
// They are not always the smaller one first. Snappy on twenty bytes of strings
// makes them longer, and a writer that has already written the page keeps it,
// so a reader that assumes the compressed size is the smaller of the two is
// wrong on the first page of this file.
func TestPagesCompressed(t *testing.T) {
	r, size, m := openFile(t, "chunks.parquet")

	if codec := m.RowGroups[0].Columns[0].Meta.Codec; codec != Snappy {
		t.Fatalf("the file is %s compressed, and these numbers are snappy's", codec)
	}

	want := [][]shape{
		{
			{kind: DictionaryPage, values: 3, encoding: Plain,
				compressed: 20, uncompressed: 18, viaCodec: true},
			{kind: DataPage, values: 3, encoding: RLEDictionary,
				compressed: 12, uncompressed: 10, viaCodec: true},
		},
		{
			{kind: DictionaryPage, values: 3, encoding: Plain,
				compressed: 22, uncompressed: 24, viaCodec: true},
			{kind: DataPage, values: 3, encoding: RLEDictionary,
				compressed: 12, uncompressed: 10, viaCodec: true},
		},
	}

	for i := range want {
		pages := walk(t, r, size, &m.RowGroups[0].Columns[i])
		if len(pages) != len(want[i]) {
			t.Fatalf("column %d holds %d pages, want %d", i, len(pages), len(want[i]))
		}
		for j, w := range want[i] {
			w.check(t, fmt.Sprintf("column %d page %d", i, j), pages[j])
		}
	}
}

// TestPagesEmpty walks a file with no rows in it.
//
// Each column is one dictionary page of no entries and no bytes, and no data
// page at all, which is a chunk that a decoder has to cope with rather than a
// file to refuse. The page header is fourteen bytes and the body is nothing.
func TestPagesEmpty(t *testing.T) {
	r, size, m := openFile(t, "empty.parquet")

	for i := range m.RowGroups[0].Columns {
		c := &m.RowGroups[0].Columns[i]
		pages := walk(t, r, size, c)
		if len(pages) != 1 {
			t.Fatalf("%s holds %d pages, want the one empty dictionary",
				strings.Join(c.Meta.Path, "."), len(pages))
		}
		shape{kind: DictionaryPage, encoding: Plain, viaCodec: true}.
			check(t, strings.Join(c.Meta.Path, "."), pages[0])
	}
}

// TestPagesValues walks every chunk of every file and checks the walk against
// what the footer said was in it.
//
// This is the property that makes a page walk worth anything. The footer counts
// the values of a chunk and the pages count their own, and a walk that lost a
// page or read a length wrong disagrees with the footer even when every header
// it read parsed cleanly.
func TestPagesValues(t *testing.T) {
	files := []string{
		"alltypes.parquet", "chunks.parquet", "nested.parquet",
		"pages.parquet", "empty.parquet",
	}

	for _, name := range files {
		t.Run(name, func(t *testing.T) {
			r, size, m := openFile(t, name)
			for i := range m.RowGroups {
				for j := range m.RowGroups[i].Columns {
					c := &m.RowGroups[i].Columns[j]
					column := strings.Join(c.Meta.Path, ".")

					var values, rows int64
					for k, p := range walk(t, r, size, c) {
						if p.Kind == DictionaryPage {
							if k != 0 {
								t.Errorf("%s has a dictionary at page %d, and a "+
									"dictionary comes first or not at all", column, k)
							}
							continue
						}
						values += int64(p.NumValues)
						rows += int64(p.NumRows)
					}

					if values != c.Meta.NumValues {
						t.Errorf("the pages of %s hold %d values and the footer says %d",
							column, values, c.Meta.NumValues)
					}
					if rows != 0 && rows != m.RowGroups[i].NumRows {
						t.Errorf("the pages of %s cover %d rows and the row group has %d",
							column, rows, m.RowGroups[i].NumRows)
					}
				}
			}
		})
	}
}

// TestPageStats reads the statistics a page carries.
//
// Almost nothing writes them any more, since the column index says the same
// thing in one place instead of once per page, but a file that has them has
// them and dropping them on the floor would be a silent loss.
func TestPageStats(t *testing.T) {
	r, size, m := openFile(t, "pages.parquet")

	pages := walk(t, r, size, &m.RowGroups[0].Columns[1])
	s := pages[1].Stats
	if string(s.MinValue) != "alpha" || string(s.MaxValue) != "gamma" {
		t.Errorf("the data page of word runs from %q to %q, want alpha to gamma",
			s.MinValue, s.MaxValue)
	}
	if !s.HasNullCount || s.NullCount != 0 {
		t.Errorf("the data page of word says %d nulls and has a count: %v",
			s.NullCount, s.HasNullCount)
	}
}

// TestStart is where a chunk begins, which is the dictionary page when there is
// one and the first data page when there is not.
func TestStart(t *testing.T) {
	tests := []struct {
		name       string
		dictionary int64
		data       int64
		want       int64
	}{
		{"a column with a dictionary", 100, 140, 100},
		{"a column without one", 0, 140, 140},

		// A column of a file with no rows in it, which pyarrow writes as an
		// empty dictionary page and no data page.
		{"a column with no data pages", 100, 0, 100},

		// Some writers put the data page offset in both fields rather than
		// leaving the dictionary one out, and one or two have written a
		// dictionary offset that points past the data. Neither is a chunk that
		// starts anywhere other than at its first page.
		{"a dictionary offset copied from the data", 140, 140, 140},
		{"a dictionary offset behind the data", 200, 140, 140},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := ColumnChunk{Meta: ColumnMeta{
				DictionaryPageOffset: tt.dictionary,
				DataPageOffset:       tt.data,
			}}
			if got := c.Start(); got != tt.want {
				t.Errorf("the chunk starts at %d, want %d", got, tt.want)
			}
		})
	}
}

// chunkOf makes a chunk out of bytes, the way ReadPages would out of a file.
func chunkOf(t *testing.T, b []byte) *Pages {
	t.Helper()

	c := &ColumnChunk{Meta: ColumnMeta{
		Path:                []string{"x"},
		TotalCompressedSize: int64(len(b)),
	}}
	p, err := ReadPages(bytes.NewReader(b), int64(len(b)), c)
	if err != nil {
		t.Fatalf("reading a chunk of %d bytes: %v", len(b), err)
	}
	return p
}

// TestPagesRefused is the headers a writer does not write.
//
// Every one of them is a number that contradicts another number, or a page that
// left out something the format calls required. The walk runs on those numbers,
// so a header it accepts and cannot use is a walk that reads the rest of the
// chunk as garbage.
func TestPagesRefused(t *testing.T) {
	tests := []struct {
		name  string
		chunk []byte
	}{
		{"a header that stops in the middle", []byte{0x15}},
		{"a page of a negative size", pageOf("", func(w *builder) {
			w.i32(1, int32(DataPage)).i32(2, 4).i32(3, -4)
		})},
		{"a page that is bigger uncompressed than a number holds", pageOf("", func(w *builder) {
			w.i32(1, int32(DataPage)).i32(2, -1).i32(3, 0)
		})},
		{"a data page with no data page header", pageOf("", func(w *builder) {
			w.i32(1, int32(DataPage)).i32(2, 0).i32(3, 0)
		})},
		{"a data page that does not say how its levels are encoded",
			pageOf("", func(w *builder) {
				w.i32(1, int32(DataPage)).i32(2, 0).i32(3, 0)
				w.field(5, thriftStruct)
				w.structure(func() { w.i32(1, 3).i32(2, int32(Plain)) })
			})},
		{"a data page of a negative number of values", pageOf("", func(w *builder) {
			w.i32(1, int32(DataPage)).i32(2, 0).i32(3, 0)
			w.field(5, thriftStruct)
			w.structure(func() {
				w.i32(1, -3).i32(2, int32(Plain)).i32(3, int32(RLE)).i32(4, int32(RLE))
			})
		})},
		{"a data page that does not say how it is encoded", pageOf("", func(w *builder) {
			w.i32(1, int32(DataPage)).i32(2, 0).i32(3, 0)
			w.field(5, thriftStruct)
			w.structure(func() { w.i32(1, 3).i32(3, int32(RLE)).i32(4, int32(RLE)) })
		})},
		{"a dictionary page with no dictionary page header", pageOf("", func(w *builder) {
			w.i32(1, int32(DictionaryPage)).i32(2, 0).i32(3, 0)
		})},
		{"a second version data page with no header of its own", pageOf("", func(w *builder) {
			w.i32(1, int32(DataPageV2)).i32(2, 0).i32(3, 0)
		})},
		{"a page holding more nulls than values", pageOf("", func(w *builder) {
			w.i32(1, int32(DataPageV2)).i32(2, 0).i32(3, 0)
			w.field(8, thriftStruct)
			w.structure(func() { w.i32(1, 10).i32(2, 11).i32(3, 10).i32(4, int32(Plain)) })
		})},
		{"a page holding a negative number of nulls", pageOf("", func(w *builder) {
			w.i32(1, int32(DataPageV2)).i32(2, 0).i32(3, 0)
			w.field(8, thriftStruct)
			w.structure(func() { w.i32(1, 10).i32(2, -1).i32(3, 10).i32(4, int32(Plain)) })
		})},
		{"a page covering more rows than it holds values", pageOf("", func(w *builder) {
			w.i32(1, int32(DataPageV2)).i32(2, 0).i32(3, 0)
			w.field(8, thriftStruct)
			w.structure(func() { w.i32(1, 10).i32(3, 11).i32(4, int32(Plain)) })
		})},
		{"levels longer than the page they are in", pageOf("body", func(w *builder) {
			w.i32(1, int32(DataPageV2)).i32(2, 4).i32(3, 4)
			w.field(8, thriftStruct)
			w.structure(func() {
				w.i32(1, 10).i32(3, 10).i32(4, int32(Plain)).i32(5, 3).i32(6, 2)
			})
		})},
		{"levels of a negative length", pageOf("body", func(w *builder) {
			w.i32(1, int32(DataPageV2)).i32(2, 4).i32(3, 4)
			w.field(8, thriftStruct)
			w.structure(func() {
				w.i32(1, 10).i32(3, 10).i32(4, int32(Plain)).i32(5, -1).i32(6, 1)
			})
		})},
		{"a body longer than what is left of the chunk", pageOf("short", func(w *builder) {
			w.i32(1, int32(DataPageV2)).i32(2, 500).i32(3, 500)
			w.field(8, thriftStruct)
			w.structure(func() { w.i32(1, 10).i32(3, 10).i32(4, int32(Plain)) })
		})},
		{"a body that does not match its checksum", pageOf("body", func(w *builder) {
			w.i32(1, int32(DataPageV2)).i32(2, 4).i32(3, 4).i32(4, 1)
			w.field(8, thriftStruct)
			w.structure(func() { w.i32(1, 4).i32(3, 4).i32(4, int32(Plain)) })
		})},
		{"a size written as a string", pageOf("", func(w *builder) {
			w.i32(1, int32(DataPage)).field(3, thriftBinary).binary("four")
		})},
		{"an encoding written as a string", pageOf("", func(w *builder) {
			w.i32(1, int32(DataPage)).i32(2, 0).i32(3, 0)
			w.field(5, thriftStruct)
			w.structure(func() { w.i32(1, 3).field(2, thriftBinary).binary("plain") })
		})},

		// A sub header written as anything other than a struct is read past
		// rather than refused where it sits, the same as a field this package
		// has never heard of, and what refuses it is the page then having said
		// nothing about itself.
		{"a data page header written as a number", pageOf("", func(w *builder) {
			w.i32(1, int32(DataPage)).i32(2, 0).i32(3, 0).i32(5, 7)
		})},
		{"a second version data page header written as a number", pageOf("", func(w *builder) {
			w.i32(1, int32(DataPageV2)).i32(2, 0).i32(3, 0).i32(8, 7)
		})},
		{"a dictionary page header written as a number", pageOf("", func(w *builder) {
			w.i32(1, int32(DictionaryPage)).i32(2, 0).i32(3, 0).i32(7, 7)
		})},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := chunkOf(t, tt.chunk).Next(); !errors.Is(err, ErrFormat) {
				t.Errorf("the page was read as %v, want a bad file", err)
			}
		})
	}
}

// TestPagesChunkRefused is the chunk itself rather than the pages in it. The
// offsets come out of a footer and are somebody else's numbers, so a chunk that
// claims to live past the end of the file must not turn into a read that size.
func TestPagesChunkRefused(t *testing.T) {
	const size = 1000

	tests := []struct {
		name  string
		file  int
		chunk ColumnChunk
		is    error
	}{
		{
			name:  "a chunk stored in another file",
			file:  size,
			chunk: ColumnChunk{FilePath: "elsewhere.parquet"},
			is:    ErrUnsupported,
		},
		{
			name:  "a chunk of a negative size",
			file:  size,
			chunk: ColumnChunk{Meta: ColumnMeta{DataPageOffset: 4, TotalCompressedSize: -1}},
			is:    ErrFormat,
		},
		{
			name:  "a chunk starting at a negative offset",
			file:  size,
			chunk: ColumnChunk{Meta: ColumnMeta{DataPageOffset: -4, TotalCompressedSize: 10}},
			is:    ErrFormat,
		},
		{
			name:  "a chunk starting past the end of the file",
			file:  size,
			chunk: ColumnChunk{Meta: ColumnMeta{DataPageOffset: size + 1, TotalCompressedSize: 10}},
			is:    ErrFormat,
		},
		{
			name:  "a chunk running past the end of the file",
			file:  size,
			chunk: ColumnChunk{Meta: ColumnMeta{DataPageOffset: size - 4, TotalCompressedSize: 10}},
			is:    ErrFormat,
		},
		{
			name:  "a file shorter than the size it was given",
			file:  10,
			chunk: ColumnChunk{Meta: ColumnMeta{DataPageOffset: 4, TotalCompressedSize: 100}},
			is:    io.EOF,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := bytes.NewReader(make([]byte, tt.file))
			if _, err := ReadPages(r, size, &tt.chunk); !errors.Is(err, tt.is) {
				t.Errorf("the chunk was read as %v, want %v", err, tt.is)
			}
		})
	}
}

// TestPagesTolerated is the fields a page written by a newer parquet than this
// one carries.
//
// Thrift puts a type on every field so that a reader can walk past what it does
// not know, and that is the whole reason the format can grow. A reader that
// refuses an unknown field is a reader that stops working the day somebody adds
// one, which for a page header would mean refusing every file a newer writer
// produced rather than the one field it added.
func TestPagesTolerated(t *testing.T) {
	tests := []struct {
		name  string
		kind  PageKind
		chunk []byte
	}{
		{"a field on the page header", DataPage, pageOf("body", func(w *builder) {
			w.i32(1, int32(DataPage)).i32(2, 4).i32(3, 4)
			w.field(5, thriftStruct)
			w.structure(func() {
				w.i32(1, 4).i32(2, int32(Plain)).i32(3, int32(RLE)).i32(4, int32(RLE))
			})
			w.field(20, thriftBinary).binary("something new")
		})},
		{"a field on a data page header", DataPage, pageOf("body", func(w *builder) {
			w.i32(1, int32(DataPage)).i32(2, 4).i32(3, 4)
			w.field(5, thriftStruct)
			w.structure(func() {
				w.i32(1, 4).i32(2, int32(Plain)).i32(3, int32(RLE)).i32(4, int32(RLE))
				w.field(20, thriftBinary).binary("something new")
			})
		})},
		{"a field on a second version data page header", DataPageV2,
			pageOf("body", func(w *builder) {
				w.i32(1, int32(DataPageV2)).i32(2, 4).i32(3, 4)
				w.field(8, thriftStruct)
				w.structure(func() {
					w.i32(1, 4).i32(3, 4).i32(4, int32(Plain))
					w.field(20, thriftBinary).binary("something new")
				})
			})},
		{"a field on a dictionary page header", DictionaryPage, pageOf("body", func(w *builder) {
			w.i32(1, int32(DictionaryPage)).i32(2, 4).i32(3, 4)
			w.field(7, thriftStruct)
			w.structure(func() {
				w.i32(1, 4).i32(2, int32(Plain))
				w.field(20, thriftBinary).binary("something new")
			})
		})},

		// An index page is a page type the format defined and nothing ever
		// wrote. Its header is an empty struct, so there is nothing in it to
		// check and nothing to refuse.
		{"an index page", IndexPage, pageOf("body", func(w *builder) {
			w.i32(1, int32(IndexPage)).i32(2, 4).i32(3, 4)
			w.field(6, thriftStruct)
			w.structure(func() {})
		})},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := chunkOf(t, tt.chunk).Next()
			if err != nil {
				t.Fatalf("the page: %v", err)
			}
			if p.Kind != tt.kind || string(p.Data) != "body" {
				t.Errorf("the page is a %s holding %q, want a %s holding the body",
					p.Kind, p.Data, tt.kind)
			}
		})
	}
}

// TestPagesUnknown walks past a page type this package has never heard of.
//
// The format has added a page type before and may again, and a walk only needs
// the size of a page to get past it. Refusing here would mean a file with one
// unreadable page in it is a file with no readable pages in it, and whatever
// wants to decode the page can say no itself.
func TestPagesUnknown(t *testing.T) {
	chunk := pageOf("nine bytes", func(w *builder) {
		w.i32(1, 99).i32(2, 10).i32(3, 10)
	})
	chunk = append(chunk, pageOf("body", func(w *builder) {
		w.i32(1, int32(DataPageV2)).i32(2, 4).i32(3, 4)
		w.field(8, thriftStruct)
		w.structure(func() { w.i32(1, 4).i32(3, 4).i32(4, int32(Plain)) })
	})...)

	pages := chunkOf(t, chunk)
	first, err := pages.Next()
	if err != nil {
		t.Fatalf("the unknown page: %v", err)
	}
	if first.Kind != 99 || string(first.Data) != "nine bytes" {
		t.Errorf("the unknown page is a %s holding %q", first.Kind, first.Data)
	}

	second, err := pages.Next()
	if err != nil {
		t.Fatalf("the page behind it: %v", err)
	}
	if string(second.Data) != "body" {
		t.Errorf("the page behind it holds %q, want the walk to have got past the first",
			second.Data)
	}
	if _, err := pages.Next(); !errors.Is(err, io.EOF) {
		t.Errorf("the end of the chunk is %v, want the end of the file", err)
	}
}

// TestPagesLevelEncodings reads the levels of a data page written the first
// way, where the encodings are on the page and are not always the same two.
//
// A file written by an old parquet-mr says its repetition and definition levels
// are bit packed rather than RLE, and those are two different encodings that
// happen to share a decoder for the case a page of levels usually is. A reader
// that assumes RLE reads such a file wrong.
func TestPagesLevelEncodings(t *testing.T) {
	chunk := pageOf("body", func(w *builder) {
		w.i32(1, int32(DataPage)).i32(2, 4).i32(3, 4)
		w.field(5, thriftStruct)
		w.structure(func() {
			w.i32(1, 4).i32(2, int32(PlainDictionary)).i32(3, int32(BitPacked)).i32(4, int32(BitPacked))
		})
	})

	p, err := chunkOf(t, chunk).Next()
	if err != nil {
		t.Fatalf("the page: %v", err)
	}
	if p.Encoding != PlainDictionary {
		t.Errorf("the values are %s encoded, want plain_dictionary", p.Encoding)
	}
	if p.DefinitionEncoding != BitPacked || p.RepetitionEncoding != BitPacked {
		t.Errorf("the levels are %s and %s encoded, want both of them bit packed",
			p.DefinitionEncoding, p.RepetitionEncoding)
	}
}

// TestPagesEnd is a chunk read past its end, which keeps saying the same thing.
func TestPagesEnd(t *testing.T) {
	pages := chunkOf(t, nil)
	for range 3 {
		if _, err := pages.Next(); !errors.Is(err, io.EOF) {
			t.Fatalf("the end of an empty chunk is %v, want the end of the file", err)
		}
	}
}

// TestPagesDamaged flips a byte inside a page of a real file.
//
// The checksum is the only thing in a parquet file that catches this. Every
// other number in a page is a length or a count, and a bit flipped in the
// middle of the values is a value that decodes cleanly into the wrong thing.
func TestPagesDamaged(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("testdata", "pages.parquet"))
	if err != nil {
		t.Fatalf("reading pages.parquet: %v", err)
	}
	m, err := ReadMetadata(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatalf("reading the footer: %v", err)
	}

	// The last byte of the first chunk, which is the last value of its second
	// page rather than anything the walk itself reads.
	c := &m.RowGroups[0].Columns[0]
	b[c.Start()+c.Meta.TotalCompressedSize-1] ^= 0xff

	pages, err := ReadPages(bytes.NewReader(b), int64(len(b)), c)
	if err != nil {
		t.Fatalf("reading the chunk: %v", err)
	}
	if _, err := pages.Next(); err != nil {
		t.Fatalf("the page in front of the damage: %v", err)
	}
	if _, err := pages.Next(); !errors.Is(err, ErrFormat) {
		t.Errorf("the damaged page was read as %v, want a bad file", err)
	}
}

func BenchmarkPages(b *testing.B) {
	for _, name := range []string{"alltypes.parquet", "pages.parquet"} {
		b.Run(strings.TrimSuffix(name, ".parquet"), func(b *testing.B) {
			r, size, m := openFile(b, name)
			b.ReportAllocs()
			b.ResetTimer()

			for b.Loop() {
				for i := range m.RowGroups[0].Columns {
					pages, err := ReadPages(r, size, &m.RowGroups[0].Columns[i])
					if err != nil {
						b.Fatal(err)
					}
					for {
						if _, err := pages.Next(); err != nil {
							if errors.Is(err, io.EOF) {
								break
							}
							b.Fatal(err)
						}
					}
				}
			}
		})
	}
}
