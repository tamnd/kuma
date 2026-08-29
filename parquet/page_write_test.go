package parquet_test

import (
	"bytes"
	"errors"
	"hash/crc32"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tamnd/kuma/parquet"
)

// rewrite writes a chunk's pages back out one at a time and hands back the
// bytes, which is what a caller building a column chunk out of pages it read
// somewhere else ends up with.
func rewrite(t *testing.T, pages []parquet.Page) []byte {
	t.Helper()

	var out bytes.Buffer
	for i := range pages {
		n, err := parquet.WritePage(&out, &pages[i].PageHeader, pages[i].Data)
		if err != nil {
			t.Fatalf("WritePage of page %d: %v", i, err)
		}
		if n <= int64(len(pages[i].Data)) {
			t.Fatalf("page %d wrote %d bytes for a body of %d, which leaves no header",
				i, n, len(pages[i].Data))
		}
	}
	return out.Bytes()
}

// walk reads back a run of pages written end to end, which is the same walk the
// reader does over a real column chunk.
//
// A run of pages is a column chunk with nothing else to it, so it is read as one
// rather than through a file, with a chunk that starts at nought and is as long
// as the bytes are.
func walk(t *testing.T, chunk []byte) []parquet.Page {
	t.Helper()

	c := &parquet.ColumnChunk{Meta: parquet.ColumnMeta{
		Path:                []string{"rewritten"},
		TotalCompressedSize: int64(len(chunk)),
	}}
	pages, err := parquet.ReadPages(bytes.NewReader(chunk), int64(len(chunk)), c)
	if err != nil {
		t.Fatalf("ReadPages: %v", err)
	}
	return allPages(t, pages)
}

// allPages reads pages until the walk says there are none left, and fails on
// anything that is not the end.
func allPages(t *testing.T, pages *parquet.Pages) []parquet.Page {
	t.Helper()

	var out []parquet.Page
	for {
		p, err := pages.Next()
		if errors.Is(err, io.EOF) {
			return out
		}
		if err != nil {
			t.Fatalf("page %d: %v", len(out), err)
		}
		out = append(out, p)
	}
}

// TestWritePageRoundTrip walks the pages of every column of every file in
// testdata, writes each of them back out, and walks the result.
//
// This is the test that matters, because the walk is the thing a wrong header
// breaks. A page header has no length in front of it, so the reader finds the
// second page by reading the first header and adding up, and a size that is one
// byte out gives back garbage for every page after it rather than an error on
// the page that was wrong. Getting the same pages back out of a rewritten chunk
// is the only way to know the sizes were written where the reader looks for
// them.
func TestWritePageRoundTrip(t *testing.T) {
	for _, name := range files(t) {
		t.Run(name, func(t *testing.T) {
			b := bytesOf(t, name)
			m, err := parquet.ReadMetadata(bytes.NewReader(b), int64(len(b)))
			if err != nil {
				t.Fatalf("ReadMetadata: %v", err)
			}

			seen := 0
			for g := range m.RowGroups {
				for c := range m.RowGroups[g].Columns {
					chunk := &m.RowGroups[g].Columns[c]
					path := strings.Join(chunk.Meta.Path, ".")

					pages, err := parquet.ReadPages(bytes.NewReader(b), int64(len(b)), chunk)
					if err != nil {
						t.Fatalf("ReadPages(%s): %v", path, err)
					}

					want := allPages(t, pages)
					if len(want) == 0 {
						continue
					}

					got := walk(t, rewrite(t, want))
					if len(got) != len(want) {
						t.Fatalf("%s came back with %d pages, want %d", path, len(got), len(want))
					}
					for i := range got {
						if !reflect.DeepEqual(got[i].PageHeader, want[i].PageHeader) {
							t.Errorf("%s page %d header came back different\n got %+v\nwant %+v",
								path, i, got[i].PageHeader, want[i].PageHeader)
						}
						if !bytes.Equal(got[i].Data, want[i].Data) {
							t.Errorf("%s page %d body came back different", path, i)
						}
					}
					seen += len(want)
				}
			}
			if seen == 0 {
				t.Fatalf("%s has no pages at all", name)
			}
		})
	}
}

// TestWritePageSize checks a rewritten chunk against the bytes pyarrow wrote.
//
// It will not be the same length, since the writers disagree about which fields
// are worth writing, but a chunk that came back much larger would mean this is
// writing something pyarrow leaves out and a chunk that came back much smaller
// would mean it is leaving out something pyarrow writes.
func TestWritePageSize(t *testing.T) {
	for _, name := range []string{"plain.parquet", "stats.parquet", "pages.parquet", "dictionary.parquet"} {
		t.Run(name, func(t *testing.T) {
			b := bytesOf(t, name)
			m, err := parquet.ReadMetadata(bytes.NewReader(b), int64(len(b)))
			if err != nil {
				t.Fatalf("ReadMetadata: %v", err)
			}

			var theirs, ours int64
			for g := range m.RowGroups {
				for c := range m.RowGroups[g].Columns {
					chunk := &m.RowGroups[g].Columns[c]
					pages, err := parquet.ReadPages(bytes.NewReader(b), int64(len(b)), chunk)
					if err != nil {
						t.Fatalf("ReadPages: %v", err)
					}
					theirs += chunk.Meta.TotalCompressedSize
					ours += int64(len(rewrite(t, allPages(t, pages))))
				}
			}

			if ours > theirs+theirs/10 || ours < theirs-theirs/10 {
				t.Errorf("%s rewrote %d bytes of pages against pyarrow's %d", name, ours, theirs)
			}
		})
	}
}

// TestWritePageChecksum checks the checksum is the writer's to work out.
//
// A page carries a CRC of its body or it does not, and a caller that had to
// compute one itself would be a caller that can get it wrong. So the header says
// whether to write one and this writes the right one, which means a page that
// says it has a checksum and does not match it cannot come out of here.
func TestWritePageChecksum(t *testing.T) {
	body := []byte("the body of a page, whatever it holds")
	h := parquet.PageHeader{
		Kind:             parquet.DictionaryPage,
		NumValues:        4,
		Encoding:         parquet.Plain,
		CompressedSize:   int32(len(body)),
		UncompressedSize: int32(len(body)),
		Compressed:       true,
		HasCRC:           true,

		// A value nothing computed, to check it is not the one written.
		CRC: 1234,
	}

	var out bytes.Buffer
	if _, err := parquet.WritePage(&out, &h, body); err != nil {
		t.Fatalf("WritePage: %v", err)
	}

	got := walk(t, out.Bytes())
	if len(got) != 1 {
		t.Fatalf("wrote one page and read back %d", len(got))
	}
	if !got[0].HasCRC {
		t.Fatal("the page came back with no checksum")
	}
	if want := int32(crc32.ChecksumIEEE(body)); got[0].CRC != want {
		t.Errorf("the checksum came back as %d, want %d", got[0].CRC, want)
	}
}

// TestWritePageSorted writes a dictionary page that says its entries are in
// order, which nothing in testdata does.
//
// It is worth writing even though nothing reads it yet. A sorted dictionary lets
// a reader compare indices instead of values, which turns comparing a column of
// strings into comparing a column of small integers, and a writer that sorts its
// dictionary and does not say so has done the work and thrown away the part that
// makes it useful.
func TestWritePageSorted(t *testing.T) {
	body := []byte("aaabbbccc")
	h := parquet.PageHeader{
		Kind: parquet.DictionaryPage, Encoding: parquet.Plain, NumValues: 3, Sorted: true,
		CompressedSize: int32(len(body)), UncompressedSize: int32(len(body)),
	}

	var out bytes.Buffer
	if _, err := parquet.WritePage(&out, &h, body); err != nil {
		t.Fatalf("WritePage: %v", err)
	}

	got := walk(t, out.Bytes())
	if len(got) != 1 {
		t.Fatalf("wrote one page and read back %d", len(got))
	}
	if !got[0].Sorted {
		t.Error("the page came back saying its dictionary is not sorted")
	}
}

// TestWritePageRefused is the headers that do not describe their body.
//
// None of these produce a page a reader would notice. A compressed size that is
// short leaves the walk pointing into the middle of this page for the next
// header, so the file it produces is one that reads as a different file rather
// than one that fails to read, which is why they are refused at the call.
func TestWritePageRefused(t *testing.T) {
	body := make([]byte, 20)

	tests := []struct {
		name   string
		header parquet.PageHeader
		body   []byte
	}{
		{
			name: "a compressed size longer than the body",
			header: parquet.PageHeader{Kind: parquet.DictionaryPage, Encoding: parquet.Plain,
				CompressedSize: 21, UncompressedSize: 20},
			body: body,
		},
		{
			name: "a compressed size shorter than the body",
			header: parquet.PageHeader{Kind: parquet.DictionaryPage, Encoding: parquet.Plain,
				CompressedSize: 19, UncompressedSize: 20},
			body: body,
		},
		{
			name: "levels that take more than the whole body",
			header: parquet.PageHeader{Kind: parquet.DataPageV2, Encoding: parquet.Plain,
				CompressedSize: 20, UncompressedSize: 20, NumValues: 5, NumRows: 5,
				DefinitionLength: 12, RepetitionLength: 12},
			body: body,
		},
		{
			name: "a data page that does not say how it is encoded",
			header: parquet.PageHeader{Kind: parquet.DataPage, Encoding: parquet.NoEncoding,
				DefinitionEncoding: parquet.RLE, RepetitionEncoding: parquet.RLE,
				CompressedSize: 20, UncompressedSize: 20},
			body: body,
		},
		{
			name: "a data page that does not say how its levels are encoded",
			header: parquet.PageHeader{Kind: parquet.DataPage, Encoding: parquet.Plain,
				DefinitionEncoding: parquet.NoEncoding, RepetitionEncoding: parquet.NoEncoding,
				CompressedSize: 20, UncompressedSize: 20},
			body: body,
		},
		{
			name: "a page of a negative number of bytes",
			header: parquet.PageHeader{Kind: parquet.DictionaryPage, Encoding: parquet.Plain,
				CompressedSize: -1, UncompressedSize: -1},
			body: nil,
		},
		{
			name: "a page holding more nulls than values",
			header: parquet.PageHeader{Kind: parquet.DataPageV2, Encoding: parquet.Plain,
				CompressedSize: 20, UncompressedSize: 20, NumValues: 3, NumNulls: 4},
			body: body,
		},
		{
			name: "an index page, which nothing writes",
			header: parquet.PageHeader{Kind: parquet.IndexPage,
				CompressedSize: 20, UncompressedSize: 20},
			body: body,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var out bytes.Buffer
			n, err := parquet.WritePage(&out, &tt.header, tt.body)
			if err == nil {
				t.Fatalf("WritePage wrote %d bytes for %s", n, tt.name)
			}
			if out.Len() != 0 {
				t.Errorf("WritePage wrote %d bytes before refusing", out.Len())
			}
		})
	}
}

// TestWritePageUnknownKind writes a page whose kind the format has not defined.
//
// The walk steps past one of those rather than refusing the file, since a page
// it cannot read is still a page whose size it can add up, so the writer has to
// be able to produce one for that to be worth anything.
func TestWritePageUnknownKind(t *testing.T) {
	body := []byte("something a later version of the format put here")
	h := parquet.PageHeader{
		Kind:             parquet.PageKind(99),
		CompressedSize:   int32(len(body)),
		UncompressedSize: int32(len(body)),
	}

	var out bytes.Buffer
	if _, err := parquet.WritePage(&out, &h, body); err != nil {
		t.Fatalf("WritePage: %v", err)
	}

	// A page of a kind nothing knows, then a page of one everything does. The
	// second only comes back if the walk stepped over the first by exactly the
	// right number of bytes.
	after := []byte("the next page")
	next := parquet.PageHeader{
		Kind: parquet.DictionaryPage, Encoding: parquet.Plain, NumValues: 2,
		CompressedSize: int32(len(after)), UncompressedSize: int32(len(after)),
	}
	if _, err := parquet.WritePage(&out, &next, after); err != nil {
		t.Fatalf("WritePage: %v", err)
	}

	got := walk(t, out.Bytes())
	if len(got) != 2 {
		t.Fatalf("wrote two pages and read back %d", len(got))
	}
	if got[0].Kind != 99 || !bytes.Equal(got[0].Data, body) {
		t.Errorf("the unknown page came back as %+v", got[0].PageHeader)
	}
	if got[1].Kind != parquet.DictionaryPage || !bytes.Equal(got[1].Data, after) {
		t.Errorf("the page behind it came back as %+v", got[1].PageHeader)
	}
}

// TestWritePageError checks a writer that will not take the bytes is reported
// rather than swallowed.
func TestWritePageError(t *testing.T) {
	body := []byte("a body")
	h := parquet.PageHeader{
		Kind: parquet.DictionaryPage, Encoding: parquet.Plain, NumValues: 1,
		CompressedSize: int32(len(body)), UncompressedSize: int32(len(body)),
	}

	if _, err := parquet.WritePage(refuser{}, &h, body); !errors.Is(err, os.ErrClosed) {
		t.Errorf("WritePage to a closed writer = %v, want the writer's own error", err)
	}
}

// BenchmarkWritePage writes the pages of one chunk of a real file, which is what
// a column writer does once per page and nothing else does at all.
func BenchmarkWritePage(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "pages.parquet"))
	if err != nil {
		b.Fatal(err)
	}
	m, err := parquet.ReadMetadata(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		b.Fatal(err)
	}

	chunk := &m.RowGroups[0].Columns[0]
	pages, err := parquet.ReadPages(bytes.NewReader(raw), int64(len(raw)), chunk)
	if err != nil {
		b.Fatal(err)
	}
	var all []parquet.Page
	for {
		p, err := pages.Next()
		if err != nil {
			break
		}
		all = append(all, p)
	}
	if len(all) == 0 {
		b.Fatal("no pages")
	}

	var out bytes.Buffer
	b.ResetTimer()
	for b.Loop() {
		out.Reset()
		for i := range all {
			if _, err := parquet.WritePage(&out, &all[i].PageHeader, all[i].Data); err != nil {
				b.Fatal(err)
			}
		}
	}
}
