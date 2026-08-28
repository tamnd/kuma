package parquet_test

import (
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/parquet"
)

// The two files these read are the same thousand rows written twice, once with
// each version of the data page, and every column of them is compressed
// differently: one is left alone, two are snappy, two are gzip and one is
// brotli so that there is something to be refused by name. That is the point of
// them. A reader that undoes the codec of the file rather than the codec of the
// chunk reads five of the six columns wrong, and a reader that undoes the whole
// body of a second version page puts the levels through the codec they were
// deliberately left out of.

// snappied returns the bytes as a snappy block holding them as a single
// literal, which is what a compressor emits for anything it cannot do better
// with and is enough for a page that only has to be well formed.
func snappied(p []byte) []byte {
	if len(p) == 0 {
		return []byte{0}
	}
	if len(p) > 60 {
		panic("the inline literal length only reaches sixty bytes")
	}
	b := binary.AppendUvarint(nil, uint64(len(p)))
	b = append(b, byte((len(p)-1)<<2))
	return append(b, p...)
}

// gzipped returns the bytes as a gzip member, which is what the codec the
// format calls GZIP is: a header, deflate, a length and a checksum.
func gzipped(tb testing.TB, p []byte) []byte {
	tb.Helper()

	var b bytes.Buffer
	w := gzip.NewWriter(&b)
	if _, err := w.Write(p); err != nil {
		tb.Fatalf("gzip: %v", err)
	}
	if err := w.Close(); err != nil {
		tb.Fatalf("gzip: %v", err)
	}
	return b.Bytes()
}

// compressedPage is a page of a body with a header saying what it comes to,
// which is all a decompressor looks at.
func compressedPage(body []byte, size int) parquet.Page {
	return parquet.Page{
		PageHeader: parquet.PageHeader{
			Kind:             parquet.DataPage,
			CompressedSize:   int32(len(body)),
			UncompressedSize: int32(size),
			Compressed:       true,
		},
		Data: body,
	}
}

// decompressorFor makes a decompressor and fails the test if the codec is one
// the package will not undo.
func decompressorFor(tb testing.TB, c parquet.Codec) *parquet.Decompressor {
	tb.Helper()

	d, err := parquet.NewDecompressor(c)
	if err != nil {
		tb.Fatalf("NewDecompressor(%s): %v", c, err)
	}
	return d
}

// undo runs a page through a decompressor and fails the test if it is refused.
func undo(tb testing.TB, d *parquet.Decompressor, p parquet.Page) parquet.Page {
	tb.Helper()

	out, err := d.Page(p)
	if err != nil {
		tb.Fatalf("Page: %v", err)
	}
	return out
}

// TestNewDecompressor is the codecs that can be undone and the codecs that
// cannot.
//
// The refusal is at the chunk rather than at the page on purpose, so a scan
// that cannot read a column finds out before it has read any of it, and the
// codec a decompressor was made for is worth asking for afterwards since it is
// the chunk that decided it rather than the caller.
func TestNewDecompressor(t *testing.T) {
	for _, c := range []parquet.Codec{parquet.Uncompressed, parquet.Snappy, parquet.Gzip} {
		t.Run(c.String(), func(t *testing.T) {
			if got := decompressorFor(t, c).Codec(); got != c {
				t.Errorf("the decompressor undoes %s, want %s", got, c)
			}
		})
	}

	// The four the format has that this package has not. Each of them is a
	// codec that exists and works rather than a number nobody writes, so a file
	// holding one is a file to come back to rather than a file that is wrong.
	for _, c := range []parquet.Codec{parquet.LZO, parquet.Brotli, parquet.LZ4, parquet.Zstd} {
		t.Run(c.String(), func(t *testing.T) {
			_, err := parquet.NewDecompressor(c)
			if !errors.Is(err, parquet.ErrUnsupported) {
				t.Errorf("got %v, want %v", err, parquet.ErrUnsupported)
			}
			if s := err.Error(); !strings.Contains(s, c.String()) {
				t.Errorf("the error does not say which codec it was: %s", s)
			}
		})
	}
}

// TestReadColumnCompressed reads every column of the two files that this
// package has a codec for, in both versions of the data page.
//
// The values are the whole of it. A page that has been decompressed into the
// wrong place, or one whose levels went through the codec when they should not
// have, gives a column of the right length holding the wrong things, so
// checking each value against what the script wrote is what says the page came
// back whole.
func TestReadColumnCompressed(t *testing.T) {
	for _, file := range []string{"codecs.parquet", "codecs2.parquet"} {
		t.Run(file, func(t *testing.T) {
			t.Run("uncompressed", func(t *testing.T) {
				a := readColumn(t, file, "plain")
				if a.Len() != 1000 || a.NullCount() != 0 {
					t.Fatalf("%d values, %d null", a.Len(), a.NullCount())
				}
				for i := range a.Len() {
					if got, want := string(a.Bytes(i)), fmt.Sprintf("row-%d", i); got != want {
						t.Fatalf("value %d: got %q, want %q", i, got, want)
					}
				}
			})

			t.Run("gzip", func(t *testing.T) {
				a := readColumn(t, file, "zip")
				if a.Len() != 1000 || a.NullCount() != 0 {
					t.Fatalf("%d values, %d null", a.Len(), a.NullCount())
				}
				for i, got := range a.Values[int64]() {
					if want := int64(i) * 3; got != want {
						t.Fatalf("value %d: got %d, want %d", i, got, want)
					}
				}
			})

			t.Run("snappy", func(t *testing.T) {
				a := readColumn(t, file, "snap")
				if a.Len() != 1000 || a.NullCount() != 0 {
					t.Fatalf("%d values, %d null", a.Len(), a.NullCount())
				}
				for i := range a.Len() {
					want := fmt.Sprintf("customer/2026/08/%06d", i)
					if got := string(a.Bytes(i)); got != want {
						t.Fatalf("value %d: got %q, want %q", i, got, want)
					}
				}
			})

			// The dictionary page goes through the codec the same way a data
			// page does, and it is the one page of the chunk that holds values
			// rather than indices, so a chunk whose dictionary came back wrong
			// is a chunk where every row is wrong.
			t.Run("a dictionary", func(t *testing.T) {
				a := readColumn(t, file, "word")
				dictionaryShape(t, a, dtype.String, 1000, 4)

				words := []string{"alpha", "beta", "gamma", "delta"}
				for i := range a.Len() {
					want := words[i%len(words)]
					if got := string(a.Dictionary().Bytes(a.Index(i))); got != want {
						t.Fatalf("value %d: got %q, want %q", i, got, want)
					}
				}
			})

			// The column with the levels worth having. Every fifth row is
			// missing, so the definition levels take fifty odd bytes a page,
			// and in the second version of the data page those bytes sit in
			// front of the compressed values rather than inside them.
			t.Run("with nulls", func(t *testing.T) {
				a := readColumn(t, file, "maybe")
				if a.Len() != 1000 || a.NullCount() != 200 {
					t.Fatalf("%d values, %d null", a.Len(), a.NullCount())
				}
				values := a.Values[int32]()
				for i := range a.Len() {
					if i%5 == 0 {
						if !a.IsNull(i) {
							t.Fatalf("value %d is there and should be missing", i)
						}
						continue
					}
					if values[i] != int32(i) {
						t.Fatalf("value %d: got %d, want %d", i, values[i], i)
					}
				}
			})
		})
	}
}

// TestDecompressorLevels checks that the levels of a second version data page
// come back exactly as the file wrote them.
//
// They are the piece that does not go through the codec. The header says how
// many bytes of the body they take, they sit in front of the compressed values,
// and putting the page back together means copying them across untouched. This
// walks the pages twice, once as they sit in the file and once undone, and asks
// that the bytes in front are the same and that what comes back is as long as
// the header promised.
func TestDecompressorLevels(t *testing.T) {
	b, chunk, _ := chunkOf(t, "codecs2.parquet", "maybe")
	raw, err := parquet.ReadPages(bytes.NewReader(b), int64(len(b)), chunk)
	if err != nil {
		t.Fatalf("ReadPages: %v", err)
	}

	d := decompressorFor(t, chunk.Meta.Codec)
	levels := 0
	for n := 0; ; n++ {
		p, err := raw.Next()
		if errors.Is(err, io.EOF) {
			if n == 0 {
				t.Fatal("the chunk has no pages in it")
			}
			break
		}
		if err != nil {
			t.Fatalf("page %d: %v", n, err)
		}

		want := p.Data[:p.RepetitionLength+p.DefinitionLength]
		levels += len(want)

		out := undo(t, d, p)
		if len(out.Data) != int(out.UncompressedSize) {
			t.Fatalf("page %d comes to %d bytes and says %d", n, len(out.Data), out.UncompressedSize)
		}
		if got := out.Data[:len(want)]; !bytes.Equal(got, want) {
			t.Fatalf("page %d: the levels came back as %x, want %x", n, got, want)
		}
	}
	if levels == 0 {
		t.Fatal("no page of the chunk has any levels in front of it")
	}
}

// TestDecompressorLeavesAlone is the pages that come back as they went in.
//
// Three of them, and only one is a file that is not compressed. A chunk written
// with a codec may still hold a page that was not, since the second version of
// the data page lets a writer give up on a page the codec made bigger, and a
// page of a kind nothing reads is not worth undoing at all.
func TestDecompressorLeavesAlone(t *testing.T) {
	body := []byte("levels and values, uncompressed")

	cases := []struct {
		name  string
		codec parquet.Codec
		page  parquet.Page
	}{{
		name:  "a chunk with no codec",
		codec: parquet.Uncompressed,
		page:  compressedPage(body, len(body)),
	}, {
		name:  "a page the writer gave up compressing",
		codec: parquet.Snappy,
		page: parquet.Page{
			PageHeader: parquet.PageHeader{
				Kind:             parquet.DataPageV2,
				CompressedSize:   int32(len(body)),
				UncompressedSize: int32(len(body)),
				DefinitionLength: 4,
			},
			Data: body,
		},
	}, {
		name:  "a page of a kind nothing reads",
		codec: parquet.Snappy,
		page: parquet.Page{
			PageHeader: parquet.PageHeader{
				Kind:             parquet.IndexPage,
				CompressedSize:   int32(len(body)),
				UncompressedSize: int32(len(body)),
				Compressed:       true,
			},
			Data: body,
		},
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out := undo(t, decompressorFor(t, c.codec), c.page)
			if &out.Data[0] != &c.page.Data[0] || len(out.Data) != len(c.page.Data) {
				t.Fatalf("the page came back as %q rather than as the bytes it went in as", out.Data)
			}
		})
	}
}

// TestDecompressorRefused is the pages that are not undone.
//
// Every one of them is a header and a body that contradict each other, which is
// the only thing a decompressor is in a position to notice. A page that says it
// comes to more bytes than it does, one that is cut short, one whose checksum
// says the bytes changed on the way, and the levels of a second version page
// claiming more of the body than the body has.
func TestDecompressorRefused(t *testing.T) {
	short := []byte("a body worth compressing, compressing, compressing")
	block := snappied(short[:30])

	withLevels := func(rep, def, size int32) parquet.Page {
		p := compressedPage(block, len(block))
		p.Kind = parquet.DataPageV2
		p.RepetitionLength, p.DefinitionLength = rep, def
		p.UncompressedSize = size
		return p
	}

	cases := []struct {
		name  string
		codec parquet.Codec
		page  parquet.Page
	}{{
		name:  "a snappy page that comes to more than its header says",
		codec: parquet.Snappy,
		page:  compressedPage(block, 29),
	}, {
		name:  "a snappy block whose length runs off the end of it",
		codec: parquet.Snappy,
		page:  compressedPage([]byte{0xff}, 30),
	}, {
		name:  "a snappy block that is cut short",
		codec: parquet.Snappy,
		page:  compressedPage(block[:len(block)-3], 30),
	}, {
		name:  "a gzip page that is not a gzip member",
		codec: parquet.Gzip,
		page:  compressedPage(short, 30),
	}, {
		name:  "a gzip page that is cut short",
		codec: parquet.Gzip,
		page:  compressedPage(gzipped(t, short)[:12], 30),
	}, {
		name:  "a gzip page that comes to less than its header says",
		codec: parquet.Gzip,
		page:  compressedPage(gzipped(t, short[:20]), 30),
	}, {
		name:  "a gzip page that comes to more than its header says",
		codec: parquet.Gzip,
		page:  compressedPage(gzipped(t, short), 30),
	}, {
		name:  "a gzip page whose checksum says the bytes changed",
		codec: parquet.Gzip,
		page:  compressedPage(broken(gzipped(t, short[:30])), 30),
	}, {
		name:  "a page whose levels take a negative number of bytes",
		codec: parquet.Snappy,
		page:  withLevels(-4, 0, 30),
	}, {
		name:  "a page whose levels take more bytes than it has",
		codec: parquet.Snappy,
		page:  withLevels(0, int32(len(block)+1), 30),
	}, {
		name:  "a page that comes to fewer bytes than its levels take",
		codec: parquet.Snappy,
		page:  withLevels(0, 8, 4),
	}, {
		// The two of these are the size being refused before a buffer is made
		// for it rather than after. A page of thirty bytes saying it comes to a
		// gigabyte is a number nothing could have written, and allocating it to
		// find that out is what a hostile file is counting on.
		name:  "a snappy page claiming more than the codec could write",
		codec: parquet.Snappy,
		page:  compressedPage(block, 1<<30),
	}, {
		name:  "a gzip page claiming more than the codec could write",
		codec: parquet.Gzip,
		page:  compressedPage(gzipped(t, short), 1<<30),
	}}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out, err := decompressorFor(t, c.codec).Page(c.page)
			if !errors.Is(err, parquet.ErrFormat) {
				t.Fatalf("got %v, want %v", err, parquet.ErrFormat)
			}
			if out.Data != nil {
				t.Errorf("a refused page came back holding %d bytes", len(out.Data))
			}
		})
	}
}

// TestReadColumnCorrupted reads a column whose last page has had a byte of it
// changed, which a scan has to refuse by name.
//
// The byte is the last one of the chunk, which is the top of the checksum the
// gzip member ends with, so the page decompresses and then says that what came
// out is not what went in. That is worth reaching through ReadColumn rather
// than only through the decompressor: a chunk that fails half way through is
// one where several pages have already gone into the builder, and the column
// has to come back as an error rather than as the rows that were readable.
func TestReadColumnCorrupted(t *testing.T) {
	b, chunk, c := chunkOf(t, "codecs.parquet", "zip")
	b = bytes.Clone(b)
	b[chunk.Start()+chunk.Meta.TotalCompressedSize-1] ^= 0x80

	_, err := parquet.ReadColumn(bytes.NewReader(b), int64(len(b)), chunk, c)
	if !errors.Is(err, parquet.ErrFormat) {
		t.Fatalf("got %v, want %v", err, parquet.ErrFormat)
	}
	if s := err.Error(); !strings.Contains(s, c.Name()) {
		t.Errorf("the error does not say which column it was: %s", s)
	}
}

// broken flips a bit of the last byte of a gzip member, which is the top byte
// of the length it ends with, so the member decompresses and then says the
// wrong thing about what came out.
func broken(p []byte) []byte {
	p[len(p)-1] ^= 0x80
	return p
}

// TestDecompressorReuse checks that a decompressor undoing a chunk of many
// pages allocates for the first one and not for the rest.
//
// That is the reason it is a value with a buffer in it rather than a function.
// A chunk of a real file is hundreds of pages and every one of them comes to
// about the same size, so the buffer the first page grew is the buffer the rest
// of them fit in, and the gzip reader is reset onto each page rather than made
// again.
func TestDecompressorReuse(t *testing.T) {
	b, chunk, _ := chunkOf(t, "codecs.parquet", "zip")
	d := decompressorFor(t, chunk.Meta.Codec)

	pages := func() []parquet.Page {
		p, err := parquet.ReadPages(bytes.NewReader(b), int64(len(b)), chunk)
		if err != nil {
			t.Fatalf("ReadPages: %v", err)
		}
		var all []parquet.Page
		for {
			page, err := p.Next()
			if errors.Is(err, io.EOF) {
				return all
			}
			if err != nil {
				t.Fatalf("page %d: %v", len(all), err)
			}
			all = append(all, page)
		}
	}()
	if len(pages) < 2 {
		t.Fatalf("the chunk is %d pages and this wants several", len(pages))
	}

	for _, p := range pages {
		undo(t, d, p)
	}
	if n := testing.AllocsPerRun(4, func() {
		for _, p := range pages {
			if _, err := d.Page(p); err != nil {
				t.Fatalf("Page: %v", err)
			}
		}
	}); n != 0 {
		t.Errorf("%v allocations undoing %d pages a second time", n, len(pages))
	}
}

// BenchmarkDecompress reads a compressed column of a real file, which is the
// whole path with the codec in it: the chunk is read, every page is undone, and
// the values and levels are put into an array.
//
// The rate is over the bytes the pages come to rather than the bytes they take
// in the file, since that is the work the decoders behind this do and it is
// what makes the codecs comparable. The uncompressed column is the control, and
// the difference between it and the other two is what the codec costs.
func BenchmarkDecompress(b *testing.B) {
	for _, column := range []string{"plain", "snap", "zip"} {
		b.Run(column, func(b *testing.B) {
			file, chunk, c := chunkOf(&testing.T{}, "codecs.parquet", column)
			r := bytes.NewReader(file)

			b.SetBytes(chunk.Meta.TotalUncompressedSize)
			for b.Loop() {
				if _, err := parquet.ReadColumn(r, int64(len(file)), chunk, c); err != nil {
					b.Fatalf("ReadColumn: %v", err)
				}
			}
		})
	}
}
