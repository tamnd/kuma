package parquet_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/bits"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/parquet"
)

// maxWidthTested is the widest value the format allows, which is the width of
// an index into a dictionary of two billion entries and is as wide as the byte
// in front of a page's indices can say.
const maxWidthTested = 32

// widthName names a subtest after the width it runs at.
func widthName(width int) string { return fmt.Sprintf("width %d", width) }

// pack writes values the way a bit packed run of the hybrid encoding writes
// them, which is least significant bit first with each value carrying on where
// the last one stopped.
//
// This is here so that the tests can cover every width from one to thirty two
// without thirty two tables of bytes written out by hand. It is the decoder's
// mirror image and so proves nothing on its own, which is what the vectors out
// of the format's own document and the levels out of the real files are for.
func pack(values []int32, width int) []byte {
	out := make([]byte, (len(values)*width+7)/8)
	for i, v := range values {
		for b := range width {
			if v>>uint(b)&1 == 1 {
				at := i*width + b
				out[at/8] |= 1 << uint(at%8)
			}
		}
	}
	return out
}

// packMSB writes values the way the deprecated encoding writes them, which is
// the same packing the other way up: the first value takes the top bits of the
// first byte rather than the bottom ones.
func packMSB(values []int32, width int) []byte {
	out := make([]byte, (len(values)*width+7)/8)
	for i, v := range values {
		for b := range width {
			if v>>uint(width-1-b)&1 == 1 {
				at := i*width + b
				out[at/8] |= 1 << uint(7-at%8)
			}
		}
	}
	return out
}

// packedRun writes a bit packed run, which is a count of groups of eight and
// then the packed values. The count of values has to divide by eight because
// that is the only length the format can write.
func packedRun(values []int32, width int) []byte {
	if len(values)%8 != 0 {
		panic("a packed run is written eight values at a time")
	}
	out := binary.AppendUvarint(nil, uint64(len(values)/8)<<1|1)
	return append(out, pack(values, width)...)
}

// repeatedRun writes a run of one value repeated, which is a count and the
// value in as many whole bytes as its width needs.
func repeatedRun(value int32, count, width int) []byte {
	out := binary.AppendUvarint(nil, uint64(count)<<1)
	for i := range (width + 7) / 8 {
		out = append(out, byte(value>>uint(8*i)))
	}
	return out
}

// sequence is values to encode and decode again, long enough to run over
// several groups and to land values on both sides of a byte boundary.
func sequence(n, width int) []int32 {
	mask := int32(1)<<uint(width) - 1
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(i*7+i/3) & mask
	}
	return out
}

// readAll decodes until the data runs out, in batches of the size given so that
// the tests cover a run handed back in pieces as well as in one go.
func readAll(t *testing.T, d *parquet.RLEDecoder, batch int) []int32 {
	t.Helper()

	var out []int32
	buf := make([]int32, batch)
	for {
		n, err := d.Read(buf)
		out = append(out, buf[:n]...)
		if err != nil {
			if !errors.Is(err, io.EOF) {
				t.Fatalf("Read: %v", err)
			}
			return out
		}
	}
}

// TestRLESpec decodes the examples in the format's own encodings document,
// which is the only ground truth for this that is not somebody's reader.
func TestRLESpec(t *testing.T) {
	cases := []struct {
		name  string
		data  []byte
		width int
		want  []int32
	}{
		{
			// One bit packed run of one group. The three bytes are the ones the
			// document prints, and they are the eight values in order.
			name:  "a packed group of three bit values",
			data:  []byte{0x03, 0x88, 0xc6, 0xfa},
			width: 3,
			want:  []int32{0, 1, 2, 3, 4, 5, 6, 7},
		},
		{
			// The document's other example, which is eight values of one, and
			// which a writer sends as a repeat rather than a group because a
			// repeat of eight ones is two bytes and a group of them is four.
			name:  "a repeat of eight ones",
			data:  []byte{0x10, 0x01},
			width: 3,
			want:  []int32{1, 1, 1, 1, 1, 1, 1, 1},
		},
		{
			name:  "a repeat and then a group",
			data:  []byte{0x10, 0x01, 0x03, 0x88, 0xc6, 0xfa},
			width: 3,
			want:  []int32{1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 3, 4, 5, 6, 7},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			for _, batch := range []int{1, 3, 8, 64} {
				d, err := parquet.NewRLEDecoder(c.data, c.width)
				if err != nil {
					t.Fatalf("NewRLEDecoder: %v", err)
				}
				if got := readAll(t, d, batch); !slices.Equal(got, c.want) {
					t.Errorf("read %d at a time: got %v, want %v", batch, got, c.want)
				}
			}
		})
	}
}

// TestBitPackedSpec decodes the example the same document gives for the
// deprecated encoding. The values are the ones the other example holds and the
// bytes are different, which is the whole point: the same numbers packed the
// other way up are not the same bytes.
func TestBitPackedSpec(t *testing.T) {
	d, err := parquet.NewBitPackedDecoder([]byte{0x05, 0x39, 0x77}, 3)
	if err != nil {
		t.Fatalf("NewBitPackedDecoder: %v", err)
	}

	got := make([]int32, 8)
	n, err := d.Read(got)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	want := []int32{0, 1, 2, 3, 4, 5, 6, 7}
	if !slices.Equal(got[:n], want) {
		t.Errorf("got %v, want %v", got[:n], want)
	}
	if _, err := d.Read(got); !errors.Is(err, io.EOF) {
		t.Errorf("reading past the end: got %v, want %v", err, io.EOF)
	}
}

// TestRLEWidths decodes both kinds of run at every width the format allows.
//
// The interesting ones are the widths that are not a whole number of bytes,
// where a value starts in the middle of one byte and ends in another, and
// thirty two, which is as wide as the format goes and is the case the decoder
// reads eight bytes to get at.
func TestRLEWidths(t *testing.T) {
	for width := range maxWidthTested + 1 {
		t.Run(widthName(width), func(t *testing.T) {
			want := sequence(64, width)
			data := packedRun(want, width)
			data = append(data, repeatedRun(want[0], 40, width)...)
			want = append(want, slices.Repeat([]int32{want[0]}, 40)...)

			d, err := parquet.NewRLEDecoder(data, width)
			if err != nil {
				t.Fatalf("NewRLEDecoder: %v", err)
			}
			if got := readAll(t, d, 7); !slices.Equal(got, want) {
				t.Errorf("got %v, want %v", got, want)
			}
		})
	}
}

// TestBitPackedWidths does the same for the deprecated encoding, which has no
// runs and so is only the packing.
func TestBitPackedWidths(t *testing.T) {
	for width := 1; width <= maxWidthTested; width++ {
		t.Run(widthName(width), func(t *testing.T) {
			want := sequence(50, width)
			d, err := parquet.NewBitPackedDecoder(packMSB(want, width), width)
			if err != nil {
				t.Fatalf("NewBitPackedDecoder: %v", err)
			}

			got := make([]int32, len(want)+8)
			n, err := d.Read(got)
			if err != nil {
				t.Fatalf("Read: %v", err)
			}

			// The values are packed up to a byte boundary, so the last byte may
			// hold values nobody wrote. A reader keeps the ones it asked for and
			// this checks the ones it asked for are there.
			if n < len(want) {
				t.Fatalf("got %d values, want at least %d", n, len(want))
			}
			if !slices.Equal(got[:len(want)], want) {
				t.Errorf("got %v, want %v", got[:len(want)], want)
			}
		})
	}
}

// TestRLERuns covers the shapes a run comes in that the examples do not.
func TestRLERuns(t *testing.T) {
	cases := []struct {
		name  string
		data  []byte
		width int
		want  []int32
	}{
		{
			name:  "no data at all",
			data:  nil,
			width: 4,
			want:  nil,
		},
		{
			// A width of nought is every level of a required column. The runs
			// are still counted and the values are still there, and none of
			// them takes a bit.
			name:  "a repeat of nothing",
			data:  []byte{0x0a},
			width: 0,
			want:  []int32{0, 0, 0, 0, 0},
		},
		{
			name:  "a group of nothing",
			data:  []byte{0x03},
			width: 0,
			want:  []int32{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			// A run of no values is legal and says nothing. It turns up in
			// front of a run a writer decided against after writing the header
			// of the last one.
			name:  "an empty repeat before a real run",
			data:  []byte{0x00, 0x01, 0x04, 0x07},
			width: 8,
			want:  []int32{7, 7},
		},
		{
			name:  "an empty group before a real run",
			data:  []byte{0x01, 0x04, 0x07},
			width: 8,
			want:  []int32{7, 7},
		},
		{
			// A repeat of nothing still writes the value it is not repeating,
			// so this is only the whole of the data at a width of nought.
			name:  "an empty run and nothing after it",
			data:  []byte{0x00},
			width: 0,
			want:  nil,
		},
		{
			// The value of a repeat takes as many whole bytes as its width
			// needs, so a width of ten is two bytes and the top six bits of the
			// second one are not part of it.
			name:  "a repeat of a value wider than a byte",
			data:  []byte{0x06, 0xff, 0x03},
			width: 10,
			want:  []int32{1023, 1023, 1023},
		},
		{
			name:  "a repeat of the widest value there is",
			data:  []byte{0x04, 0xff, 0xff, 0xff, 0x7f},
			width: 32,
			want:  []int32{0x7fffffff, 0x7fffffff},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, err := parquet.NewRLEDecoder(c.data, c.width)
			if err != nil {
				t.Fatalf("NewRLEDecoder: %v", err)
			}
			if got := readAll(t, d, 4); !slices.Equal(got, c.want) {
				t.Errorf("got %v, want %v", got, c.want)
			}
		})
	}
}

// TestRLERefused is the bad data. Every one of these is a number in the file
// saying there is more of it than there is, which is what the decoder is
// bounds checked against.
func TestRLERefused(t *testing.T) {
	cases := []struct {
		name  string
		data  []byte
		width int
	}{
		{
			name:  "a run header that never ends",
			data:  []byte{0x80},
			width: 4,
		},
		{
			name:  "a run header that goes on for ever",
			data:  bytes.Repeat([]byte{0xff}, 10),
			width: 4,
		},
		{
			// Ten bytes is as long as a varint gets, and the tenth one here
			// carries bits above the sixty four there is room for.
			name:  "a run header of a number too big to be a number",
			data:  append(bytes.Repeat([]byte{0xff}, 9), 0x02),
			width: 4,
		},
		{
			name:  "a repeat of more values than a page can hold",
			data:  append(binary.AppendUvarint(nil, 1<<32), 0x01),
			width: 4,
		},
		{
			name:  "a group of more values than a page can hold",
			data:  binary.AppendUvarint(nil, 1<<32|1),
			width: 4,
		},
		{
			name:  "a repeat whose value is not there",
			data:  []byte{0x04},
			width: 8,
		},
		{
			name:  "a repeat whose value is half there",
			data:  []byte{0x04, 0xff},
			width: 16,
		},
		{
			name:  "a repeated value that does not fit its width",
			data:  []byte{0x04, 0xff},
			width: 4,
		},
		{
			name:  "a group of more bytes than there are",
			data:  []byte{0x05, 0x01, 0x02},
			width: 4,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, err := parquet.NewRLEDecoder(c.data, c.width)
			if err != nil {
				t.Fatalf("NewRLEDecoder: %v", err)
			}

			got := make([]int32, 64)
			if _, err := d.Read(got); !errors.Is(err, parquet.ErrFormat) {
				t.Fatalf("got %v, want %v", err, parquet.ErrFormat)
			}
		})
	}
}

// TestRLEPartial is a run that goes bad after some values have been decoded.
// The values are still handed back, because a caller counting what it got is
// entitled to the count and the error says the rest is not coming.
func TestRLEPartial(t *testing.T) {
	data := append(repeatedRun(9, 3, 8), 0x05, 0x01)
	d, err := parquet.NewRLEDecoder(data, 8)
	if err != nil {
		t.Fatalf("NewRLEDecoder: %v", err)
	}

	got := make([]int32, 16)
	n, err := d.Read(got)
	if !errors.Is(err, parquet.ErrFormat) {
		t.Fatalf("got %v, want %v", err, parquet.ErrFormat)
	}
	if want := []int32{9, 9, 9}; !slices.Equal(got[:n], want) {
		t.Errorf("got %v, want %v", got[:n], want)
	}
}

// TestRLEWidthRefused is a width no value can be written in. Thirty two bits is
// as wide as the format goes, and a width read out of a page is a byte of
// somebody else's file like everything else here.
func TestRLEWidthRefused(t *testing.T) {
	for _, width := range []int{-1, 33, 64, 255} {
		if _, err := parquet.NewRLEDecoder(nil, width); !errors.Is(err, parquet.ErrFormat) {
			t.Errorf("NewRLEDecoder at %d bits: got %v, want %v", width, err, parquet.ErrFormat)
		}
		if _, err := parquet.NewBitPackedDecoder(nil, width); !errors.Is(err, parquet.ErrFormat) {
			t.Errorf("NewBitPackedDecoder at %d bits: got %v, want %v", width, err, parquet.ErrFormat)
		}
	}
}

// TestRLEEnd is what the decoders say when there is nothing left, which is what
// a caller reading a page in batches stops on.
func TestRLEEnd(t *testing.T) {
	d, err := parquet.NewRLEDecoder(repeatedRun(1, 2, 8), 8)
	if err != nil {
		t.Fatalf("NewRLEDecoder: %v", err)
	}

	got := make([]int32, 4)
	n, err := d.Read(got)
	if n != 2 || err != nil {
		t.Fatalf("got %d values and %v, want 2 and no error", n, err)
	}
	if n, err = d.Read(got); n != 0 || !errors.Is(err, io.EOF) {
		t.Fatalf("reading past the end: got %d values and %v, want 0 and %v", n, err, io.EOF)
	}

	// Reading nothing asks for nothing, the way an io.Reader handed an empty
	// buffer does, and says nothing about whether there is anything left.
	if n, err = d.Read(nil); n != 0 || err != nil {
		t.Fatalf("reading nothing: got %d values and %v, want 0 and no error", n, err)
	}

	b, err := parquet.NewBitPackedDecoder(nil, 4)
	if err != nil {
		t.Fatalf("NewBitPackedDecoder: %v", err)
	}
	if n, err = b.Read(got); n != 0 || !errors.Is(err, io.EOF) {
		t.Fatalf("packed levels of no bytes: got %d values and %v, want 0 and %v", n, err, io.EOF)
	}
}

// TestBitPackedNoWidth is a value of no bits, which the deprecated encoding
// cannot count and so does not read. Levels that wide belong to a required
// column, which writes none.
func TestBitPackedNoWidth(t *testing.T) {
	d, err := parquet.NewBitPackedDecoder([]byte{0xff, 0xff}, 0)
	if err != nil {
		t.Fatalf("NewBitPackedDecoder: %v", err)
	}
	n, err := d.Read(make([]int32, 8))
	if n != 0 || !errors.Is(err, io.EOF) {
		t.Fatalf("got %d values and %v, want 0 and %v", n, err, io.EOF)
	}
}

// TestRLEReset points a decoder at other bytes, which is what a scan does
// rather than allocating one per page.
func TestRLEReset(t *testing.T) {
	d, err := parquet.NewRLEDecoder(repeatedRun(1, 100, 8), 8)
	if err != nil {
		t.Fatalf("NewRLEDecoder: %v", err)
	}

	got := make([]int32, 3)
	if _, err = d.Read(got); err != nil {
		t.Fatalf("Read: %v", err)
	}

	// The reset is in the middle of a run, which is the state that has to go
	// away: a decoder that kept the run it was in would hand back ones.
	if err = d.Reset(repeatedRun(2, 4, 8), 8); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	after := readAll(t, d, 8)
	if want := []int32{2, 2, 2, 2}; !slices.Equal(after, want) {
		t.Errorf("after a reset: got %v, want %v", after, want)
	}
	if err = d.Reset(nil, 33); !errors.Is(err, parquet.ErrFormat) {
		t.Errorf("Reset at 33 bits: got %v, want %v", err, parquet.ErrFormat)
	}

	b, err := parquet.NewBitPackedDecoder(packMSB([]int32{1, 2}, 8), 8)
	if err != nil {
		t.Fatalf("NewBitPackedDecoder: %v", err)
	}
	if _, err = b.Read(got[:1]); err != nil {
		t.Fatalf("Read: %v", err)
	}
	if err = b.Reset(packMSB([]int32{3, 4}, 8), 8); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	n, err := b.Read(got)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if want := []int32{3, 4}; !slices.Equal(got[:n], want) {
		t.Errorf("after a reset: got %v, want %v", got[:n], want)
	}
	if err := b.Reset(nil, -1); !errors.Is(err, parquet.ErrFormat) {
		t.Errorf("Reset at -1 bits: got %v, want %v", err, parquet.ErrFormat)
	}
}

// pagesOf reads every page of one column of one of the files in testdata,
// together with the schema leaf the column is, which is where the width of its
// levels comes from.
func pagesOf(t *testing.T, name, column string) ([]parquet.Page, parquet.Column) {
	t.Helper()

	b := bytesOf(t, name)
	m, err := parquet.ReadMetadata(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatalf("ReadMetadata(%s): %v", name, err)
	}

	columns, err := m.Columns()
	if err != nil {
		t.Fatalf("Columns: %v", err)
	}
	i := slices.IndexFunc(columns, func(c parquet.Column) bool { return c.Name() == column })
	if i < 0 {
		t.Fatalf("%s has no column called %s", name, column)
	}

	var out []parquet.Page
	for g := range m.RowGroups {
		for c := range m.RowGroups[g].Columns {
			chunk := &m.RowGroups[g].Columns[c]
			if strings.Join(chunk.Meta.Path, ".") != column {
				continue
			}
			pages, err := parquet.ReadPages(bytes.NewReader(b), int64(len(b)), chunk)
			if err != nil {
				t.Fatalf("ReadPages: %v", err)
			}
			for {
				p, err := pages.Next()
				if err != nil {
					break
				}
				out = append(out, p)
			}
		}
	}
	if len(out) == 0 {
		t.Fatalf("%s has no pages for %s", name, column)
	}
	return out, columns[i]
}

// levels decodes as many levels as a page holds values, at the width the
// column's deepest level needs.
func levels(t *testing.T, data []byte, c parquet.Column, values int32) []int32 {
	t.Helper()

	d, err := parquet.NewRLEDecoder(data, bits.Len(uint(c.MaxDefinition)))
	if err != nil {
		t.Fatalf("NewRLEDecoder: %v", err)
	}

	out := make([]int32, values)
	n, err := d.Read(out)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if int32(n) != values {
		t.Fatalf("a page of %d values has %d levels", values, n)
	}
	return out
}

// TestRLERealLevels decodes the definition levels of a real file.
//
// The file is written by pyarrow and the column in it is null wherever the row
// number divides by three, which is what the script next to it wrote and is
// what the levels have to say. The page also says how many nulls it holds, so
// there are two independent things to be checked against and neither of them is
// this package.
//
// The levels of a page written the second way are their own bytes at the front
// of the body, as long as the header said and outside whatever compression the
// values went through.
func TestRLERealLevels(t *testing.T) {
	pages, column := pagesOf(t, "pages.parquet", "maybe")

	var got []int32
	for i, p := range pages {
		if p.Kind != parquet.DataPageV2 {
			t.Fatalf("page %d is a %s", i, p.Kind)
		}
		if p.RepetitionLength != 0 {
			t.Fatalf("a flat column with %d bytes of repetition levels", p.RepetitionLength)
		}

		read := levels(t, p.Data[:p.DefinitionLength], column, p.NumValues)
		nulls := int32(0)
		for _, l := range read {
			if l == 0 {
				nulls++
			}
		}
		if nulls != p.NumNulls {
			t.Errorf("page %d holds %d levels of nought and says %d nulls", i, nulls, p.NumNulls)
		}
		got = append(got, read...)
	}

	if len(got) != 500 {
		t.Fatalf("got %d levels, want 500", len(got))
	}
	for i, l := range got {
		want := int32(1)
		if i%3 == 0 {
			want = 0
		}
		if l != want {
			t.Fatalf("row %d has level %d, want %d", i, l, want)
		}
	}
}

// TestRLERealIndices decodes the dictionary indices of a real file.
//
// They sit behind the levels in the same page and they are the other thing this
// encoding is used for. The width is not in the schema this time: it is the
// byte in front of the indices, which is how a page says how many bits the
// largest index in it needed.
//
// The column repeats one of four words in order, so the indices have to repeat
// with a period of four whatever the dictionary happens to be in. That is
// checked rather than the indices themselves, since which word got which index
// is the writer's business.
func TestRLERealIndices(t *testing.T) {
	pages, _ := pagesOf(t, "pages.parquet", "word")

	if len(pages) != 2 || pages[0].Kind != parquet.DictionaryPage {
		t.Fatalf("want a dictionary page and a data page, got %d pages starting with a %s",
			len(pages), pages[0].Kind)
	}
	dictionary, page := pages[0], pages[1]
	if page.Encoding != parquet.RLEDictionary {
		t.Fatalf("the data page is %s, want %s", page.Encoding, parquet.RLEDictionary)
	}

	// The levels come first and the width of the indices is the byte after
	// them, then the indices run to the end of the page.
	body := page.Data[page.RepetitionLength+page.DefinitionLength:]
	d, err := parquet.NewRLEDecoder(body[1:], int(body[0]))
	if err != nil {
		t.Fatalf("NewRLEDecoder at %d bits: %v", body[0], err)
	}

	got := make([]int32, page.NumValues)
	if n, err := d.Read(got); err != nil || int32(n) != page.NumValues {
		t.Fatalf("got %d indices and %v, want %d and no error", n, err, page.NumValues)
	}

	for i, v := range got {
		if v < 0 || v >= dictionary.NumValues {
			t.Fatalf("row %d points at entry %d of a dictionary of %d",
				i, v, dictionary.NumValues)
		}
		if v != got[i%4] {
			t.Fatalf("row %d points at entry %d and row %d at %d, and they are the same word",
				i, v, i%4, got[i%4])
		}
	}
	if seen := len(slices.Compact(slices.Sorted(slices.Values(got)))); seen != 4 {
		t.Errorf("four words came back as %d entries", seen)
	}
}

// TestRLERealLevelsV1 decodes the levels of a page written the first way, where
// they are inside the body with four bytes of length in front of them rather
// than in bytes of their own.
//
// The column has three values and the middle one is missing, so its levels are
// one, nought, one. Three values is less than the eight a group holds, which is
// why the page has five levels in it that are not levels: a writer packs to the
// end of a group and a reader keeps as many as the page said it held.
func TestRLERealLevelsV1(t *testing.T) {
	pages, column := pagesOf(t, "alltypes.parquet", "small")

	page := pages[len(pages)-1]
	if page.Kind != parquet.DataPage {
		t.Fatalf("the last page is a %s, want a %s", page.Kind, parquet.DataPage)
	}

	n := binary.LittleEndian.Uint32(page.Data)
	got := levels(t, page.Data[4:4+n], column, page.NumValues)
	if want := []int32{1, 0, 1}; !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// BenchmarkRLE decodes a page of levels of each shape.
//
// The two runs are the two things this encoding is for. A column with no nulls
// in it is one repeat of a hundred thousand values, which is the case a decoder
// should not be reading a bit at a time, and levels that alternate are packed,
// which is the case it has to.
func BenchmarkRLE(b *testing.B) {
	const values = 100000

	packed := sequence(values, 3)
	cases := []struct {
		name  string
		data  []byte
		width int
	}{
		{name: "repeated", data: repeatedRun(1, values, 1), width: 1},
		{name: "packed", data: packedRun(packed, 3), width: 3},
		{name: "packed at a byte", data: packedRun(sequence(values, 8), 8), width: 8},
		{name: "packed at the widest", data: packedRun(sequence(values, 32), 32), width: 32},
	}

	dst := make([]int32, values)
	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			d, err := parquet.NewRLEDecoder(c.data, c.width)
			if err != nil {
				b.Fatalf("NewRLEDecoder: %v", err)
			}

			b.SetBytes(int64(len(c.data)))
			b.ReportAllocs()
			for b.Loop() {
				if err := d.Reset(c.data, c.width); err != nil {
					b.Fatalf("Reset: %v", err)
				}
				if _, err := d.Read(dst); err != nil {
					b.Fatalf("Read: %v", err)
				}
			}
		})
	}

	b.Run("deprecated", func(b *testing.B) {
		data := packMSB(packed, 3)
		d, err := parquet.NewBitPackedDecoder(data, 3)
		if err != nil {
			b.Fatalf("NewBitPackedDecoder: %v", err)
		}

		b.SetBytes(int64(len(data)))
		b.ReportAllocs()
		for b.Loop() {
			if err := d.Reset(data, 3); err != nil {
				b.Fatalf("Reset: %v", err)
			}
			if _, err := d.Read(dst); err != nil {
				b.Fatalf("Read: %v", err)
			}
		}
	})
}
