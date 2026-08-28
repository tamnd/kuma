package parquet_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/parquet"
)

// TestReadColumnDictionary reads the columns of a real file that are written as
// indices into a dictionary, which is nearly every column of nearly every file.
//
// The three of these are the three widths an index comes at. Three hundred
// distinct values is nine bits, so a value is read out of two bytes and the run
// is packed; four is two bits; and one distinct value is nought bits, which is
// no bytes of indices at all and a page that is only a run header.
func TestReadColumnDictionary(t *testing.T) {
	t.Run("wide", func(t *testing.T) {
		a := readColumn(t, "dictionary.parquet", "code")
		dictionaryShape(t, a, dtype.String, 1000, 300)

		for i := range a.Len() {
			want := fmt.Sprintf("c%03d", i%300)
			if got := string(a.Dictionary().Bytes(a.Index(i))); got != want {
				t.Fatalf("value %d: got %q, want %q", i, got, want)
			}
		}
	})

	// The same file's other column is the two halves together: an index is only
	// written for a row that has a value, so a page of five hundred rows holds
	// four hundred and twenty nine indices and the levels are the only thing
	// that says which rows they belong to.
	t.Run("with nulls", func(t *testing.T) {
		a := readColumn(t, "dictionary.parquet", "size")
		dictionaryShape(t, a, dtype.Int64, 1000, 300)

		if a.NullCount() != 143 {
			t.Fatalf("got %d nulls, want 143", a.NullCount())
		}
		values := a.Dictionary().Values[int64]()
		for i := range a.Len() {
			if i%7 == 0 {
				if !a.IsNull(i) {
					t.Fatalf("value %d is there and should be missing", i)
				}
				continue
			}
			if got := values[a.Index(i)]; got != int64(i%300) {
				t.Fatalf("value %d: got %d, want %d", i, got, i%300)
			}
		}
	})

	t.Run("narrow", func(t *testing.T) {
		a := readColumn(t, "pages.parquet", "word")
		dictionaryShape(t, a, dtype.String, 500, 4)

		words := []string{"alpha", "beta", "gamma", "delta"}
		for i := range a.Len() {
			if got := string(a.Dictionary().Bytes(a.Index(i))); got != words[i%4] {
				t.Fatalf("value %d: got %q, want %q", i, got, words[i%4])
			}
		}
	})

	// A column of one distinct value, whose indices are all nought and are
	// written at no bits each. The dictionary page in front of it is one value
	// and the data page behind it is a run header and nothing else.
	t.Run("of one value", func(t *testing.T) {
		a := readColumn(t, "alltypes.parquet", "day")
		dictionaryShape(t, a, dtype.Date32, 3, 1)

		for i := range a.Len() {
			if got := a.Index(i); got != 0 {
				t.Errorf("value %d indexes %d", i, got)
			}
		}
		if got := a.Dictionary().Value[int32](0); got != 20693 {
			t.Errorf("the one value is %d", got)
		}
	})
}

// dictionaryShape checks the shape of a dictionary encoded column, which is a
// column of indices and an array of values it shares with nobody here.
func dictionaryShape(t *testing.T, a *array.Array, value dtype.DataType, length, entries int) {
	t.Helper()

	want := dtype.Dictionary{Index: dtype.Int32, Value: value}
	if a.DType() != want {
		t.Fatalf("the column is a %s, want a %s", a.DType(), want)
	}
	if a.Len() != length {
		t.Fatalf("got %d values, want %d", a.Len(), length)
	}
	if a.Dictionary().Len() != entries {
		t.Fatalf("got a dictionary of %d, want one of %d", a.Dictionary().Len(), entries)
	}
}

// TestReadColumnDictionaryTypes reads a dictionary of every type a file writes
// one of.
//
// A dictionary page is the distinct values written plainly, so it goes through
// the same decoder the values of a plain page do and everything the schema
// decided about the column holds for it. The narrow integers are the ones worth
// having here: parquet writes an int8 dictionary as int32 like it writes an int8
// column, so the values are narrowed on the way into the dictionary rather than
// on the way out of it.
func TestReadColumnDictionaryTypes(t *testing.T) {
	const file = "alltypes.parquet"

	t.Run("small", func(t *testing.T) {
		a := readColumn(t, file, "small")
		if a.DType() != (dtype.Dictionary{Index: dtype.Int32, Value: dtype.Int8}) {
			t.Fatalf("the column is a %s", a.DType())
		}
		if a.NullCount() != 1 || !a.IsNull(1) {
			t.Fatalf("%d nulls and the second is null: %v", a.NullCount(), a.IsNull(1))
		}
		for i, want := range map[int]int8{0: 1, 2: -3} {
			if got := a.Dictionary().Value[int8](a.Index(i)); got != want {
				t.Errorf("value %d: got %d, want %d", i, got, want)
			}
		}
	})

	t.Run("name", func(t *testing.T) {
		a := readColumn(t, file, "name")
		if a.DType() != (dtype.Dictionary{Index: dtype.Int32, Value: dtype.String}) {
			t.Fatalf("the column is a %s", a.DType())
		}
		if a.NullCount() != 1 || !a.IsNull(2) {
			t.Fatalf("%d nulls and the third is null: %v", a.NullCount(), a.IsNull(2))
		}
		for i, want := range map[int]string{0: "GB", 1: "JP"} {
			if got := string(a.Dictionary().Bytes(a.Index(i))); got != want {
				t.Errorf("value %d: got %q, want %q", i, got, want)
			}
		}
	})

	t.Run("fixed", func(t *testing.T) {
		a := readColumn(t, file, "fixed")
		if got := string(a.Dictionary().Bytes(a.Index(1))); got != "efgh" {
			t.Errorf("got %q, want %q", got, "efgh")
		}
	})

	t.Run("weight", func(t *testing.T) {
		a := readColumn(t, file, "weight")
		if got := a.Dictionary().Value[float64](a.Index(2)); got != 3.75 {
			t.Errorf("got %v, want 3.75", got)
		}
	})
}

// TestReadColumnEmptyDictionary reads a chunk that is a dictionary of nothing
// and no data pages at all, which is a column of a file with no rows.
//
// pyarrow writes one for every column of an empty table. It is still dictionary
// encoded, so what comes back is a dictionary of no values indexed by no rows
// rather than a column that was never encoded.
func TestReadColumnEmptyDictionary(t *testing.T) {
	a := readColumn(t, "empty.parquet", "label")
	dictionaryShape(t, a, dtype.String, 0, 0)
	if a.NullCount() != 0 {
		t.Errorf("%d nulls in a column of nothing", a.NullCount())
	}
}

// TestReadColumnFallback refuses a chunk that starts as indices into a
// dictionary and gives up on it half way through.
//
// A writer keeps a dictionary while the distinct values fit in a page of their
// own and writes the rest of the chunk plainly once they do not, which leaves
// one chunk in two shapes. Reading it would mean expanding the dictionary into
// the column, and not doing that is the whole reason to read a dictionary this
// way, so the chunk is refused until there is something better to do with it.
func TestReadColumnFallback(t *testing.T) {
	b, chunk, column := chunkOf(t, "fallback.parquet", "code")

	_, err := parquet.ReadColumn(bytes.NewReader(b), int64(len(b)), chunk, column)
	if !errors.Is(err, parquet.ErrUnsupported) {
		t.Errorf("got %v, want %v", err, parquet.ErrUnsupported)
	}
}

// dictPage builds a dictionary page of int32 values written plainly.
func dictPage(values ...int32) parquet.Page {
	var body []byte
	for _, v := range values {
		body = binary.LittleEndian.AppendUint32(body, uint32(v))
	}
	return parquet.Page{
		PageHeader: parquet.PageHeader{
			Kind: parquet.DictionaryPage, Encoding: parquet.Plain,
			NumValues: int32(len(values)),
		},
		Data: body,
	}
}

// indexRun writes up to eight indices the way the values of a dictionary
// encoded page are written: the width in a byte of its own and then one packed
// run of a group of eight.
//
// A byte an index is wider than anything here needs and is the one width whose
// packing can be read off the page by eye. A run of eight is what the encoding
// packs in whether eight are wanted or not, and the ones past the end of the
// page are never asked for.
func indexRun(indices ...int32) []byte {
	var eight [8]byte
	for i, v := range indices {
		eight[i] = byte(v)
	}
	return append([]byte{8, 0x03}, eight[:]...)
}

// indexPage builds a data page of dictionary indices, which is levels behind
// four bytes of their length and then the indices.
func indexPage(levels []byte, values int32, indices ...int32) parquet.Page {
	body := binary.LittleEndian.AppendUint32(nil, uint32(len(levels)))
	body = append(body, levels...)
	body = append(body, indexRun(indices...)...)
	return parquet.Page{
		PageHeader: parquet.PageHeader{
			Kind: parquet.DataPage, Encoding: parquet.RLEDictionary,
			DefinitionEncoding: parquet.RLE, NumValues: values,
		},
		Data: body,
	}
}

// TestColumnReaderDictionaryPages reads pages built by hand, which is how the
// shapes no file here has are read.
func TestColumnReaderDictionaryPages(t *testing.T) {
	// A page in which every row is missing holds no indices, and pyarrow writes
	// nothing at all behind the levels of one rather than a width and an empty
	// run.
	t.Run("a page of nothing but nulls", func(t *testing.T) {
		r := readerOf(t, optional())
		if err := r.Page(dictPage(10, 20)); err != nil {
			t.Fatalf("Page: %v", err)
		}

		body := binary.LittleEndian.AppendUint32(nil, 2)
		page := indexPage(nil, 3)
		page.Data = append(body, 0x06, 0x00)
		if err := r.Page(page); err != nil {
			t.Fatalf("Page: %v", err)
		}

		a := finish(t, r)
		if a.Len() != 3 || a.NullCount() != 3 {
			t.Fatalf("%d values, %d null", a.Len(), a.NullCount())
		}
	})

	// The names the format used before it moved them. A file written by an old
	// writer says plain_dictionary on both the dictionary page and the data
	// pages behind it, and means the same two things by it.
	t.Run("the old name for the encoding", func(t *testing.T) {
		r := readerOf(t, optional())

		dict := dictPage(10, 20)
		dict.Encoding = parquet.PlainDictionary
		if err := r.Page(dict); err != nil {
			t.Fatalf("Page: %v", err)
		}

		page := indexPage([]byte{0x04, 0x01}, 2, 1, 0)
		page.Encoding = parquet.PlainDictionary
		if err := r.Page(page); err != nil {
			t.Fatalf("Page: %v", err)
		}

		a := finish(t, r)
		if a.Len() != 2 || a.Index(0) != 1 || a.Index(1) != 0 {
			t.Fatalf("%d values indexing %d and %d", a.Len(), a.Index(0), a.Index(1))
		}
	})

	// A required column, whose pages hold no levels at all, so the indices
	// start at the first byte of the body.
	t.Run("a required column", func(t *testing.T) {
		required := optional()
		required.Element.Repetition = parquet.Required
		required.MaxDefinition = 0

		r := readerOf(t, required)
		if err := r.Page(dictPage(10, 20)); err != nil {
			t.Fatalf("Page: %v", err)
		}

		page := indexPage(nil, 2, 1, 1)
		page.Data = indexRun(1, 1)
		if err := r.Page(page); err != nil {
			t.Fatalf("Page: %v", err)
		}

		a := finish(t, r)
		if a.Len() != 2 || a.Index(0) != 1 || a.Index(1) != 1 {
			t.Fatalf("%d values indexing %d and %d", a.Len(), a.Index(0), a.Index(1))
		}
	})
}

// TestColumnReaderRefusedDictionary is the dictionary pages and the pages of
// indices a reader has to refuse.
//
// Each of them is bytes that could be read into a column of the wrong values
// rather than into an error. An index that is not in the dictionary is the one
// that would get past a reader and turn into a read out of range a long way
// from here.
func TestColumnReaderRefusedDictionary(t *testing.T) {
	negative := dictPage(10, 20)
	negative.NumValues = -1

	short := dictPage(10, 20)
	short.NumValues = 4

	delta := dictPage(10, 20)
	delta.Encoding = parquet.DeltaByteArray

	wide := indexPage([]byte{0x04, 0x01}, 2, 0, 1)
	wide.Data[6] = 33

	cases := []struct {
		name  string
		pages []parquet.Page
		want  error
	}{
		{
			name:  "two dictionary pages",
			pages: []parquet.Page{dictPage(10, 20), dictPage(30)},
			want:  parquet.ErrFormat,
		},
		{
			name:  "a dictionary page behind the data pages",
			pages: []parquet.Page{pageV1([]byte{0x04, 0x01}, 1, 2), dictPage(10)},
			want:  parquet.ErrFormat,
		},
		{
			name:  "a dictionary page in an encoding that is not one",
			pages: []parquet.Page{delta},
			want:  parquet.ErrUnsupported,
		},
		{
			name:  "a dictionary of fewer values than nothing",
			pages: []parquet.Page{negative},
			want:  parquet.ErrFormat,
		},
		{
			name:  "a dictionary shorter than it says",
			pages: []parquet.Page{short},
			want:  parquet.ErrFormat,
		},
		{
			name:  "indices with no dictionary in front of them",
			pages: []parquet.Page{indexPage([]byte{0x04, 0x01}, 2, 0, 1)},
			want:  parquet.ErrFormat,
		},
		{
			name:  "indices at a width that is not one",
			pages: []parquet.Page{dictPage(10, 20), wide},
			want:  parquet.ErrFormat,
		},
		{
			// Two levels saying both rows are there, and no bytes behind them
			// for the indices to come out of.
			name: "a page that wants indices and has no bytes",
			pages: []parquet.Page{dictPage(10, 20), {
				PageHeader: parquet.PageHeader{
					Kind: parquet.DataPage, Encoding: parquet.RLEDictionary,
					DefinitionEncoding: parquet.RLE, NumValues: 2,
				},
				Data: []byte{2, 0, 0, 0, 0x04, 0x01},
			}},
			want: parquet.ErrFormat,
		},
		{
			// A page saying it holds ten rows, all of them present, behind a
			// run of eight indices.
			name:  "fewer indices than the levels want",
			pages: []parquet.Page{dictPage(10, 20), indexPage([]byte{0x14, 0x01}, 10, 0)},
			want:  parquet.ErrFormat,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := readerOf(t, optional())

			var err error
			for _, p := range c.pages {
				if err = r.Page(p); err != nil {
					break
				}
			}
			if !errors.Is(err, c.want) {
				t.Errorf("got %v, want %v", err, c.want)
			}
		})
	}
}

// TestColumnReaderDictionaryIndex refuses a page holding an index the
// dictionary in front of it has no value at.
//
// It is the one thing wrong with a dictionary encoded chunk that reads as a
// column rather than as an error, since an index is a number like any other and
// there is nothing in the page that says which numbers are ones. Handing it back
// would put a read out of range in whatever touched the column next.
func TestColumnReaderDictionaryIndex(t *testing.T) {
	r := readerOf(t, optional())

	if err := r.Page(dictPage(10, 20)); err != nil {
		t.Fatalf("Page: %v", err)
	}
	if err := r.Page(indexPage([]byte{0x04, 0x01}, 2, 0, 2)); err != nil {
		t.Fatalf("Page: %v", err)
	}

	if _, err := r.Finish(); !errors.Is(err, parquet.ErrFormat) {
		t.Errorf("got %v, want %v", err, parquet.ErrFormat)
	}
}

// TestColumnReaderDictionaryReuse reads two chunks of one column with one
// reader, the first of them dictionary encoded and the second not.
//
// A dictionary belongs to the chunk it was written in rather than to the
// column, so a reader that kept one would read the second chunk's plain pages as
// indices into the first chunk's values. Which of the two a chunk is is not
// known until its first page has been read.
func TestColumnReaderDictionaryReuse(t *testing.T) {
	r := readerOf(t, optional())

	if err := r.Page(dictPage(10, 20)); err != nil {
		t.Fatalf("Page: %v", err)
	}
	if err := r.Page(indexPage([]byte{0x04, 0x01}, 2, 1, 0)); err != nil {
		t.Fatalf("Page: %v", err)
	}

	a := finish(t, r)
	if a.Dictionary() == nil {
		t.Fatalf("the first chunk came back a %s", a.DType())
	}
	if a.Index(0) != 1 || a.Index(1) != 0 {
		t.Errorf("the first chunk indexes %d and %d", a.Index(0), a.Index(1))
	}
	if r.Len() != 0 {
		t.Errorf("after finishing there are %d values left", r.Len())
	}

	if err := r.Page(pageV1([]byte{0x04, 0x01}, 30, 40)); err != nil {
		t.Fatalf("Page: %v", err)
	}
	b := finish(t, r)
	if b.Dictionary() != nil {
		t.Fatalf("the second chunk kept the first chunk's dictionary")
	}
	if b.Value[int32](0) != 30 || b.Value[int32](1) != 40 {
		t.Errorf("the second chunk is %d and %d", b.Value[int32](0), b.Value[int32](1))
	}
}

// BenchmarkReadColumnDictionary reads a dictionary encoded column out of a file.
//
// It is the same path BenchmarkReadColumn measures with the values read as
// indices, which is what nearly every column of nearly every real file is. The
// column is a thousand rows of three hundred distinct strings, so expanding it
// would copy each of them three times and this does not copy them at all.
func BenchmarkReadColumnDictionary(b *testing.B) {
	for _, column := range []string{"code", "size"} {
		b.Run(column, func(b *testing.B) {
			file, chunk, c := chunkOf(&testing.T{}, "dictionary.parquet", column)
			r := bytes.NewReader(file)

			b.SetBytes(chunk.Meta.TotalCompressedSize)
			b.ReportAllocs()
			for b.Loop() {
				if _, err := parquet.ReadColumn(r, int64(len(file)), chunk, c); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
