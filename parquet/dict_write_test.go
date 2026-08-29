package parquet_test

import (
	"bytes"
	"strconv"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/parquet"
)

// chunkPages is the pages of one column chunk of the first row group, which is
// where a test looks to find out how a chunk was written rather than what it
// holds.
func chunkPages(tb testing.TB, raw []byte, col int) []parquet.Page {
	tb.Helper()

	m, err := parquet.ReadMetadata(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		tb.Fatalf("ReadMetadata: %v", err)
	}

	// A chunk with a dictionary starts at it, since a reader that seeks to the
	// chunk has to read the dictionary before the indices behind it mean
	// anything.
	c := &m.RowGroups[0].Columns[col]
	if off := c.Meta.DictionaryPageOffset; off != 0 && c.Start() != off {
		tb.Fatalf("column %d starts at %d and its dictionary page is at %d",
			col, c.Start(), off)
	}

	pages, err := parquet.ReadPages(bytes.NewReader(raw), int64(len(raw)), c)
	if err != nil {
		tb.Fatalf("ReadPages(%d): %v", col, err)
	}

	var out []parquet.Page
	for {
		p, err := pages.Next()
		if err != nil {
			break
		}
		out = append(out, p)
	}
	if len(out) == 0 {
		tb.Fatalf("column %d came back with no pages", col)
	}
	return out
}

// dictChunk says whether a chunk was written as a dictionary, and fails the test
// if the pages of it do not agree with each other.
//
// The agreement is the whole point. A chunk that is a dictionary page followed
// by indices is one any reader takes, and a chunk that is a dictionary page
// followed by plain values is one the reader here refuses by name, so a test
// that looked only at the first page would pass on a file nothing can open.
func dictChunk(tb testing.TB, pages []parquet.Page) bool {
	tb.Helper()

	dict := pages[0].Kind == parquet.DictionaryPage
	want := parquet.Plain
	if dict {
		want = parquet.RLEDictionary
	}

	for i, p := range pages[1:] {
		if p.Kind != parquet.DataPage {
			tb.Fatalf("page %d is a %s, want a data page", i+1, p.Kind)
		}
		if p.Encoding != want {
			tb.Fatalf("page %d is %s, want %s", i+1, p.Encoding, want)
		}
	}
	return dict
}

// TestWriteDictPages writes a column of every type and asks which of them went
// down as a dictionary.
//
// Every type that has one gets one here, since the columns are four rows of a
// handful of values. The two that never get one are the point: a boolean has two
// values and a page holds one in a bit, so a dictionary of them is larger than
// the values, and a float cannot have one at all because a dictionary is a map
// and a negative zero, a positive zero and a NaN do not behave in one.
func TestWriteDictPages(t *testing.T) {
	want := everyType(t)
	_, raw := writtenBytes(t, want, nil)

	plain := map[string]bool{"flag": true, "ratio": true, "weight": true}
	for i, f := range want.Schema.Fields {
		if got := dictChunk(t, chunkPages(t, raw, i)); got == plain[f.Name] {
			t.Errorf("%s was written as a dictionary: %t", f.Name, got)
		}
	}
}

// TestWriteDictPlain checks that the plain option writes no dictionary at all.
//
// It is the file the writer wrote before there were dictionary pages in it, and
// it is worth keeping reachable: it is the largest file anything will open, it
// costs no hash of any value, and it is what a chunk falls back to anyway.
func TestWriteDictPlain(t *testing.T) {
	want := everyType(t)
	got, raw := writtenBytes(t, want, &parquet.WriteOptions{Plain: true})
	same(t, got, want)

	m, err := parquet.ReadMetadata(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}

	for i, f := range want.Schema.Fields {
		if off := m.RowGroups[0].Columns[i].Meta.DictionaryPageOffset; off != 0 {
			t.Errorf("%s says its dictionary page is at %d, want none", f.Name, off)
		}
		if dictChunk(t, chunkPages(t, raw, i)) {
			t.Errorf("%s was written as a dictionary and the file asked for none", f.Name)
		}
	}
}

// TestWriteDictEncoded checks that a column written as a dictionary comes back
// as one when the reader is asked for it.
//
// This is the pair of calls that makes the encoding worth writing. A column of
// country codes goes out as two hundred and fifty strings and a run of small
// integers, and it comes back the same way, so a group by on it never touches
// the strings at all.
func TestWriteDictEncoded(t *testing.T) {
	codes := []string{"gb", "us", "jp", "de"}
	want := tableOf(t,
		[]dtype.Field{{Name: "code", Type: dtype.String}},
		func(b *array.Builder) {
			for i := range 1000 {
				b.AppendString(codes[i%len(codes)])
			}
		},
	)

	var buf bytes.Buffer
	if _, err := parquet.Write(&buf, want, nil); err != nil {
		t.Fatalf("Write: %v", err)
	}

	got, err := parquet.Read(bytes.NewReader(buf.Bytes()), int64(buf.Len()),
		&parquet.Options{Dictionary: true})
	if err != nil {
		t.Fatalf("Read: %v", err)
	}

	a := got.Columns[0].Chunks()[0]
	if a.Dictionary() == nil {
		t.Fatal("the column came back expanded, want it dictionary encoded")
	}
	if n := a.Dictionary().Len(); n != len(codes) {
		t.Errorf("the dictionary holds %d values, want %d", n, len(codes))
	}

	// The values are compared through the encoding rather than the schemas,
	// since the column came back as a dictionary of strings and went out as
	// strings, which is the whole of what was asked for.
	g, w := cells(got.Columns[0]), cells(want.Columns[0])
	for row := range w {
		if g[row] != w[row] {
			t.Fatalf("row %d: got %s, want %s", row, g[row], w[row])
		}
	}
}

// TestWriteDictSingle checks a chunk with one distinct value in it.
//
// An index into a dictionary of one takes no bits at all, which is the right
// answer and the one that is easy to write as a bit by mistake. The page says
// how many values it has and every one of them is the same, so there is nothing
// left to write down.
func TestWriteDictSingle(t *testing.T) {
	want := tableOf(t,
		[]dtype.Field{{Name: "code", Type: dtype.String}},
		func(b *array.Builder) {
			for range 1000 {
				b.AppendString("gb")
			}
		},
	)

	got, raw := writtenBytes(t, want, nil)
	same(t, got, want)

	pages := chunkPages(t, raw, 0)
	if !dictChunk(t, pages) {
		t.Fatal("the column was written plainly, want a dictionary")
	}
	if n := pages[0].NumValues; n != 1 {
		t.Errorf("the dictionary page holds %d values, want 1", n)
	}
	if width := pages[1].Data[0]; width != 0 {
		t.Errorf("an index into a dictionary of one takes %d bits, want 0", width)
	}
}

// TestWriteDictGroups checks that each row group gets its own dictionary.
//
// The indices of a page mean nothing outside the chunk they were written in, so
// a writer that carried a dictionary from one row group into the next would
// write indices into values that group has no page for. What comes out of that
// is a file that opens and hands back the wrong rows, which is why the values
// are different in every group here.
func TestWriteDictGroups(t *testing.T) {
	words := []string{"one", "two", "three", "four", "five", "six"}
	want := tableOf(t,
		[]dtype.Field{{Name: "word", Type: dtype.String}},
		func(b *array.Builder) {
			for i := range 300 {
				b.AppendString(words[i/50])
			}
		},
	)

	got, raw := writtenBytes(t, want, &parquet.WriteOptions{RowGroupSize: 50})
	same(t, got, want)

	m, err := parquet.ReadMetadata(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}
	if n := len(m.RowGroups); n != len(words) {
		t.Fatalf("%d row groups, want %d", n, len(words))
	}
	for i := range m.RowGroups {
		c := &m.RowGroups[i].Columns[0]
		if c.Meta.DictionaryPageOffset == 0 {
			t.Errorf("group %d says it has no dictionary page", i)
		}
	}
}

// TestWriteDictNulls checks the columns there is nothing to make a dictionary
// of.
//
// A chunk whose every row is missing has no values to point at, and a dictionary
// page of none is a page no reader is expecting to find behind an offset. The
// column of the null type is the same thing said in the schema rather than in
// the validity.
func TestWriteDictNulls(t *testing.T) {
	want := tableOf(t,
		[]dtype.Field{
			{Name: "gone", Type: dtype.String, Nullable: true},
			{Name: "nothing", Type: dtype.Null, Nullable: true},
		},
		func(b *array.Builder) {
			for range 10 {
				b.AppendNull()
			}
		},
		func(b *array.Builder) {
			for range 10 {
				b.AppendNull()
			}
		},
	)

	got, raw := writtenBytes(t, want, nil)
	same(t, got, want)

	for i, name := range []string{"gone", "nothing"} {
		if dictChunk(t, chunkPages(t, raw, i)) {
			t.Errorf("%s was written as a dictionary and every value of it is missing", name)
		}
	}
}

// TestWriteDictBounds checks that a dictionary chunk says what it holds.
//
// The bounds of a dictionary chunk are taken off its distinct values rather than
// off all of them, which is a saving and is the sort of saving that quietly
// gives a different answer. So this is the bounds test again on a column written
// the other way, and the values are the ones a repeated column would be.
func TestWriteDictBounds(t *testing.T) {
	table := tableOf(t,
		[]dtype.Field{
			{Name: "n", Type: dtype.Int64},
			{Name: "word", Type: dtype.String, Nullable: true},
		},
		func(b *array.Builder) {
			for i := range 100 {
				b.Append(int64(i % 7))
			}
		},
		func(b *array.Builder) {
			for i := range 100 {
				if i%10 == 0 {
					b.AppendNull()
					continue
				}
				b.AppendString(strconv.Itoa(i % 7))
			}
		},
	)

	r := reader(t, table, nil)
	b := boundsOf(t, r, 0)
	if got, want := [2]string{bound(t, b[0], 0), bound(t, b[0], 1)}, [2]string{"0", "6"}; got != want {
		t.Errorf("n runs from %v, want %v", got, want)
	}

	words := [2]string{strconv.Quote("0"), strconv.Quote("6")}
	if got := [2]string{bound(t, b[1], 0), bound(t, b[1], 1)}; got != words {
		t.Errorf("word runs from %v, want %v", got, words)
	}
	if got := b[1].Nulls; got != 10 {
		t.Errorf("word says %d of its values are missing, want 10", got)
	}
}

// TestWriteDictSmaller checks that the encoding does what it is for.
//
// A column of a few repeated strings written as a dictionary is a few strings
// and a run of small integers, and written plainly it is the strings again and
// again. The two files hold the same rows, so the difference between them is the
// whole of what the encoding buys.
func TestWriteDictSmaller(t *testing.T) {
	codes := []string{"united kingdom", "united states", "japan", "germany"}
	table := tableOf(t,
		[]dtype.Field{{Name: "country", Type: dtype.String}},
		func(b *array.Builder) {
			for i := range 10000 {
				b.AppendString(codes[i%len(codes)])
			}
		},
	)

	_, small := writtenBytes(t, table, nil)
	_, large := writtenBytes(t, table, &parquet.WriteOptions{Plain: true})
	if len(small) >= len(large) {
		t.Errorf("the dictionary file is %d bytes and the plain one is %d", len(small), len(large))
	}
}

// TestWriteDictFull checks the chunks that hold too much to be worth a
// dictionary.
//
// A dictionary stops paying once there are enough values in it that an index is
// nearly as wide as a value, and once the page holding them is larger than the
// pages it is saving. Those are two limits and a column runs into one or the
// other, so there is a column here for each: distinct numbers for the count,
// distinct long strings for the size, and distinct short strings for the count
// again on the side that measures both.
//
// What matters is not that the chunk gave up but that it gave up before it wrote
// anything. A chunk of indices at the front and values at the back is a file the
// reader next door refuses, so the round trip is the real assertion here.
func TestWriteDictFull(t *testing.T) {
	const (
		values = 1 << 17
		long   = 512
	)

	want := tableOf(t,
		[]dtype.Field{
			{Name: "n", Type: dtype.Int64},
			{Name: "short", Type: dtype.String},
			{Name: "long", Type: dtype.String},
		},
		func(b *array.Builder) {
			for i := range values + 1 {
				b.Append(int64(i))
			}
		},
		func(b *array.Builder) {
			for i := range values + 1 {
				b.AppendString(strconv.Itoa(i))
			}
		},
		func(b *array.Builder) {
			// Enough distinct values to go past the byte limit and not nearly
			// enough to go past the count.
			pad := strings.Repeat("x", long)
			for i := range values + 1 {
				b.AppendString(strconv.Itoa(i%4096) + pad)
			}
		},
	)

	got, raw := writtenBytes(t, want, nil)
	same(t, got, want)

	for i, name := range []string{"n", "short", "long"} {
		if dictChunk(t, chunkPages(t, raw, i)) {
			t.Errorf("%s was written as a dictionary and holds more than one is worth", name)
		}
	}
}
