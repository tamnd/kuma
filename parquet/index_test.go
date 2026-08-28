package parquet_test

import (
	"bytes"
	"errors"
	"testing"

	"github.com/tamnd/kuma/parquet"
)

// pageBoundsOf reads what a writer said about every page of one column of one
// row group, and fails the test if it cannot.
func pageBoundsOf(tb testing.TB, r *parquet.FileReader, group, column int) []parquet.PageBounds {
	tb.Helper()

	b, err := r.PageBounds(group, column)
	if err != nil {
		tb.Fatalf("PageBounds(%d, %d): %v", group, column, err)
	}
	return b
}

// indexReader is the file behind index.parquet and its size, which is what the
// two index readers are given, along with its footer.
func indexReader(t *testing.T, name string) (*bytes.Reader, int64, *parquet.Metadata) {
	t.Helper()

	b := bytesOf(t, name)
	m, err := parquet.ReadMetadata(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatalf("ReadMetadata(%s): %v", name, err)
	}
	return bytes.NewReader(b), int64(len(b)), m
}

// TestPageBounds is what the page index says about a file written to have one.
//
// The file holds two row groups of two hundred rows written a hundred rows to
// the page, so every column of every group is two pages and the bounds of each
// are the values of a hundred rows rather than of the group. That is the whole
// point of the index: a filter looking for one value in a group it cannot skip
// reads one page of the two.
func TestPageBounds(t *testing.T) {
	tests := []struct {
		group  int
		column int
		what   string
		pages  [2][2]int32
	}{
		{0, 0, "n", [2][2]int32{{0, 99}, {100, 199}}},
		{0, 1, "down", [2][2]int32{{300, 399}, {200, 299}}},
		{0, 2, "wave", [2][2]int32{{10, 20}, {0, 30}}},
		{1, 0, "n", [2][2]int32{{200, 299}, {300, 399}}},
		{1, 1, "down", [2][2]int32{{100, 199}, {0, 99}}},
		{1, 2, "wave", [2][2]int32{{10, 20}, {0, 30}}},
	}

	r := openFileReader(t, "index.parquet")
	for _, tt := range tests {
		b := pageBoundsOf(t, r, tt.group, tt.column)
		if len(b) != 2 {
			t.Fatalf("%s of group %d came back with %d pages, want 2", tt.what, tt.group, len(b))
		}

		for i, p := range b {
			if got := numberBounds[int32](t, p.Bounds); got != tt.pages[i] {
				t.Errorf("page %d of %s of group %d runs from %v, want %v",
					i, tt.what, tt.group, got, tt.pages[i])
			}
			if p.Count != 100 {
				t.Errorf("page %d of %s of group %d holds %d rows, want 100",
					i, tt.what, tt.group, p.Count)
			}
			if want := int64(i * 100); p.FirstRow != want {
				t.Errorf("page %d of %s of group %d starts at row %d, want %d",
					i, tt.what, tt.group, p.FirstRow, want)
			}
			if p.Offset <= 0 || p.CompressedSize <= 0 {
				t.Errorf("page %d of %s of group %d is %d bytes at %d",
					i, tt.what, tt.group, p.CompressedSize, p.Offset)
			}
		}
	}
}

// TestPageBoundsText is the same for a column of strings, whose bounds are the
// bytes of a value rather than a number in four of them.
func TestPageBoundsText(t *testing.T) {
	want := [2][2][2]string{
		{{"w0000", "w0099"}, {"w0100", "w0199"}},
		{{"w0200", "w0299"}, {"w0300", "w0399"}},
	}

	r := openFileReader(t, "index.parquet")
	for group := range want {
		b := pageBoundsOf(t, r, group, 3)
		if len(b) != 2 {
			t.Fatalf("word of group %d came back with %d pages, want 2", group, len(b))
		}

		for i, p := range b {
			if got := textBounds(t, p.Bounds); got != want[group][i] {
				t.Errorf("page %d of word of group %d runs from %q, want %q",
					i, group, got, want[group][i])
			}
		}
	}
}

// TestPageBoundsNulls is the column that is missing for a whole page.
//
// The format leaves the bounds of such a page undefined, so what says a filter
// can skip it is the index calling it a null page rather than anything it says
// about its values. The page that does hold values is bounded and counts none
// of them missing, which is the difference between a column that is nullable
// and a page that is null.
func TestPageBoundsNulls(t *testing.T) {
	r := openFileReader(t, "index.parquet")
	if err := r.Project("gap"); err != nil {
		t.Fatalf("Project: %v", err)
	}

	for group, want := range [2][2]int64{{0, 99}, {200, 299}} {
		b := pageBoundsOf(t, r, group, 0)
		if len(b) != 2 {
			t.Fatalf("gap of group %d came back with %d pages, want 2", group, len(b))
		}

		if got := numberBounds[int64](t, b[0].Bounds); got != want {
			t.Errorf("page 0 of gap of group %d runs from %v, want %v", group, got, want)
		}
		if b[0].AllNull() || b[0].Nulls != 0 || !b[0].HasNulls {
			t.Errorf("page 0 of gap of group %d counts %d of %d rows missing",
				group, b[0].Nulls, b[0].Count)
		}

		if b[1].Values != nil {
			t.Errorf("page 1 of gap of group %d has bounds, and every value in it is missing", group)
		}
		if !b[1].AllNull() || b[1].Nulls != 100 {
			t.Errorf("page 1 of gap of group %d counts %d of %d rows missing, want all of them",
				group, b[1].Nulls, b[1].Count)
		}
	}
}

// TestColumnIndex reads the index itself, which is where the order of the pages
// is.
//
// A writer that wrote its rows sorted says so, and that is what lets a reader
// looking for one value find its page by halving rather than by walking. The
// three columns here are the three answers: one runs up, one runs down, and one
// holds a page whose values are on both sides of the page before it, which is
// neither.
func TestColumnIndex(t *testing.T) {
	tests := []struct {
		column int
		what   string
		want   parquet.BoundaryOrder
	}{
		{0, "n", parquet.Ascending},
		{1, "down", parquet.Descending},
		{2, "wave", parquet.Unordered},
		{3, "word", parquet.Ascending},
		{4, "gap", parquet.Ascending},
	}

	src, size, m := indexReader(t, "index.parquet")
	group := &m.RowGroups[0]

	for _, tt := range tests {
		x, err := parquet.ReadColumnIndex(src, size, &group.Columns[tt.column])
		if err != nil {
			t.Fatalf("ReadColumnIndex(%s): %v", tt.what, err)
		}
		if x.Order != tt.want {
			t.Errorf("the pages of %s are %s, want %s", tt.what, x.Order, tt.want)
		}
		if len(x.NullPages) != 2 || len(x.Min) != 2 || len(x.Max) != 2 {
			t.Fatalf("the index of %s bounds %d pages, want 2", tt.what, len(x.NullPages))
		}
	}

	// The nullable column is the one with a page of nothing, and the one whose
	// counts of missing values say so twice over.
	x, err := parquet.ReadColumnIndex(src, size, &group.Columns[4])
	if err != nil {
		t.Fatalf("ReadColumnIndex(gap): %v", err)
	}
	if x.NullPages[0] || !x.NullPages[1] {
		t.Errorf("the pages of gap are null %v, want the second one only", x.NullPages)
	}
	if got, want := x.NullCounts, []int64{0, 100}; len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("the pages of gap count %v missing values, want %v", got, want)
	}
}

// TestOffsetIndex is where the pages are, which is what a reader that has
// decided to read one needs and what a reader that skipped them all never asks
// for.
func TestOffsetIndex(t *testing.T) {
	src, size, m := indexReader(t, "index.parquet")

	for i := range m.RowGroups {
		group := &m.RowGroups[i]
		for j := range group.Columns {
			chunk := &group.Columns[j]

			o, err := parquet.ReadOffsetIndex(src, size, chunk)
			if err != nil {
				t.Fatalf("ReadOffsetIndex(%d, %d): %v", i, j, err)
			}
			if len(o.Pages) != 2 {
				t.Fatalf("column %d of group %d locates %d pages, want 2", j, i, len(o.Pages))
			}

			// The first page of a chunk is where the chunk's data starts, which
			// the footer says as well, so the two have to agree.
			if got, want := o.Pages[0].Offset, chunk.Meta.DataPageOffset; got != want {
				t.Errorf("the first page of column %d of group %d is at %d, and the footer says %d",
					j, i, got, want)
			}
			if o.Pages[0].FirstRow != 0 || o.Pages[1].FirstRow != 100 {
				t.Errorf("the pages of column %d of group %d start at rows %d and %d, want 0 and 100",
					j, i, o.Pages[0].FirstRow, o.Pages[1].FirstRow)
			}
			if o.Pages[1].Offset <= o.Pages[0].Offset {
				t.Errorf("page 1 of column %d of group %d is at %d, in front of page 0 at %d",
					j, i, o.Pages[1].Offset, o.Pages[0].Offset)
			}
		}
	}
}

// TestPageBoundsCost is what asking costs, which is a read of the two index
// structures of the column asked about and not a page of anything.
//
// This is why the indexes are asked for per column. A scan filtering on one
// column of five reads a fifth of what reading the lot would, and none of it is
// data.
func TestPageBoundsCost(t *testing.T) {
	r := openFileReader(t, "index.parquet")
	m := r.Metadata()

	want := footerOf(t, "index.parquet")
	for group := range m.RowGroups {
		for column := range m.RowGroups[group].Columns {
			c := &m.RowGroups[group].Columns[column]
			want += int64(c.ColumnIndexLength) + int64(c.OffsetIndexLength)

			if len(pageBoundsOf(t, r, group, column)) != 2 {
				t.Fatalf("column %d of group %d came back with the wrong number of pages", column, group)
			}
		}
	}

	if got := r.BytesRead(); got != want {
		t.Errorf("reading every index cost %d bytes, want the footer and the indexes, which is %d",
			got, want)
	}
}

// TestPageBoundsMissing is a file whose writer wrote no page index, which is
// most of them: pyarrow writes one only when it is asked to.
//
// Nothing here is an error. A reader without a page index is a reader that
// skips row groups and reads all of the ones it keeps, which is where it was
// before the index existed.
func TestPageBoundsMissing(t *testing.T) {
	src, size, m := indexReader(t, "pages.parquet")
	chunk := &m.RowGroups[0].Columns[0]

	x, err := parquet.ReadColumnIndex(src, size, chunk)
	if x != nil || err != nil {
		t.Errorf("ReadColumnIndex of a file without one = %v, %v", x, err)
	}

	o, err := parquet.ReadOffsetIndex(src, size, chunk)
	if o != nil || err != nil {
		t.Errorf("ReadOffsetIndex of a file without one = %v, %v", o, err)
	}

	r := openFileReader(t, "pages.parquet")
	b, err := r.PageBounds(0, 0)
	if b != nil || err != nil {
		t.Errorf("PageBounds of a file without an index = %v, %v", b, err)
	}
	if got, want := r.BytesRead(), footerOf(t, "pages.parquet"); got != want {
		t.Errorf("asking read %d bytes, want the footer's %d", got, want)
	}
}

// TestPageBoundsUnordered is the columns whose pages can be counted and not
// compared.
//
// The column index came after the format settled how its types compare, so
// there is no older pair of bounds to fall back to and a column the file gave
// no order is a column with nothing to read. What still comes back is where the
// pages are and how many rows are in them, which is what a reader needs to read
// one.
func TestPageBoundsUnordered(t *testing.T) {
	src, size, m := indexReader(t, "index.parquet")
	columns, err := m.Columns()
	if err != nil {
		t.Fatalf("Columns: %v", err)
	}

	unordered := columns[0]
	unordered.Order = parquet.UndefinedOrder

	cases := []struct {
		name   string
		column parquet.Column
	}{
		{"a column the file gave no order", unordered},
		{"a column of nothing", withOrder(missing())},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			b, err := parquet.ReadPageBounds(src, size, &m.RowGroups[0].Columns[0], c.column, 200)
			if err != nil {
				t.Fatalf("ReadPageBounds: %v", err)
			}
			if len(b) != 2 {
				t.Fatalf("%d pages, want 2", len(b))
			}
			for i, p := range b {
				if p.Values != nil {
					t.Errorf("page %d of a column with no order came back bounded", i)
				}
				if p.Count != 100 {
					t.Errorf("page %d holds %d rows, want 100", i, p.Count)
				}
			}
		})
	}
}

// TestFileReaderPageBoundsRefused is what a reader will not answer for. The
// column is its place in the projection rather than in the file, the same as
// everywhere else a reader is asked about one.
func TestFileReaderPageBoundsRefused(t *testing.T) {
	r := openFileReader(t, "index.parquet")

	for _, i := range []int{-1, 2} {
		if _, err := r.PageBounds(i, 0); err == nil {
			t.Errorf("row group %d of a file of two was answered for", i)
		}
	}

	if err := r.Project("n", "word"); err != nil {
		t.Fatalf("Project: %v", err)
	}
	for _, i := range []int{-1, 2} {
		if _, err := r.PageBounds(0, i); err == nil {
			t.Errorf("column %d of a projection of two was answered for", i)
		}
	}

	// A footer with no chunk where the projection says one is.
	r.Metadata().RowGroups[0].Columns = nil
	if _, err := r.PageBounds(0, 0); !errors.Is(err, parquet.ErrFormat) {
		t.Errorf("got %v, want %v", err, parquet.ErrFormat)
	}
}

// TestPageSkip reads one page of four, the other three being skipped on what
// the index said about them.
//
// This is the point of the whole thing, one level down from skipping a row
// group. The filter is one value, one row group of the two cannot hold it and
// is skipped on its own bounds, and of the two pages of the group that is left
// only one has a range the value is in.
func TestPageSkip(t *testing.T) {
	const want = 150

	r := openFileReader(t, "index.parquet")
	if err := r.Project("n"); err != nil {
		t.Fatalf("Project: %v", err)
	}

	groups, pages := 0, 0
	for i := range r.NumRowGroups() {
		b := boundsOf(t, r, i)[0]
		if lo := numberBounds[int32](t, b); want < lo[0] || want > lo[1] {
			continue
		}
		groups++

		for _, p := range pageBoundsOf(t, r, i, 0) {
			if p.AllNull() {
				continue
			}
			if lo := numberBounds[int32](t, p.Bounds); want < lo[0] || want > lo[1] {
				continue
			}
			pages++

			if p.FirstRow != 100 || p.Count != 100 {
				t.Errorf("the page holding %d is the %d rows from %d, want the second hundred",
					want, p.Count, p.FirstRow)
			}
		}
	}

	if groups != 1 || pages != 1 {
		t.Errorf("%d holds in %d pages of %d row groups, want one of each", want, pages, groups)
	}
}

// BenchmarkPageBounds is the cost of narrowing a row group to one of its pages
// against the cost of reading the group.
//
// The index is not in the footer, so the first is a read of the file and not
// just work on bytes already in memory, which is what makes it worth measuring
// against the read it saves.
func BenchmarkPageBounds(b *testing.B) {
	b.Run("the page bounds", func(b *testing.B) {
		r := openFileReader(b, "index.parquet")

		b.ReportAllocs()
		for b.Loop() {
			if _, err := r.PageBounds(0, 0); err != nil {
				b.Fatalf("PageBounds: %v", err)
			}
		}
	})

	b.Run("the values", func(b *testing.B) {
		r := openFileReader(b, "index.parquet")
		if err := r.Project("n"); err != nil {
			b.Fatalf("Project: %v", err)
		}

		b.ReportAllocs()
		for b.Loop() {
			if _, err := r.RowGroup(0); err != nil {
				b.Fatalf("RowGroup: %v", err)
			}
		}
	})
}
