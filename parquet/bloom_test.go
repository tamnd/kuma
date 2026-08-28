package parquet_test

import (
	"encoding/binary"
	"errors"
	"fmt"
	"testing"

	"github.com/tamnd/kuma/parquet"
)

// The filter pyarrow wrote for bloom.parquet.
//
// Everything here is checked against a file another implementation wrote, which
// is the only check worth having: the hash, the block a value lands in and the
// bits it sets are all in the format and none of them are in the file, so a
// reader that got any of the three wrong would agree with itself and with
// nothing else.

// identifier is the value of the id column of row i of bloom.parquet, and the
// bytes a filter on it was built from, which is the number as a page writes it.
func identifier(i int) (int64, []byte) {
	v := int64(1000 + i*7)
	return v, binary.LittleEndian.AppendUint64(nil, uint64(v))
}

// name is the value of the name column of row i, which a filter holds as the
// bytes of the string with no length in front of them.
func name(i int) string { return fmt.Sprintf("user-%04d", i*7) }

// bloomOf reads the filter of one projected column of one row group and fails
// the test if it cannot.
func bloomOf(tb testing.TB, r *parquet.FileReader, group, column int) *parquet.BloomFilter {
	tb.Helper()

	f, err := r.BloomFilter(group, column)
	if err != nil {
		tb.Fatalf("BloomFilter(%d, %d): %v", group, column, err)
	}
	return f
}

// TestBloomFilter looks every value of a chunk up in the filter that was
// written for it.
//
// A filter that answers no to a value that is there is broken in the way that
// matters, since a scan believes that answer and skips the chunk, so the rows
// go missing and nothing says they did. All two hundred of them are looked up
// here, in both of the columns that have a filter and in both row groups.
func TestBloomFilter(t *testing.T) {
	r := openFileReader(t, "bloom.parquet")

	for group := range 2 {
		ids, names := bloomOf(t, r, group, 0), bloomOf(t, r, group, 1)
		if got := ids.Bytes(); got != 128 {
			t.Errorf("the filter of id of group %d is %d bytes, want 128", group, got)
		}

		for i := group * 100; i < group*100+100; i++ {
			if _, b := identifier(i); !ids.Has(b) {
				v, _ := identifier(i)
				t.Fatalf("the filter of id of group %d says %d is not in it, and row %d is", group, v, i)
			}
			if !names.HasString(name(i)) {
				t.Fatalf("the filter of name of group %d says %q is not in it, and row %d is",
					group, name(i), i)
			}
		}
	}
}

// TestBloomFilterRulesOut is what the filter is for, which is the values the
// bounds keep and the chunk does not hold.
//
// The identifiers go up in sevens, so a value one above one of them is between
// two values of the chunk and inside its bounds, and every one of those is a
// row group a filter can rule out and a bound cannot. A few of them come back
// kept anyway, that being what a bloom filter costs, and the count of those is
// what the writer's error rate bought.
func TestBloomFilterRulesOut(t *testing.T) {
	r := openFileReader(t, "bloom.parquet")
	f := bloomOf(t, r, 0, 0)

	kept := 0
	for i := range 100 {
		v, _ := identifier(i)
		b := binary.LittleEndian.AppendUint64(nil, uint64(v+1))
		if f.Has(b) {
			kept++
		}
	}

	// The writer was asked for one wrong answer in twenty over a hundred values.
	// This is a file in testdata rather than a filter built here, so the count is
	// the same on every run, and what is being checked is that the filter is
	// doing something rather than saying yes to everything.
	if kept > 20 {
		t.Errorf("the filter kept %d of 100 values the chunk does not hold, want a handful", kept)
	}
}

// TestBloomFilterMissing is a chunk whose writer wrote no filter, which is most
// chunks in most files.
//
// The filter that comes back is nil and that is not an error, and a nil filter
// answers yes to everything, so a scan can look a value up in whatever it got
// without checking whether it got anything.
func TestBloomFilterMissing(t *testing.T) {
	r := openFileReader(t, "bloom.parquet")

	f := bloomOf(t, r, 0, 2)
	if f != nil {
		t.Fatalf("the plain column came back with a filter of %d bytes, and none was written for it", f.Bytes())
	}
	if f.Bytes() != 0 {
		t.Errorf("a filter that is not there is %d bytes", f.Bytes())
	}
	if !f.Has([]byte("anything")) || !f.HasString("anything") {
		t.Error("a filter that is not there ruled a value out")
	}

	var zero parquet.BloomFilter
	if !zero.Has(nil) || zero.Bytes() != 0 {
		t.Error("a filter with no bits in it ruled a value out")
	}
}

// TestBloomFilterSkip is the row group a scan does not read.
//
// The value is one the file does not hold and both row groups say they might:
// it is inside the bounds of the first, and the second is ruled out by its
// bounds alone. That is the case the whole thing exists for, one identifier in
// a file of many, where a scan reading every group whose range covers it reads
// the file.
func TestBloomFilterSkip(t *testing.T) {
	const want = 1004

	r := openFileReader(t, "bloom.parquet")
	if err := r.Project("id"); err != nil {
		t.Fatalf("Project: %v", err)
	}
	value := binary.LittleEndian.AppendUint64(nil, uint64(want))

	bounds, filters := 0, 0
	for i := range r.NumRowGroups() {
		if lo := numberBounds[int64](t, boundsOf(t, r, i)[0]); want < lo[0] || want > lo[1] {
			continue
		}
		bounds++

		if bloomOf(t, r, i, 0).Has(value) {
			filters++
		}
	}

	if bounds != 1 {
		t.Errorf("the bounds kept %d row groups, want the one %d is inside", bounds, want)
	}
	if filters != 0 {
		t.Errorf("the filters kept %d row groups, want none, %d being in neither", filters, want)
	}
}

// TestBloomFilterProjection reads the filter of a column by its place in the
// projection rather than in the file, which is where every other thing a reader
// says about a column is read from.
func TestBloomFilterProjection(t *testing.T) {
	r := openFileReader(t, "bloom.parquet")
	if err := r.Project("name", "id"); err != nil {
		t.Fatalf("Project: %v", err)
	}

	if !bloomOf(t, r, 0, 0).HasString(name(0)) {
		t.Error("the first column of the projection is not the one the projection put there")
	}
	if _, b := identifier(0); !bloomOf(t, r, 0, 1).Has(b) {
		t.Error("the second column of the projection is not the one the projection put there")
	}
}

// TestBloomFilterRefused is what a reader will not answer for, which is the
// same list as everywhere else it is asked about a column of a row group.
func TestBloomFilterRefused(t *testing.T) {
	r := openFileReader(t, "bloom.parquet")

	for _, i := range []int{-1, 2} {
		if _, err := r.BloomFilter(i, 0); err == nil {
			t.Errorf("row group %d of a file of two was answered for", i)
		}
	}
	for _, i := range []int{-1, 3} {
		if _, err := r.BloomFilter(0, i); err == nil {
			t.Errorf("column %d of a projection of three was answered for", i)
		}
	}

	r.Metadata().RowGroups[0].Columns = nil
	if _, err := r.BloomFilter(0, 0); !errors.Is(err, parquet.ErrFormat) {
		t.Errorf("got %v, want %v", err, parquet.ErrFormat)
	}
}

// TestBloomFilterCost is what asking costs, which is one read of the filter of
// the column asked about and nothing else.
//
// The filter is at the end of the file rather than in the footer, so this is a
// read that a scan pays for before it knows whether it saved anything. It is a
// hundred and forty four bytes here against a row group of a few thousand,
// which is the trade at any size worth making it at.
func TestBloomFilterCost(t *testing.T) {
	r := openFileReader(t, "bloom.parquet")
	m := r.Metadata()

	want := footerOf(t, "bloom.parquet")
	for group := range m.RowGroups {
		for column := range 2 {
			want += int64(m.RowGroups[group].Columns[column].Meta.BloomFilterLength)

			if bloomOf(t, r, group, column) == nil {
				t.Fatalf("column %d of group %d came back without a filter", column, group)
			}
		}
	}

	if got := r.BytesRead(); got != want {
		t.Errorf("reading every filter cost %d bytes, want the footer and the filters, which is %d",
			got, want)
	}
}

// BenchmarkBloomFilter is the cost of ruling a row group out against the cost
// of reading it and finding nothing.
//
// The lookup includes reading the filter out of the file, since that is what a
// scan actually does once per chunk. Everything after the read is eight loads
// out of one cache line and is not what the numbers here are about.
func BenchmarkBloomFilter(b *testing.B) {
	value := binary.LittleEndian.AppendUint64(nil, uint64(1004))

	b.Run("the filter", func(b *testing.B) {
		r := openFileReader(b, "bloom.parquet")

		b.ReportAllocs()
		for b.Loop() {
			f, err := r.BloomFilter(0, 0)
			if err != nil {
				b.Fatalf("BloomFilter: %v", err)
			}
			if f.Has(value) {
				b.Fatal("the filter kept a row group holding nothing wanted")
			}
		}
	})

	b.Run("the values", func(b *testing.B) {
		r := openFileReader(b, "bloom.parquet")
		if err := r.Project("id"); err != nil {
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
