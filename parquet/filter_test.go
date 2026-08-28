package parquet_test

import (
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/parquet"
)

// groupsOf asks which row groups a filter keeps and fails the test if it cannot
// be asked.
func groupsOf(tb testing.TB, r *parquet.FileReader, filter ...parquet.Predicate) []int {
	tb.Helper()

	g, err := r.RowGroups(filter...)
	if err != nil {
		tb.Fatalf("RowGroups: %v", err)
	}
	return g
}

// TestRowGroupsBounds skips row groups on what the footer says their columns
// hold.
//
// stats.parquet is twelve rows of n running from nought to eleven in three
// groups of four, so every group covers a range of its own and every comparison
// against a number has a different answer. That is the whole of the pushdown on
// a column written in order, which is the column a scan is usually filtered on.
func TestRowGroupsBounds(t *testing.T) {
	r := openFileReader(t, "stats.parquet")

	for _, c := range []struct {
		op   kernel.CompareOp
		v    int64
		want []int
	}{
		// The three groups hold 0 to 3, 4 to 7 and 8 to 11.
		{kernel.OpEq, 5, []int{1}},
		{kernel.OpEq, 0, []int{0}},
		{kernel.OpEq, 11, []int{2}},
		{kernel.OpEq, 12, nil},
		{kernel.OpEq, -1, nil},
		{kernel.OpLt, 4, []int{0}},
		{kernel.OpLt, 5, []int{0, 1}},
		{kernel.OpLt, 0, nil},
		{kernel.OpLe, 0, []int{0}},
		{kernel.OpLe, 3, []int{0}},
		{kernel.OpLe, 4, []int{0, 1}},
		{kernel.OpGt, 7, []int{2}},
		{kernel.OpGt, 6, []int{1, 2}},
		{kernel.OpGt, 11, nil},
		{kernel.OpGe, 8, []int{2}},
		{kernel.OpGe, 12, nil},
		{kernel.OpGe, 0, []int{0, 1, 2}},

		// Not equal only rules a group out when every value in it is the value
		// being avoided, which no group of four different numbers is.
		{kernel.OpNe, 5, []int{0, 1, 2}},
	} {
		got := groupsOf(t, r, parquet.Where("n", c.op, c.v))
		if !slices.Equal(got, c.want) {
			t.Errorf("n %s %d keeps %v, want %v", c.op, c.v, got, c.want)
		}
	}
}

// TestRowGroupsText is the same on a column of words, which is the case the
// format took two goes at ordering and the reason Bounds is careful about which
// pair of them to read.
func TestRowGroupsText(t *testing.T) {
	r := openFileReader(t, "stats.parquet")

	for _, c := range []struct {
		op   kernel.CompareOp
		v    string
		want []int
	}{
		// The groups hold "" to "echo", "mike" to "papa" and "sierra" to "zulu".
		{kernel.OpEq, "zulu", []int{2}},
		{kernel.OpEq, "", []int{0}},
		{kernel.OpEq, "oscar", []int{1}},
		{kernel.OpEq, "quebec", nil},
		{kernel.OpLt, "mike", []int{0}},
		{kernel.OpGt, "papa", []int{2}},
		{kernel.OpGe, "", []int{0, 1, 2}},
	} {
		got := groupsOf(t, r, parquet.WhereString("word", c.op, c.v))
		if !slices.Equal(got, c.want) {
			t.Errorf("word %s %q keeps %v, want %v", c.op, c.v, got, c.want)
		}
	}
}

// TestRowGroupsUnsigned filters a column holding a value that is negative when
// it is read as the signed integer parquet writes it in.
//
// The largest value of the first group is four thousand million and something,
// which the deprecated pair of bounds would have called the smallest, so a
// reader that read the wrong pair would skip the group that holds it.
func TestRowGroupsUnsigned(t *testing.T) {
	r := openFileReader(t, "stats.parquet")

	for _, c := range []struct {
		v    uint32
		want []int
	}{
		// The groups hold 1 to 4294967295, 10 to 13 and 20 to 23, so the first
		// of them covers the other two and is kept by everything.
		{4294967295, []int{0}},
		{1, []int{0}},
		{11, []int{0, 1}},
		{22, []int{0, 2}},
		{4294967294, []int{0}},
		{0, nil},
	} {
		got := groupsOf(t, r, parquet.Where("size", kernel.OpEq, c.v))
		if !slices.Equal(got, c.want) {
			t.Errorf("size == %d keeps %v, want %v", c.v, got, c.want)
		}
	}
}

// TestRowGroupsNaN filters a float column with a NaN in it.
//
// A writer leaves NaN out of the bounds it writes, so the group holding one is
// bounded by the rest of its values and the NaN row is invisible to a skip. That
// is safe for five of the six comparisons, since every one of them is false of a
// NaN and a row that cannot match is a row there is no harm in skipping past.
//
// Looking for NaN itself keeps nothing, since nothing equals it, and everything
// is unequal to it.
func TestRowGroupsNaN(t *testing.T) {
	r := openFileReader(t, "stats.parquet")

	// The groups hold 0.5 to 3.5, 1.0 to 3.0 with a NaN as well, and -1.0 to 2.0.
	for _, c := range []struct {
		op   kernel.CompareOp
		v    float64
		want []int
	}{
		{kernel.OpEq, 3.5, []int{0}},
		{kernel.OpEq, -1.0, []int{2}},
		{kernel.OpEq, 1.0, []int{0, 1, 2}},
		{kernel.OpLt, 0.0, []int{2}},
		{kernel.OpGt, 3.4, []int{0}},
	} {
		got := groupsOf(t, r, parquet.Where("ratio", c.op, c.v))
		if !slices.Equal(got, c.want) {
			t.Errorf("ratio %s %v keeps %v, want %v", c.op, c.v, got, c.want)
		}
	}

	nan := parquet.Where("ratio", kernel.OpEq, math.NaN())
	if got := groupsOf(t, r, nan); len(got) != 0 {
		t.Errorf("ratio == NaN keeps %v, want none", got)
	}
	nan.Op = kernel.OpNe
	if got, want := groupsOf(t, r, nan), []int{0, 1, 2}; !slices.Equal(got, want) {
		t.Errorf("ratio != NaN keeps %v, want %v", got, want)
	}
}

// TestRowGroupsAllNull filters a column that is missing everywhere.
//
// Nothing compares to a value that is not there, so no comparison holds anywhere
// in the column and every group goes. The writer wrote no bounds for it at all,
// which is what makes this a question about the null count rather than about a
// range.
func TestRowGroupsAllNull(t *testing.T) {
	r := openFileReader(t, "stats.parquet")

	for _, op := range []kernel.CompareOp{
		kernel.OpEq, kernel.OpNe, kernel.OpLt, kernel.OpLe, kernel.OpGt, kernel.OpGe,
	} {
		if got := groupsOf(t, r, parquet.Where("absent", op, int32(1))); len(got) != 0 {
			t.Errorf("absent %s 1 keeps %v, want none", op, got)
		}
	}
}

// TestRowGroupsBool filters a column of two values, both of which are in every
// group, so the bounds rule nothing out.
func TestRowGroupsBool(t *testing.T) {
	r := openFileReader(t, "stats.parquet")

	p := parquet.Predicate{
		Column: "flag",
		Op:     kernel.OpEq,
		Value:  array.OfBools(true),
	}
	if got, want := groupsOf(t, r, p), []int{0, 1, 2}; !slices.Equal(got, want) {
		t.Errorf("flag == true keeps %v, want %v", got, want)
	}
}

// TestRowGroupsAnd checks that a list of predicates is an and.
//
// Each of these on its own keeps a group and the two together keep none, which
// is the case a scan with a filter on two columns is.
func TestRowGroupsAnd(t *testing.T) {
	r := openFileReader(t, "stats.parquet")

	n := parquet.Where("n", kernel.OpLt, int64(4))
	word := parquet.WhereString("word", kernel.OpGt, "sierra")

	if got, want := groupsOf(t, r, n), []int{0}; !slices.Equal(got, want) {
		t.Errorf("n < 4 keeps %v, want %v", got, want)
	}
	if got, want := groupsOf(t, r, word), []int{2}; !slices.Equal(got, want) {
		t.Errorf("word > sierra keeps %v, want %v", got, want)
	}
	if got := groupsOf(t, r, n, word); len(got) != 0 {
		t.Errorf("both together keep %v, want none", got)
	}
}

// TestRowGroupsNone is a filter of nothing, which keeps every group and is what
// a scan without one asks.
func TestRowGroupsNone(t *testing.T) {
	r := openFileReader(t, "stats.parquet")

	if got, want := groupsOf(t, r), []int{0, 1, 2}; !slices.Equal(got, want) {
		t.Errorf("no filter keeps %v, want %v", got, want)
	}
}

// TestRowGroupsUnprojected filters on a column the reader is not going to read.
//
// Filtering on a column and selecting it are different questions, and a scan
// that skips most of a file on a timestamp it never returns is the ordinary
// shape of one. So the filter names the columns of the file rather than the
// columns of the projection, and narrowing the projection to nothing leaves it
// working.
func TestRowGroupsUnprojected(t *testing.T) {
	r := openFileReader(t, "stats.parquet")
	if err := r.Project(); err != nil {
		t.Fatalf("Project(): %v", err)
	}

	got := groupsOf(t, r, parquet.Where("n", kernel.OpGe, int64(8)))
	if want := []int{2}; !slices.Equal(got, want) {
		t.Errorf("n >= 8 keeps %v, want %v", got, want)
	}
}

// TestRowGroupsBloom skips a row group on a bloom filter, which is the case the
// bounds cannot answer.
//
// bloom.parquet holds identifiers going up in sevens, so a value between two of
// them is inside the range of a group and is not in the group. The bounds keep
// it and the filter throws it out, which is the whole reason a writer is ever
// asked for one.
func TestRowGroupsBloom(t *testing.T) {
	r := openFileReader(t, "bloom.parquet")

	for _, c := range []struct {
		v    int64
		want []int
	}{
		// The groups hold 1000 to 1693 and 1700 to 2393, every seventh number.
		{1000, []int{0}},
		{1007, []int{0}},
		{1693, []int{0}},
		{1700, []int{1}},
		{2393, []int{1}},

		// Inside the bounds of a group and not in it.
		{1004, nil},
		{1500, nil},
		{2000, nil},

		// Outside the bounds of both, which the bounds answer on their own.
		{999, nil},
		{3000, nil},
	} {
		got := groupsOf(t, r, parquet.Where("id", kernel.OpEq, c.v))
		if !slices.Equal(got, c.want) {
			t.Errorf("id == %d keeps %v, want %v", c.v, got, c.want)
		}
	}
}

// TestRowGroupsBloomText is the same on the column of names, since a byte array
// is hashed as itself and a number is hashed in the width the file wrote it in.
func TestRowGroupsBloomText(t *testing.T) {
	r := openFileReader(t, "bloom.parquet")

	for _, c := range []struct {
		v    string
		want []int
	}{
		{"user-0000", []int{0}},
		{"user-0007", []int{0}},
		{"user-0700", []int{1}},
		{"user-0008", nil},
		{"user-0500", nil},
	} {
		got := groupsOf(t, r, parquet.WhereString("name", kernel.OpEq, c.v))
		if !slices.Equal(got, c.want) {
			t.Errorf("name == %q keeps %v, want %v", c.v, got, c.want)
		}
	}
}

// TestRowGroupsBloomOnly checks that the filter is asked only where it can
// answer.
//
// It answers equality and nothing else, since a filter holds hashes and a hash
// says nothing about which values are larger. The plain column has no filter
// written for it at all, which is what most columns of most files are, and it is
// skipped on its bounds like any other.
func TestRowGroupsBloomOnly(t *testing.T) {
	r := openFileReader(t, "bloom.parquet")

	// 1004 is not in the file and both groups keep it, because a comparison
	// other than equality reads the bounds and nothing else.
	got := groupsOf(t, r, parquet.Where("id", kernel.OpNe, int64(1004)))
	if want := []int{0, 1}; !slices.Equal(got, want) {
		t.Errorf("id != 1004 keeps %v, want %v", got, want)
	}
	got = groupsOf(t, r, parquet.Where("id", kernel.OpGe, int64(1004)))
	if want := []int{0, 1}; !slices.Equal(got, want) {
		t.Errorf("id >= 1004 keeps %v, want %v", got, want)
	}

	// The column without a filter, which the bounds still narrow.
	for _, c := range []struct {
		v    int32
		want []int
	}{
		{50, []int{0}},
		{150, []int{1}},
		{500, nil},
	} {
		got := groupsOf(t, r, parquet.Where("plain", kernel.OpEq, c.v))
		if !slices.Equal(got, c.want) {
			t.Errorf("plain == %d keeps %v, want %v", c.v, got, c.want)
		}
	}
}

// TestRowGroupsBloomBytesRead checks that a filter that answers no leaves the
// pages of the file alone.
//
// What it is allowed to read is the footer and the filters themselves, which
// together are a small part of a file and none of its data. A reader that went
// on to read the column would give the same answer and would have given up the
// point of asking.
func TestRowGroupsBloomBytesRead(t *testing.T) {
	r := openFileReader(t, "bloom.parquet")
	footer := footerOf(t, "bloom.parquet")

	if got := groupsOf(t, r, parquet.Where("id", kernel.OpEq, int64(1004))); len(got) != 0 {
		t.Fatalf("id == 1004 keeps %v, want none", got)
	}
	read := r.BytesRead()
	if read <= footer {
		t.Errorf("the filter read %d bytes and the footer is %d, want the bloom filters as well",
			read, footer)
	}

	// Two bitsets and two headers, which for a hundred distinct values at one
	// wrong answer in twenty is a few hundred bytes.
	if room := footer + 8192; read > room {
		t.Errorf("the filter read %d bytes, want no more than %d", read, room)
	}
}

// TestRowGroupsEmpty filters a file that holds a schema and no rows, which has
// no row groups to keep or skip.
func TestRowGroupsEmpty(t *testing.T) {
	r := openFileReader(t, "empty.parquet")

	if got := groupsOf(t, r, parquet.Where("id", kernel.OpEq, int64(1))); len(got) != 0 {
		t.Errorf("a file of no rows keeps %v, want none", got)
	}
}

// TestRowGroupsErrors covers the filters that are refused rather than answered.
//
// All of them are mistakes in the query rather than facts about the file, so
// they are reported before any of the file is read. A filter that was quietly
// dropped instead would read the whole file and look like it worked.
func TestRowGroupsErrors(t *testing.T) {
	r := openFileReader(t, "stats.parquet")

	for _, c := range []struct {
		name string
		p    parquet.Predicate
		want string
	}{
		{
			"a column the file does not have",
			parquet.Where("nope", kernel.OpEq, int64(1)),
			"no column called",
		},
		{
			"an operator that is not one of the six",
			parquet.Where("n", kernel.CompareOp(9), int64(1)),
			"unknown operator",
		},
		{
			"no value at all",
			parquet.Predicate{Column: "n", Op: kernel.OpEq},
			"compares against no value",
		},
		{
			"more than one value",
			parquet.Predicate{Column: "n", Op: kernel.OpEq, Value: array.Of(int64(1), int64(2))},
			"compares against 2 values",
		},
		{
			"a value that is not there",
			parquet.Predicate{Column: "n", Op: kernel.OpEq, Value: array.NewNull(1)},
			"a value that is not there",
		},
		{
			"a value of a type the column cannot be compared with",
			parquet.WhereString("n", kernel.OpEq, "five"),
			"cannot compare n",
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := r.RowGroups(c.p)
			if err == nil {
				t.Fatal("that worked")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("got %q, want it to mention %q", err, c.want)
			}
		})
	}
}

// TestRowGroupsFileErrors covers the filters that are refused because of what
// the file says rather than because of what was asked.
//
// All of them are a footer contradicting itself, which is the same thing reading
// the values of the column would run into. A filter that shrugged and read the
// row group instead would be reading a file it had already found to be wrong.
func TestRowGroupsFileErrors(t *testing.T) {
	n := parquet.Where("n", kernel.OpEq, int64(1))

	t.Run("a row group holding no chunk for the column", func(t *testing.T) {
		r := openFileReader(t, "stats.parquet")
		r.Metadata().RowGroups[0].Columns = nil

		if _, err := r.RowGroups(n); !errors.Is(err, parquet.ErrFormat) {
			t.Errorf("got %v, want %v", err, parquet.ErrFormat)
		}
	})

	t.Run("a chunk that contradicts itself", func(t *testing.T) {
		r := openFileReader(t, "stats.parquet")
		r.Metadata().RowGroups[0].Columns[0].Meta.Stats.NullCount = 99

		if _, err := r.RowGroups(n); !errors.Is(err, parquet.ErrFormat) {
			t.Errorf("got %v, want %v", err, parquet.ErrFormat)
		}
	})

	t.Run("a bloom filter that is not in the file", func(t *testing.T) {
		r := openFileReader(t, "bloom.parquet")
		r.Metadata().RowGroups[0].Columns[0].Meta.BloomFilterOffset = 1 << 40

		// The bounds keep the group, so the filter is read and the offset in
		// the footer is found to point past the end of the file.
		id := parquet.Where("id", kernel.OpEq, int64(1007))
		if _, err := r.RowGroups(id); !errors.Is(err, parquet.ErrFormat) {
			t.Errorf("got %v, want %v", err, parquet.ErrFormat)
		}
	})
}

// TestKeepNoBounds checks that a chunk whose writer said nothing about it is
// read.
//
// Most files written before the format had statistics are this, and so is any
// column of any file whose writer was told not to write them. There is nothing
// to skip on, so nothing is skipped.
func TestKeepNoBounds(t *testing.T) {
	p := parquet.Where("n", kernel.OpEq, int64(5))

	keep, err := p.Keep(parquet.Bounds{Count: 4})
	if err != nil {
		t.Fatalf("Keep: %v", err)
	}
	if !keep {
		t.Error("a chunk that said nothing about itself was skipped")
	}
}

// TestKeepOneValue checks the one range that rules out a not equal, which is a
// chunk whose values are all the same.
//
// It is the case a partition column is: a file of a day of orders holds the same
// date in every row, so a filter looking for another date reads none of it, and
// one looking for anything but that date reads none of it either.
func TestKeepOneValue(t *testing.T) {
	b := parquet.Bounds{Count: 4, Values: array.Of(int64(7), int64(7))}

	for _, c := range []struct {
		op   kernel.CompareOp
		v    int64
		want bool
	}{
		{kernel.OpNe, 7, false},
		{kernel.OpNe, 8, true},
		{kernel.OpEq, 7, true},
		{kernel.OpEq, 8, false},
		{kernel.OpLt, 7, false},
		{kernel.OpLe, 7, true},
		{kernel.OpGt, 7, false},
		{kernel.OpGe, 7, true},
	} {
		got, err := parquet.Where("n", c.op, c.v).Keep(b)
		if err != nil {
			t.Fatalf("Keep: %v", err)
		}
		if got != c.want {
			t.Errorf("a chunk of nothing but 7 with n %s %d gives %v, want %v",
				c.op, c.v, got, c.want)
		}
	}
}

// TestKeepOneFloat is the same chunk of one value on a column of floats, where
// the not equal that ruled it out cannot be trusted.
//
// A writer leaves NaN out of the bounds, so a chunk bounded by 7.0 either way
// holds nothing but sevens or holds sevens and a NaN, and the footer does not
// say which. A NaN is unequal to everything, so the second of those has a row
// the filter wanted and the chunk is read.
func TestKeepOneFloat(t *testing.T) {
	b := parquet.Bounds{Count: 4, Values: array.Of(7.0, 7.0)}

	for _, c := range []struct {
		op   kernel.CompareOp
		v    float64
		want bool
	}{
		{kernel.OpNe, 7.0, true},
		{kernel.OpEq, 7.0, true},
		{kernel.OpEq, 8.0, false},
		{kernel.OpGt, 7.0, false},
	} {
		got, err := parquet.Where("ratio", c.op, c.v).Keep(b)
		if err != nil {
			t.Fatalf("Keep: %v", err)
		}
		if got != c.want {
			t.Errorf("a chunk of nothing but 7.0 with ratio %s %v gives %v, want %v",
				c.op, c.v, got, c.want)
		}
	}
}

// TestKeepBadValue checks that a predicate the reader will not answer is refused
// by Keep as well as by RowGroups, since a caller walking the row groups itself
// reaches it the other way round.
func TestKeepBadValue(t *testing.T) {
	b := parquet.Bounds{Count: 4, Values: array.Of(int64(0), int64(3))}

	if _, err := (parquet.Predicate{Column: "n", Op: kernel.OpEq}).Keep(b); err == nil {
		t.Error("a predicate with no value worked")
	}
	if _, err := parquet.WhereString("n", kernel.OpEq, "five").Keep(b); err == nil {
		t.Error("a predicate against the wrong type worked")
	}
}

// BenchmarkRowGroups is what a filter costs before anything is read.
//
// The bounds are in the footer, which is in hand, so the answer for a file of
// three row groups is arithmetic on six numbers. The bloom filter case reads the
// file, which is what the second half of this measures against the first.
func BenchmarkRowGroups(b *testing.B) {
	b.Run("bounds", func(b *testing.B) {
		r := openFileReader(b, "stats.parquet")
		p := parquet.Where("n", kernel.OpGe, int64(8))
		b.ReportAllocs()
		for b.Loop() {
			if _, err := r.RowGroups(p); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("bloom", func(b *testing.B) {
		r := openFileReader(b, "bloom.parquet")
		p := parquet.Where("id", kernel.OpEq, int64(1004))
		b.ReportAllocs()
		for b.Loop() {
			if _, err := r.RowGroups(p); err != nil {
				b.Fatal(err)
			}
		}
	})
}
