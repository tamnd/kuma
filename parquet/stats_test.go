package parquet_test

import (
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/parquet"
)

// boundsOf reads what a writer said about one row group and fails the test if
// it cannot.
func boundsOf(tb testing.TB, r *parquet.FileReader, i int) []parquet.Bounds {
	tb.Helper()

	b, err := r.Bounds(i)
	if err != nil {
		tb.Fatalf("Bounds(%d): %v", i, err)
	}
	return b
}

// numberBounds is the two numbers a bound holds, which is what a filter on a
// numeric column compares itself against.
func numberBounds[T array.Numeric](tb testing.TB, b parquet.Bounds) [2]T {
	tb.Helper()

	if b.Values == nil || b.Values.Len() != 2 {
		tb.Fatal("the chunk came back with no bounds, want a smallest and a largest value")
	}
	v := numberColumn[T](b.Values)
	return [2]T{v[0], v[1]}
}

// textBounds is the same for a column of strings.
func textBounds(tb testing.TB, b parquet.Bounds) [2]string {
	tb.Helper()

	if b.Values == nil || b.Values.Len() != 2 {
		tb.Fatal("the chunk came back with no bounds, want a smallest and a largest value")
	}
	v := textColumn(b.Values)
	return [2]string{v[0], v[1]}
}

// withOrder puts the order the format defines on a column, which is what every
// file written this decade says about every column it holds.
func withOrder(c parquet.Column) parquet.Column {
	c.Order = parquet.TypeDefinedOrder
	return c
}

// TestReadBounds is what a writer said about every column of a file written to
// have something worth saying.
//
// The three row groups hold ranges of their own, which is what a scan skips on,
// and the columns are the ones that are easy to read wrongly. The unsigned one
// holds a value that is negative when it is read signed, and comparing it that
// way would put the largest value of the group in front of the smallest. The
// string one starts at the empty string, which is a value and not a missing
// bound. The float one has a NaN among its values, which the writer left out of
// its bounds the way the format tells it to, so the group is bounded by the
// values either side of it.
func TestReadBounds(t *testing.T) {
	tests := []struct {
		group int
		n     [2]int64
		word  [2]string
		size  [2]uint32
		ratio [2]float64
	}{
		{0, [2]int64{0, 3}, [2]string{"", "echo"}, [2]uint32{1, math.MaxUint32}, [2]float64{0.5, 3.5}},
		{1, [2]int64{4, 7}, [2]string{"mike", "papa"}, [2]uint32{10, 13}, [2]float64{1, 3}},
		{2, [2]int64{8, 11}, [2]string{"sierra", "zulu"}, [2]uint32{20, 23}, [2]float64{-1, 2}},
	}

	r := openFileReader(t, "stats.parquet")
	for _, tt := range tests {
		b := boundsOf(t, r, tt.group)
		if len(b) != 6 {
			t.Fatalf("group %d came back with %d columns of bounds, want 6", tt.group, len(b))
		}

		if got := numberBounds[int64](t, b[0]); got != tt.n {
			t.Errorf("group %d holds n from %v, want %v", tt.group, got, tt.n)
		}
		if got := textBounds(t, b[1]); got != tt.word {
			t.Errorf("group %d holds word from %q, want %q", tt.group, got, tt.word)
		}
		if got := numberBounds[uint32](t, b[2]); got != tt.size {
			t.Errorf("group %d holds size from %v, want %v", tt.group, got, tt.size)
		}
		if got := numberBounds[float64](t, b[3]); got != tt.ratio {
			t.Errorf("group %d holds ratio from %v, want %v", tt.group, got, tt.ratio)
		}
		if b[4].Values != nil {
			t.Errorf("group %d bounds a column that is missing everywhere", tt.group)
		}
		if b[5].Values == nil || b[5].Values.Bool(0) || !b[5].Values.Bool(1) {
			t.Errorf("group %d holds flag from %v, want false to true", tt.group, b[5].Values)
		}
	}
}

// TestReadBoundsCounts is what a chunk says about how many values it holds and
// how many of them are missing, which is the half of the statistics that needs
// no ordering to be useful.
func TestReadBoundsCounts(t *testing.T) {
	r := openFileReader(t, "stats.parquet")
	b := boundsOf(t, r, 0)

	for i, x := range b {
		if x.Count != 4 {
			t.Errorf("column %d says it holds %d values, want the group's 4", i, x.Count)
		}
		if !x.HasNulls {
			t.Errorf("column %d does not say how many values are missing", i)
		}
	}

	// The column that is missing everywhere is the one a filter of any kind
	// skips, since none of its values is anything.
	if !b[4].AllNull() || b[4].Nulls != 4 {
		t.Errorf("the column that is missing everywhere counts %d nulls of %d", b[4].Nulls, b[4].Count)
	}
	if b[0].AllNull() || b[0].Nulls != 0 {
		t.Errorf("a column with nothing missing counts %d nulls", b[0].Nulls)
	}

	// The bounds pyarrow writes are values out of the chunk. A truncated one
	// would still bound the chunk and would not answer a question about the
	// value itself.
	if !b[0].MinExact || !b[0].MaxExact {
		t.Error("the bounds of n are not values out of the chunk, want them exact")
	}
	if b[4].MinExact || b[4].MaxExact {
		t.Error("a column with no bounds says its bounds are exact")
	}
}

// TestReadBoundsEmpty is a row group of no rows.
//
// Every filter skips it, which falls out of the counts rather than out of the
// bounds: a chunk holding nothing holds nothing that matches. The writer wrote
// no statistics at all for it, having nothing to say, so this is also the case
// where a chunk counts as all null without a null count on it.
func TestReadBoundsEmpty(t *testing.T) {
	r := openFileReader(t, "empty.parquet")

	for i, b := range boundsOf(t, r, 0) {
		if b.Values != nil {
			t.Errorf("column %d of a group of no rows has bounds", i)
		}
		if b.Count != 0 || !b.AllNull() {
			t.Errorf("column %d of a group of no rows holds %d values, %d of them missing",
				i, b.Count, b.Nulls)
		}
	}
}

// TestReadBoundsDeprecated reads the pair of bounds the format left behind.
//
// A file that did not say how its columns compare has said nothing about what
// the pair that came with the question means, so a reader is back on the pair
// that came before it, and on the types whose order nobody ever disagreed
// about. The file read here writes both pairs and writes the old one for
// exactly the types that rule allows, which is the same rule from the writer's
// side of it.
func TestReadBoundsDeprecated(t *testing.T) {
	m := read(t, "stats.parquet")
	columns, err := m.Columns()
	if err != nil {
		t.Fatalf("Columns: %v", err)
	}
	group := &m.RowGroups[0]

	for i, want := range []bool{true, false, false, true, false, true} {
		if got := group.Columns[i].Meta.Stats.Min != nil; got != want {
			t.Errorf("the writer wrote the old bounds of %s: %v, want %v",
				columns[i].Name(), got, want)
		}
	}

	// A file that left the orders out, which is every file written before the
	// format had them and none written since.
	for i := range columns {
		columns[i].Order = parquet.UndefinedOrder
	}

	t.Run("an integer keeps its old bounds", func(t *testing.T) {
		b, err := parquet.ReadBounds(columns[0], &group.Columns[0].Meta)
		if err != nil {
			t.Fatalf("ReadBounds: %v", err)
		}
		if got, want := numberBounds[int64](t, b), [2]int64{0, 3}; got != want {
			t.Errorf("n runs from %v, want %v", got, want)
		}
		if !b.MinExact || !b.MaxExact {
			t.Error("the old bounds are not exact, and nothing ever truncated one")
		}
	})

	t.Run("a string does not", func(t *testing.T) {
		meta := group.Columns[1].Meta
		meta.Stats.Min, meta.Stats.Max = []byte("mike"), []byte("papa")

		b, err := parquet.ReadBounds(columns[1], &meta)
		if err != nil {
			t.Fatalf("ReadBounds: %v", err)
		}
		if b.Values != nil {
			t.Error("a string was bounded by a comparison the writers of the day disagreed on")
		}
	})

	t.Run("nor does a wide unsigned integer", func(t *testing.T) {
		meta := group.Columns[2].Meta
		meta.Stats.Min = binary.LittleEndian.AppendUint32(nil, 1)
		meta.Stats.Max = binary.LittleEndian.AppendUint32(nil, math.MaxUint32)

		b, err := parquet.ReadBounds(columns[2], &meta)
		if err != nil {
			t.Fatalf("ReadBounds: %v", err)
		}
		if b.Values != nil {
			t.Error("a uint32 was bounded by a signed comparison, which reads its largest value as -1")
		}
	})

	// A uint16 goes in an int32 that is never negative, so the comparison the
	// old writers used puts it in the order an unsigned one would.
	t.Run("a narrow one does", func(t *testing.T) {
		c := parquet.Column{
			Path:    []string{"small"},
			Element: parquet.SchemaElement{Name: "small", Type: parquet.Int32, Repetition: parquet.Required},
			Type:    dtype.Uint16,
		}
		meta := parquet.ColumnMeta{NumValues: 4, Stats: parquet.Statistics{
			Min: binary.LittleEndian.AppendUint32(nil, 7),
			Max: binary.LittleEndian.AppendUint32(nil, math.MaxUint16),
		}}

		b, err := parquet.ReadBounds(c, &meta)
		if err != nil {
			t.Fatalf("ReadBounds: %v", err)
		}
		if got, want := numberBounds[uint16](t, b), [2]uint16{7, math.MaxUint16}; got != want {
			t.Errorf("small runs from %v, want %v", got, want)
		}
	})
}

// TestReadBoundsUnordered is the columns nothing can be said about, whatever
// their chunks claim.
//
// An int96 is two numbers in twelve bytes and the writers that wrote them
// compared the bytes, an interval is months and days and milliseconds and no
// two of those can be put on one line, and a column of nothing has nothing to
// order. A column with no type at all is not a column.
func TestReadBoundsUnordered(t *testing.T) {
	legacy := read(t, "legacy.parquet")
	columns, err := legacy.Columns()
	if err != nil {
		t.Fatalf("Columns: %v", err)
	}

	interval := parquet.Column{
		Path: []string{"span"},
		Element: parquet.SchemaElement{
			Name: "span", Type: parquet.FixedLenByteArray,
			Repetition: parquet.Required, TypeLength: 12, Converted: parquet.ConvertedInterval,
		},
		Type: dtype.Interval{Unit: dtype.MonthDayNano},
	}
	blank := missing()
	blank.Type = nil

	cases := []struct {
		name   string
		column parquet.Column
	}{
		{"a timestamp written as an int96", columns[0]},
		{"an interval", withOrder(interval)},
		{"a column of nothing", withOrder(missing())},
		{"a column of no type at all", withOrder(blank)},
	}

	twelve := make([]byte, 12)
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			meta := parquet.ColumnMeta{NumValues: 2, Stats: parquet.Statistics{
				MinValue: twelve, MaxValue: twelve, Min: twelve, Max: twelve,
			}}

			b, err := parquet.ReadBounds(c.column, &meta)
			if err != nil {
				t.Fatalf("ReadBounds: %v", err)
			}
			if b.Values != nil {
				t.Error("a column with no order came back bounded")
			}
		})
	}
}

// TestReadBoundsNaN drops the bounds of a chunk bounded by a NaN.
//
// NaN compares false against everything, itself included, so a chunk bounded by
// one has no range for a filter to be kept out of and reading it as a bound
// would skip row groups holding rows the filter wanted. The writers leave NaN
// out of their bounds, which is why this has to be written in by hand to be
// read at all.
func TestReadBoundsNaN(t *testing.T) {
	double := binary.LittleEndian.AppendUint64(nil, math.Float64bits(math.NaN()))
	single := binary.LittleEndian.AppendUint32(nil, math.Float32bits(float32(math.NaN())))

	t.Run("a double", func(t *testing.T) {
		m := read(t, "stats.parquet")
		columns, err := m.Columns()
		if err != nil {
			t.Fatalf("Columns: %v", err)
		}

		meta := m.RowGroups[0].Columns[3].Meta
		meta.Stats.MinValue = double

		b, err := parquet.ReadBounds(columns[3], &meta)
		if err != nil {
			t.Fatalf("ReadBounds: %v", err)
		}
		if b.Values != nil {
			t.Error("a chunk whose smallest value is a NaN came back bounded")
		}
	})

	// The float column of a file that has one of every type, which is bounded
	// until the NaN goes in.
	t.Run("a float", func(t *testing.T) {
		m := read(t, "alltypes.parquet")
		columns, err := m.Columns()
		if err != nil {
			t.Fatalf("Columns: %v", err)
		}

		meta := m.RowGroups[0].Columns[5].Meta
		if columns[5].Name() != "ratio" {
			t.Fatalf("column 5 is %s, want ratio", columns[5].Name())
		}

		b, err := parquet.ReadBounds(columns[5], &meta)
		if err != nil {
			t.Fatalf("ReadBounds: %v", err)
		}
		if b.Values == nil {
			t.Fatal("a float column with bounds came back with none")
		}

		meta.Stats.MaxValue = single
		if b, err = parquet.ReadBounds(columns[5], &meta); err != nil {
			t.Fatalf("ReadBounds: %v", err)
		}
		if b.Values != nil {
			t.Error("a chunk whose largest value is a NaN came back bounded")
		}
	})
}

// TestReadBoundsRefused is the footers that contradict themselves and the
// bounds this package cannot read.
//
// A bound is what a row group is skipped on, and skipping the wrong one changes
// an answer without saying so, which is why none of these is read as far as it
// will go and then acted on.
func TestReadBoundsRefused(t *testing.T) {
	m := read(t, "stats.parquet")
	columns, err := m.Columns()
	if err != nil {
		t.Fatalf("Columns: %v", err)
	}
	n := m.RowGroups[0].Columns[0].Meta

	short := n
	short.Stats.MinValue = []byte{1}

	long := n
	long.Stats.MaxValue = make([]byte, 12)

	overcount := n
	overcount.Stats.NullCount = 5

	negative := n
	negative.Stats.NullCount = -1

	price := parquet.ColumnMeta{NumValues: 2, Stats: parquet.Statistics{
		MinValue: make([]byte, 4), MaxValue: make([]byte, 4),
	}}

	cases := []struct {
		name   string
		column parquet.Column
		meta   parquet.ColumnMeta
		want   error
	}{
		{"more missing values than values", columns[0], overcount, parquet.ErrFormat},
		{"a count of missing values below nought", columns[0], negative, parquet.ErrFormat},
		{"a bound too short to be a value", columns[0], short, parquet.ErrFormat},
		{"a bound with more in it than one value", columns[0], long, parquet.ErrFormat},
		{"a type that is not assembled yet", withOrder(decimal()), price, parquet.ErrUnsupported},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := parquet.ReadBounds(c.column, &c.meta); !errors.Is(err, c.want) {
				t.Errorf("got %v, want %v", err, c.want)
			}
		})
	}

	// A column of a type nothing can be built for, which the builder is the one
	// that knows. The error is its rather than this package's, so it is the
	// column being named that says it came from here.
	t.Run("a type nothing can be built for", func(t *testing.T) {
		c := parquet.Column{
			Path:    []string{"wide"},
			Element: parquet.SchemaElement{Name: "wide", Type: parquet.ByteArray, Repetition: parquet.Required},
			Type:    dtype.LargeString,
			Order:   parquet.TypeDefinedOrder,
		}
		meta := parquet.ColumnMeta{NumValues: 2, Stats: parquet.Statistics{
			MinValue: []byte("a"), MaxValue: []byte("z"),
		}}

		_, err := parquet.ReadBounds(c, &meta)
		if err == nil {
			t.Fatal("a column nothing can be built for was bounded")
		}
		if !strings.Contains(err.Error(), "wide") {
			t.Errorf("%v does not say which column it was", err)
		}
	})
}

// TestFileReaderBounds asks a row group what it holds without reading any of
// it.
func TestFileReaderBounds(t *testing.T) {
	r := openFileReader(t, "stats.parquet")
	if err := r.Project("word", "n"); err != nil {
		t.Fatalf("Project: %v", err)
	}

	b := boundsOf(t, r, 2)
	if len(b) != 2 {
		t.Fatalf("%d columns of bounds, want the two that were projected", len(b))
	}
	if got, want := textBounds(t, b[0]), [2]string{"sierra", "zulu"}; got != want {
		t.Errorf("word runs from %q, want %q", got, want)
	}
	if got, want := numberBounds[int64](t, b[1]), [2]int64{8, 11}; got != want {
		t.Errorf("n runs from %v, want %v", got, want)
	}

	// None of that came out of the file. The bounds are in the footer, which is
	// what was read to open it.
	if got, want := r.BytesRead(), footerOf(t, "stats.parquet"); got != want {
		t.Errorf("asking read %d bytes, want the footer's %d", got, want)
	}

	// A reader narrowed to nothing has nothing to say, which is what counting
	// the rows of a file costs.
	if err := r.Project(); err != nil {
		t.Fatalf("Project: %v", err)
	}
	if got := boundsOf(t, r, 0); len(got) != 0 {
		t.Errorf("a reader of no columns came back with %d of them", len(got))
	}
}

// TestFileReaderBoundsRefused is the row groups a reader will not answer for.
func TestFileReaderBoundsRefused(t *testing.T) {
	t.Run("a row group the file does not have", func(t *testing.T) {
		r := openFileReader(t, "stats.parquet")
		for _, i := range []int{-1, 3} {
			if _, err := r.Bounds(i); err == nil {
				t.Errorf("row group %d of a file of three was answered for", i)
			}
		}
	})

	t.Run("a row group holding no chunk for a column", func(t *testing.T) {
		r := openFileReader(t, "stats.parquet")
		r.Metadata().RowGroups[0].Columns = nil

		if _, err := r.Bounds(0); !errors.Is(err, parquet.ErrFormat) {
			t.Errorf("got %v, want %v", err, parquet.ErrFormat)
		}
	})

	t.Run("a chunk that contradicts itself", func(t *testing.T) {
		r := openFileReader(t, "stats.parquet")
		r.Metadata().RowGroups[0].Columns[0].Meta.Stats.NullCount = 99

		if _, err := r.Bounds(0); !errors.Is(err, parquet.ErrFormat) {
			t.Errorf("got %v, want %v", err, parquet.ErrFormat)
		}
	})
}

// TestBoundsSkip reads one row group of three, the other two being skipped on
// what the footer said about them.
//
// This is what the whole of it is for. The filter is one value of one column,
// the groups hold ranges of their own, and the two that cannot hold the value
// are never opened: the bytes that came out of the file are the footer and one
// column chunk, and not a page of the rest.
func TestBoundsSkip(t *testing.T) {
	const want = 6

	r := openFileReader(t, "stats.parquet")
	if err := r.Project("n"); err != nil {
		t.Fatalf("Project: %v", err)
	}

	rows, read := 0, 0
	for i := range r.NumRowGroups() {
		b := boundsOf(t, r, i)[0]
		if b.AllNull() {
			continue
		}
		if lo := numberBounds[int64](t, b); want < lo[0] || want > lo[1] {
			continue
		}

		read++
		batch := rowGroup(t, r, i)
		for _, v := range numberColumn[int64](batch.Columns[0]) {
			if v == want {
				rows++
			}
		}
	}

	if read != 1 || rows != 1 {
		t.Errorf("found %d rows holding %d in %d row groups, want one of each", rows, want, read)
	}

	cost := footerOf(t, "stats.parquet") + r.Metadata().RowGroups[1].Columns[0].Meta.TotalCompressedSize
	if got := r.BytesRead(); got != cost {
		t.Errorf("the scan read %d bytes, want the footer and one chunk, which is %d", got, cost)
	}
}

// BenchmarkBounds is the cost of deciding not to read a row group against the
// cost of reading it.
//
// The two are the same call over the same three columns of the same row group,
// one of them off the footer that is already in memory and the other off the
// file. Five hundred rows is a small group, so the ratio between them is the
// least a real file would show.
func BenchmarkBounds(b *testing.B) {
	b.Run("the bounds", func(b *testing.B) {
		r := openFileReader(b, "pages.parquet")

		b.ReportAllocs()
		for b.Loop() {
			if _, err := r.Bounds(0); err != nil {
				b.Fatalf("Bounds: %v", err)
			}
		}
	})

	b.Run("the values", func(b *testing.B) {
		r := openFileReader(b, "pages.parquet")

		b.ReportAllocs()
		for b.Loop() {
			if _, err := r.RowGroup(0); err != nil {
				b.Fatalf("RowGroup: %v", err)
			}
		}
	})
}
