package parquet_test

import (
	"bytes"
	"math"
	"strconv"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/parquet"
)

// reader is a table written and opened again, for the tests that ask what the
// footer says rather than what the columns hold.
func reader(tb testing.TB, t *array.Table, opts *parquet.WriteOptions) *parquet.FileReader {
	tb.Helper()

	var buf bytes.Buffer
	if _, err := parquet.Write(&buf, t, opts); err != nil {
		tb.Fatalf("Write: %v", err)
	}
	r, err := parquet.NewFileReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		tb.Fatalf("NewFileReader: %v", err)
	}
	return r
}

// bound is one end of a bound as text, using the same reading of a value as the
// round trip tests, so a failure names a value rather than a run of bytes.
func bound(tb testing.TB, b parquet.Bounds, i int) string {
	tb.Helper()

	if b.Values == nil {
		tb.Fatal("the chunk came back with no bounds, want a smallest and a largest value")
	}
	if b.Values.Len() != 2 {
		tb.Fatalf("the chunk came back with %d bounds, want 2", b.Values.Len())
	}
	return cell(b.Values, i)
}

// TestWriteBounds writes a column of every type and asks the footer what each
// of them holds.
//
// This is the round trip test again with the bounds instead of the values, and
// the values are the ends of each type's range for the same reason: a bound
// written at the wrong width or compared with the wrong sign is a bound that
// looks right until the largest value goes past the middle of the range. The
// unsigned columns are the ones that catch it. A uint32 of 4294967295 travels in
// an int32 as -1, so a writer comparing it signed would call it the smallest
// value of the chunk and a scan for large values would skip the group holding
// them.
func TestWriteBounds(t *testing.T) {
	want := []struct {
		column string
		lo, hi string
		nulls  int64
	}{
		{"flag", "false", "true", 0},
		{"small", "-128", "127", 1},
		{"short", "-32768", "32767", 0},
		{"count", "-1", "2147483647", 1},
		{"total", "-9223372036854775808", "9223372036854775807", 0},
		{"byte", "0", "255", 1},
		{"word", "0", "65535", 0},
		{"unsigned", "0", "4294967295", 1},
		{"big", "0", "18446744073709551615", 0},
		{"ratio", "-1.5", "3.25", 1},
		{"weight", "-1.5", "1e+300", 0},
		{"name", strconv.Quote(""), strconv.Quote("three"), 1},
		{"blob", strconv.Quote(""), strconv.Quote("\x03\x04\x05"), 0},
		{"hash", strconv.Quote("\x00\x00\x00\x00"), strconv.Quote("wxyz"), 1},
		{"day", "-1", "19000", 0},
		{"clock", "0", "86399999", 1},
		{"precise", "0", "86399999999999", 0},
		{"moment", "-1", "1700000000000000", 1},
		{"local", "-1", "1700000000000", 0},
	}

	r := reader(t, everyType(t), nil)
	b := boundsOf(t, r, 0)
	if len(b) != len(want) {
		t.Fatalf("%d columns of bounds, want %d", len(b), len(want))
	}

	for i, w := range want {
		if got := bound(t, b[i], 0); got != w.lo {
			t.Errorf("%s starts at %s, want %s", w.column, got, w.lo)
		}
		if got := bound(t, b[i], 1); got != w.hi {
			t.Errorf("%s ends at %s, want %s", w.column, got, w.hi)
		}

		// The bounds are values out of the chunk rather than values either side
		// of it, since nothing here is long enough to be worth cutting down.
		if !b[i].MinExact || !b[i].MaxExact {
			t.Errorf("%s says its bounds are not values out of the chunk", w.column)
		}
		if b[i].Count != 4 {
			t.Errorf("%s says it holds %d values, want the group's 4", w.column, b[i].Count)
		}
		if !b[i].HasNulls || b[i].Nulls != w.nulls {
			t.Errorf("%s counts %d missing values, want %d", w.column, b[i].Nulls, w.nulls)
		}
	}
}

// TestWriteOrders is the line in the footer that makes the bounds worth
// reading.
//
// A file that does not say how its columns compare has said nothing about what
// the pair of bounds on every chunk of it means, so a reader is right to fall
// back on the older pair and to ignore that one on every type the writers of
// the day disagreed about. That would be a file whose strings and whose wide
// unsigned integers are never skipped on, which is most of the skipping worth
// having, and the whole of the difference is one field per leaf.
func TestWriteOrders(t *testing.T) {
	_, raw := writtenBytes(t, everyType(t), nil)

	m, err := parquet.ReadMetadata(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatalf("ReadMetadata: %v", err)
	}

	columns, err := m.Columns()
	if err != nil {
		t.Fatalf("Columns: %v", err)
	}
	if len(m.Orders) != len(columns) {
		t.Fatalf("the footer gives an order to %d columns of %d", len(m.Orders), len(columns))
	}
	for _, c := range columns {
		if c.Order != parquet.TypeDefinedOrder {
			t.Errorf("%s compares by order %d, want the one the format defines", c.Name(), c.Order)
		}
	}
}

// TestWriteBoundsGroups writes a table in several row groups and asks each of
// them what it holds.
//
// The bounds of a chunk are the chunk's own, which is the whole of what makes
// them worth writing: a footer where every group covers the range of the file
// is a footer nothing can be skipped on. So this is the test of the one piece
// of state in the writer, and a tracker that kept what the group before it saw
// would come out of here with three groups that all start at 0.
func TestWriteBoundsGroups(t *testing.T) {
	want := [][2]string{{"0", "3"}, {"4", "7"}, {"8", "9"}}

	table := tableOf(t,
		[]dtype.Field{{Name: "n", Type: dtype.Int64}, {Name: "word", Type: dtype.String}},
		func(b *array.Builder) {
			for i := range 10 {
				b.Append(int64(i))
			}
		},
		func(b *array.Builder) {
			for i := range 10 {
				b.AppendString(strconv.Itoa(i))
			}
		},
	)

	r := reader(t, table, &parquet.WriteOptions{RowGroupSize: 4})
	if got := r.NumRowGroups(); got != len(want) {
		t.Fatalf("%d row groups, want %d", got, len(want))
	}

	for i, w := range want {
		b := boundsOf(t, r, i)
		if got := [2]string{bound(t, b[0], 0), bound(t, b[0], 1)}; got != w {
			t.Errorf("group %d holds n from %v, want %v", i, got, w)
		}
		words := [2]string{strconv.Quote(w[0]), strconv.Quote(w[1])}
		if got := [2]string{bound(t, b[1], 0), bound(t, b[1], 1)}; got != words {
			t.Errorf("group %d holds word from %v, want %v", i, got, words)
		}
	}
}

// TestWriteBoundsPages writes a chunk of more than one page.
//
// A chunk's bounds cover its whole chunk however many pages that took, so this
// is the case where the tracker is asked twice before it is read and the case
// where a writer that took the bounds a page at a time would write the last
// page's. There is nothing per page in the footer to write them to yet.
func TestWriteBoundsPages(t *testing.T) {
	const rows = 1000

	table := tableOf(t,
		[]dtype.Field{{Name: "n", Type: dtype.Int32, Nullable: true}},
		func(b *array.Builder) {
			for i := range rows {
				if i%3 == 0 {
					b.AppendNull()
					continue
				}
				b.Append(int32(rows - i))
			}
		},
	)

	r := reader(t, table, &parquet.WriteOptions{PageSize: 64})
	if got := r.NumRowGroups(); got != 1 {
		t.Fatalf("%d row groups, want 1", got)
	}

	b := boundsOf(t, r, 0)[0]
	if got, want := [2]string{bound(t, b, 0), bound(t, b, 1)}, [2]string{"2", "999"}; got != want {
		t.Errorf("n runs from %v, want %v", got, want)
	}
	if got, want := b.Nulls, int64(334); got != want {
		t.Errorf("n counts %d missing values, want %d", got, want)
	}
}

// TestWriteBoundsNulls writes the columns there is nothing to say about.
//
// A column that is missing everywhere has no smallest value, and a column of
// the null type has no values at all. Both come back bounded by nothing and
// counted as all null, which is what a filter of any kind skips a group on, and
// it is the count rather than the bounds that says so. A boolean gets a column
// of its own here because it is tracked its own way, there being two values it
// can hold and no comparison to do.
func TestWriteBoundsNulls(t *testing.T) {
	table := tableOf(t,
		[]dtype.Field{
			{Name: "missing", Type: dtype.Int64, Nullable: true},
			{Name: "nothing", Type: dtype.Null, Nullable: true},
			{Name: "gone", Type: dtype.Bool, Nullable: true},
			{Name: "present", Type: dtype.Int64, Nullable: true},
			{Name: "flag", Type: dtype.Bool, Nullable: true},
		},
		func(b *array.Builder) { b.AppendNulls(5) },
		func(b *array.Builder) { b.AppendNulls(5) },
		func(b *array.Builder) { b.AppendNulls(5) },
		func(b *array.Builder) { b.AppendValues([]int64{1, 2, 3, 4, 5}) },
		func(b *array.Builder) {
			b.AppendNull()
			b.AppendBools([]bool{true, false, true})
			b.AppendNull()
		},
	)

	b := boundsOf(t, reader(t, table, nil), 0)
	for i, name := range []string{"missing", "nothing", "gone"} {
		if b[i].Values != nil {
			t.Errorf("%s came back bounded and every value of it is missing", name)
		}
		if !b[i].AllNull() || b[i].Nulls != 5 {
			t.Errorf("%s counts %d missing values of %d", name, b[i].Nulls, b[i].Count)
		}
	}

	if got, want := [2]string{bound(t, b[3], 0), bound(t, b[3], 1)}, [2]string{"1", "5"}; got != want {
		t.Errorf("present runs from %v, want %v", got, want)
	}
	if b[3].AllNull() || !b[3].HasNulls || b[3].Nulls != 0 {
		t.Error("present says something is missing from it")
	}

	if got, want := [2]string{bound(t, b[4], 0), bound(t, b[4], 1)}, [2]string{"false", "true"}; got != want {
		t.Errorf("flag runs from %v, want %v", got, want)
	}
	if b[4].Nulls != 2 {
		t.Errorf("flag counts %d missing values, want 2", b[4].Nulls)
	}
}

// TestWriteBoundsNaN leaves a NaN out of the bounds it would otherwise be.
//
// NaN compares false against everything, itself included, so a chunk bounded by
// one is a chunk no filter can be kept out of, and a reader is right to throw
// such bounds away. A writer that let one through would be writing a file that
// loses the skipping on that column, which is why it is left out here instead.
// A column of nothing but NaN then has no bounds at all, and none of its values
// is missing, so this is the one case where a chunk is bounded by nothing
// without being all null.
func TestWriteBoundsNaN(t *testing.T) {
	table := tableOf(t,
		[]dtype.Field{
			{Name: "ratio", Type: dtype.Float32},
			{Name: "weight", Type: dtype.Float64},
			{Name: "broken", Type: dtype.Float64},
		},
		func(b *array.Builder) {
			b.AppendValues([]float32{-1.5, float32(math.NaN()), 3.25})
		},
		func(b *array.Builder) {
			b.AppendValues([]float64{math.NaN(), 2, math.Inf(-1)})
		},
		func(b *array.Builder) {
			b.AppendValues([]float64{math.NaN(), math.NaN(), math.NaN()})
		},
	)

	b := boundsOf(t, reader(t, table, nil), 0)
	if got, want := [2]string{bound(t, b[0], 0), bound(t, b[0], 1)}, [2]string{"-1.5", "3.25"}; got != want {
		t.Errorf("ratio runs from %v, want %v", got, want)
	}
	if got, want := [2]string{bound(t, b[1], 0), bound(t, b[1], 1)}, [2]string{"-Inf", "2"}; got != want {
		t.Errorf("weight runs from %v, want %v", got, want)
	}

	if b[2].Values != nil {
		t.Error("a column of nothing but NaN came back bounded")
	}
	if b[2].AllNull() || b[2].Nulls != 0 {
		t.Errorf("a column of nothing but NaN counts %d of its values missing", b[2].Nulls)
	}
}

// TestWriteSkipped reads a file this writer produced and skips most of it.
//
// This is what the statistics are for, and it is the only test here that says
// so end to end. A filter for one value reads the footer, finds that two of the
// three groups hold ranges it cannot be in, and reads the one that is left, so
// what it costs is the footer and a third of the file. Before the bounds went
// in, the same read on the same file read all of it and got the same answer,
// which is why the byte count is checked rather than the rows alone.
func TestWriteSkipped(t *testing.T) {
	const rows = 300

	table := tableOf(t,
		[]dtype.Field{{Name: "id", Type: dtype.Int64}, {Name: "word", Type: dtype.String}},
		func(b *array.Builder) {
			for i := range rows {
				b.Append(int64(i))
			}
		},
		func(b *array.Builder) {
			for i := range rows {
				b.AppendString(names[i%len(names)])
			}
		},
	)

	var buf bytes.Buffer
	if _, err := parquet.Write(&buf, table, &parquet.WriteOptions{RowGroupSize: 100}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	raw := bytes.NewReader(buf.Bytes())
	size := int64(buf.Len())

	r, err := parquet.NewFileReader(raw, size)
	if err != nil {
		t.Fatalf("NewFileReader: %v", err)
	}
	groups, err := r.RowGroups(parquet.Where("id", kernel.OpEq, int64(150)))
	if err != nil {
		t.Fatalf("RowGroups: %v", err)
	}
	if len(groups) != 1 || groups[0] != 1 {
		t.Fatalf("a filter for one value left %v to read, want the second group alone", groups)
	}

	got, err := parquet.Read(raw, size, &parquet.Options{
		Columns: []string{"id"},
		Filter:  []parquet.Predicate{parquet.Where("id", kernel.OpEq, int64(150))},
	})
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if got.NumRows() != 1 {
		t.Fatalf("%d rows came back, want the one that matches", got.NumRows())
	}
	if v := cells(got.Columns[0])[0]; v != "150" {
		t.Errorf("row %s came back, want 150", v)
	}

	// The whole file with no filter, for the comparison. What is being checked
	// is that the filtered read left most of it alone rather than reading it
	// and throwing the rows away.
	whole, err := parquet.NewFileReader(raw, size)
	if err != nil {
		t.Fatalf("NewFileReader: %v", err)
	}
	if err := whole.Project("id"); err != nil {
		t.Fatalf("Project: %v", err)
	}
	for i := range whole.NumRowGroups() {
		if _, err := whole.RowGroup(i); err != nil {
			t.Fatalf("RowGroup(%d): %v", i, err)
		}
	}

	if err := r.Project("id"); err != nil {
		t.Fatalf("Project: %v", err)
	}
	if _, err := r.RowGroup(groups[0]); err != nil {
		t.Fatalf("RowGroup(%d): %v", groups[0], err)
	}
	if r.BytesRead() >= whole.BytesRead() {
		t.Errorf("a filtered read took %d bytes and the whole file takes %d",
			r.BytesRead(), whole.BytesRead())
	}
}
