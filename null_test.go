package kuma_test

import (
	"errors"
	"testing"
	"time"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// gaps returns a frame with a hole in three different places, so that no two
// columns are missing the same rows and a rule about several columns at once
// has something to be wrong about.
//
//	sym qty px
//	.   1   1.5
//	b   .   2.5
//	c   3   3.5
//	.   .   4.5
func gaps(t *testing.T) *kuma.Frame[kuma.Dynamic] {
	t.Helper()

	f, err := kuma.NewFrame(
		nullKeys(t, "", "b", "c", "").Rename("sym"),
		nullInts(t, 1, 0, 3, 0),
		kuma.NewSeries("px", 1.5, 2.5, 3.5, 4.5).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	return f
}

// nullInts returns an int64 column called qty where a zero means the value is
// missing. Rename it when a test wants it called something else.
func nullInts(t *testing.T, vals ...int64) kuma.Column {
	t.Helper()

	b, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	for _, v := range vals {
		if v == 0 {
			b.AppendNull()
			continue
		}
		b.Append(v)
	}

	data, err := array.NewChunked(dtype.Int64, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	c, err := kuma.NewColumn("qty", data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	return c
}

// bits renders a boolean column as a string of t and f, and fails if it has any
// nulls, because a null check is never itself in doubt.
func bits(t *testing.T, c kuma.Column) string {
	t.Helper()

	if c.NullCount() != 0 {
		t.Fatalf("the mask has %d nulls in it, want none", c.NullCount())
	}

	out := make([]byte, c.Len())
	for i := range out {
		out[i] = 'f'
		if c.Data().Bool(i) {
			out[i] = 't'
		}
	}
	return string(out)
}

func TestColumnNullMask(t *testing.T) {
	c, err := gaps(t).Column("sym")
	if err != nil {
		t.Fatalf("Column: %v", err)
	}

	if got := bits(t, c.NullMask()); got != "tfft" {
		t.Errorf("NullMask gave %q, want %q", got, "tfft")
	}
	if got := bits(t, c.ValidMask()); got != "fttf" {
		t.Errorf("ValidMask gave %q, want %q", got, "fttf")
	}
	if got := c.NullMask().Name(); got != "sym" {
		t.Errorf("the mask is named %q, want it to keep the name %q", got, "sym")
	}
}

func TestColumnHasNulls(t *testing.T) {
	f := gaps(t)

	for _, name := range []string{"sym", "qty"} {
		c, err := f.Column(name)
		if err != nil {
			t.Fatalf("Column: %v", err)
		}
		if !c.HasNulls() {
			t.Errorf("column %q says it has no nulls, but it has %d", name, c.NullCount())
		}
	}

	px, err := f.Column("px")
	if err != nil {
		t.Fatalf("Column: %v", err)
	}
	if px.HasNulls() {
		t.Error("the px column says it has nulls, but every value is there")
	}
}

func TestColumnFillNull(t *testing.T) {
	c, err := gaps(t).Column("qty")
	if err != nil {
		t.Fatalf("Column: %v", err)
	}

	filled, err := c.FillNull(int64(7))
	if err != nil {
		t.Fatalf("FillNull: %v", err)
	}
	if filled.HasNulls() {
		t.Errorf("the filled column still has %d nulls", filled.NullCount())
	}

	want := []int64{1, 7, 3, 7}
	got := filled.MustAs[int64]().Values()
	for i, v := range want {
		if got[i] != v {
			t.Errorf("row %d is %d, want %d", i, got[i], v)
		}
	}

	if c.NullCount() != 2 {
		t.Errorf("filling changed the column it was called on, which now has %d nulls", c.NullCount())
	}
}

func TestColumnFillNullNothingToDo(t *testing.T) {
	c, err := gaps(t).Column("px")
	if err != nil {
		t.Fatalf("Column: %v", err)
	}

	filled, err := c.FillNull(0.0)
	if err != nil {
		t.Fatalf("FillNull: %v", err)
	}
	if filled.Data() != c.Data() {
		t.Error("filling a column with nothing missing built a new column, want the one it was given")
	}
}

func TestColumnFillNullWrongType(t *testing.T) {
	c, err := gaps(t).Column("sym")
	if err != nil {
		t.Fatalf("Column: %v", err)
	}

	_, err = c.FillNull(int64(7))
	if !errors.Is(err, kuma.ErrWrongType) {
		t.Errorf("filling a string column with an int64 gave %v, want ErrWrongType", err)
	}
}

// TestColumnFillNullEveryType is the check that the fill has no per type code
// in it. A string and a timestamp go through the same gather as an int64 does.
func TestColumnFillNullEveryType(t *testing.T) {
	sym, err := gaps(t).Column("sym")
	if err != nil {
		t.Fatalf("Column: %v", err)
	}

	filled, err := sym.FillNull("z")
	if err != nil {
		t.Fatalf("FillNull: %v", err)
	}
	for i, want := range []string{"z", "b", "c", "z"} {
		if got := string(filled.Data().Bytes(i)); got != want {
			t.Errorf("row %d is %q, want %q", i, got, want)
		}
	}
}

// TestColumnFillNullTimestamp is the one type that needs work, because a
// time.Time is nanoseconds and the column may be counting something else.
func TestColumnFillNullTimestamp(t *testing.T) {
	when := time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)

	for _, unit := range []dtype.TimeUnit{
		dtype.Second, dtype.Millisecond, dtype.Microsecond, dtype.Nanosecond,
	} {
		ts := dtype.Timestamp{Unit: unit}

		b, err := array.NewBuilder(ts)
		if err != nil {
			t.Fatalf("NewBuilder: %v", err)
		}
		b.AppendNull()

		data, err := array.NewChunked(ts, b.Finish())
		if err != nil {
			t.Fatalf("NewChunked: %v", err)
		}
		c, err := kuma.NewColumn("at", data)
		if err != nil {
			t.Fatalf("NewColumn: %v", err)
		}

		filled, err := c.FillNull(when)
		if err != nil {
			t.Fatalf("FillNull on a %s column: %v", ts, err)
		}
		if got := filled.MustAs[time.Time]().Value(0); !got.Equal(when) {
			t.Errorf("a %s column filled with %s reads back as %s", ts, when, got)
		}
	}
}

func TestColumnDropNulls(t *testing.T) {
	c, err := gaps(t).Column("qty")
	if err != nil {
		t.Fatalf("Column: %v", err)
	}

	short := c.DropNulls()
	if short.Len() != 2 {
		t.Fatalf("dropping the nulls left %d rows, want 2", short.Len())
	}
	for i, want := range []int64{1, 3} {
		if got := short.MustAs[int64]().Value(i); got != want {
			t.Errorf("row %d is %d, want %d", i, got, want)
		}
	}
}

func TestSeriesNullMask(t *testing.T) {
	s, err := gaps(t).Series[int64]("qty")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}

	if !s.HasNulls() {
		t.Error("the series says it has no nulls, but two values are missing")
	}
	if got := bits(t, s.NullMask().Column()); got != "ftft" {
		t.Errorf("NullMask gave %q, want %q", got, "ftft")
	}
	if got := bits(t, s.ValidMask().Column()); got != "tftf" {
		t.Errorf("ValidMask gave %q, want %q", got, "tftf")
	}

	filled := s.FillNull(9)
	if got := filled.Values(); got[1] != 9 || got[3] != 9 || got[0] != 1 {
		t.Errorf("FillNull gave %v, want [1 9 3 9]", got)
	}
	if filled.Name() != "qty" {
		t.Errorf("the filled series is named %q, want %q", filled.Name(), "qty")
	}

	if got := s.DropNulls().Values(); len(got) != 2 || got[0] != 1 || got[1] != 3 {
		t.Errorf("DropNulls gave %v, want [1 3]", got)
	}
}

// TestSeriesValidMaskFilters is the reason the mask is a Series[bool] and not a
// slice of bool. It goes straight back into Filter.
func TestSeriesValidMaskFilters(t *testing.T) {
	f := gaps(t)

	qty, err := f.Series[int64]("qty")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}

	got, err := f.FilterMask(qty.ValidMask())
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	// The rows where qty is there, whatever the other columns hold.
	want := []string{". 1 1.5", "c 3 3.5"}
	if lines := rows(t, got); !equalLines(lines, want) {
		t.Errorf("filtering on the valid mask gave %q, want %q", lines, want)
	}
}

func TestFrameHasNulls(t *testing.T) {
	if !gaps(t).HasNulls() {
		t.Error("the frame says it has no nulls, but three values are missing")
	}

	full, err := gaps(t).Select("px")
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if full.HasNulls() {
		t.Error("a frame of one complete column says it has nulls")
	}
}

func TestFrameNullCounts(t *testing.T) {
	got := gaps(t).NullCounts()
	want := []int{2, 2, 0}

	if len(got) != len(want) {
		t.Fatalf("NullCounts gave %d numbers, want %d", len(got), len(want))
	}
	for i, n := range want {
		if got[i] != n {
			t.Errorf("column %d has %d nulls, want %d", i, got[i], n)
		}
	}
}

func TestFrameIsNull(t *testing.T) {
	mask := gaps(t).IsNull()

	want := []string{"tfft", "ftft", "ffff"}
	if got := columnBits(t, mask); !equalLines(got, want) {
		t.Errorf("IsNull gave %q, want %q", got, want)
	}
	if got := mask.Names(); got[0] != "sym" || got[1] != "qty" || got[2] != "px" {
		t.Errorf("IsNull gave the columns %q, want the names of the frame it came from", got)
	}
}

func TestFrameIsNotNull(t *testing.T) {
	want := []string{"fttf", "tftf", "tttt"}
	if got := columnBits(t, gaps(t).IsNotNull()); !equalLines(got, want) {
		t.Errorf("IsNotNull gave %q, want %q", got, want)
	}
}

// columnBits renders a frame of boolean columns as one string per column, which
// is the way round a null mask is easiest to read.
func columnBits(t *testing.T, f *kuma.Frame[kuma.Dynamic]) []string {
	t.Helper()

	out := make([]string, f.NumCols())
	for i := range f.NumCols() {
		out[i] = bits(t, f.ColumnAt(i))
	}
	return out
}

func TestFrameFillNull(t *testing.T) {
	got, err := gaps(t).FillNull("sym", "z")
	if err != nil {
		t.Fatalf("FillNull: %v", err)
	}

	want := []string{"z 1 1.5", "b . 2.5", "c 3 3.5", "z . 4.5"}
	if lines := rows(t, got); !equalLines(lines, want) {
		t.Errorf("FillNull gave %q, want %q", lines, want)
	}
}

func TestFrameFillNullErrors(t *testing.T) {
	f := gaps(t)

	_, err := f.FillNull("nope", "z")
	if !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("filling a column that is not there gave %v, want ErrNoColumn", err)
	}

	if _, wrong := f.FillNull("qty", "z"); !errors.Is(wrong, kuma.ErrWrongType) {
		t.Errorf("filling an int64 column with a string gave %v, want ErrWrongType", wrong)
	}
}

func TestFrameDropNulls(t *testing.T) {
	tests := []struct {
		names []string
		want  []string
	}{
		{nil, []string{"c 3 3.5"}},
		{[]string{"sym"}, []string{"b . 2.5", "c 3 3.5"}},
		{[]string{"qty"}, []string{". 1 1.5", "c 3 3.5"}},
		{[]string{"px"}, []string{". 1 1.5", "b . 2.5", "c 3 3.5", ". . 4.5"}},
		{[]string{"sym", "qty"}, []string{"c 3 3.5"}},
	}

	for _, tt := range tests {
		got, err := gaps(t).DropNulls(tt.names...)
		if err != nil {
			t.Fatalf("DropNulls%v: %v", tt.names, err)
		}
		if lines := rows(t, got); !equalLines(lines, tt.want) {
			t.Errorf("DropNulls%v gave %q, want %q", tt.names, lines, tt.want)
		}
	}
}

func TestFrameKeepAtLeast(t *testing.T) {
	tests := []struct {
		present int
		want    []string
	}{
		{0, []string{". 1 1.5", "b . 2.5", "c 3 3.5", ". . 4.5"}},
		{1, []string{". 1 1.5", "b . 2.5", "c 3 3.5", ". . 4.5"}},
		{2, []string{". 1 1.5", "b . 2.5", "c 3 3.5"}},
		{3, []string{"c 3 3.5"}},
		{4, nil},
	}

	for _, tt := range tests {
		got, err := gaps(t).KeepAtLeast(tt.present)
		if err != nil {
			t.Fatalf("KeepAtLeast(%d): %v", tt.present, err)
		}
		if lines := rows(t, got); !equalLines(lines, tt.want) {
			t.Errorf("KeepAtLeast(%d) gave %q, want %q", tt.present, lines, tt.want)
		}
	}
}

// TestFrameKeepAtLeastIsHowAll is the pandas how="all", which is this with the
// threshold set to one, over the columns that are allowed to be missing.
func TestFrameKeepAtLeastIsHowAll(t *testing.T) {
	got, err := gaps(t).KeepAtLeast(1, "sym", "qty")
	if err != nil {
		t.Fatalf("KeepAtLeast: %v", err)
	}

	want := []string{". 1 1.5", "b . 2.5", "c 3 3.5"}
	if lines := rows(t, got); !equalLines(lines, want) {
		t.Errorf("KeepAtLeast gave %q, want %q", lines, want)
	}
}

// TestFrameKeepAtLeastNothingToDo is the shortcut for a frame that cannot fail
// the rule, which is the common case of complete data.
func TestFrameKeepAtLeastNothingToDo(t *testing.T) {
	f := gaps(t)

	for _, tt := range []struct {
		present int
		names   []string
	}{
		{0, nil},
		{-1, nil},
		{1, []string{"px"}},
	} {
		got, err := f.KeepAtLeast(tt.present, tt.names...)
		if err != nil {
			t.Fatalf("KeepAtLeast(%d, %v): %v", tt.present, tt.names, err)
		}
		if got != f {
			t.Errorf("KeepAtLeast(%d, %v) built a new frame, want the one it was given",
				tt.present, tt.names)
		}
	}
}

func TestFrameDropNullsErrors(t *testing.T) {
	f := gaps(t)

	_, err := f.DropNulls("z")
	if !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("DropNulls on a column that is not there gave %v, want ErrNoColumn", err)
	}

	if _, atLeast := f.KeepAtLeast(1, "z"); !errors.Is(atLeast, kuma.ErrNoColumn) {
		t.Errorf("KeepAtLeast on a column that is not there gave %v, want ErrNoColumn", atLeast)
	}
}

// TestDropNullsChunkedColumn is the case the row loop has to get right when the
// column it is reading is in more than one piece, since a chunked column counts
// its nulls per chunk and the row numbers do not restart.
func TestDropNullsChunkedColumn(t *testing.T) {
	head := nullInts(t, 1, 0)
	tail := nullInts(t, 0, 4)

	both, err := array.NewChunked(dtype.Int64, append(head.Data().Chunks(), tail.Data().Chunks()...)...)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	c, err := kuma.NewColumn("qty", both)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	f, err := kuma.NewFrame(c)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	got, err := f.DropNulls()
	if err != nil {
		t.Fatalf("DropNulls: %v", err)
	}
	if lines := rows(t, got); !equalLines(lines, []string{"1", "4"}) {
		t.Errorf("DropNulls over two chunks gave %q, want [1 4]", lines)
	}
}
