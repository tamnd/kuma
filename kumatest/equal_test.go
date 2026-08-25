package kumatest_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kumatest"
)

// recorder is a [kumatest.TB] that keeps what was reported rather than failing,
// so that a test of a test helper can read the report.
type recorder struct {
	helped  int
	reports []string
}

func (r *recorder) Helper() { r.helped++ }

func (r *recorder) Errorf(format string, args ...any) {
	r.reports = append(r.reports, fmt.Sprintf(format, args...))
}

// only returns the one report, and fails the test if there is not exactly one.
func (r *recorder) only(t *testing.T) string {
	t.Helper()
	if len(r.reports) != 1 {
		t.Fatalf("%d reports, want one: %q", len(r.reports), r.reports)
	}
	return r.reports[0]
}

func TestEqualFramesSaysNothingAboutTwoFramesThatMatch(t *testing.T) {
	got := trades([]string{"AAPL", "MSFT"}, []float64{150, 100}, []int64{1, 2})
	want := trades([]string{"AAPL", "MSFT"}, []float64{150, 100}, []int64{1, 2})

	var r recorder
	kumatest.EqualFrames(&r, got, want, nil)

	if len(r.reports) != 0 {
		t.Errorf("two frames that match were reported as %q", r.reports)
	}
	if r.helped == 0 {
		t.Error("EqualFrames did not call Helper, so a failure will point at the wrong line")
	}
}

// TestEqualFramesReportsTheWholeDifferenceAtOnce is the property that makes the
// report readable. A call per cell would arrive interleaved with whatever else
// the test printed.
func TestEqualFramesReportsTheWholeDifferenceAtOnce(t *testing.T) {
	got := trades([]string{"AAPL", "MSFT", "GOOG", "AMZN"},
		[]float64{150.25, 100, 200, 300}, []int64{1, 2, 3, 4})
	want := trades([]string{"AAPL", "MSFT", "GOOG", "META"},
		[]float64{150.5, 100, 200, 300}, []int64{1, 2, 3, 5})

	var r recorder
	kumatest.EqualFrames(&r, got, want, nil)

	const wantReport = `frames differ in 2 of 4 rows

  row | column | got    | want
------+--------+--------+------
    0 | price  | 150.25 | 150.5
    3 | symbol | AMZN   | META
    3 | qty    | 4      | 5`

	if report := r.only(t); report != wantReport {
		t.Errorf("the report is\n%s\nand it should be\n%s", report, wantReport)
	}
}

func TestEqualSeriesAndEqualColumns(t *testing.T) {
	a := kuma.NewSeries("qty", int64(1), 2, 3)
	b := kuma.NewSeries("qty", int64(1), 9, 3)

	var same recorder
	kumatest.EqualSeries(&same, a, a, nil)
	kumatest.EqualColumns(&same, a.Column(), a.Column(), nil)
	if len(same.reports) != 0 {
		t.Errorf("a series compared with itself was reported as %q", same.reports)
	}

	var r recorder
	kumatest.EqualSeries(&r, a, b, nil)

	const wantReport = `series differ in 1 of 3 rows

  row | got | want
------+-----+-----
    1 | 2   | 9`

	if report := r.only(t); report != wantReport {
		t.Errorf("the report is\n%s\nand it should be\n%s", report, wantReport)
	}

	var c recorder
	kumatest.EqualColumns(&c, a.Column(), b.Column(), nil)
	if report := c.only(t); !strings.HasPrefix(report, "columns differ in 1 of 3 rows") {
		t.Errorf("the report is\n%s\nand it should be about columns", report)
	}
}

// TestDiffFramesReportsTheShapeFirst covers the differences that are about the
// two frames rather than about a value in them.
func TestDiffFramesReportsTheShapeFirst(t *testing.T) {
	cases := []struct {
		name string
		got  *kuma.Frame[kuma.Dynamic]
		want *kuma.Frame[kuma.Dynamic]
		says []string
	}{
		{
			name: "different column names",
			got:  frame(kuma.NewSeries("price", 1.0).Column()),
			want: frame(kuma.NewSeries("px", 1.0).Column()),
			says: []string{"the columns are [price] where they should be [px]"},
		},
		{
			name: "a column that is not there",
			got:  frame(kuma.NewSeries("price", 1.0).Column()),
			want: frame(kuma.NewSeries("price", 1.0).Column(),
				kuma.NewSeries("qty", int64(1)).Column()),
			says: []string{"the columns are [price] where they should be [price qty]"},
		},
		{
			name: "a column of the wrong type",
			got:  frame(kuma.NewSeries("qty", int32(1)).Column()),
			want: frame(kuma.NewSeries("qty", int64(1)).Column()),
			says: []string{"column qty is int32 where it should be int64"},
		},
		{
			name: "a different number of rows",
			got:  frame(kuma.NewSeries("qty", int64(1), 2, 3).Column()),
			want: frame(kuma.NewSeries("qty", int64(1)).Column()),
			says: []string{"there are 3 rows where there should be 1"},
		},
		{
			name: "a name that is only spaces",
			got:  frame(kuma.NewSeries(" ", 1.0).Column()),
			want: frame(kuma.NewSeries("  ", 1.0).Column()),
			says: []string{`the columns are [" "] where they should be ["  "]`},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			report := kumatest.DiffFrames(c.got, c.want, nil)
			if report == "" {
				t.Fatal("two frames that are not the same shape were reported as equal")
			}
			if !strings.HasPrefix(report, "frames differ") {
				t.Errorf("the report starts %q", firstLine(report))
			}
			for _, says := range c.says {
				if !strings.Contains(report, says) {
					t.Errorf("the report is\n%s\nand it should mention %q", report, says)
				}
			}
		})
	}
}

// TestDiffFramesComparesWhatItCanOfTwoDifferentShapes checks that a frame with
// an extra row on the end still has its other rows compared, which is the case
// where the difference in the values is what went wrong and the row count is
// the symptom.
func TestDiffFramesComparesWhatItCanOfTwoDifferentShapes(t *testing.T) {
	got := frame(kuma.NewSeries("qty", int64(1), 9, 3).Column())
	want := frame(kuma.NewSeries("qty", int64(1), 2).Column())

	report := kumatest.DiffFrames(got, want, nil)
	for _, says := range []string{
		"frames differ in 1 of the rows they both have",
		"there are 3 rows where there should be 2",
		"    1 | qty    | 9   | 2",
	} {
		if !strings.Contains(report, says) {
			t.Errorf("the report is\n%s\nand it should mention %q", report, says)
		}
	}
}

// TestDiffFramesReportsAColumnOfTheWrongTypeAndCarriesOn checks that one column
// with the wrong type does not hide a difference in the column beside it.
func TestDiffFramesReportsAColumnOfTheWrongTypeAndCarriesOn(t *testing.T) {
	got := frame(kuma.NewSeries("qty", int32(1)).Column(),
		kuma.NewSeries("symbol", "AAPL").Column())
	want := frame(kuma.NewSeries("qty", int64(1)).Column(),
		kuma.NewSeries("symbol", "MSFT").Column())

	report := kumatest.DiffFrames(got, want, nil)
	for _, says := range []string{
		"column qty is int32 where it should be int64",
		"    0 | symbol | AAPL | MSFT",
	} {
		if !strings.Contains(report, says) {
			t.Errorf("the report is\n%s\nand it should mention %q", report, says)
		}
	}
}

func TestDiffFramesOnANilFrame(t *testing.T) {
	f := frame(kuma.NewSeries("qty", int64(1)).Column())

	if d := kumatest.DiffFrames[kuma.Dynamic](nil, nil, nil); d != "" {
		t.Errorf("two frames that are both nil were reported as %q", d)
	}
	if d := kumatest.DiffFrames(nil, f, nil); !strings.Contains(d,
		"there is no frame where there should be one of 1 rows x 1 cols") {
		t.Errorf("a missing frame was reported as %q", d)
	}
	if d := kumatest.DiffFrames(f, nil, nil); !strings.Contains(d,
		"there is a frame of 1 rows x 1 cols where there should be none") {
		t.Errorf("a frame that should not be there was reported as %q", d)
	}
}

// TestDiffFramesIgnoresHowTheColumnsAreChunked checks that equality is about
// the values rather than about how they are stored. A frame read in one go and
// the same frame read in pieces are the same frame.
func TestDiffFramesIgnoresHowTheColumnsAreChunked(t *testing.T) {
	whole := frame(kuma.NewSeries("qty", int64(1), 2, 3, 4).Column())

	head := frame(kuma.NewSeries("qty", int64(1), 2).Column())
	tail := frame(kuma.NewSeries("qty", int64(3), 4).Column())
	pieces, err := kuma.Concat(head, tail)
	if err != nil {
		t.Fatalf("Concat: %v", err)
	}
	if n := pieces.ColumnAt(0).Data().NumChunks(); n != 2 {
		t.Fatalf("the concatenated frame is %d chunks, and this test needs more than one", n)
	}

	if d := kumatest.DiffFrames(whole, pieces, nil); d != "" {
		t.Errorf("the same values in a different number of chunks were reported as\n%s", d)
	}
}

// TestNullIsNotAValue is the distinction the library exists to keep, so it is
// the one comparison worth writing out on its own.
func TestNullIsNotAValue(t *testing.T) {
	value := frame(kuma.NewSeries("qty", int64(0)).Column())
	missing := frame(nulls("qty", dtype.Int64, 1))

	if d := kumatest.DiffFrames(missing, missing, nil); d != "" {
		t.Errorf("a missing value did not equal itself: %s", d)
	}
	if d := kumatest.DiffFrames(value, missing, nil); d == "" {
		t.Error("a zero was reported as equal to a missing value")
	}
	if d := kumatest.DiffFrames(missing, value, nil); d == "" {
		t.Error("a missing value was reported as equal to a zero")
	}
}

// TestFloatsWithAnAllowance covers the arithmetic in the float comparison, one
// pair of numbers per rule.
func TestFloatsWithAnAllowance(t *testing.T) {
	cases := []struct {
		name  string
		got   float64
		want  float64
		opts  *kumatest.Options
		equal bool
	}{
		{name: "the same number", got: 1.5, want: 1.5, equal: true},
		{name: "a different number", got: 1.5, want: 1.6},
		{name: "zero of either sign", got: 0, want: math.Copysign(0, -1), equal: true},
		{
			name: "within a fraction",
			got:  1000, want: 1000.0001,
			opts:  &kumatest.Options{Fraction: 1e-6},
			equal: true,
		},
		{
			name: "outside a fraction",
			got:  1000, want: 1001,
			opts: &kumatest.Options{Fraction: 1e-6},
		},
		{
			name: "within a margin",
			got:  0, want: 1e-9,
			opts:  &kumatest.Options{Margin: 1e-6},
			equal: true,
		},
		{
			name: "outside a margin",
			got:  0, want: 1e-3,
			opts: &kumatest.Options{Margin: 1e-6},
		},
		{
			name: "near zero, where only a margin helps",
			got:  1e-12, want: 2e-12,
			opts: &kumatest.Options{Fraction: 1e-6},
		},
		{name: "two NaNs", got: math.NaN(), want: math.NaN(), equal: true},
		{
			name: "two NaNs when they are asked to differ",
			got:  math.NaN(), want: math.NaN(),
			opts: &kumatest.Options{NaNsDiffer: true},
		},
		{name: "a NaN and a number", got: math.NaN(), want: 1},
		{name: "a number and a NaN", got: 1, want: math.NaN()},
		{name: "two infinities", got: math.Inf(1), want: math.Inf(1), equal: true},
		{name: "infinities of either sign", got: math.Inf(1), want: math.Inf(-1)},
		{
			name: "an infinity and a number, however large the allowance",
			got:  math.Inf(1), want: 1,
			opts: &kumatest.Options{Margin: 1e300},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := kuma.NewSeries("px", c.got)
			want := kuma.NewSeries("px", c.want)

			d := kumatest.DiffSeries(got, want, c.opts)
			if equal := d == ""; equal != c.equal {
				t.Errorf("%v and %v were reported as %q, and they should be equal: %v",
					c.got, c.want, d, c.equal)
			}
		})
	}
}

// TestFloat32sTakeTheSameAllowance checks that the tolerance reaches the
// narrower float as well, since a float32 column is where a sum drifts soonest.
func TestFloat32sTakeTheSameAllowance(t *testing.T) {
	got := kuma.NewSeries("px", float32(1000))
	want := kuma.NewSeries("px", float32(1000.05))

	if d := kumatest.DiffSeries(got, want, &kumatest.Options{Fraction: 1e-4}); d != "" {
		t.Errorf("two float32 values within the allowance were reported as\n%s", d)
	}
	if d := kumatest.DiffSeries(got, want, nil); d == "" {
		t.Error("two float32 values that differ were reported as equal")
	}
}

// TestOnlySoManyCellsArePrinted covers the limit on the report, which is what
// keeps a frame that is wrong from top to bottom down to something a person
// will read.
func TestOnlySoManyCellsArePrinted(t *testing.T) {
	a := make([]int64, 100)
	b := make([]int64, 100)
	for i := range b {
		b[i] = 1
	}
	got := kuma.NewSeries("qty", a...)
	want := kuma.NewSeries("qty", b...)

	report := kumatest.DiffSeries(got, want, nil)
	if n := cellLines(report); n != 10 {
		t.Errorf("the report holds %d cells, want the first ten:\n%s", n, report)
	}
	if !strings.HasSuffix(report, "and 90 more") {
		t.Errorf("the report does not say how many were left out:\n%s", report)
	}
	if !strings.HasPrefix(report, "series differ in 100 of 100 rows") {
		t.Errorf("the report starts %q, and the counts should be of everything",
			firstLine(report))
	}

	few := kumatest.DiffSeries(got, want, &kumatest.Options{MaxCells: 2})
	if n := cellLines(few); n != 2 {
		t.Errorf("with MaxCells 2 the report holds %d cells:\n%s", n, few)
	}

	all := kumatest.DiffSeries(got, want, &kumatest.Options{MaxCells: -1})
	if n := cellLines(all); n != 100 {
		t.Errorf("with MaxCells -1 the report holds %d cells, want all 100", n)
	}
	if strings.Contains(all, "more") {
		t.Errorf("the report says there are more when there are not:\n%s", all)
	}
}

// cellLines is how many rows the table in a report has, which is how many of
// the differing cells it printed.
func cellLines(report string) int {
	_, table, ok := strings.Cut(report, "\n\n")
	if !ok {
		return 0
	}
	table, _, _ = strings.Cut(table, "\n\nand ")

	lines := strings.Split(strings.TrimRight(table, "\n"), "\n")
	return len(lines) - 2 // the header and the rule under it
}

// TestTheValuesAreWrittenTheWayAFrameWritesThem checks that the report goes
// through the printer rather than through some second opinion of its own, which
// is what makes a difference in a timestamp readable.
func TestTheValuesAreWrittenTheWayAFrameWritesThem(t *testing.T) {
	got := kuma.NewSeries("note", "a b")
	want := kuma.NewSeries("note", "a b ")

	report := kumatest.DiffSeries(got, want, nil)
	if !strings.Contains(report, `a b | "a b "`) {
		t.Errorf("the report is\n%s\nand the trailing space should be quoted", report)
	}

	long := kuma.NewSeries("note", strings.Repeat("x", 100))
	short := kuma.NewSeries("note", "x")
	narrow := kumatest.DiffSeries(long, short, &kumatest.Options{
		Print: &kuma.PrintOptions{MaxWidth: 8},
	})
	if !strings.Contains(narrow, "xxxxx... | x") {
		t.Errorf("the report is\n%s\nand the long value should be cut short", narrow)
	}
}

// TestASeriesWithADifferentNameOrType covers the notes a single column gets,
// which are the ones a frame reports about its columns.
func TestASeriesWithADifferentNameOrType(t *testing.T) {
	qty := kuma.NewSeries("qty", int64(1))

	if d := kumatest.DiffSeries(qty, qty.Rename("quantity"), nil); !strings.Contains(d,
		"the name is qty where it should be quantity") {
		t.Errorf("a series with the wrong name was reported as %q", d)
	}

	narrow, err := qty.Cast[int32](dtype.Int32)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	d := kumatest.DiffColumns(narrow.Column(), qty.Column(), nil)
	if !strings.Contains(d, "the type is int32 where it should be int64") {
		t.Errorf("a column of the wrong type was reported as %q", d)
	}
	if strings.Contains(d, "row") {
		t.Errorf("the values were compared across two types:\n%s", d)
	}

	if d := kumatest.DiffSeries(qty, qty.Head(0), nil); !strings.Contains(d,
		"there are 1 rows where there should be 0") {
		t.Errorf("a series of the wrong length was reported as %q", d)
	}
}

// TestATypeThatCannotBeComparedYet is the guard for the nested types. They
// cannot hold a value today, so a column of one is empty, and the point of the
// test is that the report says so rather than calling the two equal.
func TestATypeThatCannotBeComparedYet(t *testing.T) {
	dt := dtype.List{Elem: dtype.Int64}
	data, err := array.NewChunked(dt)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	col, err := kuma.NewColumn("xs", data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}

	if d := kumatest.DiffColumns(col, col, nil); !strings.Contains(d,
		"the type is list<int64>, which this package cannot compare yet") {
		t.Errorf("a list column was reported as %q", d)
	}
	if d := kumatest.DiffFrames(frame(col), frame(col), nil); !strings.Contains(d,
		"column xs is list<int64>, which this package cannot compare yet") {
		t.Errorf("a frame of one list column was reported as %q", d)
	}
}

// TestEveryTypeIsCompared runs a column of each type past the comparison, since
// the switch over the types is the part of this package most likely to grow a
// missing case.
func TestEveryTypeIsCompared(t *testing.T) {
	for _, dt := range randomTypes() {
		t.Run(dt.String(), func(t *testing.T) {
			opts := &kumatest.RandomOptions{Rows: 64, Types: []dtype.DataType{dt}, Seed: 1}
			same := *opts
			other := *opts
			other.Seed = 2

			if d := kumatest.DiffFrames(kumatest.Random(opts), kumatest.Random(&same), nil); d != "" {
				t.Errorf("the same column did not equal itself:\n%s", d)
			}
			if dt.Kind() == dtype.NullKind {
				// Every value of a null column is missing, so two of them are
				// equal whatever the seed was.
				return
			}
			if d := kumatest.DiffFrames(kumatest.Random(opts), kumatest.Random(&other), nil); d == "" {
				t.Error("two columns of different values were reported as equal")
			}
		})
	}
}

// trades builds the frame most of these tests compare.
func trades(sym []string, px []float64, qty []int64) *kuma.Frame[kuma.Dynamic] {
	return frame(
		kuma.NewSeries("symbol", sym...).Column(),
		kuma.NewSeries("price", px...).Column(),
		kuma.NewSeries("qty", qty...).Column())
}

// frame builds a frame out of columns. It panics rather than taking a
// *testing.T, since a test that cannot build its own data has a mistake in it
// rather than a result to report, and the examples in this package want the
// same helper without one.
func frame(cols ...kuma.Column) *kuma.Frame[kuma.Dynamic] {
	f, err := kuma.NewFrame(cols...)
	if err != nil {
		panic(err)
	}
	return f
}

// nulls builds a column of n missing values.
func nulls(name string, dt dtype.DataType, n int) kuma.Column {
	b, err := array.NewBuilder(dt)
	if err != nil {
		panic(err)
	}
	b.AppendNulls(n)

	data, err := array.NewChunked(dt, b.Finish())
	if err != nil {
		panic(err)
	}
	col, err := kuma.NewColumn(name, data)
	if err != nil {
		panic(err)
	}
	return col
}

// firstLine is the headline of a report, for a failure that does not need the
// whole thing.
func firstLine(report string) string {
	line, _, _ := strings.Cut(report, "\n")
	return line
}
