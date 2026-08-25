package kuma_test

import (
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// lines joins the expected output of a printer, which is written a line at a
// time because that is how it is read.
func lines(s ...string) string { return strings.Join(s, "\n") }

// numbers returns a column of a fixed width type, built by hand because the
// typed constructors only cover the types Go has.
func numbers[T array.Numeric](t *testing.T, name string, dt dtype.DataType, vs ...T) kuma.Column {
	t.Helper()

	b, err := array.NewBuilder(dt)
	if err != nil {
		t.Fatalf("NewBuilder(%s): %v", dt, err)
	}
	for _, v := range vs {
		b.Append(v)
	}
	return finish(t, name, dt, b)
}

// blobs returns a column of a type whose values are bytes rather than numbers.
func blobs(t *testing.T, name string, dt dtype.DataType, vs ...[]byte) kuma.Column {
	t.Helper()

	b, err := array.NewBuilder(dt)
	if err != nil {
		t.Fatalf("NewBuilder(%s): %v", dt, err)
	}
	for _, v := range vs {
		b.AppendBytes(v)
	}
	return finish(t, name, dt, b)
}

// finish turns a builder into a column.
func finish(t *testing.T, name string, dt dtype.DataType, b *array.Builder) kuma.Column {
	t.Helper()

	data, err := array.NewChunked(dt, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	c, err := kuma.NewColumn(name, data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	return c
}

// cell renders one value on its own, which is what most of the tests below are
// really asking about.
func cell(t *testing.T, c kuma.Column, i int) string {
	t.Helper()

	out := c.Render(&kuma.PrintOptions{MaxWidth: -1})
	got := strings.Split(out, "\n")
	return strings.TrimSpace(got[5+i])
}

func TestFrameString(t *testing.T) {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT").Column(),
		nullableInts(t, 2).Column(),
		kuma.NewSeries("price", 189.5, 411.25).Column(),
		kuma.NewSeries("live", true, false).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	want := lines(
		"kuma.Frame[kuma.Dynamic] 2 rows x 4 cols",
		"",
		"  symbol |   qty |   price | live",
		"  string | int64 | float64 | bool",
		"---------+-------+---------+------",
		"  AAPL   |  null |   189.5 | true",
		"  MSFT   |     1 |  411.25 | false",
	)
	if got := f.String(); got != want {
		t.Errorf("String() =\n%s\nwant\n%s", got, want)
	}
}

// TestFrameStringElidesRows checks the middle of a long frame being left out,
// and that the half kept is the front when the count is odd.
func TestFrameStringElidesRows(t *testing.T) {
	vs := make([]int64, 100)
	for i := range vs {
		vs[i] = int64(i)
	}
	f, err := kuma.NewFrame(kuma.NewSeries("i", vs...).Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	want := lines(
		"kuma.Frame[kuma.Dynamic] 100 rows x 1 cols",
		"",
		"      i",
		"  int64",
		"-------",
		"      0",
		"      1",
		"    ...",
		"     99",
	)
	if got := f.Render(&kuma.PrintOptions{MaxRows: 3}); got != want {
		t.Errorf("Render() =\n%s\nwant\n%s", got, want)
	}
}

func TestFrameStringElidesColumns(t *testing.T) {
	cols := make([]kuma.Column, 6)
	for i := range cols {
		cols[i] = kuma.NewSeries("c"+strconv.Itoa(i), int64(i)).Column()
	}
	f, err := kuma.NewFrame(cols...)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	want := lines(
		"kuma.Frame[kuma.Dynamic] 1 rows x 6 cols",
		"",
		"     c0 |    c1 | ... |    c5",
		"  int64 | int64 | ... | int64",
		"--------+-------+-----+------",
		"      0 |     1 | ... |     5",
	)
	if got := f.Render(&kuma.PrintOptions{MaxCols: 3}); got != want {
		t.Errorf("Render() =\n%s\nwant\n%s", got, want)
	}
}

// TestFrameStringNoRows is the frame a filter that matched nothing gives back.
// The columns are still worth seeing, so the header stays.
func TestFrameStringNoRows(t *testing.T) {
	f, err := kuma.NewFrame(kuma.NewSeries("qty", int64(1), 2).Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	want := lines(
		"kuma.Frame[kuma.Dynamic] 0 rows x 1 cols",
		"",
		"    qty",
		"  int64",
		"-------",
	)
	if got := f.Head(0).String(); got != want {
		t.Errorf("String() =\n%s\nwant\n%s", got, want)
	}
}

func TestFrameStringNoColumns(t *testing.T) {
	f, err := kuma.NewFrame()
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	if got, want := f.String(), "kuma.Frame[kuma.Dynamic] 0 rows x 0 cols"; got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
}

func TestSeriesString(t *testing.T) {
	s := nullableInts(t, 3)

	want := lines(
		"kuma.Series[int64] 3 rows",
		"",
		"    qty",
		"  int64",
		"-------",
		"   null",
		"      1",
		"      2",
	)
	if got := s.String(); got != want {
		t.Errorf("String() =\n%s\nwant\n%s", got, want)
	}
}

func TestColumnString(t *testing.T) {
	c := kuma.NewSeries("symbol", "AAPL", "MSFT").Column()

	want := lines(
		"kuma.Column 2 rows",
		"",
		"  symbol",
		"  string",
		"--------",
		"  AAPL",
		"  MSFT",
	)
	if got := c.String(); got != want {
		t.Errorf("String() =\n%s\nwant\n%s", got, want)
	}
}

// TestPrintNull covers both ways a cell can have nothing in it: a missing value
// in a column of a real type, and a column whose type is the absence of one.
func TestPrintNull(t *testing.T) {
	s := nullableInts(t, 2)
	if got := cell(t, s.Column(), 0); got != "null" {
		t.Errorf("a missing value printed as %q", got)
	}

	got := s.Render(&kuma.PrintOptions{Null: "NA"})
	if !strings.Contains(got, "NA") {
		t.Errorf("Null was not used:\n%s", got)
	}

	b, err := array.NewBuilder(dtype.Null)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	b.AppendNulls(2)
	if got := cell(t, finish(t, "nothing", dtype.Null, b), 0); got != "null" {
		t.Errorf("a null column printed as %q", got)
	}
}

func TestPrintNumbers(t *testing.T) {
	tests := []struct {
		name string
		col  kuma.Column
		want []string
	}{
		{"int8", numbers(t, "n", dtype.Int8, int8(-128), 127), []string{"-128", "127"}},
		{"int16", numbers(t, "n", dtype.Int16, int16(-32768), 32767), []string{"-32768", "32767"}},
		{"int32", numbers(t, "n", dtype.Int32, int32(-1), 2147483647), []string{"-1", "2147483647"}},
		{"int64", numbers(t, "n", dtype.Int64, int64(-1), 1<<62), []string{"-1", "4611686018427387904"}},
		{"uint8", numbers(t, "n", dtype.Uint8, uint8(0), 255), []string{"0", "255"}},
		{"uint16", numbers(t, "n", dtype.Uint16, uint16(0), 65535), []string{"0", "65535"}},
		{"uint32", numbers(t, "n", dtype.Uint32, uint32(0), 4294967295), []string{"0", "4294967295"}},
		{"uint64", numbers(t, "n", dtype.Uint64, uint64(0), 1<<63), []string{"0", "9223372036854775808"}},
		{"float32", numbers(t, "n", dtype.Float32, float32(1.5), 0.1), []string{"1.5", "0.1"}},
		{"float64", numbers(t, "n", dtype.Float64, 1.5, 0.1), []string{"1.5", "0.1"}},
		{"bool", kuma.NewSeries("b", true, false).Column(), []string{"true", "false"}},
		{"string", kuma.NewSeries("s", "a", "").Column(), []string{"a", ""}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for i, want := range tt.want {
				if got := cell(t, tt.col, i); got != want {
					t.Errorf("value %d printed as %q, want %q", i, got, want)
				}
			}
		})
	}
}

// TestPrintFloatsAreNotRounded is the promise the doc comment makes. Two prices
// a hundredth of a penny apart have to look different, since the reason anyone
// prints a frame is to find out why two numbers disagree.
func TestPrintFloatsAreNotRounded(t *testing.T) {
	tenth, fifth := 0.1, 0.2
	c := numbers(t, "px", dtype.Float64, tenth+fifth, 0.3)

	a, b := cell(t, c, 0), cell(t, c, 1)
	if a == b {
		t.Errorf("both values printed as %q", a)
	}
	if a != "0.30000000000000004" {
		t.Errorf("got %q", a)
	}
}

func TestPrintTemporal(t *testing.T) {
	// 2026-08-25 09:30:00 UTC, and the day it falls on.
	const stamp = 1787650200
	const day = stamp / 86400

	tests := []struct {
		name string
		col  kuma.Column
		want string
	}{
		{"date32", numbers(t, "d", dtype.Date32, int32(day)), "2026-08-25"},
		{"date64", numbers(t, "d", dtype.Date64, int64(day)*86400*1000), "2026-08-25"},
		{
			"time32 seconds",
			numbers(t, "t", dtype.Time32{Unit: dtype.Second}, int32(34200)),
			"09:30:00",
		},
		{
			"time32 milliseconds",
			numbers(t, "t", dtype.Time32{Unit: dtype.Millisecond}, int32(34200_250)),
			"09:30:00.25",
		},
		{
			"time64 nanoseconds",
			numbers(t, "t", dtype.Time64{Unit: dtype.Nanosecond}, int64(34200)*1e9+1),
			"09:30:00.000000001",
		},
		{
			"timestamp with no zone",
			numbers(t, "ts", dtype.Timestamp{Unit: dtype.Second}, int64(stamp)),
			"2026-08-25 09:30:00",
		},
		{
			"timestamp in microseconds",
			numbers(t, "ts", dtype.Timestamp{Unit: dtype.Microsecond}, int64(stamp)*1e6+500),
			"2026-08-25 09:30:00.0005",
		},
		{
			"timestamp in a zone",
			numbers(t, "ts",
				dtype.Timestamp{Unit: dtype.Second, Zone: "Asia/Tokyo"}, int64(stamp)),
			"2026-08-25 18:30:00",
		},
		{
			"timestamp before the epoch",
			numbers(t, "ts", dtype.Timestamp{Unit: dtype.Millisecond}, int64(-1500)),
			"1969-12-31 23:59:58.5",
		},
		{
			"duration",
			numbers(t, "dur", dtype.Duration{Unit: dtype.Millisecond}, int64(90_000)),
			"1m30s",
		},
		{
			"duration in microseconds",
			numbers(t, "dur", dtype.Duration{Unit: dtype.Microsecond}, int64(1500)),
			"1.5ms",
		},
		{
			"duration in nanoseconds",
			numbers(t, "dur", dtype.Duration{Unit: dtype.Nanosecond}, int64(1)),
			"1ns",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cell(t, tt.col, 0); got != tt.want {
				t.Errorf("printed as %q, want %q", got, tt.want)
			}
		})
	}
}

// TestPrintTemporalZoneThatIsNotInstalled covers the fallback. A zone name the
// machine cannot resolve prints in UTC rather than failing, because whether
// tzdata is installed is a property of the machine and not of the data.
func TestPrintTemporalZoneThatIsNotInstalled(t *testing.T) {
	c := numbers(t, "ts",
		dtype.Timestamp{Unit: dtype.Second, Zone: "Mars/Olympus_Mons"}, int64(0))

	if got, want := cell(t, c, 0), "1970-01-01 00:00:00"; got != want {
		t.Errorf("printed as %q, want %q", got, want)
	}
}

// TestPrintTemporalOutOfRange covers the counts that no date can be made of.
// They print as the numbers they are, because a year in five digits reads as a
// typo and a raw count reads as what it is.
func TestPrintTemporalOutOfRange(t *testing.T) {
	tests := []struct {
		name string
		col  kuma.Column
		want string
	}{
		{"date32", numbers(t, "d", dtype.Date32, int32(1<<30)), "92771293593600s"},
		{
			"timestamp",
			numbers(t, "ts", dtype.Timestamp{Unit: dtype.Second}, int64(1)<<60),
			"1152921504606846976s",
		},
		{
			"time32 past midnight",
			numbers(t, "t", dtype.Time32{Unit: dtype.Second}, int32(86400)),
			"86400s",
		},
		{
			"time32 before midnight",
			numbers(t, "t", dtype.Time32{Unit: dtype.Second}, int32(-1)),
			"-1s",
		},
		{
			"duration too long for a Duration",
			numbers(t, "dur", dtype.Duration{Unit: dtype.Second}, int64(1)<<40),
			"1099511627776s",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cell(t, tt.col, 0); got != tt.want {
				t.Errorf("printed as %q, want %q", got, tt.want)
			}
		})
	}
}

func TestPrintBytes(t *testing.T) {
	tests := []struct {
		name string
		col  kuma.Column
		want string
	}{
		{
			"binary",
			blobs(t, "b", dtype.Binary, []byte{0x00, 0xff}),
			"0x00ff",
		},
		{
			"fixed size binary",
			blobs(t, "b", dtype.FixedSizeBinary{ByteWidth: 3}, []byte("abc")),
			"0x616263",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cell(t, tt.col, 0); got != tt.want {
				t.Errorf("printed as %q, want %q", got, tt.want)
			}
		})
	}
}

func TestPrintDecimal(t *testing.T) {
	// The bytes are the integer least significant byte first, which is how a
	// decimal is stored.
	le := func(n int64, width int) []byte {
		b := make([]byte, width)
		neg := n < 0
		if neg {
			n = -n
		}
		for i := range b {
			b[i] = byte(n)
			n >>= 8
		}
		if neg {
			carry := uint16(1)
			for i := range b {
				v := uint16(^b[i]) + carry
				b[i], carry = byte(v), v>>8
			}
		}
		return b
	}

	tests := []struct {
		name  string
		dt    dtype.DataType
		value []byte
		want  string
	}{
		{"pounds and pence", dtype.Decimal128{Precision: 18, Scale: 2}, le(1234, 16), "12.34"},
		{"negative", dtype.Decimal128{Precision: 18, Scale: 2}, le(-1234, 16), "-12.34"},
		{"under a penny", dtype.Decimal128{Precision: 18, Scale: 4}, le(25, 16), "0.0025"},
		{"just under zero", dtype.Decimal128{Precision: 18, Scale: 4}, le(-25, 16), "-0.0025"},
		{"nothing", dtype.Decimal128{Precision: 18, Scale: 2}, le(0, 16), "0"},
		{"whole numbers", dtype.Decimal128{Precision: 18, Scale: 0}, le(1234, 16), "1234"},
		{"counted in thousands", dtype.Decimal128{Precision: 18, Scale: -3}, le(12, 16), "12000"},
		{"wide", dtype.Decimal256{Precision: 50, Scale: 8}, le(1<<62, 32), "46116860184.27387904"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := blobs(t, "amount", tt.dt, tt.value)
			if got := cell(t, c, 0); got != tt.want {
				t.Errorf("printed as %q, want %q", got, tt.want)
			}
		})
	}
}

func TestPrintInterval(t *testing.T) {
	tests := []struct {
		name  string
		unit  dtype.IntervalUnit
		value []byte
		want  string
	}{
		{"a year and two months", dtype.YearMonth, []byte{14, 0, 0, 0}, "14mo"},
		{
			"three days and an hour",
			dtype.DayTime,
			[]byte{3, 0, 0, 0, 0x80, 0xee, 0x36, 0x00},
			"3d 1h0m0s",
		},
		{
			"a month, a day and a second",
			dtype.MonthDayNano,
			[]byte{1, 0, 0, 0, 1, 0, 0, 0, 0x00, 0xca, 0x9a, 0x3b, 0, 0, 0, 0},
			"1mo 1d 1s",
		},
		{
			"nothing at all",
			dtype.MonthDayNano,
			make([]byte, 16),
			"0s",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := blobs(t, "every", dtype.Interval{Unit: tt.unit}, tt.value)
			if got := cell(t, c, 0); got != tt.want {
				t.Errorf("printed as %q, want %q", got, tt.want)
			}
		})
	}
}

// TestPrintTextThatWouldWreckTheTable covers the values that move the cursor
// rather than drawing. A newline in a value has to come out as two characters,
// or every row after it is in the wrong place.
func TestPrintTextThatWouldWreckTheTable(t *testing.T) {
	c := kuma.NewSeries("note", "two\nlines", "a tab\there", "plain").Column()

	for i, want := range []string{`"two\nlines"`, `"a tab\there"`, "plain"} {
		if got := cell(t, c, i); got != want {
			t.Errorf("value %d printed as %q, want %q", i, got, want)
		}
	}

	// The same goes for the name, which is a string a file gave us and not
	// something a Go programmer typed.
	f, err := kuma.NewFrame(kuma.NewSeries("two\nlines", int64(1)).Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	if got, want := strings.Count(f.String(), "\n"), 5; got != want {
		t.Errorf("the table is %d lines, want %d:\n%s", got+1, want+1, f.String())
	}
}

// TestPrintSpaceAtTheEdgeOfAValue checks that a value whose text begins or ends
// in a space is quoted. That space is a real difference between two values, it
// is invisible in a padded table, and left alone it hangs off the end of the
// line as trailing whitespace.
func TestPrintSpaceAtTheEdgeOfAValue(t *testing.T) {
	c := kuma.NewSeries("note", "trailing ", " leading", "in the middle", " ", "").Column()

	for i, want := range []string{`"trailing "`, `" leading"`, "in the middle", `" "`, ""} {
		if got := cell(t, c, i); got != want {
			t.Errorf("value %d printed as %q, want %q", i, got, want)
		}
	}

	// A column name out of a file gets the same treatment, and the whole table
	// has no line with whitespace on the end of it.
	f, err := kuma.NewFrame(kuma.NewSeries("qty ", int64(1)).Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	out := f.String()
	if !strings.Contains(out, `"qty "`) {
		t.Errorf("the name was not quoted:\n%s", out)
	}
	if strings.Contains(out, " \n") || strings.HasSuffix(out, " ") {
		t.Errorf("a line ends in whitespace:\n%q", out)
	}
}

// TestPrintWideColumn pads by more than one run of blanks, which is the case a
// long column name over short values produces.
func TestPrintWideColumn(t *testing.T) {
	name := strings.Repeat("x", 100)
	f, err := kuma.NewFrame(
		kuma.NewSeries(name, int64(1)).Column(),
		kuma.NewSeries("b", int64(2)).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	out := f.Render(&kuma.PrintOptions{MaxWidth: -1})
	want := lines(
		"kuma.Frame[kuma.Dynamic] 1 rows x 2 cols",
		"",
		"  "+name+" |     b",
		"  "+strings.Repeat(" ", 95)+"int64 | int64",
		strings.Repeat("-", 103)+"+------",
		"  "+strings.Repeat(" ", 99)+"1 |     2",
	)
	if out != want {
		t.Errorf("printed as\n%s\nwant\n%s", out, want)
	}
}

func TestPrintWidth(t *testing.T) {
	c := kuma.NewSeries("note", "abcdefghij").Column()

	tests := []struct {
		width int
		want  string
	}{
		{-1, "abcdefghij"},
		{20, "abcdefghij"},
		{10, "abcdefghij"},
		{9, "abcdef..."},
		{4, "a..."},
		{3, "abc"},
		{1, "a"},
	}

	for _, tt := range tests {
		t.Run(strconv.Itoa(tt.width), func(t *testing.T) {
			out := c.Render(&kuma.PrintOptions{MaxWidth: tt.width})
			if got := strings.TrimSpace(strings.Split(out, "\n")[5]); got != tt.want {
				t.Errorf("at width %d the value printed as %q, want %q", tt.width, got, tt.want)
			}
		})
	}
}

// TestPrintWidthCountsRunes checks that a value is cut at a character rather
// than at a byte, since cutting a UTF-8 sequence in half produces text that is
// not text.
func TestPrintWidthCountsRunes(t *testing.T) {
	c := kuma.NewSeries("note", strings.Repeat("\u00e9", 10)).Column()

	out := c.Render(&kuma.PrintOptions{MaxWidth: 6})
	if got, want := strings.TrimSpace(strings.Split(out, "\n")[5]), "\u00e9\u00e9\u00e9..."; got != want {
		t.Errorf("printed as %q, want %q", got, want)
	}
}

// TestPrintEverythingShown is the whole frame, which is what someone asking for
// a diff wants and what kumatest will ask for.
func TestPrintEverythingShown(t *testing.T) {
	vs := make([]int64, 50)
	f, err := kuma.NewFrame(kuma.NewSeries("i", vs...).Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	out := f.Render(&kuma.PrintOptions{MaxRows: -1, MaxCols: -1})
	if got, want := strings.Count(out, "\n"), 54; got != want {
		t.Errorf("the table is %d lines, want %d", got+1, want+1)
	}
	if strings.Contains(out, "...") {
		t.Errorf("something was left out:\n%s", out)
	}
}

func ExampleFrame_Render() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "NVDA").Column(),
		kuma.NewSeries("price", 189.5, 411.2, 121.0).Column(),
	)
	if err != nil {
		panic(err)
	}

	fmt.Println(f.Render(&kuma.PrintOptions{MaxRows: 2}))
	// Output:
	// kuma.Frame[kuma.Dynamic] 3 rows x 2 cols
	//
	//   symbol |   price
	//   string | float64
	// ---------+--------
	//   AAPL   |   189.5
	//   ...    |     ...
	//   NVDA   |     121
}
