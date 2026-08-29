package arrowgo

import (
	"strings"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	arrowarray "github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// arrayCases is one column of every type that crosses, written as the JSON
// arrow-go builds an array from. Every one of them has a null in it, because
// the validity bitmap is shared rather than copied and a column with no nulls
// does not have one to share.
var arrayCases = []struct {
	name string
	dt   arrow.DataType
	json string
	want dtype.DataType
}{
	{"bool", arrow.FixedWidthTypes.Boolean, `[true, false, null, true]`, dtype.Bool},
	{"int8", arrow.PrimitiveTypes.Int8, `[1, -2, null, 127]`, dtype.Int8},
	{"int16", arrow.PrimitiveTypes.Int16, `[1, -2, null, 32767]`, dtype.Int16},
	{"int32", arrow.PrimitiveTypes.Int32, `[1, -2, null, 2147483647]`, dtype.Int32},
	{"int64", arrow.PrimitiveTypes.Int64, `[1, -2, null, 90071992547409]`, dtype.Int64},
	{"uint8", arrow.PrimitiveTypes.Uint8, `[1, 2, null, 255]`, dtype.Uint8},
	{"uint16", arrow.PrimitiveTypes.Uint16, `[1, 2, null, 65535]`, dtype.Uint16},
	{"uint32", arrow.PrimitiveTypes.Uint32, `[1, 2, null, 4294967295]`, dtype.Uint32},
	{"uint64", arrow.PrimitiveTypes.Uint64, `[1, 2, null, 18446744073709551615]`, dtype.Uint64},
	{"float32", arrow.PrimitiveTypes.Float32, `[1.5, -2.25, null, 0]`, dtype.Float32},
	{"float64", arrow.PrimitiveTypes.Float64, `[1.5, -2.25, null, 0]`, dtype.Float64},
	{"utf8", arrow.BinaryTypes.String, stringJSON, dtype.String},
	{"large_utf8", arrow.BinaryTypes.LargeString, stringJSON, dtype.String},
	{"string_view", arrow.BinaryTypes.StringView, stringJSON, dtype.String},
	{"binary", arrow.BinaryTypes.Binary, binaryJSON, dtype.Binary},
	{"large_binary", arrow.BinaryTypes.LargeBinary, binaryJSON, dtype.Binary},
	{"binary_view", arrow.BinaryTypes.BinaryView, binaryJSON, dtype.Binary},
	{"date32", arrow.FixedWidthTypes.Date32, `[0, 19000, null, -1]`, dtype.Date32},
	{"date64", arrow.FixedWidthTypes.Date64, `[0, 86400000, null, -86400000]`, dtype.Date64},
	{"time32s", arrow.FixedWidthTypes.Time32s, `[0, 3600, null, 86399]`, dtype.Time32{Unit: dtype.Second}},
	{"time32ms", arrow.FixedWidthTypes.Time32ms, `[0, 3600000, null, 1]`, dtype.Time32{Unit: dtype.Millisecond}},
	{"time64us", arrow.FixedWidthTypes.Time64us, `[0, 3600000000, null, 1]`, dtype.Time64{Unit: dtype.Microsecond}},
	{"time64ns", arrow.FixedWidthTypes.Time64ns, `[0, 3600000000000, null, 1]`, dtype.Time64{Unit: dtype.Nanosecond}},
	{
		name: "timestamp",
		dt:   &arrow.TimestampType{Unit: arrow.Microsecond, TimeZone: "UTC"},
		json: `["2026-08-29T08:00:00Z", "1970-01-01T00:00:00Z", null, "2000-01-01T00:00:00Z"]`,
		want: dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"},
	},
	{"duration", arrow.FixedWidthTypes.Duration_s, `[0, 60, null, -1]`, dtype.Duration{Unit: dtype.Second}},
	{
		name: "month_interval",
		dt:   arrow.FixedWidthTypes.MonthInterval,
		json: `[{"months": 1}, {"months": 12}, null, {"months": -3}]`,
		want: dtype.Interval{Unit: dtype.YearMonth},
	},
	{
		name: "day_time_interval",
		dt:   arrow.FixedWidthTypes.DayTimeInterval,
		json: `[{"days": 1, "milliseconds": 2}, {"days": 0, "milliseconds": 0}, null, {"days": -1, "milliseconds": 3}]`,
		want: dtype.Interval{Unit: dtype.DayTime},
	},
	{
		name: "month_day_nano_interval",
		dt:   arrow.FixedWidthTypes.MonthDayNanoInterval,
		json: `[{"months": 1, "days": 2, "nanoseconds": 3}, {"months": 0, "days": 0, "nanoseconds": 0}, null, {"months": -1, "days": -2, "nanoseconds": -3}]`,
		want: dtype.Interval{Unit: dtype.MonthDayNano},
	},
	{
		name: "fixed_size_binary",
		dt:   &arrow.FixedSizeBinaryType{ByteWidth: 4},
		json: `["YWJjZA==", "ZWZnaA==", null, "aWprbA=="]`,
		want: dtype.FixedSizeBinary{ByteWidth: 4},
	},
	{
		name: "decimal128",
		dt:   &arrow.Decimal128Type{Precision: 18, Scale: 2},
		json: `["12.34", "-0.01", null, "0.00"]`,
		want: dtype.Decimal128{Precision: 18, Scale: 2},
	},
	{
		name: "decimal256",
		dt:   &arrow.Decimal256Type{Precision: 50, Scale: 3},
		json: `["12.340", "-0.001", null, "0.000"]`,
		want: dtype.Decimal256{Precision: 50, Scale: 3},
	},
}

// The two variable width columns, written once because six of the rows above
// share them. The third value is longer than the twelve bytes a view holds
// inline, so the block path is exercised as well as the inline one.
const (
	stringJSON = `["AAPL", "a symbol far too long to sit inside a view", null, ""]`
	binaryJSON = `["QUFQTA==", "YSBzeW1ib2wgZmFyIHRvbyBsb25nIHRvIHNpdCBpbnNpZGUgYSB2aWV3", null, ""]`
)

// TestArrayRoundTrip takes every column out to kuma and back, and compares what
// arrives with what was sent. The comparison is on the printed values rather
// than on the types, since the three string layouts all come back as the one
// kuma stores and that is the intended answer rather than a mismatch.
func TestArrayRoundTrip(t *testing.T) {
	for _, c := range arrayCases {
		t.Run(c.name, func(t *testing.T) {
			in := fromJSON(t, c.dt, c.json)
			defer in.Release()

			mid, err := ImportArray(in)
			if err != nil {
				t.Fatalf("ImportArray: %v", err)
			}
			if !dtype.Equal(mid.DType(), c.want) {
				t.Errorf("came in as %s, want %s", mid.DType(), c.want)
			}
			if mid.Len() != in.Len() {
				t.Errorf("came in %d values long, want %d", mid.Len(), in.Len())
			}
			if mid.NullCount() != in.NullN() {
				t.Errorf("came in with %d nulls, want %d", mid.NullCount(), in.NullN())
			}

			out, err := ExportArray(mid)
			if err != nil {
				t.Fatalf("ExportArray: %v", err)
			}
			defer out.Release()

			if got, want := out.String(), in.String(); got != want {
				t.Errorf("came back as\n%s\nand went out as\n%s", got, want)
			}
		})
	}
}

// TestArrayRoundTripSliced is the same over a slice of each column, which is
// what an array with an offset on it is. The offset is carried rather than
// resolved, so this is the case where the two sides have to agree about what
// the shared buffers mean.
func TestArrayRoundTripSliced(t *testing.T) {
	for _, c := range arrayCases {
		t.Run(c.name, func(t *testing.T) {
			whole := fromJSON(t, c.dt, c.json)
			defer whole.Release()

			in := arrowarray.NewSlice(whole, 1, 4)
			defer in.Release()

			mid, err := ImportArray(in)
			if err != nil {
				t.Fatalf("ImportArray: %v", err)
			}
			if mid.Len() != 3 {
				t.Fatalf("came in %d values long, want 3", mid.Len())
			}
			if mid.NullCount() != 1 {
				t.Errorf("came in with %d nulls, want 1", mid.NullCount())
			}

			out, err := ExportArray(mid)
			if err != nil {
				t.Fatalf("ExportArray: %v", err)
			}
			defer out.Release()

			if got, want := out.String(), in.String(); got != want {
				t.Errorf("came back as\n%s\nand went out as\n%s", got, want)
			}
		})
	}
}

// TestImportSharesTheValueBuffer is the test the package exists to pass. The
// bytes on the kuma side have to be the bytes on the arrow-go side, and the
// only way to see that is to compare where they are rather than what is in
// them.
func TestImportSharesTheValueBuffer(t *testing.T) {
	in := fromJSON(t, arrow.PrimitiveTypes.Float64, `[1.5, 2.5, null, 4.5]`)
	defer in.Release()

	got, err := ImportArray(in)
	if err != nil {
		t.Fatalf("ImportArray: %v", err)
	}

	if a, b := &got.Buffer().Bytes()[0], &in.Data().Buffers()[1].Bytes()[0]; a != b {
		t.Errorf("the values are at %p on the kuma side and %p on the arrow side, "+
			"which means they were copied", a, b)
	}
	if a, b := &got.Validity().Bytes()[0], &in.Data().Buffers()[0].Bytes()[0]; a != b {
		t.Errorf("the validity bitmap is at %p on the kuma side and %p on the arrow side, "+
			"which means it was copied", a, b)
	}
}

// TestExportSharesTheValueBuffer is the same in the other direction.
func TestExportSharesTheValueBuffer(t *testing.T) {
	b, err := array.NewBuilder(dtype.Float64)
	if err != nil {
		t.Fatal(err)
	}
	b.AppendValues([]float64{1.5, 2.5})
	b.AppendNull()
	b.Append(4.5)
	in := b.Finish()

	out, err := ExportArray(in)
	if err != nil {
		t.Fatalf("ExportArray: %v", err)
	}
	defer out.Release()

	if a, c := &in.Buffer().Bytes()[0], &out.Data().Buffers()[1].Bytes()[0]; a != c {
		t.Errorf("the values are at %p on the kuma side and %p on the arrow side, "+
			"which means they were copied", a, c)
	}
	if a, c := &in.Validity().Bytes()[0], &out.Data().Buffers()[0].Bytes()[0]; a != c {
		t.Errorf("the validity bitmap is at %p on the kuma side and %p on the arrow side, "+
			"which means it was copied", a, c)
	}
}

// TestViewsShareTheirBlocks is the string half of the same claim. A view column
// arrives whole: the views are the same memory read as views, and the block
// holding the value that did not fit inline is the same block.
func TestViewsShareTheirBlocks(t *testing.T) {
	in := fromJSON(t, arrow.BinaryTypes.StringView, stringJSON)
	defer in.Release()

	got, err := ImportArray(in)
	if err != nil {
		t.Fatalf("ImportArray: %v", err)
	}

	blocks := got.Strings().Blocks()
	if len(blocks) != len(in.Data().Buffers())-2 {
		t.Fatalf("came in with %d blocks for %d data buffers", len(blocks), len(in.Data().Buffers())-2)
	}
	for i, b := range blocks {
		if len(b.Bytes()) == 0 {
			continue
		}
		if x, y := &b.Bytes()[0], &in.Data().Buffers()[i+2].Bytes()[0]; x != y {
			t.Errorf("block %d is at %p on the kuma side and %p on the arrow side", i, x, y)
		}
	}

	out, err := ExportArray(got)
	if err != nil {
		t.Fatalf("ExportArray: %v", err)
	}
	defer out.Release()

	if x, y := &out.Data().Buffers()[2].Bytes()[0], &in.Data().Buffers()[2].Bytes()[0]; x != y {
		t.Errorf("the block came back at %p and went out at %p", x, y)
	}
}

// TestOffsetStringsAreConverted is the one path that copies. It is here to say
// that it gives the right answer, and that the answer is in the view layout
// afterwards rather than still in offsets.
func TestOffsetStringsAreConverted(t *testing.T) {
	for _, dt := range []arrow.DataType{arrow.BinaryTypes.String, arrow.BinaryTypes.LargeString} {
		t.Run(dt.String(), func(t *testing.T) {
			in := fromJSON(t, dt, stringJSON)
			defer in.Release()

			got, err := ImportArray(in)
			if err != nil {
				t.Fatalf("ImportArray: %v", err)
			}
			if got.Strings() == nil {
				t.Fatal("came in with no views, and every string kuma holds is a view")
			}

			want := []string{"AAPL", "a symbol far too long to sit inside a view", "", ""}
			for i, w := range want {
				if v := string(got.Bytes(i)); v != w {
					t.Errorf("value %d is %q, want %q", i, v, w)
				}
			}
			if !got.IsNull(2) {
				t.Error("value 2 came in present, and it was sent as a null")
			}
		})
	}
}

// TestDictionaryRoundTrip carries a dictionary encoded column both ways, which
// is two columns and an agreement about which is which.
func TestDictionaryRoundTrip(t *testing.T) {
	dt := &arrow.DictionaryType{
		IndexType: arrow.PrimitiveTypes.Int32,
		ValueType: arrow.BinaryTypes.StringView,
	}

	// Written out of its two halves rather than from JSON, because arrow-go has
	// no builder for a dictionary whose values are views and that is the layout
	// under test.
	indices := fromJSON(t, arrow.PrimitiveTypes.Int32, `[0, 1, null, 0]`)
	defer indices.Release()
	values := fromJSON(t, arrow.BinaryTypes.StringView, `["AAPL", "MSFT"]`)
	defer values.Release()

	in := arrowarray.NewDictionaryArray(dt, indices, values)
	defer in.Release()

	mid, err := ImportArray(in)
	if err != nil {
		t.Fatalf("ImportArray: %v", err)
	}
	if mid.Dictionary() == nil {
		t.Fatal("came in with no dictionary on it")
	}
	if n := mid.Dictionary().Len(); n != 2 {
		t.Errorf("the dictionary has %d values, want 2", n)
	}
	if got := string(mid.Dictionary().Bytes(mid.Index(3))); got != "AAPL" {
		t.Errorf("value 3 is %q, want AAPL", got)
	}

	out, err := ExportArray(mid)
	if err != nil {
		t.Fatalf("ExportArray: %v", err)
	}
	defer out.Release()

	if got, want := out.String(), in.String(); got != want {
		t.Errorf("came back as %s and went out as %s", got, want)
	}
}

// TestNullColumn is the type with no values in it at all, which has no buffer
// to share and so is the one column that is built rather than wrapped.
func TestNullColumn(t *testing.T) {
	in := arrowarray.NewNull(5)
	defer in.Release()

	mid, err := ImportArray(in)
	if err != nil {
		t.Fatalf("ImportArray: %v", err)
	}
	if mid.Len() != 5 || mid.NullCount() != 5 {
		t.Fatalf("came in as %d values with %d nulls, want 5 and 5", mid.Len(), mid.NullCount())
	}

	out, err := ExportArray(mid)
	if err != nil {
		t.Fatalf("ExportArray: %v", err)
	}
	defer out.Release()

	if out.Len() != 5 || out.DataType().ID() != arrow.NULL {
		t.Errorf("came back as %d values of %s, want 5 of null", out.Len(), out.DataType())
	}
}

// TestEmptyColumn is the length that catches an unchecked index, since a
// zero length column has no first byte to take the address of.
func TestEmptyColumn(t *testing.T) {
	for _, dt := range []arrow.DataType{
		arrow.PrimitiveTypes.Int64,
		arrow.BinaryTypes.StringView,
		arrow.BinaryTypes.String,
		arrow.FixedWidthTypes.Boolean,
	} {
		t.Run(dt.String(), func(t *testing.T) {
			in := fromJSON(t, dt, `[]`)
			defer in.Release()

			mid, err := ImportArray(in)
			if err != nil {
				t.Fatalf("ImportArray: %v", err)
			}
			if mid.Len() != 0 {
				t.Fatalf("came in %d values long, want 0", mid.Len())
			}

			out, err := ExportArray(mid)
			if err != nil {
				t.Fatalf("ExportArray: %v", err)
			}
			defer out.Release()

			if out.Len() != 0 {
				t.Errorf("came back %d values long, want 0", out.Len())
			}
		})
	}
}

// TestNoNullsCarriesNoBitmap checks that a column with nothing missing does not
// grow a validity bitmap on the way through, since a bitmap of all ones is a
// cost every kernel then pays for nothing.
func TestNoNullsCarriesNoBitmap(t *testing.T) {
	in := fromJSON(t, arrow.PrimitiveTypes.Int64, `[1, 2, 3]`)
	defer in.Release()

	mid, err := ImportArray(in)
	if err != nil {
		t.Fatalf("ImportArray: %v", err)
	}
	if mid.Validity() != nil {
		t.Error("came in with a validity bitmap and there was nothing missing")
	}

	out, err := ExportArray(mid)
	if err != nil {
		t.Fatalf("ExportArray: %v", err)
	}
	defer out.Release()

	if b := out.Data().Buffers()[0]; b != nil {
		t.Error("went out with a validity bitmap and there was nothing missing")
	}
	if out.NullN() != 0 {
		t.Errorf("went out claiming %d nulls", out.NullN())
	}
}

func TestImportArrayErrors(t *testing.T) {
	if _, err := ImportArray(nil); err == nil {
		t.Error("ImportArray(nil) said nothing")
	}

	in := fromJSON(t, arrow.ListOf(arrow.PrimitiveTypes.Int64), `[[1, 2], null]`)
	defer in.Release()
	if _, err := ImportArray(in); err == nil {
		t.Error("ImportArray said nothing about a list column")
	}
}

// TestImportArrayWithNoValueBuffer is the malformed case, which cannot be built
// through arrow-go's own constructors and so is assembled by hand.
func TestImportArrayWithNoValueBuffer(t *testing.T) {
	data := arrowarray.NewData(arrow.PrimitiveTypes.Int64, 2, []*memory.Buffer{nil, nil}, nil, 0, 0)
	defer data.Release()

	_, err := ImportArray(arrowarray.MakeFromData(data))
	if err == nil {
		t.Fatal("ImportArray said nothing")
	}
	if !strings.Contains(err.Error(), "no value buffer") {
		t.Errorf("ImportArray says %q", err)
	}
}

// TestImportArrayWithAShortValidityBitmap is the other malformed case, where
// the bitmap does not have a bit for every value.
func TestImportArrayWithAShortValidityBitmap(t *testing.T) {
	data := arrowarray.NewData(arrow.PrimitiveTypes.Int64, 1000,
		[]*memory.Buffer{memory.NewBufferBytes(make([]byte, 4)), memory.NewBufferBytes(make([]byte, 8000))},
		nil, 1, 0)
	defer data.Release()

	_, err := ImportArray(arrowarray.MakeFromData(data))
	if err == nil {
		t.Fatal("ImportArray said nothing")
	}
	if !strings.Contains(err.Error(), "validity bitmap") {
		t.Errorf("ImportArray says %q", err)
	}
}

// TestImportArrayWithAShortViewBuffer is the same for the views, where the
// buffer has fewer than sixteen bytes per value in it.
func TestImportArrayWithAShortViewBuffer(t *testing.T) {
	data := arrowarray.NewData(arrow.BinaryTypes.StringView, 4,
		[]*memory.Buffer{nil, memory.NewBufferBytes(make([]byte, 32))}, nil, 0, 0)
	defer data.Release()

	_, err := ImportArray(arrowarray.MakeFromData(data))
	if err == nil {
		t.Fatal("ImportArray said nothing")
	}
	if !strings.Contains(err.Error(), "bytes of views") {
		t.Errorf("ImportArray says %q", err)
	}
}

// TestImportArrayWithAViewPointingNowhere is the reason the views are checked
// at all: a view naming a block that is not there is a read of memory that
// belongs to something else.
func TestImportArrayWithAViewPointingNowhere(t *testing.T) {
	view := make([]byte, 16)
	view[0] = 40 // a length past what fits inline, so the block is read
	view[8] = 7  // block seven, of the none that came with it

	data := arrowarray.NewData(arrow.BinaryTypes.StringView, 1,
		[]*memory.Buffer{nil, memory.NewBufferBytes(view)}, nil, 0, 0)
	defer data.Release()

	_, err := ImportArray(arrowarray.MakeFromData(data))
	if err == nil {
		t.Fatal("ImportArray said nothing")
	}
	if !strings.Contains(err.Error(), "block") {
		t.Errorf("ImportArray says %q", err)
	}
}

func TestExportArrayErrors(t *testing.T) {
	if _, err := ExportArray(nil); err == nil {
		t.Error("ExportArray(nil) said nothing")
	}
}

// fromJSON builds an arrow-go array, which is how the fixtures above are
// written. The offset it returns is for a reader that streams, and there is
// nothing streaming here.
func fromJSON(t *testing.T, dt arrow.DataType, data string) arrow.Array {
	t.Helper()

	a, _, err := arrowarray.FromJSON(memory.DefaultAllocator, dt, strings.NewReader(data))
	if err != nil {
		t.Fatalf("building a %s column from %s: %v", dt, data, err)
	}
	return a
}
