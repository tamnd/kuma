package arrowgo

import (
	"strings"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"

	"github.com/tamnd/kuma/dtype"
)

// TestTypesRoundTrip is the table of everything that crosses. Each row goes out
// and comes back, and the answer has to be the type it started as, parameters
// included.
func TestTypesRoundTrip(t *testing.T) {
	cases := []dtype.DataType{
		dtype.Null,
		dtype.Bool,
		dtype.Int8, dtype.Int16, dtype.Int32, dtype.Int64,
		dtype.Uint8, dtype.Uint16, dtype.Uint32, dtype.Uint64,
		dtype.Float32, dtype.Float64,
		dtype.String, dtype.Binary,
		dtype.Date32, dtype.Date64,
		dtype.Time32{Unit: dtype.Second},
		dtype.Time32{Unit: dtype.Millisecond},
		dtype.Time64{Unit: dtype.Microsecond},
		dtype.Time64{Unit: dtype.Nanosecond},
		dtype.Timestamp{Unit: dtype.Second},
		dtype.Timestamp{Unit: dtype.Millisecond, Zone: "UTC"},
		dtype.Timestamp{Unit: dtype.Microsecond, Zone: "Europe/London"},
		dtype.Timestamp{Unit: dtype.Nanosecond},
		dtype.Duration{Unit: dtype.Second},
		dtype.Duration{Unit: dtype.Nanosecond},
		dtype.Interval{Unit: dtype.YearMonth},
		dtype.Interval{Unit: dtype.DayTime},
		dtype.Interval{Unit: dtype.MonthDayNano},
		dtype.FixedSizeBinary{ByteWidth: 16},
		dtype.Decimal128{Precision: 18, Scale: 2},
		dtype.Decimal256{Precision: 50, Scale: -3},
		dtype.Dictionary{Index: dtype.Int32, Value: dtype.String},
		dtype.Dictionary{Index: dtype.Uint8, Value: dtype.Int64},
	}

	for _, want := range cases {
		t.Run(want.String(), func(t *testing.T) {
			out, err := ExportType(want)
			if err != nil {
				t.Fatalf("ExportType: %v", err)
			}
			got, err := ImportType(out)
			if err != nil {
				t.Fatalf("ImportType(%s): %v", out, err)
			}
			if !dtype.Equal(got, want) {
				t.Errorf("%s went out as %s and came back as %s", want, out, got)
			}
		})
	}
}

// TestLargeTypesRoundTrip is the pair kuma keeps only so that a file holding
// one can be read. They go out as themselves and come back as the view layout,
// which is the whole point of keeping them.
func TestLargeTypesRoundTrip(t *testing.T) {
	cases := []struct {
		from  dtype.DataType
		arrow arrow.DataType
		back  dtype.DataType
	}{
		{dtype.LargeString, arrow.BinaryTypes.LargeString, dtype.String},
		{dtype.LargeBinary, arrow.BinaryTypes.LargeBinary, dtype.Binary},
	}

	for _, c := range cases {
		t.Run(c.from.String(), func(t *testing.T) {
			out, err := ExportType(c.from)
			if err != nil {
				t.Fatalf("ExportType: %v", err)
			}
			if out.ID() != c.arrow.ID() {
				t.Fatalf("%s went out as %s, want %s", c.from, out, c.arrow)
			}
			got, err := ImportType(out)
			if err != nil {
				t.Fatalf("ImportType: %v", err)
			}
			if !dtype.Equal(got, c.back) {
				t.Errorf("%s came back as %s, want %s", c.from, got, c.back)
			}
		})
	}
}

// TestImportTypeFolds covers the layouts that are several types on the arrow-go
// side and one on the kuma side.
func TestImportTypeFolds(t *testing.T) {
	cases := []struct {
		in   arrow.DataType
		want dtype.DataType
	}{
		{arrow.BinaryTypes.String, dtype.String},
		{arrow.BinaryTypes.LargeString, dtype.String},
		{arrow.BinaryTypes.StringView, dtype.String},
		{arrow.BinaryTypes.Binary, dtype.Binary},
		{arrow.BinaryTypes.LargeBinary, dtype.Binary},
		{arrow.BinaryTypes.BinaryView, dtype.Binary},
	}

	for _, c := range cases {
		t.Run(c.in.String(), func(t *testing.T) {
			got, err := ImportType(c.in)
			if err != nil {
				t.Fatalf("ImportType: %v", err)
			}
			if !dtype.Equal(got, c.want) {
				t.Errorf("%s came in as %s, want %s", c.in, got, c.want)
			}
		})
	}
}

// TestExportStringsAsViews pins the direction that matters for speed: a kuma
// string goes out as a view and not as offsets, because that is the layout it
// is already in.
func TestExportStringsAsViews(t *testing.T) {
	cases := []struct {
		in   dtype.DataType
		want arrow.Type
	}{
		{dtype.String, arrow.STRING_VIEW},
		{dtype.Binary, arrow.BINARY_VIEW},
	}

	for _, c := range cases {
		t.Run(c.in.String(), func(t *testing.T) {
			got, err := ExportType(c.in)
			if err != nil {
				t.Fatalf("ExportType: %v", err)
			}
			if got.ID() != c.want {
				t.Errorf("%s went out as %s, want %v", c.in, got, c.want)
			}
		})
	}
}

// TestDictionaryLosesOrdered is the one thing a type does not carry, written
// down as a test so that it is a decision rather than an oversight.
func TestDictionaryLosesOrdered(t *testing.T) {
	in := &arrow.DictionaryType{
		IndexType: arrow.PrimitiveTypes.Int32,
		ValueType: arrow.BinaryTypes.StringView,
		Ordered:   true,
	}

	got, err := ImportType(in)
	if err != nil {
		t.Fatalf("ImportType: %v", err)
	}
	want := dtype.Dictionary{Index: dtype.Int32, Value: dtype.String}
	if !dtype.Equal(got, want) {
		t.Fatalf("came in as %s, want %s", got, want)
	}

	out, err := ExportType(got)
	if err != nil {
		t.Fatalf("ExportType: %v", err)
	}
	if out.(*arrow.DictionaryType).Ordered {
		t.Error("went back out ordered, and the flag was not carried to say so")
	}
}

func TestImportTypeErrors(t *testing.T) {
	cases := []struct {
		name string
		in   arrow.DataType
		want string
	}{
		{"nothing at all", nil, "nil arrow type"},
		{"a list", arrow.ListOf(arrow.PrimitiveTypes.Int64), "does not cross"},
		{"a struct", arrow.StructOf(arrow.Field{Name: "a", Type: arrow.PrimitiveTypes.Int64}), "does not cross"},
		{"a map", arrow.MapOf(arrow.BinaryTypes.String, arrow.PrimitiveTypes.Int64), "does not cross"},
		{"a fixed size list", arrow.FixedSizeListOf(4, arrow.PrimitiveTypes.Int64), "does not cross"},
		{
			name: "a dictionary of a list",
			in: &arrow.DictionaryType{
				IndexType: arrow.PrimitiveTypes.Int32,
				ValueType: arrow.ListOf(arrow.PrimitiveTypes.Int64),
			},
			want: "the values of a",
		},
		{
			name: "a dictionary indexed by a list",
			in: &arrow.DictionaryType{
				IndexType: arrow.ListOf(arrow.PrimitiveTypes.Int64),
				ValueType: arrow.BinaryTypes.StringView,
			},
			want: "the index of a",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := ImportType(c.in)
			if err == nil {
				t.Fatal("ImportType said nothing")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("ImportType says %q, and it should mention %q", err, c.want)
			}
		})
	}
}

func TestExportTypeErrors(t *testing.T) {
	cases := []struct {
		name string
		in   dtype.DataType
		want string
	}{
		{"nothing at all", nil, "nil kuma type"},
		{"a list", dtype.List{Elem: dtype.Int64}, "does not cross"},
		{"a struct", dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}}, "does not cross"},
		{"a map", dtype.Map{Key: dtype.String, Value: dtype.Int64}, "does not cross"},
		{"an interval of no unit", dtype.Interval{Unit: 9}, "no interval unit"},
		{"a dictionary of a list", dtype.Dictionary{Index: dtype.Int32, Value: dtype.List{Elem: dtype.Int64}}, "the values of a"},
		{"a dictionary indexed by a list", dtype.Dictionary{Index: dtype.List{Elem: dtype.Int64}, Value: dtype.String}, "the index of a"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := ExportType(c.in)
			if err == nil {
				t.Fatal("ExportType said nothing")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("ExportType says %q, and it should mention %q", err, c.want)
			}
		})
	}
}

// TestSchemaRoundTrip takes a schema out and back with the metadata on it, at
// both levels, since Arrow metadata is a list and the order is part of it.
func TestSchemaRoundTrip(t *testing.T) {
	want := dtype.Schema{
		Fields: []dtype.Field{
			{Name: "symbol", Type: dtype.String, Nullable: false},
			{Name: "price", Type: dtype.Float64, Nullable: true},
			{
				Name:     "ts",
				Type:     dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"},
				Nullable: true,
				Metadata: dtype.Metadata{{Key: "unit", Value: "us"}, {Key: "source", Value: "feed"}},
			},
		},
		Metadata: dtype.Metadata{{Key: "written_by", Value: "kuma"}},
	}

	out, err := ExportSchema(want)
	if err != nil {
		t.Fatalf("ExportSchema: %v", err)
	}
	got, err := ImportSchema(out)
	if err != nil {
		t.Fatalf("ImportSchema: %v", err)
	}
	if !got.Equal(want) {
		t.Errorf("the schema came back as\n%s\nand went out as\n%s", got, want)
	}
}

// TestSchemaWithNoMetadata checks the empty case comes back as nil rather than
// as an empty slice, since Equal compares lengths and a reader that stored
// nothing should get nothing.
func TestSchemaWithNoMetadata(t *testing.T) {
	want := dtype.Schema{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64, Nullable: true}}}

	out, err := ExportSchema(want)
	if err != nil {
		t.Fatalf("ExportSchema: %v", err)
	}
	got, err := ImportSchema(out)
	if err != nil {
		t.Fatalf("ImportSchema: %v", err)
	}
	if got.Metadata != nil {
		t.Errorf("the schema came back with %v for metadata, want nil", got.Metadata)
	}
	if got.Fields[0].Metadata != nil {
		t.Errorf("the field came back with %v for metadata, want nil", got.Fields[0].Metadata)
	}
}

func TestSchemaErrors(t *testing.T) {
	if _, err := ImportSchema(nil); err == nil {
		t.Error("ImportSchema(nil) said nothing")
	}

	_, err := ExportSchema(dtype.Schema{Fields: []dtype.Field{{Name: "xs", Type: dtype.List{Elem: dtype.Int64}}}})
	if err == nil {
		t.Fatal("ExportSchema said nothing about a list column")
	}
	if !strings.Contains(err.Error(), `the column "xs"`) {
		t.Errorf("ExportSchema says %q, and it should name the column", err)
	}

	in := arrow.NewSchema([]arrow.Field{{Name: "xs", Type: arrow.ListOf(arrow.PrimitiveTypes.Int64)}}, nil)
	if _, err := ImportSchema(in); err == nil {
		t.Error("ImportSchema said nothing about a list column")
	}
}
