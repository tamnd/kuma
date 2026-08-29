package parquet_test

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/parquet"
)

// TestSetSchemaRoundTrip takes the schema of every file in testdata, writes it
// back out and reads it again.
//
// This is the test that says the two halves agree. Between them the files hold
// every logical type this package reads, a two page column, three level lists,
// a map, a struct, a list of lists and a list of structs, and none of that is
// something a writer gets right by accident: a group written with the wrong
// number of children swallows the columns after it, and an annotation left off
// a node changes what the column means.
func TestSetSchemaRoundTrip(t *testing.T) {
	for _, name := range files(t) {
		t.Run(name, func(t *testing.T) {
			want, err := read(t, name).Schema()
			if err != nil {
				t.Fatalf("Schema: %v", err)
			}

			var m parquet.Metadata
			if err = m.SetSchema(want); err != nil {
				t.Fatalf("SetSchema: %v", err)
			}

			got, err := m.Schema()
			if err != nil {
				t.Fatalf("reading back the schema: %v", err)
			}
			if !got.Equal(want) {
				t.Errorf("the schema came back different\n got %s\nwant %s", got, want)
			}
		})
	}
}

// TestSetSchemaFooter puts a written schema through a whole footer, which is
// the thing a file writer will do with it. The round trip above goes through
// the structures and this one goes through the bytes.
func TestSetSchemaFooter(t *testing.T) {
	want := dtype.Schema{
		Fields: []dtype.Field{
			{Name: "id", Type: dtype.Int64},
			{Name: "name", Type: dtype.String, Nullable: true},
			{Name: "tags", Type: dtype.List{Elem: dtype.String}, Nullable: true},
			{Name: "when", Type: dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}, Nullable: true},
		},
		Metadata: dtype.Metadata{{Key: "written by", Value: "a test"}},
	}

	m := parquet.Metadata{Version: 2, NumRows: 0}
	if err := m.SetSchema(want); err != nil {
		t.Fatalf("SetSchema: %v", err)
	}

	got, err := footer(t, &m).Schema()
	if err != nil {
		t.Fatalf("reading back the schema: %v", err)
	}
	if !got.Equal(want) {
		t.Errorf("the schema came back different\n got %s\nwant %s", got, want)
	}
}

// TestSetSchemaNodes checks the nodes themselves rather than what they read
// back as.
//
// A round trip proves the two halves of this package agree and says nothing
// about whether either of them agrees with the format, so the elements here are
// written out in full and are the ones pyarrow writes for the same columns.
// Every field of every node is given, including the ones that say a writer left
// something out, because those are not the zero of their types and a node that
// takes the zero value claims to be a required boolean.
func TestSetSchemaNodes(t *testing.T) {
	for _, c := range []struct {
		name  string
		field dtype.Field
		want  []parquet.SchemaElement
	}{
		{
			name:  "a boolean, which is the one type with nothing to say about it",
			field: dtype.Field{Name: "flag", Type: dtype.Bool},
			want: []parquet.SchemaElement{{
				Name: "flag", Type: parquet.Boolean, Repetition: parquet.Required,
				Converted: parquet.NoConverted,
			}},
		},
		{
			name:  "an int32, which is a parquet type and carries no annotation",
			field: dtype.Field{Name: "count", Type: dtype.Int32, Nullable: true},
			want: []parquet.SchemaElement{{
				Name: "count", Type: parquet.Int32, Repetition: parquet.Optional,
				Converted: parquet.NoConverted,
			}},
		},
		{
			name:  "an int8, which travels in an int32 and says so twice",
			field: dtype.Field{Name: "small", Type: dtype.Int8, Nullable: true},
			want: []parquet.SchemaElement{{
				Name: "small", Type: parquet.Int32, Repetition: parquet.Optional,
				Converted: parquet.ConvertedInt8,
				Logical: parquet.LogicalType{
					Kind: parquet.IntegerLogical, BitWidth: 8, Signed: true,
				},
			}},
		},
		{
			name:  "a uint64, which is an int64 with a note saying it is not one",
			field: dtype.Field{Name: "big", Type: dtype.Uint64},
			want: []parquet.SchemaElement{{
				Name: "big", Type: parquet.Int64, Repetition: parquet.Required,
				Converted: parquet.ConvertedUint64,
				Logical: parquet.LogicalType{
					Kind: parquet.IntegerLogical, BitWidth: 64,
				},
			}},
		},
		{
			name:  "a string",
			field: dtype.Field{Name: "name", Type: dtype.String, Nullable: true},
			want: []parquet.SchemaElement{{
				Name: "name", Type: parquet.ByteArray, Repetition: parquet.Optional,
				Converted: parquet.ConvertedUTF8,
				Logical:   parquet.LogicalType{Kind: parquet.StringLogical},
			}},
		},
		{
			name:  "a large string, which parquet has no larger version of",
			field: dtype.Field{Name: "name", Type: dtype.LargeString, Nullable: true},
			want: []parquet.SchemaElement{{
				Name: "name", Type: parquet.ByteArray, Repetition: parquet.Optional,
				Converted: parquet.ConvertedUTF8,
				Logical:   parquet.LogicalType{Kind: parquet.StringLogical},
			}},
		},
		{
			name:  "bytes with nothing said about them",
			field: dtype.Field{Name: "blob", Type: dtype.Binary, Nullable: true},
			want: []parquet.SchemaElement{{
				Name: "blob", Type: parquet.ByteArray, Repetition: parquet.Optional,
				Converted: parquet.NoConverted,
			}},
		},
		{
			name:  "bytes of a width the schema gives",
			field: dtype.Field{Name: "hash", Type: dtype.FixedSizeBinary{ByteWidth: 16}},
			want: []parquet.SchemaElement{{
				Name: "hash", Type: parquet.FixedLenByteArray, TypeLength: 16,
				Repetition: parquet.Required, Converted: parquet.NoConverted,
			}},
		},
		{
			name:  "a decimal of nine digits, which fits in four bytes",
			field: dtype.Field{Name: "price", Type: dtype.Decimal128{Precision: 9, Scale: 2}, Nullable: true},
			want: []parquet.SchemaElement{{
				Name: "price", Type: parquet.FixedLenByteArray, TypeLength: 4,
				Repetition: parquet.Optional, Converted: parquet.ConvertedDecimal,
				Logical: parquet.LogicalType{
					Kind: parquet.DecimalLogical, Precision: 9, Scale: 2,
				},
				Precision: 9, Scale: 2,
			}},
		},
		{
			name:  "a decimal of forty digits, which fits in seventeen",
			field: dtype.Field{Name: "huge", Type: dtype.Decimal256{Precision: 40, Scale: 4}},
			want: []parquet.SchemaElement{{
				Name: "huge", Type: parquet.FixedLenByteArray, TypeLength: 17,
				Repetition: parquet.Required, Converted: parquet.ConvertedDecimal,
				Logical: parquet.LogicalType{
					Kind: parquet.DecimalLogical, Precision: 40, Scale: 4,
				},
				Precision: 40, Scale: 4,
			}},
		},
		{
			name:  "a date",
			field: dtype.Field{Name: "day", Type: dtype.Date32, Nullable: true},
			want: []parquet.SchemaElement{{
				Name: "day", Type: parquet.Int32, Repetition: parquet.Optional,
				Converted: parquet.ConvertedDate,
				Logical:   parquet.LogicalType{Kind: parquet.DateLogical},
			}},
		},
		{
			name:  "a time of day in milliseconds",
			field: dtype.Field{Name: "clock", Type: dtype.Time32{Unit: dtype.Millisecond}},
			want: []parquet.SchemaElement{{
				Name: "clock", Type: parquet.Int32, Repetition: parquet.Required,
				Converted: parquet.ConvertedTimeMillis,
				Logical: parquet.LogicalType{
					Kind: parquet.TimeLogical, Unit: parquet.Millis,
				},
			}},
		},
		{
			name:  "a time of day in nanoseconds, which no converted type says",
			field: dtype.Field{Name: "clock", Type: dtype.Time64{Unit: dtype.Nanosecond}},
			want: []parquet.SchemaElement{{
				Name: "clock", Type: parquet.Int64, Repetition: parquet.Required,
				Converted: parquet.NoConverted,
				Logical: parquet.LogicalType{
					Kind: parquet.TimeLogical, Unit: parquet.Nanos,
				},
			}},
		},
		{
			name: "an instant, which is a timestamp with a zone",
			field: dtype.Field{
				Name: "moment",
				Type: dtype.Timestamp{Unit: dtype.Microsecond, Zone: "Asia/Tokyo"},
			},
			want: []parquet.SchemaElement{{
				Name: "moment", Type: parquet.Int64, Repetition: parquet.Required,
				Converted: parquet.ConvertedTimestampMicros,
				Logical: parquet.LogicalType{
					Kind: parquet.TimestampLogical, Unit: parquet.Micros, UTC: true,
				},
			}},
		},
		{
			name: "an instant in nanoseconds, which is past where the converted types stop",
			field: dtype.Field{
				Name: "moment",
				Type: dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "UTC"},
			},
			want: []parquet.SchemaElement{{
				Name: "moment", Type: parquet.Int64, Repetition: parquet.Required,
				Converted: parquet.NoConverted,
				Logical: parquet.LogicalType{
					Kind: parquet.TimestampLogical, Unit: parquet.Nanos, UTC: true,
				},
			}},
		},
		{
			name: "a wall clock reading, which the converted types cannot say",
			field: dtype.Field{
				Name: "local", Type: dtype.Timestamp{Unit: dtype.Millisecond}, Nullable: true,
			},
			want: []parquet.SchemaElement{{
				Name: "local", Type: parquet.Int64, Repetition: parquet.Optional,
				Converted: parquet.NoConverted,
				Logical: parquet.LogicalType{
					Kind: parquet.TimestampLogical, Unit: parquet.Millis,
				},
			}},
		},
		{
			name:  "a column of nothing",
			field: dtype.Field{Name: "nothing", Type: dtype.Null, Nullable: true},
			want: []parquet.SchemaElement{{
				Name: "nothing", Type: parquet.Int32, Repetition: parquet.Optional,
				Converted: parquet.NoConverted,
				Logical:   parquet.LogicalType{Kind: parquet.UnknownLogical},
			}},
		},
		{
			name:  "a dictionary, which is how a page is written rather than a type",
			field: dtype.Field{Name: "code", Type: dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String}, Nullable: true},
			want: []parquet.SchemaElement{{
				Name: "code", Type: parquet.ByteArray, Repetition: parquet.Optional,
				Converted: parquet.ConvertedUTF8,
				Logical:   parquet.LogicalType{Kind: parquet.StringLogical},
			}},
		},
		{
			name:  "a list, which is three nodes",
			field: dtype.Field{Name: "tags", Type: dtype.List{Elem: dtype.String}, Nullable: true},
			want: []parquet.SchemaElement{
				{
					Name: "tags", Type: parquet.NoType, Repetition: parquet.Optional,
					NumChildren: 1, Converted: parquet.ConvertedList,
					Logical: parquet.LogicalType{Kind: parquet.ListLogical},
				},
				{
					Name: "list", Type: parquet.NoType, Repetition: parquet.Repeated,
					NumChildren: 1, Converted: parquet.NoConverted,
				},
				{
					Name: "element", Type: parquet.ByteArray, Repetition: parquet.Optional,
					Converted: parquet.ConvertedUTF8,
					Logical:   parquet.LogicalType{Kind: parquet.StringLogical},
				},
			},
		},
		{
			name:  "a fixed size list, which parquet writes as a list of any size",
			field: dtype.Field{Name: "point", Type: dtype.FixedSizeList{Elem: dtype.Float64, Len: 3}},
			want: []parquet.SchemaElement{
				{
					Name: "point", Type: parquet.NoType, Repetition: parquet.Required,
					NumChildren: 1, Converted: parquet.ConvertedList,
					Logical: parquet.LogicalType{Kind: parquet.ListLogical},
				},
				{
					Name: "list", Type: parquet.NoType, Repetition: parquet.Repeated,
					NumChildren: 1, Converted: parquet.NoConverted,
				},
				{
					Name: "element", Type: parquet.Double, Repetition: parquet.Optional,
					Converted: parquet.NoConverted,
				},
			},
		},
		{
			name:  "a large list, which parquet writes as an ordinary one",
			field: dtype.Field{Name: "runs", Type: dtype.LargeList{Elem: dtype.Int64}, Nullable: true},
			want: []parquet.SchemaElement{
				{
					Name: "runs", Type: parquet.NoType, Repetition: parquet.Optional,
					NumChildren: 1, Converted: parquet.ConvertedList,
					Logical: parquet.LogicalType{Kind: parquet.ListLogical},
				},
				{
					Name: "list", Type: parquet.NoType, Repetition: parquet.Repeated,
					NumChildren: 1, Converted: parquet.NoConverted,
				},
				{
					Name: "element", Type: parquet.Int64, Repetition: parquet.Optional,
					Converted: parquet.NoConverted,
				},
			},
		},
		{
			name: "a struct, which is a group with no annotation on it",
			field: dtype.Field{Name: "point", Nullable: true, Type: dtype.Struct{Fields: []dtype.Field{
				{Name: "x", Type: dtype.Float64},
				{Name: "y", Type: dtype.Float64, Nullable: true},
			}}},
			want: []parquet.SchemaElement{
				{
					Name: "point", Type: parquet.NoType, Repetition: parquet.Optional,
					NumChildren: 2, Converted: parquet.NoConverted,
				},
				{
					Name: "x", Type: parquet.Double, Repetition: parquet.Required,
					Converted: parquet.NoConverted,
				},
				{
					Name: "y", Type: parquet.Double, Repetition: parquet.Optional,
					Converted: parquet.NoConverted,
				},
			},
		},
		{
			name:  "a map, which is a group holding a repeated pair",
			field: dtype.Field{Name: "props", Type: dtype.Map{Key: dtype.String, Value: dtype.Int64}, Nullable: true},
			want: []parquet.SchemaElement{
				{
					Name: "props", Type: parquet.NoType, Repetition: parquet.Optional,
					NumChildren: 1, Converted: parquet.ConvertedMap,
					Logical: parquet.LogicalType{Kind: parquet.MapLogical},
				},
				{
					Name: "key_value", Type: parquet.NoType, Repetition: parquet.Repeated,
					NumChildren: 2, Converted: parquet.NoConverted,
				},
				{
					Name: "key", Type: parquet.ByteArray, Repetition: parquet.Required,
					Converted: parquet.ConvertedUTF8,
					Logical:   parquet.LogicalType{Kind: parquet.StringLogical},
				},
				{
					Name: "value", Type: parquet.Int64, Repetition: parquet.Optional,
					Converted: parquet.NoConverted,
				},
			},
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			var m parquet.Metadata
			if err := m.SetSchema(dtype.Schema{Fields: []dtype.Field{c.field}}); err != nil {
				t.Fatalf("SetSchema: %v", err)
			}

			root := parquet.SchemaElement{
				Name: "schema", Type: parquet.NoType, Repetition: parquet.Required,
				NumChildren: 1, Converted: parquet.NoConverted,
			}
			if len(m.Nodes) == 0 || !reflect.DeepEqual(m.Nodes[0], root) {
				t.Fatalf("the root is %+v and a root is %+v", m.Nodes[0], root)
			}
			if got := m.Nodes[1:]; !reflect.DeepEqual(got, c.want) {
				t.Errorf("the nodes are wrong\n got %+v\nwant %+v", got, c.want)
			}
		})
	}
}

// TestSetSchemaMetadata checks that the pairs on a schema go into the footer
// and that a schema with none writes none.
//
// The second half is the difference between absent and empty that the footer
// writer cares about everywhere: a file that says nothing and a file that says
// there is nothing are different files, and the smaller one is right.
func TestSetSchemaMetadata(t *testing.T) {
	var m parquet.Metadata
	if err := m.SetSchema(dtype.Schema{Metadata: dtype.Metadata{
		{Key: "a", Value: "1"},
		{Key: "b", Value: "2"},
	}}); err != nil {
		t.Fatalf("SetSchema: %v", err)
	}

	want := []parquet.KeyValue{{Key: "a", Value: "1"}, {Key: "b", Value: "2"}}
	if !reflect.DeepEqual(m.KeyValue, want) {
		t.Errorf("the metadata is %+v and it should be %+v", m.KeyValue, want)
	}

	if err := m.SetSchema(dtype.Schema{}); err != nil {
		t.Fatalf("SetSchema: %v", err)
	}
	if m.KeyValue != nil {
		t.Errorf("a schema with no metadata wrote %+v", m.KeyValue)
	}
	if len(m.Nodes) != 1 {
		t.Errorf("a schema with no columns wrote %d nodes and it should write the root", len(m.Nodes))
	}
}

// TestSetSchemaRefused is every type kuma has and parquet has not.
//
// Each of them could be written as something near enough: a duration is an
// int64, a date64 is a date with the days multiplied out, an interval is the
// same three counts with the last one divided down. Each of those is a column
// that reads back as a different type or as different values, so each is an
// error naming the column instead. The last few check that the refusal comes
// back out of a group rather than being lost on the way.
func TestSetSchemaRefused(t *testing.T) {
	deep := dtype.DataType(dtype.Int32)
	for range 40 {
		deep = dtype.List{Elem: deep}
	}

	for _, c := range []struct {
		name  string
		field dtype.Field
	}{
		{"a duration", dtype.Field{Name: "took", Type: dtype.Duration{Unit: dtype.Nanosecond}}},
		{"an interval", dtype.Field{Name: "every", Type: dtype.Interval{Unit: dtype.MonthDayNano}}},
		{"a date in milliseconds", dtype.Field{Name: "day", Type: dtype.Date64}},
		{"a time of day in seconds", dtype.Field{Name: "clock", Type: dtype.Time32{Unit: dtype.Second}}},
		{"a timestamp in seconds", dtype.Field{Name: "when", Type: dtype.Timestamp{Unit: dtype.Second}}},
		{"bytes of no width", dtype.Field{Name: "nothing", Type: dtype.FixedSizeBinary{}}},
		{"a struct with no fields", dtype.Field{Name: "empty", Type: dtype.Struct{}}},
		{"a list that nests deeper than a file", dtype.Field{Name: "deep", Type: deep}},
		{"a list of durations", dtype.Field{Name: "took", Type: dtype.List{Elem: dtype.Duration{}}}},
		{"a map keyed by durations", dtype.Field{Name: "took", Type: dtype.Map{
			Key: dtype.Duration{}, Value: dtype.Int64,
		}}},
		{"a map of durations", dtype.Field{Name: "took", Type: dtype.Map{
			Key: dtype.String, Value: dtype.Duration{},
		}}},
		{"a struct holding a duration", dtype.Field{Name: "row", Type: dtype.Struct{Fields: []dtype.Field{
			{Name: "took", Type: dtype.Duration{}},
		}}}},
		{"a dictionary of durations", dtype.Field{Name: "took", Type: dtype.Dictionary{
			Index: dtype.Uint32, Value: dtype.Duration{},
		}}},
	} {
		t.Run(c.name, func(t *testing.T) {
			var m parquet.Metadata
			err := m.SetSchema(dtype.Schema{Fields: []dtype.Field{c.field}})
			if !errors.Is(err, parquet.ErrUnsupported) {
				t.Fatalf("writing %s gave %v", c.name, err)
			}
			if !strings.Contains(err.Error(), c.field.Name) {
				t.Errorf("the error does not name the column: %v", err)
			}
		})
	}
}

// TestSetSchemaInvalid is a schema that is not a schema, which is refused
// before any of it is written.
//
// The check is dtype's rather than this package's, since a schema with two
// columns of one name or a field with no type is wrong wherever it turns up.
// It is worth doing here because everything below reads a type without asking
// whether it means anything.
func TestSetSchemaInvalid(t *testing.T) {
	for _, c := range []struct {
		name   string
		schema dtype.Schema
	}{
		{"a field with no name", dtype.Schema{Fields: []dtype.Field{{Type: dtype.Int64}}}},
		{"a field with no type", dtype.Schema{Fields: []dtype.Field{{Name: "id"}}}},
		{"two fields of one name", dtype.Schema{Fields: []dtype.Field{
			{Name: "id", Type: dtype.Int64},
			{Name: "id", Type: dtype.String},
		}}},
		{"a decimal of no digits", dtype.Schema{Fields: []dtype.Field{
			{Name: "price", Type: dtype.Decimal128{Precision: 0, Scale: 2}},
		}}},
	} {
		t.Run(c.name, func(t *testing.T) {
			var m parquet.Metadata
			err := m.SetSchema(c.schema)
			if err == nil {
				t.Fatal("it was written")
			}
			if !strings.HasPrefix(err.Error(), "parquet: ") {
				t.Errorf("the error does not say who refused it: %v", err)
			}
			if len(m.Nodes) != 0 {
				t.Errorf("it wrote %d nodes before refusing", len(m.Nodes))
			}
		})
	}
}

// TestSetSchemaColumns walks the leaves of a written schema, which is what a
// file writer does with it: the columns in the order they come out here are the
// order every row group holds its chunks in, and the levels are what a page
// writer needs to know how many of them to write.
func TestSetSchemaColumns(t *testing.T) {
	var m parquet.Metadata
	err := m.SetSchema(dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64},
		{Name: "name", Type: dtype.String, Nullable: true},
		{Name: "tags", Type: dtype.List{Elem: dtype.Int32}, Nullable: true},
	}})
	if err != nil {
		t.Fatalf("SetSchema: %v", err)
	}

	columns, err := m.Columns()
	if err != nil {
		t.Fatalf("Columns: %v", err)
	}

	for i, want := range []struct {
		name        string
		definition  int
		repetition  int
		columnsType dtype.DataType
	}{
		{"id", 0, 0, dtype.Int64},
		{"name", 1, 0, dtype.String},
		{"tags.list.element", 3, 1, dtype.Int32},
	} {
		if i >= len(columns) {
			t.Fatalf("the schema has %d columns and it should have 3", len(columns))
		}
		got := columns[i]
		if got.Name() != want.name {
			t.Errorf("column %d is %q and it should be %q", i, got.Name(), want.name)
		}
		if got.MaxDefinition != want.definition || got.MaxRepetition != want.repetition {
			t.Errorf("%s is %d and %d levels deep and it should be %d and %d",
				want.name, got.MaxDefinition, got.MaxRepetition, want.definition, want.repetition)
		}
		if !dtype.Equal(got.Type, want.columnsType) {
			t.Errorf("%s holds %s and it should hold %s", want.name, got.Type, want.columnsType)
		}
	}
}

// BenchmarkSetSchema writes the schema of the file with the most types in it,
// which is what a file writer pays once per file.
func BenchmarkSetSchema(b *testing.B) {
	f, err := os.Open(filepath.Join("testdata", "alltypes.parquet"))
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		b.Fatal(err)
	}
	read, err := parquet.ReadMetadata(f, info.Size())
	if err != nil {
		b.Fatal(err)
	}
	s, err := read.Schema()
	if err != nil {
		b.Fatal(err)
	}

	var m parquet.Metadata
	for b.Loop() {
		if err := m.SetSchema(s); err != nil {
			b.Fatal(err)
		}
	}
}
