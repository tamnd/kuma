package parquet_test

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/parquet"
)

// A schema is a flat list of nodes where every group says how many of the nodes
// after it belong to it, and the shapes a reader has to cope with are mostly
// shapes pyarrow will not write. So the tests below spell the list out, one
// helper per line of the schema they stand for, and the helpers do nothing but
// fill in a struct.

// schemaOf wraps nodes in a root that claims the given number of fields.
func schemaOf(fields int32, nodes ...parquet.SchemaElement) *parquet.Metadata {
	root := parquet.SchemaElement{Name: "schema", NumChildren: fields}
	return &parquet.Metadata{Nodes: append([]parquet.SchemaElement{root}, nodes...)}
}

// leaf is a column.
//
// The annotation has to be set to nothing rather than left alone, because every
// enumeration the format defines starts at zero and absent is not zero: the
// zero of a converted type is the one meaning a string. A reader gets this
// right by writing the absent values in before it reads a node, and a test that
// builds a node has to write them in itself.
func leaf(name string, r parquet.Repetition, t parquet.Type) parquet.SchemaElement {
	return parquet.SchemaElement{
		Name:       name,
		Type:       t,
		Repetition: r,
		Converted:  parquet.NoConverted,
	}
}

// group is a node with fields under it, which has no type of its own.
func group(name string, r parquet.Repetition, fields int32) parquet.SchemaElement {
	return parquet.SchemaElement{
		Name:        name,
		Type:        parquet.NoType,
		Repetition:  r,
		NumChildren: fields,
		Converted:   parquet.NoConverted,
	}
}

// means annotates a node the way a file written this decade does.
func means(e parquet.SchemaElement, l parquet.LogicalType) parquet.SchemaElement {
	e.Logical = l
	return e
}

// converts annotates a node the older way, which is what a file written before
// the logical types existed has and all it has.
func converts(e parquet.SchemaElement, c parquet.ConvertedType) parquet.SchemaElement {
	e.Converted = c
	return e
}

// wide gives a fixed length column its width.
func wide(e parquet.SchemaElement, width int32) parquet.SchemaElement {
	e.TypeLength = width
	return e
}

// TestSchemaFile is the schema of a file holding a column of every flat type,
// read as kuma types. It is the mapping the whole file is for: fifteen columns
// written as eight physical types, and the annotations are what tell them
// apart.
func TestSchemaFile(t *testing.T) {
	s, err := read(t, "alltypes.parquet").Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}

	want := []dtype.Field{
		{Name: "flag", Type: dtype.Bool},
		{Name: "small", Type: dtype.Int8, Nullable: true},
		{Name: "count", Type: dtype.Int32, Nullable: true},
		{Name: "total", Type: dtype.Int64, Nullable: true},
		{Name: "unsigned", Type: dtype.Uint32, Nullable: true},
		{Name: "ratio", Type: dtype.Float32, Nullable: true},
		{Name: "weight", Type: dtype.Float64, Nullable: true},
		{Name: "name", Type: dtype.String, Nullable: true},
		{Name: "blob", Type: dtype.Binary, Nullable: true},
		{Name: "fixed", Type: dtype.FixedSizeBinary{ByteWidth: 4}, Nullable: true},
		{Name: "price", Type: dtype.Decimal128{Precision: 9, Scale: 2}, Nullable: true},
		{Name: "day", Type: dtype.Date32, Nullable: true},
		{Name: "clock", Type: dtype.Time64{Unit: dtype.Microsecond}, Nullable: true},
		{Name: "moment", Type: dtype.Timestamp{Unit: dtype.Millisecond, Zone: "UTC"}, Nullable: true},
		{Name: "local", Type: dtype.Timestamp{Unit: dtype.Microsecond}, Nullable: true},
	}
	compare(t, s, want)

	// The Arrow schema pyarrow attached to the file is still on the schema,
	// because nothing here reads it and dropping it would lose what it says.
	if v, ok := s.Metadata.Get("ARROW:schema"); !ok || v == "" {
		t.Error("the arrow schema the file carries is not on the schema")
	}
}

// TestSchemaNested is the same thing for a file of lists, maps and structs,
// which is where the schema stops being a list of columns and starts being a
// tree.
func TestSchemaNested(t *testing.T) {
	s, err := read(t, "nested.parquet").Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}

	person := dtype.Struct{Fields: []dtype.Field{
		{Name: "name", Type: dtype.String, Nullable: true},
		{Name: "age", Type: dtype.Int32, Nullable: true},
	}}
	want := []dtype.Field{
		{Name: "id", Type: dtype.Int32},
		{Name: "tags", Type: dtype.List{Elem: dtype.String}, Nullable: true},
		{Name: "counts", Type: dtype.List{Elem: dtype.Int32}, Nullable: true},
		{Name: "props", Type: dtype.Map{Key: dtype.String, Value: dtype.Int64}, Nullable: true},
		{Name: "point", Type: dtype.Struct{Fields: []dtype.Field{
			{Name: "x", Type: dtype.Float64},
			{Name: "y", Type: dtype.Float64, Nullable: true},
		}}, Nullable: true},
		{Name: "matrix", Type: dtype.List{Elem: dtype.List{Elem: dtype.Int32}}, Nullable: true},
		{Name: "people", Type: dtype.List{Elem: person}, Nullable: true},
	}
	compare(t, s, want)
}

// TestSchemaSmallFiles is the two files with nothing interesting in them, one
// with two row groups and one with no rows at all. A file with no rows still
// has a schema, and it is the whole point of writing one.
func TestSchemaSmallFiles(t *testing.T) {
	chunks, err := read(t, "chunks.parquet").Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}
	compare(t, chunks, []dtype.Field{
		{Name: "code", Type: dtype.String, Nullable: true},
		{Name: "n", Type: dtype.Int64, Nullable: true},
	})

	empty, err := read(t, "empty.parquet").Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}
	compare(t, empty, []dtype.Field{
		{Name: "id", Type: dtype.Int64, Nullable: true},
		{Name: "label", Type: dtype.String, Nullable: true},
	})
}

// compare checks a schema against the fields it should hold.
func compare(t *testing.T, got dtype.Schema, want []dtype.Field) {
	t.Helper()

	if len(got.Fields) != len(want) {
		t.Fatalf("the schema has %d fields, want %d:\n%s", len(got.Fields), len(want), got)
	}
	for i, w := range want {
		if !got.Fields[i].Equal(w) {
			t.Errorf("field %d is %s, want %s", i, got.Fields[i], w)
		}
	}
}

// TestSchemaTypes is every pair of a physical type and an annotation that means
// something, written by hand because no single writer produces all of them.
//
// The converted types are half of this table and they are not history: a file
// written by an old Hive or by Spark with the legacy flag set has them and
// nothing else, and reading such a file as raw integers and blobs is the bug
// this table exists to catch.
func TestSchemaTypes(t *testing.T) {
	int32Leaf := leaf("x", parquet.Required, parquet.Int32)
	int64Leaf := leaf("x", parquet.Required, parquet.Int64)
	bytesLeaf := leaf("x", parquet.Required, parquet.ByteArray)
	fixedLeaf := wide(leaf("x", parquet.Required, parquet.FixedLenByteArray), 4)

	// A decimal written the old way keeps its scale and precision on the node
	// itself, since the converted type had nowhere else to put them.
	oldDecimal := func(e parquet.SchemaElement, precision, scale int32) parquet.SchemaElement {
		e.Precision, e.Scale = precision, scale
		return converts(e, parquet.ConvertedDecimal)
	}
	integer := func(bits int8, signed bool) parquet.LogicalType {
		return parquet.LogicalType{Kind: parquet.IntegerLogical, BitWidth: bits, Signed: signed}
	}

	tests := []struct {
		name string
		node parquet.SchemaElement
		want dtype.DataType
	}{
		{"a boolean", leaf("x", parquet.Required, parquet.Boolean), dtype.Bool},
		{"a float", leaf("x", parquet.Required, parquet.Float), dtype.Float32},
		{"a double", leaf("x", parquet.Required, parquet.Double), dtype.Float64},
		{"the timestamp impala wrote", leaf("x", parquet.Required, parquet.Int96),
			dtype.Timestamp{Unit: dtype.Nanosecond}},

		{"a plain int32", int32Leaf, dtype.Int32},
		{"an int8", means(int32Leaf, integer(8, true)), dtype.Int8},
		{"an int16", means(int32Leaf, integer(16, true)), dtype.Int16},
		{"an int32", means(int32Leaf, integer(32, true)), dtype.Int32},
		{"a uint8", means(int32Leaf, integer(8, false)), dtype.Uint8},
		{"a uint16", means(int32Leaf, integer(16, false)), dtype.Uint16},
		{"a uint32", means(int32Leaf, integer(32, false)), dtype.Uint32},
		{"a date", means(int32Leaf, parquet.LogicalType{Kind: parquet.DateLogical}), dtype.Date32},
		{"a time of day", means(int32Leaf, parquet.LogicalType{
			Kind: parquet.TimeLogical, Unit: parquet.Millis,
		}), dtype.Time32{Unit: dtype.Millisecond}},
		{"a small decimal", means(int32Leaf, parquet.LogicalType{
			Kind: parquet.DecimalLogical, Precision: 9, Scale: 2,
		}), dtype.Decimal128{Precision: 9, Scale: 2}},

		{"an int8 the old way", converts(int32Leaf, parquet.ConvertedInt8), dtype.Int8},
		{"an int16 the old way", converts(int32Leaf, parquet.ConvertedInt16), dtype.Int16},
		{"an int32 the old way", converts(int32Leaf, parquet.ConvertedInt32), dtype.Int32},
		{"a uint8 the old way", converts(int32Leaf, parquet.ConvertedUint8), dtype.Uint8},
		{"a uint16 the old way", converts(int32Leaf, parquet.ConvertedUint16), dtype.Uint16},
		{"a uint32 the old way", converts(int32Leaf, parquet.ConvertedUint32), dtype.Uint32},
		{"a date the old way", converts(int32Leaf, parquet.ConvertedDate), dtype.Date32},
		{"a time of day the old way", converts(int32Leaf, parquet.ConvertedTimeMillis),
			dtype.Time32{Unit: dtype.Millisecond}},
		{"a small decimal the old way", oldDecimal(int32Leaf, 9, 2),
			dtype.Decimal128{Precision: 9, Scale: 2}},

		{"a plain int64", int64Leaf, dtype.Int64},
		{"an int64", means(int64Leaf, integer(64, true)), dtype.Int64},
		{"a uint64", means(int64Leaf, integer(64, false)), dtype.Uint64},
		{"an int32 written wide", means(int64Leaf, integer(32, true)), dtype.Int32},
		{"a time in micros", means(int64Leaf, parquet.LogicalType{
			Kind: parquet.TimeLogical, Unit: parquet.Micros,
		}), dtype.Time64{Unit: dtype.Microsecond}},
		{"a time in nanos", means(int64Leaf, parquet.LogicalType{
			Kind: parquet.TimeLogical, Unit: parquet.Nanos,
		}), dtype.Time64{Unit: dtype.Nanosecond}},
		{"an instant", means(int64Leaf, parquet.LogicalType{
			Kind: parquet.TimestampLogical, Unit: parquet.Millis, UTC: true,
		}), dtype.Timestamp{Unit: dtype.Millisecond, Zone: "UTC"}},
		{"a wall clock reading", means(int64Leaf, parquet.LogicalType{
			Kind: parquet.TimestampLogical, Unit: parquet.Micros,
		}), dtype.Timestamp{Unit: dtype.Microsecond}},
		{"an instant in nanos", means(int64Leaf, parquet.LogicalType{
			Kind: parquet.TimestampLogical, Unit: parquet.Nanos, UTC: true,
		}), dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "UTC"}},
		{"a decimal in an int64", means(int64Leaf, parquet.LogicalType{
			Kind: parquet.DecimalLogical, Precision: 18, Scale: 4,
		}), dtype.Decimal128{Precision: 18, Scale: 4}},

		{"an int64 the old way", converts(int64Leaf, parquet.ConvertedInt64), dtype.Int64},
		{"a uint64 the old way", converts(int64Leaf, parquet.ConvertedUint64), dtype.Uint64},
		{"a time in micros the old way", converts(int64Leaf, parquet.ConvertedTimeMicros),
			dtype.Time64{Unit: dtype.Microsecond}},
		{"an instant in millis the old way", converts(int64Leaf, parquet.ConvertedTimestampMillis),
			dtype.Timestamp{Unit: dtype.Millisecond, Zone: "UTC"}},
		{"an instant in micros the old way", converts(int64Leaf, parquet.ConvertedTimestampMicros),
			dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}},
		{"a decimal in an int64 the old way", oldDecimal(int64Leaf, 18, 4),
			dtype.Decimal128{Precision: 18, Scale: 4}},

		{"a blob", bytesLeaf, dtype.Binary},
		{"a string", means(bytesLeaf, parquet.LogicalType{Kind: parquet.StringLogical}), dtype.String},
		{"one of a fixed set of names", means(bytesLeaf,
			parquet.LogicalType{Kind: parquet.EnumLogical}), dtype.String},
		{"a json document", means(bytesLeaf,
			parquet.LogicalType{Kind: parquet.JSONLogical}), dtype.String},
		{"a bson document", means(bytesLeaf,
			parquet.LogicalType{Kind: parquet.BSONLogical}), dtype.Binary},
		{"a decimal too big for an int64", means(bytesLeaf, parquet.LogicalType{
			Kind: parquet.DecimalLogical, Precision: 30, Scale: 6,
		}), dtype.Decimal128{Precision: 30, Scale: 6}},
		{"a string the old way", converts(bytesLeaf, parquet.ConvertedUTF8), dtype.String},
		{"a name the old way", converts(bytesLeaf, parquet.ConvertedEnum), dtype.String},
		{"a json document the old way", converts(bytesLeaf, parquet.ConvertedJSON), dtype.String},
		{"a bson document the old way", converts(bytesLeaf, parquet.ConvertedBSON), dtype.Binary},
		{"a big decimal the old way", oldDecimal(bytesLeaf, 30, 6),
			dtype.Decimal128{Precision: 30, Scale: 6}},

		{"four bytes of anything", fixedLeaf, dtype.FixedSizeBinary{ByteWidth: 4}},
		{"a uuid", means(wide(fixedLeaf, 16), parquet.LogicalType{Kind: parquet.UUIDLogical}),
			dtype.FixedSizeBinary{ByteWidth: 16}},
		{"a decimal in sixteen bytes", means(wide(fixedLeaf, 16), parquet.LogicalType{
			Kind: parquet.DecimalLogical, Precision: 38, Scale: 10,
		}), dtype.Decimal128{Precision: 38, Scale: 10}},
		{"a decimal wider than that", means(wide(fixedLeaf, 32), parquet.LogicalType{
			Kind: parquet.DecimalLogical, Precision: 50, Scale: 10,
		}), dtype.Decimal256{Precision: 50, Scale: 10}},
		{"a decimal in bytes the old way", oldDecimal(wide(fixedLeaf, 16), 38, 10),
			dtype.Decimal128{Precision: 38, Scale: 10}},
		{"an interval", converts(wide(fixedLeaf, 12), parquet.ConvertedInterval),
			dtype.Interval{Unit: dtype.MonthDayNano}},

		{"a column of nothing", means(int32Leaf,
			parquet.LogicalType{Kind: parquet.UnknownLogical}), dtype.Null},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := schemaOf(1, tt.node).Schema()
			if err != nil {
				t.Fatalf("Schema: %v", err)
			}
			if got := s.Fields[0].Type; !dtype.Equal(got, tt.want) {
				t.Fatalf("read as %s, want %s", got, tt.want)
			}
		})
	}
}

// TestSchemaTypesRefused is a column whose annotation and physical type
// disagree, which is a file that says two things that cannot both be true. The
// annotation is not ignored: a column of dates that says it is a string was
// written by something that was wrong about it, and reading it as either one
// hands back values that mean nothing.
func TestSchemaTypesRefused(t *testing.T) {
	int32Leaf := leaf("x", parquet.Required, parquet.Int32)
	int64Leaf := leaf("x", parquet.Required, parquet.Int64)
	bytesLeaf := leaf("x", parquet.Required, parquet.ByteArray)
	fixedLeaf := wide(leaf("x", parquet.Required, parquet.FixedLenByteArray), 4)

	tests := []struct {
		name string
		node parquet.SchemaElement
		want error
	}{
		{"an int32 meaning a string", means(int32Leaf,
			parquet.LogicalType{Kind: parquet.StringLogical}), parquet.ErrFormat},
		{"an int32 converting as a string", converts(int32Leaf, parquet.ConvertedUTF8),
			parquet.ErrFormat},
		{"a time of day in an int32 in micros", means(int32Leaf, parquet.LogicalType{
			Kind: parquet.TimeLogical, Unit: parquet.Micros,
		}), parquet.ErrFormat},
		{"an int32 holding sixty four bits", means(int32Leaf,
			parquet.LogicalType{Kind: parquet.IntegerLogical, BitWidth: 64, Signed: true}),
			parquet.ErrFormat},
		{"an integer of a width nobody has", means(int32Leaf,
			parquet.LogicalType{Kind: parquet.IntegerLogical, BitWidth: 7, Signed: true}),
			parquet.ErrFormat},
		{"a time of day in an int64 in millis", means(int64Leaf, parquet.LogicalType{
			Kind: parquet.TimeLogical, Unit: parquet.Millis,
		}), parquet.ErrFormat},
		{"a time of day with no unit", means(int64Leaf,
			parquet.LogicalType{Kind: parquet.TimeLogical}), parquet.ErrFormat},
		{"a timestamp with no unit", means(int64Leaf,
			parquet.LogicalType{Kind: parquet.TimestampLogical}), parquet.ErrFormat},
		{"an int64 meaning a date", means(int64Leaf,
			parquet.LogicalType{Kind: parquet.DateLogical}), parquet.ErrFormat},
		{"an int64 converting as a date", converts(int64Leaf, parquet.ConvertedDate),
			parquet.ErrFormat},
		{"a blob meaning a date", means(bytesLeaf,
			parquet.LogicalType{Kind: parquet.DateLogical}), parquet.ErrFormat},
		{"a blob converting as a date", converts(bytesLeaf, parquet.ConvertedDate),
			parquet.ErrFormat},
		{"fixed bytes of no width", wide(fixedLeaf, 0), parquet.ErrFormat},
		{"fixed bytes of a negative width", wide(fixedLeaf, -8), parquet.ErrFormat},
		{"a uuid of the wrong size", means(fixedLeaf,
			parquet.LogicalType{Kind: parquet.UUIDLogical}), parquet.ErrFormat},
		{"an interval of the wrong size", converts(fixedLeaf, parquet.ConvertedInterval),
			parquet.ErrFormat},
		{"fixed bytes meaning a string", means(fixedLeaf,
			parquet.LogicalType{Kind: parquet.StringLogical}), parquet.ErrFormat},
		{"fixed bytes converting as a string", converts(fixedLeaf, parquet.ConvertedUTF8),
			parquet.ErrFormat},
		{"a half precision float", means(fixedLeaf,
			parquet.LogicalType{Kind: parquet.Float16Logical}), parquet.ErrUnsupported},
		{"a decimal of no digits", means(int32Leaf,
			parquet.LogicalType{Kind: parquet.DecimalLogical, Scale: 2}), parquet.ErrFormat},
		{"a decimal of more digits than there are", means(bytesLeaf,
			parquet.LogicalType{Kind: parquet.DecimalLogical, Precision: 80}),
			parquet.ErrUnsupported},
		// A node with no type and no fields is a group with nothing in it,
		// which the format forbids and no writer produces. It is refused rather
		// than read as a struct of no fields, because it is really a node
		// whose type went missing.
		{"a node that is neither a column nor a group",
			leaf("x", parquet.Required, parquet.NoType), parquet.ErrFormat},
		{"a physical type this reader has never heard of",
			leaf("x", parquet.Required, parquet.Type(9)), parquet.ErrUnsupported},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := schemaOf(1, tt.node)
			if s, err := m.Schema(); !errors.Is(err, tt.want) {
				t.Fatalf("read as %s, %v, want %v", s, err, tt.want)
			}
			// The columns come out of the same conversion, so a schema that is
			// refused has to be refused there too.
			if c, err := m.Columns(); !errors.Is(err, tt.want) {
				t.Fatalf("read as %v, %v, want %v", c, err, tt.want)
			}
		})
	}
}

// TestSchemaShapes is the nested shapes, including the ones the format only
// describes in its notes on what older writers did. A list is three levels deep
// in a file written this decade and two in one written before that, and a
// reader that only knows the new shape reads an old list as a struct holding a
// list, which is wrong in a way nothing downstream can recover from.
func TestSchemaShapes(t *testing.T) {
	tests := []struct {
		name  string
		nodes []parquet.SchemaElement
		want  dtype.Field
	}{
		{
			name: "a list, three levels deep",
			nodes: []parquet.SchemaElement{
				means(group("tags", parquet.Optional, 1),
					parquet.LogicalType{Kind: parquet.ListLogical}),
				group("list", parquet.Repeated, 1),
				leaf("element", parquet.Optional, parquet.Int32),
			},
			want: dtype.Field{Name: "tags", Type: dtype.List{Elem: dtype.Int32}, Nullable: true},
		},
		{
			name: "a list annotated the old way",
			nodes: []parquet.SchemaElement{
				converts(group("tags", parquet.Required, 1), parquet.ConvertedList),
				group("list", parquet.Repeated, 1),
				leaf("element", parquet.Required, parquet.Int32),
			},
			want: dtype.Field{Name: "tags", Type: dtype.List{Elem: dtype.Int32}},
		},
		{
			name: "a list whose elements are the repeated node",
			nodes: []parquet.SchemaElement{
				converts(group("tags", parquet.Optional, 1), parquet.ConvertedList),
				leaf("element", parquet.Repeated, parquet.Int32),
			},
			want: dtype.Field{Name: "tags", Type: dtype.List{Elem: dtype.Int32}, Nullable: true},
		},
		{
			name: "a list of structs written by a tool that called it array",
			nodes: []parquet.SchemaElement{
				converts(group("people", parquet.Optional, 1), parquet.ConvertedList),
				group("array", parquet.Repeated, 1),
				leaf("name", parquet.Optional, parquet.Int32),
			},
			want: dtype.Field{Name: "people", Nullable: true, Type: dtype.List{
				Elem: dtype.Struct{Fields: []dtype.Field{
					{Name: "name", Type: dtype.Int32, Nullable: true},
				}},
			}},
		},
		{
			name: "a list of structs written by a tool that added tuple to the name",
			nodes: []parquet.SchemaElement{
				converts(group("people", parquet.Optional, 1), parquet.ConvertedList),
				group("people_tuple", parquet.Repeated, 1),
				leaf("name", parquet.Optional, parquet.Int32),
			},
			want: dtype.Field{Name: "people", Nullable: true, Type: dtype.List{
				Elem: dtype.Struct{Fields: []dtype.Field{
					{Name: "name", Type: dtype.Int32, Nullable: true},
				}},
			}},
		},
		{
			name: "a list of structs known by its having two fields in there",
			nodes: []parquet.SchemaElement{
				converts(group("people", parquet.Optional, 1), parquet.ConvertedList),
				group("entry", parquet.Repeated, 2),
				leaf("name", parquet.Optional, parquet.Int32),
				leaf("age", parquet.Optional, parquet.Int32),
			},
			want: dtype.Field{Name: "people", Nullable: true, Type: dtype.List{
				Elem: dtype.Struct{Fields: []dtype.Field{
					{Name: "name", Type: dtype.Int32, Nullable: true},
					{Name: "age", Type: dtype.Int32, Nullable: true},
				}},
			}},
		},
		{
			name: "a list of lists",
			nodes: []parquet.SchemaElement{
				means(group("matrix", parquet.Optional, 1),
					parquet.LogicalType{Kind: parquet.ListLogical}),
				group("list", parquet.Repeated, 1),
				means(group("element", parquet.Optional, 1),
					parquet.LogicalType{Kind: parquet.ListLogical}),
				group("list", parquet.Repeated, 1),
				leaf("element", parquet.Optional, parquet.Int32),
			},
			want: dtype.Field{Name: "matrix", Nullable: true,
				Type: dtype.List{Elem: dtype.List{Elem: dtype.Int32}}},
		},
		{
			name: "a repeated field that says nothing about being a list",
			nodes: []parquet.SchemaElement{
				leaf("tags", parquet.Repeated, parquet.Int32),
			},
			want: dtype.Field{Name: "tags", Type: dtype.List{Elem: dtype.Int32}},
		},
		{
			name: "a repeated group that says nothing about being a list",
			nodes: []parquet.SchemaElement{
				group("people", parquet.Repeated, 1),
				leaf("name", parquet.Optional, parquet.Int32),
			},
			want: dtype.Field{Name: "people", Type: dtype.List{
				Elem: dtype.Struct{Fields: []dtype.Field{
					{Name: "name", Type: dtype.Int32, Nullable: true},
				}},
			}},
		},
		{
			name: "a map",
			nodes: []parquet.SchemaElement{
				means(group("props", parquet.Optional, 1),
					parquet.LogicalType{Kind: parquet.MapLogical}),
				group("key_value", parquet.Repeated, 2),
				means(leaf("key", parquet.Required, parquet.ByteArray),
					parquet.LogicalType{Kind: parquet.StringLogical}),
				leaf("value", parquet.Optional, parquet.Int64),
			},
			want: dtype.Field{Name: "props", Nullable: true,
				Type: dtype.Map{Key: dtype.String, Value: dtype.Int64}},
		},
		{
			name: "a map annotated the old way",
			nodes: []parquet.SchemaElement{
				converts(group("props", parquet.Required, 1), parquet.ConvertedMap),
				converts(group("key_value", parquet.Repeated, 2),
					parquet.ConvertedMapKeyValue),
				leaf("key", parquet.Required, parquet.Int32),
				leaf("value", parquet.Optional, parquet.Int64),
			},
			want: dtype.Field{Name: "props",
				Type: dtype.Map{Key: dtype.Int32, Value: dtype.Int64}},
		},
		{
			name: "a map annotated on the wrong node, which some writers did",
			nodes: []parquet.SchemaElement{
				converts(group("props", parquet.Optional, 1), parquet.ConvertedMapKeyValue),
				group("map", parquet.Repeated, 2),
				leaf("key", parquet.Required, parquet.Int32),
				leaf("value", parquet.Required, parquet.Int64),
			},
			want: dtype.Field{Name: "props", Nullable: true,
				Type: dtype.Map{Key: dtype.Int32, Value: dtype.Int64}},
		},
		{
			name: "a map of lists",
			nodes: []parquet.SchemaElement{
				means(group("props", parquet.Optional, 1),
					parquet.LogicalType{Kind: parquet.MapLogical}),
				group("key_value", parquet.Repeated, 2),
				leaf("key", parquet.Required, parquet.Int32),
				means(group("value", parquet.Optional, 1),
					parquet.LogicalType{Kind: parquet.ListLogical}),
				group("list", parquet.Repeated, 1),
				leaf("element", parquet.Optional, parquet.Int64),
			},
			want: dtype.Field{Name: "props", Nullable: true,
				Type: dtype.Map{Key: dtype.Int32, Value: dtype.List{Elem: dtype.Int64}}},
		},
		{
			name: "a plain group, which is a struct",
			nodes: []parquet.SchemaElement{
				group("point", parquet.Optional, 2),
				leaf("x", parquet.Required, parquet.Double),
				leaf("y", parquet.Optional, parquet.Double),
			},
			want: dtype.Field{Name: "point", Nullable: true, Type: dtype.Struct{Fields: []dtype.Field{
				{Name: "x", Type: dtype.Float64},
				{Name: "y", Type: dtype.Float64, Nullable: true},
			}}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s, err := schemaOf(1, tt.nodes...).Schema()
			if err != nil {
				t.Fatalf("Schema: %v", err)
			}
			if len(s.Fields) != 1 {
				t.Fatalf("read as %d fields, want one", len(s.Fields))
			}
			if !s.Fields[0].Equal(tt.want) {
				t.Fatalf("read as %s, want %s", s.Fields[0], tt.want)
			}
		})
	}
}

// TestSchemaShapesRefused is a group annotated as something it is not shaped
// like, which is the other half of trusting an annotation.
func TestSchemaShapesRefused(t *testing.T) {
	list := means(group("x", parquet.Optional, 1),
		parquet.LogicalType{Kind: parquet.ListLogical})
	mapping := means(group("x", parquet.Optional, 1),
		parquet.LogicalType{Kind: parquet.MapLogical})

	tests := []struct {
		name  string
		nodes []parquet.SchemaElement
		want  error
	}{
		{
			name: "a list with nothing in it",
			nodes: []parquet.SchemaElement{
				means(group("x", parquet.Optional, 0),
					parquet.LogicalType{Kind: parquet.ListLogical}),
			},
			want: parquet.ErrFormat,
		},
		{
			name: "a list holding two fields",
			nodes: []parquet.SchemaElement{
				means(group("x", parquet.Optional, 2),
					parquet.LogicalType{Kind: parquet.ListLogical}),
				leaf("a", parquet.Repeated, parquet.Int32),
				leaf("b", parquet.Repeated, parquet.Int32),
			},
			want: parquet.ErrFormat,
		},
		{
			name: "a list whose one field does not repeat",
			nodes: []parquet.SchemaElement{
				list,
				leaf("element", parquet.Optional, parquet.Int32),
			},
			want: parquet.ErrFormat,
		},
		{
			name: "a list of something that is not a type",
			nodes: []parquet.SchemaElement{
				list,
				group("list", parquet.Repeated, 1),
				leaf("element", parquet.Optional, parquet.NoType),
			},
			want: parquet.ErrFormat,
		},
		{
			name: "a two level list of something that is not a type",
			nodes: []parquet.SchemaElement{
				list,
				leaf("element", parquet.Repeated, parquet.NoType),
			},
			want: parquet.ErrFormat,
		},
		{
			name: "a map with nothing in it",
			nodes: []parquet.SchemaElement{
				means(group("x", parquet.Optional, 0),
					parquet.LogicalType{Kind: parquet.MapLogical}),
			},
			want: parquet.ErrFormat,
		},
		{
			name: "a map holding a second field",
			nodes: []parquet.SchemaElement{
				means(group("x", parquet.Optional, 2),
					parquet.LogicalType{Kind: parquet.MapLogical}),
				group("key_value", parquet.Repeated, 2),
				leaf("key", parquet.Required, parquet.Int32),
				leaf("value", parquet.Optional, parquet.Int32),
				leaf("extra", parquet.Optional, parquet.Int32),
			},
			want: parquet.ErrFormat,
		},
		{
			name: "a map whose entries do not repeat",
			nodes: []parquet.SchemaElement{
				mapping,
				group("key_value", parquet.Optional, 2),
				leaf("key", parquet.Required, parquet.Int32),
				leaf("value", parquet.Optional, parquet.Int32),
			},
			want: parquet.ErrFormat,
		},
		{
			name: "a map whose entries are a column",
			nodes: []parquet.SchemaElement{
				mapping,
				leaf("key_value", parquet.Repeated, parquet.Int32),
			},
			want: parquet.ErrFormat,
		},
		{
			name: "a map whose entries hold three fields",
			nodes: []parquet.SchemaElement{
				mapping,
				group("key_value", parquet.Repeated, 3),
				leaf("key", parquet.Required, parquet.Int32),
				leaf("value", parquet.Optional, parquet.Int32),
				leaf("extra", parquet.Optional, parquet.Int32),
			},
			want: parquet.ErrFormat,
		},
		{
			name: "a map with a key and no value, which is a set",
			nodes: []parquet.SchemaElement{
				mapping,
				group("key_value", parquet.Repeated, 1),
				leaf("key", parquet.Required, parquet.Int32),
			},
			want: parquet.ErrUnsupported,
		},
		{
			name: "a map whose key is not a type",
			nodes: []parquet.SchemaElement{
				mapping,
				group("key_value", parquet.Repeated, 2),
				leaf("key", parquet.Required, parquet.NoType),
				leaf("value", parquet.Optional, parquet.Int32),
			},
			want: parquet.ErrFormat,
		},
		{
			name: "a map whose value is not a type",
			nodes: []parquet.SchemaElement{
				mapping,
				group("key_value", parquet.Repeated, 2),
				leaf("key", parquet.Required, parquet.Int32),
				leaf("value", parquet.Optional, parquet.NoType),
			},
			want: parquet.ErrFormat,
		},
		{
			name: "a struct holding something that is not a type",
			nodes: []parquet.SchemaElement{
				group("point", parquet.Optional, 1),
				leaf("x", parquet.Required, parquet.NoType),
			},
			want: parquet.ErrFormat,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := schemaOf(1, tt.nodes...)
			if s, err := m.Schema(); !errors.Is(err, tt.want) {
				t.Fatalf("read as %s, %v, want %v", s, err, tt.want)
			}
			// A column under a group that cannot be converted is refused with
			// it, since the leaves are read by walking the same tree.
			if _, err := m.Columns(); err != nil && !errors.Is(err, parquet.ErrFormat) {
				t.Fatalf("the columns read as %v, want a format error or none", err)
			}
		})
	}
}

// TestColumns is the leaves of a flat file, which is one column per field and
// no levels worth speaking of. A required column has no levels at all, and an
// optional one has a definition level of one, meaning present.
func TestColumns(t *testing.T) {
	columns, err := read(t, "alltypes.parquet").Columns()
	if err != nil {
		t.Fatalf("Columns: %v", err)
	}
	if len(columns) != 15 {
		t.Fatalf("the file has %d columns, want 15", len(columns))
	}

	flag := columns[0]
	if flag.Name() != "flag" || !slices.Equal(flag.Path, []string{"flag"}) {
		t.Errorf("the first column is %q, want flag", flag.Name())
	}
	if flag.Element.Type != parquet.Boolean {
		t.Errorf("flag is written as a %s, want a boolean", flag.Element.Type)
	}
	if flag.MaxDefinition != 0 || flag.MaxRepetition != 0 {
		t.Errorf("flag is at levels %d and %d, want a required column at zero and zero",
			flag.MaxDefinition, flag.MaxRepetition)
	}

	for _, c := range columns[1:] {
		if c.MaxDefinition != 1 || c.MaxRepetition != 0 {
			t.Errorf("%s is at levels %d and %d, want an optional column at one and zero",
				c.Name(), c.MaxDefinition, c.MaxRepetition)
		}
	}

	// The type on a column is the type of one value, which for a flat file is
	// the same thing the schema says.
	s, err := read(t, "alltypes.parquet").Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}
	for i, c := range columns {
		if !dtype.Equal(c.Type, s.Fields[i].Type) {
			t.Errorf("%s is a %s as a column and a %s in the schema", c.Name(), c.Type, s.Fields[i].Type)
		}
	}
}

// TestColumnsNested is the levels, which are the whole reason a nested file can
// be read back at all.
//
// A value is missing if its definition level is below the maximum, and the
// maximum counts the optional and repeated nodes on the way down. The
// repetition level says which list a value continues, and it counts the
// repeated nodes only, so a list of lists has two and a flat column has none.
func TestColumnsNested(t *testing.T) {
	columns, err := read(t, "nested.parquet").Columns()
	if err != nil {
		t.Fatalf("Columns: %v", err)
	}

	tests := []struct {
		path       string
		definition int
		repetition int
		typ        dtype.DataType
	}{
		{"id", 0, 0, dtype.Int32},
		{"tags.list.element", 3, 1, dtype.String},
		{"counts.list.element", 2, 1, dtype.Int32},
		{"props.key_value.key", 2, 1, dtype.String},
		{"props.key_value.value", 3, 1, dtype.Int64},
		{"point.x", 1, 0, dtype.Float64},
		{"point.y", 2, 0, dtype.Float64},
		{"matrix.list.element.list.element", 5, 2, dtype.Int32},
		{"people.list.element.name", 4, 1, dtype.String},
		{"people.list.element.age", 4, 1, dtype.Int32},
	}

	if len(columns) != len(tests) {
		t.Fatalf("the file has %d columns, want %d", len(columns), len(tests))
	}
	for i, want := range tests {
		got := columns[i]
		if got.Name() != want.path {
			t.Fatalf("column %d is %q, want %q", i, got.Name(), want.path)
		}
		if got.MaxDefinition != want.definition || got.MaxRepetition != want.repetition {
			t.Errorf("%s is at levels %d and %d, want %d and %d",
				want.path, got.MaxDefinition, got.MaxRepetition, want.definition, want.repetition)
		}
		if !dtype.Equal(got.Type, want.typ) {
			t.Errorf("%s holds %s, want %s", want.path, got.Type, want.typ)
		}
	}

	// The columns are in the order the row groups hold their chunks, which is
	// what makes a chunk findable at all.
	for i, c := range read(t, "nested.parquet").RowGroups[0].Columns {
		if got := strings.Join(c.Meta.Path, "."); got != columns[i].Name() {
			t.Errorf("chunk %d is %q and column %d is %q", i, got, i, columns[i].Name())
		}
	}
}

// TestTreeRefused is a schema whose child counts do not describe a tree.
//
// The counts are the only thing holding the shape, so a file that gets them
// wrong has no schema rather than a broken one, and every one of these is a
// number a hostile file would rather this reader believed.
func TestTreeRefused(t *testing.T) {
	deep := []parquet.SchemaElement{}
	for range 70 {
		deep = append(deep, group("down", parquet.Optional, 1))
	}
	deep = append(deep, leaf("bottom", parquet.Optional, parquet.Int32))

	tests := []struct {
		name  string
		nodes []parquet.SchemaElement
	}{
		{"a file with no schema at all", nil},
		{"a root claiming more fields than there are", []parquet.SchemaElement{
			{Name: "schema", NumChildren: 4},
			leaf("x", parquet.Optional, parquet.Int32),
		}},
		{"a root claiming fewer fields than there are", []parquet.SchemaElement{
			{Name: "schema", NumChildren: 1},
			leaf("x", parquet.Optional, parquet.Int32),
			leaf("y", parquet.Optional, parquet.Int32),
		}},
		{"a group claiming a field that belongs to nobody", []parquet.SchemaElement{
			{Name: "schema", NumChildren: 2},
			group("a", parquet.Optional, 1),
			leaf("x", parquet.Optional, parquet.Int32),
		}},
		{"a count that is negative", []parquet.SchemaElement{
			{Name: "schema", NumChildren: 1},
			group("a", parquet.Optional, -1),
			leaf("x", parquet.Optional, parquet.Int32),
		}},
		{"a schema nested deeper than anything means to be", append(
			[]parquet.SchemaElement{{Name: "schema", NumChildren: 1}}, deep...)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &parquet.Metadata{Nodes: tt.nodes}
			if _, err := m.Tree(); !errors.Is(err, parquet.ErrFormat) {
				t.Fatalf("Tree: %v, want a format error", err)
			}
			if _, err := m.Schema(); !errors.Is(err, parquet.ErrFormat) {
				t.Fatalf("Schema: %v, want a format error", err)
			}
			if _, err := m.Columns(); !errors.Is(err, parquet.ErrFormat) {
				t.Fatalf("Columns: %v, want a format error", err)
			}
		})
	}
}

// TestTree walks the tree a file describes, which is the shape the flat list
// leaves implicit.
func TestTree(t *testing.T) {
	root, err := read(t, "nested.parquet").Tree()
	if err != nil {
		t.Fatalf("Tree: %v", err)
	}
	if root.Leaf() {
		t.Fatal("the root is a column")
	}
	if len(root.Children) != 7 {
		t.Fatalf("the root has %d fields, want 7", len(root.Children))
	}

	// Down the left hand side of the deepest column in the file, which is a
	// list of lists and is five nodes below the root.
	names := []string{}
	for n := &root.Children[5]; ; n = &n.Children[0] {
		names = append(names, n.Name)
		if n.Leaf() {
			break
		}
	}
	want := []string{"matrix", "list", "element", "list", "element"}
	if !slices.Equal(names, want) {
		t.Errorf("the path down to the innermost value is %v, want %v", names, want)
	}

	// Every node of the tree is a node of the list, and nothing was invented on
	// the way through.
	m := read(t, "nested.parquet")
	count := 0
	var walk func(n *parquet.Node)
	walk = func(n *parquet.Node) {
		count++
		for i := range n.Children {
			walk(&n.Children[i])
		}
	}
	walk(&root)
	if count != len(m.Nodes) {
		t.Errorf("the tree has %d nodes and the file has %d", count, len(m.Nodes))
	}
}

func BenchmarkSchema(b *testing.B) {
	for _, name := range []string{"alltypes.parquet", "nested.parquet"} {
		m := read(b, name)
		b.Run(strings.TrimSuffix(name, ".parquet"), func(b *testing.B) {
			for b.Loop() {
				if _, err := m.Schema(); err != nil {
					b.Fatalf("Schema: %v", err)
				}
			}
		})
	}
}

func BenchmarkColumns(b *testing.B) {
	for _, name := range []string{"alltypes.parquet", "nested.parquet"} {
		m := read(b, name)
		b.Run(strings.TrimSuffix(name, ".parquet"), func(b *testing.B) {
			for b.Loop() {
				if _, err := m.Columns(); err != nil {
					b.Fatalf("Columns: %v", err)
				}
			}
		})
	}
}
