package ipc_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

// schemaCases are the schemas that go out and come back. The want field is
// what comes back when it is not what went out, which happens for the text
// layouts kuma does not keep: a large string is written as a large string and
// read as a string, the same collapse the format strings make, since there is
// one kuma layout for text and four Arrow ones.
var schemaCases = []struct {
	name   string
	schema dtype.Schema
	want   dtype.Schema
}{
	{name: "empty"},
	{
		name: "primitives",
		schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "null", Type: dtype.Null, Nullable: true},
			{Name: "bool", Type: dtype.Bool},
			{Name: "int8", Type: dtype.Int8},
			{Name: "int16", Type: dtype.Int16},
			{Name: "int32", Type: dtype.Int32},
			{Name: "int64", Type: dtype.Int64},
			{Name: "uint8", Type: dtype.Uint8},
			{Name: "uint16", Type: dtype.Uint16},
			{Name: "uint32", Type: dtype.Uint32},
			{Name: "uint64", Type: dtype.Uint64},
			{Name: "float32", Type: dtype.Float32},
			{Name: "float64", Type: dtype.Float64},
		}},
	},
	{
		name: "text",
		schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "string", Type: dtype.String},
			{Name: "binary", Type: dtype.Binary},
			{Name: "large string", Type: dtype.LargeString},
			{Name: "large binary", Type: dtype.LargeBinary},
			{Name: "fixed", Type: dtype.FixedSizeBinary{ByteWidth: 16}},
		}},
		want: dtype.Schema{Fields: []dtype.Field{
			{Name: "string", Type: dtype.String},
			{Name: "binary", Type: dtype.Binary},
			{Name: "large string", Type: dtype.String},
			{Name: "large binary", Type: dtype.Binary},
			{Name: "fixed", Type: dtype.FixedSizeBinary{ByteWidth: 16}},
		}},
	},
	{
		name: "temporal",
		schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "date32", Type: dtype.Date32},
			{Name: "date64", Type: dtype.Date64},
			{Name: "time32 s", Type: dtype.Time32{Unit: dtype.Second}},
			{Name: "time32 ms", Type: dtype.Time32{Unit: dtype.Millisecond}},
			{Name: "time64 us", Type: dtype.Time64{Unit: dtype.Microsecond}},
			{Name: "time64 ns", Type: dtype.Time64{Unit: dtype.Nanosecond}},
			{Name: "naive", Type: dtype.Timestamp{Unit: dtype.Microsecond}},
			{Name: "utc", Type: dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "UTC"}},
			{Name: "zoned", Type: dtype.Timestamp{Unit: dtype.Second, Zone: "Europe/London"}},
			{Name: "offset", Type: dtype.Timestamp{Unit: dtype.Millisecond, Zone: "+01:00"}},
			{Name: "duration s", Type: dtype.Duration{Unit: dtype.Second}},
			{Name: "duration ms", Type: dtype.Duration{Unit: dtype.Millisecond}},
			{Name: "duration us", Type: dtype.Duration{Unit: dtype.Microsecond}},
			{Name: "duration ns", Type: dtype.Duration{Unit: dtype.Nanosecond}},
			{Name: "year month", Type: dtype.Interval{Unit: dtype.YearMonth}},
			{Name: "day time", Type: dtype.Interval{Unit: dtype.DayTime}},
			{Name: "month day nano", Type: dtype.Interval{Unit: dtype.MonthDayNano}},
		}},
	},
	{
		name: "decimals",
		schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "money", Type: dtype.Decimal128{Precision: 18, Scale: 2}},
			{Name: "no scale", Type: dtype.Decimal128{Precision: 38, Scale: 0}},
			{Name: "wide", Type: dtype.Decimal256{Precision: 60, Scale: 10}},
		}},
	},
	{
		name: "nested",
		schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "list", Type: dtype.List{Elem: dtype.Int32}, Nullable: true},
			{Name: "large list", Type: dtype.LargeList{Elem: dtype.String}},
			{Name: "fixed list", Type: dtype.FixedSizeList{Elem: dtype.Float64, Len: 3}},
			{Name: "list of list", Type: dtype.List{Elem: dtype.List{Elem: dtype.Int64}}},
			{Name: "map", Type: dtype.Map{Key: dtype.String, Value: dtype.Int64}},
			{Name: "struct", Type: dtype.Struct{Fields: []dtype.Field{
				{Name: "x", Type: dtype.Float64},
				{Name: "y", Type: dtype.Float64, Nullable: true},
				{Name: "inner", Type: dtype.Struct{Fields: []dtype.Field{
					{Name: "deep", Type: dtype.Int8, Nullable: true},
				}}},
			}}},
		}},
		want: dtype.Schema{Fields: []dtype.Field{
			{Name: "list", Type: dtype.List{Elem: dtype.Int32}, Nullable: true},
			{Name: "large list", Type: dtype.LargeList{Elem: dtype.String}},
			{Name: "fixed list", Type: dtype.FixedSizeList{Elem: dtype.Float64, Len: 3}},
			{Name: "list of list", Type: dtype.List{Elem: dtype.List{Elem: dtype.Int64}}},
			{Name: "map", Type: dtype.Map{Key: dtype.String, Value: dtype.Int64}},
			{Name: "struct", Type: dtype.Struct{Fields: []dtype.Field{
				{Name: "x", Type: dtype.Float64},
				{Name: "y", Type: dtype.Float64, Nullable: true},
				{Name: "inner", Type: dtype.Struct{Fields: []dtype.Field{
					{Name: "deep", Type: dtype.Int8, Nullable: true},
				}}},
			}}},
		}},
	},
	{
		name: "dictionary",
		schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "small", Type: dtype.Dictionary{Index: dtype.Int8, Value: dtype.String}},
			{Name: "wide", Type: dtype.Dictionary{Index: dtype.Int32, Value: dtype.String}},
			{Name: "unsigned", Type: dtype.Dictionary{Index: dtype.Uint16, Value: dtype.Binary}},
		}},
	},
	{
		name: "metadata",
		schema: dtype.Schema{
			Fields: []dtype.Field{
				{Name: "one", Type: dtype.Int64, Metadata: dtype.Metadata{
					{Key: "unit", Value: "meters"},
					{Key: "unit", Value: "the same key twice"},
					{Key: "empty", Value: ""},
				}},
				{Name: "two", Type: dtype.Float64},
			},
			Metadata: dtype.Metadata{
				{Key: "written by", Value: "kuma"},
				{Key: "unicode", Value: "\u30af\u30de"},
			},
		},
	},
	{
		name: "names",
		schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "", Type: dtype.Int64},
			{Name: "with space", Type: dtype.Int64},
			{Name: "\u30af\u30de", Type: dtype.Int64},
			{Name: strings.Repeat("long", 100), Type: dtype.Int64},
		}},
	},
}

func TestSchemaRoundTrip(t *testing.T) {
	for _, c := range schemaCases {
		t.Run(c.name, func(t *testing.T) {
			b, err := ipc.EncodeSchema(c.schema)
			if err != nil {
				t.Fatalf("EncodeSchema: %v", err)
			}
			if len(b)%8 != 0 {
				t.Errorf("the message is %d bytes, which is not a multiple of eight", len(b))
			}

			got, err := ipc.DecodeSchema(b)
			if err != nil {
				t.Fatalf("DecodeSchema: %v", err)
			}
			want := c.want
			if want.Fields == nil && want.Metadata == nil {
				want = c.schema
			}
			if !got.Equal(want) {
				t.Errorf("came back as\n%v\nwant\n%v", got, want)
			}

			// What comes back is what goes out again. The one type that changes
			// changes on the first trip and not on any after it, so this is
			// what says the collapse is a mapping rather than a slide.
			again, err := ipc.EncodeSchema(got)
			if err != nil {
				t.Fatalf("EncodeSchema of what came back: %v", err)
			}
			twice, err := ipc.DecodeSchema(again)
			if err != nil {
				t.Fatalf("DecodeSchema of what came back: %v", err)
			}
			if !twice.Equal(got) {
				t.Errorf("the second trip gave\n%v\nwant\n%v", twice, got)
			}
		})
	}
}

// TestEncodeSchemaError covers the types that cannot be written down. Every one
// of them is a type that was built by hand with a parameter that is not a real
// type, since everything the constructors produce can be written.
func TestEncodeSchemaError(t *testing.T) {
	cases := []struct {
		name string
		typ  dtype.DataType
		want error
	}{
		{"nil", nil, ipc.ErrType},
		{"time32 in nanoseconds", dtype.Time32{Unit: dtype.Nanosecond}, ipc.ErrType},
		{"time64 in seconds", dtype.Time64{Unit: dtype.Second}, ipc.ErrType},
		{"timestamp with no unit", dtype.Timestamp{Unit: dtype.TimeUnit(9)}, ipc.ErrType},
		{"duration with no unit", dtype.Duration{Unit: dtype.TimeUnit(9)}, ipc.ErrType},
		{"interval with no unit", dtype.Interval{Unit: dtype.IntervalUnit(9)}, ipc.ErrType},
		{"negative width", dtype.FixedSizeBinary{ByteWidth: -1}, ipc.ErrType},
		{"negative list", dtype.FixedSizeList{Elem: dtype.Int64, Len: -1}, ipc.ErrType},
		{"dictionary of strings", dtype.Dictionary{Index: dtype.String, Value: dtype.String}, ipc.ErrType},
		{"list of nothing", dtype.List{Elem: nil}, ipc.ErrType},
		{"struct of nothing", dtype.Struct{Fields: []dtype.Field{{Name: "x"}}}, ipc.ErrType},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := dtype.Schema{Fields: []dtype.Field{{Name: "column", Type: c.typ}}}
			_, err := ipc.EncodeSchema(s)
			if !errors.Is(err, c.want) {
				t.Fatalf("EncodeSchema: %v, want %v", err, c.want)
			}
		})
	}
}

// TestDecodeSchemaError covers the bytes that are not a schema. The message
// itself is the interesting half, since a reader of it is holding a file
// somebody else wrote.
func TestDecodeSchemaError(t *testing.T) {
	good, err := ipc.EncodeSchema(dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64},
	}})
	if err != nil {
		t.Fatalf("EncodeSchema: %v", err)
	}

	cases := []struct {
		name string
		msg  []byte
	}{
		{"nothing", nil},
		{"a prefix and no message", good[:8]},
		{"half a prefix", good[:5]},
		{"the end of a stream", []byte{0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0}},
		{"a message that stops halfway", good[:len(good)-8]},
		{"a length of minus one", []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0}},
		{"a prefix and rubbish", append(append([]byte(nil), good[:8]...),
			make([]byte, len(good)-8)...)},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := ipc.DecodeSchema(c.msg); !errors.Is(err, ipc.ErrMessage) {
				t.Fatalf("DecodeSchema: %v, want ErrMessage", err)
			}
		})
	}
}

// TestDecodeSchemaCorrupt changes one byte at a time and checks that nothing
// panics and that what does come back is a schema this package would write.
//
// The check on the way out matters as much as the one on the way in. A message
// with a byte changed that reads as a schema of nine hundred columns of
// something is a schema that will be handed to a builder, and a decoder that
// lets one through has not really refused it.
func TestDecodeSchemaCorrupt(t *testing.T) {
	good, err := ipc.EncodeSchema(dtype.Schema{
		Fields: []dtype.Field{
			{Name: "id", Type: dtype.Int64},
			{Name: "name", Type: dtype.String, Nullable: true},
			{Name: "at", Type: dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}},
			{Name: "tags", Type: dtype.List{Elem: dtype.String}},
		},
		Metadata: dtype.Metadata{{Key: "k", Value: "v"}},
	})
	if err != nil {
		t.Fatalf("EncodeSchema: %v", err)
	}

	for i := range good {
		for _, b := range []byte{0x00, 0x01, 0x7F, 0xFF} {
			bad := append([]byte(nil), good...)
			bad[i] = b

			s, err := ipc.DecodeSchema(bad)
			if err != nil {
				continue
			}
			if _, err := ipc.EncodeSchema(s); err != nil {
				t.Fatalf("byte %d set to %#x read as %v, which cannot be written back: %v",
					i, b, s, err)
			}
		}
	}
}

// TestDecodeSchemaTrailing checks that a schema at the front of a stream reads
// without being cut out of it first, which is what the stream reader will do
// with it.
func TestDecodeSchemaTrailing(t *testing.T) {
	want := dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64}}}
	b, err := ipc.EncodeSchema(want)
	if err != nil {
		t.Fatalf("EncodeSchema: %v", err)
	}

	got, err := ipc.DecodeSchema(append(b, make([]byte, 64)...))
	if err != nil {
		t.Fatalf("DecodeSchema: %v", err)
	}
	if !got.Equal(want) {
		t.Errorf("came back as %v, want %v", got, want)
	}
}

// TestSchemaShared checks that a schema of like columns does not carry one
// copy of the same description per column.
//
// A column of int64 costs a name, a field table and an offset. What it must
// not cost is another description of what int64 is, or another vtable saying
// what a field table looks like, and at thirty something bytes a column it
// plainly does not. A thousand column table is a real thing, and a reader of
// one pays this on every batch.
func TestSchemaShared(t *testing.T) {
	const columns = 1000
	const budget = 48

	s := dtype.Schema{Fields: make([]dtype.Field, columns)}
	for i := range s.Fields {
		s.Fields[i] = dtype.Field{Name: "c" + string(rune('0'+i%10)), Type: dtype.Int64}
	}

	b, err := ipc.EncodeSchema(s)
	if err != nil {
		t.Fatalf("EncodeSchema: %v", err)
	}
	if len(b) > columns*budget {
		t.Errorf("%d columns of int64 are %d bytes, which is %d a column and more than %d",
			columns, len(b), len(b)/columns, budget)
	}

	got, err := ipc.DecodeSchema(b)
	if err != nil {
		t.Fatalf("DecodeSchema: %v", err)
	}
	if !got.Equal(s) {
		t.Error("a schema of shared type tables did not come back the way it went in")
	}
}

func BenchmarkEncodeSchema(b *testing.B) {
	s := benchSchema(100)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := ipc.EncodeSchema(s); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeSchema(b *testing.B) {
	msg, err := ipc.EncodeSchema(benchSchema(100))
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := ipc.DecodeSchema(msg); err != nil {
			b.Fatal(err)
		}
	}
}

// benchSchema is a wide schema of the types a real table holds, which is what
// the cost of reading a schema is paid on: a file with one column is not where
// this shows up.
func benchSchema(n int) dtype.Schema {
	types := []dtype.DataType{
		dtype.Int64,
		dtype.Float64,
		dtype.String,
		dtype.Bool,
		dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"},
		dtype.Decimal128{Precision: 18, Scale: 2},
	}
	s := dtype.Schema{Fields: make([]dtype.Field, n)}
	for i := range s.Fields {
		s.Fields[i] = dtype.Field{
			Name:     "column_" + string(rune('a'+i%26)) + string(rune('0'+i%10)),
			Type:     types[i%len(types)],
			Nullable: i%2 == 0,
		}
	}
	return s
}
