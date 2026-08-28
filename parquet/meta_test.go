package parquet

import (
	"errors"
	"testing"
)

// logical writes the logical type union with one member set and reads it back.
//
// The union is a struct with one field, and which field it is says what the
// type is. Only a handful of the members turn up in the files in testdata,
// because pyarrow writes what an Arrow schema can hold, so the rest are written
// here and read the same way a file's would be.
func logical(t *testing.T, id int16, f func(*builder)) LogicalType {
	t.Helper()

	w := &builder{}
	w.structure(func() {
		w.field(id, thriftStruct)
		w.structure(func() { f(w) })
	})

	var l LogicalType
	if err := l.read(w.reader(), thriftStruct); err != nil {
		t.Fatalf("reading the union with field %d set: %v", id, err)
	}
	return l
}

// unit writes the time unit union, which is three empty structs and a choice of
// which one is there.
func unit(w *builder, id int16) {
	w.field(2, thriftStruct)
	w.structure(func() {
		w.field(id, thriftStruct)
		w.structure(func() {})
	})
}

// TestLogicalPlain covers the members that are an empty struct and whose only
// content is which field they arrived in.
func TestLogicalPlain(t *testing.T) {
	tests := []struct {
		id   int16
		want LogicalKind
	}{
		{1, StringLogical},
		{2, MapLogical},
		{3, ListLogical},
		{4, EnumLogical},
		{6, DateLogical},
		{11, UnknownLogical},
		{12, JSONLogical},
		{13, BSONLogical},
		{14, UUIDLogical},
		{15, Float16Logical},
	}

	for _, tt := range tests {
		t.Run(tt.want.String(), func(t *testing.T) {
			if got := logical(t, tt.id, func(*builder) {}); got.Kind != tt.want {
				t.Fatalf("field %d read as %s, want %s", tt.id, got.Kind, tt.want)
			}
		})
	}
}

func TestLogicalDecimal(t *testing.T) {
	got := logical(t, 5, func(w *builder) {
		w.field(1, thriftInt32).varint(4)
		w.field(2, thriftInt32).varint(18)
	})
	if got.Kind != DecimalLogical || got.Scale != 4 || got.Precision != 18 {
		t.Fatalf("read as %s of scale %d and precision %d, want a decimal of 4 and 18",
			got.Kind, got.Scale, got.Precision)
	}
}

func TestLogicalTime(t *testing.T) {
	tests := []struct {
		name string
		id   int16
		utc  bool
		unit int16
		want TimeUnit
		kind LogicalKind
	}{
		{"a time in millis", 7, false, 1, Millis, TimeLogical},
		{"a time in micros", 7, false, 2, Micros, TimeLogical},
		{"a time in nanos", 7, true, 3, Nanos, TimeLogical},
		{"a timestamp in millis", 8, true, 1, Millis, TimestampLogical},
		{"a timestamp in micros", 8, false, 2, Micros, TimestampLogical},
		{"a timestamp in nanos", 8, true, 3, Nanos, TimestampLogical},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := logical(t, tt.id, func(w *builder) {
				if tt.utc {
					w.field(1, thriftTrue)
				} else {
					w.field(1, thriftFalse)
				}
				unit(w, tt.unit)
			})
			if got.Kind != tt.kind || got.Unit != tt.want || got.UTC != tt.utc {
				t.Fatalf("read as a %s in %s, utc %v, want a %s in %s, utc %v",
					got.Kind, got.Unit, got.UTC, tt.kind, tt.want, tt.utc)
			}
		})
	}

	// A unit union with a member this package has never heard of, which leaves
	// the unit unset rather than refusing the column.
	got := logical(t, 8, func(w *builder) { unit(w, 9) })
	if got.Kind != TimestampLogical || got.Unit != NoUnit {
		t.Fatalf("an unknown unit read as %s in %s, want a timestamp with no unit", got.Kind, got.Unit)
	}
}

func TestLogicalInteger(t *testing.T) {
	for _, width := range []int8{8, 16, 32, 64} {
		for _, signed := range []bool{true, false} {
			got := logical(t, 10, func(w *builder) {
				w.field(1, thriftInt8).raw(byte(width))
				if signed {
					w.field(2, thriftTrue)
				} else {
					w.field(2, thriftFalse)
				}
			})
			if got.Kind != IntegerLogical || got.BitWidth != width || got.Signed != signed {
				t.Errorf("read as a %s of %d bits, signed %v, want an integer of %d bits, signed %v",
					got.Kind, got.BitWidth, got.Signed, width, signed)
			}
		}
	}
}

// TestLogicalUnknownMember is a union member this package has never heard of,
// which is what a file written by a newer parquet holds. It has to be read past
// rather than refused, and it leaves the type unset, since a column whose
// meaning is unknown is still a column whose values are there.
func TestLogicalUnknownMember(t *testing.T) {
	// Nine is the one the format left out, and a hundred is one nobody has
	// defined yet.
	for _, id := range []int16{9, 100} {
		got := logical(t, id, func(w *builder) {
			w.field(1, thriftInt32).varint(1)
		})
		if got.Kind != NoLogical {
			t.Errorf("the union with field %d set read as %s, want nothing", id, got.Kind)
		}
	}

	// The union written as something other than a struct, which is a footer
	// that disagrees with itself. It is read past for the same reason.
	var l LogicalType
	w := (&builder{}).varint(3)
	if err := l.read(w.reader(), thriftInt32); err != nil {
		t.Fatalf("a union written as an int32: %v", err)
	}
	if l.Kind != NoLogical {
		t.Errorf("a union written as an int32 read as %s, want nothing", l.Kind)
	}
}

// TestLogicalParametersRefused checks that a member whose parameters are not
// what they say is refused rather than read as something else.
func TestLogicalParametersRefused(t *testing.T) {
	tests := []struct {
		name  string
		id    int16
		write func(*builder)
	}{
		{"a decimal whose scale is a string", 5, func(w *builder) {
			w.field(1, thriftBinary).binary("four")
		}},
		{"a decimal whose precision runs off the end", 5, func(w *builder) {
			w.field(2, thriftInt32).raw(0x80)
		}},
		{"a timestamp whose utc flag is an int", 8, func(w *builder) {
			w.field(1, thriftInt32).varint(1)
		}},
		{"an integer whose width is a string", 10, func(w *builder) {
			w.field(1, thriftBinary).binary("8")
		}},
		{"an integer of more bits than a byte holds", 10, func(w *builder) {
			w.field(1, thriftInt32).varint(1000)
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			w := &builder{}
			w.structure(func() {
				w.field(tt.id, thriftStruct)
				w.structure(func() { tt.write(w) })
			})

			var l LogicalType
			if err := l.read(w.reader(), thriftStruct); !errors.Is(err, ErrFormat) {
				t.Fatalf("read as %+v, %v, want a format error", l, err)
			}
		})
	}
}

// TestLogicalUnknownParameters is a member this package knows about carrying a
// parameter it does not. Same rule as an unknown member: read past it, keep
// what was understood, and give back a type the rest of the reader can use.
func TestLogicalUnknownParameters(t *testing.T) {
	decimal := logical(t, 5, func(w *builder) {
		w.field(1, thriftInt32).varint(2)
		w.field(2, thriftInt32).varint(9)
		w.field(6, thriftBinary).binary("something new")
	})
	if decimal.Scale != 2 || decimal.Precision != 9 {
		t.Errorf("the decimal is scale %d and precision %d, want 2 and 9",
			decimal.Scale, decimal.Precision)
	}

	stamp := logical(t, 8, func(w *builder) {
		w.field(1, thriftTrue)
		unit(w, 2)
		w.field(6, thriftInt32).varint(1)
	})
	if !stamp.UTC || stamp.Unit != Micros {
		t.Errorf("the timestamp is in %s, utc %v, want micros and utc", stamp.Unit, stamp.UTC)
	}

	number := logical(t, 10, func(w *builder) {
		w.field(1, thriftInt8).raw(16)
		w.field(2, thriftTrue)
		w.field(6, thriftList).list(1, thriftInt32).varint(1)
	})
	if number.BitWidth != 16 || !number.Signed {
		t.Errorf("the integer is %d bits, signed %v, want 16 and signed", number.BitWidth, number.Signed)
	}

	// The unit written as something other than a struct, which is read past the
	// same way the union itself is.
	odd := logical(t, 7, func(w *builder) {
		w.field(2, thriftBinary).binary("micros")
	})
	if odd.Kind != TimeLogical || odd.Unit != NoUnit {
		t.Errorf("a time whose unit is a string read as %s in %s, want a time with no unit",
			odd.Kind, odd.Unit)
	}
}

// TestLogicalNotStructs is every member with parameters written as something
// that is not a struct. There is nothing to read out of one, and the member is
// still which field it arrived in, so what comes back is the kind on its own.
func TestLogicalNotStructs(t *testing.T) {
	tests := []struct {
		id   int16
		want LogicalKind
	}{
		{5, DecimalLogical},
		{7, TimeLogical},
		{8, TimestampLogical},
		{10, IntegerLogical},
	}

	for _, tt := range tests {
		t.Run(tt.want.String(), func(t *testing.T) {
			w := &builder{}
			w.structure(func() { w.field(tt.id, thriftInt32).varint(1) })

			var l LogicalType
			if err := l.read(w.reader(), thriftStruct); err != nil {
				t.Fatalf("read: %v", err)
			}
			if l.Kind != tt.want {
				t.Fatalf("read as %s, want %s", l.Kind, tt.want)
			}
			if l.Scale != 0 || l.Precision != 0 || l.Unit != NoUnit || l.BitWidth != 0 {
				t.Fatalf("read as %+v, want no parameters", l)
			}
		})
	}
}

// TestSchemaElementDefaults checks that a schema element with nothing in it
// comes back saying so rather than saying it is a required boolean.
//
// The zero of every one of these is a value the format uses, so the reader sets
// the absent ones itself. A group node has no physical type, and reading it as
// a boolean is how a nested schema turns into a broken flat one.
func TestSchemaElementDefaults(t *testing.T) {
	var e SchemaElement
	if err := e.read((&builder{}).structure(func() {}).reader()); err != nil {
		t.Fatalf("read: %v", err)
	}
	if e.Type != NoType || e.Repetition != NoRepetition || e.Converted != NoConverted {
		t.Fatalf("an empty element is a %s %s converting as %s, want none of the three",
			e.Repetition, e.Type, e.Converted)
	}
	if e.Logical.Kind != NoLogical {
		t.Errorf("an empty element means %s, want nothing", e.Logical.Kind)
	}

	// Read again into the same element, which is what a list of them does, so a
	// field the second one leaves out has to come back absent rather than
	// holding what the first one said.
	w := &builder{}
	w.structure(func() {
		w.field(1, thriftInt32).varint(int64(ByteArray))
		w.field(3, thriftInt32).varint(int64(Optional))
	})
	if err := e.read(w.reader()); err != nil {
		t.Fatalf("read: %v", err)
	}
	if e.Type != ByteArray || e.Repetition != Optional {
		t.Fatalf("the element is a %s %s, want an optional byte_array", e.Repetition, e.Type)
	}
	if err := e.read((&builder{}).structure(func() {}).reader()); err != nil {
		t.Fatalf("read: %v", err)
	}
	if e.Type != NoType || e.Repetition != NoRepetition {
		t.Fatalf("reading an empty element over a full one left a %s %s, want none of either",
			e.Repetition, e.Type)
	}
}

// TestKeyValueUnknownField is the metadata a writer hangs off a file, with a
// field this reader does not know about in the middle of it.
func TestKeyValueUnknownField(t *testing.T) {
	w := &builder{}
	w.structure(func() {
		w.field(1, thriftBinary).binary("ARROW:schema")
		w.field(2, thriftBinary).binary("blob")
		w.field(7, thriftList).list(2, thriftInt32).varint(1).varint(2)
	})

	var kv KeyValue
	if err := kv.read(w.reader()); err != nil {
		t.Fatalf("read: %v", err)
	}
	if kv.Key != "ARROW:schema" || kv.Value != "blob" {
		t.Fatalf("read %q = %q, want ARROW:schema = blob", kv.Key, kv.Value)
	}
}
