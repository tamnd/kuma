package ipc_test

import (
	"errors"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

// mapping is one type, the format string it exports as, the children that
// travel with it, and the type it comes back as.
//
// The three parts are one table because they are one claim. A format string
// that is written one way and read another is the failure this boundary exists
// to prevent, and splitting the table into a writer test and a reader test is
// how the two halves drift apart.
type mapping struct {
	typ      dtype.DataType
	format   string
	children []dtype.Field

	// back is the type the format string reads as, when that is not typ. Only
	// the text and byte layouts set it. See the note on ipc.Type.
	back dtype.DataType
}

var mappings = []mapping{
	{typ: dtype.Null, format: "n"},
	{typ: dtype.Bool, format: "b"},
	{typ: dtype.Int8, format: "c"},
	{typ: dtype.Int16, format: "s"},
	{typ: dtype.Int32, format: "i"},
	{typ: dtype.Int64, format: "l"},
	{typ: dtype.Uint8, format: "C"},
	{typ: dtype.Uint16, format: "S"},
	{typ: dtype.Uint32, format: "I"},
	{typ: dtype.Uint64, format: "L"},
	{typ: dtype.Float32, format: "f"},
	{typ: dtype.Float64, format: "g"},

	{typ: dtype.String, format: "vu"},
	{typ: dtype.Binary, format: "vz"},
	{typ: dtype.LargeString, format: "U", back: dtype.String},
	{typ: dtype.LargeBinary, format: "Z", back: dtype.Binary},
	{typ: dtype.FixedSizeBinary{ByteWidth: 16}, format: "w:16"},
	{typ: dtype.FixedSizeBinary{ByteWidth: 0}, format: "w:0"},

	{typ: dtype.Date32, format: "tdD"},
	{typ: dtype.Date64, format: "tdm"},
	{typ: dtype.Time32{Unit: dtype.Second}, format: "tts"},
	{typ: dtype.Time32{Unit: dtype.Millisecond}, format: "ttm"},
	{typ: dtype.Time64{Unit: dtype.Microsecond}, format: "ttu"},
	{typ: dtype.Time64{Unit: dtype.Nanosecond}, format: "ttn"},

	// A timestamp with no zone is naive local time and keeps its trailing
	// colon, which is what tells it apart from one in UTC.
	{typ: dtype.Timestamp{Unit: dtype.Second}, format: "tss:"},
	{typ: dtype.Timestamp{Unit: dtype.Millisecond}, format: "tsm:"},
	{typ: dtype.Timestamp{Unit: dtype.Microsecond}, format: "tsu:"},
	{typ: dtype.Timestamp{Unit: dtype.Nanosecond}, format: "tsn:"},
	{typ: dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}, format: "tsu:UTC"},
	{typ: dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "Europe/London"}, format: "tsn:Europe/London"},

	// A fixed offset zone has a colon in it, so the zone runs to the end of
	// the string rather than to the next separator.
	{typ: dtype.Timestamp{Unit: dtype.Second, Zone: "+01:00"}, format: "tss:+01:00"},

	{typ: dtype.Duration{Unit: dtype.Second}, format: "tDs"},
	{typ: dtype.Duration{Unit: dtype.Millisecond}, format: "tDm"},
	{typ: dtype.Duration{Unit: dtype.Microsecond}, format: "tDu"},
	{typ: dtype.Duration{Unit: dtype.Nanosecond}, format: "tDn"},
	{typ: dtype.Interval{Unit: dtype.YearMonth}, format: "tiM"},
	{typ: dtype.Interval{Unit: dtype.DayTime}, format: "tiD"},
	{typ: dtype.Interval{Unit: dtype.MonthDayNano}, format: "tin"},

	{typ: dtype.Decimal128{Precision: 18, Scale: 2}, format: "d:18,2"},
	{typ: dtype.Decimal128{Precision: 38, Scale: -3}, format: "d:38,-3"},
	{typ: dtype.Decimal256{Precision: 50, Scale: 8}, format: "d:50,8,256"},

	{
		typ:      dtype.List{Elem: dtype.Int64},
		format:   "+l",
		children: []dtype.Field{{Name: "item", Type: dtype.Int64, Nullable: true}},
	},
	{
		typ:      dtype.LargeList{Elem: dtype.String},
		format:   "+L",
		children: []dtype.Field{{Name: "item", Type: dtype.String, Nullable: true}},
	},
	{
		typ:      dtype.FixedSizeList{Elem: dtype.Float32, Len: 3},
		format:   "+w:3",
		children: []dtype.Field{{Name: "item", Type: dtype.Float32, Nullable: true}},
	},
	{
		typ: dtype.Struct{Fields: []dtype.Field{
			{Name: "a", Type: dtype.Int64},
			{Name: "b", Type: dtype.String, Nullable: true},
		}},
		format: "+s",
		children: []dtype.Field{
			{Name: "a", Type: dtype.Int64},
			{Name: "b", Type: dtype.String, Nullable: true},
		},
	},
	{
		typ:    dtype.Map{Key: dtype.String, Value: dtype.Int64},
		format: "+m",
		children: []dtype.Field{{Name: "entries", Type: dtype.Struct{Fields: []dtype.Field{
			{Name: "key", Type: dtype.String},
			{Name: "value", Type: dtype.Int64, Nullable: true},
		}}}},
	},

	// A dictionary exports as its index type. The values travel in the
	// dictionary member of the schema, so the string that comes back is the
	// index and not the dictionary.
	{
		typ:    dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
		format: "I",
		back:   dtype.Uint32,
	},

	// Two levels of nesting, to prove that a child is a whole type rather than
	// a leaf. The children of the outer list are what an exporter would have
	// built one level down.
	{
		typ:      dtype.List{Elem: dtype.List{Elem: dtype.Int32}},
		format:   "+l",
		children: []dtype.Field{{Name: "item", Type: dtype.List{Elem: dtype.Int32}, Nullable: true}},
	},
}

func TestFormat(t *testing.T) {
	for _, tt := range mappings {
		got, err := ipc.Format(tt.typ)
		if err != nil {
			t.Errorf("Format(%s) = %v", tt.typ, err)
			continue
		}
		if got != tt.format {
			t.Errorf("Format(%s) = %q, want %q", tt.typ, got, tt.format)
		}
	}
}

func TestRoundTrip(t *testing.T) {
	for _, tt := range mappings {
		format, err := ipc.Format(tt.typ)
		if err != nil {
			t.Errorf("Format(%s) = %v", tt.typ, err)
			continue
		}
		got, err := ipc.Type(format, tt.children)
		if err != nil {
			t.Errorf("Type(%q) = %v", format, err)
			continue
		}
		want := tt.typ
		if tt.back != nil {
			want = tt.back
		}
		if !dtype.Equal(got, want) {
			t.Errorf("Type(Format(%s)) = %s, want %s", tt.typ, got, want)
		}
	}
}

// unknown is a DataType from outside this module, which is what a caller gets
// by writing four lines. Nothing can be said about it, and saying so is better
// than returning a format string for whichever kind it claims to be.
type unknown struct{}

func (unknown) Kind() dtype.Kind { return dtype.InvalidKind }
func (unknown) String() string   { return "unknown" }

func TestFormatErrors(t *testing.T) {
	tests := []struct {
		name string
		typ  dtype.DataType
	}{
		{"nil", nil},
		{"not ours", unknown{}},
		{"time32 in nanoseconds", dtype.Time32{Unit: dtype.Nanosecond}},
		{"time64 in seconds", dtype.Time64{Unit: dtype.Second}},
		{"timestamp with no unit", dtype.Timestamp{Unit: dtype.TimeUnit(9)}},
		{"duration with no unit", dtype.Duration{Unit: dtype.TimeUnit(9)}},
		{"interval with no unit", dtype.Interval{Unit: dtype.IntervalUnit(9)}},
		{"negative binary width", dtype.FixedSizeBinary{ByteWidth: -1}},
		{"negative list length", dtype.FixedSizeList{Elem: dtype.Int64, Len: -1}},
		{"dictionary indexed by text", dtype.Dictionary{Index: dtype.String, Value: dtype.String}},
		{"dictionary indexed by nothing", dtype.Dictionary{Value: dtype.String}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ipc.Format(tt.typ)
			if err == nil {
				t.Fatalf("Format = %q, want an error", got)
			}
			if !errors.Is(err, ipc.ErrType) {
				t.Errorf("Format = %v, want ErrType", err)
			}
		})
	}
}
