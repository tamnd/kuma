package dtype_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
)

// TestValidateAccepts runs every type the package can describe through
// Validate, so a rule that is too strict fails here rather than in the reader
// that produced the type.
func TestValidateAccepts(t *testing.T) {
	extra := []dtype.DataType{
		dtype.Decimal128{Precision: 1, Scale: 0},
		dtype.Decimal128{Precision: 38, Scale: 38},
		dtype.Decimal128{Precision: 18, Scale: -18},
		dtype.Decimal256{Precision: 76, Scale: 0},
		dtype.FixedSizeBinary{ByteWidth: 0},
		dtype.FixedSizeList{Elem: dtype.Int64, Len: 0},
		dtype.Timestamp{Unit: dtype.Second, Zone: "Not/AZone"},
		dtype.List{Elem: dtype.Struct{Fields: []dtype.Field{
			{Name: "inner", Type: dtype.Map{Key: dtype.String, Value: dtype.List{Elem: dtype.Float64}}},
		}}},
		dtype.Dictionary{Index: dtype.Int8, Value: dtype.List{Elem: dtype.String}},
	}

	for _, typ := range append(append([]dtype.DataType{}, allTypes...), extra...) {
		if err := dtype.Validate(typ); err != nil {
			t.Errorf("Validate(%s) = %v, want no error", typ, err)
		}
	}
}

func TestValidateRejects(t *testing.T) {
	tests := []struct {
		name string
		typ  dtype.DataType
		want string
	}{
		{"nil", nil, "nil type"},

		{"time32 in microseconds", dtype.Time32{Unit: dtype.Microsecond}, "time32 unit must be s or ms"},
		{"time32 in nanoseconds", dtype.Time32{Unit: dtype.Nanosecond}, "time32 unit must be s or ms"},
		{"time64 in seconds", dtype.Time64{Unit: dtype.Second}, "time64 unit must be us or ns"},
		{"time64 in milliseconds", dtype.Time64{Unit: dtype.Millisecond}, "time64 unit must be us or ns"},
		{"timestamp with an unknown unit", dtype.Timestamp{Unit: 9}, "timestamp has unknown unit"},
		{"duration with an unknown unit", dtype.Duration{Unit: 9}, "duration has unknown unit"},
		{"interval with an unknown unit", dtype.Interval{Unit: 9}, "interval has unknown unit"},

		{"decimal128 with no precision", dtype.Decimal128{Precision: 0}, "precision 0 out of range"},
		{"decimal128 with negative precision", dtype.Decimal128{Precision: -1}, "precision -1 out of range"},
		{"decimal128 with too much precision", dtype.Decimal128{Precision: 39}, "precision 39 out of range 1 to 38"},
		{"decimal256 with too much precision", dtype.Decimal256{Precision: 77}, "precision 77 out of range 1 to 76"},
		{"decimal scale above precision", dtype.Decimal128{Precision: 4, Scale: 5}, "scale 5 out of range -4 to 4"},
		{"decimal scale below precision", dtype.Decimal128{Precision: 4, Scale: -5}, "scale -5 out of range -4 to 4"},

		{"fixed_size_binary of negative width", dtype.FixedSizeBinary{ByteWidth: -1}, "negative width -1"},
		{"fixed_size_list of negative length", dtype.FixedSizeList{Elem: dtype.Int64, Len: -1}, "negative length -1"},

		{"list of nothing", dtype.List{}, "list: nil type"},
		{"large_list of nothing", dtype.LargeList{}, "large_list: nil type"},
		{"fixed_size_list of nothing", dtype.FixedSizeList{Len: 2}, "fixed_size_list: nil type"},
		{"list of a bad element", dtype.List{Elem: dtype.Time32{Unit: dtype.Nanosecond}}, "list: time32 unit"},

		{"map with no key", dtype.Map{Value: dtype.Int64}, "map key: nil type"},
		{"map with no value", dtype.Map{Key: dtype.String}, "map value: nil type"},
		{
			"map with a bad value",
			dtype.Map{Key: dtype.String, Value: dtype.Decimal128{Precision: 0}},
			"map value: decimal128 precision",
		},

		{"dictionary with no index", dtype.Dictionary{Value: dtype.String}, "dictionary index must be an integer"},
		{
			"dictionary indexed by a string",
			dtype.Dictionary{Index: dtype.String, Value: dtype.String},
			"dictionary index must be an integer type, have string",
		},
		{
			"dictionary indexed by a float",
			dtype.Dictionary{Index: dtype.Float64, Value: dtype.String},
			"dictionary index must be an integer",
		},
		{"dictionary with no value", dtype.Dictionary{Index: dtype.Uint32}, "dictionary value: nil type"},
		{
			"dictionary of dictionary",
			dtype.Dictionary{
				Index: dtype.Uint32,
				Value: dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			},
			"dictionary of dictionary",
		},

		{
			"struct with an unnamed field",
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}, {Type: dtype.Int64}}},
			"struct: field 1 has no name",
		},
		{
			"struct with two fields of one name",
			dtype.Struct{Fields: []dtype.Field{
				{Name: "a", Type: dtype.Int64},
				{Name: "a", Type: dtype.String},
			}},
			`struct: two fields named "a"`,
		},
		{
			"struct with a bad field type",
			dtype.Struct{Fields: []dtype.Field{{Name: "when", Type: dtype.Time64{Unit: dtype.Second}}}},
			`struct: field "when": time64 unit`,
		},
		{
			"the failure two levels down names both levels",
			dtype.List{Elem: dtype.Struct{Fields: []dtype.Field{
				{Name: "when", Type: dtype.Time64{Unit: dtype.Second}},
			}}},
			`list: struct: field "when": time64 unit`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := dtype.Validate(tt.typ)
			if err == nil {
				t.Fatalf("Validate(%s) = nil, want an error mentioning %q", name(tt.typ), tt.want)
			}
			if !strings.HasPrefix(err.Error(), "dtype: ") {
				t.Errorf("error %q does not start with the package name", err)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error %q does not contain %q", err, tt.want)
			}
			// One prefix per error, however many levels deep the cause was.
			if n := strings.Count(err.Error(), "dtype: "); n != 1 {
				t.Errorf("error %q repeats the package name %d times", err, n)
			}
		})
	}
}

// unknown is a type from outside the package that reports a kind no type has.
// Validate has to reject it rather than accept it by falling off the end of the
// switch.
type unknown struct{}

func (unknown) Kind() dtype.Kind { return dtype.InvalidKind }
func (unknown) String() string   { return "unknown" }

// wild reports a kind past the end of the table.
type wild struct{}

func (wild) Kind() dtype.Kind { return dtype.Kind(200) }
func (wild) String() string   { return "wild" }

func TestValidateForeignType(t *testing.T) {
	for _, typ := range []dtype.DataType{unknown{}, wild{}} {
		err := dtype.Validate(typ)
		if err == nil {
			t.Fatalf("Validate(%s) = nil, want an error", typ)
		}
		if !strings.Contains(err.Error(), "unknown type") {
			t.Errorf("error %q does not say the type is unknown", err)
		}
	}
}

// TestValidateTooDeep builds a type nested past the limit, which is what a
// self-referential type from outside the package would look like from inside
// the walk.
func TestValidateTooDeep(t *testing.T) {
	var deep dtype.DataType
	deep = dtype.Int64
	for range dtype.MaxNestingDepth + 1 {
		deep = dtype.List{Elem: deep}
	}

	err := dtype.Validate(deep)
	if !errors.Is(err, dtype.ErrTooDeep) {
		t.Fatalf("Validate = %v, want ErrTooDeep", err)
	}
	// The depth limit is the one error that must not be wrapped once per level,
	// or the message is sixty four copies of the word list.
	if strings.Count(err.Error(), "list") != 0 {
		t.Errorf("error %q carries the nesting it gave up on", err)
	}

	// One below the limit still has to pass, or the limit is off by one and
	// nobody notices until a schema this deep shows up.
	var shallow dtype.DataType
	shallow = dtype.Int64
	for range dtype.MaxNestingDepth - 1 {
		shallow = dtype.List{Elem: shallow}
	}
	if err := dtype.Validate(shallow); err != nil {
		t.Errorf("Validate at depth %d = %v, want no error", dtype.MaxNestingDepth-1, err)
	}
}

func TestSchemaValidate(t *testing.T) {
	tests := []struct {
		name   string
		schema dtype.Schema
		want   string
	}{
		{"empty", dtype.Schema{}, ""},
		{"valid", sample(), ""},
		{
			"unnamed field",
			dtype.Schema{Fields: []dtype.Field{{Type: dtype.Int64}}},
			"schema: field 0 has no name",
		},
		{
			"duplicate names",
			dtype.Schema{Fields: []dtype.Field{
				{Name: "id", Type: dtype.Int64},
				{Name: "id", Type: dtype.String},
			}},
			`schema: two fields named "id"`,
		},
		{
			"field with no type",
			dtype.Schema{Fields: []dtype.Field{{Name: "id"}}},
			`schema: field "id": nil type`,
		},
		{
			"field with a bad type",
			dtype.Schema{Fields: []dtype.Field{
				{Name: "price", Type: dtype.Decimal128{Precision: 100}},
			}},
			`schema: field "price": decimal128 precision 100 out of range`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.schema.Validate()
			if tt.want == "" {
				if err != nil {
					t.Fatalf("Validate = %v, want no error", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("Validate = nil, want an error mentioning %q", tt.want)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error %q does not contain %q", err, tt.want)
			}
		})
	}
}

func TestFieldValidate(t *testing.T) {
	tests := []struct {
		name  string
		field dtype.Field
		want  string
	}{
		{"valid", dtype.Field{Name: "id", Type: dtype.Int64}, ""},
		{"no name", dtype.Field{Type: dtype.Int64}, "field has no name"},
		{"no type", dtype.Field{Name: "id"}, `field "id": nil type`},
		{
			"bad type",
			dtype.Field{Name: "when", Type: dtype.Time32{Unit: dtype.Nanosecond}},
			`field "when": time32 unit`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.field.Validate()
			if tt.want == "" {
				if err != nil {
					t.Fatalf("Validate = %v, want no error", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("Validate = nil, want an error mentioning %q", tt.want)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error %q does not contain %q", err, tt.want)
			}
		})
	}
}
