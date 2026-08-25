package dtype_test

import (
	"testing"

	"github.com/tamnd/kuma/dtype"
)

// allTypes is one value of every type the package can describe, valid unless
// the test says otherwise. Several tests walk it, so a type added to the
// package and not added here is a type with no coverage.
var allTypes = []dtype.DataType{
	dtype.Null,
	dtype.Bool,
	dtype.Int8, dtype.Int16, dtype.Int32, dtype.Int64,
	dtype.Uint8, dtype.Uint16, dtype.Uint32, dtype.Uint64,
	dtype.Float32, dtype.Float64,
	dtype.String, dtype.Binary,
	dtype.LargeString, dtype.LargeBinary,
	dtype.Date32, dtype.Date64,
	dtype.FixedSizeBinary{ByteWidth: 16},
	dtype.Time32{Unit: dtype.Second},
	dtype.Time32{Unit: dtype.Millisecond},
	dtype.Time64{Unit: dtype.Microsecond},
	dtype.Time64{Unit: dtype.Nanosecond},
	dtype.Timestamp{Unit: dtype.Microsecond},
	dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"},
	dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "Europe/London"},
	dtype.Duration{Unit: dtype.Nanosecond},
	dtype.Interval{Unit: dtype.YearMonth},
	dtype.Interval{Unit: dtype.DayTime},
	dtype.Interval{Unit: dtype.MonthDayNano},
	dtype.Decimal128{Precision: 18, Scale: 2},
	dtype.Decimal256{Precision: 50, Scale: 8},
	dtype.List{Elem: dtype.Int64},
	dtype.LargeList{Elem: dtype.String},
	dtype.FixedSizeList{Elem: dtype.Float32, Len: 3},
	dtype.Struct{Fields: []dtype.Field{
		{Name: "a", Type: dtype.Int64},
		{Name: "b", Type: dtype.String, Nullable: true},
	}},
	dtype.Map{Key: dtype.String, Value: dtype.Int64},
	dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
}

func TestTypeNames(t *testing.T) {
	tests := []struct {
		typ  dtype.DataType
		want string
		kind dtype.Kind
	}{
		{dtype.Null, "null", dtype.NullKind},
		{dtype.Bool, "bool", dtype.BoolKind},
		{dtype.Int8, "int8", dtype.Int8Kind},
		{dtype.Int16, "int16", dtype.Int16Kind},
		{dtype.Int32, "int32", dtype.Int32Kind},
		{dtype.Int64, "int64", dtype.Int64Kind},
		{dtype.Uint8, "uint8", dtype.Uint8Kind},
		{dtype.Uint16, "uint16", dtype.Uint16Kind},
		{dtype.Uint32, "uint32", dtype.Uint32Kind},
		{dtype.Uint64, "uint64", dtype.Uint64Kind},
		{dtype.Float32, "float32", dtype.Float32Kind},
		{dtype.Float64, "float64", dtype.Float64Kind},
		{dtype.String, "string", dtype.StringKind},
		{dtype.Binary, "binary", dtype.BinaryKind},
		{dtype.LargeString, "large_string", dtype.LargeStringKind},
		{dtype.LargeBinary, "large_binary", dtype.LargeBinaryKind},
		{dtype.Date32, "date32", dtype.Date32Kind},
		{dtype.Date64, "date64", dtype.Date64Kind},
		{dtype.FixedSizeBinary{ByteWidth: 16}, "fixed_size_binary[16]", dtype.FixedSizeBinaryKind},
		{dtype.Time32{Unit: dtype.Second}, "time32[s]", dtype.Time32Kind},
		{dtype.Time32{Unit: dtype.Millisecond}, "time32[ms]", dtype.Time32Kind},
		{dtype.Time64{Unit: dtype.Microsecond}, "time64[us]", dtype.Time64Kind},
		{dtype.Time64{Unit: dtype.Nanosecond}, "time64[ns]", dtype.Time64Kind},
		{dtype.Timestamp{Unit: dtype.Second}, "timestamp[s]", dtype.TimestampKind},
		{dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}, "timestamp[us, tz=UTC]", dtype.TimestampKind},
		{dtype.Duration{Unit: dtype.Nanosecond}, "duration[ns]", dtype.DurationKind},
		{dtype.Interval{Unit: dtype.YearMonth}, "interval[year_month]", dtype.IntervalKind},
		{dtype.Interval{Unit: dtype.DayTime}, "interval[day_time]", dtype.IntervalKind},
		{dtype.Interval{Unit: dtype.MonthDayNano}, "interval[month_day_nano]", dtype.IntervalKind},
		{dtype.Decimal128{Precision: 18, Scale: 2}, "decimal128(18, 2)", dtype.Decimal128Kind},
		{dtype.Decimal256{Precision: 50, Scale: -3}, "decimal256(50, -3)", dtype.Decimal256Kind},
		{dtype.List{Elem: dtype.Int64}, "list<int64>", dtype.ListKind},
		{dtype.List{Elem: dtype.List{Elem: dtype.Int64}}, "list<list<int64>>", dtype.ListKind},
		{dtype.LargeList{Elem: dtype.String}, "large_list<string>", dtype.LargeListKind},
		{dtype.FixedSizeList{Elem: dtype.Float32, Len: 3}, "fixed_size_list<float32>[3]", dtype.FixedSizeListKind},
		{dtype.Struct{}, "struct<>", dtype.StructKind},
		{
			dtype.Struct{Fields: []dtype.Field{
				{Name: "a", Type: dtype.Int64},
				{Name: "b", Type: dtype.String},
			}},
			"struct<a: int64 not null, b: string not null>",
			dtype.StructKind,
		},
		{dtype.Map{Key: dtype.String, Value: dtype.Int64}, "map<string, int64>", dtype.MapKind},
		{dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String}, "dictionary<uint32, string>", dtype.DictionaryKind},
	}

	for _, tt := range tests {
		if got := tt.typ.String(); got != tt.want {
			t.Errorf("String() = %q, want %q", got, tt.want)
		}
		if got := tt.typ.Kind(); got != tt.kind {
			t.Errorf("%s: Kind() = %v, want %v", tt.want, got, tt.kind)
		}
	}
}

// TestNamesAreUnique holds the promise made in the DataType doc comment, that
// two types print the same only if they are the same type. Error messages and
// test failures both lean on it.
func TestNamesAreUnique(t *testing.T) {
	seen := make(map[string]dtype.DataType, len(allTypes))
	for _, typ := range allTypes {
		name := typ.String()
		if prev, dup := seen[name]; dup {
			t.Errorf("%v and %v both print as %q", prev, typ, name)
		}
		seen[name] = typ
	}
}

// TestNilChildNames checks that printing a half built type says which part is
// missing rather than panicking, since that is the shape a type has while an
// error about it is being written.
func TestNilChildNames(t *testing.T) {
	tests := []struct {
		typ  dtype.DataType
		want string
	}{
		{dtype.List{}, "list<<nil>>"},
		{dtype.LargeList{}, "large_list<<nil>>"},
		{dtype.FixedSizeList{Len: 2}, "fixed_size_list<<nil>>[2]"},
		{dtype.Map{}, "map<<nil>, <nil>>"},
		{dtype.Dictionary{}, "dictionary<<nil>, <nil>>"},
		{dtype.Struct{Fields: []dtype.Field{{Name: "a"}}}, "struct<a: <nil> not null>"},
	}

	for _, tt := range tests {
		if got := tt.typ.String(); got != tt.want {
			t.Errorf("String() = %q, want %q", got, tt.want)
		}
	}
}

func TestKindString(t *testing.T) {
	if got := dtype.InvalidKind.String(); got != "invalid" {
		t.Errorf("InvalidKind.String() = %q, want %q", got, "invalid")
	}
	if got := dtype.Kind(200).String(); got != "Kind(200)" {
		t.Errorf("Kind(200).String() = %q, want %q", got, "Kind(200)")
	}

	// Every kind between invalid and the last one must have a distinct name, or
	// a dispatch table keyed on kind prints two entries the same way.
	seen := make(map[string]bool)
	for k := dtype.InvalidKind; k <= dtype.DictionaryKind; k++ {
		name := k.String()
		if name == "" {
			t.Errorf("kind %d has no name", k)
		}
		if seen[name] {
			t.Errorf("two kinds print as %q", name)
		}
		seen[name] = true
	}
}

// TestEveryKindIsReachable checks that no kind constant was added without a
// type that reports it. It is the test that fails when half a type lands.
func TestEveryKindIsReachable(t *testing.T) {
	seen := make(map[dtype.Kind]bool, len(allTypes))
	for _, typ := range allTypes {
		seen[typ.Kind()] = true
	}
	for k := dtype.NullKind; k <= dtype.DictionaryKind; k++ {
		if !seen[k] {
			t.Errorf("no type in allTypes reports kind %v", k)
		}
	}
}

func TestBits(t *testing.T) {
	tests := []struct {
		typ   dtype.DataType
		bits  int
		fixed bool
	}{
		{dtype.Bool, 1, true},
		{dtype.Int8, 8, true},
		{dtype.Int16, 16, true},
		{dtype.Int32, 32, true},
		{dtype.Int64, 64, true},
		{dtype.Uint8, 8, true},
		{dtype.Uint16, 16, true},
		{dtype.Uint32, 32, true},
		{dtype.Uint64, 64, true},
		{dtype.Float32, 32, true},
		{dtype.Float64, 64, true},
		{dtype.Date32, 32, true},
		{dtype.Date64, 64, true},
		{dtype.Time32{}, 32, true},
		{dtype.Time64{}, 64, true},
		{dtype.Timestamp{}, 64, true},
		{dtype.Duration{}, 64, true},
		{dtype.Interval{Unit: dtype.YearMonth}, 32, true},
		{dtype.Interval{Unit: dtype.DayTime}, 64, true},
		{dtype.Interval{Unit: dtype.MonthDayNano}, 128, true},
		{dtype.Interval{Unit: 9}, 0, true},
		{dtype.Decimal128{Precision: 18}, 128, true},
		{dtype.Decimal256{Precision: 50}, 256, true},
		{dtype.FixedSizeBinary{ByteWidth: 16}, 128, true},

		// Null has no storage and the variable width types have no one width.
		{dtype.Null, 0, false},
		{dtype.String, 0, false},
		{dtype.Binary, 0, false},
		{dtype.LargeString, 0, false},
		{dtype.LargeBinary, 0, false},
		{dtype.List{Elem: dtype.Int64}, 0, false},
		{dtype.Struct{}, 0, false},
		{dtype.Map{}, 0, false},
		{dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String}, 0, false},
		{nil, 0, false},
	}

	for _, tt := range tests {
		bits, ok := dtype.Bits(tt.typ)
		if ok != tt.fixed || bits != tt.bits {
			t.Errorf("Bits(%s) = (%d, %v), want (%d, %v)",
				name(tt.typ), bits, ok, tt.bits, tt.fixed)
		}
	}
}

func TestEqual(t *testing.T) {
	tests := []struct {
		name string
		a, b dtype.DataType
		want bool
	}{
		{"same singleton", dtype.Int64, dtype.Int64, true},
		{"different singletons", dtype.Int64, dtype.Int32, false},
		{"signed against unsigned", dtype.Int64, dtype.Uint64, false},
		{"both nil", nil, nil, true},
		{"nil against a type", nil, dtype.Int64, false},
		{"a type against nil", dtype.Int64, nil, false},

		{
			"same timestamp",
			dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"},
			dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"},
			true,
		},
		{
			"timestamps differing only in unit",
			dtype.Timestamp{Unit: dtype.Microsecond},
			dtype.Timestamp{Unit: dtype.Nanosecond},
			false,
		},
		{
			"timestamps differing only in zone",
			dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"},
			dtype.Timestamp{Unit: dtype.Microsecond, Zone: "Europe/London"},
			false,
		},
		{
			"naive against zoned",
			dtype.Timestamp{Unit: dtype.Microsecond},
			dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"},
			false,
		},
		{"time32 against time64", dtype.Time32{}, dtype.Time64{}, false},

		{
			"same decimal",
			dtype.Decimal128{Precision: 18, Scale: 2},
			dtype.Decimal128{Precision: 18, Scale: 2},
			true,
		},
		{
			"decimals differing in scale",
			dtype.Decimal128{Precision: 18, Scale: 2},
			dtype.Decimal128{Precision: 18, Scale: 3},
			false,
		},
		{
			"decimal128 against decimal256",
			dtype.Decimal128{Precision: 18, Scale: 2},
			dtype.Decimal256{Precision: 18, Scale: 2},
			false,
		},

		{"same list", dtype.List{Elem: dtype.Int64}, dtype.List{Elem: dtype.Int64}, true},
		{"lists of different elements", dtype.List{Elem: dtype.Int64}, dtype.List{Elem: dtype.Int32}, false},
		{"list against large_list", dtype.List{Elem: dtype.Int64}, dtype.LargeList{Elem: dtype.Int64}, false},
		{
			"same large_list",
			dtype.LargeList{Elem: dtype.String},
			dtype.LargeList{Elem: dtype.String},
			true,
		},
		{
			"nested lists",
			dtype.List{Elem: dtype.List{Elem: dtype.Int64}},
			dtype.List{Elem: dtype.List{Elem: dtype.Int64}},
			true,
		},
		{
			"nested lists differing at the bottom",
			dtype.List{Elem: dtype.List{Elem: dtype.Int64}},
			dtype.List{Elem: dtype.List{Elem: dtype.Float64}},
			false,
		},
		{
			"same fixed_size_list",
			dtype.FixedSizeList{Elem: dtype.Float32, Len: 3},
			dtype.FixedSizeList{Elem: dtype.Float32, Len: 3},
			true,
		},
		{
			"fixed_size_lists of different lengths",
			dtype.FixedSizeList{Elem: dtype.Float32, Len: 3},
			dtype.FixedSizeList{Elem: dtype.Float32, Len: 4},
			false,
		},
		{
			"fixed_size_lists of different elements",
			dtype.FixedSizeList{Elem: dtype.Float32, Len: 3},
			dtype.FixedSizeList{Elem: dtype.Float64, Len: 3},
			false,
		},

		{"empty structs", dtype.Struct{}, dtype.Struct{}, true},
		{
			"same struct",
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			true,
		},
		{
			"structs with different field names",
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Struct{Fields: []dtype.Field{{Name: "b", Type: dtype.Int64}}},
			false,
		},
		{
			"structs with different field counts",
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Struct{Fields: []dtype.Field{
				{Name: "a", Type: dtype.Int64},
				{Name: "b", Type: dtype.Int64},
			}},
			false,
		},
		{
			"structs with the same fields in a different order",
			dtype.Struct{Fields: []dtype.Field{
				{Name: "a", Type: dtype.Int64},
				{Name: "b", Type: dtype.String},
			}},
			dtype.Struct{Fields: []dtype.Field{
				{Name: "b", Type: dtype.String},
				{Name: "a", Type: dtype.Int64},
			}},
			false,
		},
		{
			"structs differing only in nullability",
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64, Nullable: true}}},
			false,
		},

		{
			"same map",
			dtype.Map{Key: dtype.String, Value: dtype.Int64},
			dtype.Map{Key: dtype.String, Value: dtype.Int64},
			true,
		},
		{
			"maps with different keys",
			dtype.Map{Key: dtype.String, Value: dtype.Int64},
			dtype.Map{Key: dtype.Int32, Value: dtype.Int64},
			false,
		},
		{
			"maps with different values",
			dtype.Map{Key: dtype.String, Value: dtype.Int64},
			dtype.Map{Key: dtype.String, Value: dtype.Float64},
			false,
		},

		{
			"same dictionary",
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			true,
		},
		{
			"dictionaries with different index widths",
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Uint8, Value: dtype.String},
			false,
		},
		{
			"dictionaries with different values",
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.Binary},
			false,
		},
		{
			"a dictionary of strings is not a string",
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			dtype.String,
			false,
		},

		{
			"same fixed_size_binary",
			dtype.FixedSizeBinary{ByteWidth: 16},
			dtype.FixedSizeBinary{ByteWidth: 16},
			true,
		},
		{
			"fixed_size_binary of different widths",
			dtype.FixedSizeBinary{ByteWidth: 16},
			dtype.FixedSizeBinary{ByteWidth: 32},
			false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := dtype.Equal(tt.a, tt.b); got != tt.want {
				t.Errorf("Equal(%s, %s) = %v, want %v", name(tt.a), name(tt.b), got, tt.want)
			}
			// Equality is symmetric, and the recursive cases are where an
			// implementation stops being so.
			if got := dtype.Equal(tt.b, tt.a); got != tt.want {
				t.Errorf("Equal(%s, %s) = %v, want %v", name(tt.b), name(tt.a), got, tt.want)
			}
		})
	}
}

// TestEqualReflexive checks that every type equals itself, which the table
// above only covers for the cases someone thought to write down.
func TestEqualReflexive(t *testing.T) {
	for _, typ := range allTypes {
		if !dtype.Equal(typ, typ) {
			t.Errorf("Equal(%s, %s) = false", typ, typ)
		}
	}
}

// TestEqualDistinct checks that no two of the types are equal to each other,
// which is the other half of the promise and the half that catches a Kind check
// that forgot to compare the parameters.
func TestEqualDistinct(t *testing.T) {
	for i, a := range allTypes {
		for j, b := range allTypes {
			if i == j {
				continue
			}
			if dtype.Equal(a, b) {
				t.Errorf("Equal(%s, %s) = true, want false", a, b)
			}
		}
	}
}

// odd is a DataType from outside the package, which is what Equal has to
// tolerate rather than crash on. It claims a kind it is not.
type odd struct{}

func (odd) Kind() dtype.Kind { return dtype.ListKind }
func (odd) String() string   { return "odd" }

func TestEqualForeignType(t *testing.T) {
	if dtype.Equal(dtype.List{Elem: dtype.Int64}, odd{}) {
		t.Error("a list equals a foreign type claiming to be a list")
	}
	if dtype.Equal(odd{}, dtype.List{Elem: dtype.Int64}) {
		t.Error("a foreign type claiming to be a list equals a list")
	}
	if !dtype.Equal(odd{}, odd{}) {
		t.Error("a foreign type does not equal itself")
	}
}

func TestPredicates(t *testing.T) {
	// Each row lists the predicates that must report true for the type. Every
	// other predicate must report false, so a type that starts matching one it
	// should not is a failure without anyone updating the row.
	tests := []struct {
		typ  dtype.DataType
		true []string
	}{
		{nil, nil},
		{dtype.Null, nil},
		{dtype.Bool, []string{"fixed"}},
		{dtype.Int8, []string{"signed", "integer", "numeric", "fixed"}},
		{dtype.Int64, []string{"signed", "integer", "numeric", "fixed"}},
		{dtype.Uint8, []string{"unsigned", "integer", "numeric", "fixed"}},
		{dtype.Uint64, []string{"unsigned", "integer", "numeric", "fixed"}},
		{dtype.Float32, []string{"float", "numeric", "fixed"}},
		{dtype.Float64, []string{"float", "numeric", "fixed"}},
		{dtype.String, []string{"string"}},
		{dtype.LargeString, []string{"string"}},
		{dtype.Binary, []string{"binary"}},
		{dtype.LargeBinary, []string{"binary"}},
		{dtype.FixedSizeBinary{ByteWidth: 16}, []string{"binary", "fixed"}},
		{dtype.Date32, []string{"temporal", "fixed"}},
		{dtype.Date64, []string{"temporal", "fixed"}},
		{dtype.Time32{}, []string{"temporal", "fixed"}},
		{dtype.Time64{Unit: dtype.Nanosecond}, []string{"temporal", "fixed"}},
		{dtype.Timestamp{}, []string{"temporal", "fixed"}},
		{dtype.Duration{}, []string{"temporal", "fixed"}},
		{dtype.Interval{}, []string{"temporal", "fixed"}},
		{dtype.Decimal128{Precision: 18}, []string{"decimal", "numeric", "fixed"}},
		{dtype.Decimal256{Precision: 50}, []string{"decimal", "numeric", "fixed"}},
		{dtype.List{Elem: dtype.Int64}, []string{"nested"}},
		{dtype.LargeList{Elem: dtype.Int64}, []string{"nested"}},
		{dtype.FixedSizeList{Elem: dtype.Int64, Len: 2}, []string{"nested"}},
		{dtype.Struct{}, []string{"nested"}},
		{dtype.Map{Key: dtype.String, Value: dtype.Int64}, []string{"nested"}},
		{dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String}, nil},
	}

	preds := map[string]func(dtype.DataType) bool{
		"signed":   dtype.IsSigned,
		"unsigned": dtype.IsUnsigned,
		"integer":  dtype.IsInteger,
		"float":    dtype.IsFloat,
		"decimal":  dtype.IsDecimal,
		"numeric":  dtype.IsNumeric,
		"temporal": dtype.IsTemporal,
		"string":   dtype.IsString,
		"binary":   dtype.IsBinary,
		"nested":   dtype.IsNested,
		"fixed": func(typ dtype.DataType) bool {
			_, ok := dtype.Bits(typ)
			return ok
		},
	}

	for _, tt := range tests {
		want := make(map[string]bool, len(tt.true))
		for _, p := range tt.true {
			if _, known := preds[p]; !known {
				t.Fatalf("the table names a predicate that does not exist: %q", p)
			}
			want[p] = true
		}
		for p, fn := range preds {
			if got := fn(tt.typ); got != want[p] {
				t.Errorf("%s predicate on %s = %v, want %v", p, name(tt.typ), got, want[p])
			}
		}
	}
}

func TestTimeUnitString(t *testing.T) {
	tests := []struct {
		unit dtype.TimeUnit
		want string
		ok   bool
	}{
		{dtype.Second, "s", true},
		{dtype.Millisecond, "ms", true},
		{dtype.Microsecond, "us", true},
		{dtype.Nanosecond, "ns", true},
		{dtype.TimeUnit(9), "TimeUnit(9)", false},
	}

	for _, tt := range tests {
		if got := tt.unit.String(); got != tt.want {
			t.Errorf("String() = %q, want %q", got, tt.want)
		}
		if got := tt.unit.Valid(); got != tt.ok {
			t.Errorf("%s: Valid() = %v, want %v", tt.want, got, tt.ok)
		}
	}
}

func TestIntervalUnitString(t *testing.T) {
	tests := []struct {
		unit dtype.IntervalUnit
		want string
		ok   bool
	}{
		{dtype.YearMonth, "year_month", true},
		{dtype.DayTime, "day_time", true},
		{dtype.MonthDayNano, "month_day_nano", true},
		{dtype.IntervalUnit(9), "IntervalUnit(9)", false},
	}

	for _, tt := range tests {
		if got := tt.unit.String(); got != tt.want {
			t.Errorf("String() = %q, want %q", got, tt.want)
		}
		if got := tt.unit.Valid(); got != tt.ok {
			t.Errorf("%s: Valid() = %v, want %v", tt.want, got, tt.ok)
		}
	}
}

func TestStructField(t *testing.T) {
	s := dtype.Struct{Fields: []dtype.Field{
		{Name: "a", Type: dtype.Int64},
		{Name: "b", Type: dtype.String, Nullable: true},
	}}

	f, ok := s.Field("b")
	if !ok {
		t.Fatal(`Field("b") not found`)
	}
	if !f.Equal(dtype.Field{Name: "b", Type: dtype.String, Nullable: true}) {
		t.Errorf(`Field("b") = %v`, f)
	}
	if _, ok := s.Field("missing"); ok {
		t.Error(`Field("missing") found something`)
	}
}

// name is String with a nil check, for the test failure messages.
func name(t dtype.DataType) string {
	if t == nil {
		return "<nil>"
	}
	return t.String()
}
