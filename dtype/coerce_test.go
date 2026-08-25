package dtype_test

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
)

func TestCoerce(t *testing.T) {
	tests := []struct {
		name string
		a, b dtype.DataType
		want dtype.DataType // nil means the two do not combine
	}{
		{"the same type", dtype.Int64, dtype.Int64, dtype.Int64},

		// The rule the whole package exists for. Every one of these is a silent
		// upcast in pandas and an error here.
		{"int64 and float64", dtype.Int64, dtype.Float64, nil},
		{"int32 and int64", dtype.Int32, dtype.Int64, nil},
		{"int64 and uint64", dtype.Int64, dtype.Uint64, nil},
		{"float32 and float64", dtype.Float32, dtype.Float64, nil},
		{"int64 and bool", dtype.Int64, dtype.Bool, nil},
		{"int64 and string", dtype.Int64, dtype.String, nil},
		{"string and large_string", dtype.String, dtype.LargeString, nil},
		{"binary and string", dtype.Binary, dtype.String, nil},
		{
			"timestamps of different units",
			dtype.Timestamp{Unit: dtype.Microsecond},
			dtype.Timestamp{Unit: dtype.Nanosecond},
			nil,
		},
		{
			"timestamps of different zones",
			dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"},
			dtype.Timestamp{Unit: dtype.Microsecond, Zone: "Europe/London"},
			nil,
		},
		{
			"decimals of different scales",
			dtype.Decimal128{Precision: 18, Scale: 2},
			dtype.Decimal128{Precision: 18, Scale: 4},
			nil,
		},
		{"date32 and date64", dtype.Date32, dtype.Date64, nil},

		// Null has no values, so there is nothing to convert.
		{"null and int64", dtype.Null, dtype.Int64, dtype.Int64},
		{"int64 and null", dtype.Int64, dtype.Null, dtype.Int64},
		{"null and null", dtype.Null, dtype.Null, dtype.Null},
		{
			"null and a nested type",
			dtype.Null,
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
		},

		// Dictionary encoding is storage rather than meaning.
		{
			"a dictionary and its value type",
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			dtype.String,
			dtype.String,
		},
		{
			"a dictionary and another value type",
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			dtype.Int64,
			nil,
		},
		{
			"two dictionaries of different value types",
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.Int64},
			nil,
		},
		{
			"a dictionary and null",
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			dtype.Null,
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
		},

		// The nested types combine element by element.
		{
			"a list of null and a list of int64",
			dtype.List{Elem: dtype.Null},
			dtype.List{Elem: dtype.Int64},
			dtype.List{Elem: dtype.Int64},
		},
		{
			"lists of incompatible elements",
			dtype.List{Elem: dtype.Int64},
			dtype.List{Elem: dtype.Float64},
			nil,
		},
		{"a list and a large_list", dtype.List{Elem: dtype.Int64}, dtype.LargeList{Elem: dtype.Int64}, nil},
		{
			"large_lists of null and int64",
			dtype.LargeList{Elem: dtype.Null},
			dtype.LargeList{Elem: dtype.Int64},
			dtype.LargeList{Elem: dtype.Int64},
		},
		{
			"fixed_size_lists of the same length",
			dtype.FixedSizeList{Elem: dtype.Null, Len: 3},
			dtype.FixedSizeList{Elem: dtype.Int64, Len: 3},
			dtype.FixedSizeList{Elem: dtype.Int64, Len: 3},
		},
		{
			"fixed_size_lists of different lengths",
			dtype.FixedSizeList{Elem: dtype.Int64, Len: 3},
			dtype.FixedSizeList{Elem: dtype.Int64, Len: 4},
			nil,
		},
		{
			"fixed_size_lists of incompatible elements",
			dtype.FixedSizeList{Elem: dtype.Int64, Len: 3},
			dtype.FixedSizeList{Elem: dtype.Float64, Len: 3},
			nil,
		},
		{
			"maps",
			dtype.Map{Key: dtype.String, Value: dtype.Null},
			dtype.Map{Key: dtype.String, Value: dtype.Int64},
			dtype.Map{Key: dtype.String, Value: dtype.Int64},
		},
		{
			"maps of incompatible keys",
			dtype.Map{Key: dtype.String, Value: dtype.Int64},
			dtype.Map{Key: dtype.Int32, Value: dtype.Int64},
			nil,
		},
		{
			"maps of incompatible values",
			dtype.Map{Key: dtype.String, Value: dtype.Int64},
			dtype.Map{Key: dtype.String, Value: dtype.Float64},
			nil,
		},
		{
			"structs, where nullable on either side is nullable in the result",
			dtype.Struct{Fields: []dtype.Field{
				{Name: "a", Type: dtype.Null},
				{Name: "b", Type: dtype.String, Nullable: true},
			}},
			dtype.Struct{Fields: []dtype.Field{
				{Name: "a", Type: dtype.Int64},
				{Name: "b", Type: dtype.String},
			}},
			dtype.Struct{Fields: []dtype.Field{
				{Name: "a", Type: dtype.Int64},
				{Name: "b", Type: dtype.String, Nullable: true},
			}},
		},
		{
			"structs of different field counts",
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Struct{},
			nil,
		},
		{
			"structs of different field names",
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Struct{Fields: []dtype.Field{{Name: "b", Type: dtype.Int64}}},
			nil,
		},
		{
			"structs of incompatible field types",
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Float64}}},
			nil,
		},

		{"nil on the left", nil, dtype.Int64, nil},
		{"nil on the right", dtype.Int64, nil, nil},
		{"nil on both sides", nil, nil, nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			checkCoerce(t, tt.a, tt.b, tt.want)
			// Combining two columns cannot depend on which one was written
			// first, since Concat has no left and right.
			checkCoerce(t, tt.b, tt.a, tt.want)
		})
	}
}

func checkCoerce(t *testing.T, a, b, want dtype.DataType) {
	t.Helper()

	got, err := dtype.Coerce(a, b)
	if want == nil {
		if err == nil {
			t.Errorf("Coerce(%s, %s) = %s, want an error", name(a), name(b), got)
			return
		}
		for _, part := range []string{name(a), name(b), "cast one side explicitly"} {
			if !strings.Contains(err.Error(), part) {
				t.Errorf("error %q does not mention %s", err, part)
			}
		}
		return
	}
	if err != nil {
		t.Errorf("Coerce(%s, %s) = %v, want %s", name(a), name(b), err, want)
		return
	}
	if !dtype.Equal(got, want) {
		t.Errorf("Coerce(%s, %s) = %s, want %s", name(a), name(b), got, want)
	}
}

// TestCoerceIsReflexive checks that every type combines with itself, which is
// the case Concat of two chunks of one column takes.
func TestCoerceIsReflexive(t *testing.T) {
	for _, typ := range allTypes {
		got, err := dtype.Coerce(typ, typ)
		if err != nil {
			t.Errorf("Coerce(%s, %s) = %v", typ, typ, err)
			continue
		}
		if !dtype.Equal(got, typ) {
			t.Errorf("Coerce(%s, %s) = %s", typ, typ, got)
		}
	}
}

// TestCoerceRefusesEveryPair walks every pair of different types and checks
// that the only ones that combine are the ones written down above. It is the
// test that fails if a rule is added without being argued for.
func TestCoerceRefusesEveryPair(t *testing.T) {
	// Two types that are not equal combine only if one of them is null, or if
	// they are a dictionary and the value type it decodes to.
	combines := func(a, b dtype.DataType) bool {
		if a.Kind() == dtype.NullKind || b.Kind() == dtype.NullKind {
			return true
		}
		pair := a.String() + " against " + b.String()
		return pair == "dictionary<uint32, string> against string" ||
			pair == "string against dictionary<uint32, string>"
	}

	for _, a := range allTypes {
		for _, b := range allTypes {
			if dtype.Equal(a, b) {
				continue
			}
			_, err := dtype.Coerce(a, b)
			if (err == nil) != combines(a, b) {
				t.Errorf("Coerce(%s, %s) = %v", a, b, err)
			}
		}
	}
}

// fake claims a kind without being the struct that kind normally means, which
// an implementation outside this package is free to do. Nothing here may assert
// on a type without checking the assertion.
type fake struct{ kind dtype.Kind }

// Kind returns the kind the test asked for.
func (f fake) Kind() dtype.Kind { return f.kind }

// String returns a name that says where the type came from.
func (f fake) String() string { return "fake<" + f.kind.String() + ">" }

func TestCoerceForeignType(t *testing.T) {
	genuine := []dtype.DataType{
		dtype.List{Elem: dtype.Int64},
		dtype.LargeList{Elem: dtype.Int64},
		dtype.FixedSizeList{Elem: dtype.Int64, Len: 3},
		dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
		dtype.Map{Key: dtype.String, Value: dtype.Int64},
		dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
	}

	for _, typ := range genuine {
		impostor := fake{kind: typ.Kind()}
		if _, err := dtype.Coerce(typ, impostor); err == nil {
			t.Errorf("Coerce(%s, %s) returned no error", typ, impostor)
		}
		if _, err := dtype.Coerce(impostor, typ); err == nil {
			t.Errorf("Coerce(%s, %s) returned no error", impostor, typ)
		}
	}
}

// TestCoerceDictionaryIndex covers the one place where two types are widened
// rather than refused. It is allowed here and nowhere else, because nothing
// reads a dictionary index, so making it wider changes nothing a caller can
// see, while making a column wider changes what comes out of the other end.
func TestCoerceDictionaryIndex(t *testing.T) {
	tests := []struct {
		a    dtype.DataType
		b    dtype.DataType
		want dtype.DataType
	}{
		// The same index type on both sides stays where it is.
		{
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.Null},
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
		},
		// Two of the same sign take the wider one.
		{
			dtype.Dictionary{Index: dtype.Uint8, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
		},
		{
			dtype.Dictionary{Index: dtype.Int64, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Int16, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Int64, Value: dtype.String},
		},
		// One of each has to be signed and wide enough for the unsigned side,
		// which costs a doubling.
		{
			dtype.Dictionary{Index: dtype.Int8, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Uint8, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Int16, Value: dtype.String},
		},
		{
			dtype.Dictionary{Index: dtype.Int32, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Uint16, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Int32, Value: dtype.String},
		},
		{
			dtype.Dictionary{Index: dtype.Int8, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Int64, Value: dtype.String},
		},
		// A uint64 next to anything signed does not fit in any integer type, so
		// the result gives up the encoding rather than the values.
		{
			dtype.Dictionary{Index: dtype.Int8, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Uint64, Value: dtype.String},
			dtype.String,
		},
		// An index that is not an integer type is not a dictionary anyone can
		// read, and Validate says so. Coerce still has to answer without
		// panicking, and decoding is the answer that loses nothing.
		{
			dtype.Dictionary{Index: dtype.String, Value: dtype.Int64},
			dtype.Dictionary{Index: dtype.String, Value: dtype.Null},
			dtype.Int64,
		},
	}

	for _, tt := range tests {
		checkCoerce(t, tt.a, tt.b, tt.want)
		checkCoerce(t, tt.b, tt.a, tt.want)
	}
}

func TestCoerceTooDeep(t *testing.T) {
	var a, b dtype.DataType
	a, b = dtype.Int64, dtype.Null
	for range dtype.MaxNestingDepth + 2 {
		a = dtype.List{Elem: a}
		b = dtype.List{Elem: b}
	}

	if _, err := dtype.Coerce(a, b); err == nil {
		t.Error("Coerce of a type nested past the limit returned no error")
	}
}

func TestCoerceLiteral(t *testing.T) {
	tests := []struct {
		name    string
		column  dtype.DataType
		literal dtype.DataType
		want    dtype.DataType // nil means the two do not combine
	}{
		{"the same type", dtype.Int64, dtype.Int64, dtype.Int64},

		// The column keeps its own storage. Comparing a uint32 column against
		// the literal 0 must not widen the column to int64.
		{"an integer literal against uint32", dtype.Uint32, dtype.Int64, dtype.Uint32},
		{"an integer literal against int8", dtype.Int8, dtype.Int64, dtype.Int8},
		{"an integer literal against float64", dtype.Float64, dtype.Int64, dtype.Float64},
		{"a float literal against float32", dtype.Float32, dtype.Float64, dtype.Float32},
		{"an integer literal against decimal128", dtype.Decimal128{Precision: 18, Scale: 2}, dtype.Int64, dtype.Decimal128{Precision: 18, Scale: 2}},

		// A float literal against an exact column is the one refusal, because
		// 1.5 has no int64 spelling and 0.1 has no exact decimal one either.
		{"a float literal against int64", dtype.Int64, dtype.Float64, nil},
		{"a float literal against decimal128", dtype.Decimal128{Precision: 18, Scale: 2}, dtype.Float64, nil},

		{"a null literal", dtype.Int64, dtype.Null, dtype.Int64},
		{"a literal against a null column", dtype.Null, dtype.Int64, dtype.Int64},

		{"a string literal against large_string", dtype.LargeString, dtype.String, dtype.LargeString},
		{"a string literal against int64", dtype.Int64, dtype.String, nil},
		{"an integer literal against string", dtype.String, dtype.Int64, nil},
		{"a binary literal against large_binary", dtype.LargeBinary, dtype.Binary, dtype.LargeBinary},
		{"a binary literal against fixed_size_binary", dtype.FixedSizeBinary{ByteWidth: 16}, dtype.Binary, dtype.FixedSizeBinary{ByteWidth: 16}},
		{"a string literal against binary", dtype.Binary, dtype.String, nil},

		// A bool column takes a bool and nothing else, which is the pandas
		// behavior this package does not want.
		{"an integer literal against bool", dtype.Bool, dtype.Int64, nil},
		{"a bool literal against int64", dtype.Int64, dtype.Bool, nil},

		{
			"a timestamp literal of another unit",
			dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"},
			dtype.Timestamp{Unit: dtype.Nanosecond},
			dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"},
		},
		{
			"a date literal against a timestamp column",
			dtype.Timestamp{Unit: dtype.Microsecond},
			dtype.Date32,
			nil,
		},
		{
			"an integer literal against a timestamp column",
			dtype.Timestamp{Unit: dtype.Microsecond},
			dtype.Int64,
			nil,
		},

		// The column keeps its encoding, so the comparison happens once per
		// distinct value rather than once per row.
		{
			"a string literal against a dictionary of strings",
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			dtype.String,
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
		},
		{
			"an integer literal against a dictionary of strings",
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			dtype.Int64,
			nil,
		},

		{
			"a list literal of null",
			dtype.List{Elem: dtype.Int64},
			dtype.List{Elem: dtype.Null},
			dtype.List{Elem: dtype.Int64},
		},
		{
			"a list literal of the wrong element",
			dtype.List{Elem: dtype.Int64},
			dtype.List{Elem: dtype.Float64},
			nil,
		},

		{"nil column", nil, dtype.Int64, nil},
		{"nil literal", dtype.Int64, nil, nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := dtype.CoerceLiteral(tt.column, tt.literal)
			if tt.want == nil {
				if err == nil {
					t.Fatalf("CoerceLiteral(%s, %s) = %s, want an error",
						name(tt.column), name(tt.literal), got)
				}
				if !strings.Contains(err.Error(), "cast the column") {
					t.Errorf("error %q does not say what to do about it", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("CoerceLiteral(%s, %s) = %v, want %s",
					name(tt.column), name(tt.literal), err, tt.want)
			}
			if !dtype.Equal(got, tt.want) {
				t.Errorf("CoerceLiteral(%s, %s) = %s, want %s",
					name(tt.column), name(tt.literal), got, tt.want)
			}
		})
	}
}

// TestCoerceLiteralKeepsTheColumn is the property that matters more than any
// single row of the table: whatever the literal was, the column's type comes
// out unchanged, so no expression silently rewrites the storage of a column.
func TestCoerceLiteralKeepsTheColumn(t *testing.T) {
	for _, column := range allTypes {
		if column.Kind() == dtype.NullKind {
			continue // a null column has no type of its own to keep
		}
		for _, literal := range allTypes {
			got, err := dtype.CoerceLiteral(column, literal)
			if err != nil {
				continue
			}
			if !dtype.Equal(got, column) {
				t.Errorf("CoerceLiteral(%s, %s) = %s, want the column type",
					column, literal, got)
			}
		}
	}
}

func TestCoerceLiteralTooDeep(t *testing.T) {
	var column, literal dtype.DataType
	column, literal = dtype.Int64, dtype.Null
	for range dtype.MaxNestingDepth + 2 {
		column = dtype.List{Elem: column}
		literal = dtype.List{Elem: literal}
	}

	if _, err := dtype.CoerceLiteral(column, literal); err == nil {
		t.Error("CoerceLiteral of a type nested past the limit returned no error")
	}
}
