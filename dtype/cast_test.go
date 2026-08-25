package dtype_test

import (
	"testing"

	"github.com/tamnd/kuma/dtype"
)

func TestCanCast(t *testing.T) {
	tests := []struct {
		from dtype.DataType
		to   dtype.DataType
		want bool
	}{
		// Numbers convert to each other, and whether a particular value
		// survives the trip is checked a row at a time somewhere else.
		{dtype.Int64, dtype.Int8, true},
		{dtype.Int8, dtype.Int64, true},
		{dtype.Uint64, dtype.Int8, true},
		{dtype.Int64, dtype.Float64, true},
		{dtype.Float64, dtype.Int64, true},
		{dtype.Float32, dtype.Decimal128{Precision: 18, Scale: 2}, true},
		{dtype.Decimal128{Precision: 18, Scale: 2}, dtype.Decimal256{Precision: 50, Scale: 8}, true},
		{dtype.Decimal128{Precision: 18, Scale: 2}, dtype.Int64, true},

		// A number is nonzero or it is not, and a bool is one or zero. Both
		// directions are what every other engine does.
		{dtype.Int64, dtype.Bool, true},
		{dtype.Bool, dtype.Int64, true},
		{dtype.Bool, dtype.Float64, true},

		// Anything can be printed.
		{dtype.Int64, dtype.String, true},
		{dtype.Bool, dtype.LargeString, true},
		{dtype.Timestamp{Unit: dtype.Microsecond}, dtype.String, true},
		{dtype.Interval{Unit: dtype.MonthDayNano}, dtype.String, true},
		{dtype.List{Elem: dtype.Int64}, dtype.String, true},
		{dtype.Map{Key: dtype.String, Value: dtype.Int64}, dtype.String, true},

		// Text parses into anything, because a string column is what a CSV
		// reader produces before anyone has said what the columns hold.
		{dtype.String, dtype.Int64, true},
		{dtype.String, dtype.Bool, true},
		{dtype.String, dtype.Timestamp{Unit: dtype.Microsecond}, true},
		{dtype.String, dtype.Interval{Unit: dtype.DayTime}, true},
		{dtype.String, dtype.LargeString, true},
		{dtype.String, dtype.Binary, true},
		{dtype.LargeString, dtype.Int32, true},

		// Text does not become a list, a struct or a map. Parsing JSON is a
		// function with options, not a cast.
		{dtype.String, dtype.List{Elem: dtype.Int64}, false},
		{dtype.String, dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}}, false},
		{dtype.String, dtype.Map{Key: dtype.String, Value: dtype.Int64}, false},

		// Bytes and text are the same buffer with a question about encoding
		// over it. Bytes and numbers are not.
		{dtype.Binary, dtype.String, true},
		{dtype.Binary, dtype.LargeBinary, true},
		{dtype.Binary, dtype.FixedSizeBinary{ByteWidth: 16}, true},
		{dtype.FixedSizeBinary{ByteWidth: 16}, dtype.Binary, true},
		{dtype.Binary, dtype.Int64, false},
		{dtype.Int64, dtype.Binary, false},
		{dtype.Binary, dtype.Bool, false},
		{dtype.Bool, dtype.Binary, false},

		// The temporal types that are a count since an origin swap units and
		// swap origins, and the number underneath is reachable.
		{dtype.Timestamp{Unit: dtype.Nanosecond}, dtype.Timestamp{Unit: dtype.Second}, true},
		{dtype.Timestamp{Unit: dtype.Microsecond}, dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}, true},
		{dtype.Date32, dtype.Date64, true},
		{dtype.Date32, dtype.Timestamp{Unit: dtype.Microsecond}, true},
		{dtype.Timestamp{Unit: dtype.Microsecond}, dtype.Date32, true},
		{dtype.Date32, dtype.Int64, true},
		{dtype.Int64, dtype.Timestamp{Unit: dtype.Microsecond}, true},
		{dtype.Time32{Unit: dtype.Second}, dtype.Time64{Unit: dtype.Nanosecond}, true},
		{dtype.Time64{Unit: dtype.Nanosecond}, dtype.Int64, true},

		// A duration is a span and a timestamp is a point, so turning one into
		// the other needs an origin nobody has stated. A time of day is the
		// exception, since it is already a count from a midnight.
		{dtype.Duration{Unit: dtype.Nanosecond}, dtype.Timestamp{Unit: dtype.Nanosecond}, false},
		{dtype.Timestamp{Unit: dtype.Nanosecond}, dtype.Duration{Unit: dtype.Nanosecond}, false},
		{dtype.Duration{Unit: dtype.Nanosecond}, dtype.Date32, false},
		{dtype.Duration{Unit: dtype.Nanosecond}, dtype.Time64{Unit: dtype.Nanosecond}, true},
		{dtype.Time64{Unit: dtype.Nanosecond}, dtype.Duration{Unit: dtype.Nanosecond}, true},
		{dtype.Duration{Unit: dtype.Second}, dtype.Duration{Unit: dtype.Nanosecond}, true},
		{dtype.Duration{Unit: dtype.Second}, dtype.Int64, true},

		// A time is not a truth value and it is not an opaque blob either.
		{dtype.Timestamp{Unit: dtype.Microsecond}, dtype.Bool, false},
		{dtype.Date32, dtype.Binary, false},

		// An interval is three counts and no single number is one of them, so
		// it only trades with another interval.
		{dtype.Interval{Unit: dtype.YearMonth}, dtype.Interval{Unit: dtype.MonthDayNano}, true},
		{dtype.Interval{Unit: dtype.YearMonth}, dtype.Int64, false},
		{dtype.Int64, dtype.Interval{Unit: dtype.YearMonth}, false},
		{dtype.Interval{Unit: dtype.DayTime}, dtype.Duration{Unit: dtype.Millisecond}, false},
		{dtype.Duration{Unit: dtype.Millisecond}, dtype.Interval{Unit: dtype.DayTime}, false},
		{dtype.Interval{Unit: dtype.YearMonth}, dtype.Timestamp{Unit: dtype.Second}, false},

		// The three list shaped types differ in how rows are delimited, so a
		// cast between any two of them is a cast of the elements.
		{dtype.List{Elem: dtype.Int64}, dtype.List{Elem: dtype.String}, true},
		{dtype.List{Elem: dtype.Int64}, dtype.LargeList{Elem: dtype.Int32}, true},
		{dtype.List{Elem: dtype.Int64}, dtype.FixedSizeList{Elem: dtype.Int64, Len: 3}, true},
		{dtype.FixedSizeList{Elem: dtype.Float32, Len: 3}, dtype.List{Elem: dtype.Float64}, true},
		{dtype.List{Elem: dtype.Int64}, dtype.List{Elem: dtype.Interval{Unit: dtype.DayTime}}, false},
		{dtype.List{Elem: dtype.Int64}, dtype.Map{Key: dtype.String, Value: dtype.Int64}, false},
		{dtype.List{Elem: dtype.Int64}, dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}}, false},

		// A struct casts field by field, and the names have to line up so that
		// no column's values quietly land in a different field.
		{
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.String}}},
			true,
		},
		{
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Struct{Fields: []dtype.Field{{Name: "b", Type: dtype.Int64}}},
			false,
		},
		{
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Struct{Fields: []dtype.Field{
				{Name: "a", Type: dtype.Int64},
				{Name: "b", Type: dtype.Int64},
			}},
			false,
		},
		{
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Interval{Unit: dtype.DayTime}}}},
			false,
		},
		// Nullability is not a cast. A field that says it holds no nulls and
		// one that says it might are the same values either way, and whether
		// there is actually a null in there is a question about the data.
		{
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64, Nullable: true}}},
			true,
		},

		{
			dtype.Map{Key: dtype.String, Value: dtype.Int64},
			dtype.Map{Key: dtype.String, Value: dtype.Float64},
			true,
		},
		{
			dtype.Map{Key: dtype.String, Value: dtype.Int64},
			dtype.Map{Key: dtype.Int64, Value: dtype.Int64},
			true,
		},
		{
			dtype.Map{Key: dtype.String, Value: dtype.Int64},
			dtype.Map{Key: dtype.String, Value: dtype.Interval{Unit: dtype.DayTime}},
			false,
		},
		{
			dtype.Map{Key: dtype.String, Value: dtype.Int64},
			dtype.List{Elem: dtype.Int64},
			false,
		},

		// Dictionary encoding is storage, so it is peeled off one side and put
		// back on the other and the values decide.
		{dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String}, dtype.String, true},
		{dtype.String, dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String}, true},
		{dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String}, dtype.Int64, true},
		{
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String},
			dtype.Dictionary{Index: dtype.Uint8, Value: dtype.Int64},
			true,
		},
		{
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.Int64},
			dtype.Interval{Unit: dtype.DayTime},
			false,
		},
		{
			dtype.Interval{Unit: dtype.DayTime},
			dtype.Dictionary{Index: dtype.Uint32, Value: dtype.Int64},
			false,
		},

		// A column of nothing becomes a column of nothing in the new type, and
		// casting to null is throwing the values away on purpose.
		{dtype.Null, dtype.Int64, true},
		{dtype.Int64, dtype.Null, true},
		{dtype.Null, dtype.Struct{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}}, true},

		{nil, dtype.Int64, false},
		{dtype.Int64, nil, false},
		{nil, nil, false},
	}

	for _, tt := range tests {
		if got := dtype.CanCast(tt.from, tt.to); got != tt.want {
			t.Errorf("CanCast(%s, %s) = %v, want %v",
				name(tt.from), name(tt.to), got, tt.want)
		}
	}
}

func TestCanCastToItself(t *testing.T) {
	for _, typ := range allTypes {
		if !dtype.CanCast(typ, typ) {
			t.Errorf("CanCast(%s, %s) = false, want true", typ, typ)
		}
	}
}

func TestCanCastToString(t *testing.T) {
	// Printing is the one cast that is always available, and a good deal of
	// debugging depends on it being available without conditions.
	for _, typ := range allTypes {
		if !dtype.CanCast(typ, dtype.String) {
			t.Errorf("CanCast(%s, string) = false, want true", typ)
		}
		if !dtype.CanCast(typ, dtype.LargeString) {
			t.Errorf("CanCast(%s, large_string) = false, want true", typ)
		}
	}
}

func TestCanCastNull(t *testing.T) {
	for _, typ := range allTypes {
		if !dtype.CanCast(dtype.Null, typ) {
			t.Errorf("CanCast(null, %s) = false, want true", typ)
		}
		if !dtype.CanCast(typ, dtype.Null) {
			t.Errorf("CanCast(%s, null) = false, want true", typ)
		}
	}
}

func TestCanCastWhateverCoerceAllows(t *testing.T) {
	// Coerce is the strict half and CanCast is the loose half, and the two
	// have to agree in one direction: if two columns combine into a type
	// without anyone asking, both of them have to be castable to it. A pair
	// that fails this is a plan that type checks and then cannot be run.
	for _, a := range allTypes {
		for _, b := range allTypes {
			want, err := dtype.Coerce(a, b)
			if err != nil {
				continue
			}
			if !dtype.CanCast(a, want) {
				t.Errorf("Coerce(%s, %s) = %s, but CanCast(%s, %s) is false", a, b, want, a, want)
			}
			if !dtype.CanCast(b, want) {
				t.Errorf("Coerce(%s, %s) = %s, but CanCast(%s, %s) is false", a, b, want, b, want)
			}
		}
	}
}

func TestCanCastTooDeep(t *testing.T) {
	// Two different types, so that the walk actually descends rather than
	// stopping at the first Equal.
	var deepFrom, deepTo dtype.DataType
	deepFrom, deepTo = dtype.Int64, dtype.Int32
	for range dtype.MaxNestingDepth + 1 {
		deepFrom = dtype.List{Elem: deepFrom}
		deepTo = dtype.List{Elem: deepTo}
	}
	if dtype.CanCast(deepFrom, deepTo) {
		t.Error("CanCast past the nesting limit = true, want false")
	}

	var shallowFrom, shallowTo dtype.DataType
	shallowFrom, shallowTo = dtype.Int64, dtype.Int32
	for range dtype.MaxNestingDepth - 2 {
		shallowFrom = dtype.List{Elem: shallowFrom}
		shallowTo = dtype.List{Elem: shallowTo}
	}
	if !dtype.CanCast(shallowFrom, shallowTo) {
		t.Errorf("CanCast at depth %d = false, want true", dtype.MaxNestingDepth-2)
	}
}

func TestCanCastForeignType(t *testing.T) {
	// DataType is exported, so a type from another package can claim a kind and
	// not be the struct that kind usually means. Nothing here may assert
	// without checking.
	var foreign dtype.DataType = odd{}
	if dtype.CanCast(foreign, dtype.Int64) {
		t.Error("CanCast(foreign list, int64) = true, want false")
	}
	if dtype.CanCast(dtype.Int64, foreign) {
		t.Error("CanCast(int64, foreign list) = true, want false")
	}
	if !dtype.CanCast(foreign, dtype.String) {
		t.Error("CanCast(foreign list, string) = false, want true")
	}

	// A type that reports a kind this package has never heard of is not
	// something any kernel knows how to read, so nothing casts to it or from
	// it, printing included.
	var alien dtype.DataType = unknown{}
	if dtype.CanCast(alien, dtype.String) {
		t.Error("CanCast(unknown, string) = true, want false")
	}
	if dtype.CanCast(dtype.String, alien) {
		t.Error("CanCast(string, unknown) = true, want false")
	}
}
