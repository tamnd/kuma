package dtype_test

import (
	"fmt"

	"github.com/tamnd/kuma/dtype"
)

func ExampleSchema() {
	orders := dtype.Schema{
		Fields: []dtype.Field{
			{Name: "id", Type: dtype.Int64},
			{Name: "customer", Type: dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String}},
			{Name: "placed", Type: dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}},
			{Name: "total", Type: dtype.Decimal128{Precision: 18, Scale: 2}, Nullable: true},
		},
	}

	if err := orders.Validate(); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(orders)
	// Output:
	// schema<id: int64 not null, customer: dictionary<uint32, string> not null, placed: timestamp[us, tz=UTC] not null, total: decimal128(18, 2)>
}

func ExampleSchema_Select() {
	orders := dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64},
		{Name: "customer", Type: dtype.String},
		{Name: "total", Type: dtype.Float64},
	}}

	narrow, err := orders.Select("customer", "total")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(narrow.Names())

	// A name that is not there says so, and says what is.
	_, err = orders.Select("totl")
	fmt.Println(err)
	// Output:
	// [customer total]
	// dtype: no field named "totl", have id, customer, total
}

func ExampleEqual() {
	// Two timestamps of different resolutions are the same kind and not the
	// same type. Concatenating them without a cast would silently multiply
	// every value by a thousand.
	us := dtype.Timestamp{Unit: dtype.Microsecond}
	ns := dtype.Timestamp{Unit: dtype.Nanosecond}

	fmt.Println(us.Kind() == ns.Kind())
	fmt.Println(dtype.Equal(us, ns))
	// Output:
	// true
	// false
}

func ExampleValidate() {
	// A time of day in nanoseconds does not fit in 32 bits.
	fmt.Println(dtype.Validate(dtype.Time32{Unit: dtype.Nanosecond}))

	// The error names the part of the tree that is wrong.
	fmt.Println(dtype.Validate(dtype.List{Elem: dtype.Decimal128{Precision: 0}}))

	fmt.Println(dtype.Validate(dtype.List{Elem: dtype.Int64}))
	// Output:
	// dtype: time32 unit must be s or ms, have ns
	// dtype: list: decimal128 precision 0 out of range 1 to 38
	// <nil>
}

func ExampleCoerce() {
	// An int64 column and a float64 column have nothing in common that keeps
	// every value, so the caller is told to say which one they meant. pandas
	// would return float64 here and round the low bits off any id large enough
	// to matter.
	fmt.Println(dtype.Coerce(dtype.Int64, dtype.Float64))

	// A column read as all nulls, which is what an empty JSON array gives,
	// takes the type of whatever it is concatenated with.
	fmt.Println(dtype.Coerce(dtype.List{Elem: dtype.Null}, dtype.List{Elem: dtype.Int64}))

	// Dictionary encoding is how the values are stored and not what they are.
	fmt.Println(dtype.Coerce(dtype.Dictionary{Index: dtype.Uint32, Value: dtype.String}, dtype.String))
	// Output:
	// <nil> dtype: cannot combine int64 and float64, cast one side explicitly
	// list<int64> <nil>
	// string <nil>
}

func ExampleCoerceLiteral() {
	// Comparing a uint32 column against the literal 0 is not a type error and
	// does not widen the column.
	fmt.Println(dtype.CoerceLiteral(dtype.Uint32, dtype.Int64))

	// A float literal against an integer column is refused, because 1.5 has no
	// int64 spelling and rounding it quietly is the mistake this package is
	// trying not to make.
	fmt.Println(dtype.CoerceLiteral(dtype.Int64, dtype.Float64))
	// Output:
	// uint32 <nil>
	// <nil> dtype: cannot use a float64 literal with a int64 column, cast the column or write a int64 literal
}

func ExampleCanCast() {
	// A cast is what the caller writes when Coerce has refused, so it allows
	// much more. Whether a particular row survives is decided when the values
	// are read.
	fmt.Println(dtype.CanCast(dtype.Int64, dtype.Float64))
	fmt.Println(dtype.CanCast(dtype.String, dtype.Timestamp{Unit: dtype.Microsecond}))

	// A duration is a span and a timestamp is a point, and turning one into the
	// other needs an origin that nobody has stated.
	fmt.Println(dtype.CanCast(dtype.Duration{Unit: dtype.Second}, dtype.Timestamp{Unit: dtype.Second}))
	// Output:
	// true
	// true
	// false
}

func ExampleBits() {
	// Bool is one bit per value, not one byte, which anything sizing a buffer
	// has to know.
	for _, t := range []dtype.DataType{dtype.Bool, dtype.Int32, dtype.String} {
		bits, fixed := dtype.Bits(t)
		fmt.Println(t, bits, fixed)
	}
	// Output:
	// bool 1 true
	// int32 32 true
	// string 0 false
}
