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
