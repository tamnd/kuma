package ipc_test

import (
	"fmt"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

func ExampleFormat() {
	types := []dtype.DataType{
		dtype.Int64,
		dtype.String,
		dtype.Timestamp{Unit: dtype.Microsecond, Zone: "Europe/London"},
		dtype.Decimal128{Precision: 18, Scale: 2},
		dtype.List{Elem: dtype.Int64},
	}
	for _, t := range types {
		format, err := ipc.Format(t)
		if err != nil {
			fmt.Println(err)
			continue
		}
		fmt.Printf("%-40s %s\n", t, format)
	}
	// Output:
	// int64                                    l
	// string                                   vu
	// timestamp[us, tz=Europe/London]          tsu:Europe/London
	// decimal128(18, 2)                        d:18,2
	// list<int64>                              +l
}

// A list is two calls, one for the element and one for the list, because the
// format string of a list says only that it is a list. An importer works its
// way up from the leaves, which is the order the C structs are laid out in.
func ExampleType() {
	elem, err := ipc.Type("l", nil)
	if err != nil {
		fmt.Println(err)
		return
	}
	list, err := ipc.Type("+l", []dtype.Field{{Name: "item", Type: elem, Nullable: true}})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(list)

	// The three text layouts all arrive as one type, since kuma stores text
	// one way and an import converts whatever it is given.
	for _, format := range []string{"u", "U", "vu"} {
		t, err := ipc.Type(format, nil)
		if err != nil {
			fmt.Println(err)
			continue
		}
		fmt.Printf("%s is %s\n", format, t)
	}
	// Output:
	// list<int64>
	// u is string
	// U is string
	// vu is string
}
