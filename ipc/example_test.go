package ipc_test

import (
	"encoding/binary"
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

// A column of strings from another library usually arrives in the offset
// layout, which kuma does not store. Import converts it to views, which is
// sixteen bytes per value, and leaves the text itself where it is.
func ExampleImport() {
	// What pyarrow would hand over for the three values below: the offsets of
	// each value in the data buffer, and the data buffer.
	offsets := []byte{}
	for _, n := range []uint32{0, 3, 6, 12} {
		offsets = binary.NativeEndian.AppendUint32(offsets, n)
	}
	data := []byte("onetwothree!")

	a, err := ipc.Import("u", ipc.Layout{
		Length:  3,
		Buffers: [][]byte{nil, offsets, data},
	})
	if err != nil {
		fmt.Println(err)
		return
	}

	for i := range a.Len() {
		fmt.Printf("%d %s\n", i, a.Bytes(i))
	}

	// Going back out is the layout kuma stores, so the format string changes
	// and nothing is copied.
	format, err := ipc.Format(a.DType())
	if err != nil {
		fmt.Println(err)
		return
	}
	l, err := ipc.Export(a)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Printf("%s in %d buffers\n", format, len(l.Buffers))
	// Output:
	// 0 one
	// 1 two
	// 2 three!
	// vu in 2 buffers
}
