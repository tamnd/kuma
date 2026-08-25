package strview_test

import (
	"fmt"
	"slices"

	"github.com/tamnd/kuma/strview"
)

func ExampleBuilder() {
	var b strview.Builder
	b.AppendString("kuma")
	b.AppendString("a value that is too long to live inside its view")

	d := b.Finish()
	for i := range d.Len() {
		fmt.Printf("%d bytes, inline %v: %s\n", d.View(i).Len(), d.View(i).IsInline(), d.At(i))
	}
	// Output:
	// 4 bytes, inline true: kuma
	// 48 bytes, inline false: a value that is too long to live inside its view
}

// ExampleView shows the layout of a short value. The first four bytes are the
// length and the rest is the value, zero padded, so nothing else has to be read
// to get it back.
func ExampleView() {
	v := strview.MakeInline([]byte("kuma"))
	fmt.Printf("%v\n", v[:])
	fmt.Println(v)
	// Output:
	// [4 0 0 0 107 117 109 97 0 0 0 0 0 0 0 0]
	// inline("kuma")
}

// ExampleView_prefix shows the other half of the layout. A long value keeps its
// first four bytes in the view, in the same place a short value keeps the start
// of its own, so a comparison that is settled in the first four bytes never
// reaches a data block.
func ExampleView_prefix() {
	var b strview.Builder
	b.AppendString("kuma")
	b.AppendString("kumamoto prefecture, in the south")
	d := b.Finish()

	for i := range d.Len() {
		p := d.View(i).Prefix()
		fmt.Printf("%q\n", p[:])
	}
	// Output:
	// "kuma"
	// "kuma"
}

func ExampleData_Compare() {
	var b strview.Builder
	for _, s := range []string{"pear", "apple", "a rather longer name than the others", "banana"} {
		b.AppendString(s)
	}
	d := b.Finish()

	order := make([]int, d.Len())
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, d.Compare)

	for _, i := range order {
		fmt.Println(string(d.At(i)))
	}
	// Output:
	// a rather longer name than the others
	// apple
	// banana
	// pear
}

func ExampleData_EqualValue() {
	var b strview.Builder
	b.AppendString("tokyo")
	b.AppendString("kumamoto")
	d := b.Finish()

	for i := range d.Len() {
		fmt.Println(string(d.At(i)), d.EqualValue(i, []byte("kumamoto")))
	}
	// Output:
	// tokyo false
	// kumamoto true
}

// ExampleNewData opens a column that was built somewhere else, which is what
// receiving one over Arrow IPC amounts to. The views are checked against the
// blocks before anything is read, so a view that points past the end of its
// block is an error here rather than a wrong answer later.
func ExampleNewData() {
	var b strview.Builder
	b.AppendString("a value that does not fit inline")
	d := b.Finish()

	views, blocks := d.Views(), d.Blocks()
	views[0][12] = 0xff // move the value's offset out past the end of its block

	if _, err := strview.NewData(views, blocks); err != nil {
		fmt.Println(err)
	}
	// Output:
	// strview: view 0: wants bytes 255 to 287 of a block that is 32 long
}
