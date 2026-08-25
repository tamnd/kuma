package buffer_test

import (
	"fmt"

	"github.com/tamnd/kuma/buffer"
)

func ExampleNew() {
	b := buffer.New(8)
	copy(b.Bytes(), "kuma")

	fmt.Println(b.Len(), b.Cap(), b.Aligned())
	fmt.Printf("%q\n", b.Bytes())
	// Output:
	// 8 64 true
	// "kuma\x00\x00\x00\x00"
}

func ExampleBuffer_Append() {
	var b buffer.Buffer
	b.Append([]byte("hello "))
	b.Append([]byte("world"))

	fmt.Println(string(b.Bytes()))
	// Output: hello world
}

// ExampleBuffer_Padded shows the property the padding exists for. A buffer
// holding one byte of data still hands a kernel a full block to load, so the
// last elements of a column do not need a separate scalar loop.
func ExampleBuffer_Padded() {
	b := buffer.New(1)

	fmt.Println(len(b.Bytes()), len(b.Padded()))
	// Output: 1 64
}

func ExampleBuffer_Resize() {
	b := buffer.New(4)
	copy(b.Bytes(), "kuma")

	b.Resize(2)
	fmt.Printf("%q\n", b.Bytes())

	// Growing back does not bring the old bytes with it.
	b.Resize(4)
	fmt.Printf("%q\n", b.Bytes())
	// Output:
	// "ku"
	// "ku\x00\x00"
}

// ExamplePool is the shape an operator uses: take a buffer, use it, give it
// back before returning. The defer is what makes the lifetime obvious to
// somebody reading the code later, which is the condition for using a pool at
// all.
func ExamplePool() {
	var pool buffer.Pool

	scratch := pool.Get(1024)
	defer pool.Put(scratch)

	copy(scratch.Bytes(), "working space")
	fmt.Println(scratch.Len(), scratch.Cap())
	// Output: 1024 1024
}
