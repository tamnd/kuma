package kernel_test

import (
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// FuzzTake builds a column out of the first input and a list of positions out
// of the second, and checks the one property a gather has: value k of the
// result is the value that was at position idx[k].
//
// The fuzzer is choosing the chunk boundaries and where the nulls are, which is
// the part a hand written test gets tired of varying and the part a gather gets
// wrong.
func FuzzTake(f *testing.F) {
	f.Add([]byte{3, 0, 4}, []byte{1, 0, 2, 255})
	f.Add([]byte{1, 1, 1, 1}, []byte{3, 2, 1, 0})
	f.Add([]byte{}, []byte{255, 255})
	f.Add([]byte{8}, []byte{})

	f.Fuzz(func(t *testing.T, layout, positions []byte) {
		if len(layout) > 32 || len(positions) > 256 {
			t.Skip("a big input proves nothing a small one does not")
		}

		src := column(t, layout)
		idx := make([]int, 0, len(positions))
		for _, p := range positions {
			// A quarter of the byte range asks for a null, which is enough of
			// them to land next to every other case.
			if p >= 192 || src.Len() == 0 {
				idx = append(idx, -1)
				continue
			}
			idx = append(idx, int(p)%src.Len())
		}

		checkTake(t, kernel.Take(src, idx), src, idx)
	})
}

// FuzzFilter checks a filter against a walk over the mask, which is the
// definition of what a filter does rather than another way of writing the
// implementation.
func FuzzFilter(f *testing.F) {
	f.Add([]byte{3, 0, 4}, []byte{1, 2, 3})
	f.Add([]byte{5}, []byte{0})
	f.Add([]byte{2, 2}, []byte{255, 255})

	f.Fuzz(func(t *testing.T, layout, choices []byte) {
		if len(layout) > 32 || len(choices) > 32 {
			t.Skip("a big input proves nothing a small one does not")
		}

		src := column(t, layout)
		mask := boolColumn(t, choices, src.Len())

		var want []any
		for i := range mask.Len() {
			if !mask.IsNull(i) && mask.Bool(i) {
				want = append(want, valueAt(t, src, i))
			}
		}

		got := kernel.Filter(src, mask)
		if got.Len() != len(want) {
			t.Fatalf("the filter kept %d values, want %d", got.Len(), len(want))
		}
		for k, v := range want {
			if have := valueAt(t, got, k); have != v {
				t.Errorf("value %d is %v, want %v", k, have, v)
			}
		}
	})
}

// column builds an int64 column whose chunk lengths and nulls come from the
// bytes given, so that the fuzzer decides the shape.
func column(t *testing.T, layout []byte) *array.Chunked {
	t.Helper()

	b, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}

	var chunks []*array.Array
	next := int64(0)
	for _, size := range layout {
		for i := range int(size % 9) {
			if (int(size)+i)%3 == 0 {
				b.AppendNull()
			} else {
				b.Append(next)
			}
			next++
		}
		chunks = append(chunks, b.Finish())
	}

	c, err := array.NewChunked(dtype.Int64, chunks...)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	return c
}

// boolColumn builds a mask of n values out of the bytes given, in as many
// chunks as it takes, with a null wherever the byte is odd.
func boolColumn(t *testing.T, choices []byte, n int) *array.Chunked {
	t.Helper()

	b, err := array.NewBuilder(dtype.Bool)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}

	var chunks []*array.Array
	for i := range n {
		c := byte(0)
		if len(choices) > 0 {
			c = choices[i%len(choices)]
		}
		if c%4 == 1 {
			b.AppendNull()
		} else {
			b.AppendBool(c&2 != 0)
		}

		// A chunk boundary every few values, so that the mask is chunked
		// differently from the column it is filtering.
		if i%5 == 4 {
			chunks = append(chunks, b.Finish())
		}
	}
	chunks = append(chunks, b.Finish())

	mask, err := array.NewChunked(dtype.Bool, chunks...)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	return mask
}
