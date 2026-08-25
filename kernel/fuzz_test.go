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

// FuzzCastNumber casts an int64 column to a numeric type the fuzzer picks, and
// checks the two properties a cast has.
//
// Nothing is dropped: the result is as long as the source and a null stays a
// null. Nothing is changed: a value that arrives becomes the same value when it
// is cast back, which is the whole promise for a destination it fits in.
//
// The loose cast is the one under test because it is the one that has to answer
// for every value rather than stopping at the first awkward one, so a run gets
// through the whole column instead of finding the same first row every time.
func FuzzCastNumber(f *testing.F) {
	f.Add([]byte{3, 0, 4}, uint8(0))
	f.Add([]byte{5}, uint8(9))
	f.Add([]byte{2, 2}, uint8(4))

	f.Fuzz(func(t *testing.T, layout []byte, pick uint8) {
		if len(layout) > 32 {
			t.Skip("a big input proves nothing a small one does not")
		}

		to := castTargets[int(pick)%len(castTargets)]
		src := column(t, layout)

		got, err := kernel.TryCast(src, to)
		if err != nil {
			t.Fatalf("TryCast to %s: %v", to, err)
		}
		if got.Len() != src.Len() {
			t.Fatalf("the result has %d values, want %d", got.Len(), src.Len())
		}

		// Back the way it came. A float32 loses digits on the way out and does
		// not come home, so the round trip is only asked of the types that hold
		// an int64 exactly.
		if to.Kind() == dtype.Float32Kind {
			return
		}
		back, err := kernel.TryCast(got, dtype.Int64)
		if err != nil {
			t.Fatalf("TryCast back to int64: %v", err)
		}

		for i := range src.Len() {
			if src.IsNull(i) {
				if !got.IsNull(i) {
					t.Fatalf("value %d was missing and came back as %v", i, valueAt(t, got, i))
				}
				continue
			}
			if got.IsNull(i) {
				// The value did not fit, which is allowed, and the only way
				// that can be true of an int64 is a narrower destination.
				continue
			}
			if have, want := valueAt(t, back, i), valueAt(t, src, i); have != want {
				t.Errorf("value %d went to %s and came back as %v, want %v", i, to, have, want)
			}
		}
	})
}

// castTargets is what FuzzCastNumber picks from. Every one of them holds at
// least some int64 values exactly, so a value that survives the trip out has to
// survive the trip home.
var castTargets = []dtype.DataType{
	dtype.Int8, dtype.Int16, dtype.Int32, dtype.Int64,
	dtype.Uint8, dtype.Uint16, dtype.Uint32, dtype.Uint64,
	dtype.Float32, dtype.Float64,
	dtype.Timestamp{Unit: dtype.Nanosecond},
}

// FuzzCastText writes numbers out as text and reads them back, which is the
// round trip a CSV file is and the one where a lost digit is a wrong number
// rather than a parse error.
func FuzzCastText(f *testing.F) {
	f.Add([]byte{3, 0, 4})
	f.Add([]byte{7})
	f.Add([]byte{})

	f.Fuzz(func(t *testing.T, layout []byte) {
		if len(layout) > 32 {
			t.Skip("a big input proves nothing a small one does not")
		}

		src := column(t, layout)

		text, err := kernel.Cast(src, dtype.String)
		if err != nil {
			t.Fatalf("Cast to text: %v", err)
		}
		back, err := kernel.Cast(text, dtype.Int64)
		if err != nil {
			t.Fatalf("Cast back from text: %v", err)
		}

		for i := range src.Len() {
			if have, want := valueAt(t, back, i), valueAt(t, src, i); have != want {
				t.Errorf("value %d printed as %v and read back as %v, want %v",
					i, valueAt(t, text, i), have, want)
			}
		}
	})
}

// FuzzSort checks the three things a sorted order has to be, whatever the
// values and however the column is cut into chunks: a permutation of the
// positions, non decreasing in the values, and with the missing values in one
// block at the end.
func FuzzSort(f *testing.F) {
	f.Add([]byte{3, 1, 2}, false, false)
	f.Add([]byte{9, 9, 9, 9}, true, false)
	f.Add([]byte{}, false, true)
	f.Add([]byte{200, 0, 7, 255, 1, 1}, true, true)

	f.Fuzz(func(t *testing.T, values []byte, descending, nullsFirst bool) {
		if len(values) > 64 {
			t.Skip("a big input proves nothing a small one does not")
		}

		src := byteColumn(t, values)
		o := kernel.Order{Column: src, Descending: descending, NullsFirst: nullsFirst}

		idx, err := kernel.SortIndex(o)
		if err != nil {
			t.Fatalf("SortIndex: %v", err)
		}
		if len(idx) != src.Len() {
			t.Fatalf("the order has %d positions, the column has %d values", len(idx), src.Len())
		}

		seen := make([]bool, len(idx))
		for _, i := range idx {
			if seen[i] {
				t.Fatalf("position %d appears twice", i)
			}
			seen[i] = true
		}

		// Walking the result, a value never comes before one it should come
		// after, and a null never appears in the middle of the values when it
		// was asked to go at the end.
		wasNull := false
		var last int64
		first := true
		for k, i := range idx {
			if src.IsNull(i) {
				if !nullsFirst {
					wasNull = true
					continue
				}
				if !first {
					t.Fatalf("a null is at position %d, after a value, with nulls first", k)
				}
				continue
			}
			if wasNull {
				t.Fatalf("a value is at position %d, after a null, with nulls last", k)
			}

			v := src.Value[int64](i)
			if !first {
				if !descending && v < last {
					t.Fatalf("value %d is %d, after %d, ascending", k, v, last)
				}
				if descending && v > last {
					t.Fatalf("value %d is %d, after %d, descending", k, v, last)
				}
			}
			last, first = v, false
		}
	})
}

// byteColumn builds an int64 column out of the bytes given, one value per byte,
// with a null wherever the byte is 255, in as many chunks as it takes.
//
// The values repeat, which is the point: a sort of values that are all
// different never exercises a tie, and the ties are where the stability and the
// null placement live.
func byteColumn(t *testing.T, values []byte) *array.Chunked {
	t.Helper()

	b, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}

	var chunks []*array.Array
	for i, v := range values {
		if v == 255 {
			b.AppendNull()
		} else {
			b.Append(int64(v) - 128)
		}
		// A chunk boundary every so often, at a place the input decides, so
		// that a value and the boundary next to it are fuzzed together.
		if i%7 == 6 {
			chunks = append(chunks, b.Finish())
		}
	}
	chunks = append(chunks, b.Finish())

	c, err := array.NewChunked(dtype.Int64, chunks...)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	return c
}
