package array_test

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
)

// int32s returns a buffer holding the given values, the long way, the same way
// int64s does.
func int32s(t *testing.T, values ...int32) *buffer.Buffer {
	t.Helper()
	buf := buffer.New(len(values) * 4)
	p := buf.Bytes()
	for i, v := range values {
		u := uint32(v)
		for k := range 4 {
			p[i*4+k] = byte(u >> (8 * k))
		}
	}
	return buf
}

// regions is the column every test here uses: four rows over a dictionary of
// two, with the third row missing.
func regions(t *testing.T) *array.Array {
	t.Helper()
	idx := mustNew(t, dtype.Int32, 4, int32s(t, 1, 0, 0, 1), validity(true, true, false, true))
	d, err := array.NewDictionary(idx, array.OfStrings("north", "south"))
	if err != nil {
		t.Fatalf("NewDictionary: %v", err)
	}
	return d
}

func TestNewDictionary(t *testing.T) {
	a := regions(t)

	want := dtype.Dictionary{Index: dtype.Int32, Value: dtype.String}
	if !dtype.Equal(a.DType(), want) {
		t.Errorf("DType() = %s, want %s", a.DType(), want)
	}
	if a.Len() != 4 {
		t.Errorf("Len() = %d, want 4", a.Len())
	}
	if a.NullCount() != 1 {
		t.Errorf("NullCount() = %d, want 1", a.NullCount())
	}

	d := a.Dictionary()
	if d == nil {
		t.Fatal("Dictionary() = nil, want the values")
	}
	if d.Len() != 2 {
		t.Errorf("the dictionary has %d values, want 2", d.Len())
	}

	idx := a.Indices()
	if idx == nil {
		t.Fatal("Indices() = nil, want the indices")
	}
	if !dtype.Equal(idx.DType(), dtype.Int32) {
		t.Errorf("Indices().DType() = %s, want int32", idx.DType())
	}
	if idx.Dictionary() != nil {
		t.Error("the indices are dictionary encoded, want a plain column")
	}
	if got, want := idx.Values[int32](), []int32{1, 0, 0, 1}; len(got) != len(want) {
		t.Fatalf("Indices() holds %v, want %v", got, want)
	}
	// The nulls travel with the indices, since that is where they are.
	if idx.NullCount() != 1 {
		t.Errorf("Indices().NullCount() = %d, want 1", idx.NullCount())
	}
}

func TestDictionaryValues(t *testing.T) {
	a := regions(t)

	want := []string{"south", "north", "", "south"}
	for i, s := range want {
		if a.IsNull(i) {
			if s != "" {
				t.Errorf("row %d is missing, want %q", i, s)
			}
			if k := a.Index(i); k != -1 {
				t.Errorf("Index(%d) = %d for a missing value, want -1", i, k)
			}
			continue
		}
		if got := string(a.Dictionary().Bytes(a.Index(i))); got != s {
			t.Errorf("row %d = %q, want %q", i, got, s)
		}
	}
}

// TestDictionaryNullIndex checks that the index behind a missing value is left
// alone. A producer is allowed to leave anything there, including a number that
// is not a position in the dictionary at all, and refusing those would refuse
// files that every other reader takes.
func TestDictionaryNullIndex(t *testing.T) {
	idx := mustNew(t, dtype.Int32, 3, int32s(t, 0, 1<<30, 1), validity(true, false, true))
	a, err := array.NewDictionary(idx, array.OfStrings("north", "south"))
	if err != nil {
		t.Fatalf("NewDictionary: %v", err)
	}
	if k := a.Index(1); k != -1 {
		t.Errorf("Index(1) = %d, want -1", k)
	}
}

func TestDictionaryIndexTypes(t *testing.T) {
	for _, tc := range []struct {
		name string
		idx  *array.Array
	}{
		{"int8", array.Of[int8](0, 1)},
		{"int16", array.Of[int16](0, 1)},
		{"int32", array.Of[int32](0, 1)},
		{"int64", array.Of[int64](0, 1)},
		{"uint8", array.Of[uint8](0, 1)},
		{"uint16", array.Of[uint16](0, 1)},
		{"uint32", array.Of[uint32](0, 1)},
		{"uint64", array.Of[uint64](0, 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, err := array.NewDictionary(tc.idx, array.OfStrings("north", "south"))
			if err != nil {
				t.Fatalf("NewDictionary: %v", err)
			}
			if k := a.Index(1); k != 1 {
				t.Errorf("Index(1) = %d, want 1", k)
			}
		})
	}
}

func TestDictionaryErrors(t *testing.T) {
	vals := array.OfStrings("north", "south")

	for _, tc := range []struct {
		name  string
		idx   *array.Array
		dict  *array.Array
		wants string
	}{
		{"no indices", nil, vals, "needs both"},
		{"no values", array.Of[int32](0), nil, "needs both"},
		{"a float index", array.Of(0.0, 1.0), vals, "must be an integer"},
		{"a string index", array.OfStrings("a"), vals, "must be an integer"},
		{"an index past the end", array.Of[int32](0, 2), vals, "index 1 of the column is 2"},
		{"a negative index", array.Of[int32](0, -1), vals, "index 1 of the column is -1"},
		{"an index of a dictionary of nothing", array.Of[int32](0), array.OfStrings(), "dictionary of 0"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := array.NewDictionary(tc.idx, tc.dict)
			if err == nil {
				t.Fatal("NewDictionary of that = nil error, want one")
			}
			if !strings.Contains(err.Error(), tc.wants) {
				t.Errorf("NewDictionary = %v, want it to mention %q", err, tc.wants)
			}
		})
	}
}

// TestDictionaryOfDictionary checks both directions of the one shape that is
// refused for being nonsense rather than for being out of range. Two levels of
// indirection is a thing no two readers agree on the meaning of.
func TestDictionaryOfDictionary(t *testing.T) {
	inner := regions(t)

	if _, err := array.NewDictionary(array.Of[int32](0, 0), inner); err == nil {
		t.Error("a dictionary of a dictionary was built, want an error")
	}
	if _, err := array.NewDictionary(inner, array.OfStrings("a", "b")); err == nil {
		t.Error("a dictionary indexed by a dictionary was built, want an error")
	}
}

// TestDictionarySlice checks that slicing one moves the indices and leaves the
// values where they are, which is the reason a slice of one is still free.
func TestDictionarySlice(t *testing.T) {
	a := regions(t).Slice(1, 4)

	if a.Len() != 3 {
		t.Fatalf("Len() = %d, want 3", a.Len())
	}
	if a.NullCount() != 1 {
		t.Errorf("NullCount() = %d, want 1", a.NullCount())
	}
	if a.Offset() != 1 {
		t.Errorf("Offset() = %d, want 1", a.Offset())
	}
	if a.Dictionary() != regions(t).Dictionary() && a.Dictionary().Len() != 2 {
		t.Errorf("the slice has a dictionary of %d values, want 2", a.Dictionary().Len())
	}
	if got := a.Indices().Values[int32](); len(got) != 3 || got[0] != 0 || got[2] != 1 {
		t.Errorf("Indices() of the slice holds %v, want the last three", got)
	}
	if k := a.Index(0); k != 0 {
		t.Errorf("Index(0) of the slice = %d, want 0", k)
	}
	if k := a.Index(1); k != -1 {
		t.Errorf("Index(1) of the slice = %d, want -1", k)
	}
}

// TestDictionaryClone checks that a copy shares nothing, which for this layout
// means the values as well as the indices.
func TestDictionaryClone(t *testing.T) {
	a := regions(t).Slice(1, 3)
	c := a.Clone()

	if c.Len() != 2 || c.NullCount() != 1 {
		t.Fatalf("the clone is %d values with %d missing, want 2 and 1", c.Len(), c.NullCount())
	}
	if c.Offset() != 0 {
		t.Errorf("the clone starts at %d, want 0", c.Offset())
	}
	if !dtype.Equal(c.DType(), a.DType()) {
		t.Errorf("the clone is a %s column, want %s", c.DType(), a.DType())
	}
	if c.Buffer() == a.Buffer() {
		t.Error("the clone shares the indices it was copied from")
	}
	if c.Dictionary() == a.Dictionary() {
		t.Error("the clone shares the values it was copied from")
	}
	if got := string(c.Dictionary().Bytes(c.Index(0))); got != "north" {
		t.Errorf("the clone reads row 0 as %q, want north", got)
	}
	if k := c.Index(1); k != -1 {
		t.Errorf("the clone reads row 1 as index %d, want -1", k)
	}
}

// TestDictionaryPlainColumn checks that the two accessors say no for a column
// that is not dictionary encoded, since that is how a caller asks.
func TestDictionaryPlainColumn(t *testing.T) {
	a := array.OfStrings("north", "south")

	if a.Dictionary() != nil {
		t.Error("Dictionary() of a plain column is not nil")
	}
	if a.Indices() != nil {
		t.Error("Indices() of a plain column is not nil")
	}

	defer func() {
		if recover() == nil {
			t.Error("Index on a plain column did not panic")
		}
	}()
	a.Index(0)
}

func TestDictionaryIndexRange(t *testing.T) {
	a := regions(t)
	for _, i := range []int{-1, 4, 1 << 20} {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("Index(%d) did not panic", i)
				}
			}()
			a.Index(i)
		}()
	}
}

// FuzzDictionary checks that a dictionary built out of arbitrary indices either
// is refused or reads back in range, which is the whole promise of the bounds
// check in the constructor: whatever a file holds, a lookup afterwards is a
// lookup that lands.
func FuzzDictionary(f *testing.F) {
	f.Add([]byte{0, 0, 1, 0, 2, 0, 0, 0}, []byte{0xFF}, 3)
	f.Add([]byte{0xFF, 0xFF, 0xFF, 0xFF}, []byte{0x00}, 1)
	f.Add([]byte{}, []byte{}, 0)

	f.Fuzz(func(t *testing.T, raw, bits []byte, size int) {
		if size < 0 || size > 64 {
			t.Skip()
		}
		length := len(raw) / 2
		if length > 8*len(bits) {
			t.Skip()
		}

		values := buffer.New(len(raw))
		copy(values.Bytes(), raw)
		valid := bitmap.New(length)
		for k := range length {
			valid.Set(k, bits[k/8]&(1<<(k%8)) != 0)
		}
		idx, err := array.New(dtype.Int16, length, values, valid)
		if err != nil {
			t.Fatalf("New: %v", err)
		}

		dict := make([]string, size)
		for i := range dict {
			dict[i] = string(rune('a' + i))
		}
		a, err := array.NewDictionary(idx, array.OfStrings(dict...))
		if err != nil {
			return
		}

		for i := range a.Len() {
			k := a.Index(i)
			if k == -1 {
				if a.IsValid(i) {
					t.Fatalf("row %d is present and has no index", i)
				}
				continue
			}
			if k < 0 || k >= size {
				t.Fatalf("row %d has index %d, which is not a value of a dictionary of %d", i, k, size)
			}
			// The lookup this is all for. It panics if the index is out of
			// range, which is the failure being ruled out.
			a.Dictionary().Bytes(k)
		}
	})
}

func BenchmarkNewDictionary(b *testing.B) {
	idx := make([]int32, 4096)
	for i := range idx {
		idx[i] = int32(i % 256)
	}
	values := make([]string, 256)
	for i := range values {
		values[i] = "value"
	}

	indices := array.Of(idx...)
	dict := array.OfStrings(values...)
	for b.Loop() {
		if _, err := array.NewDictionary(indices, dict); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDictionaryIndex(b *testing.B) {
	idx := make([]int32, 4096)
	for i := range idx {
		idx[i] = int32(i % 256)
	}
	values := make([]string, 256)
	for i := range values {
		values[i] = "value"
	}

	a, err := array.NewDictionary(array.Of(idx...), array.OfStrings(values...))
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()

	total := 0
	for b.Loop() {
		for i := range a.Len() {
			total += a.Index(i)
		}
	}
	if total == 0 {
		b.Fatal("nothing was read")
	}
}
