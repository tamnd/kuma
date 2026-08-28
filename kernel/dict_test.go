package kernel_test

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// codeType is a column of country codes, which is what dictionary encoding is
// for: a few distinct strings and a great many rows.
var codeType = dtype.Dictionary{Index: dtype.Int32, Value: dtype.String}

// codes builds one chunk of a dictionary encoded column out of the values and
// the positions in them, where minus one is a null.
func codes(t *testing.T, values *array.Array, at ...int32) *array.Array {
	t.Helper()

	b, err := array.NewBuilder(dtype.Int32)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	for _, i := range at {
		if i < 0 {
			b.AppendNull()
			continue
		}
		b.Append(i)
	}

	col, err := array.NewDictionary(b.Finish(), values)
	if err != nil {
		t.Fatalf("NewDictionary: %v", err)
	}
	return col
}

// dictColumn builds a dictionary encoded column of one chunk.
func dictColumn(t *testing.T, values *array.Array, at ...int32) *array.Chunked {
	t.Helper()

	c, err := array.NewChunked(codeType, codes(t, values, at...))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	return c
}

// wantCodes checks that a dictionary encoded column reads as the given strings,
// where an empty string is a null.
func wantCodes(t *testing.T, c *array.Chunked, want ...string) {
	t.Helper()

	if c.Len() != len(want) {
		t.Fatalf("the column has %d values, want %d", c.Len(), len(want))
	}
	for i, w := range want {
		var have string
		if !c.IsNull(i) {
			a, k := c.At(i)
			have = string(a.Dictionary().Bytes(a.Index(k)))
		}
		if have != w {
			t.Errorf("value %d is %q, want %q", i, have, w)
		}
	}
}

// TestTakeDictionary is the ordinary case: a gather over a dictionary encoded
// column reads the values behind the indices, and the result is dictionary
// encoded rather than expanded into strings.
func TestTakeDictionary(t *testing.T) {
	src := dictColumn(t, array.OfStrings("GB", "JP", "US"), 0, 1, 2, -1, 1)

	idx := []int{4, 0, 3, 2, -1, 1}
	got := kernel.Take(src, idx)
	checkTake(t, got, src, idx)
	wantCodes(t, got, "JP", "GB", "", "US", "", "JP")
}

// TestTakeDictionaryShares is the reason the encoding is worth having. The
// values are not copied and not walked, so the result points at the array the
// column it came from points at.
func TestTakeDictionaryShares(t *testing.T) {
	values := array.OfStrings("GB", "JP", "US")
	src := dictColumn(t, values, 0, 1, 2, 0)

	got := kernel.Take(src, []int{3, 2, 1, 0})
	if d := got.Chunk(0).Dictionary(); d != values {
		t.Errorf("the result points at %p, want the %p it was taken from", d, values)
	}
}

// TestTakeDictionaryChunks is a column whose chunks all read from the one
// dictionary, which is what a column read out of one file looks like. The
// indices are gathered across the chunk boundary and the values stay where they
// are.
func TestTakeDictionaryChunks(t *testing.T) {
	values := array.OfStrings("GB", "JP", "US")
	src, err := array.NewChunked(codeType,
		codes(t, values, 0, 1),
		codes(t, values, 2, -1, 0),
	)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	if src.NumChunks() != 2 {
		t.Fatalf("the column has %d chunks, want 2", src.NumChunks())
	}

	idx := []int{4, 1, 3, 0, 2}
	got := kernel.Take(src, idx)
	checkTake(t, got, src, idx)
	wantCodes(t, got, "GB", "JP", "", "GB", "US")
	if d := got.Chunk(0).Dictionary(); d != values {
		t.Errorf("the result points at %p, want the %p its chunks agreed on", d, values)
	}
}

// TestTakeDictionaryDisagree is two chunks that do not read from the same
// dictionary, which is what putting two files end to end produces. The result
// gets both dictionaries laid end to end and the indices of the second chunk
// shifted past the values of the first.
func TestTakeDictionaryDisagree(t *testing.T) {
	left := array.OfStrings("GB", "JP")
	right := array.OfStrings("US", "FR", "DE")
	src, err := array.NewChunked(codeType,
		codes(t, left, 0, 1, 0),
		codes(t, right, 2, -1, 1),
	)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	idx := []int{5, 0, 3, 4, 1}
	got := kernel.Take(src, idx)
	checkTake(t, got, src, idx)
	wantCodes(t, got, "FR", "GB", "DE", "", "JP")

	if d := got.Chunk(0).Dictionary(); d.Len() != left.Len()+right.Len() {
		t.Errorf("the result reads from %d values, want the %d of both chunks",
			d.Len(), left.Len()+right.Len())
	}
}

// TestTakeDictionaryRepeated is the same two chunks again in the other order
// and then again. A dictionary that has already been laid down is not laid down
// a second time, so three chunks over two dictionaries give two.
func TestTakeDictionaryRepeated(t *testing.T) {
	left := array.OfStrings("GB", "JP")
	right := array.OfStrings("US", "FR", "DE")
	src, err := array.NewChunked(codeType,
		codes(t, left, 0),
		codes(t, right, 1),
		codes(t, left, 1),
	)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	idx := []int{2, 1, 0}
	got := kernel.Take(src, idx)
	checkTake(t, got, src, idx)
	wantCodes(t, got, "JP", "FR", "GB")

	if d := got.Chunk(0).Dictionary(); d.Len() != 5 {
		t.Errorf("the result reads from %d values, want 5", d.Len())
	}
}

// TestTakeDictionaryNullValue is a dictionary that holds a null of its own,
// which means a value that is missing rather than a row that is. The gather has
// to keep the two apart, since the index is there and points at nothing.
func TestTakeDictionaryNullValue(t *testing.T) {
	b, err := array.NewBuilder(dtype.String)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	b.AppendString("GB")
	b.AppendNull()
	values := b.Finish()

	// checkTake is no use here, because it reads a row pointing at a missing
	// value as a missing row and those are the two things this is about.
	src := dictColumn(t, values, 0, 1, -1)
	got := kernel.Take(src, []int{2, 1, 0})

	if got.Len() != 3 {
		t.Fatalf("the result has %d values, want 3", got.Len())
	}
	if got.NullCount() != 1 {
		t.Errorf("the result counts %d nulls, want 1", got.NullCount())
	}
	if !got.IsNull(0) {
		t.Error("the row that was missing is there, want a missing row")
	}
	if got.IsNull(1) {
		t.Error("the row pointing at the missing value is null itself, want a row that is there")
	}
	if !got.Chunk(0).Dictionary().IsNull(got.Chunk(0).Index(1)) {
		t.Error("the value behind that row is there, want the missing one")
	}
}

// TestTakeDictionaryIndexTypes runs the gather once per index type, since each
// of them is a case of its own and a missing one would be found by whoever read
// a Parquet file that used it.
func TestTakeDictionaryIndexTypes(t *testing.T) {
	values := array.OfStrings("GB", "JP", "US")
	tests := []dtype.DataType{
		dtype.Int8, dtype.Int16, dtype.Int32, dtype.Int64,
		dtype.Uint8, dtype.Uint16, dtype.Uint32, dtype.Uint64,
	}

	for _, index := range tests {
		t.Run(index.String(), func(t *testing.T) {
			b, err := array.NewBuilder(index)
			if err != nil {
				t.Fatalf("NewBuilder: %v", err)
			}
			appendIndices(t, b, index, 2, 0, 1)
			b.AppendNull()

			a, err := array.NewDictionary(b.Finish(), values)
			if err != nil {
				t.Fatalf("NewDictionary: %v", err)
			}
			dt := dtype.Dictionary{Index: index, Value: dtype.String}
			src, err := array.NewChunked(dt, a)
			if err != nil {
				t.Fatalf("NewChunked: %v", err)
			}

			idx := []int{3, 2, 1, 0}
			got := kernel.Take(src, idx)
			checkTake(t, got, src, idx)
			if !dtype.Equal(got.DType(), dt) {
				t.Errorf("the result is a %s column, want %s", got.DType(), dt)
			}
		})
	}
}

// appendIndices adds index values of whichever integer type the dictionary is
// indexed by.
func appendIndices(t *testing.T, b *array.Builder, index dtype.DataType, at ...int) {
	t.Helper()

	for _, i := range at {
		switch index.Kind() {
		case dtype.Int8Kind:
			b.Append(int8(i))
		case dtype.Int16Kind:
			b.Append(int16(i))
		case dtype.Int32Kind:
			b.Append(int32(i))
		case dtype.Int64Kind:
			b.Append(int64(i))
		case dtype.Uint8Kind:
			b.Append(uint8(i))
		case dtype.Uint16Kind:
			b.Append(uint16(i))
		case dtype.Uint32Kind:
			b.Append(uint32(i))
		case dtype.Uint64Kind:
			b.Append(uint64(i))
		default:
			t.Fatalf("no way to index a dictionary by %s", index)
		}
	}
}

// TestTakeDictionaryNumbers is a dictionary of something that is not a string.
// Nothing in the gather knows what a value is, so this is here to say so.
func TestTakeDictionaryNumbers(t *testing.T) {
	values := array.Of[int64](10, 20, 30)
	b, err := array.NewBuilder(dtype.Int32)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	b.AppendValues([]int32{2, 0, 1})

	a, err := array.NewDictionary(b.Finish(), values)
	if err != nil {
		t.Fatalf("NewDictionary: %v", err)
	}
	dt := dtype.Dictionary{Index: dtype.Int32, Value: dtype.Int64}
	src, err := array.NewChunked(dt, a)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	idx := []int{1, 2, 0}
	got := kernel.Take(src, idx)
	checkTake(t, got, src, idx)
}

// TestTakeDictionarySliced checks that the gather reads the column it was given
// rather than the buffers underneath it, which for a sliced column start
// somewhere else.
func TestTakeDictionarySliced(t *testing.T) {
	whole := dictColumn(t, array.OfStrings("GB", "JP", "US"), 0, 1, 2, -1, 0)
	src := whole.Slice(1, 4)

	idx := []int{0, 1, 2}
	got := kernel.Take(src, idx)
	checkTake(t, got, src, idx)
	wantCodes(t, got, "JP", "US", "")
}

// TestTakeDictionaryEmpty is a column with no values in it, which has no chunks
// and so no dictionary to point the result at.
func TestTakeDictionaryEmpty(t *testing.T) {
	src, err := array.NewChunked(codeType)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	got := kernel.Take(src, nil)
	if got.Len() != 0 {
		t.Errorf("the result has %d values, want none", got.Len())
	}
	if !dtype.Equal(got.DType(), codeType) {
		t.Errorf("the result is a %s column, want %s", got.DType(), codeType)
	}
}

// TestTakeDictionaryTooMany is the one thing that cannot be done. Two chunks
// with different dictionaries need one dictionary holding both, and an int8
// index cannot name more than a hundred and twenty eight values.
func TestTakeDictionaryTooMany(t *testing.T) {
	dt := dtype.Dictionary{Index: dtype.Int8, Value: dtype.String}
	chunks := make([]*array.Array, 2)
	for k := range chunks {
		b, err := array.NewBuilder(dtype.String)
		if err != nil {
			t.Fatalf("NewBuilder: %v", err)
		}
		for i := range 100 {
			b.AppendString(strings.Repeat("x", k+1) + string(rune('0'+i%10)))
		}

		idx, err := array.NewBuilder(dtype.Int8)
		if err != nil {
			t.Fatalf("NewBuilder: %v", err)
		}
		idx.Append(int8(k))

		chunks[k], err = array.NewDictionary(idx.Finish(), b.Finish())
		if err != nil {
			t.Fatalf("NewDictionary: %v", err)
		}
	}

	src, err := array.NewChunked(dt, chunks...)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	defer func() {
		p, ok := recover().(string)
		if !ok {
			t.Fatal("a gather that cannot number its values did not panic")
		}
		if !strings.Contains(p, "200 values") {
			t.Errorf("the panic says %q, want it to say how many values there were", p)
		}
	}()
	kernel.Take(src, []int{0, 1})
}

// TestFilterDictionary is the kernel every predicate goes through. A filter is
// a gather, so the column that comes out is still dictionary encoded and still
// points at the values that went in.
func TestFilterDictionary(t *testing.T) {
	values := array.OfStrings("GB", "JP", "US")
	src := dictColumn(t, values, 0, 1, 2, -1, 1)
	mask := col(t, dtype.Bool, []any{true, false, true, true, nil})

	got := kernel.Filter(src, mask)
	wantCodes(t, got, "GB", "US", "")
	if d := got.Chunk(0).Dictionary(); d != values {
		t.Errorf("the result points at %p, want the %p it was filtered from", d, values)
	}
}

const (
	benchDictRows   = 100_000
	benchDictValues = 250
)

// benchDictColumn is a column of low cardinality strings, the shape dictionary
// encoding is for.
func benchDictColumn(b *testing.B) (dict, plain *array.Chunked) {
	b.Helper()

	values, err := array.NewBuilder(dtype.String)
	if err != nil {
		b.Fatalf("NewBuilder: %v", err)
	}
	for i := range benchDictValues {
		values.AppendString("value number " + string(rune('a'+i%26)) + strings.Repeat("x", i%17))
	}
	vals := values.Finish()

	idx, err := array.NewBuilder(dtype.Int32)
	if err != nil {
		b.Fatalf("NewBuilder: %v", err)
	}
	strs, err := array.NewBuilder(dtype.String)
	if err != nil {
		b.Fatalf("NewBuilder: %v", err)
	}
	for i := range benchDictRows {
		at := i % benchDictValues
		idx.Append(int32(at))
		strs.AppendBytes(vals.Bytes(at))
	}

	a, err := array.NewDictionary(idx.Finish(), vals)
	if err != nil {
		b.Fatalf("NewDictionary: %v", err)
	}
	dict, err = array.NewChunked(codeType, a)
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	plain, err = array.NewChunked(dtype.String, strs.Finish())
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}
	return dict, plain
}

// BenchmarkTakeDictionary is the gather that made this worth writing, against
// the same column stored as plain strings. Both read every row once, in an
// order that is not the one they are in, which is what a sort or a join hands a
// gather.
func BenchmarkTakeDictionary(b *testing.B) {
	dict, plain := benchDictColumn(b)

	idx := make([]int, benchDictRows)
	for i := range idx {
		idx[i] = (i*7919 + 13) % benchDictRows
	}

	b.Run("dictionary", func(b *testing.B) {
		for b.Loop() {
			kernel.Take(dict, idx)
		}
	})
	b.Run("strings", func(b *testing.B) {
		for b.Loop() {
			kernel.Take(plain, idx)
		}
	})
}
