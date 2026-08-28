package kernel_test

import (
	"errors"
	"math"
	"slices"
	"strconv"
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

	dt := dtype.Dictionary{Index: dtype.Int32, Value: values.DType()}
	c, err := array.NewChunked(dt, codes(t, values, at...))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	return c
}

// indexed is dictColumn with a say in the index type, which is what a column
// read out of a file that had few enough values to number in a byte comes back
// as. Minus one is a null here too.
func indexed(t *testing.T, index dtype.DataType, values *array.Array, at ...int) *array.Chunked {
	t.Helper()

	b, err := array.NewBuilder(index)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	for _, i := range at {
		if i < 0 {
			b.AppendNull()
			continue
		}
		appendIndices(t, b, index, i)
	}

	a, err := array.NewDictionary(b.Finish(), values)
	if err != nil {
		t.Fatalf("NewDictionary: %v", err)
	}
	dt := dtype.Dictionary{Index: index, Value: values.DType()}
	c, err := array.NewChunked(dt, a)
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

// TestGroupByDictionary is the ordinary case. Rows group by the string behind
// the index, and the keys come back dictionary encoded rather than expanded.
func TestGroupByDictionary(t *testing.T) {
	values := array.OfStrings("GB", "JP", "US")
	src := dictColumn(t, values, 1, 0, 1, 2, -1, 0)

	g, err := kernel.GroupBy(src)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if want := []int{0, 1, 0, 2, 3, 1}; !slices.Equal(g.IDs(), want) {
		t.Errorf("the groups are %v, want %v", g.IDs(), want)
	}
	if want := []int{2, 2, 1, 1}; !slices.Equal(g.Sizes(), want) {
		t.Errorf("the sizes are %v, want %v", g.Sizes(), want)
	}

	keys := g.Keys()[0]
	if !dtype.Equal(keys.DType(), codeType) {
		t.Errorf("the keys are a %s column, want %s", keys.DType(), codeType)
	}
	wantCodes(t, keys, "JP", "GB", "US", "")
}

// TestGroupByDictionaryChunks is two chunks that hold the same strings in a
// different order, which is what reading two files gives. The index of a row
// says nothing about the group it belongs to until it is read through the
// dictionary of its own chunk, and this is the test that says so.
func TestGroupByDictionaryChunks(t *testing.T) {
	left := array.OfStrings("GB", "JP")
	right := array.OfStrings("JP", "US", "GB")
	src, err := array.NewChunked(codeType,
		codes(t, left, 0, 1),
		codes(t, right, 0, 2, 1),
	)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	g, err := kernel.GroupBy(src)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if want := []int{0, 1, 1, 0, 2}; !slices.Equal(g.IDs(), want) {
		t.Errorf("the groups are %v, want %v", g.IDs(), want)
	}
	if want := []int{2, 2, 1}; !slices.Equal(g.Sizes(), want) {
		t.Errorf("the sizes are %v, want %v", g.Sizes(), want)
	}
	wantCodes(t, g.Keys()[0], "GB", "JP", "US")
}

// TestGroupByDictionaryNullValue is a dictionary that holds a null of its own.
// A row pointing at it is a row whose value is missing, so it groups with the
// rows that have no value at all rather than with the strings.
func TestGroupByDictionaryNullValue(t *testing.T) {
	b, err := array.NewBuilder(dtype.String)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	b.AppendString("GB")
	b.AppendNull()
	values := b.Finish()

	src := dictColumn(t, values, 0, 1, -1, 1, 0)
	g, err := kernel.GroupBy(src)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if want := []int{0, 1, 1, 1, 0}; !slices.Equal(g.IDs(), want) {
		t.Errorf("the groups are %v, want %v", g.IDs(), want)
	}
	if want := []int{2, 3}; !slices.Equal(g.Sizes(), want) {
		t.Errorf("the sizes are %v, want %v", g.Sizes(), want)
	}

	keys := g.Keys()[0]
	if got := valueAt(t, keys, 0); got != "GB" {
		t.Errorf("the first key is %v, want GB", got)
	}
	if got := valueAt(t, keys, 1); got != nil {
		t.Errorf("the second key is %v, want a missing value", got)
	}
}

// TestGroupByDictionaryDuplicates is a dictionary holding the same string
// twice, which nothing forbids and which two indices then stand for. The rows
// have to land in one group, so this is the case where the indices cannot be
// the key and the values have to be read after all.
func TestGroupByDictionaryDuplicates(t *testing.T) {
	values := array.OfStrings("GB", "JP", "GB")
	src := dictColumn(t, values, 0, 2, 1, 0)

	g, err := kernel.GroupBy(src)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if want := []int{0, 0, 1, 0}; !slices.Equal(g.IDs(), want) {
		t.Errorf("the groups are %v, want %v", g.IDs(), want)
	}
	wantCodes(t, g.Keys()[0], "GB", "JP")

	n, err := kernel.NUnique(src, kernel.OneGroup(src.Len()))
	if err != nil {
		t.Fatalf("NUnique: %v", err)
	}
	if got := n.Value[int64](0); got != 2 {
		t.Errorf("the column has %d distinct values, want 2", got)
	}
}

// TestGroupByDictionaryMixed is a dictionary encoded key next to a plain one,
// which is what a group by over a table read from Parquet looks like.
func TestGroupByDictionaryMixed(t *testing.T) {
	code := dictColumn(t, array.OfStrings("GB", "JP"), 0, 0, 1, 0)
	day := col(t, dtype.Int32, []any{int32(1), int32(2), int32(1), int32(1)})

	g, err := kernel.GroupBy(code, day)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if want := []int{0, 1, 2, 0}; !slices.Equal(g.IDs(), want) {
		t.Errorf("the groups are %v, want %v", g.IDs(), want)
	}
	wantCodes(t, g.Keys()[0], "GB", "GB", "JP")

	wantDay := []any{int32(1), int32(2), int32(1)}
	for i, want := range wantDay {
		if got := valueAt(t, g.Keys()[1], i); got != want {
			t.Errorf("day of group %d is %v, want %v", i, got, want)
		}
	}
}

// TestGroupByDictionaryIndexTypes runs a group by once per index type, since
// reading the indices of a chunk is a case for each of them.
func TestGroupByDictionaryIndexTypes(t *testing.T) {
	values := array.OfStrings("GB", "JP", "US")
	indexes := []dtype.DataType{
		dtype.Int8, dtype.Int16, dtype.Int32, dtype.Int64,
		dtype.Uint8, dtype.Uint16, dtype.Uint32, dtype.Uint64,
	}

	for _, index := range indexes {
		t.Run(index.String(), func(t *testing.T) {
			b, err := array.NewBuilder(index)
			if err != nil {
				t.Fatalf("NewBuilder: %v", err)
			}
			appendIndices(t, b, index, 2, 0, 2)
			b.AppendNull()

			a, err := array.NewDictionary(b.Finish(), values)
			if err != nil {
				t.Fatalf("NewDictionary: %v", err)
			}
			src, err := array.NewChunked(dtype.Dictionary{Index: index, Value: dtype.String}, a)
			if err != nil {
				t.Fatalf("NewChunked: %v", err)
			}

			g, err := kernel.GroupBy(src)
			if err != nil {
				t.Fatalf("GroupBy: %v", err)
			}
			if want := []int{0, 1, 0, 2}; !slices.Equal(g.IDs(), want) {
				t.Errorf("the groups are %v, want %v", g.IDs(), want)
			}
			wantCodes(t, g.Keys()[0], "US", "GB", "")
		})
	}
}

// TestGroupByDictionarySliced groups a column that starts partway into its
// buffers, which is what a slice of a batch is.
func TestGroupByDictionarySliced(t *testing.T) {
	whole := dictColumn(t, array.OfStrings("GB", "JP", "US"), 0, 1, 2, 1, 0)
	src := whole.Slice(1, 4)

	g, err := kernel.GroupBy(src)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if want := []int{0, 1, 0}; !slices.Equal(g.IDs(), want) {
		t.Errorf("the groups are %v, want %v", g.IDs(), want)
	}
	wantCodes(t, g.Keys()[0], "JP", "US")
}

// TestAggDictionary covers the aggregations that read a dictionary encoded
// column without decoding it. A count skips the row pointing at the missing
// value the same way it skips the row that has no value, first and last skip
// them too, and a distinct count counts the strings rather than the indices.
func TestAggDictionary(t *testing.T) {
	b, err := array.NewBuilder(dtype.String)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	b.AppendString("GB")
	b.AppendNull()
	b.AppendString("JP")
	values := b.Finish()

	// Two groups, taken one row each in turn: the first holds GB, a missing
	// value and GB again, the second a missing row, JP and JP.
	src := dictColumn(t, values, 0, -1, 1, 2, 0, 2)
	by := col(t, dtype.Int32, []any{int32(0), int32(1), int32(0), int32(1), int32(0), int32(1)})

	g, err := kernel.GroupBy(by)
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}

	counts := kernel.Count(src, g)
	for i, want := range []int64{2, 2} {
		if got := counts.Value[int64](i); got != want {
			t.Errorf("group %d counts %d values, want %d", i, got, want)
		}
	}
	if got := kernel.Size(g); got.Value[int64](0) != 3 {
		t.Errorf("group 0 has %d rows, want 3", got.Value[int64](0))
	}

	first := kernel.First(src, g)
	wantCodes(t, first, "GB", "JP")
	last := kernel.Last(src, g)
	wantCodes(t, last, "GB", "JP")

	n, err := kernel.NUnique(src, g)
	if err != nil {
		t.Fatalf("NUnique: %v", err)
	}
	for i, want := range []int64{1, 1} {
		if got := n.Value[int64](i); got != want {
			t.Errorf("group %d has %d distinct values, want %d", i, got, want)
		}
	}
}

// TestCompareDictionary is the ordinary comparison: a dictionary encoded column
// against one value, which is what a filter on a column read out of Parquet is.
// The answers are the answers the same column of plain strings would give, and
// the two ways of having no value both give a null.
func TestCompareDictionary(t *testing.T) {
	values := arrayOf(t, dtype.String, "GB", "JP", nil)
	src := dictColumn(t, values, 0, 1, 2, -1, 1)
	one := col(t, dtype.String, []any{"JP"})

	tests := []struct {
		op   kernel.CompareOp
		want []any
	}{
		{kernel.OpEq, []any{false, true, nil, nil, true}},
		{kernel.OpNe, []any{true, false, nil, nil, false}},
		{kernel.OpLt, []any{true, false, nil, nil, false}},
		{kernel.OpLe, []any{true, true, nil, nil, true}},
		{kernel.OpGt, []any{false, false, nil, nil, false}},
		{kernel.OpGe, []any{false, true, nil, nil, true}},
	}

	for _, tt := range tests {
		t.Run(tt.op.String(), func(t *testing.T) {
			got, err := kernel.Compare(src, one, tt.op)
			if err != nil {
				t.Fatalf("Compare: %v", err)
			}
			if have := answers(t, got); !same(have, tt.want) {
				t.Errorf("the answers are %v, want %v", have, tt.want)
			}
		})
	}

	// The same question from the other side, since either operand may be the
	// encoded one and each is read through its own cursor.
	got, err := kernel.Compare(one, src, kernel.OpGt)
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}
	want := []any{true, false, nil, nil, false}
	if have := answers(t, got); !same(have, want) {
		t.Errorf("the answers from the other side are %v, want %v", have, want)
	}
}

// TestCompareDictionaryBoth is two encoded columns against each other, which is
// two files read into one table. They hold different dictionaries and number
// them differently, so an answer that came from the indices would be wrong on
// every row.
func TestCompareDictionaryBoth(t *testing.T) {
	left := dictColumn(t, array.OfStrings("GB", "JP", "US"), 0, 1, 2, 1)
	right := indexed(t, dtype.Int8, array.OfStrings("JP", "GB"), 0, 0, 1, 1)

	tests := []struct {
		op   kernel.CompareOp
		want []any
	}{
		{kernel.OpEq, []any{false, true, false, false}},
		{kernel.OpLt, []any{true, false, false, false}},
		{kernel.OpGt, []any{false, false, true, true}},
	}

	for _, tt := range tests {
		t.Run(tt.op.String(), func(t *testing.T) {
			got, err := kernel.Compare(left, right, tt.op)
			if err != nil {
				t.Fatalf("Compare: %v", err)
			}
			if have := answers(t, got); !same(have, tt.want) {
				t.Errorf("the answers are %v, want %v", have, tt.want)
			}
		})
	}
}

// TestCompareDictionaryChunks is a column whose chunks number their values
// differently, which is what putting two files end to end gives. Every row is
// read through the dictionary of the chunk it is in.
func TestCompareDictionaryChunks(t *testing.T) {
	src, err := array.NewChunked(codeType,
		codes(t, array.OfStrings("GB", "JP"), 0, 1),
		codes(t, array.OfStrings("JP", "US"), 0, 1),
	)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	got, err := kernel.Compare(src, col(t, dtype.String, []any{"JP"}), kernel.OpEq)
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}
	want := []any{false, true, true, false}
	if have := answers(t, got); !same(have, want) {
		t.Errorf("the answers are %v, want %v", have, want)
	}
}

// TestCompareDictionaryNumbers is a dictionary of something other than strings,
// with a NaN in it. The encoding does not change what a NaN is, so it is
// unordered against everything and only != is true of it.
func TestCompareDictionaryNumbers(t *testing.T) {
	values := arrayOf(t, dtype.Float64, 1.5, math.NaN(), 2.5)
	src := dictColumn(t, values, 0, 1, 2)
	one := col(t, dtype.Float64, []any{1.5})

	tests := []struct {
		op   kernel.CompareOp
		want []any
	}{
		{kernel.OpEq, []any{true, false, false}},
		{kernel.OpNe, []any{false, true, true}},
		{kernel.OpLt, []any{false, false, false}},
	}

	for _, tt := range tests {
		t.Run(tt.op.String(), func(t *testing.T) {
			got, err := kernel.Compare(src, one, tt.op)
			if err != nil {
				t.Fatalf("Compare: %v", err)
			}
			if have := answers(t, got); !same(have, tt.want) {
				t.Errorf("the answers are %v, want %v", have, tt.want)
			}
		})
	}
}

// TestCompareDictionaryIndexTypes compares and orders a column once per index
// type. The indices of a chunk are read as the type they were written as, that
// type is looked at once per chunk rather than once per row, and a case missing
// from that would be found by whoever read a file that used it.
func TestCompareDictionaryIndexTypes(t *testing.T) {
	values := array.OfStrings("GB", "JP", "US")
	one := col(t, dtype.String, []any{"JP"})
	indexes := []dtype.DataType{
		dtype.Int8, dtype.Int16, dtype.Int32, dtype.Int64,
		dtype.Uint8, dtype.Uint16, dtype.Uint32, dtype.Uint64,
	}

	for _, index := range indexes {
		t.Run(index.String(), func(t *testing.T) {
			src := indexed(t, index, values, 2, 0, 1, -1)

			got, err := kernel.Compare(src, one, kernel.OpEq)
			if err != nil {
				t.Fatalf("Compare: %v", err)
			}
			want := []any{false, false, true, nil}
			if have := answers(t, got); !same(have, want) {
				t.Errorf("the answers are %v, want %v", have, want)
			}

			order, err := kernel.SortIndex(kernel.Order{Column: src})
			if err != nil {
				t.Fatalf("SortIndex: %v", err)
			}
			if wantOrder := []int{1, 2, 0, 3}; !slices.Equal(order, wantOrder) {
				t.Errorf("the order is %v, want %v", order, wantOrder)
			}
		})
	}
}

// TestCompareDictionaryFilter is the whole of what this is for: a predicate on
// an encoded column keeps the rows it names and the column that comes out is
// still encoded and still points at the values that went in.
func TestCompareDictionaryFilter(t *testing.T) {
	values := array.OfStrings("GB", "JP", "US")
	src := dictColumn(t, values, 0, 1, 2, -1, 1)

	mask, err := kernel.Compare(src, col(t, dtype.String, []any{"JP"}), kernel.OpEq)
	if err != nil {
		t.Fatalf("Compare: %v", err)
	}

	got := kernel.Filter(src, mask)
	wantCodes(t, got, "JP", "JP")
	if d := got.Chunk(0).Dictionary(); d != values {
		t.Errorf("the result points at %p, want the %p it was filtered from", d, values)
	}
}

// TestSortDictionary orders an encoded column by the strings behind its indices
// rather than by the indices, which are the order the writer happened to meet
// the values in and here are the reverse of the order they sort in.
func TestSortDictionary(t *testing.T) {
	values := arrayOf(t, dtype.String, "US", "GB", nil, "JP")
	src := dictColumn(t, values, 0, 1, 2, -1, 3)

	tests := []struct {
		name string
		key  kernel.Order
		want []int
	}{
		{"ascending", kernel.Order{Column: src}, []int{1, 4, 0, 2, 3}},
		{"descending", kernel.Order{Column: src, Descending: true}, []int{0, 4, 1, 2, 3}},
		{"nulls first", kernel.Order{Column: src, NullsFirst: true}, []int{2, 3, 1, 4, 0}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := kernel.SortIndex(tt.key)
			if err != nil {
				t.Fatalf("SortIndex: %v", err)
			}
			if !slices.Equal(got, tt.want) {
				t.Errorf("the order is %v, want %v", got, tt.want)
			}
		})
	}
}

// TestSortDictionaryChunks sorts across chunks that number their values
// differently, so a sort reading the indices would put them in the wrong order
// and a sort reading the wrong chunk's values would read the wrong strings.
func TestSortDictionaryChunks(t *testing.T) {
	src, err := array.NewChunked(codeType,
		codes(t, array.OfStrings("GB", "US"), 0, 1),
		codes(t, array.OfStrings("JP", "AA"), 0, 1),
	)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	got, err := kernel.SortIndex(kernel.Order{Column: src})
	if err != nil {
		t.Fatalf("SortIndex: %v", err)
	}
	if want := []int{3, 0, 2, 1}; !slices.Equal(got, want) {
		t.Errorf("the order is %v, want %v", got, want)
	}

	wantCodes(t, kernel.Take(src, got), "AA", "GB", "JP", "US")
}

// TestSortDictionaryValues sorts dictionaries of the other two shapes a value
// can have, since the numbers are read through a typed slice of the dictionary
// and the booleans are bits and are read one at a time.
func TestSortDictionaryValues(t *testing.T) {
	numbers := dictColumn(t, arrayOf(t, dtype.Int64, int64(30), int64(10), int64(20)), 0, 1, 2, -1)
	got, err := kernel.SortIndex(kernel.Order{Column: numbers})
	if err != nil {
		t.Fatalf("SortIndex: %v", err)
	}
	if want := []int{1, 2, 0, 3}; !slices.Equal(got, want) {
		t.Errorf("the numbers are ordered %v, want %v", got, want)
	}

	bools := dictColumn(t, arrayOf(t, dtype.Bool, true, false), 0, 1, 0)
	got, err = kernel.SortIndex(kernel.Order{Column: bools})
	if err != nil {
		t.Fatalf("SortIndex: %v", err)
	}
	if want := []int{1, 0, 2}; !slices.Equal(got, want) {
		t.Errorf("the booleans are ordered %v, want %v", got, want)
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

// BenchmarkGroupByDictionary groups the same column stored both ways. The
// dictionary side keys the rows by their indices, so it hashes four bytes where
// the other hashes a string, and it pays a pass over the two hundred and fifty
// values first to find out that it may.
func BenchmarkGroupByDictionary(b *testing.B) {
	dict, plain := benchDictColumn(b)

	b.Run("dictionary", func(b *testing.B) {
		for b.Loop() {
			if _, err := kernel.GroupBy(dict); err != nil {
				b.Fatalf("GroupBy: %v", err)
			}
		}
	})
	b.Run("strings", func(b *testing.B) {
		for b.Loop() {
			if _, err := kernel.GroupBy(plain); err != nil {
				b.Fatalf("GroupBy: %v", err)
			}
		}
	})
}

// BenchmarkCompareDictionary is a predicate on the same column stored both
// ways, which is the filter a scan of a Parquet file carries. The dictionary
// side pays a load of the index before the load of the value, and it gets the
// two hundred and fifty values back out of cache rather than walking a hundred
// thousand strings.
func BenchmarkCompareDictionary(b *testing.B) {
	dict, plain := benchDictColumn(b)

	one, err := array.NewChunked(dtype.String, plain.Chunk(0).Slice(0, 1))
	if err != nil {
		b.Fatalf("NewChunked: %v", err)
	}

	b.Run("dictionary", func(b *testing.B) {
		for b.Loop() {
			if _, err := kernel.Compare(dict, one, kernel.OpEq); err != nil {
				b.Fatalf("Compare: %v", err)
			}
		}
	})
	b.Run("strings", func(b *testing.B) {
		for b.Loop() {
			if _, err := kernel.Compare(plain, one, kernel.OpEq); err != nil {
				b.Fatalf("Compare: %v", err)
			}
		}
	})
}

// BenchmarkSortDictionary sorts the same column stored both ways. A sort reads
// every row log n times, so this is where the extra load per value is paid the
// most times over and where the values staying in cache is worth the most.
func BenchmarkSortDictionary(b *testing.B) {
	dict, plain := benchDictColumn(b)

	b.Run("dictionary", func(b *testing.B) {
		for b.Loop() {
			if _, err := kernel.SortIndex(kernel.Order{Column: dict}); err != nil {
				b.Fatalf("SortIndex: %v", err)
			}
		}
	})
	b.Run("strings", func(b *testing.B) {
		for b.Loop() {
			if _, err := kernel.SortIndex(kernel.Order{Column: plain}); err != nil {
				b.Fatalf("SortIndex: %v", err)
			}
		}
	})
}

// TestCastDictionary is the ordinary case: a column of numbers written as
// strings and read out of a file that encoded them becomes a column of numbers.
// The result is a plain int64 column, since that is what was asked for.
func TestCastDictionary(t *testing.T) {
	c := dictColumn(t, arrayOf(t, dtype.String, "10", "20", "30"), 2, 0, 1, -1, 0)
	got, err := kernel.Cast(c, dtype.Int64)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}

	if !dtype.Equal(got.DType(), dtype.Int64) {
		t.Fatalf("the result is a %s column, want int64", got.DType())
	}
	want := []any{int64(30), int64(10), int64(20), nil, int64(10)}
	for i, w := range want {
		if have := valueAt(t, got, i); have != w {
			t.Errorf("value %d is %v, want %v", i, have, w)
		}
	}
	if got.NullCount() != 1 {
		t.Errorf("the result counts %d nulls, want 1", got.NullCount())
	}
}

// TestCastDictionaryChunks is a column whose chunks read from the one
// dictionary, which is what a column read out of one file looks like. The
// values are cast once between them and the rows of both chunks come out right.
func TestCastDictionaryChunks(t *testing.T) {
	values := arrayOf(t, dtype.String, "1", "2", "3")
	c, err := array.NewChunked(codeType,
		codes(t, values, 0, 1),
		codes(t, values, 2, -1, 0))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	got, err := kernel.Cast(c, dtype.Int32)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	if got.NumChunks() != 2 {
		t.Errorf("the result has %d chunks, want 2", got.NumChunks())
	}
	want := []any{int32(1), int32(2), int32(3), nil, int32(1)}
	for i, w := range want {
		if have := valueAt(t, got, i); have != w {
			t.Errorf("value %d is %v, want %v", i, have, w)
		}
	}
}

// TestCastDictionaryDisagree is two chunks holding different dictionaries,
// which is what putting two files end to end gives. Each is cast on its own and
// the rows read the values of their own chunk.
func TestCastDictionaryDisagree(t *testing.T) {
	c, err := array.NewChunked(codeType,
		codes(t, arrayOf(t, dtype.String, "1", "2"), 1, 0),
		codes(t, arrayOf(t, dtype.String, "7", "8"), 0, 1))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	got, err := kernel.Cast(c, dtype.Int64)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	want := []any{int64(2), int64(1), int64(7), int64(8)}
	for i, w := range want {
		if have := valueAt(t, got, i); have != w {
			t.Errorf("value %d is %v, want %v", i, have, w)
		}
	}
}

// TestCastDictionaryNullValue is a dictionary that holds a null of its own,
// meaning a value that is missing rather than a row that is. A row pointing at
// it has no value to cast and comes out missing.
func TestCastDictionaryNullValue(t *testing.T) {
	c := dictColumn(t, arrayOf(t, dtype.String, "5", nil, "6"), 0, 1, 2)
	got, err := kernel.Cast(c, dtype.Int64)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}

	want := []any{int64(5), nil, int64(6)}
	for i, w := range want {
		if have := valueAt(t, got, i); have != w {
			t.Errorf("value %d is %v, want %v", i, have, w)
		}
	}
}

// TestCastDictionaryIndexTypes casts once per index type, since reaching the
// rows of a chunk goes through the index and each width is a case of its own.
func TestCastDictionaryIndexTypes(t *testing.T) {
	for _, index := range []dtype.DataType{
		dtype.Int8, dtype.Int16, dtype.Int32, dtype.Int64,
		dtype.Uint8, dtype.Uint16, dtype.Uint32, dtype.Uint64,
	} {
		t.Run(index.String(), func(t *testing.T) {
			c := indexed(t, index, arrayOf(t, dtype.String, "4", "9"), 1, 0, -1)
			got, err := kernel.Cast(c, dtype.Int64)
			if err != nil {
				t.Fatalf("Cast: %v", err)
			}
			want := []any{int64(9), int64(4), nil}
			for i, w := range want {
				if have := valueAt(t, got, i); have != w {
					t.Errorf("value %d is %v, want %v", i, have, w)
				}
			}
		})
	}
}

// TestCastDictionaryBadValue is a value no parser will take, with a row
// pointing at it. The whole cast fails and says which row it was, counted from
// the start of the column and not from the start of the chunk.
func TestCastDictionaryBadValue(t *testing.T) {
	values := arrayOf(t, dtype.String, "1", "n/a")
	c, err := array.NewChunked(codeType,
		codes(t, values, 0, 0),
		codes(t, values, 0, 1, 0))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	_, err = kernel.Cast(c, dtype.Int64)
	if err == nil {
		t.Fatal("the cast took a value that is not a number")
	}

	var ce *kernel.CastError
	if !errors.As(err, &ce) {
		t.Fatalf("Cast returned %v, want a *CastError", err)
	}
	if ce.Row != 3 {
		t.Errorf("the error blames row %d, want 3", ce.Row)
	}
	if !dtype.Equal(ce.From, codeType) {
		t.Errorf("the error says the column is a %s, want %s", ce.From, codeType)
	}
	if !dtype.Equal(ce.To, dtype.Int64) {
		t.Errorf("the error says the cast was to %s, want int64", ce.To)
	}
	if !strings.Contains(ce.Value, "n/a") {
		t.Errorf("the error prints the value as %s, want it to mention n/a", ce.Value)
	}
}

// TestCastDictionaryUnusedBadValue is the decision this file makes about a
// dictionary holding more than the column uses. A writer that carried a value
// over from another row group has not put anything wrong in the rows, so a
// value nothing points at does not fail the cast.
func TestCastDictionaryUnusedBadValue(t *testing.T) {
	c := dictColumn(t, arrayOf(t, dtype.String, "1", "n/a", "2"), 0, 2, -1, 0)
	got, err := kernel.Cast(c, dtype.Int64)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}

	want := []any{int64(1), int64(2), nil, int64(1)}
	for i, w := range want {
		if have := valueAt(t, got, i); have != w {
			t.Errorf("value %d is %v, want %v", i, have, w)
		}
	}
}

// TestTryCastDictionary is the same bad value with the strictness turned off,
// where the rows pointing at it come out missing and the rest come out.
func TestTryCastDictionary(t *testing.T) {
	c := dictColumn(t, arrayOf(t, dtype.String, "1", "n/a", "2"), 1, 0, 1, 2)
	got, err := kernel.TryCast(c, dtype.Int64)
	if err != nil {
		t.Fatalf("TryCast: %v", err)
	}

	want := []any{nil, int64(1), nil, int64(2)}
	for i, w := range want {
		if have := valueAt(t, got, i); have != w {
			t.Errorf("value %d is %v, want %v", i, have, w)
		}
	}
	if got.NullCount() != 2 {
		t.Errorf("the result counts %d nulls, want 2", got.NullCount())
	}
}

// TestCastDictionaryKeeps is a cast to another dictionary type, which keeps the
// encoding. The values are cast and the indices are left where they are, so a
// hundred thousand rows over three values convert three values.
func TestCastDictionaryKeeps(t *testing.T) {
	c := dictColumn(t, arrayOf(t, dtype.String, "10", "20", "30"), 2, 0, -1, 1)
	to := dtype.Dictionary{Index: dtype.Int32, Value: dtype.Int64}

	got, err := kernel.Cast(c, to)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	if !dtype.Equal(got.DType(), to) {
		t.Fatalf("the result is a %s column, want %s", got.DType(), to)
	}
	if d := got.Chunk(0).Dictionary(); d.Len() != 3 {
		t.Errorf("the result holds %d values, want 3", d.Len())
	}

	want := []any{int64(30), int64(10), nil, int64(20)}
	for i, w := range want {
		if have := valueAt(t, got, i); have != w {
			t.Errorf("value %d is %v, want %v", i, have, w)
		}
	}
}

// TestCastDictionaryIndex is a cast that changes nothing but the index type,
// which is what narrowing a column read out of a file to what it turned out to
// need looks like. There is nothing to convert, so the values are the ones that
// were already there rather than a copy of them.
func TestCastDictionaryIndex(t *testing.T) {
	values := arrayOf(t, dtype.String, "de", "fr", "it")
	c := dictColumn(t, values, 0, 2, -1, 1)

	to := dtype.Dictionary{Index: dtype.Int8, Value: dtype.String}
	got, err := kernel.Cast(c, to)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	if !dtype.Equal(got.DType(), to) {
		t.Fatalf("the result is a %s column, want %s", got.DType(), to)
	}
	if d := got.Chunk(0).Dictionary(); d != values {
		t.Error("the result copied the values, want the ones it was given")
	}
	if k := got.Chunk(0).Indices().DType(); !dtype.Equal(k, dtype.Int8) {
		t.Errorf("the result is indexed by %s, want int8", k)
	}
	wantCodes(t, got, "de", "it", "", "fr")
}

// TestCastDictionaryIndexTooSmall is the one thing that cannot be done. An
// index of a type that cannot name every value of the dictionary would leave
// rows pointing nowhere, so it is refused before anything is cast.
func TestCastDictionaryIndexTooSmall(t *testing.T) {
	b, err := array.NewBuilder(dtype.String)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	for i := range 200 {
		b.AppendString(strings.Repeat("a", i%9+1))
	}
	c := dictColumn(t, b.Finish(), 0, 1, 2)

	to := dtype.Dictionary{Index: dtype.Int8, Value: dtype.String}
	_, err = kernel.Cast(c, to)
	if err == nil {
		t.Fatal("the cast numbered two hundred values in an int8")
	}
	if !strings.Contains(err.Error(), "can name") {
		t.Errorf("Cast said %q, want it to say what an int8 can name", err)
	}
}

// TestCastDictionaryIndexNotInteger is a destination that is a dictionary of
// something that cannot be a position. It is a mistake in the type rather than
// in the data, and it is refused before any value is read.
func TestCastDictionaryIndexNotInteger(t *testing.T) {
	c := dictColumn(t, arrayOf(t, dtype.String, "1", "2"), 0, 1)

	to := dtype.Dictionary{Index: dtype.Float64, Value: dtype.Int64}
	_, err := kernel.Cast(c, to)
	if err == nil {
		t.Fatal("the cast indexed a dictionary by a float")
	}
	if !strings.Contains(err.Error(), "indexed by an integer") {
		t.Errorf("Cast said %q, want it to say an index is an integer", err)
	}
}

// TestCastDictionaryNotYet is a cast of the values that this package has not
// learned. The encoding does not change the answer, so it is the same error the
// same cast gives on a plain column.
func TestCastDictionaryNotYet(t *testing.T) {
	c := dictColumn(t, arrayOf(t, dtype.String, "2026-08-29"), 0)

	_, err := kernel.Cast(c, dtype.Timestamp{Unit: dtype.Second})
	if err == nil {
		t.Fatal("the cast parsed a date")
	}
	if !strings.Contains(err.Error(), "not implemented yet") {
		t.Errorf("Cast said %q, want it to say the cast is not written yet", err)
	}
}

// TestCastToDictionary is encoding a plain column, which is the other half of
// this and is not written yet. It is worth a test because the error has to say
// so rather than the cast quietly producing something else.
func TestCastToDictionary(t *testing.T) {
	c := col(t, dtype.String, []any{"de", "fr", "de"})

	to := dtype.Dictionary{Index: dtype.Int32, Value: dtype.String}
	_, err := kernel.Cast(c, to)
	if err == nil {
		t.Fatal("the cast encoded a plain column")
	}
	if !strings.Contains(err.Error(), "not implemented yet") {
		t.Errorf("Cast said %q, want it to say the cast is not written yet", err)
	}
}

// TestCastDictionaryToNull throws every value away, which is allowed because it
// takes saying so. The rows are still there and every one of them is missing.
func TestCastDictionaryToNull(t *testing.T) {
	c := dictColumn(t, arrayOf(t, dtype.String, "de", "fr"), 0, 1, 0)
	got, err := kernel.Cast(c, dtype.Null)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	if got.Len() != 3 || got.NullCount() != 3 {
		t.Errorf("the result has %d values and %d nulls, want 3 and 3", got.Len(), got.NullCount())
	}
}

// TestCastDictionarySame is a cast to the type the column already has, which is
// the column itself and no work at all.
func TestCastDictionarySame(t *testing.T) {
	c := dictColumn(t, arrayOf(t, dtype.String, "de", "fr"), 1, 0)
	got, err := kernel.Cast(c, codeType)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	if got != c {
		t.Error("the cast built a new column, want the one it was given")
	}
}

// TestCastDictionaryNumbers is a dictionary of something that is not a string,
// since nothing in here knows what a value is and this is the test that says so.
func TestCastDictionaryNumbers(t *testing.T) {
	c := dictColumn(t, arrayOf(t, dtype.Int64, int64(1), int64(2)), 0, 1, 1)
	got, err := kernel.Cast(c, dtype.Float64)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}

	want := []any{1.0, 2.0, 2.0}
	for i, w := range want {
		if have := valueAt(t, got, i); have != w {
			t.Errorf("value %d is %v, want %v", i, have, w)
		}
	}
}

// TestCastDictionaryEmpty is a column with no chunks in it, which has no
// dictionary to cast and no rows to expand.
func TestCastDictionaryEmpty(t *testing.T) {
	c, err := array.NewChunked(codeType)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	got, err := kernel.Cast(c, dtype.Int64)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	if got.Len() != 0 {
		t.Errorf("the result has %d values, want none", got.Len())
	}
	if !dtype.Equal(got.DType(), dtype.Int64) {
		t.Errorf("the result is a %s column, want int64", got.DType())
	}
}

// benchDictParse is a column of numbers written as text, stored both dictionary
// encoded and plain. Parsing is the expensive cast and the one a Parquet file
// of low cardinality strings actually asks for.
func benchDictParse(b *testing.B) (dict, plain *array.Chunked) {
	b.Helper()

	values, err := array.NewBuilder(dtype.String)
	if err != nil {
		b.Fatalf("NewBuilder: %v", err)
	}
	for i := range benchDictValues {
		values.AppendString(strconv.Itoa(i * 37))
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

// BenchmarkCastDictionary parses the same column stored both ways. The keep
// case is the one the encoding was built for, since it parses two hundred and
// fifty numbers and copies nothing, and the decode case still parses only those
// and then writes the hundred thousand rows the caller asked for.
func BenchmarkCastDictionary(b *testing.B) {
	dict, plain := benchDictParse(b)
	keep := dtype.Dictionary{Index: dtype.Int32, Value: dtype.Int64}

	b.Run("keep", func(b *testing.B) {
		for b.Loop() {
			if _, err := kernel.Cast(dict, keep); err != nil {
				b.Fatalf("Cast: %v", err)
			}
		}
	})
	b.Run("decode", func(b *testing.B) {
		for b.Loop() {
			if _, err := kernel.Cast(dict, dtype.Int64); err != nil {
				b.Fatalf("Cast: %v", err)
			}
		}
	})
	b.Run("strings", func(b *testing.B) {
		for b.Loop() {
			if _, err := kernel.Cast(plain, dtype.Int64); err != nil {
				b.Fatalf("Cast: %v", err)
			}
		}
	})
}

// TestCastDictionarySliced checks that the cast reads the column it was given
// rather than the buffers underneath it, which for a sliced column start
// somewhere else.
func TestCastDictionarySliced(t *testing.T) {
	values := arrayOf(t, dtype.String, "1", "2", "3")
	whole := codes(t, values, 0, 1, 2, -1, 0)
	c, err := array.NewChunked(codeType, whole.Slice(1, 4))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	got, err := kernel.Cast(c, dtype.Int64)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	want := []any{int64(2), int64(3), nil}
	for i, w := range want {
		if have := valueAt(t, got, i); have != w {
			t.Errorf("value %d is %v, want %v", i, have, w)
		}
	}

	keep, err := kernel.Cast(c, dtype.Dictionary{Index: dtype.Int16, Value: dtype.Int64})
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	for i, w := range want {
		if have := valueAt(t, keep, i); have != w {
			t.Errorf("value %d of the encoded result is %v, want %v", i, have, w)
		}
	}
}
