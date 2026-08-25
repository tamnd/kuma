package kernel_test

import (
	"math"
	"slices"
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

// FuzzGroupBy checks the definition of a grouping against the grouping itself:
// two rows are in the same group exactly when their keys are equal, which is a
// walk over every pair rather than another way of writing the implementation.
//
// It also adds the groups up and checks the total against the total of the
// whole column, since an aggregation that loses a row or counts one twice is
// the failure worth catching and it does not show up in the group numbers.
func FuzzGroupBy(f *testing.F) {
	f.Add([]byte{1, 1, 2}, []byte{9})
	f.Add([]byte{255, 255, 0}, []byte{1, 2, 3})
	f.Add([]byte{}, []byte{})
	f.Add([]byte{5, 5, 5, 5, 5, 5, 5, 5}, []byte{255, 1, 255})

	f.Fuzz(func(t *testing.T, keys, values []byte) {
		if len(keys) > 64 {
			t.Skip("a big input proves nothing a small one does not")
		}

		k := byteColumn(t, keys)
		g, err := kernel.GroupBy(k)
		if err != nil {
			t.Fatalf("GroupBy: %v", err)
		}
		if g.Len() != k.Len() {
			t.Fatalf("the grouping covers %d rows, the column has %d", g.Len(), k.Len())
		}

		ids := g.IDs()
		for i := range ids {
			for j := range ids {
				same := valueAt(t, k, i) == valueAt(t, k, j)
				if (ids[i] == ids[j]) != same {
					t.Fatalf("rows %d and %d are %v and %v, in groups %d and %d",
						i, j, valueAt(t, k, i), valueAt(t, k, j), ids[i], ids[j])
				}
			}
		}

		// The key of a group is the key of its rows, and the sizes are how many
		// of them there are.
		sizes := make([]int, g.NumGroups())
		for _, id := range ids {
			sizes[id]++
		}
		for id := range g.NumGroups() {
			if sizes[id] != g.Sizes()[id] {
				t.Fatalf("group %d has %d rows, Sizes says %d", id, sizes[id], g.Sizes()[id])
			}
			want := valueAt(t, k, g.FirstRows()[id])
			if got := valueAt(t, g.Keys()[0], id); got != want {
				t.Fatalf("the key of group %d is %v, want %v", id, got, want)
			}
		}

		// Now a column of its own to add up, cut into chunks the fuzzer picks,
		// so the walk over the values and the walk over the groups have to line
		// up whatever shape either of them is.
		v := sameLength(t, values, k.Len())
		parts, err := kernel.Sum(v, g)
		if err != nil {
			t.Fatalf("Sum: %v", err)
		}
		whole, err := kernel.Sum(v, kernel.OneGroup(v.Len()))
		if err != nil {
			t.Fatalf("Sum over one group: %v", err)
		}

		var total int64
		for i := range parts.Len() {
			total += parts.Value[int64](i)
		}
		if whole.Len() == 0 {
			if total != 0 {
				t.Fatalf("the groups add up to %d with no rows to add up", total)
			}
			return
		}
		if want := whole.Value[int64](0); total != want {
			t.Fatalf("the groups add up to %d, the whole column adds up to %d", total, want)
		}
	})
}

// sameLength returns an int64 column of exactly n values built out of the bytes
// given, repeating or stopping as it has to, with a null wherever the byte is
// 255.
func sameLength(t *testing.T, values []byte, n int) *array.Chunked {
	t.Helper()

	b, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}

	var chunks []*array.Array
	for i := range n {
		v := byte(0)
		if len(values) > 0 {
			v = values[i%len(values)]
		}
		if v == 255 {
			b.AppendNull()
		} else {
			b.Append(int64(v) - 128)
		}
		// A chunk boundary at a different spacing from the one the keys use, so
		// that the two walks are never in step.
		if i%3 == 2 {
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

// FuzzAggregate checks the things that have to hold between the aggregations
// whatever the values are, since the answers themselves are not something a
// fuzzer can know.
//
// The properties are that a quantile never goes down as q goes up, that the
// median and every quantile sit between the smallest value and the largest, that
// a distinct count is between zero and the number of values that are there, that
// a variance is never negative, and that squaring the standard deviation gives
// the variance back.
func FuzzAggregate(f *testing.F) {
	f.Add([]byte{1, 1, 2}, []byte{9})
	f.Add([]byte{0}, []byte{255})
	f.Add([]byte{7, 7, 7, 7}, []byte{1, 200, 3, 4})
	f.Add([]byte{}, []byte{})

	f.Fuzz(func(t *testing.T, keys, values []byte) {
		if len(keys) > 64 {
			t.Skip("a big input proves nothing a small one does not")
		}

		k := byteColumn(t, keys)
		g, err := kernel.GroupBy(k)
		if err != nil {
			t.Fatalf("GroupBy: %v", err)
		}
		v := sameLength(t, values, k.Len())

		low := aggregate(t, "Min", func() (*array.Chunked, error) { return kernel.Min(v, g) })
		high := aggregate(t, "Max", func() (*array.Chunked, error) { return kernel.Max(v, g) })
		count := kernel.Count(v, g)
		distinct := aggregate(t, "NUnique", func() (*array.Chunked, error) { return kernel.NUnique(v, g) })
		variance := aggregate(t, "Var", func() (*array.Chunked, error) { return kernel.Var(v, g, 0) })
		sd := aggregate(t, "Std", func() (*array.Chunked, error) { return kernel.Std(v, g, 0) })

		// Every q the loop below asks for, in order, with the median where it
		// belongs so it is checked against its neighbors too.
		var qs []*array.Chunked
		for _, q := range []float64{0, 0.25, 0.5, 0.75, 1} {
			if q == 0.5 {
				qs = append(qs, aggregate(t, "Median",
					func() (*array.Chunked, error) { return kernel.Median(v, g) }))
				continue
			}
			qs = append(qs, aggregate(t, "Quantile",
				func() (*array.Chunked, error) { return kernel.Quantile(v, g, q, kernel.Linear) }))
		}

		for id := range g.NumGroups() {
			n := count.Value[int64](id)
			if d := distinct.Value[int64](id); d < 0 || d > n {
				t.Fatalf("group %d has %d values and %d distinct ones", id, n, d)
			}

			if n == 0 {
				// Nothing to be the smallest of, so every one of these is
				// missing and there is nothing else to check.
				for _, c := range append([]*array.Chunked{low, high, variance, sd}, qs...) {
					if !c.IsNull(id) {
						t.Fatalf("group %d has no values and an answer anyway", id)
					}
				}
				continue
			}

			if x := variance.Value[float64](id); x < 0 {
				t.Fatalf("group %d has a variance of %v", id, x)
			}
			if s, x := sd.Value[float64](id), variance.Value[float64](id); math.Abs(s*s-x) > 1e-9*(1+x) {
				t.Fatalf("group %d has a standard deviation of %v and a variance of %v", id, s, x)
			}

			smallest, largest := float64(low.Value[int64](id)), float64(high.Value[int64](id))
			last := math.Inf(-1)
			for i, c := range qs {
				got := c.Value[float64](id)
				if got < smallest || got > largest {
					t.Fatalf("quantile %d of group %d is %v, outside %v to %v",
						i, id, got, smallest, largest)
				}
				if got < last {
					t.Fatalf("quantile %d of group %d is %v, below the one before it at %v",
						i, id, got, last)
				}
				last = got
			}
		}
	})
}

// aggregate runs one aggregation where a failure means the fuzzer found a bug
// rather than a column these do not work on, since the column is always int64.
func aggregate(t *testing.T, name string, f func() (*array.Chunked, error)) *array.Chunked {
	t.Helper()

	c, err := f()
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return c
}

// FuzzJoin builds two sides out of the two inputs and checks the properties
// every join has, which is more than a table of hand written answers can reach
// once the fuzzer starts choosing the chunk boundaries and where the nulls go.
//
// An inner join contains exactly the pairs whose keys are equal and present, a
// left join is an inner join plus one unmatched row for each left row that
// contributed nothing, a semi join is the distinct left rows of an inner join,
// an anti join is the rest of them, and the two together are every left row
// exactly once.
func FuzzJoin(f *testing.F) {
	f.Add([]byte{1, 2, 3}, []byte{2, 3, 4})
	f.Add([]byte{1, 1, 1}, []byte{1, 1})
	f.Add([]byte{255, 1}, []byte{255, 255})
	f.Add([]byte{}, []byte{1})
	f.Add([]byte{1}, []byte{})

	f.Fuzz(func(t *testing.T, lb, rb []byte) {
		if len(lb) > 32 || len(rb) > 32 {
			t.Skip("a big input proves nothing a small one does not")
		}

		lk, rk := byteColumn(t, lb), byteColumn(t, rb)
		left := kernel.Side{Rows: lk.Len(), Keys: []*array.Chunked{lk}}
		right := kernel.Side{Rows: rk.Len(), Keys: []*array.Chunked{rk}}

		// The pairs an inner join has to produce, worked out the slow and
		// obvious way. A missing key matches nothing, including another one.
		var wantL, wantR []int
		for i := range left.Rows {
			if lk.IsNull(i) {
				continue
			}
			for j := range right.Rows {
				if rk.IsNull(j) {
					continue
				}
				if valueAt(t, lk, i) == valueAt(t, rk, j) {
					wantL = append(wantL, i)
					wantR = append(wantR, j)
				}
			}
		}

		inner, err := kernel.Join(left, right, kernel.InnerJoin)
		if err != nil {
			t.Fatalf("inner join: %v", err)
		}
		if !slices.Equal(inner.Left, wantL) || !slices.Equal(inner.Right, wantR) {
			t.Fatalf("the inner join is %v %v, want %v %v",
				inner.Left, inner.Right, wantL, wantR)
		}

		// A left join is the inner join with an unmatched row wherever a left
		// row produced nothing, and the left column is still in order.
		outer, err := kernel.Join(left, right, kernel.LeftJoin)
		if err != nil {
			t.Fatalf("left join: %v", err)
		}
		if !slices.IsSorted(outer.Left) {
			t.Fatalf("the left join is out of order: %v", outer.Left)
		}
		matched := map[int]bool{}
		for _, i := range wantL {
			matched[i] = true
		}
		want := len(wantL)
		for i := range left.Rows {
			if !matched[i] {
				want++
			}
		}
		if outer.Len() != want {
			t.Fatalf("the left join has %d rows, want %d", outer.Len(), want)
		}
		for k, j := range outer.Right {
			if j < 0 && matched[outer.Left[k]] {
				t.Fatalf("left row %d matched and has an unmatched row anyway", outer.Left[k])
			}
		}

		// Semi and anti split the left rows in two, each row landing in exactly
		// one of them, and semi is the ones the inner join used.
		semi, err := kernel.Join(left, right, kernel.SemiJoin)
		if err != nil {
			t.Fatalf("semi join: %v", err)
		}
		anti, err := kernel.Join(left, right, kernel.AntiJoin)
		if err != nil {
			t.Fatalf("anti join: %v", err)
		}
		if semi.Right != nil || anti.Right != nil {
			t.Fatal("a semi or anti join took something from the right side")
		}
		if semi.Len()+anti.Len() != left.Rows {
			t.Fatalf("semi and anti have %d and %d rows, the left side has %d",
				semi.Len(), anti.Len(), left.Rows)
		}
		for _, i := range semi.Left {
			if !matched[i] {
				t.Fatalf("the semi join kept left row %d, which matched nothing", i)
			}
		}
		for _, i := range anti.Left {
			if matched[i] {
				t.Fatalf("the anti join kept left row %d, which matched", i)
			}
		}

		// An outer join is the left join plus the right rows nothing used, and
		// every right row appears at least once.
		full, err := kernel.Join(left, right, kernel.OuterJoin)
		if err != nil {
			t.Fatalf("outer join: %v", err)
		}
		seen := map[int]bool{}
		for _, j := range full.Right {
			seen[j] = true
		}
		for j := range right.Rows {
			if !seen[j] {
				t.Fatalf("right row %d is in no outer join row", j)
			}
		}

		// A right join is the left join of the swapped sides, so the two have
		// to agree about how many pairs there are.
		back, err := kernel.Join(right, left, kernel.RightJoin)
		if err != nil {
			t.Fatalf("right join: %v", err)
		}
		if back.Len() != outer.Len() {
			t.Fatalf("the right join has %d rows and the left join has %d",
				back.Len(), outer.Len())
		}
	})
}
