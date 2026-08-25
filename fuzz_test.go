package kuma_test

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// FuzzFrameSlice checks a sliced frame against a flat model of the same rows.
// The fuzzer picks the chunk lengths of the column underneath as well as the
// range, so a range that lands on a chunk boundary and one that falls inside a
// chunk are both reachable, and the nullability in the schema has to agree with
// the nulls that are actually in the range.
func FuzzFrameSlice(f *testing.F) {
	f.Add([]byte{3, 5, 1, 7}, 2, 11)
	f.Add([]byte{1}, 0, 1)
	f.Add([]byte{0, 4, 0}, 1, 3)
	f.Add([]byte{}, 0, 0)

	f.Fuzz(func(t *testing.T, lengths []byte, i, j int) {
		if len(lengths) > 32 {
			t.Skip()
		}

		var (
			chunks  []*array.Array
			present []bool
			next    int64
		)
		for _, n := range lengths {
			b, err := array.NewBuilder(dtype.Int64)
			if err != nil {
				t.Fatalf("NewBuilder: %v", err)
			}
			for range int(n) % 20 {
				if next%4 == 0 {
					b.AppendNull()
					present = append(present, false)
				} else {
					b.Append(next)
					present = append(present, true)
				}
				next++
			}
			chunks = append(chunks, b.Finish())
		}

		c, err := array.NewChunked(dtype.Int64, chunks...)
		if err != nil {
			t.Fatalf("NewChunked: %v", err)
		}
		col, err := kuma.NewColumn("qty", c)
		if err != nil {
			t.Fatalf("NewColumn: %v", err)
		}
		frame, err := kuma.NewFrame(col)
		if err != nil {
			t.Fatalf("NewFrame: %v", err)
		}
		if frame.NumRows() != len(present) {
			t.Fatalf("%s, want %d rows", frame, len(present))
		}
		if i < 0 || j < i || j > frame.NumRows() {
			t.Skip()
		}

		cut := frame.Slice(i, j)
		if cut.NumRows() != j-i {
			t.Fatalf("Slice(%d, %d) has %d rows, want %d", i, j, cut.NumRows(), j-i)
		}

		s, err := cut.Series[int64]("qty")
		if err != nil {
			t.Fatalf("Series: %v", err)
		}

		nulls := 0
		for k := range s.Len() {
			if !present[i+k] {
				nulls++
			}
			if s.IsValid(k) != present[i+k] {
				t.Fatalf("Slice(%d, %d) has value %d %v, want %v",
					i, j, k, s.IsValid(k), present[i+k])
			}
			if present[i+k] && s.Value(k) != int64(i+k) {
				t.Fatalf("Slice(%d, %d).Value(%d) = %d, want %d", i, j, k, s.Value(k), i+k)
			}
		}
		if s.NullCount() != nulls {
			t.Fatalf("Slice(%d, %d) has %d nulls, want %d", i, j, s.NullCount(), nulls)
		}

		// Values agrees with Value, whichever way the chunks fell.
		values := s.Values()
		if len(values) != s.Len() {
			t.Fatalf("Values() has %d values, the column has %d", len(values), s.Len())
		}
		for k := range values {
			if values[k] != s.Value(k) {
				t.Fatalf("Values()[%d] = %d, Value(%d) = %d", k, values[k], k, s.Value(k))
			}
		}

		// A frame's schema describes its data, so the column is nullable if and
		// only if the range it holds has a null in it.
		if got := cut.Schema().Fields[0].Nullable; got != (nulls > 0) {
			t.Fatalf("Slice(%d, %d) is nullable %v with %d nulls in it", i, j, got, nulls)
		}
	})
}

// FuzzColumnError checks the message for a column that is not there. It is the
// error users see most often, it is built out of whatever names the data had,
// and none of that is under our control, so the properties worth holding are
// that it never panics, that it names the column that was asked for, and that a
// suggestion is always one of the names the frame actually has.
func FuzzColumnError(f *testing.F) {
	f.Add("sym", "symbol,price,qty,side")
	f.Add("", "")
	f.Add("price", "price")
	f.Add(strings.Repeat("x", 300), "x,xx,xxx")

	f.Fuzz(func(t *testing.T, name, names string) {
		var have []string
		if names != "" {
			have = strings.Split(names, ",")
		}

		// The message is read back below to find the suggestion in it, so the
		// names have to be ones that cannot forge a line of it. That is a limit
		// of the test rather than of the code: the property under test is which
		// name gets suggested, not whether our own output can be parsed.
		if strings.ContainsAny(name+names, ":\n") {
			t.Skip()
		}

		err := &kuma.ColumnError{Op: "Select", Name: name, Have: have}
		msg := err.Error()

		if !errors.Is(err, kuma.ErrNoColumn) {
			t.Fatal("the error is not an ErrNoColumn")
		}
		if !strings.Contains(msg, "Select") {
			t.Fatalf("the message does not name the operation: %q", msg)
		}

		const marker = "did you mean: "
		k := strings.Index(msg, marker)
		if k < 0 {
			// A name that is only the wrong case is always close enough, so
			// the absence of a suggestion says there is no such name.
			for _, h := range have {
				if strings.EqualFold(h, name) {
					t.Fatalf("no suggestion for %q, which is %q in another case", name, h)
				}
			}
			return
		}

		suggestion := strings.TrimSuffix(msg[k+len(marker):], "?")
		if !slices.Contains(have, suggestion) {
			t.Fatalf("the message suggests %q, which is not one of %q", suggestion, have)
		}
	})
}

// FuzzConcat checks stacked frames against a flat model of the same values. The
// fuzzer picks how many frames there are and how many rows each one has, so a
// frame with no rows in the middle of the list and a list with one frame in it
// are both reachable.
func FuzzConcat(f *testing.F) {
	f.Add([]byte{2, 3})
	f.Add([]byte{0, 1, 0})
	f.Add([]byte{5})
	f.Add([]byte{})

	f.Fuzz(func(t *testing.T, lengths []byte) {
		if len(lengths) == 0 || len(lengths) > 16 {
			t.Skip()
		}

		var (
			frames []*kuma.Frame[kuma.Dynamic]
			want   []int64
			next   int64
		)
		for _, n := range lengths {
			values := make([]int64, int(n)%20)
			for i := range values {
				values[i] = next
				next++
			}
			want = append(want, values...)

			frame, err := kuma.NewFrame(kuma.NewSeries("v", values...).Column())
			if err != nil {
				t.Fatalf("NewFrame: %v", err)
			}
			frames = append(frames, frame)
		}

		got, err := kuma.Concat(frames...)
		if err != nil {
			t.Fatalf("Concat: %v", err)
		}
		if got.NumRows() != len(want) {
			t.Fatalf("there are %d rows, want %d", got.NumRows(), len(want))
		}

		s, err := got.Series[int64]("v")
		if err != nil {
			t.Fatalf("Series: %v", err)
		}
		for i, v := range want {
			if s.Value(i) != v {
				t.Fatalf("row %d is %d, want %d", i, s.Value(i), v)
			}
		}

		// Stacking the same frames in a union has to give the same answer,
		// since they all hold the same one column.
		union, err := kuma.ConcatUnion(frames...)
		if err != nil {
			t.Fatalf("ConcatUnion: %v", err)
		}
		if union.NumRows() != len(want) {
			t.Fatalf("the union has %d rows, want %d", union.NumRows(), len(want))
		}
		if union.NumCols() != 1 {
			t.Fatalf("the union has %d columns, want 1", union.NumCols())
		}

		// A slice of the stacked frame is the same rows, which is the check
		// that the chunk boundaries the stacking created are where it says.
		if len(want) > 1 {
			mid := len(want) / 2
			tail, err := kuma.Concat(got.Slice(0, mid), got.Slice(mid, len(want)))
			if err != nil {
				t.Fatalf("Concat of the halves: %v", err)
			}
			half, err := tail.Series[int64]("v")
			if err != nil {
				t.Fatalf("Series: %v", err)
			}
			if !slices.Equal(half.Values(), s.Values()) {
				t.Fatal("stacking the two halves back up gave different values")
			}
		}
	})
}
