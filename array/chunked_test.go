package array_test

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// chunkedInts returns a column whose chunks are the given lengths, holding
// 0, 1, 2 and so on across all of them, with every seventh value missing, and
// the model to check it against.
func chunkedInts(t *testing.T, lengths ...int) (*array.Chunked, []bool) {
	t.Helper()

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
		for range n {
			if next%7 == 0 {
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
	return c, present
}

func TestChunked(t *testing.T) {
	c, present := chunkedInts(t, 3, 5, 1, 7)

	if c.Len() != 16 {
		t.Fatalf("Len() = %d, want 16", c.Len())
	}
	if c.NumChunks() != 4 {
		t.Fatalf("NumChunks() = %d, want 4", c.NumChunks())
	}
	if !dtype.Equal(c.DType(), dtype.Int64) {
		t.Fatalf("DType() = %s", c.DType())
	}

	nulls := 0
	for i := range c.Len() {
		if !present[i] {
			nulls++
		}
		if c.IsValid(i) != present[i] || c.IsNull(i) == present[i] {
			t.Fatalf("IsValid(%d) = %v, want %v", i, c.IsValid(i), present[i])
		}
		if !present[i] {
			continue
		}
		if got := c.Value[int64](i); got != int64(i) {
			t.Fatalf("Value(%d) = %d, want %d", i, got, i)
		}
	}
	if c.NullCount() != nulls {
		t.Fatalf("NullCount() = %d, want %d", c.NullCount(), nulls)
	}

	want := "array.Chunked{int64, len 16, nulls 3, chunks 4}"
	if got := c.String(); got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
}

// TestChunkedSlice walks every range of a column with four chunks in it, so
// every combination of starting and ending inside a chunk, on a boundary, or
// past a whole chunk gets checked.
func TestChunkedSlice(t *testing.T) {
	c, present := chunkedInts(t, 3, 5, 1, 7)

	for i := range c.Len() + 1 {
		for j := i; j <= c.Len(); j++ {
			s := c.Slice(i, j)
			if s.Len() != j-i {
				t.Fatalf("Slice(%d, %d).Len() = %d, want %d", i, j, s.Len(), j-i)
			}

			nulls := 0
			for k := range s.Len() {
				if !present[i+k] {
					nulls++
				}
				if s.IsValid(k) != present[i+k] {
					t.Fatalf("Slice(%d, %d).IsValid(%d) = %v, want %v",
						i, j, k, s.IsValid(k), present[i+k])
				}
				if present[i+k] && s.Value[int64](k) != int64(i+k) {
					t.Fatalf("Slice(%d, %d).Value(%d) = %d, want %d",
						i, j, k, s.Value[int64](k), i+k)
				}
			}
			if s.NullCount() != nulls {
				t.Fatalf("Slice(%d, %d).NullCount() = %d, want %d", i, j, s.NullCount(), nulls)
			}
		}
	}
}

// TestChunkedSliceSharesWholeChunks is the reason slicing a chunked column of a
// million values is cheap. A chunk the range covers whole is handed over as it
// is, with no null count and no new array.
func TestChunkedSliceSharesWholeChunks(t *testing.T) {
	c, _ := chunkedInts(t, 3, 5, 1, 7)

	s := c.Slice(1, 15)
	if s.NumChunks() != 4 {
		t.Fatalf("NumChunks() = %d, want 4", s.NumChunks())
	}
	if s.Chunk(1) != c.Chunk(1) || s.Chunk(2) != c.Chunk(2) {
		t.Error("a chunk covered whole by the range was copied instead of shared")
	}
	if s.Chunk(0) == c.Chunk(0) || s.Chunk(3) == c.Chunk(3) {
		t.Error("a chunk covered in part by the range was shared instead of sliced")
	}

	// The whole column sliced from end to end is every chunk shared.
	whole := c.Slice(0, c.Len())
	for i := range whole.NumChunks() {
		if whole.Chunk(i) != c.Chunk(i) {
			t.Errorf("chunk %d of the whole column was copied", i)
		}
	}
}

func TestChunkedEmpty(t *testing.T) {
	c, err := array.NewChunked(dtype.Int64)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	if c.Len() != 0 || c.NullCount() != 0 || c.NumChunks() != 0 {
		t.Fatalf("%s, want an empty column", c)
	}
	if s := c.Slice(0, 0); s.Len() != 0 {
		t.Errorf("Slice(0, 0) of an empty column has %d values", s.Len())
	}
	if got := len(c.Chunks()); got != 0 {
		t.Errorf("Chunks() has %d entries", got)
	}
}

// TestChunkedDropsEmptyChunks is the rule that keeps every lookup from having
// to step over a chunk that can never hold the answer.
func TestChunkedDropsEmptyChunks(t *testing.T) {
	empty := array.Of[int64]()

	c, err := array.NewChunked(dtype.Int64, empty, array.Of[int64](1, 2), empty, array.Of[int64](3), empty)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	if c.NumChunks() != 2 {
		t.Fatalf("NumChunks() = %d, want 2", c.NumChunks())
	}
	if c.Len() != 3 {
		t.Fatalf("Len() = %d, want 3", c.Len())
	}
	for i := range c.Len() {
		if got := c.Value[int64](i); got != int64(i+1) {
			t.Errorf("Value(%d) = %d, want %d", i, got, i+1)
		}
	}
}

func TestChunkedAppend(t *testing.T) {
	first, err := array.NewChunked(dtype.Int64, array.Of[int64](1, 2))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	second, err := first.Append(array.Of[int64](3), array.Of[int64](4, 5))
	if err != nil {
		t.Fatalf("Append: %v", err)
	}

	if first.Len() != 2 || first.NumChunks() != 1 {
		t.Errorf("Append changed the column it was called on: %s", first)
	}
	if second.Len() != 5 || second.NumChunks() != 3 {
		t.Fatalf("%s, want 5 values in 3 chunks", second)
	}
	for i := range second.Len() {
		if got := second.Value[int64](i); got != int64(i+1) {
			t.Errorf("Value(%d) = %d, want %d", i, got, i+1)
		}
	}

	// Appending twice to the same column gives two columns, not one column and
	// one surprise, which is the thing a shared backing slice would get wrong.
	a, err := first.Append(array.Of[int64](10))
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	b, err := first.Append(array.Of[int64](20))
	if err != nil {
		t.Fatalf("Append: %v", err)
	}
	if a.Value[int64](2) != 10 || b.Value[int64](2) != 20 {
		t.Errorf("the two appends interfered: %d and %d", a.Value[int64](2), b.Value[int64](2))
	}
}

func TestChunkedBoolAndBytes(t *testing.T) {
	bools, err := array.NewChunked(dtype.Bool, array.OfBools(true, false), array.OfBools(false, true, true))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	for i, want := range []bool{true, false, false, true, true} {
		if got := bools.Bool(i); got != want {
			t.Errorf("Bool(%d) = %v, want %v", i, got, want)
		}
	}

	strs, err := array.NewChunked(dtype.String, array.OfStrings("kuma"), array.OfStrings("bear", "cub"))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	for i, want := range []string{"kuma", "bear", "cub"} {
		if got := string(strs.Bytes(i)); got != want {
			t.Errorf("Bytes(%d) = %q, want %q", i, got, want)
		}
	}
}

func TestNewChunkedErrors(t *testing.T) {
	tests := []struct {
		name   string
		dt     dtype.DataType
		chunks []*array.Array
		want   string
	}{
		{"nil dtype", nil, nil, "nil dtype"},
		{"nil chunk", dtype.Int64, []*array.Array{array.Of[int64](1), nil}, "chunk 1 is nil"},
		{"wrong type", dtype.Int64, []*array.Array{array.Of[int64](1), array.Of[int32](2)}, "chunk 1 is a int32 column, want int64"},
		{"same kind, different parameters", dtype.Timestamp{Unit: dtype.Second},
			[]*array.Array{mustTimestamp(t, dtype.Timestamp{Unit: dtype.Nanosecond})}, "chunk 0 is a"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, err := array.NewChunked(tt.dt, tt.chunks...)
			if err == nil {
				t.Fatalf("NewChunked returned %s, want an error", c)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error is %q, want it to mention %q", err, tt.want)
			}
		})
	}
}

func mustTimestamp(t *testing.T, dt dtype.DataType) *array.Array {
	t.Helper()
	b, err := array.NewBuilder(dt)
	if err != nil {
		t.Fatalf("NewBuilder(%s): %v", dt, err)
	}
	b.Append[int64](1)
	return b.Finish()
}

func TestChunkedPanics(t *testing.T) {
	c, _ := chunkedInts(t, 3, 5)

	tests := []struct {
		name string
		fn   func()
		want string
	}{
		{"index too high", func() { c.IsValid(8) }, "index out of range"},
		{"index negative", func() { c.Value[int64](-1) }, "index out of range"},
		{"index on an empty column", func() {
			empty, err := array.NewChunked(dtype.Int64)
			if err != nil {
				t.Fatalf("NewChunked: %v", err)
			}
			empty.IsValid(0)
		}, "index out of range"},
		{"chunk out of range", func() { c.Chunk(2) }, "chunk index out of range"},
		{"slice past the end", func() { c.Slice(0, 9) }, "Slice(0, 9) of a column of 8 values"},
		{"slice backwards", func() { c.Slice(5, 4) }, "Slice(5, 4) of a column of 8 values"},
		{"slice from before the start", func() { c.Slice(-1, 4) }, "Slice(-1, 4) of a column of 8 values"},
		{"the wrong type", func() { c.Value[float64](0) }, "cannot read a int64 column as float64"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("did not panic")
				}
				msg, ok := r.(string)
				if !ok {
					t.Fatalf("panicked with %T, want a string", r)
				}
				if !strings.Contains(msg, tt.want) {
					t.Errorf("panicked with %q, want it to mention %q", msg, tt.want)
				}
			}()
			tt.fn()
		})
	}
}

// TestChunkedSliceOfSlice checks that a sliced column slices again correctly,
// since the second slice is working from starts that no longer begin at zero in
// the arrays underneath.
func TestChunkedSliceOfSlice(t *testing.T) {
	c, present := chunkedInts(t, 4, 4, 4)

	s := c.Slice(2, 11).Slice(1, 6)
	if s.Len() != 5 {
		t.Fatalf("Len() = %d, want 5", s.Len())
	}
	for k := range s.Len() {
		i := 3 + k
		if s.IsValid(k) != present[i] {
			t.Fatalf("IsValid(%d) = %v, want %v", k, s.IsValid(k), present[i])
		}
		if present[i] && s.Value[int64](k) != int64(i) {
			t.Fatalf("Value(%d) = %d, want %d", k, s.Value[int64](k), i)
		}
	}
}
