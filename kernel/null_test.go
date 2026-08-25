package kernel_test

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// mask renders a boolean column as a string of t and f, so a test can write
// down what it expects in one line.
func mask(t *testing.T, c *array.Chunked) string {
	t.Helper()

	if c.DType() != dtype.Bool {
		t.Fatalf("the mask is a %s column", c.DType())
	}
	if c.NullCount() != 0 {
		t.Fatal("the mask has nulls in it, and whether a value is missing is always known")
	}

	var b strings.Builder
	for i := range c.Len() {
		if c.Bool(i) {
			b.WriteByte('t')
			continue
		}
		b.WriteByte('f')
	}
	return b.String()
}

func TestIsNull(t *testing.T) {
	tests := []struct {
		name   string
		col    *array.Chunked
		isNull string
	}{
		{"mixed", col(t, dtype.Int64, []any{int64(1), nil, int64(3)}), "ftf"},
		{"none missing", col(t, dtype.Int64, []any{int64(1), int64(2)}), "ff"},
		{"all missing", col(t, dtype.Int64, []any{nil, nil}), "tt"},
		{"empty", col(t, dtype.Int64), ""},
		{"chunked", col(t, dtype.Int64,
			[]any{int64(1), nil}, []any{nil, int64(4)}), "fttf"},
		{"strings", col(t, dtype.String, []any{"a", nil}), "ft"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := mask(t, kernel.IsNull(tt.col)); got != tt.isNull {
				t.Errorf("IsNull is %q, want %q", got, tt.isNull)
			}

			// IsNotNull is the same answer the other way round, every time.
			want := strings.Map(func(r rune) rune {
				if r == 't' {
					return 'f'
				}
				return 't'
			}, tt.isNull)
			if got := mask(t, kernel.IsNotNull(tt.col)); got != want {
				t.Errorf("IsNotNull is %q, want %q", got, want)
			}
		})
	}
}

func TestIsNullOfNothing(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("IsNull of a nil column did not panic")
		}
	}()
	kernel.IsNull(nil)
}

func TestFillNull(t *testing.T) {
	c := col(t, dtype.Int64, []any{int64(1), nil, int64(3)}, []any{nil})

	got, err := kernel.FillNull(c, array.Of(int64(7)))
	if err != nil {
		t.Fatalf("FillNull: %v", err)
	}
	if got.NullCount() != 0 {
		t.Errorf("%d values are still missing", got.NullCount())
	}

	want := []int64{1, 7, 3, 7}
	for i, v := range want {
		if got.Value[int64](i) != v {
			t.Errorf("row %d is %d, want %d", i, got.Value[int64](i), v)
		}
	}
}

func TestFillNullStrings(t *testing.T) {
	c := col(t, dtype.String, []any{"a", nil, "c"})

	got, err := kernel.FillNull(c, array.OfStrings("?"))
	if err != nil {
		t.Fatalf("FillNull: %v", err)
	}

	want := []string{"a", "?", "c"}
	for i, v := range want {
		if string(got.Bytes(i)) != v {
			t.Errorf("row %d is %q, want %q", i, got.Bytes(i), v)
		}
	}
}

// TestFillNullNothingToDo is the column with no nulls, which is handed back as
// it is rather than copied.
func TestFillNullNothingToDo(t *testing.T) {
	c := col(t, dtype.Int64, []any{int64(1), int64(2)})

	got, err := kernel.FillNull(c, array.Of(int64(7)))
	if err != nil {
		t.Fatalf("FillNull: %v", err)
	}
	if got != c {
		t.Error("filling a column with nothing missing built a new one")
	}
}

func TestFillNullEveryType(t *testing.T) {
	tests := []struct {
		name string
		col  *array.Chunked
		fill *array.Array
	}{
		{"bool", col(t, dtype.Bool, []any{true, nil}), array.OfBools(false)},
		{"int8", col(t, dtype.Int8, []any{int8(1), nil}), array.Of(int8(7))},
		{"uint32", col(t, dtype.Uint32, []any{uint32(1), nil}), array.Of(uint32(7))},
		{"float64", col(t, dtype.Float64, []any{1.5, nil}), array.Of(2.5)},
		{"string", col(t, dtype.String, []any{"a", nil}), array.OfStrings("z")},
		{"timestamp", col(t, dtype.Timestamp{Unit: dtype.Nanosecond}, []any{int64(1), nil}), tsArray(t, 7)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := kernel.FillNull(tt.col, tt.fill)
			if err != nil {
				t.Fatalf("FillNull: %v", err)
			}
			if got.NullCount() != 0 {
				t.Errorf("%d values are still missing", got.NullCount())
			}
			if got.DType() != tt.col.DType() {
				t.Errorf("the result is a %s column, want %s", got.DType(), tt.col.DType())
			}
		})
	}
}

// tsArray returns a one value timestamp array, which array.Of cannot build
// because the Go type it takes decides the dtype and int64 means int64.
func tsArray(t *testing.T, v int64) *array.Array {
	t.Helper()

	b, err := array.NewBuilder(dtype.Timestamp{Unit: dtype.Nanosecond})
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	b.Append(v)
	return b.Finish()
}

func TestFillNullErrors(t *testing.T) {
	c := col(t, dtype.Int64, []any{int64(1), nil})

	if _, err := kernel.FillNull(c, nil); err == nil {
		t.Error("filling with no value succeeded")
	}
	if _, err := kernel.FillNull(c, array.Of(int64(1), int64(2))); err == nil {
		t.Error("filling with two values succeeded")
	}
	if _, err := kernel.FillNull(c, array.Of(1.0)); err == nil {
		t.Error("filling an int64 column with a float64 succeeded")
	}

	missing := col(t, dtype.Int64, []any{nil})
	if _, err := kernel.FillNull(c, missing.Chunk(0)); err == nil {
		t.Error("filling nulls with a null succeeded")
	}
}

func TestFillNullOfNothing(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("filling a nil column did not panic")
		}
	}()
	if _, err := kernel.FillNull(nil, array.Of(int64(1))); err != nil {
		t.Fatalf("FillNull: %v", err)
	}
}

// keepCase is one call to KeepIndex written down as the columns, the threshold
// and the rows that should survive.
type keepCase struct {
	name    string
	cols    [][]any
	present int
	want    []int
}

func TestKeepIndex(t *testing.T) {
	// The three columns have a hole in a different place each, so no two of
	// them fail on the same row.
	sym := []any{nil, "b", "c", nil}
	qty := []any{int64(1), nil, int64(3), nil}
	full := []any{int64(9), int64(9), int64(9), int64(9)}

	tests := []keepCase{
		{"none of them", [][]any{qty}, 0, []int{0, 1, 2, 3}},
		{"below none of them", [][]any{qty}, -1, []int{0, 1, 2, 3}},
		{"one column", [][]any{qty}, 1, []int{0, 2}},
		{"one complete column", [][]any{full}, 1, []int{0, 1, 2, 3}},
		{"either of two", [][]any{qty, full}, 1, []int{0, 1, 2, 3}},
		{"both of two", [][]any{qty, full}, 2, []int{0, 2}},
		{"more than there are", [][]any{qty}, 2, nil},
		{"two of three", [][]any{qty, full, full}, 2, []int{0, 1, 2, 3}},
		{"all three", [][]any{qty, full, full}, 3, []int{0, 2}},
	}

	for _, tt := range tests {
		cols := make([]*array.Chunked, len(tt.cols))
		for i, values := range tt.cols {
			cols[i] = col(t, dtype.Int64, values)
		}

		got := kernel.KeepIndex(cols, 4, tt.present)
		if !equalInts(got, tt.want) {
			t.Errorf("%s: KeepIndex gave %v, want %v", tt.name, got, tt.want)
		}
	}

	// The string column takes the same path, and a mixture of types in one call
	// is the ordinary case, since the columns of a frame are whatever they are.
	mixed := []*array.Chunked{col(t, dtype.String, sym), col(t, dtype.Int64, qty)}
	if got := kernel.KeepIndex(mixed, 4, 2); !equalInts(got, []int{2}) {
		t.Errorf("KeepIndex over two types gave %v, want [2]", got)
	}
	if got := kernel.KeepIndex(mixed, 4, 1); !equalInts(got, []int{0, 1, 2}) {
		t.Errorf("KeepIndex over two types gave %v, want [0 1 2]", got)
	}
}

// TestKeepIndexChunked is the case the row numbering has to get right, since
// each chunk counts its own nulls and its own positions.
func TestKeepIndexChunked(t *testing.T) {
	one := col(t, dtype.Int64, []any{int64(1), nil}, []any{nil, int64(4)})
	two := col(t, dtype.Int64, []any{int64(1), int64(2), int64(3)}, []any{nil})

	if got := kernel.KeepIndex([]*array.Chunked{one}, 4, 1); !equalInts(got, []int{0, 3}) {
		t.Errorf("KeepIndex gave %v, want [0 3]", got)
	}
	if got := kernel.KeepIndex([]*array.Chunked{one, two}, 4, 2); !equalInts(got, []int{0}) {
		t.Errorf("KeepIndex gave %v, want [0]", got)
	}
	if got := kernel.KeepIndex([]*array.Chunked{one, two}, 4, 1); !equalInts(got, []int{0, 1, 2, 3}) {
		t.Errorf("KeepIndex gave %v, want [0 1 2 3]", got)
	}
}

// TestKeepIndexSliced is the same again for a chunk that does not begin at the
// start of the bitmap it shares, which is what slicing a column leaves behind.
func TestKeepIndexSliced(t *testing.T) {
	c := col(t, dtype.Int64, []any{int64(1), nil, int64(3), nil, int64(5)}).Slice(1, 5)

	if got := kernel.KeepIndex([]*array.Chunked{c}, 4, 1); !equalInts(got, []int{1, 3}) {
		t.Errorf("KeepIndex over a sliced column gave %v, want [1 3]", got)
	}

	full := col(t, dtype.Int64, []any{int64(1), int64(2), int64(3), int64(4), int64(5)}).Slice(1, 5)
	if got := kernel.KeepIndex([]*array.Chunked{c, full}, 4, 2); !equalInts(got, []int{1, 3}) {
		t.Errorf("KeepIndex over a sliced column gave %v, want [1 3]", got)
	}
}

// TestKeepIndexNullColumn is the type that says every value is missing, which
// has a null count and no bitmap to read it from.
func TestKeepIndexNullColumn(t *testing.T) {
	c, err := array.NewChunked(dtype.Null, array.NewNull(4))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	if got := kernel.KeepIndex([]*array.Chunked{c}, 4, 1); len(got) != 0 {
		t.Errorf("KeepIndex over a null column gave %v, want nothing", got)
	}

	full := col(t, dtype.Int64, []any{int64(1), int64(2), int64(3), int64(4)})
	if got := kernel.KeepIndex([]*array.Chunked{c, full}, 4, 1); !equalInts(got, []int{0, 1, 2, 3}) {
		t.Errorf("KeepIndex over a null column gave %v, want every row", got)
	}
	if got := kernel.KeepIndex([]*array.Chunked{c, full}, 4, 2); len(got) != 0 {
		t.Errorf("KeepIndex over a null column gave %v, want nothing", got)
	}
}

func TestKeepIndexWrongLength(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("a column of the wrong length did not panic")
		}
	}()

	c := col(t, dtype.Int64, []any{int64(1), nil})
	indexSink = kernel.KeepIndex([]*array.Chunked{c}, 3, 1)
}

func TestKeepIndexNilColumn(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("a nil column did not panic")
		}
	}()
	indexSink = kernel.KeepIndex([]*array.Chunked{nil}, 3, 1)
}

func equalInts(got, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// TestIsNullOfNullColumn is the type that says every value is missing, which
// has a null count and no bitmap to read it from.
func TestIsNullOfNullColumn(t *testing.T) {
	c, err := array.NewChunked(dtype.Null, array.NewNull(3))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	if got := mask(t, kernel.IsNull(c)); got != "ttt" {
		t.Errorf("IsNull over a null column gave %q, want %q", got, "ttt")
	}
	if got := mask(t, kernel.IsNotNull(c)); got != "fff" {
		t.Errorf("IsNotNull over a null column gave %q, want %q", got, "fff")
	}
}

// TestKeepIndexCompleteChunk is a column that has nulls in one chunk and not in
// the other, so the chunk that cannot fail is counted rather than read.
func TestKeepIndexCompleteChunk(t *testing.T) {
	c := col(t, dtype.Int64, []any{int64(1), int64(2)}, []any{nil, int64(4)})

	if got := kernel.KeepIndex([]*array.Chunked{c}, 4, 1); !equalInts(got, []int{0, 1, 3}) {
		t.Errorf("KeepIndex gave %v, want [0 1 3]", got)
	}

	full := col(t, dtype.Int64, []any{int64(1), int64(2), int64(3), int64(4)})
	if got := kernel.KeepIndex([]*array.Chunked{c, full}, 4, 2); !equalInts(got, []int{0, 1, 3}) {
		t.Errorf("KeepIndex gave %v, want [0 1 3]", got)
	}
}
