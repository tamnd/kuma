package array_test

import (
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
)

// int64List is the type most of these use, a list of numbers being the shape a
// repeated column in a file usually has.
var int64List = dtype.List{Elem: dtype.Int64}

// lists builds a list array out of the rows given, with no nulls.
func lists(t *testing.T, rows ...[]int64) *array.Array {
	t.Helper()

	b, err := array.NewListBuilder(int64List)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}
	for _, r := range rows {
		b.Elem().AppendValues(r)
		b.Append()
	}
	return b.Finish()
}

// rowsOf reads a list array back as the rows it holds, with a null row coming
// back nil so that a test can tell it from an empty one.
func rowsOf(t *testing.T, a *array.Array) [][]int64 {
	t.Helper()

	out := make([][]int64, a.Len())
	for i := range a.Len() {
		if a.IsNull(i) {
			continue
		}
		out[i] = slices.Clone(a.List(i).Values[int64]())
	}
	return out
}

func TestListBuilder(t *testing.T) {
	a := lists(t, []int64{1, 2, 3}, nil, []int64{4})

	if a.Len() != 3 {
		t.Errorf("Len = %d, want 3", a.Len())
	}
	if a.NullCount() != 0 {
		t.Errorf("NullCount = %d, want 0", a.NullCount())
	}
	if !dtype.Equal(a.DType(), int64List) {
		t.Errorf("DType = %s, want %s", a.DType(), int64List)
	}
	if want := []int32{0, 3, 3, 4}; !slices.Equal(a.Offsets(), want) {
		t.Errorf("Offsets = %v, want %v", a.Offsets(), want)
	}
	if a.Child().Len() != 4 {
		t.Errorf("the child is %d elements, want 4", a.Child().Len())
	}

	want := [][]int64{{1, 2, 3}, {}, {4}}
	for i, w := range want {
		if got := a.List(i).Values[int64](); !slices.Equal(got, w) {
			t.Errorf("row %d is %v, want %v", i, got, w)
		}
	}
}

// TestListBuilderEmptyRowIsNotNull is the difference the bitmap exists for,
// since the offsets of an empty row and of a null row are the same two numbers.
func TestListBuilderEmptyRowIsNotNull(t *testing.T) {
	b, err := array.NewListBuilder(int64List)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}
	b.Append()
	b.AppendNull()
	a := b.Finish()

	if a.IsNull(0) {
		t.Error("the empty row reads as null")
	}
	if !a.IsNull(1) {
		t.Error("the null row reads as present")
	}
	if a.NullCount() != 1 {
		t.Errorf("NullCount = %d, want 1", a.NullCount())
	}
	if got := a.List(1).Len(); got != 0 {
		t.Errorf("the null row holds %d elements, want 0", got)
	}
}

// TestListBuilderNullAfterValues is the case the bitmap is filled in late for,
// where nothing was missing until it was.
func TestListBuilderNullAfterValues(t *testing.T) {
	b, err := array.NewListBuilder(int64List)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}
	b.Elem().AppendValues([]int64{1})
	b.Append()
	b.AppendNull()
	b.Elem().AppendValues([]int64{2, 3})
	b.Append()

	a := b.Finish()
	if got, want := rowsOf(t, a), [][]int64{{1}, nil, {2, 3}}; !slices.EqualFunc(got, want, slices.Equal) {
		t.Errorf("the rows are %v, want %v", got, want)
	}
	if a.IsValid(0) != true || a.IsValid(2) != true {
		t.Error("a row appended before the first null reads as missing")
	}
}

// TestListBuilderDropsAnUnclosedRow is what happens to elements appended and
// never turned into a row, which is that they are half a row and go nowhere.
func TestListBuilderDropsAnUnclosedRow(t *testing.T) {
	b, err := array.NewListBuilder(int64List)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}
	b.Elem().AppendValues([]int64{1, 2})
	b.Append()
	b.Elem().AppendValues([]int64{3, 4, 5})

	a := b.Finish()
	if a.Len() != 1 {
		t.Errorf("Len = %d, want 1", a.Len())
	}
	if got := a.Child().Len(); got != 2 {
		t.Errorf("the child is %d elements, want 2", got)
	}
}

func TestListBuilderReset(t *testing.T) {
	b, err := array.NewListBuilder(int64List)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}
	b.Elem().AppendValues([]int64{1, 2})
	b.Append()
	b.AppendNull()

	// Finish resets, so a second Finish is a column of nothing rather than the
	// same column again.
	first := b.Finish()
	second := b.Finish()

	if first.Len() != 2 {
		t.Errorf("the first Finish gave %d rows, want 2", first.Len())
	}
	if second.Len() != 0 {
		t.Errorf("the second Finish gave %d rows, want 0", second.Len())
	}
	if second.NullCount() != 0 {
		t.Errorf("the second Finish kept %d nulls", second.NullCount())
	}
	if got := first.List(0).Values[int64](); !slices.Equal(got, []int64{1, 2}) {
		t.Errorf("the first column changed to %v", got)
	}
}

func TestListBuilderGrow(t *testing.T) {
	b, err := array.NewListBuilder(int64List)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}
	b.Grow(64)
	b.Elem().Grow(256)

	if b.Len() != 0 {
		t.Errorf("Grow added %d rows", b.Len())
	}
	b.Elem().AppendValues([]int64{7})
	b.Append()

	a := b.Finish()
	if got := a.List(0).Values[int64](); !slices.Equal(got, []int64{7}) {
		t.Errorf("the row is %v, want [7]", got)
	}
}

func TestListBuilderRejectsWhatItCannotBuild(t *testing.T) {
	if _, err := array.NewListBuilder(nil); err == nil {
		t.Error("NewListBuilder on a nil dtype gave no error")
	}
	if _, err := array.NewListBuilder(dtype.Int64); err == nil {
		t.Error("NewListBuilder on an int64 column gave no error")
	}
}

// offsets turns a run of numbers into the buffer NewList wants.
func offsets(t *testing.T, vals ...int32) *buffer.Buffer {
	t.Helper()

	buf := buffer.New(0)
	for _, v := range vals {
		buf.Append([]byte{byte(v), byte(v >> 8), byte(v >> 16), byte(v >> 24)})
	}
	return buf
}

func TestNewList(t *testing.T) {
	child := array.Of[int64](10, 20, 30, 40)

	a, err := array.NewList(int64List, 2, offsets(t, 0, 3, 4), child, nil)
	if err != nil {
		t.Fatalf("NewList: %v", err)
	}
	if got, want := rowsOf(t, a), [][]int64{{10, 20, 30}, {40}}; !slices.EqualFunc(got, want, slices.Equal) {
		t.Errorf("the rows are %v, want %v", got, want)
	}
	if a.Child() != child {
		t.Error("Child gave back something other than the child it was built over")
	}
}

// TestNewListKeepsTheChildWhole is the rule that the offsets are positions in
// the shared child, so a column that does not reach the start of it still has
// all of it.
func TestNewListKeepsTheChildWhole(t *testing.T) {
	child := array.Of[int64](10, 20, 30, 40)

	a, err := array.NewList(int64List, 1, offsets(t, 2, 4), child, nil)
	if err != nil {
		t.Fatalf("NewList: %v", err)
	}
	if got := a.Child().Len(); got != 4 {
		t.Errorf("the child is %d elements, want all 4", got)
	}
	if got := a.List(0).Values[int64](); !slices.Equal(got, []int64{30, 40}) {
		t.Errorf("the row is %v, want [30 40]", got)
	}
}

func TestNewListWithNulls(t *testing.T) {
	valid := bitmap.New(2)
	valid.Set(0, true)

	a, err := array.NewList(int64List, 2, offsets(t, 0, 2, 2), array.Of[int64](1, 2), valid)
	if err != nil {
		t.Fatalf("NewList: %v", err)
	}
	if a.NullCount() != 1 {
		t.Errorf("NullCount = %d, want 1", a.NullCount())
	}
	if !a.IsNull(1) {
		t.Error("row 1 reads as present")
	}
}

func TestNewListMistakes(t *testing.T) {
	child := array.Of[int64](1, 2, 3)

	tests := []struct {
		name    string
		dt      dtype.DataType
		length  int
		offsets *buffer.Buffer
		child   *array.Array
		want    string
	}{
		{"nil dtype", nil, 1, offsets(t, 0, 1), child, "nil dtype"},
		{"not a list", dtype.Int64, 1, offsets(t, 0, 1), child, "NewList on a"},
		{"large list", dtype.LargeList{Elem: dtype.Int64}, 1, offsets(t, 0, 1), child, "NewList on a"},
		{"negative length", int64List, -1, offsets(t, 0), child, "negative length"},
		{"nil child", int64List, 1, offsets(t, 0, 1), nil, "nil child"},
		{"wrong child", dtype.List{Elem: dtype.Float64}, 1, offsets(t, 0, 1), child, "over a int64 child"},
		{"nil offsets", int64List, 1, nil, child, "nil offsets"},
		{"short offsets", int64List, 4, offsets(t, 0, 1), child, "bytes of offsets"},
		{"negative offset", int64List, 1, offsets(t, -1, 1), child, "starts at offset"},
		{"backwards", int64List, 2, offsets(t, 0, 2, 1), child, "back to"},
		{"past the child", int64List, 1, offsets(t, 0, 9), child, "over a child of 3"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := array.NewList(tt.dt, tt.length, tt.offsets, tt.child, nil)
			if err == nil {
				t.Fatalf("NewList gave no error")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("the message is %q, want it to mention %q", err, tt.want)
			}
		})
	}
}

func TestNewListShortValidity(t *testing.T) {
	_, err := array.NewList(int64List, 2, offsets(t, 0, 1, 2), array.Of[int64](1, 2), bitmap.New(1))
	if err == nil {
		t.Fatal("NewList with a bitmap too short gave no error")
	}
}

func TestListSlice(t *testing.T) {
	a := lists(t, []int64{1}, []int64{2, 3}, nil, []int64{4, 5, 6})

	s := a.Slice(1, 3)
	if s.Len() != 2 {
		t.Fatalf("Len = %d, want 2", s.Len())
	}
	if got, want := rowsOf(t, s), [][]int64{{2, 3}, {}}; !slices.EqualFunc(got, want, slices.Equal) {
		t.Errorf("the rows are %v, want %v", got, want)
	}

	// The offsets are absolute, so slicing does not rewrite them and the child
	// is still the whole child.
	if want := []int32{1, 3, 3}; !slices.Equal(s.Offsets(), want) {
		t.Errorf("Offsets = %v, want %v", s.Offsets(), want)
	}
	if got := s.Child().Len(); got != 6 {
		t.Errorf("the child is %d elements, want all 6", got)
	}
}

func TestListSliceNulls(t *testing.T) {
	b, err := array.NewListBuilder(int64List)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}
	b.Elem().AppendValues([]int64{1})
	b.Append()
	b.AppendNull()
	b.AppendNull()
	b.Elem().AppendValues([]int64{2})
	b.Append()

	a := b.Finish()
	s := a.Slice(1, 3)
	if s.NullCount() != 2 {
		t.Errorf("NullCount = %d, want 2", s.NullCount())
	}
	if s.Slice(0, 0).NullCount() != 0 {
		t.Error("an empty slice counted a null")
	}
}

// TestListClone is the copy that shares nothing, which for a list means the
// offsets move back to zero and the child comes along cut down to what the rows
// in range reach.
func TestListClone(t *testing.T) {
	a := lists(t, []int64{1}, []int64{2, 3}, []int64{4, 5, 6})

	c := a.Slice(1, 3).Clone()
	if got, want := rowsOf(t, c), [][]int64{{2, 3}, {4, 5, 6}}; !slices.EqualFunc(got, want, slices.Equal) {
		t.Errorf("the rows are %v, want %v", got, want)
	}
	if want := []int32{0, 2, 5}; !slices.Equal(c.Offsets(), want) {
		t.Errorf("Offsets = %v, want %v", c.Offsets(), want)
	}
	if got := c.Child().Len(); got != 5 {
		t.Errorf("the child is %d elements, want the 5 the rows reach", got)
	}
	if c.Offset() != 0 {
		t.Errorf("Offset = %d, want 0", c.Offset())
	}
}

func TestListCloneKeepsNulls(t *testing.T) {
	b, err := array.NewListBuilder(int64List)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}
	b.Elem().AppendValues([]int64{1})
	b.Append()
	b.AppendNull()

	c := b.Finish().Clone()
	if c.NullCount() != 1 {
		t.Errorf("NullCount = %d, want 1", c.NullCount())
	}
	if !c.IsNull(1) {
		t.Error("the null row came back present")
	}
}

func TestListChunked(t *testing.T) {
	c, err := array.NewChunked(int64List,
		lists(t, []int64{1, 2}),
		lists(t, []int64{3}, nil),
	)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	if c.Len() != 3 {
		t.Errorf("Len = %d, want 3", c.Len())
	}
}

// TestListAccessorsOnOtherColumns is what the list methods do when the column is
// not a list, which is nothing for the two that can and a panic for the one that
// cannot.
func TestListAccessorsOnOtherColumns(t *testing.T) {
	a := array.Of[int64](1, 2, 3)

	if a.Child() != nil {
		t.Error("Child on an int64 column gave something")
	}
	if a.Offsets() != nil {
		t.Error("Offsets on an int64 column gave something")
	}

	defer func() {
		if recover() == nil {
			t.Error("List on an int64 column did not panic")
		}
	}()
	a.List(0)
}

func TestListOutOfRange(t *testing.T) {
	a := lists(t, []int64{1})

	defer func() {
		if recover() == nil {
			t.Error("List past the end did not panic")
		}
	}()
	a.List(1)
}

func TestListBuilderNegativeGrow(t *testing.T) {
	b, err := array.NewListBuilder(int64List)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}

	defer func() {
		if recover() == nil {
			t.Error("a negative Grow did not panic")
		}
	}()
	b.Grow(-1)
}

// TestNewSaysWhereToGoForAList is the error a caller gets for handing a list
// type to the constructors that cannot build one.
func TestNewSaysWhereToGoForAList(t *testing.T) {
	_, err := array.New(int64List, 1, buffer.New(64), nil)
	if err == nil {
		t.Fatal("New on a list column gave no error")
	}
	if !strings.Contains(err.Error(), "NewList") {
		t.Errorf("the message is %q, want it to point at NewList", err)
	}

	_, err = array.NewBuilder(dtype.LargeList{Elem: dtype.Int64})
	if err == nil {
		t.Fatal("NewBuilder on a large list gave no error")
	}
	if !strings.Contains(err.Error(), "store it as a list") {
		t.Errorf("the message is %q, want it to say what to store instead", err)
	}
}

func TestListBuilderElemIsTheSameBuilder(t *testing.T) {
	b, err := array.NewListBuilder(int64List)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}
	if b.Elem() != b.Elem() {
		t.Error("Elem gave two different builders")
	}
	if !dtype.Equal(b.Elem().DType(), dtype.Int64) {
		t.Errorf("the element builder is for %s, want int64", b.Elem().DType())
	}
	if !dtype.Equal(b.DType(), int64List) {
		t.Errorf("DType = %s, want %s", b.DType(), int64List)
	}
	if b.NullCount() != 0 {
		t.Errorf("NullCount = %d, want 0", b.NullCount())
	}
}
