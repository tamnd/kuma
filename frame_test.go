package kuma_test

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/dtype"
)

// trades returns a small frame of the shape most of these tests want: three
// columns of four rows, of three different types.
func trades(t *testing.T) *kuma.Frame[kuma.Dynamic] {
	t.Helper()

	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "AAPL", "NVDA").Column(),
		kuma.NewSeries("price", 189.5, 411.2, 190.1, 121.0).Column(),
		kuma.NewSeries("qty", int64(100), 50, 25, 400).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	return f
}

func TestNewFrame(t *testing.T) {
	f := trades(t)

	rows, cols := f.Shape()
	if rows != 4 || cols != 3 {
		t.Fatalf("Shape() = %d, %d, want 4, 3", rows, cols)
	}
	if f.NumRows() != 4 || f.NumCols() != 3 {
		t.Fatalf("%s, want 4 rows and 3 columns", f)
	}

	want := []string{"symbol", "price", "qty"}
	got := f.Names()
	if len(got) != len(want) {
		t.Fatalf("Names() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Names() = %v, want %v", got, want)
		}
		if f.Index(want[i]) != i {
			t.Errorf("Index(%q) = %d, want %d", want[i], f.Index(want[i]), i)
		}
	}
	if f.Index("nope") != -1 {
		t.Errorf("Index of a column that is not there = %d, want -1", f.Index("nope"))
	}

	schema := f.Schema()
	if schema.Len() != 3 {
		t.Fatalf("Schema() = %s", schema)
	}
	if !dtype.Equal(schema.Fields[1].Type, dtype.Float64) {
		t.Errorf("the price column is a %s, want float64", schema.Fields[1].Type)
	}

	// The schema is a copy, so writing to it does not reach back into the frame.
	schema.Fields[0].Name = "changed"
	if f.Names()[0] != "symbol" {
		t.Error("writing to the schema changed the frame")
	}
}

func TestNewFrameEmpty(t *testing.T) {
	f, err := kuma.NewFrame()
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	if f.NumRows() != 0 || f.NumCols() != 0 {
		t.Fatalf("%s, want an empty frame", f)
	}
	if got := f.Names(); len(got) != 0 {
		t.Errorf("Names() = %v, want nothing", got)
	}
	if _, err := f.Column("qty"); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("Column on an empty frame gave %v, want ErrNoColumn", err)
	}
}

func TestNewFrameErrors(t *testing.T) {
	tests := []struct {
		name string
		cols []kuma.Column
		want error
		text string
	}{
		{
			name: "columns of different length",
			cols: []kuma.Column{
				kuma.NewSeries("a", int64(1), 2, 3).Column(),
				kuma.NewSeries("b", int64(1)).Column(),
			},
			want: kuma.ErrLength,
			text: `column "b" has 1 rows, "a" has 3`,
		},
		{
			name: "two columns of one name",
			cols: []kuma.Column{
				kuma.NewSeries("a", int64(1)).Column(),
				kuma.NewSeries("a", int64(2)).Column(),
			},
			want: kuma.ErrDuplicateColumn,
			text: `two columns are called "a"`,
		},
		{
			name: "a column with no values",
			cols: []kuma.Column{{}},
			want: kuma.ErrNoValues,
			text: "column 0 has no values",
		},
		{
			name: "a column with no name",
			cols: []kuma.Column{kuma.NewSeries("", int64(1)).Column()},
			want: nil,
			text: "field 0 has no name",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := kuma.NewFrame(tt.cols...)
			if err == nil {
				t.Fatalf("NewFrame returned %s, want an error", f)
			}
			if tt.want != nil && !errors.Is(err, tt.want) {
				t.Errorf("error is %v, want it to be %v", err, tt.want)
			}
			if !strings.Contains(err.Error(), tt.text) {
				t.Errorf("error is %q, want it to mention %q", err, tt.text)
			}
		})
	}
}

func TestFrameColumn(t *testing.T) {
	f := trades(t)

	c, err := f.Column("price")
	if err != nil {
		t.Fatalf("Column: %v", err)
	}
	if c.Name() != "price" || c.Len() != 4 {
		t.Errorf("Column gave %s", c)
	}

	if got := f.ColumnAt(2); got.Name() != "qty" {
		t.Errorf("ColumnAt(2) = %s, want qty", got)
	}
	if got := f.Columns(); len(got) != 3 || got[0].Name() != "symbol" {
		t.Errorf("Columns() = %v", got)
	}

	_, err = f.Column("prices")
	if !errors.Is(err, kuma.ErrNoColumn) {
		t.Fatalf("error is %v, want it to be ErrNoColumn", err)
	}
}

func TestFrameSeries(t *testing.T) {
	f := trades(t)

	prices, err := f.Series[float64]("price")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := prices.Values(); len(got) != 4 || got[1] != 411.2 {
		t.Errorf("the prices are %v", got)
	}

	symbols, err := f.Series[string]("symbol")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := symbols.Value(3); got != "NVDA" {
		t.Errorf("Value(3) = %q, want NVDA", got)
	}

	if _, err := f.Series[int64]("price"); !errors.Is(err, kuma.ErrWrongType) {
		t.Errorf("reading the price column as an int64 gave %v, want ErrWrongType", err)
	}
	if _, err := f.Series[float64]("nope"); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("error is %v, want it to be ErrNoColumn", err)
	}
}

func TestFrameSelect(t *testing.T) {
	f := trades(t)

	out, err := f.Select("qty", "symbol")
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if got := out.Names(); len(got) != 2 || got[0] != "qty" || got[1] != "symbol" {
		t.Errorf("Names() = %v, want the order they were asked for", got)
	}
	if out.NumRows() != 4 {
		t.Errorf("Select changed the row count to %d", out.NumRows())
	}
	if out.ColumnAt(0).Data() != f.ColumnAt(2).Data() {
		t.Error("Select copied a column")
	}
	if f.NumCols() != 3 {
		t.Error("Select changed the frame it was called on")
	}

	none, err := f.Select()
	if err != nil {
		t.Fatalf("Select with no names: %v", err)
	}
	if none.NumCols() != 0 {
		t.Errorf("Select with no names gave %s", none)
	}

	if _, err := f.Select("symbol", "symbol"); !errors.Is(err, kuma.ErrDuplicateColumn) {
		t.Errorf("selecting a column twice gave %v, want ErrDuplicateColumn", err)
	}
	if _, err := f.Select("symbol", "nope"); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("error is %v, want it to be ErrNoColumn", err)
	}
}

func TestFrameDrop(t *testing.T) {
	f := trades(t)

	out, err := f.Drop("price")
	if err != nil {
		t.Fatalf("Drop: %v", err)
	}
	if got := out.Names(); len(got) != 2 || got[0] != "symbol" || got[1] != "qty" {
		t.Errorf("Names() = %v, want the order the frame had", got)
	}

	all, err := f.Drop("symbol", "price", "qty")
	if err != nil {
		t.Fatalf("Drop: %v", err)
	}
	if all.NumCols() != 0 {
		t.Errorf("dropping every column gave %s", all)
	}

	if _, err := f.Drop("nope"); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("error is %v, want it to be ErrNoColumn", err)
	}
}

func TestFrameRename(t *testing.T) {
	f := trades(t)

	out, err := f.Rename("qty", "quantity")
	if err != nil {
		t.Fatalf("Rename: %v", err)
	}
	if got := out.Names(); got[2] != "quantity" {
		t.Errorf("Names() = %v, want the column renamed in place", got)
	}
	if f.Names()[2] != "qty" {
		t.Error("Rename changed the frame it was called on")
	}

	if _, err := f.Rename("qty", "price"); !errors.Is(err, kuma.ErrDuplicateColumn) {
		t.Errorf("renaming a column onto another gave %v, want ErrDuplicateColumn", err)
	}
	if _, err := f.Rename("nope", "x"); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("error is %v, want it to be ErrNoColumn", err)
	}
}

func TestFrameWithColumn(t *testing.T) {
	f := trades(t)

	added, err := f.WithColumn(kuma.NewSeries("side", "BUY", "SELL", "BUY", "BUY").Column())
	if err != nil {
		t.Fatalf("WithColumn: %v", err)
	}
	if got := added.Names(); len(got) != 4 || got[3] != "side" {
		t.Errorf("Names() = %v, want side at the end", got)
	}

	replaced, err := f.WithColumn(kuma.NewSeries("price", 1.0, 2.0, 3.0, 4.0).Column())
	if err != nil {
		t.Fatalf("WithColumn: %v", err)
	}
	if replaced.NumCols() != 3 {
		t.Fatalf("replacing a column gave %s", replaced)
	}
	if got := replaced.ColumnAt(1).MustAs[float64]().Value(0); got != 1.0 {
		t.Errorf("the price column holds %v, want the new values", got)
	}
	if got := f.ColumnAt(1).MustAs[float64]().Value(0); got != 189.5 {
		t.Error("WithColumn changed the frame it was called on")
	}

	_, err = f.WithColumn(kuma.NewSeries("short", int64(1)).Column())
	if !errors.Is(err, kuma.ErrLength) {
		t.Errorf("adding a column of the wrong length gave %v, want ErrLength", err)
	}

	// The first column of an empty frame sets the row count, so any length goes.
	empty, err := kuma.NewFrame()
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	first, err := empty.WithColumn(kuma.NewSeries("a", int64(1), 2).Column())
	if err != nil {
		t.Fatalf("WithColumn on an empty frame: %v", err)
	}
	if first.NumRows() != 2 {
		t.Errorf("%s, want 2 rows", first)
	}
}

func TestFrameSlice(t *testing.T) {
	f := trades(t)

	cut := f.Slice(1, 3)
	if cut.NumRows() != 2 || cut.NumCols() != 3 {
		t.Fatalf("%s, want 2 rows and 3 columns", cut)
	}
	if got := cut.ColumnAt(0).MustAs[string]().Values(); got[0] != "MSFT" || got[1] != "AAPL" {
		t.Errorf("the symbols are %v", got)
	}
	if f.NumRows() != 4 {
		t.Error("Slice changed the frame it was called on")
	}

	if got := f.Head(2).ColumnAt(2).MustAs[int64]().Values(); len(got) != 2 || got[0] != 100 {
		t.Errorf("Head(2) holds %v", got)
	}
	if got := f.Tail(1).ColumnAt(2).MustAs[int64]().Values(); len(got) != 1 || got[0] != 400 {
		t.Errorf("Tail(1) holds %v", got)
	}
	if got := f.Head(-1).NumRows(); got != 3 {
		t.Errorf("Head(-1) has %d rows, want 3", got)
	}
	if got := f.Tail(-1).NumRows(); got != 3 {
		t.Errorf("Tail(-1) has %d rows, want 3", got)
	}
}

// TestFrameSliceRebuildsTheSchema is the other half of the rule that a frame's
// schema describes its data. Slicing away the rows with nulls in them gives a
// frame whose columns are not nullable.
func TestFrameSliceRebuildsTheSchema(t *testing.T) {
	f, err := kuma.NewFrame(nullableInts(t, 10).Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	if !f.Schema().Fields[0].Nullable {
		t.Fatal("the column is not nullable, want it nullable")
	}
	if f.Slice(1, 3).Schema().Fields[0].Nullable {
		t.Error("a range with no nulls in it is still nullable")
	}
	if !f.Slice(0, 4).Schema().Fields[0].Nullable {
		t.Error("a range with nulls in it is not nullable")
	}
}

func TestFramePanics(t *testing.T) {
	f := trades(t)

	tests := []struct {
		name string
		fn   func()
		want string
	}{
		{"column index too high", func() { columnSink = f.ColumnAt(3) }, "column index 3 out of range"},
		{"column index negative", func() { columnSink = f.ColumnAt(-1) }, "out of range"},
		{"slice past the end", func() { frameSink = f.Slice(0, 5) }, "Slice(0, 5) of a frame of 4 rows"},
		{"slice backwards", func() { frameSink = f.Slice(3, 2) }, "Slice(3, 2) of a frame of 4 rows"},
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

func TestFrameString(t *testing.T) {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT").Column(),
		nullableInts(t, 2).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	want := strings.Join([]string{
		"kuma.Frame[kuma.Dynamic] 2 rows x 2 cols",
		"  symbol: string",
		"  qty: int64, 1 null",
	}, "\n")
	if got := f.String(); got != want {
		t.Errorf("String() =\n%s\nwant\n%s", got, want)
	}
}

func TestFrameTake(t *testing.T) {
	f := trades(t)

	idx := []int{3, 0, 0}
	got := f.Take(idx)
	if got.NumRows() != 3 || got.NumCols() != 3 {
		t.Fatalf("%s, want 3 rows and 3 columns", got)
	}
	if names := got.Names(); !slices.Equal(names, f.Names()) {
		t.Errorf("Take gave the columns %v, want %v", names, f.Names())
	}

	if want := []string{"NVDA", "AAPL", "AAPL"}; !slices.Equal(got.ColumnAt(0).MustAs[string]().Values(), want) {
		t.Errorf("the symbols are %v, want %v", got.ColumnAt(0).MustAs[string]().Values(), want)
	}
	if want := []int64{400, 100, 100}; !slices.Equal(got.ColumnAt(2).MustAs[int64]().Values(), want) {
		t.Errorf("the quantities are %v, want %v", got.ColumnAt(2).MustAs[int64]().Values(), want)
	}
	if f.NumRows() != 4 {
		t.Error("Take changed the frame it was called on")
	}

	// A position below zero is a row of nulls, one in every column.
	nulls := f.Take([]int{-1, 1})
	for _, c := range nulls.Columns() {
		if !c.IsNull(0) {
			t.Errorf("column %q is not null in the row that matched nothing", c.Name())
		}
	}
	if !nulls.Schema().Fields[0].Nullable {
		t.Error("the schema says the column is not nullable, and it has a null in it")
	}
}

func TestFrameTakeNothing(t *testing.T) {
	got := trades(t).Take(nil)
	if got.NumRows() != 0 || got.NumCols() != 3 {
		t.Fatalf("%s, want no rows and 3 columns", got)
	}
	if got.Schema().Fields[0].Nullable {
		t.Error("a column of no values is nullable, want it not nullable")
	}
}

func TestFrameFilter(t *testing.T) {
	f := trades(t)
	mask := kuma.NewSeries("keep", false, true, false, true)

	got, err := f.Filter(mask)
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if got.NumRows() != 2 {
		t.Fatalf("%s, want 2 rows", got)
	}
	if want := []string{"MSFT", "NVDA"}; !slices.Equal(got.ColumnAt(0).MustAs[string]().Values(), want) {
		t.Errorf("the symbols are %v, want %v", got.ColumnAt(0).MustAs[string]().Values(), want)
	}

	if _, err := f.Filter(kuma.NewSeries("keep", true, false)); !errors.Is(err, kuma.ErrLength) {
		t.Errorf("a short mask gave %v, want an ErrLength", err)
	}
}

// TestFrameTakePanics also covers a frame with no columns, where there is no
// column to notice that the position is out of range and the frame has to say
// so itself.
func TestFrameTakePanics(t *testing.T) {
	empty, err := kuma.NewFrame()
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	tests := []struct {
		name string
		call func()
	}{
		{"past the end", func() { trades(t).Take([]int{4}) }},
		{"out of an empty frame", func() { empty.Take([]int{0}) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Fatal("it did not panic")
				}
			}()
			tt.call()
		})
	}
}

func TestFrameCast(t *testing.T) {
	f := mustFrame(t,
		kuma.NewSeries("symbol", "AAPL", "MSFT").Column(),
		kuma.NewSeries("qty", int32(100), int32(250)).Column(),
	)

	got, err := f.Cast("qty", dtype.Int64)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	if names := got.Names(); !slices.Equal(names, []string{"symbol", "qty"}) {
		t.Errorf("Cast gave the columns %v, want symbol and qty in that order", names)
	}
	if dt := got.ColumnAt(1).DType(); !dtype.Equal(dt, dtype.Int64) {
		t.Errorf("qty is a %s column, want int64", dt)
	}
	if dt := f.ColumnAt(1).DType(); !dtype.Equal(dt, dtype.Int32) {
		t.Error("Cast changed the frame it was called on")
	}
	if got.NumRows() != 2 {
		t.Errorf("Cast gave %d rows, want 2", got.NumRows())
	}

	// The schema is read off the data, so it has to say int64 as well.
	if dt := got.Schema().Fields[1].Type; !dtype.Equal(dt, dtype.Int64) {
		t.Errorf("the schema says qty is a %s, want int64", dt)
	}
}

func TestFrameCastErrors(t *testing.T) {
	f := mustFrame(t,
		kuma.NewSeries("qty", int64(1), int64(400)).Column(),
	)

	if _, err := f.Cast("nope", dtype.Int64); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("casting a column that is not there gave %v, want ErrNoColumn", err)
	}
	if _, err := f.TryCast("nope", dtype.Int64); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("casting a column that is not there gave %v, want ErrNoColumn", err)
	}

	if _, err := f.Cast("qty", dtype.Int8); err == nil {
		t.Fatal("casting 400 into an int8 succeeded")
	} else if !strings.Contains(err.Error(), `"qty"`) {
		t.Errorf("the message is %q, want it to name the column", err.Error())
	}

	got, err := f.TryCast("qty", dtype.Int8)
	if err != nil {
		t.Fatalf("TryCast: %v", err)
	}
	if got.ColumnAt(0).NullCount() != 1 {
		t.Errorf("TryCast gave %d nulls, want 1", got.ColumnAt(0).NullCount())
	}
}

// mustFrame is NewFrame where a failure is a broken test rather than a result.
func mustFrame(t *testing.T, cols ...kuma.Column) *kuma.Frame[kuma.Dynamic] {
	t.Helper()

	f, err := kuma.NewFrame(cols...)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	return f
}
