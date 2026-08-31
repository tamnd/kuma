package kuma_test

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// listColumn builds a list of numbers column, where a nil row is a missing row
// and an empty one is a row that is there and holds nothing.
func listColumn(t *testing.T, name string, rows ...[]int64) kuma.Column {
	t.Helper()

	dt := dtype.List{Elem: dtype.Int64}
	b, err := array.NewListBuilder(dt)
	if err != nil {
		t.Fatalf("NewListBuilder: %v", err)
	}
	for _, r := range rows {
		if r == nil {
			b.AppendNull()
			continue
		}
		b.Elem().AppendValues(r)
		b.Append()
	}

	data, err := array.NewChunked(dt, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	col, err := kuma.NewColumn(name, data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	return col
}

// tagged returns a frame of two symbols, one with three sizes against it and one
// with none.
//
//	symbol sizes
//	AAPL   [1 2 3]
//	MSFT   [7]
func tagged(t *testing.T) *kuma.Frame[kuma.Dynamic] {
	t.Helper()

	return mustFrame(t,
		kuma.NewSeries("symbol", "AAPL", "MSFT").Column(),
		listColumn(t, "sizes", []int64{1, 2, 3}, []int64{7}),
	)
}

// rowsOfFrame prints a frame a row at a time so a test can say what it wants as
// a list of strings, with a missing value printed as a dot.
func rowsOfFrame(t *testing.T, f *kuma.Frame[kuma.Dynamic]) []string {
	t.Helper()

	out := make([]string, f.NumRows())
	for i := range f.NumRows() {
		var sb strings.Builder
		for k, c := range f.Columns() {
			if k > 0 {
				sb.WriteByte(' ')
			}
			if c.IsNull(i) {
				sb.WriteByte('.')
				continue
			}
			switch c.DType().Kind() {
			case dtype.StringKind:
				sb.WriteString(string(c.Data().Chunks()[0].Bytes(i)))
			default:
				fmt.Fprintf(&sb, "%d", c.Data().Chunks()[0].Values[int64]()[i])
			}
		}
		out[i] = sb.String()
	}
	return out
}

// dtypeOf is the type of a named column, which is two lines everywhere else.
func dtypeOf(t *testing.T, f *kuma.Frame[kuma.Dynamic], name string) dtype.DataType {
	t.Helper()

	c, err := f.Column(name)
	if err != nil {
		t.Fatalf("Column(%q): %v", name, err)
	}
	return c.DType()
}

func wantFrame(t *testing.T, f *kuma.Frame[kuma.Dynamic], want []string) {
	t.Helper()

	got := rowsOfFrame(t, f)
	if len(got) != len(want) {
		t.Fatalf("the frame has %d rows, want %d:\n%v", len(got), len(want), got)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("row %d is %q, want %q", i, got[i], want[i])
		}
	}
}

func TestExplode(t *testing.T) {
	got, err := tagged(t).Explode("sizes")
	if err != nil {
		t.Fatalf("Explode: %v", err)
	}

	wantFrame(t, got, []string{"AAPL 1", "AAPL 2", "AAPL 3", "MSFT 7"})
	if dt := dtypeOf(t, got, "sizes"); !dtype.Equal(dt, dtype.Int64) {
		t.Errorf("the exploded column is %s, want int64", dt)
	}
	if dt := dtypeOf(t, got, "symbol"); !dtype.Equal(dt, dtype.String) {
		t.Errorf("the column beside it is %s, want the string it was", dt)
	}
	if names := got.Names(); len(names) != 2 || names[0] != "symbol" || names[1] != "sizes" {
		t.Errorf("the columns are %v, want symbol and sizes in that order", names)
	}
}

// TestExplodeKeepsEmptyRows is the rule that a row with nothing in it is still a
// row, so that the row count of the answer does not depend on which column was
// taken apart.
func TestExplodeKeepsEmptyRows(t *testing.T) {
	f := mustFrame(t,
		kuma.NewSeries("symbol", "AAPL", "MSFT", "GOOG", "AMZN").Column(),
		listColumn(t, "sizes", []int64{1, 2}, []int64{}, nil, []int64{3}),
	)

	got, err := f.Explode("sizes")
	if err != nil {
		t.Fatalf("Explode: %v", err)
	}
	wantFrame(t, got, []string{"AAPL 1", "AAPL 2", "MSFT .", "GOOG .", "AMZN 3"})
}

func TestExplodeTwoColumns(t *testing.T) {
	f := mustFrame(t,
		kuma.NewSeries("symbol", "AAPL", "MSFT").Column(),
		listColumn(t, "sizes", []int64{1, 2}, []int64{7}),
		listColumn(t, "prices", []int64{10, 20}, []int64{70}),
	)

	got, err := f.Explode("sizes", "prices")
	if err != nil {
		t.Fatalf("Explode: %v", err)
	}
	wantFrame(t, got, []string{"AAPL 1 10", "AAPL 2 20", "MSFT 7 70"})
}

// TestExplodeTwoColumnsDisagree is the one thing about several columns worth
// saying: a row that is two elements in one column and three in another has no
// answer, so it is an error that says which row.
func TestExplodeTwoColumnsDisagree(t *testing.T) {
	f := mustFrame(t,
		kuma.NewSeries("symbol", "AAPL", "MSFT").Column(),
		listColumn(t, "sizes", []int64{1, 2}, []int64{7}),
		listColumn(t, "prices", []int64{10, 20}, []int64{70, 80}),
	)

	_, err := f.Explode("sizes", "prices")
	if err == nil {
		t.Fatal("two columns that disagree were allowed")
	}
	if !errors.Is(err, kuma.ErrLength) {
		t.Errorf("error is %v, want one that is ErrLength", err)
	}
	for _, want := range []string{"row 1", `1 value in "sizes"`, `2 values in "prices"`} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error is %q, want it to mention %q", err, want)
		}
	}
}

// TestExplodeNullAgainstOneValue is the pair the rule about empty rows decides:
// a missing row becomes one row, so it agrees with a column holding one value
// there.
func TestExplodeNullAgainstOneValue(t *testing.T) {
	f := mustFrame(t,
		kuma.NewSeries("symbol", "AAPL", "MSFT").Column(),
		listColumn(t, "sizes", []int64{1}, nil),
		listColumn(t, "prices", []int64{10}, []int64{70}),
	)

	got, err := f.Explode("sizes", "prices")
	if err != nil {
		t.Fatalf("Explode: %v", err)
	}
	wantFrame(t, got, []string{"AAPL 1 10", "MSFT . 70"})
}

func TestExplodeNoRows(t *testing.T) {
	f := mustFrame(t,
		kuma.NewSeries[string]("symbol").Column(),
		listColumn(t, "sizes"),
	)

	got, err := f.Explode("sizes")
	if err != nil {
		t.Fatalf("Explode: %v", err)
	}
	if got.NumRows() != 0 || got.NumCols() != 2 {
		t.Errorf("the frame is %d by %d, want 0 by 2", got.NumRows(), got.NumCols())
	}
	if dt := dtypeOf(t, got, "sizes"); !dtype.Equal(dt, dtype.Int64) {
		t.Errorf("the exploded column is %s, want int64", dt)
	}
}

func TestExplodeMistakes(t *testing.T) {
	f := tagged(t)

	tests := []struct {
		name  string
		names []string
		want  error
		says  string
	}{
		{
			name:  "no column at all",
			names: nil,
			want:  kuma.ErrNoColumn,
			says:  "needs the name of a column",
		},
		{
			name:  "a column that is not there",
			names: []string{"nope"},
			want:  kuma.ErrNoColumn,
			says:  "nope",
		},
		{
			name:  "a column that holds one value per row",
			names: []string{"symbol"},
			want:  kuma.ErrWrongType,
			says:  "one value per row",
		},
		{
			name:  "one good name and one bad one",
			names: []string{"sizes", "symbol"},
			want:  kuma.ErrWrongType,
			says:  "symbol",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := f.Explode(tt.names...)
			if err == nil {
				t.Fatalf("Explode(%v) was allowed", tt.names)
			}
			if !errors.Is(err, tt.want) {
				t.Errorf("error is %v, want one that is %v", err, tt.want)
			}
			if !strings.Contains(err.Error(), tt.says) {
				t.Errorf("error is %q, want it to mention %q", err, tt.says)
			}
		})
	}
}

// tag is the result of exploding the sizes of tagged, which is the struct the
// typed steps below are told. A list column cannot be bound to a field today,
// so this is a struct that only means anything after the explode, which is the
// reason ExplodeAs is worth having at all.
type tag struct {
	Symbol string `kuma:"symbol"`
	Sizes  int64  `kuma:"sizes"`
}

// tagCols is what kumagen writes for tag.
var tagCols = struct {
	Symbol kuma.StrCol[tag]
	Sizes  kuma.I64Col[tag]
}{
	Symbol: kuma.NewStrCol[tag]("symbol"),
	Sizes:  kuma.NewI64Col[tag]("sizes"),
}

func TestLazyExplode(t *testing.T) {
	got, err := tagged(t).Lazy().Explode("sizes").Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	wantFrame(t, got, []string{"AAPL 1", "AAPL 2", "AAPL 3", "MSFT 7"})
	if dt := dtypeOf(t, got, "sizes"); !dtype.Equal(dt, dtype.Int64) {
		t.Errorf("the exploded column is %s, want int64", dt)
	}
}

// TestLazyExplodeIsTheSameAsEager is the rule the whole lazy API rests on,
// checked on the step that changes the shape of a frame most.
func TestLazyExplodeIsTheSameAsEager(t *testing.T) {
	f := mustFrame(t,
		kuma.NewSeries("symbol", "AAPL", "MSFT", "GOOG").Column(),
		listColumn(t, "sizes", []int64{1, 2}, nil, []int64{3}),
	)

	lazy, err := f.Lazy().Explode("sizes").Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	eager, err := f.Explode("sizes")
	if err != nil {
		t.Fatalf("Explode: %v", err)
	}

	if lazy.String() != eager.String() {
		t.Errorf("the lazy query gave\n%s\nand the eager one gave\n%s", lazy, eager)
	}
}

// TestLazyExplodeSchema is what the query says it will produce before it runs,
// which is the half of an explode a plan can work out on its own.
func TestLazyExplodeSchema(t *testing.T) {
	s, err := tagged(t).Lazy().Explode("sizes").Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}

	f, ok := s.Field("sizes")
	if !ok {
		t.Fatal("the exploded column is not in the schema")
	}
	if !dtype.Equal(f.Type, dtype.Int64) {
		t.Errorf("the schema says sizes is a %s, want int64", f.Type)
	}
	if !f.Nullable {
		t.Error("the schema says sizes cannot be missing, and an empty row becomes a missing value")
	}
}

func TestLazyExplodeString(t *testing.T) {
	q := tagged(t).Lazy().Explode("sizes").Head(2)

	want := strings.Join([]string{
		"Limit 2",
		"  Explode sizes",
		"    Scan frame",
	}, "\n")
	if got := q.String(); got != want {
		t.Errorf("String() =\n%s\nwant\n%s", got, want)
	}
}

// TestLazyExplodeMistakes are the errors the plan finds before anything is
// read, which is every one of them except two columns that disagree.
func TestLazyExplodeMistakes(t *testing.T) {
	f := tagged(t)

	tests := []struct {
		name  string
		names []string
		want  error
		says  string
	}{
		{"no column at all", nil, kuma.ErrNoColumn, "no column to take apart"},
		{"a column that is not there", []string{"size"}, kuma.ErrNoColumn, "did you mean: sizes?"},
		{"a column that holds one value per row", []string{"symbol"}, kuma.ErrWrongType, "one value per row"},
		{"a column named twice", []string{"sizes", "sizes"}, kuma.ErrDuplicateColumn, "names sizes twice"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := f.Lazy().Explode(tt.names...).Collect(t.Context())
			if err == nil {
				t.Fatalf("Explode(%v) was allowed", tt.names)
			}
			if !errors.Is(err, tt.want) {
				t.Errorf("error is %v, want one that is %v", err, tt.want)
			}
			if !strings.Contains(err.Error(), tt.says) {
				t.Errorf("error is %q, want it to mention %q", err, tt.says)
			}
		})
	}
}

// TestLazyExplodeOfTwoColumnsThatDisagree is the one mistake the plan cannot
// find, since how many elements a row holds is what the data decides.
func TestLazyExplodeOfTwoColumnsThatDisagree(t *testing.T) {
	f := mustFrame(t,
		kuma.NewSeries("symbol", "AAPL").Column(),
		listColumn(t, "sizes", []int64{1, 2}),
		listColumn(t, "prices", []int64{10}),
	)

	q := f.Lazy().Explode("sizes", "prices")
	if err := q.Validate(); err != nil {
		t.Fatalf("the query was turned away before it ran: %v", err)
	}

	_, err := q.Collect(t.Context())
	if !errors.Is(err, kuma.ErrLength) {
		t.Errorf("error is %v, want one that is ErrLength", err)
	}
}

func TestExplodeAs(t *testing.T) {
	got, err := tagged(t).ExplodeAs[tag]("sizes")
	if err != nil {
		t.Fatalf("ExplodeAs: %v", err)
	}

	if got.NumRows() != 4 || got.NumCols() != 2 {
		t.Fatalf("the frame is %d by %d, want 4 by 2", got.NumRows(), got.NumCols())
	}
	if names := got.Names(); !slices.Equal(names, []string{"symbol", "sizes"}) {
		t.Errorf("the columns are %v, want symbol and sizes", names)
	}

	// The frame is typed, so a handle written for tag reads it.
	sizes, err := tagCols.Sizes.Series(got)
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if want := []int64{1, 2, 3, 7}; !slices.Equal(sizes.Values(), want) {
		t.Errorf("the sizes are %v, want %v", sizes.Values(), want)
	}
}

func TestLazyExplodeAs(t *testing.T) {
	got, err := tagged(t).Lazy().ExplodeAs[tag]("sizes").Collect(t.Context())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	symbols, err := tagCols.Symbol.Series(got)
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if want := []string{"AAPL", "AAPL", "AAPL", "MSFT"}; !slices.Equal(symbols.Values(), want) {
		t.Errorf("the symbols are %v, want %v", symbols.Values(), want)
	}
}

// TestLazyExplodeAsIsCheckedWhereItIsWritten is what the typed steps buy over a
// collect and a bind: the struct is compared against what the plan produces at
// the line it was written, so a field that reads the column as the list it used
// to be is an error from there.
func TestLazyExplodeAsIsCheckedWhereItIsWritten(t *testing.T) {
	type listy struct {
		Sizes []int64 `kuma:"sizes"`
	}

	q := tagged(t).Lazy().ExplodeAs[listy]("sizes")
	if err := q.Validate(); err == nil {
		t.Fatal("a struct that reads the exploded column as a list was allowed")
	}

	// The same struct against the column before the explode is the same
	// mistake, since a field cannot read a list column either way.
	if _, err := tagged(t).SelectAs[listy](); err == nil {
		t.Error("a struct that reads a list column was allowed")
	}
}

func TestExplodeAsMistakes(t *testing.T) {
	f := tagged(t)

	if _, err := f.ExplodeAs[tag]("symbol"); !errors.Is(err, kuma.ErrWrongType) {
		t.Errorf("exploding a column of values gives %v, want ErrWrongType", err)
	}
	if _, err := f.ExplodeAs[tag]("nope"); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("exploding a column that is not there gives %v, want ErrNoColumn", err)
	}

	// The struct names a column the explode did not produce, which is the
	// mistake the As steps exist to report and the one they report late.
	type other struct {
		Volume int64 `kuma:"volume"`
	}
	if _, err := f.ExplodeAs[other]("sizes"); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("a struct naming a column that is not there gives %v, want ErrNoColumn", err)
	}
}

// TestExplodeLeavesTheFrameAlone is the immutability every operation here
// promises, checked on the one that changes the shape of the frame most.
func TestExplodeLeavesTheFrameAlone(t *testing.T) {
	f := tagged(t)

	if _, err := f.Explode("sizes"); err != nil {
		t.Fatalf("Explode: %v", err)
	}
	if f.NumRows() != 2 {
		t.Errorf("the frame has %d rows after being exploded, want 2", f.NumRows())
	}
	if dt := dtypeOf(t, f, "sizes"); dtype.Equal(dt, dtype.Int64) {
		t.Errorf("the frame's own column became %s", dt)
	}
}
