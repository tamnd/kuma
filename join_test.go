package kuma_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// sectors returns the small right hand frame most of these tests join onto. It
// has AAPL and MSFT and not the TSLA the left frame has, and it lists MSFT
// before AAPL so that the output order is a real check rather than a
// coincidence.
func sectors(t *testing.T) *kuma.Frame[kuma.Dynamic] {
	t.Helper()

	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "MSFT", "AAPL").Column(),
		kuma.NewSeries("sector", "software", "hardware").Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	return f
}

// rows renders a frame as one string per row, for comparing against a table.
func rows[S any](t *testing.T, f *kuma.Frame[S]) []string {
	t.Helper()

	out := make([]string, f.NumRows())
	for i := range f.NumRows() {
		var b strings.Builder
		for j := range f.NumCols() {
			if j > 0 {
				b.WriteByte(' ')
			}
			c := f.ColumnAt(j)
			if c.IsNull(i) {
				b.WriteByte('.')
				continue
			}
			switch c.DType() {
			case dtype.String:
				b.WriteString(string(c.Data().Bytes(i)))
			case dtype.Int64:
				fmt.Fprintf(&b, "%d", c.Data().Value[int64](i))
			case dtype.Float64:
				fmt.Fprintf(&b, "%g", c.Data().Value[float64](i))
			case dtype.Bool:
				if c.Data().Bool(i) {
					b.WriteByte('t')
					continue
				}
				b.WriteByte('f')
			default:
				t.Fatalf("no way to print a %s column", c.DType())
			}
		}
		out[i] = b.String()
	}
	return out
}

func TestFrameJoin(t *testing.T) {
	left := trades(t)
	right := sectors(t)

	tests := []struct {
		how   kuma.JoinType
		names string
		want  []string
	}{
		{kuma.InnerJoin, "symbol,price,qty,sector", []string{
			"AAPL 189.5 100 hardware",
			"MSFT 411.2 50 software",
			"AAPL 190.1 25 hardware",
		}},
		{kuma.LeftJoin, "symbol,price,qty,sector", []string{
			"AAPL 189.5 100 hardware",
			"MSFT 411.2 50 software",
			"AAPL 190.1 25 hardware",
			"NVDA 121 400 .",
		}},
		{kuma.RightJoin, "symbol,price,qty,sector", []string{
			"MSFT 411.2 50 software",
			"AAPL 189.5 100 hardware",
			"AAPL 190.1 25 hardware",
		}},
		{kuma.OuterJoin, "symbol,price,qty,sector", []string{
			"AAPL 189.5 100 hardware",
			"MSFT 411.2 50 software",
			"AAPL 190.1 25 hardware",
			"NVDA 121 400 .",
		}},
		{kuma.SemiJoin, "symbol,price,qty", []string{
			"AAPL 189.5 100",
			"MSFT 411.2 50",
			"AAPL 190.1 25",
		}},
		{kuma.AntiJoin, "symbol,price,qty", []string{
			"NVDA 121 400",
		}},
	}

	for _, tt := range tests {
		t.Run(tt.how.String(), func(t *testing.T) {
			got, err := left.Join(right, kuma.Using("symbol"), tt.how)
			if err != nil {
				t.Fatalf("Join: %v", err)
			}
			if names := strings.Join(got.Names(), ","); names != tt.names {
				t.Errorf("the columns are %q, want %q", names, tt.names)
			}
			if lines := rows(t, got); !equalLines(lines, tt.want) {
				t.Errorf("the rows are\n%s\nwant\n%s",
					strings.Join(lines, "\n"), strings.Join(tt.want, "\n"))
			}
		})
	}
}

func equalLines(got, want []string) bool {
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

// TestFrameJoinOuterKey is the reason the shared key column is a gather from
// both sides rather than from the left one. A row that came in from the right
// has no left position, and taking the key from the left column would give it
// a null for a key it plainly has.
func TestFrameJoinOuterKey(t *testing.T) {
	left, err := kuma.NewFrame(
		kuma.NewSeries("k", "a", "b").Column(),
		kuma.NewSeries("l", int64(1), 2).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	right, err := kuma.NewFrame(
		kuma.NewSeries("k", "b", "c").Column(),
		kuma.NewSeries("r", int64(9), 8).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	got, err := left.Join(right, kuma.Using("k"), kuma.OuterJoin)
	if err != nil {
		t.Fatalf("Join: %v", err)
	}
	want := []string{"a 1 .", "b 2 9", "c . 8"}
	if lines := rows(t, got); !equalLines(lines, want) {
		t.Errorf("the rows are\n%s\nwant\n%s",
			strings.Join(lines, "\n"), strings.Join(want, "\n"))
	}

	// The same thing on a right join, where every output row came from the
	// right side and only some of them have a left one.
	got, err = left.Join(right, kuma.Using("k"), kuma.RightJoin)
	if err != nil {
		t.Fatalf("Join: %v", err)
	}
	want = []string{"b 2 9", "c . 8"}
	if lines := rows(t, got); !equalLines(lines, want) {
		t.Errorf("the rows are\n%s\nwant\n%s",
			strings.Join(lines, "\n"), strings.Join(want, "\n"))
	}
}

func TestFrameJoinDifferentNames(t *testing.T) {
	left := trades(t)
	right, err := kuma.NewFrame(
		kuma.NewSeries("ticker", "MSFT", "AAPL").Column(),
		kuma.NewSeries("sector", "software", "hardware").Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	got, err := left.Join(right, []kuma.On{{Left: "symbol", Right: "ticker"}}, kuma.InnerJoin)
	if err != nil {
		t.Fatalf("Join: %v", err)
	}
	// Both key columns are kept, because a caller who wrote two names probably
	// wants to see both.
	if names := strings.Join(got.Names(), ","); names != "symbol,price,qty,ticker,sector" {
		t.Errorf("the columns are %q", names)
	}
	want := []string{
		"AAPL 189.5 100 AAPL hardware",
		"MSFT 411.2 50 MSFT software",
		"AAPL 190.1 25 AAPL hardware",
	}
	if lines := rows(t, got); !equalLines(lines, want) {
		t.Errorf("the rows are\n%s\nwant\n%s",
			strings.Join(lines, "\n"), strings.Join(want, "\n"))
	}
}

func TestFrameJoinSeveralKeys(t *testing.T) {
	left, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "AAPL", "MSFT").Column(),
		kuma.NewSeries("side", "BUY", "SELL", "BUY").Column(),
		kuma.NewSeries("qty", int64(1), 2, 3).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	right, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT").Column(),
		kuma.NewSeries("side", "SELL", "BUY").Column(),
		kuma.NewSeries("fee", 0.5, 0.25).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	got, err := left.InnerJoin(right, "symbol", "side")
	if err != nil {
		t.Fatalf("InnerJoin: %v", err)
	}
	if names := strings.Join(got.Names(), ","); names != "symbol,side,qty,fee" {
		t.Errorf("the columns are %q", names)
	}
	want := []string{"AAPL SELL 2 0.5", "MSFT BUY 3 0.25"}
	if lines := rows(t, got); !equalLines(lines, want) {
		t.Errorf("the rows are\n%s\nwant\n%s",
			strings.Join(lines, "\n"), strings.Join(want, "\n"))
	}
}

func TestFrameLeftJoin(t *testing.T) {
	got, err := trades(t).LeftJoin(sectors(t), "symbol")
	if err != nil {
		t.Fatalf("LeftJoin: %v", err)
	}
	if got.NumRows() != 4 {
		t.Errorf("there are %d rows, want every left row", got.NumRows())
	}
	if !got.ColumnAt(3).IsNull(3) {
		t.Error("NVDA found a sector")
	}
}

func TestFrameCrossJoin(t *testing.T) {
	left, err := kuma.NewFrame(kuma.NewSeries("a", int64(1), 2).Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	right, err := kuma.NewFrame(kuma.NewSeries("b", "x", "y", "z").Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	got, err := left.CrossJoin(right)
	if err != nil {
		t.Fatalf("CrossJoin: %v", err)
	}
	want := []string{"1 x", "1 y", "1 z", "2 x", "2 y", "2 z"}
	if lines := rows(t, got); !equalLines(lines, want) {
		t.Errorf("the rows are\n%s\nwant\n%s",
			strings.Join(lines, "\n"), strings.Join(want, "\n"))
	}
}

// TestFrameJoinNullKey is the rule a join takes from SQL and not from pandas: a
// missing key matches nothing, including another missing key.
func TestFrameJoinNullKey(t *testing.T) {
	left, err := kuma.NewFrame(nullKeys(t, "a", "", "c"))
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	right, err := kuma.NewFrame(
		nullKeys(t, "", "c"),
		kuma.NewSeries("v", int64(7), 8).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	got, err := left.InnerJoin(right, "k")
	if err != nil {
		t.Fatalf("InnerJoin: %v", err)
	}
	want := []string{"c 8"}
	if lines := rows(t, got); !equalLines(lines, want) {
		t.Errorf("the rows are\n%s\nwant\n%s",
			strings.Join(lines, "\n"), strings.Join(want, "\n"))
	}
}

// nullKeys returns a string column called k where an empty string means a
// missing value.
func nullKeys(t *testing.T, vals ...string) kuma.Column {
	t.Helper()

	b, err := array.NewBuilder(dtype.String)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	for _, v := range vals {
		if v == "" {
			b.AppendNull()
			continue
		}
		b.AppendString(v)
	}

	data, err := array.NewChunked(dtype.String, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	c, err := kuma.NewColumn("k", data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	return c
}

func TestFrameJoinErrors(t *testing.T) {
	left := trades(t)
	right := sectors(t)

	// A nil right frame has no schema for the compiler to work the type
	// parameter out from, so this is the one call that has to write it.
	if _, err := left.Join[kuma.Dynamic](nil, kuma.Using("symbol"), kuma.InnerJoin); err == nil {
		t.Error("joining onto nothing succeeded")
	}
	if _, err := left.Join(right, nil, kuma.InnerJoin); err == nil {
		t.Error("an inner join with no keys succeeded")
	}
	if _, err := left.Join(right, kuma.Using("symbol"), kuma.CrossJoin); err == nil {
		t.Error("a cross join with keys succeeded")
	}
	if _, err := left.InnerJoin(right, "nope"); err == nil {
		t.Error("joining on a column that is not there succeeded")
	}
	if _, err := left.Join(right, []kuma.On{{Left: "symbol", Right: "nope"}},
		kuma.InnerJoin); err == nil {
		t.Error("joining onto a column that is not there succeeded")
	}

	// A name that is on both sides and is not a key, which pandas would rename
	// to price_x and price_y and this refuses.
	clash, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL").Column(),
		kuma.NewSeries("price", 1.0).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	if _, err := left.InnerJoin(clash, "symbol"); err == nil {
		t.Error("a join with two columns called price succeeded")
	}
}

// TestFrameJoinKeyTypes is the one place a join is stricter than the key
// encoding. An int8 and an int64 holding the same number match, but the shared
// output column has to be one type, so keys of two types have to be named
// apart.
func TestFrameJoinKeyTypes(t *testing.T) {
	left, err := kuma.NewFrame(kuma.NewSeries("k", int8(1), 2).Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	right, err := kuma.NewFrame(
		kuma.NewSeries("k", int64(2), 3).Column(),
		kuma.NewSeries("v", "two", "three").Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	if _, mixed := left.InnerJoin(right, "k"); mixed == nil {
		t.Error("joining an int8 key to an int64 key under one name succeeded")
	}

	// Renamed apart, the join itself is fine and the widening does its job.
	renamed, err := right.Rename("k", "k64")
	if err != nil {
		t.Fatalf("Rename: %v", err)
	}
	got, err := left.Join(renamed, []kuma.On{{Left: "k", Right: "k64"}}, kuma.InnerJoin)
	if err != nil {
		t.Fatalf("Join: %v", err)
	}
	if got.NumRows() != 1 {
		t.Errorf("there are %d rows, want the one key both sides have", got.NumRows())
	}
}

func TestUsing(t *testing.T) {
	got := kuma.Using("a", "b")
	want := []kuma.On{{Left: "a", Right: "a"}, {Left: "b", Right: "b"}}
	if len(got) != len(want) {
		t.Fatalf("Using gave %d keys, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("key %d is %v, want %v", i, got[i], want[i])
		}
	}
	if kuma.Using() != nil && len(kuma.Using()) != 0 {
		t.Error("Using with no names gave keys")
	}
}

// TestFrameJoinRefused is the error from the kernel coming through, which is a
// key column of a type there is no encoding for. The kernel tests cover which
// types those are.
func TestFrameJoinRefused(t *testing.T) {
	data, err := array.NewChunked(dtype.List{Elem: dtype.Int64})
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	lists, err := kuma.NewColumn("l", data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	empty, err := kuma.NewFrame(lists)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	if _, err := empty.InnerJoin(empty, "l"); err == nil {
		t.Error("joining on a list column succeeded")
	}
}
