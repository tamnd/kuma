package kernel_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// sideOf returns a join side over one int64 key column written as one chunk,
// with nil meaning a missing key.
func sideOf(t *testing.T, keys ...any) kernel.Side {
	t.Helper()

	c := col(t, dtype.Int64, keys)
	return kernel.Side{Rows: c.Len(), Keys: []*array.Chunked{c}}
}

// pairsString renders the result the way the tables below write it, which is a
// row per line as "left right" with a dot where nothing matched.
func pairsString(p kernel.Pairs) string {
	var b strings.Builder
	for i, l := range p.Left {
		if i > 0 {
			b.WriteByte(' ')
		}
		write := func(v int) {
			if v < 0 {
				b.WriteByte('.')
				return
			}
			fmt.Fprintf(&b, "%d", v)
		}
		write(l)
		if p.Right != nil {
			b.WriteByte(':')
			write(p.Right[i])
		}
	}
	return b.String()
}

func TestJoin(t *testing.T) {
	// Left  0:10  1:20  2:30  3:20
	// Right 0:20  1:40  2:20
	//
	// So 20 matches twice, 10 and 30 match nothing, and 40 is unmatched on the
	// right.
	left := []any{int64(10), int64(20), int64(30), int64(20)}
	right := []any{int64(20), int64(40), int64(20)}

	tests := []struct {
		how  kernel.JoinType
		want string
	}{
		{kernel.InnerJoin, "1:0 1:2 3:0 3:2"},
		{kernel.LeftJoin, "0:. 1:0 1:2 2:. 3:0 3:2"},
		{kernel.RightJoin, "1:0 3:0 .:1 1:2 3:2"},
		{kernel.OuterJoin, "0:. 1:0 1:2 2:. 3:0 3:2 .:1"},
		{kernel.SemiJoin, "1 3"},
		{kernel.AntiJoin, "0 2"},
	}

	for _, tt := range tests {
		t.Run(tt.how.String(), func(t *testing.T) {
			p, err := kernel.Join(sideOf(t, left...), sideOf(t, right...), tt.how)
			if err != nil {
				t.Fatalf("Join: %v", err)
			}
			if got := pairsString(p); got != tt.want {
				t.Errorf("the pairs are %q, want %q", got, tt.want)
			}
			if p.Len() != len(p.Left) {
				t.Errorf("Len says %d, there are %d rows", p.Len(), len(p.Left))
			}
		})
	}
}

// TestJoinNullMatchesNothing is the rule that separates this from pandas. A
// missing key does not match another missing key, so an inner join drops both
// and an anti join keeps the left one.
func TestJoinNullMatchesNothing(t *testing.T) {
	left := sideOf(t, int64(1), nil, int64(3))
	right := sideOf(t, nil, int64(3))

	tests := []struct {
		how  kernel.JoinType
		want string
	}{
		{kernel.InnerJoin, "2:1"},
		{kernel.LeftJoin, "0:. 1:. 2:1"},
		{kernel.OuterJoin, "0:. 1:. 2:1 .:0"},
		{kernel.SemiJoin, "2"},
		{kernel.AntiJoin, "0 1"},
	}

	for _, tt := range tests {
		t.Run(tt.how.String(), func(t *testing.T) {
			p, err := kernel.Join(left, right, tt.how)
			if err != nil {
				t.Fatalf("Join: %v", err)
			}
			if got := pairsString(p); got != tt.want {
				t.Errorf("the pairs are %q, want %q", got, tt.want)
			}
		})
	}
}

func TestJoinSeveralKeys(t *testing.T) {
	symbol := col(t, dtype.String, []any{"AAPL", "AAPL", "MSFT"})
	side := col(t, dtype.String, []any{"BUY", "SELL", "BUY"})
	left := kernel.Side{Rows: 3, Keys: []*array.Chunked{symbol, side}}

	rsymbol := col(t, dtype.String, []any{"AAPL", "MSFT", "AAPL"})
	rside := col(t, dtype.String, []any{"SELL", "BUY", "BUY"})
	right := kernel.Side{Rows: 3, Keys: []*array.Chunked{rsymbol, rside}}

	p, err := kernel.Join(left, right, kernel.InnerJoin)
	if err != nil {
		t.Fatalf("Join: %v", err)
	}
	if got, want := pairsString(p), "0:2 1:0 2:1"; got != want {
		t.Errorf("the pairs are %q, want %q", got, want)
	}
}

// TestJoinKeyBoundary is the collision the length prefix in the key encoding
// exists to stop. Without it the pair ("a","bc") and the pair ("ab","c") would
// encode the same and would join to each other.
func TestJoinKeyBoundary(t *testing.T) {
	left := kernel.Side{Rows: 1, Keys: []*array.Chunked{
		col(t, dtype.String, []any{"a"}),
		col(t, dtype.String, []any{"bc"}),
	}}
	right := kernel.Side{Rows: 1, Keys: []*array.Chunked{
		col(t, dtype.String, []any{"ab"}),
		col(t, dtype.String, []any{"c"}),
	}}

	p, err := kernel.Join(left, right, kernel.InnerJoin)
	if err != nil {
		t.Fatalf("Join: %v", err)
	}
	if p.Len() != 0 {
		t.Errorf("the keys joined to each other, giving %q", pairsString(p))
	}
}

// TestJoinWidening is the other half of that encoding: an int8 and an int64
// holding the same number are the same key.
func TestJoinWidening(t *testing.T) {
	left := col(t, dtype.Int8, []any{int8(-1), int8(127)})
	right := col(t, dtype.Int64, []any{int64(127), int64(-1)})

	p, err := kernel.Join(
		kernel.Side{Rows: 2, Keys: []*array.Chunked{left}},
		kernel.Side{Rows: 2, Keys: []*array.Chunked{right}},
		kernel.InnerJoin)
	if err != nil {
		t.Fatalf("Join: %v", err)
	}
	if got, want := pairsString(p), "0:1 1:0"; got != want {
		t.Errorf("the pairs are %q, want %q", got, want)
	}
}

func TestJoinCross(t *testing.T) {
	p, err := kernel.Join(kernel.Side{Rows: 3}, kernel.Side{Rows: 2}, kernel.CrossJoin)
	if err != nil {
		t.Fatalf("Join: %v", err)
	}
	if got, want := pairsString(p), "0:0 0:1 1:0 1:1 2:0 2:1"; got != want {
		t.Errorf("the pairs are %q, want %q", got, want)
	}

	// One empty side gives no rows, which is what multiplying by zero does.
	p, err = kernel.Join(kernel.Side{Rows: 3}, kernel.Side{}, kernel.CrossJoin)
	if err != nil {
		t.Fatalf("Join: %v", err)
	}
	if p.Len() != 0 {
		t.Errorf("crossing with nothing gave %d rows", p.Len())
	}
}

func TestJoinEmpty(t *testing.T) {
	full := sideOf(t, int64(1), int64(2))
	empty := sideOf(t)

	tests := []struct {
		how   kernel.JoinType
		left  kernel.Side
		right kernel.Side
		want  string
	}{
		{kernel.InnerJoin, full, empty, ""},
		{kernel.LeftJoin, full, empty, "0:. 1:."},
		{kernel.RightJoin, full, empty, ""},
		{kernel.LeftJoin, empty, full, ""},
		{kernel.RightJoin, empty, full, ".:0 .:1"},
		{kernel.OuterJoin, full, empty, "0:. 1:."},
		{kernel.OuterJoin, empty, full, ".:0 .:1"},
		{kernel.SemiJoin, full, empty, ""},
		{kernel.AntiJoin, full, empty, "0 1"},
	}

	for _, tt := range tests {
		t.Run(tt.how.String(), func(t *testing.T) {
			p, err := kernel.Join(tt.left, tt.right, tt.how)
			if err != nil {
				t.Fatalf("Join: %v", err)
			}
			if got := pairsString(p); got != tt.want {
				t.Errorf("the pairs are %q, want %q", got, tt.want)
			}
		})
	}
}

func TestJoinChunked(t *testing.T) {
	left := col(t, dtype.Int64, []any{int64(1), int64(2)}, []any{int64(3), int64(2)})
	right := col(t, dtype.Int64, []any{int64(2)}, []any{int64(9), int64(2)})

	p, err := kernel.Join(
		kernel.Side{Rows: 4, Keys: []*array.Chunked{left}},
		kernel.Side{Rows: 3, Keys: []*array.Chunked{right}},
		kernel.InnerJoin)
	if err != nil {
		t.Fatalf("Join: %v", err)
	}
	if got, want := pairsString(p), "1:0 1:2 3:0 3:2"; got != want {
		t.Errorf("the pairs are %q, want %q", got, want)
	}
}

// TestJoinTake is the join and the gather together, which is what a join
// actually is to anybody using one. A position below zero becomes a null, so
// an outer join needs no special handling downstream.
func TestJoinTake(t *testing.T) {
	lkey := col(t, dtype.Int64, []any{int64(1), int64(2)})
	rkey := col(t, dtype.Int64, []any{int64(2), int64(3)})

	p, err := kernel.Join(
		kernel.Side{Rows: 2, Keys: []*array.Chunked{lkey}},
		kernel.Side{Rows: 2, Keys: []*array.Chunked{rkey}},
		kernel.OuterJoin)
	if err != nil {
		t.Fatalf("Join: %v", err)
	}

	gotLeft := kernel.Take(lkey, p.Left)
	gotRight := kernel.Take(rkey, p.Right)
	checkAgg(t, gotLeft, []any{int64(1), int64(2), nil})
	checkAgg(t, gotRight, []any{nil, int64(2), int64(3)})
}

func TestJoinErrors(t *testing.T) {
	full := sideOf(t, int64(1))

	if _, err := kernel.Join(full, kernel.Side{Rows: 1}, kernel.InnerJoin); err == nil {
		t.Error("an inner join with no right keys succeeded")
	}
	if _, err := kernel.Join(full, full, kernel.CrossJoin); err == nil {
		t.Error("a cross join with keys succeeded")
	}

	lists, err := array.NewChunked(dtype.List{Elem: dtype.Int64})
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	nested := kernel.Side{Keys: []*array.Chunked{lists}}
	if _, err := kernel.Join(nested, nested, kernel.InnerJoin); err == nil {
		t.Error("joining on a list column succeeded")
	}
	if _, err := kernel.Join(nested, nested, kernel.RightJoin); err == nil {
		t.Error("a right join on a list column succeeded")
	}
	if _, err := kernel.Join(nested, nested, kernel.SemiJoin); err == nil {
		t.Error("a semi join on a list column succeeded")
	}

	// The right side builds the table and the left side probes it, so a bad
	// left key is a different code path from a bad right one.
	ok := kernel.Side{Rows: 0, Keys: []*array.Chunked{col(t, dtype.Int64, nil)}}
	if _, err := kernel.Join(nested, ok, kernel.InnerJoin); err == nil {
		t.Error("joining a list column onto an int64 column succeeded")
	}
}

func TestJoinPanics(t *testing.T) {
	full := sideOf(t, int64(1), int64(2))

	tests := []struct {
		name string
		run  func()
	}{
		{"a nil key", func() {
			bad := kernel.Side{Rows: 2, Keys: []*array.Chunked{nil}}
			_, _ = kernel.Join(bad, full, kernel.InnerJoin)
		}},
		{"a key of the wrong length", func() {
			bad := kernel.Side{Rows: 5, Keys: full.Keys}
			_, _ = kernel.Join(bad, full, kernel.InnerJoin)
		}},
		{"a negative row count", func() {
			_, _ = kernel.Join(kernel.Side{Rows: -1}, full, kernel.CrossJoin)
		}},
		{"a different number of keys", func() {
			two := kernel.Side{Rows: 2, Keys: []*array.Chunked{full.Keys[0], full.Keys[0]}}
			_, _ = kernel.Join(full, two, kernel.InnerJoin)
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Error("it did not panic")
				}
			}()
			tt.run()
		})
	}
}

func TestJoinTypeString(t *testing.T) {
	tests := []struct {
		how  kernel.JoinType
		want string
	}{
		{kernel.InnerJoin, "inner"},
		{kernel.LeftJoin, "left"},
		{kernel.RightJoin, "right"},
		{kernel.OuterJoin, "outer"},
		{kernel.SemiJoin, "semi"},
		{kernel.AntiJoin, "anti"},
		{kernel.CrossJoin, "cross"},
		{kernel.JoinType(9), "JoinType(9)"},
	}

	for _, tt := range tests {
		if got := tt.how.String(); got != tt.want {
			t.Errorf("JoinType(%d) is %q, want %q", int(tt.how), got, tt.want)
		}
	}
}
