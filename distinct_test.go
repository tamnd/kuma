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

// repeats returns a frame with the same row written twice, a row that agrees on
// one column and not on the other, and a row that agrees with nothing.
//
//	symbol day qty
//	AAPL   1   100
//	MSFT   1   50
//	AAPL   1   100
//	AAPL   2   100
func repeats(t *testing.T) *kuma.Frame[kuma.Dynamic] {
	t.Helper()

	return mustFrame(t,
		kuma.NewSeries("symbol", "AAPL", "MSFT", "AAPL", "AAPL").Column(),
		kuma.NewSeries("day", int64(1), 1, 1, 2).Column(),
		kuma.NewSeries("qty", int64(100), 50, 100, 100).Column(),
	)
}

func TestDistinct(t *testing.T) {
	tests := []struct {
		name  string
		names []string
		want  []string
	}{
		{
			name: "every column",
			want: []string{"AAPL 1 100", "MSFT 1 50", "AAPL 2 100"},
		},
		{
			name:  "one column",
			names: []string{"symbol"},
			want:  []string{"AAPL 1 100", "MSFT 1 50"},
		},
		{
			name:  "two columns",
			names: []string{"symbol", "day"},
			want:  []string{"AAPL 1 100", "MSFT 1 50", "AAPL 2 100"},
		},
		{
			// The rows are compared by the columns named and not by the ones
			// left out, so this one keeps a row per day and forgets that two
			// different symbols traded on day 1.
			name:  "a column that is not the key",
			names: []string{"day"},
			want:  []string{"AAPL 1 100", "AAPL 2 100"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := repeats(t).Distinct(tt.names...)
			if err != nil {
				t.Fatalf("Distinct: %v", err)
			}
			if r := rows(t, got); !slices.Equal(r, tt.want) {
				t.Errorf("Distinct(%q) = %v, want %v", tt.names, r, tt.want)
			}
		})
	}
}

// TestDistinctKeepsTheFirstRow is the part of the promise the columns that were
// not compared can see: the row that stays is the first of its set, so the
// values it carries along are that row's values and not a later one's.
func TestDistinctKeepsTheFirstRow(t *testing.T) {
	f := mustFrame(t,
		kuma.NewSeries("symbol", "AAPL", "AAPL", "MSFT").Column(),
		kuma.NewSeries("price", 189.5, 190.1, 411.2).Column(),
	)

	got, err := f.Distinct("symbol")
	if err != nil {
		t.Fatalf("Distinct: %v", err)
	}
	if want := []string{"AAPL 189.5", "MSFT 411.2"}; !slices.Equal(rows(t, got), want) {
		t.Errorf("Distinct = %v, want %v", rows(t, got), want)
	}
}

// TestDistinctNullsAgree is the rule a group by follows, which pandas does not:
// two rows that are both missing the value are the same row.
func TestDistinctNullsAgree(t *testing.T) {
	f := mustFrame(t,
		nullKeys(t, "a", "", "a", "").Rename("sym"),
		nullInts(t, "qty", 1, 0, 1, 0),
	)

	got, err := f.Distinct()
	if err != nil {
		t.Fatalf("Distinct: %v", err)
	}
	if want := []string{"a 1", ". ."}; !slices.Equal(rows(t, got), want) {
		t.Errorf("Distinct = %v, want %v", rows(t, got), want)
	}
}

// TestDistinctOfDistinctRows is the frame that has nothing to take out, which
// is answered with the frame itself rather than with a copy of it.
func TestDistinctOfDistinctRows(t *testing.T) {
	f := trades(t)

	got, err := f.Distinct()
	if err != nil {
		t.Fatalf("Distinct: %v", err)
	}
	if got != f {
		t.Errorf("Distinct built a new frame of %d rows, want the one it was given", got.NumRows())
	}
}

// TestDistinctKeepsTheSchemaType is the reason this is not a Dynamic frame like
// a select is. The columns are the ones that arrived, so the handles written for
// them still work and the compiler is what says so.
func TestDistinctKeepsTheSchemaType(t *testing.T) {
	f := typedTrades(t)

	got, err := f.Distinct("symbol")
	if err != nil {
		t.Fatalf("Distinct: %v", err)
	}

	only, err := got.Filter(tradeCols.Symbol.Eq("AAPL"))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if only.NumRows() != 1 {
		t.Errorf("there are %d AAPL rows left, want 1", only.NumRows())
	}
}

// TestDistinctOfNothing is a frame with no columns, which is what selecting
// none of them gives. There is nothing to compare and no rows to compare, so
// the answer is the frame.
func TestDistinctOfNothing(t *testing.T) {
	f, err := trades(t).Select()
	if err != nil {
		t.Fatalf("Select: %v", err)
	}

	got, err := f.Distinct()
	if err != nil {
		t.Fatalf("Distinct: %v", err)
	}
	if got.NumRows() != 0 || got.NumCols() != 0 {
		t.Errorf("Distinct gave %d rows and %d columns, want none of either", got.NumRows(), got.NumCols())
	}
}

func TestDistinctMistakes(t *testing.T) {
	if _, err := trades(t).Distinct("smybol"); !errors.Is(err, kuma.ErrNoColumn) {
		t.Errorf("Distinct on a name that is not there = %v, want ErrNoColumn", err)
	}

	dt := dtype.List{Elem: dtype.Int64}
	data, err := array.NewChunked(dt)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	c, err := kuma.NewColumn("v", data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}

	if _, err := mustFrame(t, c).Distinct(); err == nil {
		t.Error("the distinct rows of a list column succeeded")
	} else if !strings.Contains(err.Error(), dt.String()) {
		t.Errorf("the message is %q, want it to name the type", err.Error())
	}
}
