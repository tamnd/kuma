package plan_test

// The tests for what an explode produces. They are in their own file because
// they need a source with a list column in it, and every other test in the
// package scans one that holds a value per row.

import (
	"errors"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/plan"
)

// orders is a source with three list columns, so that a plan can take one of
// them apart, or two of them together, or one that holds lists of its own.
//
// It hands back the same fields every time it is asked rather than building
// them again, which is what a reader that worked its schema out once does, and
// what makes it worth checking that an operator does not write to them.
type orders struct{}

var orderFields = []dtype.Field{
	{Name: "symbol", Type: dtype.String},
	{Name: "tags", Type: dtype.List{Elem: dtype.String}},
	{Name: "sizes", Type: dtype.List{Elem: dtype.Int64}, Nullable: true},
	{Name: "legs", Type: dtype.List{Elem: dtype.List{Elem: dtype.Int64}}},
}

func (orders) Name() string { return "orders/*.parquet" }

func (orders) Schema() (dtype.Schema, error) {
	return dtype.Schema{Fields: orderFields}, nil
}

var listed = plan.Scan(orders{})

func TestSchemaOfAnExplode(t *testing.T) {
	cases := []struct {
		name string
		node *plan.Node
		want dtype.Schema
	}{
		{
			"one column holds what its lists held",
			plan.Explode(listed, []string{"tags"}),
			schema(field("symbol", dtype.String, false), field("tags", dtype.String, true),
				field("sizes", dtype.List{Elem: dtype.Int64}, true),
				field("legs", dtype.List{Elem: dtype.List{Elem: dtype.Int64}}, false)),
		},
		{
			"the columns stay where they were",
			plan.Explode(listed, []string{"sizes"}),
			schema(field("symbol", dtype.String, false),
				field("tags", dtype.List{Elem: dtype.String}, false),
				field("sizes", dtype.Int64, true),
				field("legs", dtype.List{Elem: dtype.List{Elem: dtype.Int64}}, false)),
		},
		{
			"two columns come apart together",
			plan.Explode(listed, []string{"sizes", "tags"}),
			schema(field("symbol", dtype.String, false), field("tags", dtype.String, true),
				field("sizes", dtype.Int64, true),
				field("legs", dtype.List{Elem: dtype.List{Elem: dtype.Int64}}, false)),
		},
		{
			"a column of lists of lists comes back a column of lists",
			plan.Explode(plan.Project(listed, []plan.Projection{{Expr: plan.Col("legs")}}), []string{"legs"}),
			schema(field("legs", dtype.List{Elem: dtype.Int64}, true)),
		},
		{
			"an explode of an explode is what is left after both",
			plan.Explode(plan.Explode(listed, []string{"legs"}), []string{"legs"}),
			schema(field("symbol", dtype.String, false),
				field("tags", dtype.List{Elem: dtype.String}, false),
				field("sizes", dtype.List{Elem: dtype.Int64}, true),
				field("legs", dtype.Int64, true)),
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.node.Schema()
			if err != nil {
				t.Fatalf("Schema: %v", err)
			}
			if !got.Equal(tt.want) {
				t.Errorf("the plan gives %s, want %s", got, tt.want)
			}
		})
	}
}

// TestExplodedColumnsAreNullable is the rule worth its own test, since it is
// the one thing about the result that is not read off the input: a row holding
// nothing becomes a row holding a missing value, so a column of lists that were
// never missing comes out as a column that can be.
func TestExplodedColumnsAreNullable(t *testing.T) {
	s, err := plan.Explode(listed, []string{"tags"}).Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}

	f, ok := s.Field("tags")
	if !ok {
		t.Fatal("the exploded column is not in the result")
	}
	if !f.Nullable {
		t.Error("the exploded column is not nullable, and an empty row becomes a missing value")
	}
	if other, _ := s.Field("symbol"); other.Nullable {
		t.Error("a column the explode did not touch became nullable")
	}
}

func TestSchemaTurnsAwayAnExplode(t *testing.T) {
	cases := []struct {
		name string
		node *plan.Node
		want string
	}{
		{
			"an explode with no column to take apart",
			plan.Explode(listed, nil),
			"no column to take apart",
		},
		{
			"an explode of a column that is not there",
			plan.Explode(listed, []string{"tag"}),
			`column "tag" not found in Explode`,
		},
		{
			"an explode of a column that holds one value per row",
			plan.Explode(listed, []string{"symbol"}),
			"holds one value per row",
		},
		{
			"an explode that names one good column and one bad one",
			plan.Explode(listed, []string{"tags", "symbol"}),
			"holds one value per row",
		},
		{
			"an explode that names a column twice",
			plan.Explode(listed, []string{"tags", "tags"}),
			"names tags twice",
		},
		{
			"an explode over a plan that is already wrong",
			plan.Explode(plan.Scan(missing{}), []string{"tags"}),
			"no such file",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.node.Schema()
			if err == nil {
				t.Fatal("the plan was allowed")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("the error is %q, want it to say %q", err, tt.want)
			}
		})
	}
}

// TestAnExplodeOfTheWrongColumnIsTheKumaError checks that the two mistakes with
// a sentinel behind them come back as the values the kuma package exports, so
// that one errors.Is covers a query wherever it went wrong.
func TestAnExplodeOfTheWrongColumnIsTheKumaError(t *testing.T) {
	if _, err := plan.Explode(listed, []string{"symbol"}).Schema(); !errors.Is(err, plan.ErrWrongType) {
		t.Errorf("the error for a column of values is %v, want it to be ErrWrongType", err)
	}
	if _, err := plan.Explode(listed, []string{"tags", "tags"}).Schema(); !errors.Is(err, plan.ErrDuplicateColumn) {
		t.Errorf("the error for a name given twice is %v, want it to be ErrDuplicateColumn", err)
	}
}

// TestAnExplodeDoesNotWriteToItsInput is the immutability every operator here
// promises, checked on the one that changes the fields it was given rather than
// building new ones. The source hands out the same fields every time, so a
// write to them would be a source that changed type after being read once.
func TestAnExplodeDoesNotWriteToItsInput(t *testing.T) {
	if _, err := plan.Explode(listed, []string{"tags"}).Schema(); err != nil {
		t.Fatalf("Schema: %v", err)
	}

	s, err := listed.Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}
	f, _ := s.Field("tags")
	if _, ok := f.Type.(dtype.List); !ok {
		t.Errorf("the source now says tags is a %s, want the list it holds", f.Type)
	}
}
