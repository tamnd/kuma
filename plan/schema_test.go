package plan_test

// The tests for what a plan produces and for what it turns away. They are one
// subject: working out the columns is what checks the query, so a plan that has
// a schema is a plan that will run and a plan that has none says why.

import (
	"errors"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/plan"
)

// quotes is the other side of the joins here. It shares the symbol column with
// trades, which is the case where a join gives one column for the pair, and it
// has one column of its own.
type quotes struct{}

func (quotes) Name() string { return "quotes/*.parquet" }

func (quotes) Schema() (dtype.Schema, error) {
	return dtype.Schema{Fields: []dtype.Field{
		{Name: "symbol", Type: dtype.String},
		{Name: "bid", Type: dtype.Float64, Nullable: true},
	}}, nil
}

// sameName is a source whose symbol column is an int32, which is the case where
// two key columns of one name cannot become one column.
type sameName struct{}

func (sameName) Name() string { return "ids" }

func (sameName) Schema() (dtype.Schema, error) {
	return dtype.Schema{Fields: []dtype.Field{{Name: "symbol", Type: dtype.Int32}}}, nil
}

// twice is a file with two columns of one name, which a reader is allowed to
// describe and a frame cannot hold.
type twice struct{}

func (twice) Name() string { return "twice.csv" }

func (twice) Schema() (dtype.Schema, error) {
	return dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64},
		{Name: "id", Type: dtype.String},
	}}, nil
}

// missing is a source that cannot say what it holds, such as a directory of
// files that is not there.
type missing struct{}

func (missing) Name() string { return "gone" }

func (missing) Schema() (dtype.Schema, error) {
	return dtype.Schema{}, errors.New("kuma: gone: no such file or directory")
}

var (
	bid = plan.Col("bid")

	// The amount is a column that is worked out rather than read, which is the
	// case where a plan knows the type and nothing about what will be in it.
	// The cast is there because kuma does not mix a float64 and an int64 on its
	// own.
	amount = plan.Arith(kernel.OpMul, price, plan.Cast(dtype.Float64, qty))

	scan     = plan.Scan(trades{})
	quoted   = plan.Scan(quotes{})
	onSymbol = []plan.JoinKey{{Left: symbol, Right: symbol}}
)

// field is a column of a schema, written short because these tables are mostly
// made of them.
func field(name string, dt dtype.DataType, nullable bool) dtype.Field {
	return dtype.Field{Name: name, Type: dt, Nullable: nullable}
}

// schema is the schema of the fields.
func schema(fields ...dtype.Field) dtype.Schema { return dtype.Schema{Fields: fields} }

func TestSchemaOfAPlan(t *testing.T) {
	cases := []struct {
		name string
		node *plan.Node
		want dtype.Schema
	}{
		{
			"a scan is what the source holds",
			scan,
			schema(field("symbol", dtype.String, false), field("price", dtype.Float64, false),
				field("qty", dtype.Int64, false)),
		},
		{
			"a narrowed scan is the columns it reads, in the order the source has them",
			plan.ScanOnly(trades{}, []string{"qty", "symbol"}),
			schema(field("symbol", dtype.String, false), field("qty", dtype.Int64, false)),
		},
		{
			"a filter leaves the columns alone",
			plan.Filter(scan, plan.Compare(kernel.OpGt, price, plan.Lit(100.0))),
			schema(field("symbol", dtype.String, false), field("price", dtype.Float64, false),
				field("qty", dtype.Int64, false)),
		},
		{
			"a projection is the columns it names",
			plan.Project(scan, []plan.Projection{{Expr: symbol}, {Expr: amount, As: "amount"}}),
			schema(field("symbol", dtype.String, false), field("amount", dtype.Float64, true)),
		},
		{
			"a renamed column keeps what its source said about it",
			plan.Project(quoted, []plan.Projection{{Expr: bid, As: "best"}}),
			schema(field("best", dtype.Float64, true)),
		},
		{
			"an aggregate is the keys and then the aggregations",
			plan.Aggregate(scan, []*plan.Expr{symbol}, []plan.Agg{
				{Func: plan.AggSum, Expr: amount, As: "volume"},
				{Func: plan.AggMean, Expr: price},
				{Func: plan.AggSize},
			}),
			schema(field("symbol", dtype.String, false), field("volume", dtype.Float64, true),
				field("price", dtype.Float64, true), field("size", dtype.Int64, false)),
		},
		{
			"an aggregate with no keys is the aggregations",
			plan.Aggregate(scan, nil, []plan.Agg{{Func: plan.AggMin, Expr: qty}}),
			schema(field("qty", dtype.Int64, true)),
		},
		{
			"an inner join gives one column for a key both sides call the same thing",
			plan.Join(scan, quoted, onSymbol, kernel.InnerJoin),
			schema(field("symbol", dtype.String, false), field("price", dtype.Float64, false),
				field("qty", dtype.Int64, false), field("bid", dtype.Float64, true)),
		},
		{
			"a left join can leave the right side missing",
			plan.Join(scan, quoted, onSymbol, kernel.LeftJoin),
			schema(field("symbol", dtype.String, false), field("price", dtype.Float64, false),
				field("qty", dtype.Int64, false), field("bid", dtype.Float64, true)),
		},
		{
			"a right join can leave the left side missing",
			plan.Join(scan, quoted, onSymbol, kernel.RightJoin),
			schema(field("symbol", dtype.String, true), field("price", dtype.Float64, true),
				field("qty", dtype.Int64, true), field("bid", dtype.Float64, true)),
		},
		{
			"a semi join is the left side and nothing else",
			plan.Join(scan, quoted, onSymbol, kernel.SemiJoin),
			schema(field("symbol", dtype.String, false), field("price", dtype.Float64, false),
				field("qty", dtype.Int64, false)),
		},
		{
			"a join on two names keeps both columns",
			plan.Join(quoted, plan.Project(scan, []plan.Projection{{Expr: symbol, As: "sym"}}),
				[]plan.JoinKey{{Left: symbol, Right: plan.Col("sym")}}, kernel.InnerJoin),
			schema(field("symbol", dtype.String, false), field("bid", dtype.Float64, true),
				field("sym", dtype.String, false)),
		},
		{
			"a cross join is both sides",
			plan.Join(quoted, plan.Project(scan, []plan.Projection{{Expr: qty}}), nil, kernel.CrossJoin),
			schema(field("symbol", dtype.String, false), field("bid", dtype.Float64, true),
				field("qty", dtype.Int64, false)),
		},
		{
			"a sort leaves the columns alone",
			plan.Sort(scan, []plan.SortKey{{Expr: price, Descending: true}}),
			schema(field("symbol", dtype.String, false), field("price", dtype.Float64, false),
				field("qty", dtype.Int64, false)),
		},
		{
			"a limit leaves the columns alone",
			plan.Limit(scan, 10, 5),
			schema(field("symbol", dtype.String, false), field("price", dtype.Float64, false),
				field("qty", dtype.Int64, false)),
		},
		{
			"a distinct leaves the columns alone",
			plan.Distinct(scan, []*plan.Expr{symbol}),
			schema(field("symbol", dtype.String, false), field("price", dtype.Float64, false),
				field("qty", dtype.Int64, false)),
		},
		{
			"a plan several operators deep is worked out from the source up",
			plan.Limit(
				plan.Sort(
					plan.Aggregate(
						plan.Filter(scan, plan.Compare(kernel.OpGt, qty, plan.Lit(int64(0)))),
						[]*plan.Expr{symbol},
						[]plan.Agg{{Func: plan.AggSum, Expr: amount, As: "volume"}}),
					[]plan.SortKey{{Expr: plan.Col("volume"), Descending: true}}),
				0, 10),
			schema(field("symbol", dtype.String, false), field("volume", dtype.Float64, true)),
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.node.Schema()
			if err != nil {
				t.Fatalf("Schema: %v", err)
			}
			if !got.Equal(tt.want) {
				t.Errorf("the plan gives\n  %s\nwant\n  %s", got, tt.want)
			}
			if err := tt.node.Validate(); err != nil {
				t.Errorf("Validate says %q about a plan that has a schema", err)
			}
		})
	}
}

func TestSchemaTurnsAway(t *testing.T) {
	cases := []struct {
		name string
		node *plan.Node
		want string
	}{
		{"a plan with no operator in it", nil, "no operator"},
		{"a scan with nothing to read", plan.Scan(nil), "nothing to read"},
		{"a source that cannot say what it holds", plan.Scan(missing{}), "no such file"},
		{"a source with two columns of one name", plan.Scan(twice{}), `two columns are called "id"`},
		{
			"a scan of a column the source does not have",
			plan.ScanOnly(trades{}, []string{"prcie"}),
			`column "prcie" not found in Scan`,
		},
		{"a filter with no condition", plan.Filter(scan, nil), "no condition"},
		{
			"a filter on a column that is not there",
			plan.Filter(scan, plan.Compare(kernel.OpGt, plan.Col("prise"), plan.Lit(1.0))),
			`column "prise" not found in Filter`,
		},
		{
			"a filter on something that is not a condition",
			plan.Filter(scan, price),
			"condition",
		},
		{"a projection of nothing", plan.Project(scan, []plan.Projection{{}}), "a projection of nothing"},
		{
			"a projection of a column that is not there",
			plan.Project(scan, []plan.Projection{{Expr: plan.Col("volume")}}),
			`column "volume" not found in Project`,
		},
		{
			"a projection that names two columns the same",
			plan.Project(scan, []plan.Projection{{Expr: price, As: "x"}, {Expr: qty, As: "x"}}),
			`two columns are called "x"`,
		},
		{
			"a group by a column that is not there",
			plan.Aggregate(scan, []*plan.Expr{plan.Col("venue")}, nil),
			`column "venue" not found in Aggregate`,
		},
		{
			"a total of a column of text",
			plan.Aggregate(scan, nil, []plan.Agg{{Func: plan.AggSum, Expr: symbol}}),
			"there is no sum of a string column",
		},
		{
			"an aggregation of nothing",
			plan.Aggregate(scan, nil, []plan.Agg{{Func: plan.AggSum}}),
			"no column to work out",
		},
		{
			"a variance with a negative ddof",
			plan.Aggregate(scan, nil, []plan.Agg{{Func: plan.AggVar, Expr: price, DDoF: -1}}),
			"ddof of -1",
		},
		{
			"a quantile outside 0 to 1",
			plan.Aggregate(scan, nil, []plan.Agg{{Func: plan.AggQuantile, Expr: price, Q: 2}}),
			"not between 0 and 1",
		},
		{
			"a quantile with no such interpolation",
			plan.Aggregate(scan, nil, []plan.Agg{{Func: plan.AggQuantile, Expr: price, Q: 0.5, How: 9}}),
			"no interpolation there is",
		},
		{
			"an aggregation named after the key it sits beside",
			plan.Aggregate(scan, []*plan.Expr{symbol}, []plan.Agg{
				{Func: plan.AggMean, Expr: price, As: "symbol"},
			}),
			`two columns are called "symbol"`,
		},
		{
			"a cross join with keys",
			plan.Join(scan, quoted, onSymbol, kernel.CrossJoin),
			"cross join with 1 keys",
		},
		{"an inner join with no keys", plan.Join(scan, quoted, nil, kernel.InnerJoin), "inner join with no keys"},
		{"a join of no known kind", plan.Join(scan, quoted, onSymbol, 99), "no known kind"},
		{
			"a join key with only one side",
			plan.Join(scan, quoted, []plan.JoinKey{{Left: symbol}}, kernel.InnerJoin),
			"only one side",
		},
		{
			"a join key that is not there on the right",
			plan.Join(scan, quoted, []plan.JoinKey{{Left: symbol, Right: plan.Col("ticker")}}, kernel.InnerJoin),
			`column "ticker" not found in Join`,
		},
		{
			"one name over two types of key column",
			plan.Join(scan, plan.Scan(sameName{}), onSymbol, kernel.InnerJoin),
			"string on one side and a int32 on the other",
		},
		{
			"a join that would give two columns of one name",
			plan.Join(scan, quoted, []plan.JoinKey{{Left: price, Right: bid}}, kernel.InnerJoin),
			`two columns are called "symbol"`,
		},
		{"a sort with nothing to sort by", plan.Sort(scan, nil), "nothing to sort by"},
		{"a sort by nothing", plan.Sort(scan, []plan.SortKey{{}}), "a sort by nothing"},
		{
			"a sort by a column that is not there",
			plan.Sort(scan, []plan.SortKey{{Expr: plan.Col("time")}}),
			`column "time" not found in Sort`,
		},
		{"a limit that skips a negative number of rows", plan.Limit(scan, -1, 10), "skips -1 rows"},
		{"a limit of a negative number of rows", plan.Limit(scan, 0, -2), "a limit of -2 rows"},
		{
			"a distinct by a column that is not there",
			plan.Distinct(scan, []*plan.Expr{plan.Col("venue")}),
			`column "venue" not found in Distinct`,
		},
		{
			"a mistake under an operator that is fine itself",
			plan.Limit(plan.Sort(plan.Filter(scan, nil), []plan.SortKey{{Expr: price}}), 0, 1),
			"no condition",
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			s, err := tt.node.Schema()
			if err == nil {
				t.Fatalf("the plan gives %s, want an error about %s", s, tt.want)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("the error is %q, want it to say %q", err, tt.want)
			}
			if got := tt.node.Validate(); got == nil || got.Error() != err.Error() {
				t.Errorf("Validate says %v, and Schema says %v", got, err)
			}
		})
	}
}

// TestSchemaOfEveryAggregation is the type each aggregation promises, written
// out rather than worked out, so that an aggregation wired to the wrong rule
// shows up here.
func TestSchemaOfEveryAggregation(t *testing.T) {
	cases := []struct {
		fn   plan.AggFunc
		want dtype.DataType // over the int64 column, or nil for one it refuses
		text dtype.DataType // over the string column, or nil for one it refuses
	}{
		{plan.AggSum, dtype.Int64, nil},
		{plan.AggMean, dtype.Float64, nil},
		{plan.AggMin, dtype.Int64, dtype.String},
		{plan.AggMax, dtype.Int64, dtype.String},
		{plan.AggCount, dtype.Int64, dtype.Int64},
		{plan.AggFirst, dtype.Int64, dtype.String},
		{plan.AggLast, dtype.Int64, dtype.String},
		{plan.AggVar, dtype.Float64, nil},
		{plan.AggStd, dtype.Float64, nil},
		{plan.AggMedian, dtype.Float64, nil},
		{plan.AggQuantile, dtype.Float64, nil},
		{plan.AggNUnique, dtype.Int64, dtype.Int64},
	}

	for _, tt := range cases {
		t.Run(tt.fn.String(), func(t *testing.T) {
			for _, over := range []struct {
				col  *plan.Expr
				want dtype.DataType
			}{{qty, tt.want}, {symbol, tt.text}} {
				n := plan.Aggregate(scan, nil, []plan.Agg{{Func: tt.fn, Expr: over.col, As: "x"}})
				s, err := n.Schema()
				switch {
				case over.want == nil && err == nil:
					t.Errorf("%s of %s gives a %s column, want it turned away",
						tt.fn, over.col, s.Fields[0].Type)
				case over.want == nil:
				case err != nil:
					t.Errorf("%s of %s says %q, want a %s column", tt.fn, over.col, err, over.want)
				case !dtype.Equal(s.Fields[0].Type, over.want):
					t.Errorf("%s of %s gives a %s column, want a %s one",
						tt.fn, over.col, s.Fields[0].Type, over.want)
				}
			}
		})
	}

	// Size reads no column, so it is the one that is not in the table.
	s, err := plan.Aggregate(scan, nil, []plan.Agg{{Func: plan.AggSize}}).Schema()
	if err != nil {
		t.Fatalf("Size: %v", err)
	}
	if !s.Equal(schema(field("size", dtype.Int64, false))) {
		t.Errorf("Size gives %s, want a size column of int64", s)
	}
}

// TestSchemaSaysWhichColumnIsMissing is the error a caller meets most often, so
// it is worth checking that it arrives whole rather than as a sentinel.
func TestSchemaSaysWhichColumnIsMissing(t *testing.T) {
	_, err := plan.Project(scan, []plan.Projection{{Expr: plan.Col("prise")}}).Schema()
	if !errors.Is(err, plan.ErrNoColumn) {
		t.Fatalf("the error is %v, want it to be ErrNoColumn", err)
	}

	var ce *plan.ColumnError
	if !errors.As(err, &ce) {
		t.Fatalf("the error is %T, want a ColumnError", err)
	}
	if ce.Op != "Project" {
		t.Errorf("the error names %q as the operator, want Project", ce.Op)
	}
	if got := err.Error(); !strings.Contains(got, "did you mean: price?") {
		t.Errorf("the error is %q, want it to suggest price", got)
	}
}

// TestSchemaOfADuplicateIsTheKumaError checks that the two column names one
// frame cannot hold come back as the error the frame itself gives, so that one
// errors.Is covers a query however far it got.
func TestSchemaOfADuplicateIsTheKumaError(t *testing.T) {
	_, err := plan.Project(scan, []plan.Projection{{Expr: price, As: "x"}, {Expr: qty, As: "x"}}).Schema()
	if !errors.Is(err, plan.ErrDuplicateColumn) {
		t.Fatalf("the error is %v, want it to be ErrDuplicateColumn", err)
	}
}

// BenchmarkSchema is what checking a plan costs, which is what a query pays
// once before it reads anything, and once more for each pass that wants to know
// what an operator produces.
func BenchmarkSchema(b *testing.B) {
	n := plan.Limit(
		plan.Sort(
			plan.Aggregate(
				plan.Filter(scan, plan.Compare(kernel.OpGt, price, plan.Lit(100.0))),
				[]*plan.Expr{symbol},
				[]plan.Agg{
					{Func: plan.AggSum, Expr: amount, As: "volume"},
					{Func: plan.AggMean, Expr: price, As: "avg"},
					{Func: plan.AggSize},
				}),
			[]plan.SortKey{{Expr: plan.Col("volume"), Descending: true}}),
		0, 10)

	for b.Loop() {
		s, err := n.Schema()
		if err != nil {
			b.Fatal(err)
		}
		schemaSink = s
	}
}

// schemaSink keeps the benchmark above from being optimized away.
var schemaSink dtype.Schema

// TestSchemaIsNotAffectedByReadingIt makes sure a plan can be asked twice, since
// the passes ask it once each and a node that changed under them would be a
// plan that means something different by the time it runs.
func TestSchemaIsNotAffectedByReadingIt(t *testing.T) {
	n := plan.Project(scan, []plan.Projection{{Expr: amount, As: "amount"}})
	first, err := n.Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}
	first.Fields[0].Name = "changed"

	second, err := n.Schema()
	if err != nil {
		t.Fatalf("Schema: %v", err)
	}
	if second.Fields[0].Name != "amount" {
		t.Errorf("the second read gives %s, want the column still called amount", second)
	}
}
