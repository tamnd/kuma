package plan

import (
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// Schema returns the columns this operator produces, and an error saying what
// is wrong with the plan when it would produce none.
//
// It works from the source down. Every operator asks its input what it has,
// checks its own expressions against that, and says what it leaves behind, so
// asking the last operator of a plan checks the whole of it. A name is resolved
// against the columns that reach the operator that used it rather than against
// the ones the files hold, which is what makes a name that a projection dropped
// two steps earlier an error here and not a surprise later.
//
// Nullable says what is known rather than what will happen. A column that comes
// through an operator unchanged keeps what its source said about it, a column
// that is worked out is nullable because arithmetic over a missing value is
// missing, and the side of an outer join that may not be there is nullable
// whatever it said before. A count is the one thing that never goes missing.
//
// The schema of a plan is not the schema of the frame it produces in one
// respect: a frame says whether a column has any missing values in it, which is
// a fact about the data, and a plan can only say whether one might.
func (n *Node) Schema() (dtype.Schema, error) {
	if n == nil {
		return dtype.Schema{}, errNoPlan
	}

	switch n.op {
	case OpScan:
		return n.scanSchema()
	case OpFilter:
		return n.filterSchema()
	case OpProject:
		return n.projectSchema()
	case OpAggregate:
		return n.aggregateSchema()
	case OpJoin:
		return n.joinSchema()
	case OpSort:
		return n.sortSchema()
	case OpLimit:
		return n.limitSchema()
	case OpDistinct:
		return n.distinctSchema()
	default:
		return n.explodeSchema()
	}
}

// Validate returns the first thing wrong with a plan, and nil for a plan that
// will run.
//
// It is [Node.Schema] without the schema. Working out what a plan produces
// means checking every expression in it against the columns that reach it, so
// there is one walk rather than two, and this is the name for the half of the
// answer a caller who only wants to know whether the query is right cares
// about.
//
// What it catches is everything that can be known without reading any data: a
// column that is not there, two columns of one name, a comparison between types
// with nothing in common, a literal that does not fit the column it was written
// against, a filter on something that is not a condition, an order on something
// there is no order for, a group by on a column there is no way to compare, an
// aggregation of a column it has no meaning for, and the arguments of an
// operator that make no sense on their own, such as a negative limit or a
// quantile at two. What it does not catch is anything the data decides, such as
// a file that turns out not to be there or a total that overflows.
func (n *Node) Validate() error {
	_, err := n.Schema()
	return err
}

func (n *Node) scanSchema() (dtype.Schema, error) {
	if n.src == nil {
		return dtype.Schema{}, errors.New("kuma: a scan with nothing to read")
	}

	s, err := n.src.Schema()
	if err != nil {
		return dtype.Schema{}, err
	}

	// A file with two columns of one name is a real file, and the schema type
	// allows it so that a reader can describe one. A frame does not, so a plan
	// that would build a frame out of it says so here rather than at the end of
	// the read.
	s, err = noDuplicates(s.Fields)
	if err != nil {
		return dtype.Schema{}, err
	}

	// A scan reading a run of rows is checked the same way a limit is, since it
	// is a limit that a pass wrote into the read.
	if r := n.rows; r != nil {
		if r.off < 0 {
			return dtype.Schema{}, fmt.Errorf("kuma: a scan that skips %d rows", r.off)
		}
		if r.n < 0 {
			return dtype.Schema{}, fmt.Errorf("kuma: a scan of %d rows", r.n)
		}
	}
	if n.read == nil {
		return s, nil
	}

	// The columns come out in the order the source has them rather than the
	// order they were named in, since a scan that is asked for two columns of a
	// file reads the file, and a read goes forwards. A projection is what puts
	// the columns in an order, and a pushdown that leaves one behind leaves the
	// order to it.
	want := make(map[string]bool, len(n.read))
	for _, name := range n.read {
		if s.Index(name) < 0 {
			return dtype.Schema{}, noColumn(n.op.String(), name, s.Names())
		}
		want[name] = true
	}
	fields := make([]dtype.Field, 0, len(want))
	for _, f := range s.Fields {
		if want[f.Name] {
			fields = append(fields, f)
		}
	}
	return dtype.Schema{Fields: fields, Metadata: s.Metadata}, nil
}

func (n *Node) filterSchema() (dtype.Schema, error) {
	s, err := n.l.Schema()
	if err != nil {
		return dtype.Schema{}, err
	}
	if n.pred == nil {
		return dtype.Schema{}, errors.New("kuma: a filter with no condition")
	}
	if err := conditionType(n.pred, s); err != nil {
		return dtype.Schema{}, inOp(n.op, err)
	}

	// A filter chooses rows and leaves the columns alone. It can only take
	// missing values away, never add them, so what the input said still holds.
	return s, nil
}

func (n *Node) projectSchema() (dtype.Schema, error) {
	s, err := n.l.Schema()
	if err != nil {
		return dtype.Schema{}, err
	}

	fields := make([]dtype.Field, len(n.cols))
	for i, p := range n.cols {
		if p.Expr == nil {
			return dtype.Schema{}, fmt.Errorf("kuma: a projection of nothing, at position %d", i)
		}
		f, err := fieldOf(p.Expr, s, p.Name())
		if err != nil {
			return dtype.Schema{}, inOp(n.op, err)
		}
		fields[i] = f
	}
	return noDuplicates(fields)
}

func (n *Node) aggregateSchema() (dtype.Schema, error) {
	s, err := n.l.Schema()
	if err != nil {
		return dtype.Schema{}, err
	}

	// The keys first and the aggregations after, which is the order a group by
	// produces them in and the order they were written in.
	fields := make([]dtype.Field, 0, len(n.by)+len(n.aggs))
	for i, e := range n.by {
		if e == nil {
			return dtype.Schema{}, fmt.Errorf("kuma: a group by nothing, at position %d", i)
		}
		f, err := fieldOf(e, s, e.String())
		if err != nil {
			return dtype.Schema{}, inOp(n.op, err)
		}
		if _, err := kernel.GroupKeyType(f.Type); err != nil {
			return dtype.Schema{}, fmt.Errorf("kuma: cannot group by %s: %w", e, err)
		}
		fields = append(fields, f)
	}

	for _, a := range n.aggs {
		f, err := aggField(a, s)
		if err != nil {
			return dtype.Schema{}, inOp(n.op, err)
		}
		fields = append(fields, f)
	}
	return noDuplicates(fields)
}

func (n *Node) joinSchema() (dtype.Schema, error) {
	left, err := n.l.Schema()
	if err != nil {
		return dtype.Schema{}, err
	}
	right, err := n.r.Schema()
	if err != nil {
		return dtype.Schema{}, err
	}
	if n.how < kernel.InnerJoin || n.how > kernel.CrossJoin {
		return dtype.Schema{}, fmt.Errorf("kuma: a join of no known kind, %s", n.how)
	}
	if (n.how == kernel.CrossJoin) != (len(n.on) == 0) {
		if n.how == kernel.CrossJoin {
			return dtype.Schema{}, fmt.Errorf("kuma: a cross join with %d keys, which pairs every row with every row and reads none of them",
				len(n.on))
		}
		return dtype.Schema{}, fmt.Errorf("kuma: a %s join with no keys", n.how)
	}

	shared, err := n.joinKeyNames(left, right)
	if err != nil {
		return dtype.Schema{}, err
	}

	// A join that keeps a row with nothing on the other side fills that side
	// with nulls, so the columns that come from it may be missing whatever they
	// held before.
	nullLeft := n.how == kernel.RightJoin || n.how == kernel.OuterJoin
	nullRight := n.how == kernel.LeftJoin || n.how == kernel.OuterJoin

	fields := make([]dtype.Field, 0, len(left.Fields)+len(right.Fields))
	for _, f := range left.Fields {
		f.Nullable = f.Nullable || nullLeft
		fields = append(fields, f)
	}

	// A semi or an anti join answers a question about the left side, so the
	// right side contributes the question and no columns.
	if n.how == kernel.SemiJoin || n.how == kernel.AntiJoin {
		return noDuplicates(fields)
	}

	for _, f := range right.Fields {
		if shared[f.Name] {
			continue
		}
		f.Nullable = f.Nullable || nullRight
		fields = append(fields, f)
	}
	return noDuplicates(fields)
}

// joinKeyNames checks the keys of a join and returns the names that the result
// has one column for rather than two.
//
// Two key columns called the same thing hold the same values wherever they
// matched, so keeping both would be a name clash over a pair of columns that
// agree. Two called different things are both kept, since a caller who wrote
// two names probably wants to see both.
func (n *Node) joinKeyNames(left, right dtype.Schema) (map[string]bool, error) {
	oneColumn := n.how != kernel.SemiJoin && n.how != kernel.AntiJoin
	shared := make(map[string]bool, len(n.on))
	for i, k := range n.on {
		if k.Left == nil || k.Right == nil {
			return nil, fmt.Errorf("kuma: a join key with only one side, at position %d", i)
		}

		lt, err := TypeOf(k.Left, left)
		if err != nil {
			return nil, inOp(n.op, err)
		}
		rt, err := TypeOf(k.Right, right)
		if err != nil {
			return nil, inOp(n.op, err)
		}
		if _, err := kernel.GroupKeyType(lt); err != nil {
			return nil, fmt.Errorf("kuma: cannot join on %s: %w", k.Left, err)
		}
		if _, err := kernel.GroupKeyType(rt); err != nil {
			return nil, fmt.Errorf("kuma: cannot join on %s: %w", k.Right, err)
		}

		name, ok := bothSides(k)
		if !ok || !oneColumn {
			continue
		}
		if !dtype.Equal(lt, rt) {
			// The key encoding widens, so the join itself is fine. Putting a
			// %s and a %s in one column under one name is not.
			return nil, fmt.Errorf("kuma: the join key %s is a %s on one side and a %s on the other: %w",
				name, lt, rt, ErrWrongType)
		}
		shared[name] = true
	}
	return shared, nil
}

func (n *Node) sortSchema() (dtype.Schema, error) {
	s, err := n.l.Schema()
	if err != nil {
		return dtype.Schema{}, err
	}
	if len(n.sort) == 0 {
		return dtype.Schema{}, errors.New("kuma: a sort with nothing to sort by")
	}

	for i, k := range n.sort {
		if k.Expr == nil {
			return dtype.Schema{}, fmt.Errorf("kuma: a sort by nothing, at position %d", i)
		}
		dt, err := TypeOf(k.Expr, s)
		if err != nil {
			return dtype.Schema{}, inOp(n.op, err)
		}
		if !kernel.HasOrder(dt) {
			return dtype.Schema{}, fmt.Errorf("kuma: cannot sort by %s, which is a %s: %w", k.Expr, dt, ErrWrongType)
		}
	}

	// A sort moves rows and leaves the columns as they are.
	return s, nil
}

func (n *Node) limitSchema() (dtype.Schema, error) {
	s, err := n.l.Schema()
	if err != nil {
		return dtype.Schema{}, err
	}
	if n.off < 0 {
		return dtype.Schema{}, fmt.Errorf("kuma: a limit that skips %d rows", n.off)
	}
	if n.n < 0 {
		return dtype.Schema{}, fmt.Errorf("kuma: a limit of %d rows", n.n)
	}
	return s, nil
}

func (n *Node) distinctSchema() (dtype.Schema, error) {
	s, err := n.l.Schema()
	if err != nil {
		return dtype.Schema{}, err
	}

	// With nothing named it compares every column, so every column has to be
	// one there is a way to compare.
	if len(n.by) == 0 {
		for _, f := range s.Fields {
			if _, err := kernel.GroupKeyType(f.Type); err != nil {
				return dtype.Schema{}, fmt.Errorf("kuma: cannot take the distinct rows of %s: %w", f.Name, err)
			}
		}
		return s, nil
	}

	for i, e := range n.by {
		if e == nil {
			return dtype.Schema{}, fmt.Errorf("kuma: a distinct by nothing, at position %d", i)
		}
		dt, err := TypeOf(e, s)
		if err != nil {
			return dtype.Schema{}, inOp(n.op, err)
		}
		if _, err := kernel.GroupKeyType(dt); err != nil {
			return dtype.Schema{}, fmt.Errorf("kuma: cannot take the distinct rows of %s: %w", e, err)
		}
	}
	return s, nil
}

// explodeSchema is the input with each named column holding what its lists held
// rather than the lists themselves.
//
// The columns keep their names and their places, since an explode changes what
// is in a column and not which columns there are. Every one it touches comes out
// nullable whatever the lists said, because a row holding nothing becomes a row
// holding a missing value, and a column with no lists in it at all still has
// that rule over it.
func (n *Node) explodeSchema() (dtype.Schema, error) {
	s, err := n.l.Schema()
	if err != nil {
		return dtype.Schema{}, err
	}
	// Wrapped rather than plain, unlike the other operators that arrive with a
	// part missing, because this one is a mistake a caller makes rather than a
	// plan that was built wrong: the eager Explode reports the same thing for
	// the same call and one errors.Is has to cover both.
	if len(n.names) == 0 {
		return dtype.Schema{}, fmt.Errorf("kuma: an explode with no column to take apart: %w", ErrNoColumn)
	}

	fields := slices.Clone(s.Fields)
	done := make(map[string]bool, len(n.names))
	for _, name := range n.names {
		i := s.Index(name)
		if i < 0 {
			return dtype.Schema{}, noColumn(n.op.String(), name, s.Names())
		}
		if done[name] {
			return dtype.Schema{}, fmt.Errorf("kuma: an explode names %s twice, "+
				"and a column that has been taken apart is not a list any more: %w", name, ErrDuplicateColumn)
		}
		list, ok := fields[i].Type.(dtype.List)
		if !ok {
			return dtype.Schema{}, fmt.Errorf("kuma: cannot explode %s, which is a %s and holds one value per row: %w",
				name, fields[i].Type, ErrWrongType)
		}
		fields[i].Type = list.Elem
		fields[i].Nullable = true
		done[name] = true
	}
	return noDuplicates(fields)
}

// fieldOf is the column an expression produces, under the name it is given.
//
// A bare column keeps the field it came from, so renaming one carries what the
// source said about it and what it holds. Anything worked out is a new column,
// and it is nullable because every step that reads a missing value produces
// one.
func fieldOf(e *Expr, s dtype.Schema, name string) (dtype.Field, error) {
	if e.Kind() == KindColumn {
		f, ok := s.Field(e.Name())
		if !ok {
			return dtype.Field{}, noColumn("", e.Name(), s.Names())
		}
		f.Name = name
		return f, nil
	}

	dt, err := TypeOf(e, s)
	if err != nil {
		return dtype.Field{}, err
	}
	return dtype.Field{Name: name, Type: dt, Nullable: true}, nil
}

// aggField is the column one aggregation produces.
func aggField(a Agg, s dtype.Schema) (dtype.Field, error) {
	// A count is the one answer that is never missing, since a group with
	// nothing in it has none of it, and a size counts rows without reading a
	// column at all.
	counted := a.Func == AggCount || a.Func == AggSize
	if a.Func == AggSize {
		return dtype.Field{Name: a.Name(), Type: dtype.Int64}, nil
	}
	if a.Expr == nil {
		return dtype.Field{}, fmt.Errorf("kuma: %s has no column to work out", a)
	}

	dt, err := TypeOf(a.Expr, s)
	if err != nil {
		return dtype.Field{}, err
	}
	out, err := aggType(a, dt)
	if err != nil {
		return dtype.Field{}, err
	}
	return dtype.Field{Name: a.Name(), Type: out, Nullable: !counted}, nil
}

// aggType is what one aggregation of a column of type dt comes out as.
//
// Every answer here is the kernel's own, so an aggregation this accepts is one
// the kernels will run and the type it promises is the type that arrives. The
// arguments are checked with it, since a quantile at two is a mistake that can
// be found without reading anything.
func aggType(a Agg, dt dtype.DataType) (dtype.DataType, error) {
	switch a.Func {
	case AggSum:
		return kernel.SumType(dt)
	case AggMean:
		return kernel.MeanType(dt)
	case AggMin, AggMax:
		return kernel.MinMaxType(dt)
	case AggCount, AggSize:
		return dtype.Int64, nil
	case AggFirst, AggLast:
		return dt, nil
	case AggVar:
		if err := checkDDoF(a); err != nil {
			return nil, err
		}
		return kernel.VarType(dt)
	case AggStd:
		if err := checkDDoF(a); err != nil {
			return nil, err
		}
		return kernel.StdType(dt)
	case AggMedian:
		return kernel.MedianType(dt)
	case AggQuantile:
		if math.IsNaN(a.Q) || a.Q < 0 || a.Q > 1 {
			return nil, fmt.Errorf("kuma: the column %s is a quantile at %v, which is not between 0 and 1", a.Name(), a.Q)
		}
		if a.How < kernel.Linear || a.How > kernel.Midpoint {
			return nil, fmt.Errorf("kuma: the column %s is a quantile with %s, which is no interpolation there is",
				a.Name(), a.How)
		}
		return kernel.QuantileType(dt)
	default:
		return kernel.NUniqueType(dt)
	}
}

// checkDDoF turns away a divisor adjustment that would divide by a negative
// number of values.
func checkDDoF(a Agg) error {
	if a.DDoF < 0 {
		return fmt.Errorf("kuma: the column %s has a ddof of %d, which would take more values off the divisor than there are values",
			a.Name(), a.DDoF)
	}
	return nil
}

// bothSides returns the name of a join key that is the same column name on both
// sides, which is the case where the result has one column for the pair.
func bothSides(k JoinKey) (string, bool) {
	if k.Left.Kind() != KindColumn || k.Right.Kind() != KindColumn {
		return "", false
	}
	if k.Left.Name() != k.Right.Name() {
		return "", false
	}
	return k.Left.Name(), true
}

// noDuplicates is the schema of the fields, or an error naming the first one
// that turns up twice.
//
// A frame is read by name, so two columns of one name is a frame with a column
// in it that nobody can get at. The check is here rather than in the frame
// because the point of a plan is to say so before the reading starts.
func noDuplicates(fields []dtype.Field) (dtype.Schema, error) {
	seen := make(map[string]struct{}, len(fields))
	for _, f := range fields {
		if _, ok := seen[f.Name]; ok {
			return dtype.Schema{}, fmt.Errorf("kuma: two columns are called %q: %w", f.Name, ErrDuplicateColumn)
		}
		seen[f.Name] = struct{}{}
	}
	return dtype.Schema{Fields: fields}, nil
}

// inOp names the operator an error came out of, when the error has somewhere to
// say it. A name that is not there is the most common mistake in a plan, and
// which step asked for it is half of what a reader needs to fix it.
func inOp(op Op, err error) error {
	var ce *ColumnError
	if !errors.As(err, &ce) || ce.Op != "" {
		return err
	}

	// A new error rather than a field written into the one that arrived, since
	// the same expression is checked against several schemas by the passes and
	// an error is not the place to keep which of them was last.
	return &ColumnError{Op: op.String(), Name: ce.Name, Have: ce.Have}
}
