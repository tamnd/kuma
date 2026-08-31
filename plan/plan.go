package plan

import (
	"slices"
	"strconv"
	"strings"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// Op is what one operator of a plan does.
type Op uint8

// The operators. They are the ones the eager frame already has, and the list
// grows as the engine does: a union, a pivot, a window and a sink are all
// operators here eventually.
const (
	OpScan      Op = iota // read a source
	OpFilter              // keep the rows a condition holds for
	OpProject             // work out a new set of columns
	OpAggregate           // work something out about each group
	OpJoin                // pair the rows of two plans by their keys
	OpSort                // put the rows in order
	OpLimit               // keep a run of rows
	OpDistinct            // keep one row of each set of equal ones
	OpExplode             // turn each element of a list column into a row
)

// String returns the name of the operator, which is the word an explain uses.
func (o Op) String() string {
	switch o {
	case OpScan:
		return "Scan"
	case OpFilter:
		return "Filter"
	case OpProject:
		return "Project"
	case OpAggregate:
		return "Aggregate"
	case OpJoin:
		return "Join"
	case OpSort:
		return "Sort"
	case OpLimit:
		return "Limit"
	case OpDistinct:
		return "Distinct"
	default:
		return "Explode"
	}
}

// Source is where a scan reads from, which is a set of files, a frame that is
// already in memory, or anything else that can say what columns it has before
// it is read.
//
// The interface is here and the implementations are not, because what a source
// is made of is nothing the plan needs to know. A plan holds one so that a pass
// can ask what columns exist and so that an explain can say where the data came
// from.
type Source interface {
	// Name is what to call the source in an explain, such as the pattern the
	// files were found by or the word frame for one already in memory.
	Name() string

	// Schema is the columns the source has, worked out without reading any of
	// the data. It is asked once when the plan is checked and the answer is
	// what every name in the plan is resolved against.
	Schema() (dtype.Schema, error)
}

// Projection is one column a projection produces: what to work out, and what to
// call it.
type Projection struct {
	// Expr is the value of the column in each row.
	Expr *Expr

	// As is what to call it. When it is empty the column is called after the
	// expression, so a projection of price is called price and one of
	// (price * qty) is called that.
	As string
}

// Name returns what the column will be called.
func (p Projection) Name() string {
	if p.As != "" {
		return p.As
	}
	return p.Expr.String()
}

// String returns the projected column as it would be written.
func (p Projection) String() string {
	if p.As == "" {
		return p.Expr.String()
	}
	return p.Expr.String() + " as " + p.As
}

// SortKey is one column of an ordering: what to sort by, and how.
//
// The zero direction is ascending with the nulls at the end, which is what the
// eager [github.com/tamnd/kuma.Frame.Sort] does and what most databases do.
// Null placement is not part of the direction, so asking for descending order
// does not move the nulls.
type SortKey struct {
	Expr       *Expr
	Descending bool
	NullsFirst bool
}

// String returns the sort key as it would be written.
func (k SortKey) String() string {
	s := k.Expr.String()
	if k.Descending {
		s += " desc"
	}
	if k.NullsFirst {
		s += " nulls first"
	}
	return s
}

// JoinKey is one pair of values two plans are joined on. The common case is the
// same column name on both sides, and the case this shape exists for is the
// same thing called two different things, which is most real data.
type JoinKey struct {
	Left  *Expr
	Right *Expr
}

// String returns the join key as it would be written.
func (k JoinKey) String() string {
	if k.Left == k.Right {
		return k.Left.String()
	}
	return k.Left.String() + " = " + k.Right.String()
}

// Node is one operator of a logical plan, and by way of its inputs the whole of
// one.
//
// It is one struct with an operator tag rather than a type per operator, for
// the reason [Expr] is: the set is closed, the tree is small, and a pass over it
// reads better as a switch than as a family of types with a method each.
//
// Which fields mean anything depends on the operator, so a node is built by one
// of the constructors below rather than written out field by field. That is
// also what keeps a node from changing after it is built, which every pass
// depends on: a pass that wants a plan with one operator different builds the
// new operator and reuses the rest of the tree, and the plan it started from is
// still there afterwards.
//
// Unlike an [Expr], a node is not shared between equal plans. There is nothing
// to gain from it. A plan is built once per query and walked by a handful of
// passes, so the lookup would cost more than the tree it saved.
type Node struct {
	op Op

	// l is the input of a one input operator and the left input of a join, and
	// r is the right input of a join. Both are nil for a scan.
	l, r *Node

	src    Source       // OpScan
	pred   *Expr        // OpFilter
	cols   []Projection // OpProject
	by     []*Expr      // OpAggregate, the group keys
	aggs   []Agg        // OpAggregate
	sort   []SortKey    // OpSort
	on     []JoinKey    // OpJoin
	how    kernel.JoinType
	off, n int64    // OpLimit
	names  []string // OpExplode
	read   []string // OpScan, the columns to read, and nil for all of them
}

// The constructors. Each one takes what its operator needs and nothing else,
// and each one copies the slices it is given, so that a plan cannot change
// under whoever is walking it.

// Scan reads a source. It is the leaf every plan starts from.
func Scan(src Source) *Node {
	return &Node{op: OpScan, src: src}
}

// ScanOnly reads the named columns of a source and leaves the rest of them
// unread.
//
// It is what a projection pushdown builds, and it is the pass that pays for the
// whole optimizer: a query over forty columns that reads two of them reads two
// of them out of the file rather than reading forty and dropping thirty eight.
// The columns come out in the order the source has them, since a scan is a read
// and putting the columns in an order is what a projection is for.
//
// A name that is not a column of the source is an error when the plan is
// checked, the same as a name written anywhere else.
func ScanOnly(src Source, names []string) *Node {
	return &Node{op: OpScan, src: src, read: slices.Clone(names)}
}

// Filter keeps the rows the condition holds for. A row where the condition is
// missing rather than false is not kept, since a row is kept when the answer is
// true and missing is not true.
func Filter(in *Node, cond *Expr) *Node {
	return &Node{op: OpFilter, l: in, pred: cond}
}

// Project works out a new set of columns from the ones the input has. It is
// what a select is, and what adding a column is, and what renaming one is.
func Project(in *Node, cols []Projection) *Node {
	return &Node{op: OpProject, l: in, cols: slices.Clone(cols)}
}

// Aggregate works out something about each group of rows that the keys agree
// on. With no keys it works the aggregations out over the whole input, which is
// what a frame with no group by does.
func Aggregate(in *Node, by []*Expr, aggs []Agg) *Node {
	return &Node{op: OpAggregate, l: in, by: slices.Clone(by), aggs: slices.Clone(aggs)}
}

// Join pairs the rows of two plans by their keys. A cross join takes no keys
// and pairs every row with every row.
func Join(left, right *Node, on []JoinKey, how kernel.JoinType) *Node {
	return &Node{op: OpJoin, l: left, r: right, on: slices.Clone(on), how: how}
}

// Sort puts the rows in order. The first key decides and each later one breaks
// the ties of the one before, and the sort is stable, so rows that every key
// calls equal come out in the order they went in.
func Sort(in *Node, keys []SortKey) *Node {
	return &Node{op: OpSort, l: in, sort: slices.Clone(keys)}
}

// Limit keeps at most n rows, having skipped the first off of them. It is what
// a head is, with an offset of zero, and it is the operator a slice pushdown
// pass sinks into a scan so that a file is only read as far as it is needed.
func Limit(in *Node, off, n int64) *Node {
	return &Node{op: OpLimit, l: in, off: off, n: n}
}

// Distinct keeps the first row of each set of rows that the given expressions
// agree on, and with none given it looks at every column. It is what pandas
// calls drop_duplicates.
func Distinct(in *Node, by []*Expr) *Node {
	return &Node{op: OpDistinct, l: in, by: slices.Clone(by)}
}

// Explode turns each element of the named list columns into a row of its own,
// repeating the other columns of the row it came from. It is what pandas calls
// explode and what SQL writes as an unnest.
//
// The columns stay where they are and under the names they had, holding the
// element type rather than a list of it, which is why this takes names rather
// than expressions: it changes columns in place and there is no place to put
// the result of an expression that is not one of them. Naming several takes
// them apart together, and then every row has to hold the same number of
// elements in all of them.
func Explode(in *Node, names []string) *Node {
	return &Node{op: OpExplode, l: in, names: slices.Clone(names)}
}

// The accessors. The slices they return are the node's own, so read them and do
// not write to them. A pass that wants something different builds a new node,
// which is cheap, rather than changing a node another plan may be holding.

// Op is what this operator does, and which of the other accessors mean
// anything.
func (n *Node) Op() Op { return n.op }

// Input is what this operator reads, which is the left input of a join and
// nothing at all for a scan.
func (n *Node) Input() *Node { return n.l }

// Right is the other input of a join, and nil for every other operator.
func (n *Node) Right() *Node { return n.r }

// Source is where a scan reads from, and nil for every other operator.
func (n *Node) Source() Source { return n.src }

// ScanColumns is the columns a scan reads, and nil for a scan that reads every
// column its source has and for every other operator.
func (n *Node) ScanColumns() []string { return n.read }

// Cond is the condition a filter keeps the rows by, and nil for every other
// operator.
func (n *Node) Cond() *Expr { return n.pred }

// Columns is what a projection works out, and nil for every other operator.
func (n *Node) Columns() []Projection { return n.cols }

// By is what an aggregate groups by, which is empty for an aggregate over the
// whole input, and what a distinct compares by, which is empty when it compares
// every column.
func (n *Node) By() []*Expr { return n.by }

// Aggs is what an aggregate works out about each group, and nil for every other
// operator.
func (n *Node) Aggs() []Agg { return n.aggs }

// SortKeys is the ordering a sort puts the rows in, and nil for every other
// operator.
func (n *Node) SortKeys() []SortKey { return n.sort }

// JoinKeys is what a join pairs the rows by, which is empty for a cross join,
// and nil for every other operator.
func (n *Node) JoinKeys() []JoinKey { return n.on }

// JoinType is which rows of the two inputs a join keeps.
func (n *Node) JoinType() kernel.JoinType { return n.how }

// SharedKeys returns the names of the join keys the result holds one column
// for rather than two, which are the keys that are a column of the same name on
// both sides.
//
// Two such columns hold the same values wherever the rows matched, so keeping
// both would be a name clash over a pair of columns that agree. It is here so
// that what [Node.Schema] promises and what an engine builds are decided by the
// same rule.
//
// It returns nothing for a node that is not a join, for a cross join, which has
// no keys, and for a semi or an anti join, which take no columns from the right
// side at all.
func (n *Node) SharedKeys() []string {
	if n.op != OpJoin || n.how == kernel.SemiJoin || n.how == kernel.AntiJoin {
		return nil
	}

	var names []string
	for _, k := range n.on {
		if k.Left == nil || k.Right == nil {
			continue
		}
		if name, ok := bothSides(k); ok {
			names = append(names, name)
		}
	}
	return names
}

// ExplodeNames is the columns an explode takes apart, and nil for every other
// operator.
func (n *Node) ExplodeNames() []string { return n.names }

// Offset is how many rows a limit skips before it starts keeping them.
func (n *Node) Offset() int64 { return n.off }

// Limit is how many rows a limit keeps, having skipped [Node.Offset] of them.
func (n *Node) Limit() int64 { return n.n }

// String returns this one operator as an explain would show it, without its
// inputs. It is one line, and it is the line the tree of a whole plan is built
// out of.
func (n *Node) String() string {
	var sb strings.Builder
	sb.WriteString(n.op.String())
	switch n.op {
	case OpScan:
		sb.WriteByte(' ')
		sb.WriteString(n.src.Name())
		if len(n.read) > 0 {
			sb.WriteString(" [")
			sb.WriteString(strings.Join(n.read, ", "))
			sb.WriteByte(']')
		}
	case OpFilter:
		sb.WriteByte(' ')
		sb.WriteString(n.pred.String())
	case OpProject:
		sb.WriteByte(' ')
		writeList(&sb, n.cols)
	case OpAggregate:
		writeBy(&sb, n.by)
		sb.WriteString(": ")
		writeList(&sb, n.aggs)
	case OpJoin:
		sb.WriteByte(' ')
		sb.WriteString(n.how.String())
		if len(n.on) > 0 {
			sb.WriteString(" on ")
			writeList(&sb, n.on)
		}
	case OpSort:
		writeBy(&sb, n.sort)
	case OpLimit:
		sb.WriteByte(' ')
		sb.WriteString(strconv.FormatInt(n.n, 10))
		if n.off != 0 {
			sb.WriteString(" offset ")
			sb.WriteString(strconv.FormatInt(n.off, 10))
		}
	case OpDistinct:
		writeBy(&sb, n.by)
	default:
		sb.WriteByte(' ')
		sb.WriteString(strings.Join(n.names, ", "))
	}
	return sb.String()
}

// writeBy writes the keys an operator works by, and nothing at all when there
// are none, since an aggregate over the whole input and a distinct over every
// column are both written without a by.
func writeBy[T interface{ String() string }](sb *strings.Builder, items []T) {
	if len(items) == 0 {
		return
	}
	sb.WriteString(" by ")
	writeList(sb, items)
}

// writeList writes the parts of an operator the way a caller wrote them, which
// is comma separated and in the order they were given.
func writeList[T interface{ String() string }](sb *strings.Builder, items []T) {
	for i, item := range items {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(item.String())
	}
}
