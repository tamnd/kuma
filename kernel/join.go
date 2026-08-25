package kernel

import (
	"fmt"

	"github.com/tamnd/kuma/array"
)

// JoinType is which rows of the two sides a join keeps.
type JoinType int

// The seven joins. They are the seven SQL has, and they mean the same things
// here.
const (
	// InnerJoin keeps the pairs that matched and nothing else.
	InnerJoin JoinType = iota

	// LeftJoin keeps every left row, with the right side missing where nothing
	// matched. RightJoin is the same thing the other way round.
	LeftJoin
	RightJoin

	// OuterJoin keeps every row of both sides. It is what SQL calls a full
	// outer join.
	OuterJoin

	// SemiJoin keeps the left rows that matched, once each, and takes nothing
	// from the right side. It is the join for "which of these have one", and it
	// is an EXISTS in SQL.
	SemiJoin

	// AntiJoin keeps the left rows that matched nothing, which is a NOT EXISTS.
	AntiJoin

	// CrossJoin pairs every left row with every right row and looks at no keys
	// at all.
	CrossJoin
)

// String returns the name of the join, as it would be written in a query.
func (j JoinType) String() string {
	switch j {
	case InnerJoin:
		return "inner"
	case LeftJoin:
		return "left"
	case RightJoin:
		return "right"
	case OuterJoin:
		return "outer"
	case SemiJoin:
		return "semi"
	case AntiJoin:
		return "anti"
	case CrossJoin:
		return "cross"
	default:
		return fmt.Sprintf("JoinType(%d)", int(j))
	}
}

// Pairs is which rows of the two sides a join put together.
//
// A join returns positions rather than a joined table for the same reason a
// sort does: the result is one set of positions applied to every column of each
// side, and building the table is [Take]'s job. It also means a caller who
// wants to know what matched, and not the table, does not pay to build one.
//
// A position below zero means nothing matched, which is exactly what Take turns
// into a null, so an outer join needs no special handling anywhere downstream.
type Pairs struct {
	// Left holds the left row of every output row, and Right the right row.
	// They are the same length except after a semi or an anti join, which take
	// nothing from the right side and leave Right nil.
	Left  []int
	Right []int
}

// Len returns how many rows the join produced.
func (p Pairs) Len() int { return len(p.Left) }

// Side is one side of a join: how many rows it has and what to match on.
//
// The row count is given rather than taken from the key columns because a cross
// join has no key columns and still has rows. For every other join the count has
// to agree with what the keys say, and it is checked.
type Side struct {
	Rows int
	Keys []*array.Chunked
}

// Join works out which rows of the two sides go together.
//
// Rows match when they agree on every key, using the same encoding [GroupBy]
// uses, so an int8 and an int64 holding the same number match and every NaN
// matches every other NaN.
//
// A missing key matches nothing, including another missing key. That is what
// SQL says and what Polars does, and it is the answer that keeps a join from
// gluing together every row whose field was left blank. It is not what pandas
// does, where merging on a column with NaN in it happily pairs the blanks up.
//
// Output order is the left side's row order, with the matches of one left row
// in the right side's row order, so the result is deterministic and reads the
// way the input did. A right join is ordered by the right side. The rows an
// outer join adds for the unmatched right rows come at the end, in right side
// order, which is where SQL puts them too.
//
// It panics if a side's row count disagrees with its key columns, if a key
// column is nil, or if the two sides have different numbers of keys, all of
// which are mistakes in the program rather than in the data. It returns an
// error for a key column of a type there is no encoding for, since that type
// usually comes from data, and for a keyed join with no keys or a cross join
// with some.
func Join(left, right Side, how JoinType) (Pairs, error) {
	checkSide("left", left)
	checkSide("right", right)

	if how == CrossJoin {
		if len(left.Keys) > 0 || len(right.Keys) > 0 {
			return Pairs{}, fmt.Errorf("kernel: a cross join has no keys, got %d and %d",
				len(left.Keys), len(right.Keys))
		}
		return cross(left.Rows, right.Rows), nil
	}
	if len(left.Keys) == 0 || len(right.Keys) == 0 {
		return Pairs{}, fmt.Errorf("kernel: a %s join needs keys, got %d and %d",
			how, len(left.Keys), len(right.Keys))
	}
	if len(left.Keys) != len(right.Keys) {
		panic(fmt.Sprintf("kernel: join on %d left keys and %d right keys",
			len(left.Keys), len(right.Keys)))
	}

	// A right join is a left join with the sides swapped, which is the whole
	// implementation of it. Doing it any other way means two copies of one loop
	// that have to stay in step.
	if how == RightJoin {
		p, err := join(right, left, LeftJoin)
		if err != nil {
			return Pairs{}, err
		}
		p.Left, p.Right = p.Right, p.Left
		return p, nil
	}
	return join(left, right, how)
}

// checkSide panics if a side does not describe itself consistently.
func checkSide(name string, s Side) {
	if s.Rows < 0 {
		panic(fmt.Sprintf("kernel: the %s side of a join has %d rows", name, s.Rows))
	}
	for i, k := range s.Keys {
		if k == nil {
			panic(fmt.Sprintf("kernel: %s join key %d is a nil column", name, i))
		}
		if k.Len() != s.Rows {
			panic(fmt.Sprintf("kernel: %s join key %d has %d values over %d rows",
				name, i, k.Len(), s.Rows))
		}
	}
}

// cross pairs every one of the left rows with every one of the right rows.
//
// This is the one join that can turn two small tables into a very large one,
// which is why it is a join a caller has to name rather than one a forgotten
// key falls into.
func cross(l, r int) Pairs {
	p := Pairs{
		Left:  make([]int, 0, l*r),
		Right: make([]int, 0, l*r),
	}
	for i := range l {
		for j := range r {
			p.Left = append(p.Left, i)
			p.Right = append(p.Right, j)
		}
	}
	return p
}

// join is the hash join, with the table always built on the right side.
//
// Which side to build on is a real choice and the answer here is the simple
// one. Building on the smaller side is the usual advice, and it is the right
// advice when the output order does not matter. It does matter here: the order
// is the left side's, and swapping the build side means either sorting the
// output afterwards or probing in an order that is not the one the caller
// asked for. RightJoin gets its swap by swapping the arguments in Join, where
// the ordering promise is swapped along with them.
func join(left, right Side, how JoinType) (Pairs, error) {
	table, err := buildTable(right)
	if err != nil {
		return Pairs{}, err
	}

	probe, err := newKeys(left.Keys)
	if err != nil {
		return Pairs{}, err
	}

	// matched[j] records whether right row j found anything, which is what an
	// outer join needs at the end and what nothing else needs at all.
	var matched []bool
	if how == OuterJoin {
		matched = make([]bool, right.Rows)
	}

	// Semi and anti take one row per left row at most and nothing from the
	// right, so they get the short loop and a nil Right.
	if how == SemiJoin || how == AntiJoin {
		want := how == SemiJoin
		p := Pairs{}
		var scratch []byte
		for i := range left.Rows {
			scratch = probe.appendRow(scratch[:0], i)

			// A missing key matches nothing, so it is kept by an anti join and
			// dropped by a semi join without the table being asked.
			hit := false
			if !probe.missing {
				_, hit = table[string(scratch)]
			}
			if hit == want {
				p.Left = append(p.Left, i)
			}
		}
		return p, nil
	}

	p := Pairs{}
	var scratch []byte
	for i := range left.Rows {
		scratch = probe.appendRow(scratch[:0], i)

		var rows []int
		if !probe.missing {
			rows = table[string(scratch)]
		}
		if len(rows) == 0 {
			if how == LeftJoin || how == OuterJoin {
				p.Left = append(p.Left, i)
				p.Right = append(p.Right, -1)
			}
			continue
		}
		for _, j := range rows {
			p.Left = append(p.Left, i)
			p.Right = append(p.Right, j)
			if matched != nil {
				matched[j] = true
			}
		}
	}

	// The right rows nothing matched, in right side order, which is where SQL
	// puts them.
	for j, hit := range matched {
		if !hit {
			p.Left = append(p.Left, -1)
			p.Right = append(p.Right, j)
		}
	}
	return p, nil
}

// buildTable indexes the rows of a side by their key bytes.
//
// The rows of one key come out in the order they went in, which is what makes
// the output order a promise rather than whatever the map felt like.
func buildTable(s Side) (map[string][]int, error) {
	ks, err := newKeys(s.Keys)
	if err != nil {
		return nil, err
	}

	table := make(map[string][]int)
	var scratch []byte
	for i := range s.Rows {
		scratch = ks.appendRow(scratch[:0], i)
		if ks.missing {
			// A missing key matches nothing, so it is not worth a slot.
			continue
		}
		table[string(scratch)] = append(table[string(scratch)], i)
	}
	return table, nil
}

// keys is the several key columns of one side of a join, encoded together.
//
// It is the same encoder [GroupBy] uses, with one thing added: whether the row
// just written had a missing value in it, which grouping does not care about
// and joining decides everything on.
type keys struct {
	cols []*key

	// missing is whether the last row written had a missing key.
	missing bool
}

// newKeys returns an encoder over the given key columns.
func newKeys(cols []*array.Chunked) (*keys, error) {
	ks := &keys{cols: make([]*key, len(cols))}
	for i, c := range cols {
		k, err := newKey(c)
		if err != nil {
			return nil, err
		}
		ks.cols[i] = k
	}
	return ks, nil
}

// appendRow writes row i of every key column onto dst and records whether any
// of them was missing.
func (ks *keys) appendRow(dst []byte, i int) []byte {
	ks.missing = false
	for _, k := range ks.cols {
		before := len(dst)
		dst = k.appendRow(dst, i)
		if len(dst) == before+1 && dst[before] == 0 {
			ks.missing = true
		}
	}
	return dst
}
