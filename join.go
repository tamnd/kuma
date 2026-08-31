package kuma

import (
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/kernel"
)

// JoinType is which rows of the two frames a join keeps.
//
// It is [kernel.JoinType] under another name, so the constants below are the
// same constants and either name may be used.
type JoinType = kernel.JoinType

// The seven joins, which are the seven SQL has and mean the same things here.
const (
	// InnerJoin keeps the pairs that matched and nothing else.
	InnerJoin = kernel.InnerJoin

	// LeftJoin keeps every left row, with the right columns missing where
	// nothing matched, and RightJoin is the same thing the other way round.
	LeftJoin  = kernel.LeftJoin
	RightJoin = kernel.RightJoin

	// OuterJoin keeps every row of both frames. It is what SQL calls a full
	// outer join.
	OuterJoin = kernel.OuterJoin

	// SemiJoin keeps the left rows that matched, once each, and takes no
	// columns from the right frame. It is an EXISTS.
	SemiJoin = kernel.SemiJoin

	// AntiJoin keeps the left rows that matched nothing, which is a NOT EXISTS.
	AntiJoin = kernel.AntiJoin

	// CrossJoin pairs every left row with every right row and looks at no keys.
	CrossJoin = kernel.CrossJoin
)

// On names the columns two frames are joined on.
//
// The common case is one name on both sides, which is what [Using] is for. This
// is for the case where the same thing is called two different things, which is
// most real data.
type On struct {
	Left  string
	Right string
}

// Using returns the join keys for columns that are called the same thing on
// both sides.
func Using(names ...string) []On {
	on := make([]On, len(names))
	for i, name := range names {
		on[i] = On{Left: name, Right: name}
	}
	return on
}

// Join returns the rows of two frames put together where their keys match.
//
// Rows match when they agree on every key in on. A missing key matches nothing,
// including another missing key, which is what SQL says and what keeps a join
// from gluing together every row whose field was left blank. It is not what
// pandas does.
//
// The result has every column of the left frame followed by every column of the
// right frame, except that the right key columns are dropped when they are
// called the same thing as the left ones, since they hold the same values and
// nobody wants both. A semi or an anti join takes no columns from the right
// frame at all. Any other name that appears on both sides is an error, because
// silently renaming one of them is how pandas ends up with columns called
// price_x and price_y.
//
// The rows come out in the left frame's order, with the matches of one left row
// in the right frame's order. A right join is ordered by the right frame, and
// the rows an outer join adds for unmatched right rows come at the end.
//
// It reports an error if a key names a column that is not there, if the two
// sides have different numbers of keys, if a key column is of a type there is
// no encoding for, or if the result would have two columns of one name.
func (f *Frame[S]) Join(other *Frame[Dynamic], on []On, how JoinType) (*Frame[Dynamic], error) {
	if other == nil {
		return nil, fmt.Errorf("kuma: Join with no frame to join to: %w", ErrNoValues)
	}

	left, right, err := joinKeys(f, other, on, how)
	if err != nil {
		return nil, err
	}

	p, err := kernel.Join(left, right, how)
	if err != nil {
		return nil, err
	}
	return joinFrame(f, other, sharedKeys(on, how), how, p)
}

// sharedKeys returns the key names the result holds one column for rather than
// two, which are the ones called the same thing on both sides. It is the rule
// the plan states in SharedKeys, over names rather than over expressions.
func sharedKeys(on []On, how JoinType) map[string]bool {
	if how == SemiJoin || how == AntiJoin {
		return nil
	}

	shared := make(map[string]bool, len(on))
	for _, o := range on {
		if o.Left == o.Right {
			shared[o.Left] = true
		}
	}
	return shared
}

// InnerJoin returns the rows of two frames put together on the named columns,
// keeping only the pairs that matched. It is [Frame.Join] for the common case.
func (f *Frame[S]) InnerJoin(other *Frame[Dynamic], names ...string) (*Frame[Dynamic], error) {
	return f.Join(other, Using(names...), InnerJoin)
}

// LeftJoin returns the rows of two frames put together on the named columns,
// keeping every left row and filling the right columns with nulls where nothing
// matched.
func (f *Frame[S]) LeftJoin(other *Frame[Dynamic], names ...string) (*Frame[Dynamic], error) {
	return f.Join(other, Using(names...), LeftJoin)
}

// CrossJoin returns every left row paired with every right row.
//
// The result has as many rows as the two frames multiplied together, which is
// why it is a method a caller has to name rather than something a forgotten key
// falls into.
func (f *Frame[S]) CrossJoin(other *Frame[Dynamic]) (*Frame[Dynamic], error) {
	return f.Join(other, nil, CrossJoin)
}

// joinKeys turns the names into the two sides the kernel wants.
func joinKeys[S any](f *Frame[S], other *Frame[Dynamic], on []On, how JoinType) (
	left, right kernel.Side, err error) {
	left = kernel.Side{Rows: f.NumRows()}
	right = kernel.Side{Rows: other.NumRows()}
	if how == CrossJoin {
		if len(on) > 0 {
			return left, right, fmt.Errorf("kuma: a cross join has no keys, got %d: %w",
				len(on), ErrLength)
		}
		return left, right, nil
	}
	if len(on) == 0 {
		return left, right, fmt.Errorf("kuma: a %s join needs keys: %w", how, ErrLength)
	}

	left.Keys = make([]*array.Chunked, len(on))
	right.Keys = make([]*array.Chunked, len(on))
	for i, o := range on {
		k, ok := f.index[o.Left]
		if !ok {
			return left, right, noColumn("Join", o.Left, f.Names())
		}
		j, ok := other.index[o.Right]
		if !ok {
			return left, right, noColumn("Join", o.Right, other.Names())
		}
		left.Keys[i] = f.cols[k].data
		right.Keys[i] = other.cols[j].data
	}
	return left, right, nil
}

// joinFrame gathers both sides at the positions the join worked out and puts
// the columns together.
//
// The shared names are the keys the result holds one column for rather than
// two, which is what [sharedKeys] works out for the eager path and the plan
// works out for the engine. The right key columns hold the same values as the
// left ones wherever the rows matched, so keeping both would be two identical
// columns and a name clash.
func joinFrame[S any](f *Frame[S], other *Frame[Dynamic], shared map[string]bool, how JoinType,
	p kernel.Pairs) (*Frame[Dynamic], error) {
	cols := make([]Column, 0, f.NumCols()+other.NumCols())
	for _, c := range f.cols {
		if !shared[c.name] {
			cols = append(cols, Column{name: c.name, data: kernel.Take(c.data, p.Left)})
			continue
		}
		key, err := joinKeyColumn(c, other.cols[other.index[c.name]], p)
		if err != nil {
			return nil, err
		}
		cols = append(cols, key)
	}

	// A semi or an anti join answers a question about the left frame, so the
	// right frame contributes nothing but the question.
	if how == SemiJoin || how == AntiJoin {
		return NewFrame(cols...)
	}

	for _, c := range other.cols {
		if shared[c.name] {
			continue
		}
		cols = append(cols, Column{name: c.name, data: kernel.Take(c.data, p.Right)})
	}
	return NewFrame(cols...)
}

// joinKeyColumn returns the one column that stands for a pair of keys called
// the same thing on both sides.
//
// The obvious answer is the left column gathered at the left positions, and it
// is wrong for a right join and for an outer join, where a row that came from
// the right side has no left position and would get a null for a key it plainly
// has. The answer that is right for all seven joins is a gather from the two
// key columns one after the other, taking the left value when there is one and
// the right value when there is not. Chaining two chunked columns is appending
// one chunk list to the other and copies nothing, so this costs one index slice.
func joinKeyColumn(left, right Column, p kernel.Pairs) (Column, error) {
	if left.DType() != right.DType() {
		// Take builds one column of one type, so two key columns that are
		// stored differently cannot share an output column. Joining them is
		// fine, since the key encoding widens; naming the result is not.
		return Column{}, fmt.Errorf(
			"kuma: the join key %q is %s on the left and %s on the right, "+
				"so rename one of them: %w",
			left.Name(), left.DType(), right.DType(), ErrWrongType)
	}

	chunks := left.data.Chunks()
	both := make([]*array.Array, 0, len(chunks)+len(right.data.Chunks()))
	both = append(both, chunks...)
	both = append(both, right.data.Chunks()...)

	chained, err := array.NewChunked(left.DType(), both...)
	if err != nil {
		return Column{}, fmt.Errorf("kuma: %w", err)
	}

	n := left.Len()
	idx := make([]int, p.Len())
	for i, l := range p.Left {
		if l >= 0 {
			idx[i] = l
			continue
		}
		idx[i] = n + p.Right[i]
	}
	return Column{name: left.Name(), data: kernel.Take(chained, idx)}, nil
}
