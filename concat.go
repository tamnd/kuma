package kuma

import (
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Concat stacks frames on top of each other, so the result has the rows of the
// first, then the rows of the second, and so on.
//
// Every frame has to hold the same columns, of the same types. The order they
// are in does not have to match, and the first frame's order is the one the
// result comes out in, because a frame read from one file and a frame read from
// another are the same table whether or not the writer put the columns in the
// same order. A column that is in one frame and not another is an error rather
// than a column of nulls, since that is much more often a mistake than an
// intention. [ConcatUnion] is the version that fills.
//
// Nothing is copied. A column is stored as a list of chunks, so stacking two
// frames is appending one list to the other, and the values stay where they
// are.
//
// The frames all have the same schema type and the result keeps it, so
// concatenating typed frames gives back a typed frame.
func Concat[S any](frames ...*Frame[S]) (*Frame[S], error) {
	if len(frames) == 0 {
		return nil, fmt.Errorf("kuma: Concat with no frames: %w", ErrNoValues)
	}
	if err := checkFrames("Concat", frames); err != nil {
		return nil, err
	}
	if len(frames) == 1 {
		// A frame never changes, so there is nothing to copy and nothing to
		// build.
		return frames[0], nil
	}

	// A frame that has all of frame 0's columns and a column of its own has as
	// many columns as frame 0 does not, so counting first is what catches the
	// extra column, and the loop below catches the missing one.
	first := frames[0]
	for j, f := range frames[1:] {
		if f.NumCols() != first.NumCols() {
			return nil, fmt.Errorf(
				"kuma: Concat: frame %d has %d columns and frame 0 has %d, "+
					"so one of them has a column the other does not: %w",
				j+1, f.NumCols(), first.NumCols(), ErrNoColumn)
		}
	}

	cols := make([]Column, first.NumCols())
	for i, c := range first.cols {
		parts := make([]*array.Chunked, len(frames))
		parts[0] = c.data

		for j, f := range frames[1:] {
			k, ok := f.index[c.name]
			if !ok {
				return nil, fmt.Errorf(
					"kuma: Concat: frame %d has no column %q, which frame 0 has: %w",
					j+1, c.name, ErrNoColumn)
			}
			other := f.cols[k]
			if other.DType() != c.DType() {
				return nil, fmt.Errorf(
					"kuma: Concat: column %q is %s in frame 0 and %s in frame %d: %w",
					c.name, c.DType(), other.DType(), j+1, ErrWrongType)
			}
			parts[j+1] = other.data
		}

		stacked, err := chain(c.DType(), parts)
		if err != nil {
			return nil, err
		}
		cols[i] = Column{name: c.name, data: stacked}
	}

	return newFrame[S](cols)
}

// ConcatUnion is [Concat] over frames that do not hold the same columns.
//
// The result has every column that any of the frames has, in the order they
// first appear, and a frame that does not have one contributes nulls for it. It
// is what pandas concat does by default and what Polars calls a diagonal
// concat.
//
// The result is a dynamic frame whatever went in, since its schema is not the
// schema of any of the frames.
func ConcatUnion(frames ...*Frame[Dynamic]) (*Frame[Dynamic], error) {
	if len(frames) == 0 {
		return nil, fmt.Errorf("kuma: ConcatUnion with no frames: %w", ErrNoValues)
	}
	if err := checkFrames("ConcatUnion", frames); err != nil {
		return nil, err
	}

	// The column names in the order they first appear, with the type the frame
	// that first had one gave it.
	var names []string
	types := make(map[string]dtype.DataType)
	for j, f := range frames {
		for _, c := range f.cols {
			dt, seen := types[c.name]
			if !seen {
				names = append(names, c.name)
				types[c.name] = c.DType()
				continue
			}
			if dt != c.DType() {
				return nil, fmt.Errorf(
					"kuma: ConcatUnion: column %q is %s in an earlier frame and %s in frame %d: %w",
					c.name, dt, c.DType(), j, ErrWrongType)
			}
		}
	}

	cols := make([]Column, len(names))
	for i, name := range names {
		dt := types[name]
		parts := make([]*array.Chunked, len(frames))
		for j, f := range frames {
			k, ok := f.index[name]
			if ok {
				parts[j] = f.cols[k].data
				continue
			}
			missing, err := nulls(dt, f.NumRows())
			if err != nil {
				return nil, err
			}
			parts[j] = missing
		}

		stacked, err := chain(dt, parts)
		if err != nil {
			return nil, err
		}
		cols[i] = Column{name: name, data: stacked}
	}
	return NewFrame(cols...)
}

// HStack puts frames side by side, so the result has the columns of the first,
// then the columns of the second, and so on.
//
// Every frame has to have the same number of rows, since row 3 of the result is
// row 3 of each of them, and no two of them may have a column of the same name.
// Rename one first, or use a join if the rows should be matched up by a key
// rather than by where they happen to be.
//
// Nothing is copied. The result holds the same columns the frames do.
func HStack(frames ...*Frame[Dynamic]) (*Frame[Dynamic], error) {
	if len(frames) == 0 {
		return nil, fmt.Errorf("kuma: HStack with no frames: %w", ErrNoValues)
	}
	if err := checkFrames("HStack", frames); err != nil {
		return nil, err
	}

	n := 0
	for _, f := range frames {
		n += f.NumCols()
	}

	cols := make([]Column, 0, n)
	for j, f := range frames {
		if f.NumRows() != frames[0].NumRows() {
			return nil, fmt.Errorf("kuma: HStack: frame %d has %d rows and frame 0 has %d: %w",
				j, f.NumRows(), frames[0].NumRows(), ErrLength)
		}
		cols = append(cols, f.cols...)
	}

	// NewFrame is what rejects two columns of one name, and it says which name
	// it was, which is the useful half of that error.
	return NewFrame(cols...)
}

// checkFrames rejects a nil frame in the list, which is a mistake in the
// program rather than something the data did but reads better as an error than
// as a nil dereference three functions further down.
func checkFrames[S any](who string, frames []*Frame[S]) error {
	for i, f := range frames {
		if f == nil {
			return fmt.Errorf("kuma: %s: frame %d is nil: %w", who, i, ErrNoValues)
		}
	}
	return nil
}

// chain puts the chunks of several columns end to end into one column.
//
// This is where stacking gets to be free. A chunked column is a list of arrays,
// so putting two of them together is putting the two lists together, and the
// values are not read, copied or moved.
func chain(dt dtype.DataType, parts []*array.Chunked) (*array.Chunked, error) {
	n := 0
	for _, p := range parts {
		n += p.NumChunks()
	}

	chunks := make([]*array.Array, 0, n)
	for _, p := range parts {
		chunks = append(chunks, p.Chunks()...)
	}

	out, err := array.NewChunked(dt, chunks...)
	if err != nil {
		return nil, fmt.Errorf("kuma: %w", err)
	}
	return out, nil
}

// nulls returns a column of n missing values of the given type.
func nulls(dt dtype.DataType, n int) (*array.Chunked, error) {
	if dt == dtype.Null {
		// The null type has no values to be missing, so there is nothing to
		// build and nothing to allocate.
		return array.NewChunked(dt, array.NewNull(n))
	}

	b, err := array.NewBuilder(dt)
	if err != nil {
		return nil, fmt.Errorf("kuma: %w", err)
	}
	b.Grow(n)
	for range n {
		b.AppendNull()
	}

	out, err := array.NewChunked(dt, b.Finish())
	if err != nil {
		return nil, fmt.Errorf("kuma: %w", err)
	}
	return out, nil
}
