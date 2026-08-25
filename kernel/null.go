package kernel

import (
	"errors"
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
)

// IsNull returns a boolean column that is true where c has no value.
//
// The result has no nulls of its own. Whether a value is missing is always
// known, even when the value is not, so the answer is a plain boolean and not a
// boolean that might itself be missing. That is the difference between this and
// a comparison, where pandas gives back NaN for a row it could not compare.
//
// It panics if c is nil.
func IsNull(c *array.Chunked) *array.Chunked {
	return nullMask(c, false)
}

// IsNotNull returns a boolean column that is true where c has a value. It is
// [IsNull] the other way round, and it is the mask a caller wants far more
// often, since dropping the rows with nothing in them is what most callers are
// really asking for.
//
// It panics if c is nil.
func IsNotNull(c *array.Chunked) *array.Chunked {
	return nullMask(c, true)
}

// nullMask builds the boolean column both of the two above want, with want
// saying which way round it goes.
//
// No value is looked at one at a time. A Bool column and a validity bitmap have
// the same layout, one bit per row, so the mask a caller is asking for is the
// bitmap the column already carries, and IsNotNull is a copy of it while IsNull
// is a copy with the words inverted. That is a byte of work per eight rows
// rather than a branch per row.
func nullMask(c *array.Chunked, want bool) *array.Chunked {
	if c == nil {
		panic("kernel: null check of a nil column")
	}

	chunks := c.Chunks()
	masks := make([]*array.Array, len(chunks))
	for i, a := range chunks {
		masks[i] = chunkMask(a, want)
	}

	out, err := array.NewChunked(dtype.Bool, masks...)
	if err != nil {
		panic("kernel: " + err.Error())
	}
	return out
}

// chunkMask builds the mask for one chunk.
func chunkMask(a *array.Array, want bool) *array.Array {
	var bits *bitmap.Bitmap
	switch {
	case a.NullCount() == 0:
		// Nothing is missing, so every answer is the same one and there is no
		// bitmap to read. A file of complete data is the common case and this is
		// the whole of the work for it.
		bits = allBits(a.Len(), want)
	case a.Validity() == nil:
		// A Null column, where the type itself says every value is missing.
		bits = allBits(a.Len(), !want)
	default:
		// Slice copies, and it is the copy the result needs anyway, so the
		// offset of a sliced chunk costs nothing beyond the shift it already
		// does.
		bits = a.Validity().Slice(a.Offset(), a.Offset()+a.Len())
		if !want {
			bits.Not()
		}
	}

	// The bytes go through a fresh buffer rather than being wrapped, so that the
	// result is aligned the way every other column is.
	buf := buffer.New(len(bits.Bytes()))
	copy(buf.Bytes(), bits.Bytes())

	out, err := array.New(dtype.Bool, a.Len(), buf, nil)
	if err != nil {
		panic("kernel: " + err.Error())
	}
	return out
}

// allBits returns a bitmap of n bits that are all set or all clear.
func allBits(n int, set bool) *bitmap.Bitmap {
	if set {
		return bitmap.NewSet(n)
	}
	return bitmap.New(n)
}

// KeepIndex returns the positions of the rows where at least present of cols
// have a value, in order. It is what dropping the rows that are too empty to be
// worth keeping comes down to, and the positions go straight into [Take].
//
// Every column has to be rows long. A present of zero or less keeps every row,
// since every row has at least nothing, and a present larger than the number of
// columns keeps none of them.
//
// A column with nothing missing is counted without being read, so a frame of
// complete data is answered by counting the columns rather than the rows. The
// columns that can fail are read through their validity bitmaps, so a row costs
// a shift and a mask rather than the binary search that finding one value in a
// chunked column costs.
func KeepIndex(cols []*array.Chunked, rows, present int) []int {
	if present <= 0 {
		return everyRow(rows)
	}

	complete := 0
	check := make([]*array.Chunked, 0, len(cols))
	for i, c := range cols {
		if c == nil {
			panic("kernel: nil column in a keep")
		}
		if c.Len() != rows {
			panic(fmt.Sprintf("kernel: column %d is %d rows, want %d", i, c.Len(), rows))
		}
		if c.NullCount() == 0 {
			complete++
			continue
		}
		check = append(check, c)
	}

	// Enough columns cannot fail, so nothing can, and the rest are not worth
	// reading.
	if complete >= present {
		return everyRow(rows)
	}

	// More columns have to be there than there are columns left that could be,
	// so no row passes.
	need := present - complete
	if need > len(check) {
		return nil
	}

	if len(check) == 1 {
		return validIndex(check[0], make([]int, 0, rows))
	}

	counts := make([]int32, rows)
	for _, c := range check {
		countValid(c, counts)
	}

	idx := make([]int, 0, rows)
	for i, n := range counts {
		if int(n) >= need {
			idx = append(idx, i)
		}
	}
	return idx
}

// everyRow returns the positions of all rows rows, which is the answer whenever
// the rule turns out to be one no row can fail.
func everyRow(rows int) []int {
	idx := make([]int, rows)
	for i := range idx {
		idx[i] = i
	}
	return idx
}

// validIndex appends the positions where c has a value to idx. It is KeepIndex
// for one column, where the count is the answer and there is nothing to add up.
func validIndex(c *array.Chunked, idx []int) []int {
	base := 0
	for _, a := range c.Chunks() {
		switch valid := a.Validity(); {
		case a.NullCount() == 0:
			for i := range a.Len() {
				idx = append(idx, base+i)
			}
		case valid != nil:
			off := a.Offset()
			for i := range a.Len() {
				if valid.Get(off + i) {
					idx = append(idx, base+i)
				}
			}
		}
		// A chunk with a null count and no bitmap is a Null column, where
		// nothing is there and nothing is kept.
		base += a.Len()
	}
	return idx
}

// countValid adds one to counts[i] for every row i where c has a value.
func countValid(c *array.Chunked, counts []int32) {
	base := 0
	for _, a := range c.Chunks() {
		switch valid := a.Validity(); {
		case a.NullCount() == 0:
			for i := range a.Len() {
				counts[base+i]++
			}
		case valid != nil:
			off := a.Offset()
			for i := range a.Len() {
				if valid.Get(off + i) {
					counts[base+i]++
				}
			}
		}
		base += a.Len()
	}
}

// FillNull returns a column with every missing value of c replaced by the one
// value in fill.
//
// There is no per type code here, and that is deliberate. The column and the
// one value fill are chained into a single column, which costs nothing because
// a chunked column is a list of chunks, and then the answer is a gather that
// takes position i where there is a value and the fill's position where there
// is not. [Take] already knows how to read every type, so this works for
// strings and timestamps and decimals without any of them being named.
//
// The result is a new column, since a fill has to write the values it changed
// and there is nowhere to write them but a new buffer. A column with nothing
// missing is handed straight back, because there is nothing to change and
// nothing to copy.
//
// It returns an error if fill is not exactly one value, if it is a different
// type from c, or if the value in it is itself missing, all of which are the
// caller asking for something that has no meaning rather than a bug in the
// program.
func FillNull(c *array.Chunked, fill *array.Array) (*array.Chunked, error) {
	if c == nil {
		panic("kernel: fill of a nil column")
	}
	if fill == nil {
		return nil, errors.New("kernel: fill with no value")
	}
	if fill.Len() != 1 {
		return nil, fmt.Errorf("kernel: fill with %d values, want exactly one", fill.Len())
	}
	if fill.DType() != c.DType() {
		return nil, fmt.Errorf("kernel: fill a %s column with a %s value",
			c.DType(), fill.DType())
	}
	if fill.IsNull(0) {
		return nil, fmt.Errorf("kernel: fill a %s column with a missing value", c.DType())
	}
	if c.NullCount() == 0 {
		return c, nil
	}

	chunks := c.Chunks()
	both := make([]*array.Array, 0, len(chunks)+1)
	both = append(both, chunks...)
	both = append(both, fill)

	chained, err := array.NewChunked(c.DType(), both...)
	if err != nil {
		return nil, err
	}

	n := c.Len()
	idx := make([]int, n)
	for i := range idx {
		if c.IsValid(i) {
			idx[i] = i
			continue
		}
		idx[i] = n
	}
	return Take(chained, idx), nil
}
