package parquet

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// Leaving unread the row groups that cannot hold what was asked for.
//
// A projection cuts a file down by column. This cuts it down by row, and on a
// big file it is worth more. A row group holds as many rows as its writer
// thought fit in memory at once, which in practice is a million, and a scan
// looking for one day of a year of orders wants one row group in three hundred
// and sixty five. Reading the other three hundred and sixty four to throw their
// rows away is the whole cost of the query.
//
// What makes it possible is that the writer already said what each chunk holds.
// The smallest and largest value of every column of every row group is in the
// footer, which was read to open the file, so a reader can work out that a group
// covering March cannot hold a row from June without touching a page of it.
// [FileReader.Bounds] is those numbers and [Predicate.Keep] is the arithmetic on
// them.
//
// Bounds do nothing for a column that is not in any order. A customer identifier
// scattered across a hundred row groups puts every customer in the middle of
// every range, so the bounds keep every group and the scan reads the file. That
// is what a bloom filter answers instead: the writer hashed every value it wrote
// into a bitset, so a reader looking for one identifier can find a clear bit and
// know for certain the group never held it. It only answers equality, and only
// on a file whose writer was asked for one, which no writer does by default.
//
// Both of them only ever skip. A group that is kept is a group that may hold a
// matching row rather than one that does, so a caller filtering rows has to
// filter them, and a file with no statistics at all reads the way it read
// before. Nothing here can change an answer, only how much of the file it took.

// Predicate is one test on the values of one column.
//
// It is a column compared against one value, which is what most of a filter on a
// scan is made of and what a writer's statistics can be asked about. A list of
// them is an and: a row group is read when every one of them may hold in it.
type Predicate struct {
	// Column is the column to test, named the way [Column.Name] names it, so a
	// leaf inside a group is "point.x" rather than "x".
	//
	// It does not have to be a projected column. Filtering on a column and
	// reading it are different questions, and skipping nine row groups in ten on
	// a timestamp that never appears in the result is the ordinary case.
	Column string

	// Op is the comparison, with the column on the left and Value on the right,
	// so [kernel.OpLt] keeps the rows below Value.
	Op kernel.CompareOp

	// Value is what to compare against, as an array holding exactly one value
	// that is not missing. [array.Of] and [array.OfStrings] build one, and
	// [Where] and [WhereString] wrap the whole struct up.
	//
	// A value that is missing is refused rather than acted on. Nothing compares
	// to it, so a filter carrying one keeps no rows at all, which is a mistake
	// worth hearing about rather than an empty table worth returning.
	Value *array.Array
}

// Where returns a predicate comparing a column of numbers against v.
//
// A column whose type carries something else, a timestamp with a unit or a
// dictionary of strings, needs a value of that same type and so needs the struct
// written out with an array of one built for it.
func Where[T array.Numeric](column string, op kernel.CompareOp, v T) Predicate {
	return Predicate{Column: column, Op: op, Value: array.Of(v)}
}

// WhereString is [Where] for a column of text.
func WhereString(column string, op kernel.CompareOp, v string) Predicate {
	return Predicate{Column: column, Op: op, Value: array.OfStrings(v)}
}

// Keep reports whether a chunk holding these bounds may hold a row that passes
// the predicate.
//
// True is the answer that costs a read and false is the one worth having. A
// chunk it keeps is one whose range overlaps what was asked for, which is not
// the same as one holding a matching row, so a caller still has to look at the
// rows. A chunk it drops holds no matching row at all.
//
// Bounds a writer did not give, or gave in a form [ReadBounds] will not act on,
// are kept. So is a chunk whose bounds cannot be compared against the value,
// which is the one thing here that is an error, since a filter written against
// the wrong type is a mistake in the query rather than a fact about the file.
//
// The bounds do not have to be a whole row group's. [PageBounds] is the same
// shape for one page, and the same arithmetic decides it.
func (p Predicate) Keep(b Bounds) (bool, error) {
	if err := p.check(); err != nil {
		return false, err
	}

	// Nothing matches a value that is not there, so a chunk of nothing else has
	// no row for any comparison, this one included.
	if b.AllNull() {
		return false, nil
	}
	if b.Values == nil {
		return true, nil
	}

	// A writer leaves NaN out of the bounds it writes, so a float chunk whose
	// smallest and largest value are the same may hold a NaN as well, and a NaN
	// is unequal to everything. That is the only comparison a value left out of
	// the bounds changes the answer to, since every other one is false of a NaN,
	// and it is not worth asking on a column that can hold one.
	if p.Op == kernel.OpNe && dtype.IsFloat(b.Values.DType()) {
		return true, nil
	}

	// Two comparisons of the pair of bounds against the value answer all six
	// operators, since each of them is a question about whether the value falls
	// below the smallest, above the largest, or between the two.
	bounds, value := chunked(b.Values.DType(), b.Values), chunked(p.Value.DType(), p.Value)
	var got [2][2]bool
	for i, op := range [2]kernel.CompareOp{kernel.OpLe, kernel.OpGe} {
		v, ok, err := p.compare(bounds, value, op)
		if err != nil {
			return false, err
		}

		// A comparison nobody can answer, which needs a bound that is missing
		// and so needs a footer this package did not write.
		if !ok {
			return true, nil
		}
		got[i] = v
	}
	return holds(p.Op, got[0][0], got[1][0], got[0][1], got[1][1]), nil
}

// holds works out whether the range from the smallest to the largest value of a
// chunk overlaps what the operator asked for.
//
// The four booleans are the smallest and the largest value each compared against
// the value being looked for, which between them say where that value sits. So
// the chunk holds something smaller when its minimum is smaller, meaning at most
// and not equal, and something larger when its maximum is larger.
//
// A value that is a NaN makes all four false, which drops the chunk for every
// operator but the one that is true of everything. That is the IEEE answer and
// the one [kernel.Compare] gives over the rows, so a skip agrees with the filter
// it is standing in for.
func holds(op kernel.CompareOp, loLe, loGe, hiLe, hiGe bool) bool {
	switch op {
	case kernel.OpEq:
		return loLe && hiGe
	case kernel.OpNe:
		// Every value of the chunk is the value being looked for, which is the
		// one range that holds nothing unequal to it.
		one := loLe && loGe && hiLe && hiGe
		return !one
	case kernel.OpLt:
		return loLe && !loGe
	case kernel.OpLe:
		return loLe
	case kernel.OpGt:
		return hiGe && !hiLe
	default: // kernel.OpGe
		return hiGe
	}
}

// compare compares the pair of bounds against the predicate's value and returns
// the two answers, and whether they mean anything, which they do not when either
// bound is a value nobody can compare.
func (p Predicate) compare(bounds, value *array.Chunked, op kernel.CompareOp) ([2]bool, bool, error) {
	var got [2]bool

	c, err := kernel.Compare(bounds, value, op)
	if err != nil {
		return got, false, fmt.Errorf("parquet: the bounds of %s: %w", p.Column, err)
	}

	a := c.Chunk(0)
	if a.IsNull(0) || a.IsNull(1) {
		return got, false, nil
	}
	return [2]bool{a.Bool(0), a.Bool(1)}, true, nil
}

// check says whether the predicate is one that can be answered at all.
func (p Predicate) check() error {
	switch {
	case p.Op > kernel.OpGe:
		return fmt.Errorf("parquet: the predicate on %s uses an unknown operator %d",
			p.Column, uint8(p.Op))
	case p.Value == nil:
		return fmt.Errorf("parquet: the predicate on %s compares against no value", p.Column)
	case p.Value.Len() != 1:
		return fmt.Errorf("parquet: the predicate on %s compares against %d values, want one",
			p.Column, p.Value.Len())
	case p.Value.IsNull(0):
		return fmt.Errorf("parquet: the predicate on %s compares against a value that is not there, which no row passes",
			p.Column)
	}
	return nil
}

// RowGroups returns the row groups that may hold a row passing every predicate,
// in the order the file holds them.
//
// This is the pushdown. Every group it leaves out is one whose statistics say it
// holds no matching row, which the reader worked out from the footer and, where
// the writer wrote one, from a bloom filter. A group it returns is one that may
// hold a matching row rather than one that does, so the rows still have to be
// filtered once they are read.
//
// With no predicates it returns every group, which is what an unfiltered scan
// does and costs nothing to ask for. A predicate naming a column the file does
// not have is an error, the same as a projection naming one, since a filter that
// was quietly dropped would read the whole file and look like it worked.
//
// It reads the footer, which is already in hand, and a bloom filter for each
// equality on a column that has one. A file whose writer wrote no bloom filters,
// which is nearly all of them, is answered without touching the file at all.
func (r *FileReader) RowGroups(filter ...Predicate) ([]int, error) {
	tests, err := r.tests(filter)
	if err != nil {
		return nil, err
	}
	return r.rowGroups(tests)
}

// rowGroups is RowGroups once the filter has been looked over, which is where a
// read that goes on to filter the rows starts, having wanted the tests for that
// anyway.
func (r *FileReader) rowGroups(tests []test) ([]int, error) {
	out := make([]int, 0, len(r.meta.RowGroups))
	for i := range r.meta.RowGroups {
		keep, err := r.keep(i, tests)
		if err != nil {
			return nil, err
		}
		if keep {
			out = append(out, i)
		}
	}
	return out, nil
}

// test is a predicate with its column found and its value looked over, which is
// the work that belongs to the filter rather than to each row group.
type test struct {
	pred Predicate

	// column is the place of the column in the file, which is where its chunk
	// sits in every row group, and slot is the place of the same column in a
	// batch, which is where the rows to compare are. The second is only filled
	// in by a read that goes on to filter the rows, since the column has to be
	// projected before there is a place in a batch to name.
	column int
	slot   int

	// hash is the value hashed the way a writer building a bloom filter would
	// have hashed it, and hashed says there is such a hash. There is not for a
	// type the format hashes some other way, and there is no point in one unless
	// the comparison is an equality, since that is the only question a bloom
	// filter answers.
	hash   uint64
	hashed bool
}

// tests looks each predicate's column up and checks that the comparison is one
// that can be made, so that a filter is refused before any of the file is read
// rather than partway through it.
func (r *FileReader) tests(filter []Predicate) ([]test, error) {
	out := make([]test, len(filter))
	for i, p := range filter {
		if err := p.check(); err != nil {
			return nil, err
		}

		k := slices.IndexFunc(r.columns, func(c Column) bool { return c.Name() == p.Column })
		if k < 0 {
			return nil, fmt.Errorf("parquet: no column called %q in the file", p.Column)
		}
		c := &r.columns[k]
		if _, err := dtype.Coerce(c.Type, p.Value.DType()); err != nil {
			return nil, fmt.Errorf("parquet: cannot compare %s, which is %s, against a %s value: %w",
				p.Column, c.Type, p.Value.DType(), err)
		}

		out[i] = test{pred: p, column: k}
		if p.Op == kernel.OpEq {
			out[i].hash, out[i].hashed = bloomHash(c, p.Value)
		}
	}
	return out, nil
}

// keep says whether row group i may hold a row that passes every test.
//
// The bounds are asked first because they are in hand, and the bloom filter only
// for a group the bounds kept, since reading one costs a read of the file and a
// group already ruled out has nothing to gain from it.
func (r *FileReader) keep(group int, tests []test) (bool, error) {
	if len(tests) == 0 {
		return true, nil
	}
	g := &r.meta.RowGroups[group]

	for i := range tests {
		t := &tests[i]
		ch, c, err := r.chunkFor(g, group, t.column)
		if err != nil {
			return false, err
		}

		b, err := ReadBounds(*c, &ch.Meta)
		if err != nil {
			return false, err
		}
		keep, err := t.pred.Keep(b)
		if err != nil {
			return false, err
		}
		if !keep {
			return false, nil
		}

		if !t.hashed {
			continue
		}
		f, err := ReadBloomFilter(r.src, r.size, ch)
		if err != nil {
			return false, err
		}
		if !f.has(t.hash) {
			return false, nil
		}
	}
	return true, nil
}

// bloomHash returns the value hashed the way the writer of a bloom filter would
// have hashed it, and whether there is such a hash.
//
// A filter holds the hash of the value as a page writes it, so this is the value
// put back into the file's own shape and run through the format's one hash. The
// shape is the physical type rather than the type the schema means, since parquet
// writes an int8 in four bytes and a filter over that column holds the hash of
// the four.
//
// A value that is not exactly the column's type has no place in this at all. A
// hash is a lookup rather than a comparison, so there is no widening one side to
// meet the other: a hash of the wrong bytes finds an empty block and reports that
// a chunk holding the value does not, which is the one mistake a skip must never
// make.
func bloomHash(c *Column, a *array.Array) (uint64, bool) {
	if !dtype.Equal(a.DType(), c.Type) {
		return 0, false
	}

	switch c.Element.Type {
	case Int32:
		v, ok := narrowValue(a)
		return xxh64(binary.LittleEndian.AppendUint32(nil, uint32(v))), ok
	case Int64:
		v, ok := wideValue(a)
		return xxh64(binary.LittleEndian.AppendUint64(nil, uint64(v))), ok
	case Float:
		if a.DType().Kind() != dtype.Float32Kind {
			return 0, false
		}
		return xxh64(binary.LittleEndian.AppendUint32(nil, math.Float32bits(a.Value[float32](0)))), true
	case Double:
		if a.DType().Kind() != dtype.Float64Kind {
			return 0, false
		}
		return xxh64(binary.LittleEndian.AppendUint64(nil, math.Float64bits(a.Value[float64](0)))), true
	case ByteArray, FixedLenByteArray:
		switch a.DType().Kind() {
		case dtype.StringKind, dtype.BinaryKind, dtype.FixedSizeBinaryKind:
			return xxh64(a.Bytes(0)), true
		default:
			return 0, false
		}
	default:
		// A boolean, whose values are bits and which the format leaves out of
		// this for want of a byte to hash, and an int96, which no writer has
		// built a filter over since before the format had filters.
		return 0, false
	}
}

// narrowValue reads a value stored in fewer bits than the file wrote it in, as
// the int32 the file wrote. Every integer of thirty two bits or fewer is written
// that way, whether it is signed or not.
func narrowValue(a *array.Array) (int32, bool) {
	switch a.DType().Kind() {
	case dtype.Int8Kind:
		return int32(a.Value[int8](0)), true
	case dtype.Int16Kind:
		return int32(a.Value[int16](0)), true
	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		return a.Value[int32](0), true
	case dtype.Uint8Kind:
		return int32(a.Value[uint8](0)), true
	case dtype.Uint16Kind:
		return int32(a.Value[uint16](0)), true
	case dtype.Uint32Kind:
		return int32(a.Value[uint32](0)), true
	default:
		return 0, false
	}
}

// wideValue reads a value the file wrote in eight bytes, as the int64 it wrote.
func wideValue(a *array.Array) (int64, bool) {
	switch a.DType().Kind() {
	case dtype.Int64Kind, dtype.Date64Kind, dtype.Time64Kind,
		dtype.TimestampKind, dtype.DurationKind:
		return a.Value[int64](0), true
	case dtype.Uint64Kind:
		return int64(a.Value[uint64](0)), true
	default:
		return 0, false
	}
}
