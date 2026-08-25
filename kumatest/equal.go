package kumatest

import (
	"bytes"
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// EqualFrames reports the difference between two frames on t, and says nothing
// when they are equal.
//
// The whole difference is one call to Errorf, so the report arrives in one
// piece rather than interleaved with whatever else is running, and the test
// carries on rather than stopping there. A test that checks three frames should
// say which of the three is wrong.
func EqualFrames[S any](t TB, got, want *kuma.Frame[S], o *Options) {
	t.Helper()

	if d := DiffFrames(got, want, o); d != "" {
		t.Errorf("%s", d)
	}
}

// EqualSeries reports the difference between two series on t, and says nothing
// when they are equal. It follows the rules in [EqualFrames].
func EqualSeries[T kuma.Value](t TB, got, want kuma.Series[T], o *Options) {
	t.Helper()

	if d := DiffSeries(got, want, o); d != "" {
		t.Errorf("%s", d)
	}
}

// EqualColumns reports the difference between two columns on t, and says
// nothing when they are equal. It follows the rules in [EqualFrames].
func EqualColumns(t TB, got, want kuma.Column, o *Options) {
	t.Helper()

	if d := DiffColumns(got, want, o); d != "" {
		t.Errorf("%s", d)
	}
}

// DiffFrames returns the difference between two frames as text, and returns the
// empty string when they are equal. It is what [EqualFrames] prints, for a
// caller who wants to decide what to do about it.
//
// Two frames whose columns are not the same names in the same order are
// reported as that and nothing more, since column three of one is not the
// column three of the other and comparing them cell by cell would bury the one
// thing worth reading in a table of noise. A column whose type is wrong is
// reported the same way and its values are left alone, while the columns either
// side of it are still compared.
func DiffFrames[S any](got, want *kuma.Frame[S], o *Options) string {
	opts := o.withDefaults()
	d := &diff{what: "frames", named: true, opts: &opts}

	switch {
	case got == nil && want == nil:
		return ""
	case got == nil:
		d.notef("there is no frame where there should be one of %s", shape(want))
		return d.String()
	case want == nil:
		d.notef("there is a frame of %s where there should be none", shape(got))
		return d.String()
	}

	names := got.Names()
	if !slices.Equal(names, want.Names()) {
		d.notef("the columns are %s where they should be %s",
			list(names), list(want.Names()))
		return d.String()
	}
	if got.NumRows() != want.NumRows() {
		d.notef("there are %d rows where there should be %d",
			got.NumRows(), want.NumRows())
	}

	pairs := make([]pair, 0, len(names))
	for i, name := range names {
		g, w := got.ColumnAt(i), want.ColumnAt(i)
		switch {
		case !dtype.Equal(g.DType(), w.DType()):
			d.notef("column %s is %s where it should be %s", name, g.DType(), w.DType())
		case !canCompare(g.DType()):
			d.notef("column %s is %s, which this package cannot compare yet",
				name, g.DType())
		default:
			pairs = append(pairs, pair{name: name, got: g, want: w})
		}
	}

	d.walk(pairs, min(got.NumRows(), want.NumRows()))
	return d.String()
}

// DiffSeries returns the difference between two series as text, and returns the
// empty string when they are equal. It is what [EqualSeries] prints.
func DiffSeries[T kuma.Value](got, want kuma.Series[T], o *Options) string {
	opts := o.withDefaults()
	d := &diff{what: "series", opts: &opts}
	d.one(got.Column(), want.Column())
	return d.String()
}

// DiffColumns returns the difference between two columns as text, and returns
// the empty string when they are equal. It is what [EqualColumns] prints.
func DiffColumns(got, want kuma.Column, o *Options) string {
	opts := o.withDefaults()
	d := &diff{what: "columns", opts: &opts}
	d.one(got, want)
	return d.String()
}

// pair is two columns of the same name and type, waiting to be compared, and
// where the walk has got to in each of them.
type pair struct {
	name string
	got  kuma.Column
	want kuma.Column

	gotAt  cursor
	wantAt cursor
}

// cursor walks the values of a column one at a time, remembering where it got
// to.
//
// Asking a chunked column for value i costs a binary search over its chunks,
// and a comparison that does that four times a cell spends longer finding the
// values than looking at them. The two sides cannot be taken a chunk at a time
// instead, since they are chunked however they were built and the boundaries
// need not line up, so this is the same answer the binary kernels came to: the
// cost per value is an increment and a comparison.
type cursor struct {
	chunks []*array.Array

	// c is the chunk the next value is in and i is where in that chunk it is.
	c, i int
}

// next returns the chunk holding the next value and where in that chunk it is.
// It must not be called more times than the column has rows.
func (c *cursor) next() (*array.Array, int) {
	for c.chunks[c.c].Len() == c.i {
		// An empty chunk is a chunk like any other and holds no value, so the
		// position moves past it rather than pointing into it.
		c.c++
		c.i = 0
	}
	a, i := c.chunks[c.c], c.i
	c.i++
	return a, i
}

// cell is one value that differs, as the report will print it.
type cell struct {
	row  int
	name string
	got  string
	want string
}

// diff is a report being put together: the lines about the two things as a
// whole, and then the cells inside them that differ.
type diff struct {
	what  string // what is being compared, in the plural
	named bool   // whether a cell has to say which column it is in
	opts  *Options

	notes []string
	cells []cell

	total   int  // how many cells differ, which is not how many are kept
	rows    int  // how many rows hold at least one of them
	scanned int  // how many rows were compared
	partial bool // whether that was fewer rows than one of the two has
}

// notef records a difference between the two as a whole, such as a shape or a
// type, rather than a difference between two values.
func (d *diff) notef(format string, args ...any) {
	d.notes = append(d.notes, fmt.Sprintf(format, args...))
}

// one compares a single pair of columns, which is what a series and a column
// diff are once the name and the type have been checked.
func (d *diff) one(got, want kuma.Column) {
	if got.Name() != want.Name() {
		d.notef("the name is %s where it should be %s", quote(got.Name()), quote(want.Name()))
	}
	if got.Len() != want.Len() {
		d.notef("there are %d rows where there should be %d", got.Len(), want.Len())
	}
	if !dtype.Equal(got.DType(), want.DType()) {
		d.notef("the type is %s where it should be %s", got.DType(), want.DType())
		return
	}
	if !canCompare(got.DType()) {
		d.notef("the type is %s, which this package cannot compare yet", got.DType())
		return
	}

	d.walk([]pair{{name: got.Name(), got: got, want: want}}, min(got.Len(), want.Len()))
}

// walk compares the first rows of every pair, a row at a time so that the cells
// come out in the order somebody reads them and so that the rows that differ
// can be counted as they go past.
func (d *diff) walk(pairs []pair, rows int) {
	d.scanned = rows
	for _, p := range pairs {
		if p.got.Len() > rows || p.want.Len() > rows {
			d.partial = true
		}
	}

	for i := range pairs {
		pairs[i].gotAt = cursor{chunks: pairs[i].got.Data().Chunks()}
		pairs[i].wantAt = cursor{chunks: pairs[i].want.Data().Chunks()}
	}

	for i := range rows {
		differs := false
		for j := range pairs {
			p := &pairs[j]
			g, gi := p.gotAt.next()
			w, wi := p.wantAt.next()
			if equalAt(g, gi, w, wi, d.opts) {
				continue
			}
			differs = true
			d.total++
			if d.opts.MaxCells < 0 || len(d.cells) < d.opts.MaxCells {
				d.cells = append(d.cells, cell{
					row:  i,
					name: p.name,
					got:  p.got.Text(i, d.opts.Print),
					want: p.want.Text(i, d.opts.Print),
				})
			}
		}
		if differs {
			d.rows++
		}
	}
}

// String returns the report, or the empty string if there is nothing to report.
func (d *diff) String() string {
	if len(d.notes) == 0 && d.total == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString(d.headline())
	for _, n := range d.notes {
		sb.WriteString("\n  ")
		sb.WriteString(n)
	}
	if len(d.cells) > 0 {
		sb.WriteString("\n\n")
		sb.WriteString(d.table())
	}
	if n := d.total - len(d.cells); n > 0 {
		fmt.Fprintf(&sb, "\n\nand %d more", n)
	}
	return sb.String()
}

// headline is the first line, which says how much differs. It is a line on its
// own so that a test log read at a glance says whether one value is wrong or
// the whole frame is.
func (d *diff) headline() string {
	switch {
	case d.total == 0:
		return d.what + " differ"
	case d.partial:
		// The row counts are in a note of their own, so the headline says how
		// much of what could be compared differed and leaves the arithmetic
		// about what could not to the line under it.
		return fmt.Sprintf("%s differ in %d of the rows they both have", d.what, d.rows)
	default:
		return fmt.Sprintf("%s differ in %d of %d rows", d.what, d.rows, d.scanned)
	}
}

// table lays the cells out under a header.
func (d *diff) table() string {
	t := newTable("row", "got", "want")
	if d.named {
		t = newTable("row", "column", "got", "want")
	}

	for _, c := range d.cells {
		if d.named {
			t.add(strconv.Itoa(c.row), c.name, c.got, c.want)
			continue
		}
		t.add(strconv.Itoa(c.row), c.got, c.want)
	}
	return t.String()
}

// equalAt reports whether value i of a is the same as value j of b. The two
// columns are known to be the same type by the time this is called.
func equalAt(a *array.Array, i int, b *array.Array, j int, o *Options) bool {
	if a.IsNull(i) || b.IsNull(j) {
		return a.IsNull(i) && b.IsNull(j)
	}

	switch a.DType().Kind() {
	case dtype.NullKind:
		// Every value of a null column is missing, so the check above has
		// already answered this.
		return true
	case dtype.BoolKind:
		return a.Bool(i) == b.Bool(j)
	case dtype.Int8Kind:
		return a.Value[int8](i) == b.Value[int8](j)
	case dtype.Int16Kind:
		return a.Value[int16](i) == b.Value[int16](j)
	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		return a.Value[int32](i) == b.Value[int32](j)
	case dtype.Int64Kind, dtype.Date64Kind, dtype.Time64Kind,
		dtype.TimestampKind, dtype.DurationKind:
		return a.Value[int64](i) == b.Value[int64](j)
	case dtype.Uint8Kind:
		return a.Value[uint8](i) == b.Value[uint8](j)
	case dtype.Uint16Kind:
		return a.Value[uint16](i) == b.Value[uint16](j)
	case dtype.Uint32Kind:
		return a.Value[uint32](i) == b.Value[uint32](j)
	case dtype.Uint64Kind:
		return a.Value[uint64](i) == b.Value[uint64](j)
	case dtype.Float32Kind:
		return closeEnough(float64(a.Value[float32](i)), float64(b.Value[float32](j)), o)
	case dtype.Float64Kind:
		return closeEnough(a.Value[float64](i), b.Value[float64](j), o)
	default:
		// The rest are the types whose values are a run of bytes, meaning the
		// strings, the binaries, the decimals and the intervals. canCompare
		// has already turned away everything that is not one of those.
		return bytes.Equal(a.Bytes(i), b.Bytes(j))
	}
}

// closeEnough reports whether two floating point numbers are equal under the
// allowances in o.
//
// The two allowances are read as either being enough rather than both having to
// hold, so a caller who sets one is not made to think about the other.
func closeEnough(a, b float64, o *Options) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return !o.NaNsDiffer && math.IsNaN(a) && math.IsNaN(b)
	}
	if a == b {
		// This is where two infinities of the same sign are equal, and where a
		// negative zero is equal to a positive one, which is what the numbers
		// say and what a test means.
		return true
	}
	if math.IsInf(a, 0) || math.IsInf(b, 0) {
		// The subtraction below would be a NaN for two infinities of opposite
		// sign, and no allowance covers the distance to a finite number.
		return false
	}

	away := math.Abs(a - b)
	return away <= o.Margin ||
		away <= o.Fraction*math.Min(math.Abs(a), math.Abs(b))
}

// canCompare reports whether this package knows how to compare two values of
// type t.
//
// The answer is yes for everything an array can hold today. It is here for the
// nested types, which cannot be built yet and which will need more than a value
// at a time when they can be, so that the day one turns up in a frame the
// report says so rather than quietly calling two of them equal.
func canCompare(t dtype.DataType) bool {
	switch t.Kind() {
	case dtype.NullKind, dtype.BoolKind,
		dtype.Int8Kind, dtype.Int16Kind, dtype.Int32Kind, dtype.Int64Kind,
		dtype.Uint8Kind, dtype.Uint16Kind, dtype.Uint32Kind, dtype.Uint64Kind,
		dtype.Float32Kind, dtype.Float64Kind,
		dtype.StringKind, dtype.BinaryKind,
		dtype.LargeStringKind, dtype.LargeBinaryKind, dtype.FixedSizeBinaryKind,
		dtype.Date32Kind, dtype.Date64Kind, dtype.Time32Kind, dtype.Time64Kind,
		dtype.TimestampKind, dtype.DurationKind, dtype.IntervalKind,
		dtype.Decimal128Kind, dtype.Decimal256Kind:
		return true
	default:
		return false
	}
}

// shape is a frame's size, for a report about a frame that is not there.
func shape[S any](f *kuma.Frame[S]) string {
	rows, cols := f.Shape()
	return fmt.Sprintf("%d rows x %d cols", rows, cols)
}

// list writes column names out in the order they are in.
func list(names []string) string {
	out := make([]string, len(names))
	for i, n := range names {
		out[i] = quote(n)
	}
	return "[" + strings.Join(out, " ") + "]"
}

// quote puts a name in quotes when it is empty or holds a space, so that a
// report about two names that look the same shows what the difference is.
func quote(name string) string {
	if name == "" || strings.ContainsAny(name, " \t\r\n\"") {
		return strconv.Quote(name)
	}
	return name
}
