package dataset

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Read reads every file of the dataset as one table.
//
// The columns are the ones the files hold, followed by the partition columns
// filled in from the paths, in the order [Dataset.Schema] has them. A file
// contributes its chunks as they arrived, so a table of a hundred files comes
// back in at least a hundred chunks and nothing is copied to join them.
//
//	t, err := dataset.Read(d, &dataset.ReadOptions{
//		Open: func(p string) (*array.Table, error) {
//			return parquet.ReadFile(p, nil)
//		},
//	})
//
// The files have to agree. Two files whose columns or types differ are
// [ErrSchema], naming both, because a dataset is one table and a table has one
// schema. A file holding a column with the same name as a partition column is
// the same error, since the path and the file would both be saying what it
// holds.
//
// A partition column is materialized here, one value repeated for every row of
// the file it came from. That is what an eager read has to do. The lazy engine
// will keep them virtual and build only the ones a query asks for, which is why
// a [Dataset] carries the values rather than the columns.
func Read(d *Dataset, opts *ReadOptions) (*array.Table, error) {
	if opts == nil || opts.Open == nil {
		return nil, fmt.Errorf("dataset: %w", ErrOpen)
	}
	if len(d.Files) == 0 {
		return nil, fmt.Errorf("dataset: nothing to read: %w", ErrNoData)
	}

	r := reader{d: d, opts: opts}
	if !opts.OmitPartitions {
		r.parts = d.Schema.Fields
	}
	for _, f := range d.Files {
		if err := r.file(f); err != nil {
			return nil, err
		}
	}
	return r.table()
}

// reader is one run of [Read]. The schema is taken from the first file and
// every file after it is checked against that one.
type reader struct {
	d    *Dataset
	opts *ReadOptions

	parts  []dtype.Field    // the partition columns, or nil when they are omitted
	schema dtype.Schema     // the files' columns, then the partition columns
	chunks [][]*array.Array // one list per column of the schema
	first  string           // the file the schema came from
}

// file reads one file and appends its chunks to the columns.
func (r *reader) file(f File) error {
	t, err := r.opts.Open(f.Path)
	if err != nil {
		return &FileError{Path: f.Path, Err: err}
	}
	if err = r.match(f, t); err != nil {
		return err
	}

	for i, c := range t.Columns {
		r.chunks[i] = append(r.chunks[i], c.Chunks()...)
	}

	// A file of no rows contributes no chunk to a partition column either. One
	// of no values would be legal, but it would leave the partition columns
	// holding a chunk the file columns have not got, for nothing.
	rows := t.NumRows()
	if rows == 0 {
		return nil
	}
	at := len(t.Columns)
	for i, p := range r.parts {
		a, err := repeat(p.Type, f.Values[i], rows)
		if err != nil {
			return &ValueError{
				Path:   f.Path,
				Column: p.Name,
				Value:  f.Values[i].String(),
				Type:   p.Type.String(),
				Err:    err,
			}
		}
		r.chunks[at+i] = append(r.chunks[at+i], a)
	}
	return nil
}

// match checks a file against the schema the table is being built with, taking
// that schema from the first file.
func (r *reader) match(f File, t *array.Table) error {
	if len(t.Columns) != t.Schema.Len() {
		return &FileError{Path: f.Path, Err: fmt.Errorf(
			"%d columns for a schema of %d fields: %w",
			len(t.Columns), t.Schema.Len(), ErrSchema)}
	}
	if len(f.Values) < len(r.parts) {
		return &FileError{Path: f.Path, Err: fmt.Errorf(
			"%d partition values for %d partition columns: %w",
			len(f.Values), len(r.parts), ErrSchema)}
	}

	if r.chunks == nil {
		return r.start(f, t)
	}
	if want := r.fileSchema(); !t.Schema.Equal(want) {
		return fmt.Errorf("dataset: %s holds %s and %s holds %s: %w",
			f.Path, t.Schema, r.first, want, ErrSchema)
	}
	return nil
}

// start takes the schema from the first file read.
func (r *reader) start(f File, t *array.Table) error {
	fields := make([]dtype.Field, 0, t.Schema.Len()+len(r.parts))
	fields = append(fields, t.Schema.Fields...)

	for _, p := range r.parts {
		if t.Schema.Index(p.Name) >= 0 {
			return fmt.Errorf(
				"dataset: %s holds a column called %q and so does its path: %w",
				f.Path, p.Name, ErrSchema)
		}
		fields = append(fields, p)
	}

	r.first = f.Path
	r.schema = dtype.Schema{Fields: fields}
	r.chunks = make([][]*array.Array, len(fields))
	return nil
}

// fileSchema is the part of the schema that came from the files, which is what
// the next file has to match.
func (r *reader) fileSchema() dtype.Schema {
	return dtype.Schema{Fields: r.schema.Fields[:len(r.schema.Fields)-len(r.parts)]}
}

// table gathers the chunks into columns.
func (r *reader) table() (*array.Table, error) {
	cols := make([]*array.Chunked, len(r.schema.Fields))
	for i := range cols {
		c, err := array.NewChunked(r.schema.Fields[i].Type, r.chunks[i]...)
		if err != nil {
			return nil, fmt.Errorf("dataset: %w", err)
		}
		cols[i] = c
	}
	return &array.Table{Schema: r.schema, Columns: cols}, nil
}

// repeat builds a chunk of n rows all holding the one partition value.
//
// The text is read once and appended n times rather than read n times, which is
// the difference between one parse and ten million of them on a file that big.
func repeat(dt dtype.DataType, v Value, n int) (*array.Array, error) {
	b, err := array.NewBuilder(dt)
	if err != nil {
		return nil, err
	}
	b.Grow(n)

	if v.Null {
		b.AppendNulls(n)
		return b.Finish(), nil
	}
	if err = fill(b, dt, v.Text, n); err != nil {
		return nil, err
	}
	return b.Finish(), nil
}

// readable says whether a partition value, which is a piece of text, can be
// read into a column of this type. It is what [fill] takes, checked up front by
// [Discover] so that a type nothing can read is caught before any walking.
//
// The wide string and binary types are not here. They are converted at the IPC
// boundary rather than built directly, so a partition column is a string or a
// binary and the boundary does the rest.
func readable(dt dtype.DataType) bool {
	switch dt.Kind() {
	case dtype.BoolKind, dtype.StringKind, dtype.BinaryKind:
		return true
	default:
		return dtype.IsInteger(dt) || dtype.IsFloat(dt)
	}
}

// fill appends the text n times, read as the type of the column.
//
// A declared column reads the permissive way, so a column declared int64 takes
// 01 as the number 1. Inference would have called that one a string, and a
// caller who declared the type asked for the number.
func fill(b *array.Builder, dt dtype.DataType, text string, n int) error {
	bits, _ := dtype.Bits(dt)

	switch k := dt.Kind(); {
	case k == dtype.StringKind || k == dtype.BinaryKind:
		// Text goes in one value at a time. The numbers below go in blocks
		// because a run of one number is a run of identical bytes, which is a
		// copy, and a string column is views and a byte store and there is no
		// bulk append that fills both.
		for range n {
			b.AppendString(text)
		}
	case k == dtype.BoolKind:
		v, err := strconv.ParseBool(text)
		if err != nil {
			return cause(err)
		}
		appendBools(b, v, n)
	case dtype.IsSigned(dt):
		v, err := strconv.ParseInt(text, 10, bits)
		if err != nil {
			return cause(err)
		}
		appendInt(b, dt, v, n)
	case dtype.IsUnsigned(dt):
		v, err := strconv.ParseUint(text, 10, bits)
		if err != nil {
			return cause(err)
		}
		appendUint(b, dt, v, n)
	case dtype.IsFloat(dt):
		v, err := strconv.ParseFloat(text, bits)
		if err != nil {
			return cause(err)
		}
		appendFloat(b, dt, v, n)
	default:
		// [Discover] turns these away when it builds the schema. It is checked
		// again here because a caller can build a [Dataset] by hand, and a type
		// nothing can fill has to be an error rather than a panic.
		return ErrUnsupportedType
	}
	return nil
}

// block is how many copies of a partition value are handed to the builder at a
// time.
//
// The whole run could go in one call, but that would mean a slice as long as
// the file, and a file is millions of rows. A block fits in the caches and the
// copies come out of it, so the cost is one memmove per block rather than a
// call per value and the memory is a constant.
const block = 4096

// appendInt appends a signed value n times, at the width of the column.
func appendInt(b *array.Builder, dt dtype.DataType, v int64, n int) {
	switch dt.Kind() {
	case dtype.Int8Kind:
		appendCopies(b, int8(v), n)
	case dtype.Int16Kind:
		appendCopies(b, int16(v), n)
	case dtype.Int32Kind:
		appendCopies(b, int32(v), n)
	default:
		appendCopies(b, v, n)
	}
}

// appendUint appends an unsigned value n times, at the width of the column.
func appendUint(b *array.Builder, dt dtype.DataType, v uint64, n int) {
	switch dt.Kind() {
	case dtype.Uint8Kind:
		appendCopies(b, uint8(v), n)
	case dtype.Uint16Kind:
		appendCopies(b, uint16(v), n)
	case dtype.Uint32Kind:
		appendCopies(b, uint32(v), n)
	default:
		appendCopies(b, v, n)
	}
}

// appendFloat appends a float value n times, at the width of the column.
func appendFloat(b *array.Builder, dt dtype.DataType, v float64, n int) {
	if dt.Kind() == dtype.Float32Kind {
		appendCopies(b, float32(v), n)
		return
	}
	appendCopies(b, v, n)
}

// appendCopies appends n copies of v, a block at a time.
func appendCopies[T array.Numeric](b *array.Builder, v T, n int) {
	buf := make([]T, min(n, block))
	for i := range buf {
		buf[i] = v
	}
	for n > 0 {
		k := min(n, len(buf))
		b.AppendValues(buf[:k])
		n -= k
	}
}

// appendBools appends n copies of v, a block at a time, the same way.
func appendBools(b *array.Builder, v bool, n int) {
	buf := make([]bool, min(n, block))
	if v {
		for i := range buf {
			buf[i] = true
		}
	}
	for n > 0 {
		k := min(n, len(buf))
		b.AppendBools(buf[:k])
		n -= k
	}
}

// cause returns what a failed parse was about, which is the syntax or the range
// error inside it rather than the wrapper carrying the function name and the
// input. Both of those are already in the [ValueError] this ends up in.
func cause(err error) error {
	var num *strconv.NumError
	if errors.As(err, &num) {
		return num.Err
	}
	return err
}
