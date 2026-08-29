package parquet

import (
	"fmt"
	"io"
	"os"
	"slices"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// Reading a whole file at once.
//
// [FileReader] hands back a row group at a time, which is what a scan wants and
// what a file larger than memory needs. What most callers want most of the time
// is the file, and this is that: the row groups read in order and each column
// put together out of the chunks the groups gave it.
//
// A column comes back in chunks rather than in one piece. Each row group
// contributed one and there is no reason to copy them into a single buffer,
// since every kernel in this library reads a chunked column and a file of a
// hundred row groups would otherwise want one allocation the size of the file.

// Options says what to read and in what shape.
//
// The zero value reads every column of the file and gives each of them the type
// the file's schema names, which is what a nil pointer means as well.
type Options struct {
	// Columns names the columns to read, in the order the table should hold
	// them, using the dotted path [Column.Name] returns. Nil reads every column
	// of the file in the order the file holds them.
	//
	// This is the projection, and it is the reason to use this format at all. A
	// file of two hundred columns keeps each of them in a run of pages of its
	// own, so naming three of them reads three runs and never touches the rest:
	// not read, not decompressed, not allocated for.
	//
	// A name the file does not have is an error, since a caller who asked for a
	// column by a name that is not there wants to hear about it rather than get
	// a table with a column missing.
	Columns []string

	// Dictionary keeps the encoding of any column the file wrote as indices
	// into a dictionary, so such a column comes back as a [dtype.Dictionary]
	// of the type the file's schema names rather than as that type.
	//
	// It is off by default because a caller who asked for a file of country
	// codes wants a column of strings, and because most writers encode nearly
	// every column whether it repeats or not, so leaving it on would mean a
	// dictionary of a million distinct values as readily as one of two hundred.
	// It is worth turning on for the columns it was meant for: a group by, a
	// join and a filter all read through the encoding, and a column of country
	// codes kept encoded is the size of a column of small integers.
	Dictionary bool

	// Filter says which rows to keep, as predicates a row has to pass every one
	// of. Nil reads every row.
	//
	// This is the other half of not reading a file and on a file written in any
	// sort of order it saves more than the projection does. A row group whose
	// statistics say it holds no matching row is never opened, so a scan of a
	// year of orders looking for one day reads one row group rather than three
	// hundred and sixty five, and [FileReader.RowGroups] is that part on its
	// own.
	//
	// The rows of the groups that are read are compared as well, so what comes
	// back is the rows that pass and not the row groups that might hold them. A
	// row whose value is missing does not pass, since nothing compares to a
	// value that is not there, and neither does a row of a group with no
	// statistics that turned out not to match.
	//
	// A predicate may name a column Columns does not. Filtering on a column and
	// reading it are different questions, and filtering on a timestamp that is
	// not wanted in the result is the ordinary case, so such a column is read to
	// compare the rows against and left out of the table.
	//
	//	t, err := parquet.Read(r, size, &parquet.Options{
	//		Columns: []string{"id", "price"},
	//		Filter:  []parquet.Predicate{parquet.Where("day", kernel.OpEq, int64(19000))},
	//	})
	Filter []Predicate
}

// Read reads a whole parquet file.
//
// The size is the size of the file, the same one [ReadMetadata] takes and for
// the same reason: a footer at the end of a file cannot be found by anything
// that only reads forwards.
//
//	t, err := parquet.Read(r, size, &parquet.Options{Columns: []string{"id", "price"}})
//
// Each column comes back in as many chunks as there were row groups holding
// rows, which is the chunking the file was written with and the chunking every
// kernel here is happy to read. A row group of no rows contributes nothing, and
// neither does one whose rows [Options.Filter] all rejected, so a filtered read
// of a file of a hundred row groups comes back in as many chunks as held a
// matching row.
func Read(r io.ReaderAt, size int64, opts *Options) (*array.Table, error) {
	f, err := NewFileReader(r, size)
	if err != nil {
		return nil, err
	}
	if opts == nil {
		opts = &Options{}
	}
	if opts.Columns != nil {
		if err := f.Project(opts.Columns...); err != nil {
			return nil, err
		}
	}
	return f.table(opts)
}

// ReadFile reads the file at path. It is [Read] over an open file, with the
// name of the file in any error the read returns.
func ReadFile(path string, opts *Options) (*array.Table, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	t, err := readOpen(f, opts)

	// A close after a read has nothing left to finish and next to nothing to
	// report, but the read error is the interesting one when there is one.
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return t, nil
}

// readOpen is ReadFile once the file is open, which is where the size is asked
// for. It is apart from ReadFile so that a file that went away underneath the
// reader is something a test can arrange.
func readOpen(f *os.File, opts *Options) (*array.Table, error) {
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	return Read(f, info.Size(), opts)
}

// table reads the row groups the filter cannot rule out, keeps the rows of them
// that pass it, and puts the chunks of each column together.
func (r *FileReader) table(opts *Options) (*array.Table, error) {
	tests, err := r.tests(opts.Filter)
	if err != nil {
		return nil, err
	}
	groups, err := r.rowGroups(tests)
	if err != nil {
		return nil, err
	}

	// The filter may want a column the caller does not, and that column has to
	// be read to compare the rows against, so it goes on the end of the
	// projection and the table is cut back to what was asked for. want is where
	// that cut falls.
	want := len(r.take)
	r.readAlso(tests)

	chunks, err := r.chunksOf(groups, tests, want)
	if err != nil {
		return nil, err
	}

	fields := slices.Clone(r.schema.Fields[:want])
	cols := make([]*array.Chunked, want)
	for j := range fields {
		cols[j] = column(fields[j], chunks[j], opts.Dictionary)
		fields[j].Type = cols[j].DType()
	}
	return &array.Table{
		Schema:  dtype.Schema{Fields: fields, Metadata: r.schema.Metadata},
		Columns: cols,
	}, nil
}

// readAlso adds whatever columns the filter needs to the projection and says
// where each test will find its column in a batch.
//
// Filtering on a column and reading it are different questions, so a predicate
// may name a column the projection does not and that column still has to be read
// to compare the rows against. A predicate on a column that is already projected
// is compared against the column that is already being read rather than reading
// it a second time, and so are two predicates on the same column.
func (r *FileReader) readAlso(tests []test) {
	if len(tests) == 0 {
		return
	}

	take := slices.Clone(r.take)
	for i := range tests {
		k := slices.Index(take, tests[i].column)
		if k < 0 {
			k = len(take)
			take = append(take, tests[i].column)
		}
		tests[i].slot = k
	}
	r.project(take)
}

// chunksOf reads the given row groups and returns the chunks of the first want
// columns, with the rows the tests reject left out.
func (r *FileReader) chunksOf(groups []int, tests []test, want int) ([][]*array.Array, error) {
	chunks := make([][]*array.Array, want)
	for _, i := range groups {
		b, err := r.RowGroup(i)
		if err != nil {
			return nil, err
		}
		if len(tests) > 0 {
			if b, err = pick(b, tests, want); err != nil {
				return nil, err
			}
		}

		// A row group of no rows holds a chunk of nothing per column, and what
		// it would say about the type of that column is nothing either, so it
		// is dropped here rather than argued with below. So is one the filter
		// emptied, which is the ordinary way for a group the statistics kept to
		// turn out to hold nothing.
		if b.Length == 0 {
			continue
		}
		for j := range want {
			chunks[j] = append(chunks[j], b.Columns[j])
		}
	}
	return chunks, nil
}

// pick returns the first want columns of a batch holding the rows that pass
// every test, which is at least one.
//
// The mask is worked out once and turned into positions once, which is what
// [kernel.Indices] is for, and only the columns going into the table are
// gathered: the ones the filter wanted have been read by then.
func pick(b Batch, tests []test, want int) (Batch, error) {
	var mask *array.Chunked
	for i := range tests {
		t := &tests[i]
		a := b.Columns[t.slot]

		m, err := kernel.Compare(chunked(a.DType(), a), chunked(t.pred.Value.DType(), t.pred.Value), t.pred.Op)
		if err != nil {
			return Batch{}, fmt.Errorf("parquet: filtering on %s: %w", t.pred.Column, err)
		}
		if mask == nil {
			mask = m
			continue
		}
		mask = and(mask, m)
	}

	// Every row passed, which is what a filter on a column the file is sorted
	// by does to most of the groups it keeps, so the arrays the row group was
	// read into are the answer and nothing is gathered.
	idx := kernel.Indices(mask)
	if len(idx) == b.Length {
		return Batch{Length: b.Length, Columns: b.Columns[:want]}, nil
	}

	out := Batch{Length: len(idx), Columns: make([]*array.Array, want)}
	for j := range want {
		a := b.Columns[j]
		out.Columns[j] = kernel.Take(chunked(a.DType(), a), idx).Chunk(0)
	}
	return out, nil
}

// and joins two masks, where both of them came out of [kernel.Compare] and so
// are the conditions it wants.
func and(a, b *array.Chunked) *array.Chunked {
	c, err := kernel.And(a, b)
	if err != nil {
		panic("parquet: " + err.Error())
	}
	return c
}

// column puts the chunks of one column together as one type.
//
// A column is one type and the chunks do not always agree on what it is, since
// a writer decides per row group whether to write indices into a dictionary and
// is allowed to change its mind partway through a file for the good reason that
// the values stopped repeating. Decoding settles that by leaving the encoding
// out of it. Keeping it settles that by taking what the chunks say when they
// all say the same thing, and decoding after all when they do not.
func column(f dtype.Field, chunks []*array.Array, keep bool) *array.Chunked {
	dt := f.Type
	if keep && len(chunks) > 0 {
		dt = chunks[0].DType()
		for _, a := range chunks[1:] {
			if !dtype.Equal(a.DType(), dt) {
				dt = f.Type
				break
			}
		}
	}

	out := make([]*array.Array, len(chunks))
	for i, a := range chunks {
		out[i] = a
		if !dtype.Equal(a.DType(), dt) {
			out[i] = decode(a, dt)
		}
	}
	return chunked(dt, out...)
}

// decode expands one dictionary encoded chunk.
//
// A chunk is either the type of its column or a dictionary of that type, since
// the values of a dictionary page are read with the same builder as the values
// of a plain one. So the only cast asked for here is the one that drops the
// encoding, which is a gather rather than a conversion and cannot fail on a
// value.
func decode(a *array.Array, dt dtype.DataType) *array.Array {
	c, err := kernel.Cast(chunked(a.DType(), a), dt)
	if err != nil {
		panic("parquet: " + err.Error())
	}
	return c.Chunk(0)
}

// chunked returns a column of chunks this package has just decided the type of,
// so the two agree and the error cannot happen.
func chunked(dt dtype.DataType, chunks ...*array.Array) *array.Chunked {
	c, err := array.NewChunked(dt, chunks...)
	if err != nil {
		panic("parquet: " + err.Error())
	}
	return c
}
