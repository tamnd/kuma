package parquet

import (
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
// kernel here is happy to read. A row group of no rows contributes nothing.
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
	return f.table(opts.Dictionary)
}

// ReadFile reads the file at path. It is [Read] over an open file.
func ReadFile(path string, opts *Options) (*array.Table, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readOpen(f, opts)
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

// table reads every row group and puts the chunks of each column together.
func (r *FileReader) table(keep bool) (*array.Table, error) {
	fields := slices.Clone(r.schema.Fields)
	chunks := make([][]*array.Array, len(fields))
	for i := range r.NumRowGroups() {
		b, err := r.RowGroup(i)
		if err != nil {
			return nil, err
		}

		// A row group of no rows holds a chunk of nothing per column, and what
		// it would say about the type of that column is nothing either, so it
		// is dropped here rather than argued with below.
		if b.Length == 0 {
			continue
		}
		for j, a := range b.Columns {
			chunks[j] = append(chunks[j], a)
		}
	}

	cols := make([]*array.Chunked, len(fields))
	for j := range fields {
		cols[j] = column(fields[j], chunks[j], keep)
		fields[j].Type = cols[j].DType()
	}
	return &array.Table{
		Schema:  dtype.Schema{Fields: fields, Metadata: r.schema.Metadata},
		Columns: cols,
	}, nil
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
