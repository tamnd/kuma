package parquet

import (
	"fmt"
	"io"
	"slices"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Reading the columns that were asked for and none of the others.
//
// Projection is what the format is for. A file of two hundred columns keeps
// each of them in a run of pages of its own and says in the footer where every
// one of those runs is, so a query that wants three of them reads three runs
// and never touches the rest: not read, not decompressed, not allocated for.
// That is the difference between a parquet file and a CSV file, and a reader
// that reads every column and throws most of them away has given it up.
//
// A FileReader is a file with a projection on it. It is made from the footer,
// which is three reads at the end of the file and none at all of the data, and
// it hands back one row group at a time in the shape the projection asked for.
//
// What it counts on the way is bytes. BytesRead is every byte that came out of
// the file through it, the footer included, which is the number a projection is
// judged by: reading three columns of a file has to cost what those three
// columns take rather than what the file takes.
//
// A row group at a time rather than a whole file at a time because that is the
// unit the format is written in and the unit a scan works in. A row group holds
// as many rows as its writer thought fit in memory at once, and the reason to
// have the boundaries is to be able to stop at them.
//
// Which of them to stop at is what Bounds is for. Nearly every writer writes the
// smallest and largest value of every chunk into the footer, so a scan with a
// filter on it can ask a row group what it holds and skip it whole. On a file
// written in any sort of order that saves more than the projection does and it
// costs nothing to ask, the numbers being in the footer that was read to open
// the file.

// Batch is the columns of one row group.
type Batch struct {
	// Length is the number of rows.
	Length int

	// Columns are the values, one array per field of the reader's schema, in
	// the order the projection named them.
	Columns []*array.Array
}

// FileReader reads the row groups of a parquet file, a projection at a time.
//
// It is made from the footer and keeps it, so making one reads the end of the
// file and nothing else. What it hands back is a Batch per row group holding
// the columns Project named, which is all of them until it is called.
//
// It keeps the buffers a column is assembled in from one row group to the next,
// so a scan of a file of a thousand row groups allocates what one of them needs
// rather than a thousand times that. That is also what makes it a thing to use
// from one goroutine at a time: two goroutines reading row groups of the same
// file want a reader each, which costs another footer and nothing more.
type FileReader struct {
	src  *counted
	size int64
	meta *Metadata

	// columns is every leaf of the file's schema, in the order a row group
	// holds its chunks, and take is the projection as indices into it.
	columns []Column
	take    []int

	// readers assemble the projected columns, one per entry of take, each made
	// when its column is first read. A reader is good for one chunk after
	// another of the same column, which is what makes it worth keeping.
	readers []*ColumnReader

	// schema is what a batch holds and kv is the file's own key and value
	// metadata, kept apart because a projection rebuilds the fields and leaves
	// the metadata alone.
	schema dtype.Schema
	kv     dtype.Metadata
}

// NewFileReader returns a reader for the parquet file in r.
//
// The size is the size of the file, the same one ReadMetadata takes and for the
// same reason: a footer at the end of a file cannot be found by anything that
// only reads forwards. Nothing but the footer is read here.
//
// The reader starts with every column of the file projected, in the order the
// row groups hold their chunks. Project is what narrows it.
func NewFileReader(r io.ReaderAt, size int64) (*FileReader, error) {
	src := &counted{src: r}
	meta, err := ReadMetadata(src, size)
	if err != nil {
		return nil, err
	}
	columns, err := meta.Columns()
	if err != nil {
		return nil, err
	}

	kv := make(dtype.Metadata, len(meta.KeyValue))
	for i, e := range meta.KeyValue {
		kv[i] = dtype.KeyValue{Key: e.Key, Value: e.Value}
	}

	f := &FileReader{src: src, size: size, meta: meta, columns: columns, kv: kv}
	take := make([]int, len(columns))
	for i := range take {
		take[i] = i
	}
	f.project(take)
	return f, nil
}

// Metadata returns the footer the reader works from.
//
// It is the reader's own rather than a copy of it, so a caller that changes it
// changes what the next read reads.
func (r *FileReader) Metadata() *Metadata { return r.meta }

// Columns returns every leaf of the file's schema, whatever is projected.
//
// These are what Project names, in the order a row group holds its chunks,
// which is the order a projection of all of them comes back in.
func (r *FileReader) Columns() []Column { return r.columns }

// Schema returns what a batch holds, which is one field per projected column.
//
// The fields are the leaves of the file's schema and are named the way a
// projection names them, so a leaf inside a group is "point.x" here and is a
// field of a field in the schema the file itself describes, which is what
// Metadata.Schema returns. A column is nullable when anything on its path is
// optional.
//
// The file's key and value metadata comes back on it whatever is projected,
// since it describes the file rather than the columns. That includes the Arrow
// schema pyarrow and Spark write under ARROW:schema, which still names every
// column of the file and not the projected ones.
//
// A chunk written as indices into a dictionary comes back dictionary encoded,
// which ColumnReader.Finish says more about, so the type of a column in a batch
// is either the type of the field here or a dictionary of it.
func (r *FileReader) Schema() dtype.Schema { return r.schema }

// NumRows returns how many rows the file holds, whatever is projected.
func (r *FileReader) NumRows() int64 { return r.meta.NumRows }

// NumRowGroups returns how many row groups the file holds.
func (r *FileReader) NumRowGroups() int { return len(r.meta.RowGroups) }

// BytesRead returns how many bytes have come out of the file through this
// reader so far, the footer included.
//
// This is what says a projection worked. Reading two columns of a file of two
// hundred has to cost what those two columns take, and the way to know it did
// is to add up the reads rather than to trust that the offsets in the footer
// were used.
func (r *FileReader) BytesRead() int64 { return r.src.n }

// Project narrows what a batch holds to the named columns, in the order they
// are named.
//
// A name is what Column.Name returns, which is the column's path joined with
// dots, so a leaf inside a group is "point.x" rather than "x". A name the file
// does not have is an error and nothing is narrowed, since a projection that
// was half applied would hand back batches nobody asked for.
//
// Naming a column twice reads it twice, which is a waste rather than a mistake.
// Naming none of them narrows the reader to no columns at all, which is what
// counting the rows of a file costs nothing with. Projecting again starts from
// the file's own columns rather than from what is projected now, so it widens
// as easily as it narrows.
func (r *FileReader) Project(names ...string) error {
	take := make([]int, len(names))
	for i, name := range names {
		take[i] = slices.IndexFunc(r.columns, func(c Column) bool { return c.Name() == name })
		if take[i] < 0 {
			return fmt.Errorf("parquet: no column called %q in the file", name)
		}
	}
	r.project(take)
	return nil
}

// project settles on a projection and builds the schema that goes with it.
func (r *FileReader) project(take []int) {
	fields := make([]dtype.Field, len(take))
	for i, k := range take {
		c := &r.columns[k]
		fields[i] = dtype.Field{Name: c.Name(), Type: c.Type, Nullable: c.MaxDefinition > 0}
	}
	r.take, r.readers = take, make([]*ColumnReader, len(take))
	r.schema = dtype.Schema{Fields: fields, Metadata: r.kv}
}

// RowGroup reads the projected columns of one row group.
//
// The columns are read in the order they were projected, one chunk each, and
// the batch comes back holding all of them. Reading the same row group twice
// reads the file twice: nothing is cached, since a row group of a real file is
// tens of megabytes and a caller that wants it twice can keep it.
func (r *FileReader) RowGroup(i int) (Batch, error) {
	g, err := r.group(i)
	if err != nil {
		return Batch{}, err
	}

	// The row count is a number out of a footer and every column is checked
	// against it, so it is worth one look before any of them is read. What it
	// has to fit in is memory, and an int is what says how much of that there
	// is.
	if g.NumRows < 0 || int64(int(g.NumRows)) != g.NumRows {
		return Batch{}, fmt.Errorf("parquet: %w: row group %d says it holds %d rows",
			ErrFormat, i, g.NumRows)
	}

	b := Batch{Length: int(g.NumRows), Columns: make([]*array.Array, len(r.take))}
	for j := range r.take {
		a, err := r.chunk(g, i, j)
		if err != nil {
			// The assembler stopped somewhere inside a chunk and what it holds
			// is half a column, so the next row group gets a new one.
			r.readers[j] = nil
			return Batch{}, err
		}
		b.Columns[j] = a
	}
	return b, nil
}

// Bounds returns what the writer said about the projected columns of one row
// group, one entry per column and in the order they were projected.
//
// This is what a row group is skipped on. The bounds are in the footer, so
// nothing is read out of the file to answer it: a scan asks each group what its
// columns hold, works out that nothing in it can match, and moves on without
// touching a page. ReadBounds is what each entry comes from and says what is in
// one and how much of it is worth acting on.
//
// A column that says nothing about itself comes back with no bounds rather than
// as an error, since a writer is allowed to write no statistics and most of the
// old ones wrote none worth reading. What is an error is a footer that
// contradicts itself, the same as it is for reading the values.
func (r *FileReader) Bounds(i int) ([]Bounds, error) {
	g, err := r.group(i)
	if err != nil {
		return nil, err
	}

	out := make([]Bounds, len(r.take))
	for j := range r.take {
		var ch *ColumnChunk
		var c *Column
		if ch, c, err = r.chunkOf(g, i, j); err != nil {
			return nil, err
		}
		if out[j], err = ReadBounds(*c, &ch.Meta); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// group returns the ith row group of the file.
func (r *FileReader) group(i int) (*RowGroup, error) {
	if i < 0 || i >= len(r.meta.RowGroups) {
		return nil, fmt.Errorf("parquet: row group %d of a file that has %d",
			i, len(r.meta.RowGroups))
	}
	return &r.meta.RowGroups[i], nil
}

// chunkOf returns the chunk holding the jth projected column of the row group g,
// which is the ith of the file, and the column it holds.
//
// A row group holds one chunk per leaf of the schema, in the same order, so the
// chunk for a column is the one in its place. The path is checked against the
// leaf's rather than searched for, since a file whose row group is in another
// order is one whose footer contradicts its own schema.
func (r *FileReader) chunkOf(g *RowGroup, group, j int) (*ColumnChunk, *Column, error) {
	k := r.take[j]
	c := &r.columns[k]
	if k >= len(g.Columns) || !slices.Equal(g.Columns[k].Meta.Path, c.Path) {
		return nil, nil, fmt.Errorf("parquet: %w: row group %d holds %d chunks and none of them is %s",
			ErrFormat, group, len(g.Columns), c.Name())
	}
	return &g.Columns[k], c, nil
}

// chunk reads the jth projected column of the row group g, which is the ith of
// the file.
func (r *FileReader) chunk(g *RowGroup, group, j int) (*array.Array, error) {
	ch, c, err := r.chunkOf(g, group, j)
	if err != nil {
		return nil, err
	}

	if r.readers[j] == nil {
		if r.readers[j], err = NewColumnReader(*c); err != nil {
			return nil, err
		}
	}

	a, err := r.readers[j].Chunk(r.src, r.size, ch)
	if err != nil {
		return nil, err
	}
	if int64(a.Len()) != g.NumRows {
		return nil, fmt.Errorf("parquet: %w: the chunk for %s holds %d values in a row group of %d rows",
			ErrFormat, c.Name(), a.Len(), g.NumRows)
	}
	return a, nil
}

// counted is an io.ReaderAt that adds up what has been read through it.
//
// The count is what a projection is judged by, so it counts what the file was
// asked for rather than what came back: a read that failed halfway still moved
// the disk and still says what the reader tried to do.
type counted struct {
	src io.ReaderAt
	n   int64
}

// ReadAt reads from the file and counts what it asked for.
func (c *counted) ReadAt(p []byte, off int64) (int, error) {
	c.n += int64(len(p))
	return c.src.ReadAt(p, off)
}
