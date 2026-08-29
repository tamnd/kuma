package arrowgo

import (
	"fmt"

	"github.com/apache/arrow-go/v18/arrow"
	arrowarray "github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"

	"github.com/tamnd/kuma/array"
)

// ImportChunked returns a kuma column holding the chunks of an arrow-go one.
// The chunks stay separate, since joining them would copy every value to gain
// nothing and both sides read a column one chunk at a time anyway.
func ImportChunked(c *arrow.Chunked) (*array.Chunked, error) {
	if c == nil {
		return nil, fmt.Errorf("arrowgo: nil arrow column")
	}

	dt, err := ImportType(c.DataType())
	if err != nil {
		return nil, err
	}

	chunks := make([]*array.Array, len(c.Chunks()))
	for i, chunk := range c.Chunks() {
		chunks[i], err = ImportArray(chunk)
		if err != nil {
			return nil, err
		}
	}

	out, err := array.NewChunked(dt, chunks...)
	if err != nil {
		return nil, fmt.Errorf("arrowgo: %w", err)
	}
	return out, nil
}

// ExportChunked returns an arrow-go column over the chunks of a kuma one.
//
// The caller owns the result and should release it when it is done, which frees
// nothing but is what the rest of arrow-go expects to be able to do.
func ExportChunked(c *array.Chunked) (*arrow.Chunked, error) {
	if c == nil {
		return nil, fmt.Errorf("arrowgo: nil kuma column")
	}

	dt, err := ExportType(c.DType())
	if err != nil {
		return nil, err
	}

	chunks := make([]arrow.Array, 0, c.NumChunks())
	for _, chunk := range c.Chunks() {
		out, cerr := ExportArray(chunk)
		if cerr != nil {
			releaseAll(chunks)
			return nil, cerr
		}
		chunks = append(chunks, out)
	}

	// NewChunked retains what it is given, so the references made above are
	// handed over here and the caller is left holding one for the whole column.
	out := arrow.NewChunked(dt, chunks)
	releaseAll(chunks)
	return out, nil
}

// ImportRecordBatch returns a kuma table holding the columns of one arrow-go
// record batch, each of them a single chunk.
func ImportRecordBatch(rec arrow.RecordBatch) (*array.Table, error) {
	if rec == nil {
		return nil, fmt.Errorf("arrowgo: nil arrow record")
	}

	schema, err := ImportSchema(rec.Schema())
	if err != nil {
		return nil, err
	}

	columns := make([]*array.Chunked, rec.NumCols())
	for i := range columns {
		a, aerr := ImportArray(rec.Column(i))
		if aerr != nil {
			return nil, fmt.Errorf("the column %q: %w", schema.Fields[i].Name, aerr)
		}
		columns[i], aerr = array.NewChunked(a.DType(), a)
		if aerr != nil {
			return nil, fmt.Errorf("arrowgo: %w", aerr)
		}
	}
	return &array.Table{Schema: schema, Columns: columns}, nil
}

// ExportRecordBatch returns an arrow-go record batch over the columns of a kuma
// table.
//
// A record batch is one array per column, so a table whose columns are chunked
// is reported rather than joined up. Use [ExportTable] for that one, which is
// the type on the arrow-go side that holds chunks.
func ExportRecordBatch(t *array.Table) (arrow.RecordBatch, error) {
	schema, columns, err := exportColumns(t)
	if err != nil {
		return nil, err
	}

	arrays := make([]arrow.Array, 0, len(columns))
	for i, c := range columns {
		if c.NumChunks() > 1 {
			releaseAll(arrays)
			return nil, fmt.Errorf("arrowgo: the column %q is %d chunks and a record batch "+
				"holds one array per column, use ExportTable", t.Schema.Fields[i].Name, c.NumChunks())
		}
		a, aerr := exportOneChunk(c, schema.Field(i).Type)
		if aerr != nil {
			releaseAll(arrays)
			return nil, fmt.Errorf("the column %q: %w", t.Schema.Fields[i].Name, aerr)
		}
		arrays = append(arrays, a)
	}

	// NewRecord retains each array, so the references made here are handed over
	// and the caller is left holding one for the record.
	rec := arrowarray.NewRecordBatch(schema, arrays, int64(t.NumRows()))
	releaseAll(arrays)
	return rec, nil
}

// ImportTable returns a kuma table holding the columns of an arrow-go one,
// chunks and all.
func ImportTable(t arrow.Table) (*array.Table, error) {
	if t == nil {
		return nil, fmt.Errorf("arrowgo: nil arrow table")
	}

	schema, err := ImportSchema(t.Schema())
	if err != nil {
		return nil, err
	}

	columns := make([]*array.Chunked, t.NumCols())
	for i := range columns {
		columns[i], err = ImportChunked(t.Column(i).Data())
		if err != nil {
			return nil, fmt.Errorf("the column %q: %w", schema.Fields[i].Name, err)
		}
	}
	return &array.Table{Schema: schema, Columns: columns}, nil
}

// ExportTable returns an arrow-go table over the columns of a kuma one.
//
// The caller owns the result and should release it when it is done.
func ExportTable(t *array.Table) (arrow.Table, error) {
	schema, columns, err := exportColumns(t)
	if err != nil {
		return nil, err
	}

	cols := make([]arrow.Column, 0, len(columns))
	for i, c := range columns {
		chunked, cerr := ExportChunked(c)
		if cerr != nil {
			return nil, fmt.Errorf("the column %q: %w", t.Schema.Fields[i].Name, cerr)
		}
		cols = append(cols, *arrow.NewColumn(schema.Field(i), chunked))
		chunked.Release()
	}

	out := arrowarray.NewTable(schema, cols, int64(t.NumRows()))
	for i := range cols {
		cols[i].Release()
	}
	return out, nil
}

// exportColumns is the checking both table exports do first: the schema
// converts, there is a column for every field, and neither half is nil.
func exportColumns(t *array.Table) (*arrow.Schema, []*array.Chunked, error) {
	if t == nil {
		return nil, nil, fmt.Errorf("arrowgo: nil kuma table")
	}
	if t.NumCols() != t.Schema.Len() {
		return nil, nil, fmt.Errorf("arrowgo: a table of %d columns with a schema of %d fields",
			t.NumCols(), t.Schema.Len())
	}

	schema, err := ExportSchema(t.Schema)
	if err != nil {
		return nil, nil, err
	}
	for i, c := range t.Columns {
		if c == nil {
			return nil, nil, fmt.Errorf("arrowgo: the column %q is nil", t.Schema.Fields[i].Name)
		}
	}
	return schema, t.Columns, nil
}

// exportOneChunk is the single array of a column, which is the chunk when there
// is one and an empty array when the column has none.
//
// A column with no chunks is a column of no rows, and arrow-go has no way to
// describe one except as an array of length zero, so a builder makes one. It
// allocates nothing that matters and it is the only place in this package that
// builds rather than wraps.
func exportOneChunk(c *array.Chunked, dt arrow.DataType) (arrow.Array, error) {
	if c.NumChunks() == 1 {
		return ExportArray(c.Chunk(0))
	}
	if dt.ID() == arrow.NULL {
		return arrowarray.NewNull(0), nil
	}

	b := arrowarray.NewBuilder(memory.DefaultAllocator, dt)
	defer b.Release()
	return b.NewArray(), nil
}

// releaseAll drops the references made while building a list that is being
// abandoned, so that an error partway through leaves nothing retained.
func releaseAll(arrays []arrow.Array) {
	for _, a := range arrays {
		a.Release()
	}
}
