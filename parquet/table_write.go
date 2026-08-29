package parquet

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/bits"
	"os"
	"slices"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Writing a whole file.
//
// This is table.go the other way round and it is where the schema writer, the
// encoders, the page writer and the footer writer become a file. Almost none of
// the work is here. What is here is the bookkeeping that keeps the footer and
// the bytes in step, because a parquet file is pages with an index at the end
// saying where every one of them is, and the index is written last out of
// numbers collected while the pages went down. An offset that is one byte out
// gives a file that opens, reports the right number of rows, and hands back
// nonsense for the column it points at.
//
// So there is one counter, the number of bytes written so far, and every offset
// in the footer is a copy of it taken at the moment the thing it points at
// started. Nothing here computes an offset by adding up sizes, since that is the
// same number arrived at twice and the second way is the one that goes wrong.
//
// The file this writes is the plain one: every value written as it is, no
// compression, no dictionary, no statistics and no page index. That is a file
// any reader opens and it is the floor the rest is built on, since a dictionary
// page holds plain values and a chunk that gives up on its dictionary falls back
// to exactly what this writes. What it costs is size, and a caller who wants the
// file smaller wants the parts that are not written yet.
//
// Only flat columns go out. A list or a group needs repetition levels, and the
// reader next door does not read them either, so a writer that produced one
// would be writing files this package cannot open. A column of a type parquet
// cannot hold is refused by the schema writer before any of this starts.

// The layout a caller who says nothing gets.
//
// A row group of a million rows is what every other writer uses and it is the
// unit a reader skips, so a smaller one means finer skipping and a bigger footer
// and a larger one means the opposite. A page of a megabyte is what the format
// suggests and it is the unit a reader decompresses, which is why it is a
// thousand times smaller than the group.
const (
	defaultRowGroupSize = 1 << 20
	defaultPageSize     = 1 << 20

	// writtenBy is what goes in the footer when the caller does not say. It is
	// free text that nothing reads to decide anything, and it is worth setting
	// because it is the first thing anyone looks at when a file turns out to be
	// wrong.
	writtenBy = "kuma"
)

// WriteOptions says how the file is laid out. The zero value, which is what a
// nil pointer means as well, writes the layout described on each field.
type WriteOptions struct {
	// RowGroupSize is how many rows go in a row group, and defaults to a
	// million.
	//
	// A row group is what a reader skips whole, so this is the granularity of
	// every filter that reads statistics. It is also what a reader has to hold
	// to read one column of one group, so a group of ten million rows of
	// strings is a large allocation on the other side of the file.
	RowGroupSize int

	// PageSize is roughly how many bytes of values go in a data page, and
	// defaults to a megabyte.
	//
	// It is roughly because a page holds a whole number of values and the last
	// one is what takes it over. A column of fixed width values divides, and a
	// column of byte arrays is added up as it goes.
	PageSize int

	// CreatedBy is the writer's name and version, which goes in the footer as
	// free text. It defaults to naming this library.
	CreatedBy string
}

// Write writes a table as a parquet file and returns how many bytes it wrote.
//
//	n, err := parquet.Write(w, t, nil)
//
// The file is written forwards in one pass: the magic, then the pages of each
// row group a column at a time, then the footer that says where they all are.
// Nothing is seeked back to, so the writer can be a pipe or a network
// connection, and nothing is buffered beyond the page being built, so a table of
// a hundred million rows costs a page rather than a file.
//
// Every value is written plain and uncompressed, which is the file any reader
// opens and the largest one. A column whose type kuma has and parquet has not is
// refused by name rather than approximated, and so is a column that is a list, a
// map or a struct, since those need repetition levels that nothing here reads
// back yet.
//
// A dictionary encoded column is written as its values, because in parquet a
// dictionary is a decision about a page rather than a type and this does not
// write dictionary pages yet. Such a column comes back as its value type.
func Write(w io.Writer, t *array.Table, opts *WriteOptions) (int64, error) {
	tw, err := newTableWriter(w, t, opts)
	if err != nil {
		return 0, err
	}
	return tw.write()
}

// WriteFile writes a table to the file at path, creating it or truncating what
// is there. It is [Write] over a new file, with the name of the file in any
// error it returns.
//
// A write that fails part way leaves what got as far as the disk, the same as
// any other half finished file. There is nothing to be done about that in one
// pass, since a parquet file is only a parquet file once its footer is on.
func WriteFile(path string, t *array.Table, opts *WriteOptions) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	_, err = Write(f, t, opts)

	// The close is where a buffered write finally fails, so its error is worth
	// reporting when the write itself had nothing to say.
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

// tableWriter is one file being written.
//
// The buffers are on it rather than in the loops because a file is thousands of
// pages of one shape after another, and an encoder that keeps its buffer between
// pages allocates for the largest page of a column rather than for every page of
// it.
type tableWriter struct {
	w io.Writer

	// n is how many bytes have gone out, which is where the next thing written
	// starts and so is every offset the footer holds.
	n int64

	table   *array.Table
	opts    WriteOptions
	meta    Metadata
	columns []Column
	values  []*pageWriter

	plain  PlainEncoder
	levels RLEEncoder
	body   []byte
	defs   []int32
}

// newTableWriter settles everything about the file that does not depend on the
// values: the options, the schema, and how each column's values turn into bytes.
//
// It is apart from the writing so that a table this cannot write is refused
// before anything has been written, since a caller handed half a file and an
// error has to work out which half.
func newTableWriter(w io.Writer, t *array.Table, opts *WriteOptions) (*tableWriter, error) {
	tw := &tableWriter{w: w, table: t}
	if opts != nil {
		tw.opts = *opts
	}
	if tw.opts.RowGroupSize <= 0 {
		tw.opts.RowGroupSize = defaultRowGroupSize
	}
	if tw.opts.PageSize <= 0 {
		tw.opts.PageSize = defaultPageSize
	}
	if tw.opts.CreatedBy == "" {
		tw.opts.CreatedBy = writtenBy
	}

	if err := tw.check(); err != nil {
		return nil, err
	}
	if err := tw.meta.SetSchema(t.Schema); err != nil {
		return nil, err
	}
	tw.columns = leaves(&tw.meta)

	tw.values = make([]*pageWriter, len(tw.columns))
	for i := range tw.columns {
		c := &tw.columns[i]

		// A column that is not one node under the root is inside a list, a map
		// or a group, and every one of those needs repetition levels. The
		// schema writer will happily write the nodes for one, and the reader
		// here will not read them back, so this is where a file nothing could
		// open is turned into an error instead.
		if len(c.Path) != 1 {
			return nil, fmt.Errorf("parquet: %w: %s is inside a list or a group, and only flat columns are written yet",
				ErrUnsupported, c.Name())
		}

		v, err := writerFor(c)
		if err != nil {
			return nil, err
		}
		tw.values[i] = v
	}

	tw.meta.Version = 1
	tw.meta.NumRows = int64(t.NumRows())
	tw.meta.CreatedBy = tw.opts.CreatedBy
	return tw, nil
}

// check refuses a table whose columns do not match the schema it carries.
//
// Nothing else looks at this again. The schema is what the file says the columns
// are and the columns are what goes in the pages, so a table where the two
// disagree is a file that says one thing and holds another, and the reader that
// finds out is somebody else's.
func (tw *tableWriter) check() error {
	t := tw.table
	if len(t.Columns) != len(t.Schema.Fields) {
		return fmt.Errorf("parquet: a table of %d columns and a schema of %d fields",
			len(t.Columns), len(t.Schema.Fields))
	}

	rows := t.NumRows()
	for i, c := range t.Columns {
		if c.Len() != rows {
			return fmt.Errorf("parquet: the column %q has %d rows and the table has %d",
				t.Schema.Fields[i].Name, c.Len(), rows)
		}
	}

	// The ordinal of a row group is two bytes wide, so a file of more groups
	// than that can hold has nowhere to say which group is which. It takes a
	// caller asking for tiny groups to get there, and the alternative to
	// refusing is writing a number that wrapped.
	groups := (rows + tw.opts.RowGroupSize - 1) / tw.opts.RowGroupSize
	if groups > math.MaxInt16 {
		return fmt.Errorf("parquet: %d rows in groups of %d is %d row groups, and a file numbers them in two bytes",
			rows, tw.opts.RowGroupSize, groups)
	}
	return nil
}

// write puts the file down: the magic, the row groups, and the footer.
func (tw *tableWriter) write() (int64, error) {
	n, err := tw.w.Write([]byte(magic))
	tw.n += int64(n)
	if err != nil {
		return tw.n, fmt.Errorf("parquet: writing the magic: %w", err)
	}

	rows := tw.table.NumRows()
	for lo := 0; lo < rows; lo += tw.opts.RowGroupSize {
		if err = tw.group(lo, min(lo+tw.opts.RowGroupSize, rows)); err != nil {
			return tw.n, err
		}
	}

	// The footer is the last thing and is the whole of what makes the bytes in
	// front of it a file. A table of no rows still gets one, since a file with a
	// schema and no rows is a file and a caller who wrote an empty table wants
	// to read an empty table back.
	n64, err := WriteMetadata(tw.w, &tw.meta)
	tw.n += n64
	return tw.n, err
}

// group writes the rows in [lo, hi) as one row group, a column at a time.
//
// A column at a time is what makes the format worth reading: the pages of one
// column end up next to each other, so a reader that wants three columns out of
// two hundred reads three runs of bytes rather than picking through all of them.
func (tw *tableWriter) group(lo, hi int) error {
	g := RowGroup{
		Columns:    make([]ColumnChunk, len(tw.columns)),
		NumRows:    int64(hi - lo),
		FileOffset: tw.n,
		Ordinal:    int16(len(tw.meta.RowGroups)),
	}

	for i := range tw.columns {
		chunk, err := tw.chunk(i, tw.table.Columns[i].Slice(lo, hi))
		if err != nil {
			return err
		}
		g.Columns[i] = chunk
		g.TotalByteSize += chunk.Meta.TotalUncompressedSize
		g.TotalCompressedSize += chunk.Meta.TotalCompressedSize
	}

	tw.meta.RowGroups = append(tw.meta.RowGroups, g)
	return nil
}

// chunk writes one column of one row group and returns what the footer has to
// say about it.
func (tw *tableWriter) chunk(i int, col *array.Chunked) (ColumnChunk, error) {
	c := &tw.columns[i]

	// A required column writes no levels at all, so the only encoding in it is
	// the one the values are in. Saying otherwise would have a reader believe a
	// page holds something it does not.
	encodings := []Encoding{Plain}
	if c.MaxDefinition > 0 {
		encodings = []Encoding{Plain, RLE}
	}

	meta := ColumnMeta{
		Type:           c.Element.Type,
		Encodings:      encodings,
		Path:           c.Path,
		Codec:          Uncompressed,
		NumValues:      int64(col.Len()),
		DataPageOffset: tw.n,
	}

	for _, a := range col.Chunks() {
		// A column the caller kept dictionary encoded is written as its values,
		// since the schema says it is the value type and there are no dictionary
		// pages yet. The cast is a gather rather than a conversion.
		if !dtype.Equal(a.DType(), c.Type) {
			a = decode(a, c.Type)
		}

		for at := 0; at < a.Len(); {
			rows := tw.values[i].rows(a, at, tw.opts.PageSize)
			n, err := tw.page(c, tw.values[i], a, at, at+rows)
			if err != nil {
				return ColumnChunk{}, err
			}

			// The two sizes are the same because nothing is compressed. They
			// are both written because a reader sizing a buffer reads the
			// uncompressed one and a reader walking the pages reads the other.
			meta.TotalUncompressedSize += n
			meta.TotalCompressedSize += n
			at += rows
		}
	}
	return ColumnChunk{Meta: meta}, nil
}

// page writes rows [i, j) of one chunk as one data page.
//
// It is the first version of the data page, which keeps the levels inside the
// body behind four bytes of their length. The second version keeps them outside
// it and says how long they are in the header, which is worth having when the
// body is compressed and is a header field for nothing when it is not.
func (tw *tableWriter) page(c *Column, v *pageWriter, a *array.Array, i, j int) (int64, error) {
	tw.body = tw.body[:0]
	if c.MaxDefinition > 0 {
		levels := tw.definitions(c, a, i, j)
		tw.body = binary.LittleEndian.AppendUint32(tw.body, uint32(len(levels)))
		tw.body = append(tw.body, levels...)
	}

	tw.plain.Reset()
	v.encode(&tw.plain, a, i, j)
	tw.body = append(tw.body, tw.plain.Bytes()...)

	h := PageHeader{
		Kind:             DataPage,
		CompressedSize:   int32(len(tw.body)),
		UncompressedSize: int32(len(tw.body)),
		NumValues:        int32(j - i),
		Encoding:         Plain,

		// Both level encodings are given even for a column that writes no
		// levels, because a data page that does not say how its levels are
		// encoded is one the reader here refuses on the way in.
		DefinitionEncoding: RLE,
		RepetitionEncoding: RLE,
	}

	n, err := WritePage(tw.w, &h, tw.body)
	tw.n += n
	return n, err
}

// definitions writes the definition levels of rows [i, j).
//
// A flat column has one optional node on its path at most, so a value is either
// all the way down or missing at the top and there are two levels to write. The
// runs are what makes this nearly free: a column with nothing missing is one
// repeat and comes to three bytes however many rows it has.
func (tw *tableWriter) definitions(c *Column, a *array.Array, i, j int) []byte {
	tw.defs = slices.Grow(tw.defs[:0], j-i)
	for k := i; k < j; k++ {
		level := int32(c.MaxDefinition)
		if a.IsNull(k) {
			level = 0
		}
		tw.defs = append(tw.defs, level)
	}
	return mustLevels(&tw.levels, bits.Len(uint(c.MaxDefinition)), tw.defs)
}

// mustLevels writes levels that this package worked out itself.
//
// The width comes from the schema this writer wrote and the values are the
// levels it just built, so an encoder that refuses either of them has been asked
// for something this file cannot produce, which is a bug here rather than
// anything a caller can arrange.
func mustLevels(e *RLEEncoder, width int, levels []int32) []byte {
	if err := e.Reset(width); err != nil {
		panic("parquet: " + err.Error())
	}
	if err := e.Write(levels); err != nil {
		panic("parquet: " + err.Error())
	}
	return e.Finish()
}

// leaves is the columns of a schema this package has just written down, so the
// walk of it cannot fail for the same reason.
func leaves(m *Metadata) []Column {
	cols, err := m.Columns()
	if err != nil {
		panic("parquet: " + err.Error())
	}
	return cols
}

// pageWriter turns a run of rows of one column into the values of a page.
//
// The values of a page are the ones that are there. A missing value is written
// in the levels and nowhere else, so a column of a million rows with all but ten
// of them missing is ten values and a few bytes of levels, and the encoder never
// sees the rest.
type pageWriter struct {
	// encode writes rows [i, j) of a, leaving out the missing ones.
	encode func(e *PlainEncoder, a *array.Array, i, j int)

	// bits is how many bits one value takes, which is what the rows of a page
	// are worked out from. It is zero for a byte array, whose values have no
	// width until they are looked at.
	bits int
}

// rows says how many rows from i on go in one page.
//
// A missing value counts against the budget for a fixed width column, since
// working out otherwise means walking the validity of every row to save nothing
// but a few pages on a column that is mostly missing. It does not for a byte
// array, which is walked anyway.
func (v *pageWriter) rows(a *array.Array, i, budget int) int {
	left := a.Len() - i
	if v.bits > 0 {
		return min(max(budget*8/v.bits, 1), left)
	}

	size := 0
	for n := range left {
		if a.IsValid(i + n) {
			// Four bytes of length and the value, which is what the plain
			// encoding writes a byte array as.
			size += 4 + len(a.Bytes(i+n))
		}
		if size >= budget {
			return n + 1
		}
	}
	return left
}

// writerFor works out how the values of a column go into a page.
//
// This is valuesFor in column.go the other way round and it is a switch on the
// same thing, which is what one value of the column means rather than how it is
// stored. The physical type follows from that and is where the widening happens:
// parquet has two integer widths and kuma has eight, so six of them are widened
// on the way out and narrowed on the way back in.
func writerFor(c *Column) (*pageWriter, error) {
	v, err := valueWriter(c.Type)
	if err != nil {
		return nil, fmt.Errorf("parquet: %w: %s is a %s and that is not written yet",
			ErrUnsupported, c.Name(), c.Type)
	}
	v.bits = valueBits(c.Element)
	return v, nil
}

// valueWriter is writerFor without the column, which is everything that depends
// on the type alone.
func valueWriter(t dtype.DataType) (*pageWriter, error) {
	switch t.Kind() {
	case dtype.NullKind:
		// A column of nothing. Every row is missing, so the levels say all of
		// it and the page has no values in it at all.
		return &pageWriter{encode: func(*PlainEncoder, *array.Array, int, int) {}}, nil

	case dtype.BoolKind:
		return boolWriter(), nil

	case dtype.Int8Kind:
		return widened[int8]((*PlainEncoder).Int32), nil
	case dtype.Int16Kind:
		return widened[int16]((*PlainEncoder).Int32), nil
	case dtype.Uint8Kind:
		return widened[uint8]((*PlainEncoder).Int32), nil
	case dtype.Uint16Kind:
		return widened[uint16]((*PlainEncoder).Int32), nil
	case dtype.Uint32Kind:
		return widened[uint32]((*PlainEncoder).Int32), nil
	case dtype.Uint64Kind:
		return widened[uint64]((*PlainEncoder).Int64), nil

	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		return direct[int32]((*PlainEncoder).Int32), nil
	case dtype.Int64Kind, dtype.Time64Kind, dtype.TimestampKind:
		return direct[int64]((*PlainEncoder).Int64), nil
	case dtype.Float32Kind:
		return direct[float32]((*PlainEncoder).Float), nil
	case dtype.Float64Kind:
		return direct[float64]((*PlainEncoder).Double), nil

	case dtype.StringKind, dtype.LargeStringKind, dtype.BinaryKind, dtype.LargeBinaryKind:
		return blobWriter((*PlainEncoder).ByteArray), nil
	case dtype.FixedSizeBinaryKind:
		return blobWriter((*PlainEncoder).Fixed), nil

	default:
		// The decimals and the intervals, which are written as the bytes of a
		// wide integer and are a conversion of their own. Everything else that
		// gets this far was refused by the schema writer already.
		return nil, ErrUnsupported
	}
}

// direct writes a column already at the width parquet stores it at, which is
// the case that costs nothing when nothing is missing.
func direct[T array.Numeric](put func(*PlainEncoder, []T)) *pageWriter {
	var buf []T
	return &pageWriter{encode: func(e *PlainEncoder, a *array.Array, i, j int) {
		vals := a.Values[T]()
		if a.NullCount() == 0 {
			// The whole run in one call and no copy, which is the path a column
			// of numbers out of a file or a kernel takes.
			put(e, vals[i:j])
			return
		}

		buf = slices.Grow(buf[:0], j-i)
		for k := i; k < j; k++ {
			if a.IsValid(k) {
				buf = append(buf, vals[k])
			}
		}
		put(e, buf)
	}}
}

// widened writes a column narrower than the type it travels in, which is every
// integer the format has no width for and every unsigned one.
//
// An unsigned value becomes the signed one with the same bits, which is what
// the annotation on the column says to undo and what the reader does undo.
func widened[T array.Numeric, W array.Numeric](put func(*PlainEncoder, []W)) *pageWriter {
	var buf []W
	return &pageWriter{encode: func(e *PlainEncoder, a *array.Array, i, j int) {
		vals := a.Values[T]()
		buf = slices.Grow(buf[:0], j-i)
		for k := i; k < j; k++ {
			if a.IsValid(k) {
				buf = append(buf, W(vals[k]))
			}
		}
		put(e, buf)
	}}
}

// boolWriter writes a column of bits as a column of bools, which is the shape the
// plain encoder packs from.
func boolWriter() *pageWriter {
	var buf []bool
	return &pageWriter{encode: func(e *PlainEncoder, a *array.Array, i, j int) {
		buf = slices.Grow(buf[:0], j-i)
		for k := i; k < j; k++ {
			if a.IsValid(k) {
				buf = append(buf, a.Bool(k))
			}
		}
		e.Boolean(buf)
	}}
}

// blobWriter writes a column whose values are bytes, which is a byte array when the
// schema gives no width and a fixed length one when it does.
//
// The values are handed over as they sit in the column rather than copied, so
// what this costs is a slice header a value.
func blobWriter(put func(*PlainEncoder, [][]byte)) *pageWriter {
	var buf [][]byte
	return &pageWriter{encode: func(e *PlainEncoder, a *array.Array, i, j int) {
		buf = slices.Grow(buf[:0], j-i)
		for k := i; k < j; k++ {
			if a.IsValid(k) {
				buf = append(buf, a.Bytes(k))
			}
		}
		put(e, buf)
	}}
}

// valueBits is how many bits one value of a column takes in a page, or zero for
// a byte array, whose values are as long as they are.
func valueBits(e SchemaElement) int {
	switch e.Type {
	case Boolean:
		return 1
	case Int32, Float:
		return 32
	case Int64, Double:
		return 64
	case FixedLenByteArray:
		return 8 * int(e.TypeLength)
	default:
		// A byte array, and the int96 nothing writes.
		return 0
	}
}
