package parquet

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/bits"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Putting a column back together.
//
// A page holds its levels and its values apart. The levels say which rows have
// a value and the values are only the rows that do, so a page of five hundred
// rows with a third of them missing holds five hundred levels and three hundred
// and thirty four values, and neither half means anything without the other.
// Assembly is walking the two together: a level that reaches the column's depth
// takes the next value, and one that does not is a null.
//
// That is the whole of it for a flat column, which is what this reads. A
// repeated column needs the repetition levels as well, which say where one row's
// list ends and the next begins, and needs an array of lists to put them in.
// Both are still to come.
//
// The values come out at the width kuma stores them at rather than the width
// the file wrote them at. Parquet has four integer widths and writes six of
// kuma's in the narrowest of them that fits, so a column of int8 arrives as
// int32 and has to be narrowed on the way in. Nothing is guessed: the schema
// already said which of the two it is, and the conversion is decided once for
// the column rather than once per value.

// ColumnReader assembles the pages of one column chunk into an array.
//
// Pages are handed to it in the order the file has them and it hands back an
// array at the end. It holds one builder and one set of buffers for the whole
// chunk, so a chunk of a hundred pages allocates what one page needs and reuses
// it.
//
// The reader is for one column, decided when it is made. What it can read is
// what the page decoders can read: a flat column, written plainly, as indices
// into a dictionary or as differences of one of the three kinds. Anything else
// is refused rather than guessed at. It reads a page that has already had its
// compression undone, which is what Chunk uses a Decompressor for.
type ColumnReader struct {
	column  Column
	builder *array.Builder

	// maxDefinition is the level a value has to reach to be present, and width
	// is how many bits a level takes. Both come out of the schema rather than
	// out of the page.
	maxDefinition int32
	width         uint

	// levels is the definition levels of the page being read, and values is how
	// the values of that page are decoded and handed to the builder. Both are
	// reused from page to page.
	levels []int32
	values *columnValues

	// dict is the dictionary of the chunk, read out of the page in front of the
	// data pages, and indices is where the rows of a dictionary encoded chunk
	// go. index is the indices of the page being read and is reused like levels.
	//
	// dict is nil until a dictionary page turns up and is what says which of the
	// two shapes a chunk is. out is the builder the rows go into and run appends
	// a run of the present ones, and the pair of them is the values builder for
	// a chunk of plain pages and the index builder for a dictionary encoded one.
	dict    *array.Array
	indices *array.Builder
	index   []int32
	out     *array.Builder
	run     func(from, count int)

	rle      RLEDecoder
	packed   BitPackedDecoder
	plain    PlainDecoder
	deltas   DeltaDecoder
	lengths  DeltaLengthDecoder
	prefixed DeltaByteArrayDecoder
}

// columnValues is how the values of one column are read and appended.
//
// The functions are built once for the column, in valuesFor, and close over the
// buffer the values are decoded into. That is what keeps the type of a
// column out of the loop over its pages: deciding that an int8 column is read
// as int32 and narrowed is a question about the schema, and asking it once per
// value would cost more than the narrowing.
type columnValues struct {
	// decode reads n values out of d into the buffer.
	decode func(d *PlainDecoder, n int) error

	// delta, deltaLength and deltaBytes read n values out of a page in one of
	// the three encodings that write differences, into the same buffer decode
	// fills. Each of them is nil for a column the format does not write that
	// encoding for, which is what makes such a page a file that contradicts its
	// own schema rather than one this package has not got round to.
	delta       func(d *DeltaDecoder, n int) error
	deltaLength func(d *DeltaLengthDecoder, n int) error
	deltaBytes  func(d *DeltaByteArrayDecoder, n int) error

	// run appends count values from the buffer, starting at from. They are all
	// present, since the nulls are put in by the caller.
	run func(from, count int)

	// take appends row k of a, which is the chunk's dictionary. It is how a
	// chunk that gives its dictionary up part way through puts back the rows it
	// had already read as indices, and it reads out of the dictionary rather
	// than out of the buffer because the buffer of a byte array column points
	// at the page it was decoded from and that page is long gone by then.
	take func(a *array.Array, k int)
}

// reads says whether a page in the encoding e is one this column's values can
// come out of.
//
// The three delta encodings each go with some of the physical types and not the
// others, and which of them a column has a reader for was settled by valuesFor
// out of the schema. Everything else is either plain or a matter for the
// dictionary, and both of those are somebody else's question.
func (v *columnValues) reads(e Encoding) bool {
	switch e {
	case DeltaBinaryPacked:
		return v.delta != nil
	case DeltaLengthByteArray:
		return v.deltaLength != nil
	case DeltaByteArray:
		return v.deltaBytes != nil
	default:
		return true
	}
}

// NewColumnReader returns a reader for the column c.
func NewColumnReader(c Column) (*ColumnReader, error) {
	if c.MaxRepetition > 0 {
		return nil, fmt.Errorf("parquet: %w: %s is repeated and lists are not read yet",
			ErrUnsupported, c.Name())
	}

	// A definition level counts the optional groups a column sits in, so it is a
	// small number in every file there is and the width it is written at follows
	// from it. Metadata.Columns is where one comes from and it counts groups it
	// has read, but a Column is a struct a caller can fill in, and a depth the
	// levels of a page could not hold is worth refusing once here rather than
	// once per page.
	if c.MaxDefinition < 0 || c.MaxDefinition > math.MaxInt32 {
		return nil, fmt.Errorf("parquet: %w: %s is %d levels deep",
			ErrFormat, c.Name(), c.MaxDefinition)
	}

	var indices *array.Builder
	b, err := array.NewBuilder(c.Type)
	if err == nil {
		// Where the rows of a dictionary encoded chunk go. The format has one
		// index type whatever the values are, so this is made along with the
		// values rather than when a dictionary page turns up, and the one check
		// that the column has a type kuma can build covers both.
		indices, err = array.NewBuilder(dtype.Int32)
	}
	if err != nil {
		return nil, fmt.Errorf("parquet: %s: %w", c.Name(), err)
	}

	values, err := valuesFor(c, b)
	if err != nil {
		return nil, err
	}

	return &ColumnReader{
		column:        c,
		builder:       b,
		maxDefinition: int32(c.MaxDefinition),
		width:         uint(bits.Len(uint(c.MaxDefinition))),
		values:        values,
		indices:       indices,
		out:           b,
		run:           values.run,
	}, nil
}

// DType returns the type of the values of the column being read.
//
// A chunk written as indices into a dictionary comes back dictionary encoded,
// so what Finish hands back for one of those is a dictionary of this type
// rather than this type. Which of the two shapes a chunk has is not known until
// its pages have been read, since a chunk that fills its dictionary and writes
// the rest of itself plainly comes back as this type after all.
func (r *ColumnReader) DType() dtype.DataType { return r.column.Type }

// Len returns how many values have been assembled, nulls included.
func (r *ColumnReader) Len() int { return r.out.Len() }

// Finish returns the values assembled so far and leaves the reader ready for
// another chunk of the same column.
//
// A chunk that was written as indices into a dictionary comes back dictionary
// encoded rather than expanded, so a column of a million rows holding two
// hundred distinct strings is a million indices and two hundred strings. That
// is the shape it was written in and the shape the kernels would rather have
// it in, and expanding it would be undoing the one thing the encoding is for.
//
// The exception is a chunk that gave its dictionary up part way through, which
// expand has already turned back into values by the time this is reached.
func (r *ColumnReader) Finish() (*array.Array, error) {
	if r.dict == nil {
		return r.builder.Finish(), nil
	}

	dict := r.dict
	r.dict, r.out, r.run = nil, r.builder, r.values.run

	// The indices are checked against the dictionary here rather than as they
	// are decoded, since it is the same walk either way and this is the one
	// that knows what to say about it. An index that is not in the dictionary
	// is a read out of range in whatever touches the column next.
	a, err := array.NewDictionary(r.indices.Finish(), dict)
	if err != nil {
		return nil, fmt.Errorf("parquet: %w: %s: %w", ErrFormat, r.column.Name(), err)
	}
	return a, nil
}

// Page assembles one page.
//
// The body is the page as it sits in the file with whatever compression the
// chunk used already undone, which is what Chunk uses a Decompressor for. The
// levels are still in front of the values, since where they are depends on
// which version of the data page it is and that is this function's business
// rather than its caller's.
func (r *ColumnReader) Page(p Page) error {
	switch p.Kind {
	case DataPage, DataPageV2:
	case DictionaryPage:
		return r.dictionary(p)
	default:
		// A page type this package has never heard of. The walk knows to step
		// over one and there is nothing in it for a column.
		return nil
	}

	if err := r.encoding(p); err != nil {
		return err
	}
	if p.NumValues < 0 {
		return fmt.Errorf("parquet: %w: a page of %s holding %d values",
			ErrFormat, r.column.Name(), p.NumValues)
	}

	body, err := r.definitions(p)
	if err != nil {
		return err
	}

	count, present := int(p.NumValues), int(p.NumValues)
	if r.maxDefinition > 0 {
		present = 0
		for _, level := range r.levels {
			if level > r.maxDefinition {
				return fmt.Errorf("parquet: %w: a level of %d in %s, whose deepest is %d",
					ErrFormat, level, r.column.Name(), r.maxDefinition)
			}
			if level == r.maxDefinition {
				present++
			}
		}

		// The second version of the data page writes the null count down. It is
		// the one thing in a page that says what the levels should have come to,
		// so a page that disagrees with itself is a page that was read wrong.
		if p.Kind == DataPageV2 && int32(count-present) != p.NumNulls {
			return fmt.Errorf("parquet: %w: a page of %s says it has %d nulls and its levels have %d",
				ErrFormat, r.column.Name(), p.NumNulls, count-present)
		}
	}

	if err := r.decodeValues(p, body, present); err != nil {
		return err
	}
	r.assemble(count, present)
	return nil
}

// decodeValues reads the values of a data page into the buffer the assembly
// takes them from.
//
// Which of them it is has already been settled by encoding, so this is a switch
// over what that let through rather than a second look at the page: a chunk with
// a dictionary in front of it is indices whatever else the page says, and
// everything that is not a difference is a value written as it is.
func (r *ColumnReader) decodeValues(p Page, body []byte, present int) error {
	if r.dict != nil {
		return r.decodeIndices(body, present)
	}
	if len(body) == 0 && present == 0 {
		// A page in which every row is missing. The delta encodings write a
		// header even when there are no values under it, and a writer that
		// leaves it out has said the same thing in the levels already. A plain
		// page of no values is no bytes either way.
		return nil
	}

	var err error
	switch p.Encoding {
	case DeltaBinaryPacked:
		if err = r.deltas.Reset(body); err == nil {
			err = r.values.delta(&r.deltas, present)
		}

	case DeltaLengthByteArray:
		if err = r.lengths.Reset(body); err == nil {
			err = r.values.deltaLength(&r.lengths, present)
		}

	case DeltaByteArray:
		if err = r.prefixed.Reset(body); err == nil {
			err = r.values.deltaBytes(&r.prefixed, present)
		}

	default:
		r.plain.Reset(body)
		err = r.values.decode(&r.plain, present)
	}

	if err != nil {
		return fmt.Errorf("parquet: %s: %w", r.column.Name(), err)
	}
	return nil
}

// definitions decodes the definition levels of a page into r.levels and returns
// the rest of the body, which is the values.
//
// The two versions of the data page keep their levels in different places. The
// second puts them in bytes of their own at the front of the body and says in
// the header how long they are. The first puts them inside the body behind four
// bytes of their length, or behind nothing at all when they are written in the
// encoding the format deprecated, which has no length because how many bytes it
// takes follows from how many values the page holds.
func (r *ColumnReader) definitions(p Page) ([]byte, error) {
	if r.maxDefinition == 0 {
		// A required column writes no levels, and neither version of the page
		// leaves room for them. Every value is present, so there is nothing to
		// decode and nothing for the assembly to walk.
		r.levels = r.levels[:0]
		return p.Data, nil
	}

	body := p.Data
	if p.Kind == DataPageV2 {
		n := int(p.RepetitionLength) + int(p.DefinitionLength)
		if p.RepetitionLength < 0 || p.DefinitionLength < 0 || n > len(body) {
			return nil, fmt.Errorf("parquet: %w: a page of %s with %d bytes of levels in %d",
				ErrFormat, r.column.Name(), n, len(body))
		}
		return body[n:], r.decodeLevels(p, body[p.RepetitionLength:n])
	}

	switch p.DefinitionEncoding {
	case RLE:
		if len(body) < 4 {
			return nil, fmt.Errorf("parquet: %w: a page of %s with %d bytes and no level length",
				ErrFormat, r.column.Name(), len(body))
		}
		n := int64(binary.LittleEndian.Uint32(body))
		if n > int64(len(body)-4) {
			return nil, fmt.Errorf("parquet: %w: %d bytes of levels of %s in a page of %d",
				ErrFormat, n, r.column.Name(), len(body)-4)
		}
		return body[4+n:], r.decodeLevels(p, body[4:4+n])

	case BitPacked:
		// The deprecated encoding writes no length. It packs every level of the
		// page end to end and pads to a byte, so how long it is follows from
		// how many values the page holds.
		n := (int(p.NumValues)*int(r.width) + 7) / 8
		if n > len(body) {
			return nil, fmt.Errorf("parquet: %w: %d bytes of packed levels of %s in a page of %d",
				ErrFormat, n, r.column.Name(), len(body))
		}
		return body[n:], r.decodeLevels(p, body[:n])

	default:
		return nil, fmt.Errorf("parquet: %w: the levels of %s are %s",
			ErrUnsupported, r.column.Name(), p.DefinitionEncoding)
	}
}

// decodeLevels reads exactly as many levels as the page holds values.
func (r *ColumnReader) decodeLevels(p Page, data []byte) error {
	r.levels = grow(r.levels, int(p.NumValues))

	// The width is the column's and was checked when the reader was made, so
	// neither of these can refuse it.
	var read func([]int32) (int, error)
	if p.DefinitionEncoding == BitPacked {
		r.packed.reset(data, r.width)
		read = r.packed.Read
	} else {
		r.rle.reset(data, r.width)
		read = r.rle.Read
	}

	for n := 0; n < len(r.levels); {
		got, err := read(r.levels[n:])
		n += got
		if err != nil {
			return fmt.Errorf("parquet: the levels of %s: %w", r.column.Name(), err)
		}
	}
	return nil
}

// assemble walks the levels and the values together, appending each run of
// present values and each run of nulls in one call.
//
// Runs rather than values because that is the shape the data comes in. A column
// with nothing missing is one run, a column that is all missing is one run of
// nulls, and a column that alternates is the worst case and is rare enough that
// the loop is not worth unrolling for it.
func (r *ColumnReader) assemble(count, present int) {
	r.out.Grow(count)
	if r.maxDefinition == 0 {
		r.run(0, present)
		return
	}

	taken := 0
	for i := 0; i < len(r.levels); {
		j := i
		if r.levels[i] == r.maxDefinition {
			for j < len(r.levels) && r.levels[j] == r.maxDefinition {
				j++
			}
			r.run(taken, j-i)
			taken += j - i
		} else {
			for j < len(r.levels) && r.levels[j] != r.maxDefinition {
				j++
			}
			r.out.AppendNulls(j - i)
		}
		i = j
	}
}

// ReadColumn reads one column chunk of a file into an array.
//
// The size is the size of the file, the same one ReadMetadata was given, and c
// is the column the chunk holds, which is one of the leaves Metadata.Columns
// returned. A chunk written as indices into a dictionary comes back dictionary
// encoded, which is what ColumnReader.Finish says more about.
//
// This is one chunk read with a reader of its own. Something walking the row
// groups of a file reads the same column again and again and wants one reader
// for all of them, which is what ColumnReader.Chunk is.
func ReadColumn(r io.ReaderAt, size int64, chunk *ColumnChunk, c Column) (*array.Array, error) {
	reader, err := NewColumnReader(c)
	if err != nil {
		return nil, err
	}
	return reader.Chunk(r, size, chunk)
}

// Chunk reads one column chunk of a file into an array.
//
// The size is the size of the file, the same one ReadMetadata was given, and
// the chunk has to be one of this column's. A reader is made for one column and
// knows what that column's pages should hold, so handing it another one's chunk
// is a way of reading the wrong values without being told.
//
// A reader is good for one chunk after another, which is what makes it worth
// keeping while a scan walks the row groups of a file: the builder and the
// buffers a column is assembled in are made once rather than once per row
// group. A reader that returned an error stopped somewhere inside a chunk and
// is holding half a column, so it is not worth handing another one.
//
// The pages have their compression undone on the way through, which is what a
// Decompressor is for. The codec is a property of the chunk, so a column
// compressed one way in one row group and another way in the next is read the
// way each of them was written.
func (r *ColumnReader) Chunk(src io.ReaderAt, size int64, chunk *ColumnChunk) (*array.Array, error) {
	codec, err := NewDecompressor(chunk.Meta.Codec)
	if err != nil {
		return nil, fmt.Errorf("parquet: %s: %w", r.column.Name(), err)
	}

	pages, err := ReadPages(src, size, chunk)
	if err != nil {
		return nil, err
	}
	for {
		p, err := pages.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		if p, err = codec.Page(p); err != nil {
			return nil, fmt.Errorf("parquet: %s: %w", r.column.Name(), err)
		}
		if err := r.Page(p); err != nil {
			return nil, err
		}
	}

	if int64(r.Len()) != chunk.Meta.NumValues {
		return nil, fmt.Errorf("parquet: %w: the chunk for %s says it has %d values and its pages hold %d",
			ErrFormat, r.column.Name(), chunk.Meta.NumValues, r.Len())
	}
	return r.Finish()
}

// valuesFor decides how a column's values are read and appended.
//
// The physical type says how they are written and the kuma type says how they
// are stored, and the two are not the same question. Parquet writes every
// integer of thirty two bits or fewer as an int32, so six of kuma's types come
// out of the same read and are told apart by the schema. Nothing here looks at
// a value to decide what it is.
func valuesFor(c Column, b *array.Builder) (*columnValues, error) {
	switch c.Type.Kind() {
	case dtype.NullKind:
		// A column the writer knew nothing about, which in practice is a column
		// whose values are all missing. Whatever physical type it was written as
		// means nothing, so the bytes of the page are not read at all and a row
		// is a null however its level reads.
		return &columnValues{
			decode: func(*PlainDecoder, int) error { return nil },
			run:    func(_, count int) { b.AppendNulls(count) },
			take:   func(*array.Array, int) { b.AppendNull() },
		}, nil

	case dtype.BoolKind:
		return booleans(b), nil

	case dtype.Int8Kind:
		return narrowed[int32, int8](b, (*PlainDecoder).Int32), nil
	case dtype.Int16Kind:
		return narrowed[int32, int16](b, (*PlainDecoder).Int32), nil
	case dtype.Uint8Kind:
		return narrowed[int32, uint8](b, (*PlainDecoder).Int32), nil
	case dtype.Uint16Kind:
		return narrowed[int32, uint16](b, (*PlainDecoder).Int32), nil
	case dtype.Uint32Kind:
		return narrowed[int32, uint32](b, (*PlainDecoder).Int32), nil
	case dtype.Uint64Kind:
		return narrowed[int64, uint64](b, (*PlainDecoder).Int64), nil

	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		return integers(b, (*PlainDecoder).Int32), nil
	case dtype.Int64Kind, dtype.Date64Kind, dtype.Time64Kind, dtype.DurationKind:
		return integers(b, (*PlainDecoder).Int64), nil
	case dtype.Float32Kind:
		return numbers(b, (*PlainDecoder).Float), nil
	case dtype.Float64Kind:
		return numbers(b, (*PlainDecoder).Double), nil

	case dtype.TimestampKind:
		// The one type that is written two ways. An int96 is the timestamp
		// parquet had before it had one, and it is turned into nanoseconds by
		// the decoder, so from here on it is an int64 like any other.
		if c.Element.Type == Int96 {
			return numbers(b, (*PlainDecoder).Int96), nil
		}
		return integers(b, (*PlainDecoder).Int64), nil

	case dtype.StringKind, dtype.BinaryKind:
		return blobs(b), nil

	case dtype.FixedSizeBinaryKind:
		return fixedBlobs(b, int(c.Element.TypeLength)), nil

	default:
		// The decimals and the intervals are the rest of the flat types and they
		// are written as bytes of a wide integer, big endian and two's
		// complement, which is a conversion of its own. The types that are not
		// flat need a repeated column first.
		return nil, fmt.Errorf("parquet: %w: %s is a %s and that is not assembled yet",
			ErrUnsupported, c.Name(), c.Type)
	}
}

// numbers reads a column whose values are stored at the width the file wrote
// them at, which is most of them.
func numbers[T array.Numeric](b *array.Builder, read func(*PlainDecoder, []T) (int, error)) *columnValues {
	var buf []T
	return &columnValues{
		decode: func(d *PlainDecoder, n int) error {
			buf = grow(buf, n)
			got, err := read(d, buf)
			return exactly(got, n, err)
		},
		run:  func(from, count int) { b.AppendValues(buf[from : from+count]) },
		take: func(a *array.Array, k int) { b.Append(a.Value[T](k)) },
	}
}

// integers is numbers for a column that can also arrive as differences, which
// is every column parquet writes as an int32 or an int64 and no other. The two
// are apart rather than one function with a flag because a delta encoded page
// of doubles is a file to refuse rather than a page to read badly, and the
// cheapest way to say so is to have nothing to read it with.
func integers[T deltaValue](b *array.Builder, read func(*PlainDecoder, []T) (int, error)) *columnValues {
	var buf []T
	return &columnValues{
		decode: func(d *PlainDecoder, n int) error {
			buf = grow(buf, n)
			got, err := read(d, buf)
			return exactly(got, n, err)
		},
		delta: func(d *DeltaDecoder, n int) error {
			buf = grow(buf, n)
			got, err := d.Read(buf)
			return exactly(got, n, err)
		},
		run:  func(from, count int) { b.AppendValues(buf[from : from+count]) },
		take: func(a *array.Array, k int) { b.Append(a.Value[T](k)) },
	}
}

// narrowed reads a column that parquet wrote wider than kuma stores it, which
// is every integer of fewer than thirty two bits and the two unsigned types
// that share a width with a signed one.
//
// The conversion cannot lose anything. The file said the column is an int8 and
// the values in it are int8 values written in four bytes, so a value that does
// not fit is a file that contradicts its own schema, and the value would have
// been wrong however it was read.
func narrowed[W deltaValue, T array.Numeric](b *array.Builder, read func(*PlainDecoder, []W) (int, error)) *columnValues {
	var wide []W
	var buf []T
	narrow := func(got, n int, readErr error) error {
		if err := exactly(got, n, readErr); err != nil {
			return err
		}
		buf = grow(buf, n)
		for i, v := range wide {
			buf[i] = T(v)
		}
		return nil
	}
	return &columnValues{
		decode: func(d *PlainDecoder, n int) error {
			wide = grow(wide, n)
			got, err := read(d, wide)
			return narrow(got, n, err)
		},
		delta: func(d *DeltaDecoder, n int) error {
			wide = grow(wide, n)
			got, err := d.Read(wide)
			return narrow(got, n, err)
		},
		run:  func(from, count int) { b.AppendValues(buf[from : from+count]) },
		take: func(a *array.Array, k int) { b.Append(a.Value[T](k)) },
	}
}

// booleans reads a column of one bit values.
func booleans(b *array.Builder) *columnValues {
	var buf []bool
	return &columnValues{
		decode: func(d *PlainDecoder, n int) error {
			buf = grow(buf, n)
			got, err := d.Boolean(buf)
			return exactly(got, n, err)
		},
		run:  func(from, count int) { b.AppendBools(buf[from : from+count]) },
		take: func(a *array.Array, k int) { b.AppendBool(a.Bool(k)) },
	}
}

// blobs reads a column of byte arrays that carry their own length, which is
// what a string or a binary column is written as.
func blobs(b *array.Builder) *columnValues {
	return blobsOf(b, func(d *PlainDecoder, dst [][]byte) (int, error) { return d.ByteArray(dst) })
}

// fixedBlobs reads a column of byte arrays whose width is in the schema rather
// than in front of every value.
func fixedBlobs(b *array.Builder, width int) *columnValues {
	v := blobsOf(b, func(d *PlainDecoder, dst [][]byte) (int, error) { return d.Fixed(dst, width) })

	// A column whose values are all the same width has no lengths worth
	// writing, so the encoding that takes the lengths out and writes them as
	// differences is not defined for it. The one that shares a prefix between
	// neighbors is, since a fixed width column is as likely to be sorted keys
	// as any other.
	v.deltaLength = nil
	return v
}

// blobsOf reads a column of byte arrays however they are written down.
//
// The values point into the page they were decoded from, or into the buffer the
// prefixed encoding stitched them in, and the builder copies them either way.
// That is what makes it safe for both of those to be reused by the next page.
func blobsOf(b *array.Builder, read func(*PlainDecoder, [][]byte) (int, error)) *columnValues {
	var buf [][]byte
	return &columnValues{
		decode: func(d *PlainDecoder, n int) error {
			buf = grow(buf, n)
			got, err := read(d, buf)
			return exactly(got, n, err)
		},
		deltaLength: func(d *DeltaLengthDecoder, n int) error {
			buf = grow(buf, n)
			got, err := d.Read(buf)
			return exactly(got, n, err)
		},
		deltaBytes: func(d *DeltaByteArrayDecoder, n int) error {
			buf = grow(buf, n)
			got, err := d.Read(buf)
			return exactly(got, n, err)
		},
		run: func(from, count int) {
			for _, v := range buf[from : from+count] {
				b.AppendBytes(v)
			}
		},
		take: func(a *array.Array, k int) { b.AppendBytes(a.Bytes(k)) },
	}
}

// exactly turns what a decoder returned into an error unless it read every
// value the page said it held. A decoder that filled the buffer returns no
// error at all, so a short read is the only thing there is to report.
//
// A page holds what its header says and a decoder stops when the bytes run out,
// so the two disagreeing means the page is shorter than it claims. Reading on
// would take the next page's bytes as this page's values.
func exactly(got, want int, err error) error {
	if got == want {
		return nil
	}
	if err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	return fmt.Errorf("%w: a page of %d values holds %d", ErrFormat, want, got)
}

// grow returns a slice of exactly n values, reusing s when it is big enough.
func grow[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n)
	}
	return s[:n]
}
