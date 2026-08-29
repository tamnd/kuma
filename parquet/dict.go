package parquet

import "fmt"

// Reading a column that was written as a dictionary.
//
// Most of a real parquet file is dictionary encoded, because most columns
// repeat themselves. A chunk of ten million country codes holds two hundred and
// fifty strings once, in a page of its own in front of the data pages, and the
// rows are indices into that page. The writer only gives up on it when the
// distinct values stop fitting in a page of their own.
//
// So the column comes back dictionary encoded rather than expanded. kuma has
// that shape already and it is the shape the file is in, so handing back ten
// million strings would mean copying the same two hundred and fifty of them
// forty thousand times each, and then every kernel that touched the column
// would work on ten million strings instead of ten million integers. The one
// thing worth doing with a dictionary on the way in is nothing.
//
// A chunk whose writer gave the dictionary up part way through is the one place
// that does not hold. It is indices at the front and plain values at the back,
// and a column is one shape or the other, so the rows read as indices are put
// back as values when the first plain page turns up and the rest of the chunk
// is read the way any plain chunk is. The work lands on the chunks that fall
// back and on no others, which is the trade the format leaves: reading such a
// chunk costs a gather, and refusing it costs the file.
//
// The indices are the RLE hybrid again, the same encoding as the levels, at a
// width the writer chose and wrote in the first byte of the page rather than
// one that follows from the schema. A width of nought is a column with one
// distinct value in it, which is a real width and reads as a run of noughts.

// dictionary reads the dictionary page of a chunk.
//
// There is one of them and it comes first, which is what makes it worth reading
// through the same decoder as a plain page: a dictionary page is the distinct
// values written plainly, so the column's own decoder reads it, and everything
// the schema decided about narrowing and conversions holds for it too.
func (r *ColumnReader) dictionary(p Page) error {
	switch {
	case r.dict != nil:
		return fmt.Errorf("parquet: %w: two dictionary pages for %s",
			ErrFormat, r.column.Name())

	case r.out.Len() > 0:
		// The indices of a page mean nothing until the values they index have
		// been read, so a dictionary behind them is a chunk that cannot be read
		// in one pass rather than one whose pages arrived in an odd order.
		return fmt.Errorf("parquet: %w: a dictionary page of %s behind its data pages",
			ErrFormat, r.column.Name())

	case p.Encoding != Plain && p.Encoding != PlainDictionary:
		// A dictionary page is plain values, and the older files say so with
		// the name the format used before it moved that name to the data pages.
		return fmt.Errorf("parquet: %w: a %s dictionary page of %s",
			ErrUnsupported, p.Encoding, r.column.Name())

	case p.NumValues < 0:
		return fmt.Errorf("parquet: %w: a dictionary page of %s holding %d values",
			ErrFormat, r.column.Name(), p.NumValues)
	}

	n := int(p.NumValues)
	r.plain.Reset(p.Data)
	if err := r.values.decode(&r.plain, n); err != nil {
		return fmt.Errorf("parquet: the dictionary of %s: %w", r.column.Name(), err)
	}

	r.builder.Grow(n)
	r.values.run(0, n)
	r.dict = r.builder.Finish()
	r.out, r.run = r.indices, r.appendIndices
	return nil
}

// encoding decides which shape a data page is in and refuses the ones that are
// none of them.
//
// It is the page that says whether the chunk is dictionary encoded rather than
// the chunk, since the encodings a chunk lists are the ones it used somewhere. A
// writer that fills its dictionary gives up on it and writes the rest of the
// chunk the way it would have written all of it, which leaves a chunk that is
// indices at the front and values at the back. Both of the writers most files
// come from do that, so it is a shape to read rather than one to refuse, and
// what it takes is expanding the dictionary into the rows already read.
func (r *ColumnReader) encoding(p Page) error {
	switch p.Encoding {
	case Plain, DeltaBinaryPacked, DeltaLengthByteArray, DeltaByteArray:
		if r.dict != nil {
			if err := r.expand(); err != nil {
				return err
			}
		}
		if !r.values.reads(p.Encoding) {
			// Each of the delta encodings goes with some of the physical types
			// and no others: differences with the integers, lengths with the
			// byte arrays that have their own, shared prefixes with those and
			// the fixed width ones. A page that says otherwise contradicts the
			// schema in front of it rather than being one this package has not
			// got round to.
			return fmt.Errorf("parquet: %w: a %s page of %s, which is a %s",
				ErrFormat, p.Encoding, r.column.Name(), r.column.Type)
		}
		return nil

	case PlainDictionary, RLEDictionary:
		if r.dict == nil {
			return fmt.Errorf("parquet: %w: a page of %s indexes a dictionary the chunk has not got",
				ErrFormat, r.column.Name())
		}
		return nil

	default:
		return fmt.Errorf("parquet: %w: a %s page of %s and only plain, dictionary and delta pages are read yet",
			ErrUnsupported, p.Encoding, r.column.Name())
	}
}

// expand puts the rows read as indices back as values, which is what a chunk
// that gives its dictionary up leaves no choice about.
//
// It is the one thing reading a dictionary is meant to avoid and there is
// nothing else to do with such a chunk: the rows behind the fallback are values
// and the ones in front of it are indices, and a column is one or the other.
// The cost lands on the chunks that fall back and on no others, and it is a
// gather of the rows already read rather than of the whole chunk, since
// everything after this point is read as values in the first place.
//
// The indices are checked here rather than left to the dictionary array that is
// never now built, so an index the file has no value for is the same error
// whichever shape the chunk turns out to have.
func (r *ColumnReader) expand() error {
	dict, rows := r.dict, r.indices.Finish()
	r.dict, r.out, r.run = nil, r.builder, r.values.run

	at := rows.Values[int32]()
	for k := range rows.Len() {
		if rows.IsNull(k) {
			r.builder.AppendNull()
			continue
		}
		if at[k] < 0 || int(at[k]) >= dict.Len() {
			return fmt.Errorf("parquet: %w: a row of %s indexes %d of a dictionary of %d",
				ErrFormat, r.column.Name(), at[k], dict.Len())
		}
		r.values.take(dict, int(at[k]))
	}
	return nil
}

// decodeIndices reads the indices of one data page into r.index.
//
// There is one of them per row that has a value, and the nulls are not written
// down here any more than the values of a plain page are. The width is the first
// byte of the body and the rest is one run of the hybrid encoding, so a page
// with no bytes at all is a page in which every row is missing.
func (r *ColumnReader) decodeIndices(body []byte, present int) error {
	if len(body) == 0 {
		if present == 0 {
			return nil
		}
		return fmt.Errorf("parquet: %w: a page of %s wants %d indices and holds no bytes",
			ErrFormat, r.column.Name(), present)
	}

	if err := r.rle.Reset(body[1:], int(body[0])); err != nil {
		return fmt.Errorf("parquet: the indices of %s: %w", r.column.Name(), err)
	}

	r.index = grow(r.index, present)
	for n := 0; n < present; {
		got, readErr := r.rle.Read(r.index[n:])
		n += got
		if readErr != nil {
			return fmt.Errorf("parquet: the indices of %s: %w",
				r.column.Name(), exactly(n, present, readErr))
		}
	}
	return nil
}

// appendIndices appends a run of the indices of the page being read, which is
// what takes the place of appending a run of values when the chunk is
// dictionary encoded.
func (r *ColumnReader) appendIndices(from, count int) {
	r.indices.AppendValues(r.index[from : from+count])
}
