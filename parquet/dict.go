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
// indices at the front and values at the back. Reading that would mean expanding
// the dictionary into the column, which is the one thing reading it this way is
// for, so it is refused by name rather than done quietly.
func (r *ColumnReader) encoding(p Page) error {
	switch p.Encoding {
	case Plain, DeltaBinaryPacked:
		if r.dict != nil {
			return fmt.Errorf("parquet: %w: %s falls back from its dictionary to %s pages",
				ErrUnsupported, r.column.Name(), p.Encoding)
		}
		if p.Encoding == DeltaBinaryPacked && r.values.delta == nil {
			// The encoding is written for the two integer widths and nothing
			// else, so this is a page that contradicts the schema in front of
			// it rather than one this package has not got round to.
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
