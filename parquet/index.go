package parquet

import (
	"fmt"
	"io"
	"strings"
)

// The two indexes that say what is in each page of a column chunk.
//
// Bounds skips a row group. A row group is as many rows as its writer thought
// fit in memory at once, which in a real file is a million, so skipping one is
// worth a lot and reading one that turned out to hold nothing wanted is worth
// nothing. The page index is the same idea one level down: a pair of structures
// at the end of the file holding the smallest and largest value of every page
// and where every page starts, so a scan that has to read a row group can still
// read three pages of it.
//
// They are two structures rather than one because they are read at different
// times. The column index is what a filter looks at, and a filter that keeps
// nothing needs nothing else. The offset index is what tells the reader where to
// go once the filter has picked its pages, and a scan that skips the whole group
// never reads it. Writing them apart is what lets a reader read them apart, and
// keeping the bounds of every page out of the footer is what keeps the footer
// small in a file of ten thousand pages.
//
// Neither is in the footer, so reading them costs a read of the file. That is
// what makes them worth asking for per column rather than up front: a scan
// filtering on one column of two hundred reads the index of that one.
//
// What is here is reading them and saying what they hold. Reading only the pages
// a filter kept is the next thing and needs the assembler to start in the middle
// of a chunk, which it cannot do yet.

// PageLocation is where one page of a column chunk is.
type PageLocation struct {
	// Offset is where the page starts in the file, and CompressedSize is how
	// long it is, header and all, as it sits in the file.
	Offset         int64
	CompressedSize int32

	// FirstRow is the row the page starts at, counted from the first row of the
	// row group rather than of the file.
	FirstRow int64
}

// OffsetIndex is where every page of one column chunk is.
type OffsetIndex struct {
	// Pages are the locations, in the order the pages are written, which is the
	// order their rows are in.
	Pages []PageLocation
}

// ColumnIndex is what a writer said about the values of every page of one
// column chunk.
//
// The three lists are parallel and are as long as the chunk has pages. The
// values are the raw bytes of the physical type, the same as in Statistics, and
// PageBounds is what decodes them.
type ColumnIndex struct {
	// NullPages says, per page, that every value in it is missing. The format
	// leaves the bounds of such a page undefined, so this is what has to be
	// read before them.
	NullPages []bool

	// Min and Max are the smallest and largest value of each page, in the order
	// the format defines for the column's type. There is no saying whether they
	// are values out of the page or values either side of it, which Statistics
	// does say, so a bound out of here is treated as inexact.
	Min [][]byte
	Max [][]byte

	// Order is whether the pages themselves are in order.
	Order BoundaryOrder

	// NullCounts is how many values of each page are missing, and is empty in a
	// file that did not count them.
	NullCounts []int64
}

// PageBounds is what the two indexes together say about one page.
//
// The Bounds are what the page holds, the same shape ReadBounds gives for a
// whole chunk, with Count being the rows of the page rather than of the group.
// The rest is where the page is, so a scan that has decided to read it can.
type PageBounds struct {
	Bounds

	Offset         int64
	CompressedSize int32
	FirstRow       int64
}

// ReadColumnIndex reads the bounds of every page of a column chunk.
//
// The size is the size of the file, the same one ReadMetadata was given, since
// the offset in the chunk is a number out of a footer and a claim to be past the
// end of the file must not turn into a read of that size.
//
// It comes back nil when the writer wrote no column index, which is most files
// written before the format had one and any file whose writer was not asked for
// it. That is not an error: a scan without a page index skips row groups and
// reads all of the ones it keeps, which is where the reader was before.
func ReadColumnIndex(r io.ReaderAt, size int64, c *ColumnChunk) (*ColumnIndex, error) {
	name := strings.Join(c.Meta.Path, ".")
	buf, err := chunkBytes(r, size, c.ColumnIndexOffset, int64(c.ColumnIndexLength), "column index", name)
	if buf == nil || err != nil {
		return nil, err
	}

	x := &ColumnIndex{}
	if err := x.read(&reader{buf: buf}); err != nil {
		return nil, fmt.Errorf("parquet: the column index of %s: %w", name, err)
	}
	if len(x.Min) != len(x.NullPages) || len(x.Max) != len(x.NullPages) {
		return nil, fmt.Errorf("parquet: %w: the column index of %s says %d pages, %d smallest values and %d largest",
			ErrFormat, name, len(x.NullPages), len(x.Min), len(x.Max))
	}
	if len(x.NullCounts) != 0 && len(x.NullCounts) != len(x.NullPages) {
		return nil, fmt.Errorf("parquet: %w: the column index of %s says %d pages and counts the missing values of %d",
			ErrFormat, name, len(x.NullPages), len(x.NullCounts))
	}
	return x, nil
}

// ReadOffsetIndex reads where every page of a column chunk is.
//
// It comes back nil when the writer wrote no offset index, on the same terms as
// ReadColumnIndex. A writer that wrote one of the two wrote both, but they are
// read one at a time because a filter that keeps no page of a chunk has no use
// for where those pages are.
func ReadOffsetIndex(r io.ReaderAt, size int64, c *ColumnChunk) (*OffsetIndex, error) {
	name := strings.Join(c.Meta.Path, ".")
	buf, err := chunkBytes(r, size, c.OffsetIndexOffset, int64(c.OffsetIndexLength), "offset index", name)
	if buf == nil || err != nil {
		return nil, err
	}

	o := &OffsetIndex{}
	if err := o.read(&reader{buf: buf}); err != nil {
		return nil, fmt.Errorf("parquet: the offset index of %s: %w", name, err)
	}
	return o, nil
}

// ReadPageBounds reads both indexes of a column chunk and decodes what they say
// about each of its pages.
//
// The rows are how many the row group holds, which is what the last page runs
// to. The index says where each page starts and never how long it is, so the
// length of a page is where the next one starts and the length of the last one
// is everything left of the group.
//
// It comes back nil when the writer wrote no page index, which is what a caller
// falls back to ReadBounds on. A chunk with a column index and no offset index
// is a footer contradicting itself rather than a chunk without an index, since
// no writer produces one without the other.
func ReadPageBounds(r io.ReaderAt, size int64, chunk *ColumnChunk, c Column, rows int64) ([]PageBounds, error) {
	x, err := ReadColumnIndex(r, size, chunk)
	if x == nil || err != nil {
		return nil, err
	}
	o, err := ReadOffsetIndex(r, size, chunk)
	if err != nil {
		return nil, err
	}
	if o == nil {
		return nil, fmt.Errorf("parquet: %w: %s has a column index and no offset index",
			ErrFormat, c.Name())
	}
	if len(o.Pages) != len(x.NullPages) {
		return nil, fmt.Errorf("parquet: %w: the index of %s bounds %d pages and locates %d",
			ErrFormat, c.Name(), len(x.NullPages), len(o.Pages))
	}
	if len(o.Pages) != 0 && o.Pages[0].FirstRow != 0 {
		return nil, fmt.Errorf("parquet: %w: the first page of %s starts at row %d of its row group",
			ErrFormat, c.Name(), o.Pages[0].FirstRow)
	}

	out := make([]PageBounds, len(o.Pages))
	for i := range o.Pages {
		p := &o.Pages[i]

		// A page runs to the start of the next one, and the last runs to the
		// end of the row group. Two pages in the wrong order would give a
		// negative length, and a page starting past the end of the group would
		// give one longer than the group.
		end := rows
		if i+1 < len(o.Pages) {
			end = o.Pages[i+1].FirstRow
		}
		if end < p.FirstRow || end > rows {
			return nil, fmt.Errorf("parquet: %w: page %d of %s holds the rows from %d to %d of a row group of %d",
				ErrFormat, i, c.Name(), p.FirstRow, end, rows)
		}

		b, err := x.bounds(&c, i, end-p.FirstRow)
		if err != nil {
			return nil, err
		}
		out[i] = PageBounds{
			Bounds:         b,
			Offset:         p.Offset,
			CompressedSize: p.CompressedSize,
			FirstRow:       p.FirstRow,
		}
	}
	return out, nil
}

// PageBounds returns what the writer said about each page of one projected
// column of one row group.
//
// The column is its place in the projection rather than in the file, the same
// as the entries Bounds hands back, so a reader projecting two columns is asked
// about page bounds one column at a time and not once for the pair. That is
// because reading them costs a read of the file: the indexes are not in the
// footer, and a scan filtering on one column has no reason to read the index of
// the other.
//
// It comes back nil when the writer wrote no page index for the chunk, and a
// caller that gets nil is where FileReader.Bounds left it, which is a row group
// it either reads or skips whole.
func (r *FileReader) PageBounds(group, column int) ([]PageBounds, error) {
	g, err := r.group(group)
	if err != nil {
		return nil, err
	}
	if column < 0 || column >= len(r.take) {
		return nil, fmt.Errorf("parquet: column %d of a projection of %d", column, len(r.take))
	}

	ch, c, err := r.chunkOf(g, group, column)
	if err != nil {
		return nil, err
	}
	return ReadPageBounds(r.src, r.size, ch, *c, g.NumRows)
}

// bounds decodes what the index said about page i, which holds the given number
// of rows.
//
// The rules are ReadBounds's rules with one pair of bounds instead of two. The
// column index came after the format settled how its types compare, so there is
// no older pair to fall back to and nothing to read at all on a column the file
// gave no order. What it does not carry is whether a bound is a value out of the
// page, so both are treated as inexact and a caller that wants the value itself
// reads the page.
func (x *ColumnIndex) bounds(c *Column, i int, rows int64) (Bounds, error) {
	b := Bounds{Count: rows}
	if len(x.NullCounts) != 0 {
		b.Nulls, b.HasNulls = x.NullCounts[i], true
		if b.Nulls < 0 || b.Nulls > rows {
			return Bounds{}, fmt.Errorf("parquet: %w: page %d of %s holds %d rows and says %d are missing",
				ErrFormat, i, c.Name(), rows, b.Nulls)
		}
	}

	// A page of nothing but values that are not there. The format leaves its
	// bounds undefined, so they are not read, and every filter skips it.
	if x.NullPages[i] {
		if b.HasNulls && b.Nulls != rows {
			return Bounds{}, fmt.Errorf("parquet: %w: page %d of %s is missing every value and counts %d of %d",
				ErrFormat, i, c.Name(), b.Nulls, rows)
		}
		b.Nulls, b.HasNulls = rows, true
		return b, nil
	}

	if !ordered(c) || c.Order != TypeDefinedOrder {
		return b, nil
	}
	lo, hi := x.Min[i], x.Max[i]
	if nan(c.Element.Type, lo) || nan(c.Element.Type, hi) {
		return b, nil
	}

	values, err := decodeBounds(c, lo, hi)
	if err != nil {
		return Bounds{}, err
	}
	b.Values = values
	return b, nil
}

// chunkBytes reads a run of bytes a chunk's metadata pointed at, which is one
// of the two indexes or a bloom filter.
//
// An offset of nought is a writer that wrote nothing there, since a parquet
// file starts with its magic and nothing else can live at the front of it.
// Everything else is a claim about a file this reader has not read, and is
// checked against the size before anything is allocated for it.
func chunkBytes(r io.ReaderAt, size, at, n int64, what, column string) ([]byte, error) {
	if at <= 0 || n <= 0 {
		return nil, nil
	}
	if at > size || n > size-at {
		return nil, fmt.Errorf("parquet: %w: the %s of %s is %d bytes at %d of a file of %d",
			ErrFormat, what, column, n, at, size)
	}

	buf := make([]byte, n)
	if _, err := r.ReadAt(buf, at); err != nil {
		return nil, fmt.Errorf("parquet: reading the %s of %s: %w", what, column, err)
	}
	return buf, nil
}

// read fills in the column index from the file.
func (x *ColumnIndex) read(r *reader) error {
	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			x.NullPages, err = flags(r, t)
		case 2:
			x.Min, err = binaries(r, t)
		case 3:
			x.Max, err = binaries(r, t)
		case 4:
			var v int32
			if v, err = r.int32(t); err == nil {
				x.Order = BoundaryOrder(v)
			}
		case 5:
			x.NullCounts, err = longs(r, t)
		default:
			// The two histograms of levels, which say how many of the values
			// of a page are at each level and are only worth anything on a
			// column of lists.
			err = r.skip(t)
		}
		return err
	})
}

// read fills in the offset index from the file.
func (o *OffsetIndex) read(r *reader) error {
	return r.fields(func(id int16, t thriftType) error {
		if id == 1 {
			var err error
			o.Pages, err = structs(r, t, (*PageLocation).read)
			return err
		}
		// The unencoded size of every page of a byte array column, which says
		// how much memory reading one would take rather than anything about
		// what is in it.
		return r.skip(t)
	})
}

// read fills in one page location from the file.
func (p *PageLocation) read(r *reader) error {
	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			p.Offset, err = r.integer(t)
		case 2:
			p.CompressedSize, err = r.int32(t)
		case 3:
			p.FirstRow, err = r.integer(t)
		default:
			err = r.skip(t)
		}
		return err
	})
}
