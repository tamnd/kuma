package parquet

import (
	"fmt"
	"hash/crc32"
	"io"
	"strings"
)

// Walking the pages of a column chunk.
//
// The footer says where a column chunk starts and how long it is, and nothing
// more. Inside it is a run of pages laid end to end, each one a Thrift header
// followed by a body, and the only way to find the second page is to read the
// first header and add up. So a chunk is walked rather than indexed, which is
// why a page header is the smallest thing in the format that has to be right:
// one wrong length and every page after it is garbage.
//
// A chunk is read in one go rather than a page at a time. A page header has no
// length in front of it, so reading one page at a time means reading a guess,
// parsing what came back, and going round again when the guess was short, and
// the thing being avoided by that is holding a chunk in memory when a scan is
// about to decode the whole chunk anyway. Projection is what keeps the reading
// small: a chunk that is not wanted is never read at all.
//
// The sizes in a header are somebody else's numbers, same as everything else in
// the footer, and they are what the walk is driven by. Every one of them is
// checked against what is left of the chunk before it is used, and a page that
// carries a checksum is checked against it.

// PageHeader is the header in front of a page.
//
// It is four structures in the format, one per page type, and one here. Which
// fields mean anything depends on Kind and is written next to each of them. The
// fields the format calls required have to be there: a data page that does not
// say how it is encoded is refused rather than read as though it were plain,
// since a page decoded with the wrong encoding is wrong data rather than an
// error.
type PageHeader struct {
	// Kind is what the page holds.
	Kind PageKind

	// CompressedSize is how many bytes the body takes in the file, which is
	// what the next page header sits behind. UncompressedSize is how many it
	// takes once the codec of the chunk has been undone, and the two are the
	// same when the chunk is not compressed.
	CompressedSize   int32
	UncompressedSize int32

	// CRC is a CRC32 of the body as it sits in the file, and HasCRC says
	// whether the writer wrote one. Most do not.
	CRC    int32
	HasCRC bool

	// NumValues is how many values the page holds, nulls included, and for a
	// dictionary page how many entries the dictionary has.
	NumValues int32

	// Encoding is how the values are encoded. A dictionary page is plain or
	// plain dictionary, and a data page that leans on the dictionary says so
	// here rather than in the chunk.
	Encoding Encoding

	// DefinitionEncoding and RepetitionEncoding are how the levels in front of
	// the values are encoded. The second version of the data page does not
	// write them because the format fixes both at RLE, so they are filled in
	// here either way and a decoder does not have to ask which page it got.
	DefinitionEncoding Encoding
	RepetitionEncoding Encoding

	// NumNulls and NumRows are the second version of the data page only. The
	// row count is what makes it worth having: a page of a repeated column
	// holds more values than rows, and this is what lets a reader skip a page
	// without decoding its levels.
	NumNulls int32
	NumRows  int32

	// DefinitionLength and RepetitionLength are how many bytes of the body the
	// levels take, in the second version of the data page. They come first,
	// repetition then definition, and they are outside whatever compression
	// the rest of the body went through. The first version writes its levels
	// inside the compressed part and says nothing about how long they are.
	DefinitionLength int32
	RepetitionLength int32

	// Compressed says whether the values in the body went through the codec of
	// the chunk. The second version of the data page may turn compression off
	// for one page, which a writer does when the page did not get smaller.
	// Everything else is compressed if the chunk is.
	Compressed bool

	// Sorted is a dictionary page whose entries are in order, which lets a
	// reader compare dictionary indices instead of values. Writers rarely say
	// so even when it is true.
	Sorted bool

	// Stats is what the writer said about the values of this page. It is the
	// same structure the chunk carries and it is usually not written, since
	// the column index replaced it.
	Stats Statistics
}

// Page is a page of a column chunk.
//
// Data is the body as it sits in the file, which is compressed if the chunk is
// and holds the levels in front of the values either way. It points into the
// bytes of the chunk rather than copying out of them, so it stays valid for as
// long as the Pages it came from.
type Page struct {
	PageHeader
	Data []byte
}

// Pages is the pages of one column chunk, in the order they were written.
type Pages struct {
	chunk []byte
	pos   int

	// read is how many pages have come back already, which is what the errors
	// count in. column is the name of the column, for the same reason.
	read   int
	column string
}

// Start is where a column chunk begins in the file.
//
// It is the dictionary page when there is one and the first data page when
// there is not, worked out from the two page offsets rather than read from the
// chunk's own FileOffset. That field is meant to say this and enough writers
// have got it wrong over the years that no reader trusts it.
//
// Zero is either offset saying it is not there, which the format leaves to a
// reader to work out and which is safe to read that way because no page can
// live at nought: the first four bytes of the file are the magic. A chunk of a
// file with no rows in it has a dictionary page and no data page at all, and
// says so with a data page offset of zero.
func (c *ColumnChunk) Start() int64 {
	dictionary, data := c.Meta.DictionaryPageOffset, c.Meta.DataPageOffset
	if dictionary > 0 && (data <= 0 || dictionary < data) {
		return dictionary
	}
	return data
}

// ReadPages reads a column chunk of a file and returns its pages.
//
// The size is the size of the file, the same one ReadMetadata was given. It is
// what the offsets in the chunk are checked against, since they are numbers out
// of a footer and a chunk claiming to start past the end of the file must not
// turn into a read of that size.
//
// The whole chunk is read here, in one ReadAt, and the pages that come back
// point into it. Reading the chunk that a scan wants is the point of having
// read the footer first.
func ReadPages(r io.ReaderAt, size int64, c *ColumnChunk) (*Pages, error) {
	column := strings.Join(c.Meta.Path, ".")
	if c.FilePath != "" {
		return nil, fmt.Errorf("parquet: %w: the chunk for %s is in %q rather than in this file",
			ErrUnsupported, column, c.FilePath)
	}

	start, n := c.Start(), c.Meta.TotalCompressedSize
	if start < 0 || n < 0 || start > size || n > size-start {
		return nil, fmt.Errorf("parquet: %w: the chunk for %s is %d bytes at %d of a file of %d",
			ErrFormat, column, n, start, size)
	}

	// A chunk of no bytes is a chunk of no pages. It is read as one rather than
	// refused because a read of nothing at the end of a file is an error in
	// every io.ReaderAt there is, and because a chunk with nothing in it is
	// what a column of a row group with no rows looks like.
	buf := make([]byte, n)
	if n > 0 {
		if _, err := r.ReadAt(buf, start); err != nil {
			return nil, fmt.Errorf("parquet: reading the chunk for %s: %w", column, err)
		}
	}
	return &Pages{chunk: buf, column: column}, nil
}

// Next reads the next page of the chunk. It returns io.EOF once the chunk has
// been walked to the end.
func (p *Pages) Next() (Page, error) {
	if p.pos >= len(p.chunk) {
		return Page{}, io.EOF
	}

	r := &reader{buf: p.chunk, pos: p.pos}
	var page Page
	if err := page.read(r); err != nil {
		return Page{}, fmt.Errorf("parquet: page %d of %s: %w", p.read, p.column, err)
	}

	n := int(page.CompressedSize)
	if n > len(p.chunk)-r.pos {
		return Page{}, fmt.Errorf("parquet: %w: page %d of %s is %d bytes with %d left in the chunk",
			ErrFormat, p.read, p.column, n, len(p.chunk)-r.pos)
	}
	page.Data = p.chunk[r.pos : r.pos+n]

	// The checksum is over the body as it sits in the file, levels and all,
	// after whatever compression the writer did. So it is checked here, before
	// anything looks at the bytes, which is the only place it is worth
	// anything: a page that fails this is a page whose values are already
	// wrong.
	if page.HasCRC {
		if sum := crc32.ChecksumIEEE(page.Data); sum != uint32(page.CRC) {
			return Page{}, fmt.Errorf("parquet: %w: page %d of %s checksums to %#08x and says %#08x",
				ErrFormat, p.read, p.column, sum, uint32(page.CRC))
		}
	}

	p.pos = r.pos + n
	p.read++
	return page, nil
}

// The field numbers below are the ones in parquet.thrift, the same as the ones
// the footer is read with.

func (p *Page) read(r *reader) error {
	p.PageHeader = PageHeader{
		Encoding:           NoEncoding,
		DefinitionEncoding: NoEncoding,
		RepetitionEncoding: NoEncoding,
		Compressed:         true,
	}

	err := r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			var v int32
			if v, err = r.int32(t); err == nil {
				p.Kind = PageKind(v)
			}
		case 2:
			p.UncompressedSize, err = r.int32(t)
		case 3:
			p.CompressedSize, err = r.int32(t)
		case 4:
			if p.CRC, err = r.int32(t); err == nil {
				p.HasCRC = true
			}
		case 5:
			err = p.readData(r, t)
		case 7:
			err = p.readDictionary(r, t)
		case 8:
			err = p.readDataV2(r, t)
		default:
			err = r.skip(t)
		}
		return err
	})
	if err != nil {
		return err
	}
	return p.check()
}

// readData reads the header of a data page as the first version of it wrote
// one, which is levels and values together in one compressed run of bytes.
func (h *PageHeader) readData(r *reader, t thriftType) error {
	if t != thriftStruct {
		return r.skip(t)
	}

	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			h.NumValues, err = r.int32(t)
		case 2:
			err = readEncoding(r, t, &h.Encoding)
		case 3:
			err = readEncoding(r, t, &h.DefinitionEncoding)
		case 4:
			err = readEncoding(r, t, &h.RepetitionEncoding)
		case 5:
			err = h.Stats.read(r, t)
		default:
			err = r.skip(t)
		}
		return err
	})
}

// readDataV2 reads the header of a data page as the second version wrote one.
// The levels are their own bytes here and their encoding is not written down,
// because the format settled it: both of them are RLE and nothing else.
func (h *PageHeader) readDataV2(r *reader, t thriftType) error {
	if t != thriftStruct {
		return r.skip(t)
	}
	h.DefinitionEncoding, h.RepetitionEncoding = RLE, RLE

	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			h.NumValues, err = r.int32(t)
		case 2:
			h.NumNulls, err = r.int32(t)
		case 3:
			h.NumRows, err = r.int32(t)
		case 4:
			err = readEncoding(r, t, &h.Encoding)
		case 5:
			h.DefinitionLength, err = r.int32(t)
		case 6:
			h.RepetitionLength, err = r.int32(t)
		case 7:
			h.Compressed, err = r.boolean(t)
		case 8:
			err = h.Stats.read(r, t)
		default:
			err = r.skip(t)
		}
		return err
	})
}

// readDictionary reads the header of a dictionary page, which is the values of
// the column written once each and in no particular order unless it says
// otherwise.
func (h *PageHeader) readDictionary(r *reader, t thriftType) error {
	if t != thriftStruct {
		return r.skip(t)
	}

	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			h.NumValues, err = r.int32(t)
		case 2:
			err = readEncoding(r, t, &h.Encoding)
		case 3:
			h.Sorted, err = r.boolean(t)
		default:
			err = r.skip(t)
		}
		return err
	})
}

// readEncoding reads a field written as an encoding. Every enumeration in the
// format is an int32 on the wire, and this is the one of them a page header has
// four of.
func readEncoding(r *reader, t thriftType, e *Encoding) error {
	v, err := r.int32(t)
	if err != nil {
		return err
	}
	*e = Encoding(v)
	return nil
}

// check is what a header has to say before the walk will believe it.
//
// Two things are being asked. The numbers the walk runs on have to be numbers:
// a negative size would step the walk backwards and a page that holds more
// nulls than values holds a count somebody made up. And the page has to have
// said what it is: a header whose type says data page and which carries no data
// page header at all left out everything a decoder needs, and is a truncated
// write or a file that was never parquet rather than a page to guess at.
func (h *PageHeader) check() error {
	if h.CompressedSize < 0 || h.UncompressedSize < 0 {
		return fmt.Errorf("%w: %d bytes compressed and %d uncompressed",
			ErrFormat, h.CompressedSize, h.UncompressedSize)
	}

	switch h.Kind {
	case DataPage:
		if h.DefinitionEncoding == NoEncoding || h.RepetitionEncoding == NoEncoding {
			return fmt.Errorf("%w: a data page that does not say how its levels are encoded",
				ErrFormat)
		}
	case DataPageV2:
		if h.NumNulls < 0 || h.NumNulls > h.NumValues {
			return fmt.Errorf("%w: a page of %d values holding %d nulls",
				ErrFormat, h.NumValues, h.NumNulls)
		}
		if h.NumRows < 0 || h.NumRows > h.NumValues {
			return fmt.Errorf("%w: a page of %d values covering %d rows",
				ErrFormat, h.NumValues, h.NumRows)
		}
		if h.DefinitionLength < 0 || h.RepetitionLength < 0 ||
			h.DefinitionLength+h.RepetitionLength > h.CompressedSize {
			return fmt.Errorf("%w: a page of %d bytes whose levels take %d and %d",
				ErrFormat, h.CompressedSize, h.RepetitionLength, h.DefinitionLength)
		}
	case DictionaryPage:
		// A dictionary page says nothing beyond what every page with values in
		// it has to say, which is checked below.
	case IndexPage:
		// An index page has no header of its own and no values in it. Nothing
		// writes one.
		return nil
	default:
		// A page type this package has never heard of is a page to walk past
		// rather than a file to refuse, since the walk only needs its size and
		// whatever wants to read it can say no itself.
		return nil
	}

	if h.NumValues < 0 {
		return fmt.Errorf("%w: a %s of %d values", ErrFormat, h.Kind, h.NumValues)
	}
	if h.Encoding == NoEncoding {
		return fmt.Errorf("%w: a %s that does not say how it is encoded", ErrFormat, h.Kind)
	}
	return nil
}
