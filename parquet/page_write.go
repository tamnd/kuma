package parquet

import (
	"fmt"
	"hash/crc32"
	"io"
)

// Writing a page.
//
// This is the walk in page.go turned around. A page is a Thrift header followed
// by a body laid straight down, with nothing in front of the header saying how
// long it is, which is why the walk has to read every header to find the next
// page and why the sizes in a header are the one thing a writer cannot get
// wrong. A compressed size that is one byte out does not produce a page that is
// one byte wrong, it produces a chunk where every page after this one is
// garbage, and the reader has no way to tell that from a file that was always
// nonsense.
//
// So the sizes are checked here rather than trusted. The reader checks them
// because they are somebody else's numbers and it must not walk off the end of a
// buffer. The writer checks them because a page it wrote that disagrees with
// itself is a bug in whatever called it, and finding that out at the call is
// worth more than finding it out when a reader somewhere refuses the file.

// WritePage writes a page: its header, then the body behind it.
//
// The body is the bytes as they go in the file, which means already encoded,
// already compressed if the chunk is compressed, and with the levels in front of
// the values. Nothing here encodes or compresses anything, and the body is
// written from where it sits rather than copied. It returns how many bytes it
// wrote, header and body together, which is what a caller adds up to know where
// the next page starts.
//
// The header has to agree with the body. CompressedSize is the length of the
// body, and for the second version of the data page the two level lengths are
// bytes of the body and come out of the front of it, so the three together are a
// claim this checks rather than takes. It asks the same things of the header
// that the reader asks of one it read, so a page written here is a page this
// package would accept. A header that does not agree with its body is refused
// and nothing is written.
//
// The checksum is the one field the caller does not fill in. HasCRC says whether
// to write one and the value is computed here over the body, so a page this
// writes can never carry a checksum that disagrees with it. The CRC field of the
// header is ignored on the way out.
func WritePage(w io.Writer, h *PageHeader, body []byte) (int64, error) {
	if err := h.writable(body); err != nil {
		return 0, err
	}

	var b writer
	h.write(&b, body)
	b.put(byte(thriftStop))

	// The header and the body go down as two writes rather than one. Joining
	// them would mean copying every page of every file through a buffer to save
	// a call, and a page is a megabyte where a header is forty bytes.
	n, err := w.Write(b.buf)
	if err != nil {
		return int64(n), fmt.Errorf("parquet: writing a %s: %w", h.Kind, err)
	}

	m, err := w.Write(body)
	if err != nil {
		return int64(n + m), fmt.Errorf("parquet: writing a %s: %w", h.Kind, err)
	}
	return int64(n + m), nil
}

// writable says whether the header describes the body it was handed.
//
// The first thing asked is everything the reader asks of a header it read, so a
// page this package writes is a page this package would accept. The rest is the
// sizes against the bytes, which the reader has no way to ask because it works
// the other way round: it takes the size and cuts the body to it.
func (h *PageHeader) writable(body []byte) error {
	if h.Kind == IndexPage {
		return fmt.Errorf("parquet: %w: an index page, which the format defined and no writer produces",
			ErrUnsupported)
	}
	if err := h.check(); err != nil {
		return err
	}

	if int(h.CompressedSize) != len(body) {
		return fmt.Errorf("parquet: a %s of %d bytes whose header says %d",
			h.Kind, len(body), h.CompressedSize)
	}

	// The level lengths of a second version data page are not checked again
	// here. They are bytes of the body and come out of the front of it, and
	// check has already had them against the compressed size, which the line
	// above has just had against the body. Checking them a second time against
	// the same number by a different route is a line that cannot fail.
	return nil
}

// The field numbers below are the ones in parquet.thrift, the same as the ones
// the pages are read with.

func (h *PageHeader) write(w *writer, body []byte) {
	w.int32Field(1, int32(h.Kind))
	w.int32Field(2, h.UncompressedSize)
	w.int32Field(3, h.CompressedSize)
	if h.HasCRC {
		w.int32Field(4, int32(crc32.ChecksumIEEE(body)))
	}

	switch h.Kind {
	case DataPage:
		h.writeData(w, 5)
	case DictionaryPage:
		h.writeDictionary(w, 7)
	case DataPageV2:
		h.writeDataV2(w, 8)
	default:
		// A page kind this package has never heard of has no header of its
		// own to write. The walk reads one of those as a page to step past, so
		// writing the sizes and nothing else produces exactly that.
	}
}

// writeData writes the header of a data page as the first version of it wrote
// one, which is levels and values together in one compressed run of bytes.
func (h *PageHeader) writeData(w *writer, id int16) {
	w.structure(id, func() {
		w.int32Field(1, h.NumValues)
		w.int32Field(2, int32(h.Encoding))
		w.int32Field(3, int32(h.DefinitionEncoding))
		w.int32Field(4, int32(h.RepetitionEncoding))
		h.Stats.write(w, 5)
	})
}

// writeDataV2 writes the header of a data page as the second version wrote one.
//
// The two level encodings are not written, because the format fixed both at RLE
// when it defined this page and there is nowhere to say otherwise. The reader
// fills them in for the same reason.
func (h *PageHeader) writeDataV2(w *writer, id int16) {
	w.structure(id, func() {
		w.int32Field(1, h.NumValues)
		w.int32Field(2, h.NumNulls)
		w.int32Field(3, h.NumRows)
		w.int32Field(4, int32(h.Encoding))
		w.int32Field(5, h.DefinitionLength)
		w.int32Field(6, h.RepetitionLength)

		// The format says a page with no is_compressed field is compressed, so
		// the field is written only to say it is not. Writing it either way
		// would be legal and would make every page of an uncompressed chunk two
		// bytes longer for nothing.
		if !h.Compressed {
			w.boolField(7, false)
		}
		h.Stats.write(w, 8)
	})
}

// writeDictionary writes the header of a dictionary page.
func (h *PageHeader) writeDictionary(w *writer, id int16) {
	w.structure(id, func() {
		w.int32Field(1, h.NumValues)
		w.int32Field(2, int32(h.Encoding))
		if h.Sorted {
			w.boolField(3, true)
		}
	})
}
