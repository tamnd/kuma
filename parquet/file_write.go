package parquet

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

// WriteMetadata writes the footer of a parquet file, and with it the last eight
// bytes of the file.
//
// A parquet file is the four bytes PAR1, then the pages, then the footer, then
// how long the footer is in four bytes, then PAR1 again. This writes the last
// three of those, so a writer that has put its pages down finishes the file with
// one call to it and a reader working backwards from the end finds everything
// else. It returns how many bytes it wrote.
//
// The footer is built whole before any of it goes out, because the length behind
// it is not known until it is finished. That is a few hundred bytes a column
// chunk, which is what a footer is.
//
// Nothing here checks that the offsets in the metadata point at anything. A
// footer is a description of a file and this writes the description it was
// given, so a caller that hands it offsets from one file and pages from another
// gets a file that opens and reads nonsense. The whole file writer is what keeps
// those two in step.
func WriteMetadata(w io.Writer, m *Metadata) (int64, error) {
	var b writer
	m.write(&b)

	// The footer is the one struct in the file that nothing else contains, so
	// the byte that says a struct has ended is written here rather than by
	// whatever wrote the field header in front of it.
	b.put(byte(thriftStop))

	tail, err := footerTrailer(len(b.buf))
	if err != nil {
		return 0, err
	}
	b.buf = append(b.buf, tail[:]...)

	n, err := w.Write(b.buf)
	if err != nil {
		return int64(n), fmt.Errorf("parquet: writing the footer: %w", err)
	}
	return int64(n), nil
}

// footerTrailer returns the eight bytes that go behind a footer of n bytes,
// which is its length and the magic.
//
// A footer of more than four gigabytes has nowhere to say so, since the length
// is four bytes and always has been. Nothing will ever write one, because a
// footer that size is a file of millions of row groups and the format falls over
// long before that, but the alternative to checking is writing a length that
// wrapped and a file that no reader can open.
func footerTrailer(n int) ([trailer]byte, error) {
	var out [trailer]byte
	if int64(n) > math.MaxUint32 {
		return out, fmt.Errorf("parquet: a footer of %d bytes, which is more than the four bytes behind it can say", n)
	}

	binary.LittleEndian.PutUint32(out[:4], uint32(n))
	copy(out[4:], magic)
	return out, nil
}
