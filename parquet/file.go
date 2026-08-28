package parquet

import (
	"encoding/binary"
	"fmt"
	"io"
)

// The magic at both ends of a parquet file. The second one is a file whose
// footer is encrypted, which says the same thing in the same place and then
// hands the reader something it cannot read without a key.
const (
	magic          = "PAR1"
	encryptedMagic = "PARE"
)

// trailer is the four byte footer length and the four byte magic behind it.
const trailer = 8

// ReadMetadata reads the footer of a parquet file.
//
// The file is read backwards, the way the format is meant to be read: the last
// eight bytes say how long the footer is and where it therefore starts, and the
// footer says where everything else is. Three reads, none of them near the
// data, which is why opening a parquet file over a network is cheap and opening
// a CSV file is not.
//
// It takes an io.ReaderAt and the size the way archive/zip does, since a footer
// at the end of a file cannot be found by anything that only reads forwards.
// The size has to be the real one: a size that is too small looks like a
// truncated file and a size that is too large looks like a file that is not
// parquet at all.
//
// The strings in the result are copied out of the footer, and the statistics
// point into it, so a Metadata holds onto the bytes of the footer and nothing
// else. The file itself is not read beyond it and is not held open.
func ReadMetadata(r io.ReaderAt, size int64) (*Metadata, error) {
	if size < int64(len(magic))+trailer {
		return nil, fmt.Errorf("parquet: %w: a file of %d bytes is too small to hold a footer",
			ErrFormat, size)
	}

	var head [len(magic)]byte
	if _, err := r.ReadAt(head[:], 0); err != nil {
		return nil, fmt.Errorf("parquet: reading the first %d bytes: %w", len(magic), err)
	}
	if string(head[:]) != magic {
		return nil, fmt.Errorf("parquet: %w: it starts with %q rather than %q",
			ErrFormat, head[:], magic)
	}

	var tail [trailer]byte
	if _, err := r.ReadAt(tail[:], size-trailer); err != nil {
		return nil, fmt.Errorf("parquet: reading the last %d bytes: %w", trailer, err)
	}
	switch end := string(tail[4:]); end {
	case magic:
	case encryptedMagic:
		return nil, fmt.Errorf("parquet: %w: the footer is encrypted", ErrUnsupported)
	default:
		return nil, fmt.Errorf("parquet: %w: it ends with %q rather than %q", ErrFormat, end, magic)
	}

	// The length is a claim about a file this reader has not read yet, so it is
	// checked against the size before it is believed. What it may not overlap
	// is the magic at the front and the trailer it was read out of.
	n := int64(binary.LittleEndian.Uint32(tail[:4]))
	if room := size - int64(len(magic)) - trailer; n > room {
		return nil, fmt.Errorf("parquet: %w: a footer of %d bytes in a file with room for %d",
			ErrFormat, n, room)
	}

	buf := make([]byte, n)
	if _, err := r.ReadAt(buf, size-trailer-n); err != nil {
		return nil, fmt.Errorf("parquet: reading the footer: %w", err)
	}

	m := &Metadata{}
	if err := m.read(&reader{buf: buf}); err != nil {
		return nil, err
	}
	return m, nil
}
