package parquet

import (
	"bytes"
	"compress/gzip"
	"errors"
	"fmt"
	"io"

	"github.com/tamnd/kuma/compress/snappy"
)

// Undoing the compression of a page.
//
// Every parquet file anybody has is compressed, and the codec is a property of
// a column chunk rather than of the file, so one file may hold a snappy column
// next to a gzip one next to one that was left alone. That is the shape of it
// here too: a decompressor is made for a chunk, holds one buffer, and hands
// back the pages of that chunk with their bodies undone.
//
// What comes out is checked against what the header said before it is believed.
// A page header carries both sizes, and the compressed bytes carry their own
// idea of how long they are, so a page that comes to something other than what
// its header promised is a file that contradicts itself and is refused rather
// than passed on to a decoder that would read whatever it found.
//
// The uncompressed size is checked before the buffer for it is made rather than
// after, since a page of a thousand bytes saying it comes to two gigabytes is a
// number to refuse rather than a number to allocate. What it is checked against
// is how far the codec could stretch the bytes it was given, which is a
// property of the codec rather than a guess.
//
// The one place the two versions of the data page differ here is the levels.
// The first version puts them inside the compressed part and says nothing about
// how long they are. The second leaves them uncompressed in front of the values
// and says in the header how many bytes they take, so undoing a page of that
// version means putting the two pieces back together, and that is why a page
// comes back in a buffer of the decompressor's own rather than as two slices.

// Decompressor undoes the compression of the pages of one column chunk.
//
// It holds the buffer the pages come back in, so a scan reading a chunk of a
// thousand pages allocates once rather than once per page. That is also what a
// caller has to know about it: a page it hands back holds until the next one is
// asked for, in the way a bufio.Scanner's token does.
//
// The zero value undoes nothing, which is what a chunk that was not compressed
// wants. Use NewDecompressor.
type Decompressor struct {
	codec Codec

	// undo is the codec, or nil for a chunk that has none. It is settled once
	// when the decompressor is made rather than looked up per page, since every
	// page of a chunk is compressed the same way and a chunk of a thousand
	// pages would ask the same question a thousand times.
	undo func(dst, src []byte) error

	// stretch is how many bytes one byte of a compressed body may come back
	// as, which is what an uncompressed size is believed up to.
	stretch int

	// buf is where a page is undone into and gzipped is the reader gzip needs,
	// both kept from page to page. src is what that reader reads from, since
	// the standard library wants an io.Reader and a page is bytes.
	buf     []byte
	gzipped *gzip.Reader
	src     bytes.Reader
	tail    [1]byte
}

// How far each codec can stretch what it is given, which is what a page saying
// what it comes to is believed up to.
//
// The longest thing a snappy tag can write is sixty four bytes out of three,
// which is under twenty two to one, and the most deflate can write is the one
// thousand and thirty two to one that the zlib documentation gives. Both are
// rounded up to a power of two here so that no honest page is ever refused for
// being a byte over a number nobody meant to be exact.
const (
	snappyStretch = 32
	gzipStretch   = 2048
)

// NewDecompressor returns a decompressor for a chunk written with the codec,
// and refuses a codec this package cannot undo.
//
// The refusal is here rather than at the first page so that a scan finds out
// what it cannot read before it reads anything.
func NewDecompressor(c Codec) (*Decompressor, error) {
	d := &Decompressor{codec: c}
	switch c {
	case Uncompressed:
	case Snappy:
		d.undo, d.stretch = d.unsnappy, snappyStretch
	case Gzip:
		d.undo, d.stretch = d.ungzip, gzipStretch
	default:
		return nil, fmt.Errorf("parquet: %w: %s pages, and only snappy and gzip are undone yet",
			ErrUnsupported, c)
	}
	return d, nil
}

// Codec is the codec the decompressor undoes.
func (d *Decompressor) Codec() Codec { return d.codec }

// Page returns the page with its body decompressed.
//
// The body it comes back with is the levels and the values together, the way an
// uncompressed page holds them, so what reads a page does not have to ask which
// codec it went through. The header is left as the file wrote it, which means
// the two sizes in it still say what the page took on disk and what it comes
// to, and it is the second one that says how long the body now is.
//
// A page of a chunk that was not compressed comes back as it went in, pointing
// into the bytes of the chunk. Everything else points into the decompressor and
// holds only until the next page is asked for.
func (d *Decompressor) Page(p Page) (Page, error) {
	levels := 0
	switch {
	case d.undo == nil, !p.Compressed:
		// A chunk with no codec, or the one page in a hundred that the second
		// version of the data page is allowed to leave alone because the codec
		// made it bigger.
		return p, nil
	case p.Kind == DataPageV2:
		levels = int(p.RepetitionLength) + int(p.DefinitionLength)
	case p.Kind != DataPage && p.Kind != DictionaryPage:
		// A page type this package has never heard of, which the walk steps
		// over and nothing reads. Undoing a codec into a body nobody will look
		// at is work for its own sake, and the page may not even be compressed.
		return p, nil
	}

	size := int(p.UncompressedSize)
	if levels < 0 || levels > len(p.Data) || size < levels {
		return Page{}, fmt.Errorf("parquet: %w: a page of %d bytes coming to %d with %d of levels",
			ErrFormat, len(p.Data), size, levels)
	}
	// In sixty four bits because the two of them are thirty two bit numbers out
	// of a file and the product of the widest pair of them is not.
	body, values := len(p.Data)-levels, size-levels
	if int64(values) > int64(d.stretch)*int64(body) {
		return Page{}, fmt.Errorf("parquet: %w: %d bytes of %s coming to %d, which the codec cannot do",
			ErrFormat, body, d.codec, values)
	}

	d.buf = grow(d.buf, size)
	copy(d.buf, p.Data[:levels])
	if err := d.undo(d.buf[levels:], p.Data[levels:]); err != nil {
		return Page{}, err
	}

	p.Data = d.buf
	return p, nil
}

// unsnappy undoes a page written in the block format of snappy, which is what
// nearly every parquet file written this decade is.
//
// The block says how long it is and so does the page header, and the two have
// to agree before a byte of it is undone. That is not belt and braces: it is
// what lets the block be undone straight into the buffer the levels were
// already copied into, since a decoder that had to grow the buffer would leave
// the values somewhere other than behind the levels.
func (d *Decompressor) unsnappy(dst, src []byte) error {
	size, err := snappy.DecodedLen(src)
	if err != nil {
		return fmt.Errorf("parquet: %w: %w", ErrFormat, err)
	}
	if size != len(dst) {
		return fmt.Errorf("parquet: %w: a snappy page of %d bytes in a header that says %d",
			ErrFormat, size, len(dst))
	}
	if _, err = snappy.Decode(dst, src); err != nil {
		return fmt.Errorf("parquet: %w: %w", ErrFormat, err)
	}
	return nil
}

// ungzip undoes a page written with the codec the format calls GZIP, which is a
// gzip member with its header and its checksum rather than raw deflate.
//
// What comes out has to be exactly as long as the header said, and the read
// after it has to be the end of the stream. A page that keeps going past its
// own uncompressed size is one whose header and body disagree, and it is worth
// saying so here rather than letting a decoder read the first half of a page
// and stop.
func (d *Decompressor) ungzip(dst, src []byte) error {
	d.src.Reset(src)

	var err error
	if d.gzipped == nil {
		d.gzipped, err = gzip.NewReader(&d.src)
	} else {
		err = d.gzipped.Reset(&d.src)
	}
	if err != nil {
		return fmt.Errorf("parquet: %w: a gzip page: %w", ErrFormat, err)
	}

	if _, err = io.ReadFull(d.gzipped, dst); err != nil {
		return fmt.Errorf("parquet: %w: a gzip page of %d bytes: %w", ErrFormat, len(dst), err)
	}
	// The read past the end is what walks off the end of the member and checks
	// the length and the checksum written there, so it is worth doing for its
	// error as much as for what it says about the length.
	n, err := d.gzipped.Read(d.tail[:])
	if n > 0 {
		return fmt.Errorf("parquet: %w: a gzip page holding more than the %d bytes it says",
			ErrFormat, len(dst))
	}
	if !errors.Is(err, io.EOF) {
		return fmt.Errorf("parquet: %w: the end of a gzip page: %w", ErrFormat, err)
	}
	return nil
}
