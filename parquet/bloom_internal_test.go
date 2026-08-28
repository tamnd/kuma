package parquet

import (
	"bytes"
	"errors"
	"testing"
)

// The filters no writer produces.
//
// A bloom filter sits at the end of the file on an offset out of the footer,
// with a header in front of it saying how long it is and which of the three
// things the format has ever defined it used. Nothing in a real file exercises
// the answers to those questions being anything else, so the headers here are
// written by hand: a hash this reader has never seen, a bitset that is not
// whole blocks, a length that runs off the end of what the footer claimed.
// Reading any of them the only way this package knows would look a value up in
// the wrong bits and answer no to a value that is there.

// bloomHeaderOf writes a header, with the member written in each of the three
// unions given as a field id. One is the member this reader knows and anything
// else is a writer from a future that has not happened.
func bloomHeaderOf(numBytes int32, algorithm, hash, compression int16) []byte {
	w := &builder{}
	w.field(1, thriftInt32).varint(int64(numBytes))

	for i, member := range []int16{algorithm, hash, compression} {
		w.field(int16(i)+2, thriftStruct)
		w.structure(func() {
			w.field(member, thriftStruct)
			w.structure(func() {})
		})
	}
	return w.raw(0).b
}

// blocks is a bitset of n blocks with every bit set, which is a filter that
// keeps everything and is the shape rather than the contents of a real one.
func blocks(n int) []byte { return bytes.Repeat([]byte{0xff}, n*bloomBlock) }

// forgedBloom lays a header and a bitset out behind a file's magic and hands
// back the chunk pointing at them, which is what a footer would have said. A
// length of nought is a writer that wrote none, which is legal and is read by
// taking enough for the header and going back for the rest.
func forgedBloom(header, bits []byte, length int32) (*bytes.Reader, int64, *ColumnChunk) {
	buf := append([]byte("PAR1"), header...)
	buf = append(buf, bits...)

	c := &ColumnChunk{Meta: ColumnMeta{Path: []string{"id"}}}
	c.Meta.BloomFilterOffset, c.Meta.BloomFilterLength = 4, length
	return bytes.NewReader(buf), int64(len(buf)), c
}

// TestReadBloomFilterUnsupported is a filter written with something this
// package cannot look a value up in.
//
// There is one algorithm, one hash and one compression in the format, and no
// way for a file to say which version of it wrote them, so a header naming
// anything else is refused rather than read as the one thing this knows. A
// filter read the wrong way says no to values that are there, and no is the
// answer a scan acts on.
func TestReadBloomFilterUnsupported(t *testing.T) {
	cases := []struct {
		name   string
		header []byte
	}{
		{"an algorithm nobody has defined", bloomHeaderOf(32, 2, 1, 1)},
		{"a hash nobody has defined", bloomHeaderOf(32, 1, 2, 1)},
		{"a compression nobody has defined", bloomHeaderOf(32, 1, 1, 2)},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			src, size, chunk := forgedBloom(c.header, blocks(1), int32(len(c.header))+bloomBlock)
			if _, err := ReadBloomFilter(src, size, chunk); !errors.Is(err, ErrUnsupported) {
				t.Errorf("got %v, want %v", err, ErrUnsupported)
			}
		})
	}
}

// TestReadBloomFilterRefused is the headers that contradict the file they are
// in.
func TestReadBloomFilterRefused(t *testing.T) {
	ok := bloomHeaderOf(32, 1, 1, 1)

	union := &builder{}
	union.field(1, thriftInt32).varint(32)
	union.field(2, thriftInt32).varint(1)

	cases := []struct {
		name   string
		header []byte
		bits   []byte
		length int32
	}{
		{"a union written as a number", union.raw(0).b, blocks(1), int32(len(union.b)) + bloomBlock},
		{"a bitset of no bytes", bloomHeaderOf(0, 1, 1, 1), nil, int32(len(ok))},
		{"a bitset of fewer bytes than nought", bloomHeaderOf(-32, 1, 1, 1), blocks(1), int32(len(ok)) + bloomBlock},
		{"a bitset that is not whole blocks", bloomHeaderOf(48, 1, 1, 1), blocks(2), int32(len(ok)) + 48},
		{"a bitset longer than the footer said", bloomHeaderOf(64, 1, 1, 1), blocks(2), int32(len(ok)) + bloomBlock},
		{"a header that stops partway", ok[:len(ok)-1], nil, int32(len(ok)) - 1},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			src, size, chunk := forgedBloom(c.header, c.bits, c.length)
			if _, err := ReadBloomFilter(src, size, chunk); !errors.Is(err, ErrFormat) {
				t.Errorf("got %v, want %v", err, ErrFormat)
			}
		})
	}
}

// TestReadBloomFilterOutside is a filter the file cannot hold.
//
// Both numbers come out of a footer, so a claim that a filter is two hundred
// bytes at the end of a file of a hundred is a claim to check before anything
// is allocated for it.
func TestReadBloomFilterOutside(t *testing.T) {
	header := bloomHeaderOf(32, 1, 1, 1)
	src, size, chunk := forgedBloom(header, blocks(1), int32(len(header))+bloomBlock)

	cases := []struct {
		name   string
		offset int64
		length int32
	}{
		{"a filter past the end of the file", size + 1, chunk.Meta.BloomFilterLength},
		{"a filter running past the end of the file", chunk.Meta.BloomFilterOffset, int32(size)},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			f := *chunk
			f.Meta.BloomFilterOffset, f.Meta.BloomFilterLength = c.offset, c.length

			if _, err := ReadBloomFilter(src, size, &f); !errors.Is(err, ErrFormat) {
				t.Errorf("got %v, want %v", err, ErrFormat)
			}
		})
	}
}

// TestReadBloomFilterNoLength is a writer that wrote no length in the footer,
// which is what the format's first writers did.
//
// The header has no length in front of it either, so what is read is enough of
// the file for one and then the bitset on its own, once the header has said how
// long it is. The header here also carries a field this package has no use for,
// which is what a later writer adding to it looks like and is read past rather
// than refused.
func TestReadBloomFilterNoLength(t *testing.T) {
	w := &builder{}
	w.field(1, thriftInt32).varint(8 * bloomBlock)
	for _, id := range []int16{2, 3, 4} {
		w.field(id, thriftStruct)
		w.structure(func() {
			w.field(1, thriftStruct)
			w.structure(func() {})
		})
	}
	w.field(5, thriftBinary).binary("what a later writer wrote here")
	header := w.raw(0).b

	bits := blocks(8)
	src, size, chunk := forgedBloom(header, bits, 0)

	f, err := ReadBloomFilter(src, size, chunk)
	if err != nil {
		t.Fatalf("ReadBloomFilter: %v", err)
	}
	if f.Bytes() != len(bits) {
		t.Fatalf("the filter is %d bytes, want %d", f.Bytes(), len(bits))
	}
	if !f.Has([]byte("anything")) {
		t.Error("a filter with every bit set ruled a value out")
	}

	// The bitset is read after the header, so a file that gives up the first and
	// not the second is its own case.
	t.Run("a file that stops answering halfway", func(t *testing.T) {
		half := &halting{at: src, ok: 1}
		if _, err := ReadBloomFilter(half, size, chunk); !errors.Is(err, errGoneIndex) {
			t.Errorf("got %v, want %v", err, errGoneIndex)
		}
	})

	t.Run("a file that stops answering", func(t *testing.T) {
		if _, err := ReadBloomFilter(gone{}, size, chunk); !errors.Is(err, errGoneIndex) {
			t.Errorf("got %v, want %v", err, errGoneIndex)
		}
	})
}

// TestBloomFilterEmptyBlock is the bits of a filter that holds nothing, which
// is what a writer that was asked for a filter on a chunk of no values leaves
// behind.
//
// Every bit is clear, so every lookup is ruled out on the first word, which is
// the answer a scan skips the chunk on and is the right one.
func TestBloomFilterEmptyBlock(t *testing.T) {
	f := &BloomFilter{bits: make([]byte, 2*bloomBlock)}

	if f.Has([]byte("anything")) || f.HasString("anything") {
		t.Error("a filter with no bits set kept a value")
	}
	if f.Bytes() != 2*bloomBlock {
		t.Errorf("the filter is %d bytes, want %d", f.Bytes(), 2*bloomBlock)
	}
}
