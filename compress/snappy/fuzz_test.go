package snappy_test

import (
	"bytes"
	"errors"
	"testing"

	"github.com/tamnd/kuma/compress/snappy"
)

// FuzzDecode throws bytes at the decoder and checks that what comes back is
// either an ErrCorrupt or exactly as many bytes as the block said it held.
//
// A decompressor is the one place in a reader where a length in somebody else's
// file turns straight into an index into memory, so what this is looking for is
// a block that reads or writes outside the buffer it was given, and it finds
// that by running at all. What it can check on top of that is small and worth
// checking anyway: a block that decompresses has to fill its buffer to the end,
// no further and no less, and it has to say the same thing into a buffer with
// something else in it as into a fresh one.
func FuzzDecode(f *testing.F) {
	for _, name := range corpus(f) {
		block, _ := blockOf(f, name)
		f.Add(block)
	}
	f.Add(built(8, literal([]byte("abcd"), 0), copyBack(5, 4, 4)))
	f.Add(built(9, literal([]byte("z"), 0), copyBack(2, 1, 8)))
	f.Add([]byte(nil))
	f.Add([]byte{0xff, 0xff, 0xff, 0xff, 0x0f})

	f.Fuzz(func(t *testing.T, data []byte) {
		got, err := snappy.Decode(nil, data)
		if err != nil {
			if !errors.Is(err, snappy.ErrCorrupt) {
				t.Fatalf("Decode = %v, want an ErrCorrupt", err)
			}
			return
		}

		size, err := snappy.DecodedLen(data)
		if err != nil {
			t.Fatalf("DecodedLen of a block that decompressed: %v", err)
		}
		if len(got) != size {
			t.Fatalf("Decode wrote %d bytes into a block of %d", len(got), size)
		}

		// A buffer with room to spare and rubbish in it, which is what the
		// second page of a column arrives at. What it holds has to be gone and
		// what comes back has to be the length of this block rather than of the
		// buffer.
		dirty := bytes.Repeat([]byte{0xff}, size+7)
		again, err := snappy.Decode(dirty, data)
		if err != nil {
			t.Fatalf("Decode into a buffer of its own: %v", err)
		}
		if !bytes.Equal(again, got) {
			t.Fatalf("Decode into a buffer of its own: %s", differ(again, got))
		}
	})
}
