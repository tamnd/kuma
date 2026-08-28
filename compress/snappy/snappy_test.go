package snappy_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tamnd/kuma/compress/snappy"
)

// corpus is the name of every case in testdata, which is a block somebody
// else's snappy wrote next to the bytes that went into it. See gen.py there.
func corpus(tb testing.TB) []string {
	tb.Helper()

	found, err := filepath.Glob("testdata/*.snappy")
	if err != nil {
		tb.Fatalf("looking for the blocks in testdata: %v", err)
	}
	if len(found) == 0 {
		tb.Fatal("no blocks in testdata")
	}

	names := make([]string, len(found))
	for i, path := range found {
		names[i] = strings.TrimSuffix(filepath.Base(path), ".snappy")
	}
	return names
}

// blockOf reads one of those cases, the compressed block and then the bytes it
// has to come back as.
func blockOf(tb testing.TB, name string) (block, want []byte) {
	tb.Helper()

	block, err := os.ReadFile(filepath.Join("testdata", name+".snappy"))
	if err != nil {
		tb.Fatalf("reading the block: %v", err)
	}
	want, err = os.ReadFile(filepath.Join("testdata", name+".raw"))
	if err != nil {
		tb.Fatalf("reading what the block holds: %v", err)
	}
	return block, want
}

func TestDecode(t *testing.T) {
	for _, name := range corpus(t) {
		t.Run(name, func(t *testing.T) {
			block, want := blockOf(t, name)

			size, err := snappy.DecodedLen(block)
			if err != nil {
				t.Fatalf("DecodedLen: %v", err)
			}
			if size != len(want) {
				t.Errorf("DecodedLen = %d, want %d", size, len(want))
			}

			got, err := snappy.Decode(nil, block)
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			if !bytes.Equal(got, want) {
				t.Errorf("Decode: %s", differ(got, want))
			}
		})
	}
}

// differ says where two runs of bytes stop agreeing, since printing several
// kilobytes of them side by side says nothing.
func differ(got, want []byte) string {
	for i := range min(len(got), len(want)) {
		if got[i] != want[i] {
			return "byte " + strconv.Itoa(i) + " is " + strconv.Itoa(int(got[i])) +
				" and should be " + strconv.Itoa(int(want[i]))
		}
	}
	return "got " + strconv.Itoa(len(got)) + " bytes and want " + strconv.Itoa(len(want))
}

// TestDecodeReuse checks that a caller handing back the last answer stops
// allocating, which is what a scan reading page after page of one column does.
func TestDecodeReuse(t *testing.T) {
	block, want := blockOf(t, "keys")

	dst, err := snappy.Decode(nil, block)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}

	allocs := testing.AllocsPerRun(10, func() {
		dst, err = snappy.Decode(dst, block)
		if err != nil {
			t.Fatalf("Decode again: %v", err)
		}
		if !bytes.Equal(dst, want) {
			t.Fatalf("Decode again: %s", differ(dst, want))
		}
	})
	if allocs != 0 {
		t.Errorf("Decode into its own answer allocated %v times, want 0", allocs)
	}
}

// TestDecodeIntoASmallerBuffer checks that a buffer too small to hold the block
// is grown rather than filled to the end of, and that one too large comes back
// the length of the block rather than the length it was.
func TestDecodeIntoASmallerBuffer(t *testing.T) {
	block, want := blockOf(t, "hello")

	for _, size := range []int{0, 1, len(want) - 1, len(want), len(want) + 100} {
		got, err := snappy.Decode(make([]byte, size), block)
		if err != nil {
			t.Fatalf("Decode into %d bytes: %v", size, err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("Decode into %d bytes: %s", size, differ(got, want))
		}
	}
}

// The builders below write blocks by hand. They are for the shapes the
// reference compressor never writes and the format allows anyway: a length
// written in more bytes than it needs, and a copy with an offset of four bytes,
// which nothing that works in blocks of sixty four kilobytes can reach.

// built returns a block that says it holds size bytes, with the tags behind the
// length.
func built(size int, tags ...[]byte) []byte {
	b := binary.AppendUvarint(nil, uint64(size))
	for _, tag := range tags {
		b = append(b, tag...)
	}
	return b
}

// literal returns a literal holding p, with its length written in count bytes
// behind the tag. A count of nought puts the length in the tag itself, which is
// what a compressor does whenever it fits.
func literal(p []byte, count int) []byte {
	if count == 0 {
		return append([]byte{byte(len(p)-1) << 2}, p...)
	}

	tag := []byte{byte(59+count) << 2}
	for i := range count {
		tag = append(tag, byte((len(p)-1)>>(8*i)))
	}
	return append(tag, p...)
}

// copyBack returns a copy of length bytes reaching offset back, written in the
// form of the tag that takes width bytes.
func copyBack(width, offset, length int) []byte {
	switch width {
	case 2:
		return []byte{byte(offset>>8)<<5 | byte(length-4)<<2 | 0x01, byte(offset)}
	case 3:
		return []byte{byte(length-1)<<2 | 0x02, byte(offset), byte(offset >> 8)}
	default:
		return []byte{byte(length-1)<<2 | 0x03,
			byte(offset), byte(offset >> 8), byte(offset >> 16), byte(offset >> 24)}
	}
}

func TestDecodeBuilt(t *testing.T) {
	for _, c := range []struct {
		name  string
		block []byte
		want  string
	}{
		{"a block of nothing", built(0), ""},
		{"one byte", built(1, literal([]byte("q"), 0)), "q"},
		{"a literal whose length is in a byte of its own",
			built(3, literal([]byte("abc"), 1)), "abc"},
		{"a literal whose length is in two bytes",
			built(3, literal([]byte("abc"), 2)), "abc"},
		{"a literal whose length is in three bytes",
			built(3, literal([]byte("abc"), 3)), "abc"},
		{"a literal whose length is in four bytes",
			built(3, literal([]byte("abc"), 4)), "abc"},
		{"a copy in the short form",
			built(8, literal([]byte("abcd"), 0), copyBack(2, 4, 4)), "abcdabcd"},
		{"the longest copy the short form holds",
			built(15, literal([]byte("abcd"), 0), copyBack(2, 4, 11)), "abcdabcdabcdabc"},
		{"a copy with a two byte offset",
			built(8, literal([]byte("abcd"), 0), copyBack(3, 4, 4)), "abcdabcd"},
		{"a copy with a four byte offset",
			built(8, literal([]byte("abcd"), 0), copyBack(5, 4, 4)), "abcdabcd"},
		{"a copy reaching further back than it is long",
			built(10, literal([]byte("abcdef"), 0), copyBack(3, 6, 4)), "abcdefabcd"},
		{"a copy that overlaps itself",
			built(6, literal([]byte("ab"), 0), copyBack(3, 2, 4)), "ababab"},
		{"a run written as a copy reaching one byte back",
			built(9, literal([]byte("z"), 0), copyBack(2, 1, 8)), "zzzzzzzzz"},
		{"a copy of what a copy wrote",
			built(12, literal([]byte("ab"), 0), copyBack(2, 2, 4), copyBack(3, 6, 6)), "abababababab"},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := snappy.Decode(nil, c.block)
			if err != nil {
				t.Fatalf("Decode: %v", err)
			}
			if string(got) != c.want {
				t.Errorf("Decode = %q, want %q", got, c.want)
			}
		})
	}
}

func TestDecodeRefused(t *testing.T) {
	for _, c := range []struct {
		name  string
		block []byte
	}{
		{"no bytes at all", nil},
		{"a length that stops half way", []byte{0x80}},
		{"a length written in more than ten bytes",
			append(bytes.Repeat([]byte{0x80}, 9), 0x02)},
		{"a length longer than any block can be",
			binary.AppendUvarint(nil, 1<<32)},
		{"a literal whose length runs off the end of the block",
			built(10, []byte{0xf4, 0x01})},
		{"a literal longer than the block says it holds",
			built(4, literal(make([]byte, 10), 1))},
		{"a literal of four bytes of length longer than any block",
			built(4, []byte{0xfc, 0xff, 0xff, 0xff, 0xff})},
		{"a literal with fewer bytes behind it than it says",
			built(10, []byte{0x24, 'a', 'b'})},
		{"a long literal with fewer bytes behind it than it says",
			built(100, []byte{0xf0, 69, 'a', 'b'})},
		{"a short copy with nothing behind its tag",
			built(10, literal([]byte("abcd"), 0), []byte{0x01})},
		{"a copy with half an offset behind its tag",
			built(10, literal([]byte("abcd"), 0), []byte{0x02, 0x00})},
		{"a copy with half a wide offset behind its tag",
			built(10, literal([]byte("abcd"), 0), []byte{0x03, 0x00, 0x00, 0x00})},
		{"a copy reaching back to before the block",
			built(8, literal([]byte("abcd"), 0), copyBack(3, 100, 4))},
		{"a copy reaching back nowhere at all",
			built(8, literal([]byte("abcd"), 0), copyBack(3, 0, 4))},
		{"a wide copy reaching back further than a block can be",
			built(8, literal([]byte("abcd"), 0), copyBack(5, 1<<31, 4))},
		{"a copy longer than the block says it holds",
			built(8, literal([]byte("abcd"), 0), copyBack(3, 4, 20))},
		{"a block that holds less than it says",
			built(20, literal([]byte("abcd"), 0))},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := snappy.Decode(nil, c.block)
			if !errors.Is(err, snappy.ErrCorrupt) {
				t.Fatalf("Decode = %q, %v, want an ErrCorrupt", got, err)
			}
			if got != nil {
				t.Errorf("Decode returned %d bytes with its error, want none", len(got))
			}
		})
	}
}

func TestDecodedLen(t *testing.T) {
	block, want := blockOf(t, "text")

	got, err := snappy.DecodedLen(block)
	if err != nil {
		t.Fatalf("DecodedLen: %v", err)
	}
	if got != len(want) {
		t.Errorf("DecodedLen = %d, want %d", got, len(want))
	}

	// It reads the length and stops, so a block with nothing behind the length
	// still answers, and that is what makes it worth having on its own.
	if got, err = snappy.DecodedLen(binary.AppendUvarint(nil, 1<<20)); err != nil {
		t.Fatalf("DecodedLen of a block with no tags: %v", err)
	}
	if got != 1<<20 {
		t.Errorf("DecodedLen of a block with no tags = %d, want %d", got, 1<<20)
	}

	if _, err = snappy.DecodedLen(nil); !errors.Is(err, snappy.ErrCorrupt) {
		t.Errorf("DecodedLen of no bytes = %v, want an ErrCorrupt", err)
	}
}

func BenchmarkDecode(b *testing.B) {
	for _, name := range corpus(b) {
		b.Run(name, func(b *testing.B) {
			block, want := blockOf(b, name)

			dst := make([]byte, len(want))
			b.SetBytes(int64(len(want)))
			b.ReportAllocs()
			b.ResetTimer()

			for b.Loop() {
				var err error
				if dst, err = snappy.Decode(dst, block); err != nil {
					b.Fatalf("Decode: %v", err)
				}
			}
		})
	}
}
