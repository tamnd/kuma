package parquet

import (
	"strings"
	"testing"
)

// The hash against the numbers everybody else gets.
//
// A hash is only right if it agrees with the one the writer used, and there is
// nothing in a file to check that against, so the check has to be against the
// values in the specification and in the reference implementation. The strings
// below are the ones the reference publishes. The lengths after them are there
// because the hash does different arithmetic on every length class, and a
// mistake in the tail is a mistake on exactly the short values a bloom filter
// on identifiers is made of.

// vector is the bytes the length cases are cut from, which is any old sequence
// as long as it is the same one the values were taken from.
func vector(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*37 + 11)
	}
	return b
}

// TestXXH64 is the published values of the hash.
func TestXXH64(t *testing.T) {
	tests := []struct {
		in   string
		want uint64
	}{
		{"", 0xef46db3751d8e999},
		{"a", 0xd24ec4f1a98c6e5b},
		{"abc", 0x44bc2cf5ad770999},
		{"abcd", 0xde0327b0d25d92cc},
		{"the quick brown fox jumps over the lazy dog", 0xed714233c5a9a792},
		{strings.Repeat("x", 100), 0x92f0de5a88a3c094},
	}

	for _, tt := range tests {
		if got := xxh64(tt.in); got != tt.want {
			t.Errorf("xxh64(%q) = %#016x, want %#016x", tt.in, got, tt.want)
		}
		if got := xxh64([]byte(tt.in)); got != tt.want {
			t.Errorf("xxh64(%q) as bytes = %#016x, want %#016x", tt.in, got, tt.want)
		}
	}
}

// TestXXH64Lengths is one input of every length that changes what the hash
// does.
//
// Under thirty two bytes it starts from a constant and over it from four
// accumulators, the tail is eaten eight bytes then four then one at a time, and
// every one of those boundaries is a place to be off by one. A value in a bloom
// filter is four or eight or twelve bytes or a name, so the short ones are not
// the edge cases here, they are the whole use.
func TestXXH64Lengths(t *testing.T) {
	tests := []struct {
		n    int
		want uint64
	}{
		{0, 0xef46db3751d8e999},
		{1, 0xf592c0c7639c4cb6},
		{2, 0x78f1ae91e3b05e99},
		{3, 0x22c08528601d4f27},
		{4, 0xfb1e5cf2f1ae4d95},
		{5, 0x9d213eca1f68f273},
		{7, 0x5613ac510496c04e},
		{8, 0x57cb2b7521f3e21a},
		{9, 0xc3f1d09775257b66},
		{15, 0x90a9714eb00e8d29},
		{16, 0xc6843cef99721174},
		{31, 0xe4a0e629e519a4ae},
		{32, 0xcc6b8aaada790b2d},
		{33, 0x35ec49850475a832},
		{63, 0xbf9f0ba3cf95b28a},
		{64, 0x155ccce4bf32befc},
		{100, 0x4826e367566ea023},
		{200, 0x2f074b6dd9094e34},
	}

	for _, tt := range tests {
		b := vector(tt.n)
		if got := xxh64(b); got != tt.want {
			t.Errorf("xxh64 of %d bytes = %#016x, want %#016x", tt.n, got, tt.want)
		}
		if got := xxh64(string(b)); got != tt.want {
			t.Errorf("xxh64 of %d bytes as a string = %#016x, want %#016x", tt.n, got, tt.want)
		}
	}
}

// BenchmarkXXH64 is the hash of the two things a bloom filter is asked about,
// which is a number as a page writes it and a name.
func BenchmarkXXH64(b *testing.B) {
	b.Run("a number", func(b *testing.B) {
		v := vector(8)
		for b.Loop() {
			sink = xxh64(v)
		}
	})

	b.Run("a name", func(b *testing.B) {
		v := "user-0042"
		for b.Loop() {
			sink = xxh64(v)
		}
	})
}

// sink is where the benchmarks put their answer, so that the work is not thrown
// away before it is done.
var sink uint64
