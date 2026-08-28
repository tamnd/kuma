package parquet

import "math/bits"

// XXH64, which is the hash a parquet bloom filter is built on.
//
// The format names one hash and never a choice of them, so this is not a
// pluggable anything: a filter written by any writer in the world was built
// with XXH64 of the value's plain encoding with a seed of nought, and a reader
// that hashes it any other way looks in the wrong place and answers no to a
// value the chunk holds. That is the one answer a filter must never give.
//
// It is written here rather than pulled in because the core of kuma is the
// standard library and nothing else, and because this is a hundred lines of
// arithmetic with a specification and a set of test vectors behind it. It is
// unexported for now. Zstandard checksums its frames with the same hash, so if
// that decompressor arrives this moves somewhere both of them can see it.

// The five constants of XXH64, which are the odd primes the reference
// implementation picked.
const (
	prime1 uint64 = 0x9e3779b185ebca87
	prime2 uint64 = 0xc2b2ae3d27d4eb4f
	prime3 uint64 = 0x165667b19e3779f9
	prime4 uint64 = 0x85ebca77c2b2ae63
	prime5 uint64 = 0x27d4eb2f165667c5
)

// text is a value to hash, which arrives as bytes out of a footer or as a
// string out of a filter and is not copied to turn one into the other.
type text interface{ ~[]byte | ~string }

// xxh64 is XXH64 of v with a seed of nought.
//
// Long inputs are eaten thirty two bytes at a time into four accumulators,
// which is what makes the hash fast and is beside the point here, since a value
// in a bloom filter is a number or a name and the loop never runs. What matters
// is that the tail is the same tail the reference implementation has, byte for
// byte, because a hash that agrees on long inputs and not on short ones would
// pass every benchmark and fail every file.
func xxh64[T text](v T) uint64 {
	n, i := len(v), 0

	var h uint64
	if n >= 32 {
		seed := uint64(0)
		v1, v2, v3, v4 := seed+prime1+prime2, seed+prime2, seed, seed-prime1
		for ; i+32 <= n; i += 32 {
			v1 = xxhRound(v1, xxhWord(v, i))
			v2 = xxhRound(v2, xxhWord(v, i+8))
			v3 = xxhRound(v3, xxhWord(v, i+16))
			v4 = xxhRound(v4, xxhWord(v, i+24))
		}

		h = bits.RotateLeft64(v1, 1) + bits.RotateLeft64(v2, 7) +
			bits.RotateLeft64(v3, 12) + bits.RotateLeft64(v4, 18)
		h = xxhMerge(h, v1)
		h = xxhMerge(h, v2)
		h = xxhMerge(h, v3)
		h = xxhMerge(h, v4)
	} else {
		h = prime5
	}
	h += uint64(n)

	for ; i+8 <= n; i += 8 {
		h ^= xxhRound(0, xxhWord(v, i))
		h = bits.RotateLeft64(h, 27)*prime1 + prime4
	}
	if i+4 <= n {
		h ^= uint64(xxhHalf(v, i)) * prime1
		h = bits.RotateLeft64(h, 23)*prime2 + prime3
		i += 4
	}
	for ; i < n; i++ {
		h ^= uint64(v[i]) * prime5
		h = bits.RotateLeft64(h, 11) * prime1
	}

	// The avalanche, which is what makes the last byte of the input reach the
	// first bit of the answer. A bloom filter takes its block from the top half
	// of the hash and its bits from the bottom, so both halves have to be worth
	// something.
	h ^= h >> 33
	h *= prime2
	h ^= h >> 29
	h *= prime3
	h ^= h >> 32
	return h
}

// xxhRound eats eight bytes into one accumulator.
func xxhRound(acc, v uint64) uint64 {
	acc += v * prime2
	return bits.RotateLeft64(acc, 31) * prime1
}

// xxhMerge folds one of the four accumulators into the answer.
func xxhMerge(acc, v uint64) uint64 {
	return (acc^xxhRound(0, v))*prime1 + prime4
}

// xxhWord and xxhHalf read a little endian number out of the value.
//
// They are written a byte at a time rather than with encoding/binary because a
// string and a slice of bytes are hashed by the same code here, and indexing is
// the one thing both of them do. The compiler puts the bytes back together into
// a load on the platforms that have one.
func xxhWord[T text](v T, i int) uint64 {
	return uint64(v[i]) | uint64(v[i+1])<<8 | uint64(v[i+2])<<16 | uint64(v[i+3])<<24 |
		uint64(v[i+4])<<32 | uint64(v[i+5])<<40 | uint64(v[i+6])<<48 | uint64(v[i+7])<<56
}

func xxhHalf[T text](v T, i int) uint32 {
	return uint32(v[i]) | uint32(v[i+1])<<8 | uint32(v[i+2])<<16 | uint32(v[i+3])<<24
}
