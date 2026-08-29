package parquet

import (
	"slices"

	"github.com/tamnd/kuma/array"
)

// Writing a column as a dictionary.
//
// This is dict.go the other way round, and it is the encoding that makes a
// parquet file smaller than the values in it. A chunk of ten million country
// codes holds two hundred and fifty strings once, in a page of its own in front
// of the data pages, and the rows are indices into that page written as runs. A
// file written without it is several times larger and slower to read, and it is
// slower to read twice over: there are more bytes to get off the disk, and the
// column comes back as ten million strings rather than as ten million small
// integers pointing at two hundred and fifty.
//
// The decision is taken per chunk and taken before anything is written. A writer
// that started a chunk as a dictionary and gave up part way through would leave
// indices at the front of the chunk and values at the back, which the reader
// next door refuses by name, and rightly: reading it would mean expanding the
// dictionary into the column, which is the one thing reading a dictionary this
// way is for. So the values are walked once to build the dictionary, and if it
// grows past what a dictionary is worth the whole chunk is written plainly
// instead. The walk costs a hash of every value and it is the price of the
// decision being right rather than nearly right.
//
// Two types have no dictionary. A boolean has two values and a page holds one in
// a bit, so a dictionary of them is larger than the values. A float is left out
// for a different reason: a dictionary is a map and equality there is not
// identity, since a negative zero equals a positive one and would come back as
// whichever went in first, and a NaN equals nothing at all including itself, so
// a chunk of them would fill the dictionary with values none of which could ever
// be found in it again.

// What a dictionary is allowed to grow to before the chunk is written plainly
// instead.
//
// The count is what the width of an index follows from, and past about this many
// values an index is wide enough that the runs stop paying for the page in front
// of them. The byte limit is the same idea for the values themselves: a
// dictionary page is a page, and one that is larger than the pages it saves is
// not saving anything. Both are what the other writers use.
const (
	maxDictValues = 1 << 17
	maxDictBytes  = 1 << 20
)

// dictionary is the distinct values of one column chunk, in the order they were
// first seen.
//
// It is an interface because what a value is depends on the column, and the two
// implementations are the two answers: a number, which is a key on its own, and
// a run of bytes, which is a key by its contents.
type dictionary interface {
	// scan folds rows [i, j) of a in and says whether the dictionary is still
	// worth having. A dictionary that says no is not asked anything else, and
	// the chunk it was being built for is written plainly.
	scan(a *array.Array, i, j int) bool

	// indices appends the index of every value of rows [i, j) that is there,
	// which is what the data pages of a dictionary encoded chunk hold. The
	// missing rows are in the levels and nowhere else, the same as the values
	// of a plain page.
	indices(a *array.Array, i, j int, out []int32) []int32

	// write puts the values down the way a plain page puts them, which is what
	// a dictionary page is.
	write(e *PlainEncoder)

	// size is how many distinct values there are. The width of an index
	// follows from it, and a dictionary of nothing is not written at all.
	size() int

	// reset empties it for the next chunk. The dictionary of a chunk is the
	// chunk's own, since the indices of its pages mean nothing anywhere else.
	reset()
}

// numberDict is the dictionary of a column of numbers, which is every integer
// kuma has and the dates, times and timestamps that travel as one.
//
// T is what the column holds and W is what parquet writes, the same pair as the
// page writer, so a uint64 is keyed as a uint64 and written as the int64 with
// the same bits.
type numberDict[T array.Numeric, W array.Numeric] struct {
	at   map[T]int32
	vals []T
	buf  []W

	// put writes the values of the dictionary page, and add folds a value into
	// the bounds of the chunk.
	put func(*PlainEncoder, []W)
	add func(T)
}

// scan folds rows [i, j) in.
func (d *numberDict[T, W]) scan(a *array.Array, i, j int) bool {
	if d.at == nil {
		d.at = make(map[T]int32)
	}

	vals := a.Values[T]()
	for k := i; k < j; k++ {
		if !a.IsValid(k) {
			continue
		}

		v := vals[k]
		if _, ok := d.at[v]; ok {
			continue
		}
		if len(d.vals) >= maxDictValues {
			return false
		}

		// A value the dictionary has not got is a value nothing else has seen
		// either, so this is also where the bounds of the chunk are told about
		// it. That is the whole of what a dictionary saves the bounds: the
		// distinct values of a column are compared rather than all of them.
		d.at[v] = int32(len(d.vals))
		d.vals = append(d.vals, v)
		d.add(v)
	}
	return true
}

// indices appends the index of every value of rows [i, j) that is there.
func (d *numberDict[T, W]) indices(a *array.Array, i, j int, out []int32) []int32 {
	vals := a.Values[T]()
	for k := i; k < j; k++ {
		if a.IsValid(k) {
			out = append(out, d.at[vals[k]])
		}
	}
	return out
}

// write puts the values down as a dictionary page.
func (d *numberDict[T, W]) write(e *PlainEncoder) {
	d.buf = slices.Grow(d.buf[:0], len(d.vals))
	for _, v := range d.vals {
		d.buf = append(d.buf, W(v))
	}
	d.put(e, d.buf)
}

// size is how many distinct values there are.
func (d *numberDict[T, W]) size() int { return len(d.vals) }

// reset empties it for the next chunk, keeping the map and the buffer, since a
// file is one chunk of a column after another and they are usually of a size.
func (d *numberDict[T, W]) reset() {
	clear(d.at)
	d.vals = d.vals[:0]
}

// blobDict is the dictionary of a column whose values are bytes, which is where
// a dictionary earns the most: a column of short strings out of a small set is
// most of what anybody stores.
//
// The keys are the values as strings, which is the one shape Go can look bytes
// up by, and the map is only ever added to with a string that is already there
// as a key, so the copy is one per distinct value rather than one per row. What
// is kept is the value where it sits in the column rather than a copy of it,
// since the column outlives the chunk being written out of it.
type blobDict struct {
	at    map[string]int32
	vals  [][]byte
	bytes int

	put func(*PlainEncoder, [][]byte)
	add func([]byte)
}

// scan folds rows [i, j) in.
func (d *blobDict) scan(a *array.Array, i, j int) bool {
	if d.at == nil {
		d.at = make(map[string]int32)
	}

	for k := i; k < j; k++ {
		if !a.IsValid(k) {
			continue
		}

		v := a.Bytes(k)
		if _, ok := d.at[string(v)]; ok {
			continue
		}
		if len(d.vals) >= maxDictValues || d.bytes+len(v) > maxDictBytes {
			return false
		}

		d.at[string(v)] = int32(len(d.vals))
		d.vals = append(d.vals, v)
		d.bytes += len(v)
		d.add(v)
	}
	return true
}

// indices appends the index of every value of rows [i, j) that is there.
func (d *blobDict) indices(a *array.Array, i, j int, out []int32) []int32 {
	for k := i; k < j; k++ {
		if a.IsValid(k) {
			out = append(out, d.at[string(a.Bytes(k))])
		}
	}
	return out
}

// write puts the values down as a dictionary page.
func (d *blobDict) write(e *PlainEncoder) { d.put(e, d.vals) }

// size is how many distinct values there are.
func (d *blobDict) size() int { return len(d.vals) }

// reset empties it for the next chunk.
func (d *blobDict) reset() {
	clear(d.at)
	d.vals = d.vals[:0]
	d.bytes = 0
}
