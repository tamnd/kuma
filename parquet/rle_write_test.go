package parquet_test

import (
	"bytes"
	"math/bits"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/parquet"
)

// encoded is what the encoder makes of vals at the given width.
func encoded(t *testing.T, width int, vals []int32) []byte {
	t.Helper()

	e, err := parquet.NewRLEEncoder(width)
	if err != nil {
		t.Fatalf("NewRLEEncoder(%d): %v", width, err)
	}
	if err := e.Write(vals); err != nil {
		t.Fatalf("Write: %v", err)
	}
	return e.Finish()
}

// decoded reads values back at the given width, taking as many as were written
// and leaving whatever the last group was padded with.
func decoded(t *testing.T, width int, data []byte, n int) []int32 {
	t.Helper()

	d, err := parquet.NewRLEDecoder(data, width)
	if err != nil {
		t.Fatalf("NewRLEDecoder(%d): %v", width, err)
	}
	out := make([]int32, n)
	got, err := d.Read(out)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if got != n {
		t.Fatalf("read %d values, want %d", got, n)
	}
	return out
}

// TestRLEEncoderBytes checks the runs the encoder picks and the bytes it writes
// them as.
//
// What makes an encoder of this worth anything is which runs it picks, since the
// same values are a legal file written any number of ways. So these are the
// bytes and not just the values: a change that still round trips and writes
// twice as much would pass a round trip and fail here.
func TestRLEEncoderBytes(t *testing.T) {
	for _, c := range []struct {
		name  string
		width int
		vals  []int32
		want  []byte
	}{
		{
			"nothing at all, which is no runs and no bytes",
			1, nil, nil,
		},
		{
			"a value repeated eight times, which is a count and the value",
			1, []int32{1, 1, 1, 1, 1, 1, 1, 1},
			[]byte{8 << 1, 0x01},
		},
		{
			"the levels of a column with no nulls, which is why this encoding exists",
			1, slices.Repeat([]int32{1}, 100000),
			[]byte{0xc0, 0x9a, 0x0c, 0x01},
		},
		{
			"eight values that are not a repeat, which is one packed group",
			2, []int32{0, 1, 2, 3, 0, 1, 2, 3},
			[]byte{1<<1 | 1, 0b11100100, 0b11100100},
		},
		{
			"seven values, which is a group padded out with noughts",
			1, []int32{1, 0, 1, 0, 1, 0, 1},
			[]byte{1<<1 | 1, 0b01010101},
		},
		{
			"a repeat of seven, which is too short to be one and is packed",
			1, []int32{1, 1, 1, 1, 1, 1, 1},
			[]byte{1<<1 | 1, 0b01111111},
		},
		{
			"a packed run and then a repeat, which is two runs",
			2, []int32{0, 1, 2, 3, 0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1},
			[]byte{1<<1 | 1, 0b11100100, 0b11100100, 8 << 1, 0x01},
		},
		{
			"a width of ten, whose repeated value takes two bytes",
			10, slices.Repeat([]int32{1000}, 8),
			[]byte{8 << 1, 0xe8, 0x03},
		},
		{
			"a width of nought, which is a column with one possible value",
			0, slices.Repeat([]int32{0}, 8),
			[]byte{8 << 1},
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			got := encoded(t, c.width, c.vals)
			if !bytes.Equal(got, c.want) {
				t.Errorf("wrote %x, want %x", got, c.want)
			}
			if len(c.vals) > 0 {
				if read := decoded(t, c.width, got, len(c.vals)); !slices.Equal(read, c.vals) {
					t.Errorf("read back %v, want %v", read, c.vals)
				}
			}
		})
	}
}

// TestRLEEncoderRoundTrip reads back what the encoder wrote, at every width and
// over the shapes the values come in.
//
// The bytes written by hand say the encoder agrees with the format on the shapes
// it was shown. This says the two halves of the package agree with each other on
// everything else, which is what a file written here and read here rests on.
func TestRLEEncoderRoundTrip(t *testing.T) {
	shapes := []struct {
		name string
		of   func(width int) []int32
	}{
		{"all the same, which is one repeat", func(int) []int32 {
			return slices.Repeat([]int32{0}, 1000)
		}},
		{"counting up and round, which packs", func(width int) []int32 {
			out := make([]int32, 1000)
			for i := range out {
				out[i] = int32(i) & int32(1<<uint(width)-1)
			}
			return out
		}},
		{"runs of seven, which are each too short to repeat", func(width int) []int32 {
			var out []int32
			for i := range 200 {
				out = append(out, slices.Repeat([]int32{int32(i) & int32(1<<uint(width)-1)}, 7)...)
			}
			return out
		}},
		{"long runs with a value in between, which is repeats and packing", func(width int) []int32 {
			var out []int32
			for i := range 20 {
				out = append(out, slices.Repeat([]int32{0}, 50)...)
				out = append(out, int32(i)&int32(1<<uint(width)-1))
			}
			return out
		}},
		{"one value, which is a group of eight with seven noughts behind it", func(int) []int32 {
			return []int32{0}
		}},
	}

	for _, s := range shapes {
		t.Run(s.name, func(t *testing.T) {
			for width := range maxTestWidth + 1 {
				vals := s.of(width)
				got := decoded(t, width, encoded(t, width, vals), len(vals))
				if !slices.Equal(got, vals) {
					t.Fatalf("at %d bits read %d values back wrong", width, len(vals))
				}
			}
		})
	}
}

// maxTestWidth is the widest column the round trip is run at, which is the
// widest the format has: a dictionary index is an int32.
const maxTestWidth = 32

// TestRLEEncoderWide checks a value at the top of every width.
//
// A value of thirty two bits is negative as an int32 and is the one the packing
// arithmetic would drop a bit of, since the accumulator it goes through holds
// sixty four and a value can start seven bits into a byte.
func TestRLEEncoderWide(t *testing.T) {
	for width := 1; width <= maxTestWidth; width++ {
		top := int32(uint32(1)<<uint(width) - 1)
		vals := []int32{top, 0, top, 0, top, 0, top, 0, top}

		got := decoded(t, width, encoded(t, width, vals), len(vals))
		if !slices.Equal(got, vals) {
			t.Errorf("at %d bits read %v, want %v", width, got, vals)
		}
	}
}

// TestRLEEncoderRuns checks that a page written in several calls is the page
// written in one.
//
// A page is filled from a column that arrives in chunks, so the levels of one
// page are written in as many calls as the column had chunks. A run that crosses
// a call has to carry on rather than being ended and started again, and this is
// the one thing here that could quietly turn a file into a bigger file that
// still reads.
func TestRLEEncoderRuns(t *testing.T) {
	vals := slices.Repeat([]int32{0, 0, 0, 0, 0, 1}, 100)

	e, err := parquet.NewRLEEncoder(1)
	if err != nil {
		t.Fatal(err)
	}
	for at := 0; at < len(vals); at += 7 {
		if err := e.Write(vals[at:min(at+7, len(vals))]); err != nil {
			t.Fatal(err)
		}
	}

	if got, want := e.Finish(), encoded(t, 1, vals); !bytes.Equal(got, want) {
		t.Errorf("wrote %x in pieces, want %x", got, want)
	}
}

// TestRLEEncoderLongPackedRun checks the run that has to be split.
//
// The header of a packed run is one byte, which is what lets it be written in
// front of the groups it counts, and one byte counts sixty three groups. So a
// page of values that never repeat is a run of sixty three groups, then another
// header, and the sixty fourth group behind it.
func TestRLEEncoderLongPackedRun(t *testing.T) {
	const groups = 65

	vals := make([]int32, 8*groups)
	for i := range vals {
		vals[i] = int32(i) & 0xff
	}

	got := encoded(t, 8, vals)
	want := []byte{63<<1 | 1}
	for i := range 63 * 8 {
		want = append(want, byte(vals[i]))
	}
	want = append(want, 2<<1|1)
	for i := 63 * 8; i < len(vals); i++ {
		want = append(want, byte(vals[i]))
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("wrote %d bytes, want %d", len(got), len(want))
	}

	if read := decoded(t, 8, got, len(vals)); !slices.Equal(read, vals) {
		t.Error("the values did not come back")
	}
}

// TestRLEEncoderFinishTwice checks that finishing a finished encoder is the same
// answer rather than more bytes.
//
// A page writer that asks for the bytes, decides the page is not full after all
// and asks again is not a thing this package does, but a caller has no way of
// knowing that from the outside and the runs are all closed either way.
func TestRLEEncoderFinishTwice(t *testing.T) {
	e, err := parquet.NewRLEEncoder(2)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Write([]int32{1, 2, 3}); err != nil {
		t.Fatal(err)
	}

	first := slices.Clone(e.Finish())
	if got := e.Finish(); !bytes.Equal(got, first) {
		t.Errorf("finished twice gives %x, want %x", got, first)
	}
}

// TestRLEEncoderReset checks that an encoder put back to the start writes what a
// new one would.
func TestRLEEncoderReset(t *testing.T) {
	e, err := parquet.NewRLEEncoder(4)
	if err != nil {
		t.Fatal(err)
	}
	if err := e.Write([]int32{1, 2, 3, 4, 5, 6, 7, 8, 9}); err != nil {
		t.Fatal(err)
	}
	if got := e.Len(); got == 0 {
		t.Error("a whole group of values wrote nothing at all")
	}

	if err := e.Reset(1); err != nil {
		t.Fatal(err)
	}
	if got := e.Len(); got != 0 {
		t.Errorf("%d bytes after a reset, want none", got)
	}
	if err := e.Write([]int32{1, 1, 1, 1, 1, 1, 1, 1}); err != nil {
		t.Fatal(err)
	}

	want := []byte{8 << 1, 0x01}
	if got := e.Finish(); !bytes.Equal(got, want) {
		t.Errorf("wrote %x, want %x", got, want)
	}
}

// TestRLEEncoderErrors checks the two things a caller can get wrong.
//
// A value that does not fit the width would be read back as a different value,
// which for levels is nulls in the wrong places and for indices is the wrong
// values out of the dictionary. Neither is something to write and find out
// about later.
func TestRLEEncoderErrors(t *testing.T) {
	for _, c := range []struct {
		name  string
		width int
		vals  []int32
		want  string
	}{
		{"a width no dictionary index could need", 33, nil, "33 bits"},
		{"a negative width", -1, nil, "-1 bits"},
		{"a value wider than the width", 2, []int32{0, 1, 4}, "a value of 4 at 2"},
		{"a negative value, which no level or index is", 4, []int32{-1}, "a value of -1 at 0"},
	} {
		t.Run(c.name, func(t *testing.T) {
			e, err := parquet.NewRLEEncoder(c.width)
			if c.vals == nil {
				if err == nil {
					t.Fatalf("a width of %d was taken", c.width)
				}
				if !strings.Contains(err.Error(), c.want) {
					t.Errorf("got %q, want it to mention %q", err, c.want)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}

			err = e.Write(c.vals)
			if err == nil {
				t.Fatalf("%v was written at %d bits", c.vals, c.width)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("got %q, want it to mention %q", err, c.want)
			}
		})
	}
}

// TestRLEEncoderRealLevels writes the definition levels of a real file and reads
// them back.
//
// The levels come out of a page pyarrow wrote, so what they are is a real
// column's nulls rather than a pattern chosen here. Comparing the bytes against
// pyarrow's own would be comparing two writers' taste in runs, which is not
// something the format asks anybody to agree on, so what is checked is that the
// levels come back and that they do not take more room than pyarrow took.
func TestRLEEncoderRealLevels(t *testing.T) {
	pages, column := pagesOf(t, "pages.parquet", "maybe")
	width := bits.Len(uint(column.MaxDefinition))

	for i, p := range pages {
		want := levels(t, p.Data[:p.DefinitionLength], column, p.NumValues)

		got := encoded(t, width, want)
		if read := decoded(t, width, got, len(want)); !slices.Equal(read, want) {
			t.Fatalf("page %d did not come back", i)
		}
		if len(got) > int(p.DefinitionLength) {
			t.Errorf("page %d took %d bytes, where pyarrow took %d", i, len(got), p.DefinitionLength)
		}
	}
}

// BenchmarkRLEWrite encodes the shapes a page of levels and a page of indices
// come in.
//
// Levels with no nulls in them are one long repeat and should cost almost
// nothing however many rows there are. Indices that jump about are packed, which
// is a few instructions per value and is what a column of a million dictionary
// encoded strings pays to be written.
func BenchmarkRLEWrite(b *testing.B) {
	const values = 100000

	ones := slices.Repeat([]int32{1}, values)
	mixed := make([]int32, values)
	indices := make([]int32, values)
	for i := range values {
		if i%3 != 0 {
			mixed[i] = 1
		}
		indices[i] = int32(i) & 0x3ff
	}

	for _, c := range []struct {
		name  string
		width int
		vals  []int32
	}{
		{"levels of a column with no nulls", 1, ones},
		{"levels of a column with nulls", 1, mixed},
		{"dictionary indices of ten bits", 10, indices},
	} {
		b.Run(c.name, func(b *testing.B) {
			e, err := parquet.NewRLEEncoder(c.width)
			if err != nil {
				b.Fatal(err)
			}
			if err := e.Write(c.vals); err != nil {
				b.Fatal(err)
			}
			b.SetBytes(int64(len(e.Finish())))

			for b.Loop() {
				if err := e.Reset(c.width); err != nil {
					b.Fatal(err)
				}
				if err := e.Write(c.vals); err != nil {
					b.Fatal(err)
				}
				e.Finish()
			}
		})
	}
}
