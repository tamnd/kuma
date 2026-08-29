package parquet

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// leaf builds a column that is written as one physical type and means another,
// which is the pair bloomHash has to put back together.
func leaf(physical Type, meant dtype.DataType) Column {
	return Column{Path: []string{"c"}, Element: SchemaElement{Type: physical}, Type: meant}
}

// openStats opens the file in testdata that carries statistics, which is the
// one the tests in here filter on. The tests outside the package read the rest
// of them and have a helper of their own for it.
func openStats(t *testing.T) *FileReader {
	t.Helper()

	b, err := os.ReadFile(filepath.Join("testdata", "stats.parquet"))
	if err != nil {
		t.Fatal(err)
	}
	r, err := NewFileReader(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// oneValue builds a column of a single value of a type array.Of cannot name.
func oneValue(t *testing.T, dt dtype.DataType, put func(*array.Builder)) *array.Array {
	t.Helper()

	b, err := array.NewBuilder(dt)
	if err != nil {
		t.Fatal(err)
	}
	put(b)
	return b.Finish()
}

// TestBloomHash checks the value is hashed as the bytes the file holds it in.
//
// A bloom filter holds the hash of a value as a page writes it, so a lookup has
// to put the value back into the file's own shape. Parquet writes every integer
// of thirty two bits or fewer in four bytes, signed or not, and eight of them in
// eight, so an int8 column and a uint32 column are hashed the same width and a
// reader that hashed the Go value would look in the wrong block of the filter
// and report that a chunk holding the value does not.
//
// The bytes are written out here rather than worked out, since the whole point
// is that they are the writer's bytes and not this reader's idea of them.
func TestBloomHash(t *testing.T) {
	for _, c := range []struct {
		name string
		col  Column
		val  *array.Array
		want []byte
	}{
		{
			"an int8 in the four bytes the file wrote it in",
			leaf(Int32, dtype.Int8), array.Of(int8(-2)),
			[]byte{0xfe, 0xff, 0xff, 0xff},
		},
		{
			"an int16",
			leaf(Int32, dtype.Int16), array.Of(int16(-300)),
			[]byte{0xd4, 0xfe, 0xff, 0xff},
		},
		{
			"an int32",
			leaf(Int32, dtype.Int32), array.Of(int32(7)),
			[]byte{0x07, 0x00, 0x00, 0x00},
		},
		{
			"a uint8",
			leaf(Int32, dtype.Uint8), array.Of(uint8(200)),
			[]byte{0xc8, 0x00, 0x00, 0x00},
		},
		{
			"a uint16",
			leaf(Int32, dtype.Uint16), array.Of(uint16(40000)),
			[]byte{0x40, 0x9c, 0x00, 0x00},
		},
		{
			"a uint32 large enough to be negative in the int32 it is written in",
			leaf(Int32, dtype.Uint32), array.Of(uint32(4294967295)),
			[]byte{0xff, 0xff, 0xff, 0xff},
		},
		{
			"a date, which is an int32 with a calendar on it",
			leaf(Int32, dtype.Date32), oneValue(t, dtype.Date32, func(b *array.Builder) {
				b.Append(int32(19000))
			}),
			[]byte{0x38, 0x4a, 0x00, 0x00},
		},
		{
			"an int64",
			leaf(Int64, dtype.Int64), array.Of(int64(-2)),
			[]byte{0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
		},
		{
			"a uint64",
			leaf(Int64, dtype.Uint64), array.Of(uint64(1)),
			[]byte{0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
		},
		{
			"a float",
			leaf(Float, dtype.Float32), array.Of(float32(1.5)),
			[]byte{0x00, 0x00, 0xc0, 0x3f},
		},
		{
			"a double",
			leaf(Double, dtype.Float64), array.Of(1.5),
			[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x3f},
		},
		{
			"a string, which is hashed without the length a page puts in front of it",
			leaf(ByteArray, dtype.String), array.OfStrings("GB"),
			[]byte("GB"),
		},
		{
			"a run of bytes",
			leaf(ByteArray, dtype.Binary), oneValue(t, dtype.Binary, func(b *array.Builder) {
				b.AppendBytes([]byte{0x00, 0x01})
			}),
			[]byte{0x00, 0x01},
		},
		{
			"a run of bytes whose width is in the schema",
			leaf(FixedLenByteArray, dtype.FixedSizeBinary{ByteWidth: 2}),
			oneValue(t, dtype.FixedSizeBinary{ByteWidth: 2}, func(b *array.Builder) {
				b.AppendBytes([]byte{0x0a, 0x0b})
			}),
			[]byte{0x0a, 0x0b},
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, ok := bloomHash(&c.col, c.val)
			if !ok {
				t.Fatal("the value could not be hashed")
			}
			if want := xxh64(c.want); got != want {
				t.Errorf("hashed to %d, want %d, which is the hash of %x", got, want, c.want)
			}
		})
	}
}

// TestBloomHashNone checks the values a filter cannot be asked about.
//
// Answering yes to all of those is the whole of what happens next, since a
// filter that is not consulted rules nothing out and the bounds are left to do
// the work. Answering with a hash of the wrong bytes would rule out a chunk that
// holds the value, which is a wrong answer rather than a slow one.
func TestBloomHashNone(t *testing.T) {
	for _, c := range []struct {
		name string
		col  Column
		val  *array.Array
	}{
		{
			"a boolean, whose values are bits and have no byte to hash",
			leaf(Boolean, dtype.Bool), array.OfBools(true),
		},
		{
			"an int96, which no writer has built a filter over",
			leaf(Int96, dtype.Int64), array.Of(int64(1)),
		},
		{
			"a value of a type the column is not",
			leaf(Int64, dtype.Int64), array.Of(int32(1)),
		},

		// A schema saying one thing about how a column is written and another
		// about what it means, which is a file rather than a column this reader
		// would build. What matters is that it comes back with no hash rather
		// than with a hash of whatever the bytes happened to be.
		{
			"four bytes said to hold something that is not four bytes",
			leaf(Int32, dtype.Float32), array.Of(float32(1)),
		},
		{
			"eight bytes said to hold something that is not eight bytes",
			leaf(Int64, dtype.Float64), array.Of(1.0),
		},
		{
			"a float said to hold a double",
			leaf(Float, dtype.Float64), array.Of(1.0),
		},
		{
			"a double said to hold a float",
			leaf(Double, dtype.Float32), array.Of(float32(1)),
		},
		{
			"a byte array said to hold a number",
			leaf(ByteArray, dtype.Int32), array.Of(int32(1)),
		},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, ok := bloomHash(&c.col, c.val); ok {
				t.Error("that was hashed")
			}
		})
	}
}

// TestKeepUnchecked checks that a predicate reaching the row groups without
// having been looked over is still refused rather than acted on.
//
// RowGroups checks the whole filter before it reads anything, so nothing it
// hands to a row group is in this state. Keep checks again because it is public
// and a caller walking the groups itself arrives the other way round, and this
// is the path that error takes back out.
func TestKeepUnchecked(t *testing.T) {
	r := openStats(t)
	bad := []test{{pred: Predicate{Column: "n", Op: kernel.OpEq}, column: 0}}

	_, err := r.keep(0, bad)
	if err == nil {
		t.Fatal("a predicate with no value worked")
	}
	if !strings.Contains(err.Error(), "compares against no value") {
		t.Errorf("got %q, want it to mention the missing value", err)
	}
}

// TestCompareUnanswered checks a bound that is not there.
//
// ReadBounds gives a pair of values or no pair at all, so this is not a state a
// file arrives in. It is what the arithmetic does with a comparison nobody can
// answer, which is to read the chunk, and it is worth being sure of because the
// other answer would skip on a comparison that never happened.
func TestCompareUnanswered(t *testing.T) {
	b, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		t.Fatal(err)
	}
	b.AppendNull()
	b.Append(int64(11))

	keep, err := Where("n", kernel.OpEq, int64(5)).Keep(Bounds{Count: 4, Values: b.Finish()})
	if err != nil {
		t.Fatalf("Keep: %v", err)
	}
	if !keep {
		t.Error("a chunk whose bounds could not be compared was skipped")
	}
}

// TestHoldsNaN checks the four booleans a NaN produces, which is all of them
// false, against every operator.
//
// Nothing is smaller than a NaN, larger than one or equal to one, so the range
// of a chunk sits nowhere in relation to it and every comparison but the one
// that is true of everything comes out false.
func TestHoldsNaN(t *testing.T) {
	for _, c := range []struct {
		op   kernel.CompareOp
		want bool
	}{
		{kernel.OpEq, false},
		{kernel.OpNe, true},
		{kernel.OpLt, false},
		{kernel.OpLe, false},
		{kernel.OpGt, false},
		{kernel.OpGe, false},
	} {
		if got := holds(c.op, false, false, false, false); got != c.want {
			t.Errorf("%s against a NaN gives %v, want %v", c.op, got, c.want)
		}
	}

	// And the same thing the long way round, so that the four booleans really
	// are what a NaN produces rather than what this test assumed.
	nan := Where("ratio", kernel.OpEq, math.NaN())
	got, err := nan.Keep(Bounds{Count: 4, Values: array.Of(0.0, 1.0)})
	if err != nil {
		t.Fatalf("Keep: %v", err)
	}
	if got {
		t.Error("a chunk of ordinary numbers was kept for a NaN")
	}
}
