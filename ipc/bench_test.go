package ipc_test

import (
	"strconv"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

// The types and the format strings a schema is actually made of. A frame
// crossing this boundary pays for one of these per column, so the number that
// matters is per call rather than per row, and a wide table is a few hundred of
// them.
var (
	benchTypes = []dtype.DataType{
		dtype.Int64,
		dtype.Float64,
		dtype.String,
		dtype.Timestamp{Unit: dtype.Microsecond, Zone: "Europe/London"},
		dtype.Decimal128{Precision: 18, Scale: 2},
		dtype.List{Elem: dtype.Int64},
	}

	benchFormats = []string{
		"l",
		"g",
		"u",
		"tsu:Europe/London",
		"d:18,2",
		"+l",
	}

	benchChildren = []dtype.Field{{Name: "item", Type: dtype.Int64, Nullable: true}}

	benchMeta = dtype.Metadata{
		{Key: "unit", Value: "meters"},
		{Key: "source", Value: "the shipping system"},
	}
)

var (
	formatSink string
	typeSink   dtype.DataType
	metaSink   dtype.Metadata
	blobSink   []byte
	layoutSink ipc.Layout
	arraySink  *array.Array
)

func BenchmarkFormat(b *testing.B) {
	b.ReportAllocs()
	for i := 0; b.Loop(); i++ {
		s, err := ipc.Format(benchTypes[i%len(benchTypes)])
		if err != nil {
			b.Fatal(err)
		}
		formatSink = s
	}
}

func BenchmarkType(b *testing.B) {
	b.ReportAllocs()
	for i := 0; b.Loop(); i++ {
		t, err := ipc.Type(benchFormats[i%len(benchFormats)], benchChildren)
		if err != nil {
			b.Fatal(err)
		}
		typeSink = t
	}
}

func BenchmarkEncodeMetadata(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		blob, err := ipc.EncodeMetadata(benchMeta)
		if err != nil {
			b.Fatal(err)
		}
		blobSink = blob
	}
}

func BenchmarkDecodeMetadata(b *testing.B) {
	blob, err := ipc.EncodeMetadata(benchMeta)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for b.Loop() {
		m, err := ipc.DecodeMetadata(blob)
		if err != nil {
			b.Fatal(err)
		}
		metaSink = m
	}
}

// The array benchmarks all use one column of this many values, so the numbers
// are per column rather than per value. Import and Export are called once per
// column per batch, and a batch is usually thousands of values, which is what
// makes the difference between the cases that walk the values and the cases
// that do not worth measuring.
const benchRows = 4096

func benchInts() *array.Array {
	values := make([]int64, benchRows)
	for i := range values {
		values[i] = int64(i)
	}
	return array.Of(values...)
}

func benchStrings() *array.Array {
	values := make([]string, benchRows)
	for i := range values {
		values[i] = "value number " + strconv.Itoa(i)
	}
	return array.OfStrings(values...)
}

func BenchmarkExport(b *testing.B) {
	for _, bb := range []struct {
		name string
		a    *array.Array
	}{
		{"int64", benchInts()},
		{"string", benchStrings()},
	} {
		b.Run(bb.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				l, err := ipc.Export(bb.a)
				if err != nil {
					b.Fatal(err)
				}
				layoutSink = l
			}
		})
	}
}

// BenchmarkImport is three different jobs under one name. A fixed width column
// wraps two buffers and counts the nulls, so it costs the same whatever the
// column holds. A column of views is checked value by value against its blocks,
// which is what buys every read after it. The two offset cases do that and
// build a view per value on top, which is the price of taking a layout kuma
// does not store.
func BenchmarkImport(b *testing.B) {
	ints, err := ipc.Export(benchInts())
	if err != nil {
		b.Fatal(err)
	}
	views, err := ipc.Export(benchStrings())
	if err != nil {
		b.Fatal(err)
	}

	values := make([]string, benchRows)
	for i := range values {
		values[i] = "value number " + strconv.Itoa(i)
	}
	narrow, _ := offsetLayout(values, false)
	wide, _ := offsetLayout(values, true)

	for _, bb := range []struct {
		name   string
		format string
		layout ipc.Layout
	}{
		{"int64", "l", ints},
		{"views", "vu", views},
		{"offsets32", "u", narrow},
		{"offsets64", "U", wide},
	} {
		b.Run(bb.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				a, err := ipc.Import(bb.format, bb.layout)
				if err != nil {
					b.Fatal(err)
				}
				arraySink = a
			}
		})
	}
}
