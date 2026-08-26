package ipc_test

import (
	"testing"

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
		{Key: "unit", Value: "metres"},
		{Key: "source", Value: "the shipping system"},
	}
)

var (
	formatSink string
	typeSink   dtype.DataType
	metaSink   dtype.Metadata
	blobSink   []byte
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
