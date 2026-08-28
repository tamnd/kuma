//go:build cgo

package ipc_test

import (
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
	"github.com/tamnd/kuma/ipc/ipctest"
)

var (
	fieldSink    dtype.Field
	importedSink *ipc.Imported
)

// BenchmarkCExportArray is what one column costs to hand to another library.
// It is a handful of small allocations and a pin per buffer, and none of it
// depends on the number of values, which is the property worth keeping.
func BenchmarkCExportArray(b *testing.B) {
	for _, bb := range []struct {
		name string
		a    *array.Array
	}{
		{"int64", benchInts()},
		{"string", benchStrings()},
	} {
		b.Run(bb.name, func(b *testing.B) {
			_, values := ipctest.Pair(b)
			b.ReportAllocs()
			for b.Loop() {
				if err := ipc.ExportArray(bb.a, values); err != nil {
					b.Fatal(err)
				}
				ipc.ReleaseArray(values)
			}
		})
	}
}

// BenchmarkCRoundTrip is the whole handoff, out through the C struct and back
// in again. It is the number to compare against the cost of a copy, since a
// copy is what this exists to avoid.
func BenchmarkCRoundTrip(b *testing.B) {
	for _, bb := range []struct {
		name string
		a    *array.Array
	}{
		{"int64", benchInts()},
		{"string", benchStrings()},
	} {
		b.Run(bb.name, func(b *testing.B) {
			schema, values := ipctest.Pair(b)
			field := dtype.Field{Name: "col", Type: bb.a.DType(), Nullable: true}

			b.ReportAllocs()
			for b.Loop() {
				if err := ipc.ExportField(field, schema); err != nil {
					b.Fatal(err)
				}
				if err := ipc.ExportArray(bb.a, values); err != nil {
					b.Fatal(err)
				}
				imported, err := ipc.ImportArray(schema, values)
				if err != nil {
					b.Fatal(err)
				}
				imported.Release()
				importedSink = imported
			}
		})
	}
}

// BenchmarkCExportField and BenchmarkCImportField are per column of a schema
// rather than per batch, and a wide table is a few hundred of them. The nested
// case is the one that allocates a struct per child.
func BenchmarkCExportField(b *testing.B) {
	for _, bb := range benchFields {
		b.Run(bb.name, func(b *testing.B) {
			schema, _ := ipctest.Pair(b)
			b.ReportAllocs()
			for b.Loop() {
				if err := ipc.ExportField(bb.field, schema); err != nil {
					b.Fatal(err)
				}
				ipc.ReleaseSchema(schema)
			}
		})
	}
}

func BenchmarkCImportField(b *testing.B) {
	for _, bb := range benchFields {
		b.Run(bb.name, func(b *testing.B) {
			schema, _ := ipctest.Pair(b)
			b.ReportAllocs()
			for b.Loop() {
				if err := ipc.ExportField(bb.field, schema); err != nil {
					b.Fatal(err)
				}
				f, err := ipc.ImportField(schema)
				if err != nil {
					b.Fatal(err)
				}
				fieldSink = f
			}
		})
	}
}

var benchFields = []struct {
	name  string
	field dtype.Field
}{
	{"int64", dtype.Field{Name: "id", Type: dtype.Int64}},
	{"timestamp", dtype.Field{
		Name: "when",
		Type: dtype.Timestamp{Unit: dtype.Microsecond, Zone: "Europe/London"},
	}},
	{"metadata", dtype.Field{Name: "id", Type: dtype.Int64, Metadata: benchMeta}},
	{"struct", dtype.Field{Name: "row", Type: dtype.Struct{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64},
		{Name: "name", Type: dtype.String, Nullable: true},
		{Name: "price", Type: dtype.Decimal128{Precision: 18, Scale: 2}},
	}}}},
}
