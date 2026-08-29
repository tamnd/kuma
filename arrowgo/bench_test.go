package arrowgo

import (
	"strconv"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	arrowarray "github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"
)

// benchRows is the size the benchmarks below are written at, which is one
// chunk of a size a reader would hand over rather than a whole file.
const benchRows = 1 << 16

// BenchmarkImportArrayFloat64 is the shared path at its plainest: one buffer
// and one bitmap change hands and nothing is read.
func BenchmarkImportArrayFloat64(b *testing.B) {
	in := benchFloats(b)
	defer in.Release()

	b.ReportAllocs()
	for b.Loop() {
		if _, err := ImportArray(in); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExportArrayFloat64(b *testing.B) {
	in := benchFloats(b)
	defer in.Release()

	out, err := ImportArray(in)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for b.Loop() {
		a, aerr := ExportArray(out)
		if aerr != nil {
			b.Fatal(aerr)
		}
		a.Release()
	}
}

// BenchmarkImportArrayStringView is the string column in the layout both sides
// hold, which walks the views once to check them and shares the blocks.
func BenchmarkImportArrayStringView(b *testing.B) {
	in := benchStrings(b, arrow.BinaryTypes.StringView)
	defer in.Release()

	b.ReportAllocs()
	for b.Loop() {
		if _, err := ImportArray(in); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkImportArrayString is the same column in the offset layout, which is
// the one path that copies. The gap between this and the view benchmark above
// is what the layout is worth at the boundary.
func BenchmarkImportArrayString(b *testing.B) {
	in := benchStrings(b, arrow.BinaryTypes.String)
	defer in.Release()

	b.ReportAllocs()
	for b.Loop() {
		if _, err := ImportArray(in); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkImportRecordBatch(b *testing.B) {
	in := benchBatch(b)
	defer in.Release()

	b.ReportAllocs()
	for b.Loop() {
		if _, err := ImportRecordBatch(in); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExportRecordBatch(b *testing.B) {
	in := benchBatch(b)
	defer in.Release()

	out, err := ImportRecordBatch(in)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for b.Loop() {
		rec, rerr := ExportRecordBatch(out)
		if rerr != nil {
			b.Fatal(rerr)
		}
		rec.Release()
	}
}

func benchFloats(b *testing.B) arrow.Array {
	b.Helper()

	bld := arrowarray.NewFloat64Builder(memory.DefaultAllocator)
	defer bld.Release()

	bld.Reserve(benchRows)
	for i := range benchRows {
		if i%16 == 0 {
			bld.AppendNull()
			continue
		}
		bld.Append(float64(i) * 1.5)
	}
	return bld.NewArray()
}

// benchStrings builds the same values in whichever string layout is asked for,
// with one value in eight too long to sit inside a view so that the blocks are
// used rather than only the headers.
func benchStrings(b *testing.B, dt arrow.DataType) arrow.Array {
	b.Helper()

	bld := arrowarray.NewBuilder(memory.DefaultAllocator, dt)
	defer bld.Release()

	sv, ok := bld.(interface{ Append(string) })
	if !ok {
		b.Fatalf("a %s builder has no way to append a value", dt)
	}
	bld.Reserve(benchRows)
	for i := range benchRows {
		if i%16 == 0 {
			bld.AppendNull()
			continue
		}
		if i%8 == 0 {
			sv.Append("a symbol far too long to sit inside a view " + strconv.Itoa(i))
			continue
		}
		sv.Append("SYM" + strconv.Itoa(i%1000))
	}
	return bld.NewArray()
}

func benchBatch(b *testing.B) arrow.RecordBatch {
	b.Helper()

	symbol := benchStrings(b, arrow.BinaryTypes.StringView)
	defer symbol.Release()
	price := benchFloats(b)
	defer price.Release()

	schema := arrow.NewSchema([]arrow.Field{
		{Name: "symbol", Type: arrow.BinaryTypes.StringView, Nullable: true},
		{Name: "price", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
	}, nil)
	return arrowarray.NewRecordBatch(schema, []arrow.Array{symbol, price}, benchRows)
}
