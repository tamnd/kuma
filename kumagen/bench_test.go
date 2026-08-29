package main

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/parquet"
)

// sink is where the generated source goes, so that nothing is optimized away.
var sink []byte

// BenchmarkLoadAndGenerate is the whole tool over one small package, which is
// what a go generate line costs every time somebody runs it. The property this
// is here to keep is the one in document 03: fast enough that running it on
// every build is not worth thinking about.
func BenchmarkLoadAndGenerate(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		p, err := loadPackage("testdata/trades")
		if err != nil {
			b.Fatalf("loadPackage: %v", err)
		}
		src, err := generate(p, "Trade")
		if err != nil {
			b.Fatalf("generate: %v", err)
		}
		sink = src
	}
}

// BenchmarkGenerate is the same without the parsing, which is the part that
// grows with the package rather than with the struct.
func BenchmarkGenerate(b *testing.B) {
	p, err := loadPackage("testdata/trades")
	if err != nil {
		b.Fatalf("loadPackage: %v", err)
	}

	b.ReportAllocs()
	for b.Loop() {
		src, err := generate(p, "Trade")
		if err != nil {
			b.Fatalf("generate: %v", err)
		}
		sink = src
	}
}

// BenchmarkReadSchemaParquet is what -from costs on a format that writes its
// schema down. Only the footer is read, so what this measures is the end of a
// file of sixty four columns rather than the eight thousand rows in it.
func BenchmarkReadSchemaParquet(b *testing.B) {
	path := filepath.Join(b.TempDir(), "wide.parquet")
	if err := parquet.WriteFile(path, wideTable(b, 64, 8192), nil); err != nil {
		b.Fatalf("parquet.WriteFile: %v", err)
	}

	b.ReportAllocs()
	for b.Loop() {
		s, err := readSchema(path, 1000)
		if err != nil {
			b.Fatal(err)
		}
		if s.Len() != 64 {
			b.Fatalf("read %d columns", s.Len())
		}
	}
}

// BenchmarkReadSchemaCSV is the same on a format that does not, where the
// answer is a guess made from the sample and the sample has to be parsed.
func BenchmarkReadSchemaCSV(b *testing.B) {
	var sb strings.Builder
	for i := range 64 {
		if i > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString("col_")
		sb.WriteString(strconv.Itoa(i))
	}
	sb.WriteByte('\n')
	row := strings.Repeat("1,", 63) + "1\n"
	for range 8192 {
		sb.WriteString(row)
	}

	path := filepath.Join(b.TempDir(), "wide.csv")
	if err := os.WriteFile(path, []byte(sb.String()), 0o600); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for b.Loop() {
		s, err := readSchema(path, 1000)
		if err != nil {
			b.Fatal(err)
		}
		if s.Len() != 64 {
			b.Fatalf("read %d columns", s.Len())
		}
	}
}

// BenchmarkRenderStruct is the writing on its own, which is the part that grows
// with the number of columns and not with the file.
func BenchmarkRenderStruct(b *testing.B) {
	fields, err := structFields("Wide", wideSchema(64))
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for b.Loop() {
		src, err := renderStruct("wide", "Wide", "wide.parquet", fields)
		if err != nil {
			b.Fatal(err)
		}
		sink = src
	}
}

// wideSchema is a schema of n int64 columns, which is the shape that makes the
// number of columns the thing being measured.
func wideSchema(n int) dtype.Schema {
	fields := make([]dtype.Field, n)
	for i := range fields {
		fields[i] = dtype.Field{Name: "col_" + strconv.Itoa(i), Type: dtype.Int64}
	}
	return dtype.Schema{Fields: fields}
}

// wideTable fills that schema in.
func wideTable(tb testing.TB, cols, rows int) *array.Table {
	tb.Helper()

	vals := make([]int64, rows)
	for i := range vals {
		vals[i] = int64(i)
	}

	schema := wideSchema(cols)
	columns := make([]*array.Chunked, cols)
	for i := range columns {
		c, err := array.NewChunked(dtype.Int64, array.Of(vals...))
		if err != nil {
			tb.Fatalf("NewChunked: %v", err)
		}
		columns[i] = c
	}
	return &array.Table{Schema: schema, Columns: columns}
}
