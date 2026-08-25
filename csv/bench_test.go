package csv_test

import (
	"bytes"
	stdcsv "encoding/csv"
	"errors"
	"io"
	"math/rand/v2"
	"strconv"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/csv"
	"github.com/tamnd/kuma/dtype"
)

// benchRows is how many rows the benchmarks read. It is enough that the per
// row cost is what is being measured and small enough that generating the
// input is not most of the run.
const benchRows = 1 << 16

var (
	tableSink *csv.Table
	fieldSink int
)

// benchFile writes a file of benchRows rows with the given columns, where a
// column is named by what it holds.
//
// The values are the same on every run, since a benchmark that reads different
// numbers each time is measuring the number generator as well.
func benchFile(kinds ...string) []byte {
	r := rand.New(rand.NewPCG(1, 2))

	var buf bytes.Buffer
	for i, k := range kinds {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.WriteString(k)
		buf.WriteString(strconv.Itoa(i))
	}
	buf.WriteByte('\n')

	syms := []string{"AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA"}
	for range benchRows {
		for i, k := range kinds {
			if i > 0 {
				buf.WriteByte(',')
			}
			switch k {
			case "int":
				buf.WriteString(strconv.FormatInt(int64(r.Uint32()), 10))
			case "float":
				buf.WriteString(strconv.FormatFloat(r.Float64()*1000, 'f', 4, 64))
			case "string":
				buf.WriteString(syms[int(r.Uint32())%len(syms)])
			case "bool":
				buf.WriteString(strconv.FormatBool(r.Uint32()%2 == 0))
			case "gappy":
				// One value in eight is missing, which is what a file written
				// out of a real system looks like.
				if r.Uint32()%8 != 0 {
					buf.WriteString(strconv.FormatInt(int64(r.Uint32()), 10))
				}
			}
		}
		buf.WriteByte('\n')
	}
	return buf.Bytes()
}

// benchRead reads the same bytes over and over, so what is measured is the
// parse rather than the disk.
func benchRead(b *testing.B, in []byte, opts *csv.Options) {
	b.Helper()
	b.SetBytes(int64(len(in)))
	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		t, err := csv.Read(bytes.NewReader(in), opts)
		if err != nil {
			b.Fatalf("Read: %v", err)
		}
		tableSink = t
	}
}

func BenchmarkRead(b *testing.B) {
	benchRead(b, benchFile("string", "int", "float", "bool"), nil)
}

func BenchmarkReadInts(b *testing.B) {
	benchRead(b, benchFile("int", "int", "int", "int"), nil)
}

func BenchmarkReadFloats(b *testing.B) {
	benchRead(b, benchFile("float", "float", "float", "float"), nil)
}

func BenchmarkReadStrings(b *testing.B) {
	benchRead(b, benchFile("string", "string", "string", "string"), nil)
}

func BenchmarkReadGappy(b *testing.B) {
	benchRead(b, benchFile("gappy", "gappy", "gappy", "gappy"), nil)
}

// BenchmarkReadTyped is the same file read with the types given rather than
// worked out, which is what a program that already knows its own file should
// do and is the difference inference costs.
func BenchmarkReadTyped(b *testing.B) {
	benchRead(b, benchFile("string", "int", "float", "bool"), &csv.Options{
		Types: map[string]dtype.DataType{
			"string0": dtype.String,
			"int1":    dtype.Int64,
			"float2":  dtype.Float64,
			"bool3":   dtype.Bool,
		},
	})
}

// BenchmarkReadQuoted is the same values with every field quoted, which is
// what a writer that does not look at the values produces and is the case the
// vectorized parser will have to work hardest on.
func BenchmarkReadQuoted(b *testing.B) {
	plain := benchFile("string", "int", "float", "bool")

	var buf bytes.Buffer
	for _, line := range bytes.Split(plain, []byte("\n")) {
		if len(line) == 0 {
			continue
		}
		for i, field := range bytes.Split(line, []byte(",")) {
			if i > 0 {
				buf.WriteByte(',')
			}
			buf.WriteByte('"')
			buf.Write(field)
			buf.WriteByte('"')
		}
		buf.WriteByte('\n')
	}
	benchRead(b, buf.Bytes(), nil)
}

// BenchmarkRecords is the floor: what encoding/csv costs on the same file with
// nothing built out of it.
//
// The difference between this and BenchmarkRead is everything this package
// does, which is the number to watch. Beating pandas means beating this too,
// so when the vectorized parser from document 05 arrives it replaces the line
// below rather than sitting on top of it.
func BenchmarkRecords(b *testing.B) {
	in := benchFile("string", "int", "float", "bool")
	b.SetBytes(int64(len(in)))
	b.ReportAllocs()

	for b.Loop() {
		r := stdcsv.NewReader(bytes.NewReader(in))
		r.ReuseRecord = true
		for {
			rec, err := r.Read()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				b.Fatalf("Read: %v", err)
			}
			fieldSink = len(rec)
		}
	}
}

// benchTable reads a generated file into the table the write benchmarks write.
func benchTable(b *testing.B, kinds ...string) *csv.Table {
	b.Helper()
	t, err := csv.Read(bytes.NewReader(benchFile(kinds...)), nil)
	if err != nil {
		b.Fatalf("Read: %v", err)
	}
	return t
}

// benchWrite writes the same table over and over into a writer that throws it
// away, so what is measured is the formatting rather than a disk.
func benchWrite(b *testing.B, t *csv.Table, opts *csv.WriteOptions) {
	b.Helper()

	var n countingWriter
	if err := csv.Write(&n, t, opts); err != nil {
		b.Fatalf("Write: %v", err)
	}
	b.SetBytes(int64(n))
	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		if err := csv.Write(io.Discard, t, opts); err != nil {
			b.Fatalf("Write: %v", err)
		}
	}
}

// countingWriter measures how big the output is, which is what the throughput
// figures are per second of.
type countingWriter int

func (w *countingWriter) Write(p []byte) (int, error) {
	*w += countingWriter(len(p))
	return len(p), nil
}

func BenchmarkWrite(b *testing.B) {
	benchWrite(b, benchTable(b, "string", "int", "float", "bool"), nil)
}

func BenchmarkWriteInts(b *testing.B) {
	benchWrite(b, benchTable(b, "int", "int", "int", "int"), nil)
}

func BenchmarkWriteFloats(b *testing.B) {
	benchWrite(b, benchTable(b, "float", "float", "float", "float"), nil)
}

func BenchmarkWriteStrings(b *testing.B) {
	benchWrite(b, benchTable(b, "string", "string", "string", "string"), nil)
}

func BenchmarkWriteGappy(b *testing.B) {
	benchWrite(b, benchTable(b, "gappy", "gappy", "gappy", "gappy"), nil)
}

func BenchmarkWriteQuoteAll(b *testing.B) {
	benchWrite(b, benchTable(b, "string", "int", "float", "bool"),
		&csv.WriteOptions{QuoteAll: true})
}

// BenchmarkWriteRecords is what the same table costs through
// [encoding/csv.Writer], which takes a []string and so allocates a string for
// every value on the way past. It is the reason this package has a writer of
// its own.
func BenchmarkWriteRecords(b *testing.B) {
	t := benchTable(b, "string", "int", "float", "bool")
	rows := t.NumRows()

	b.ReportAllocs()
	for b.Loop() {
		w := stdcsv.NewWriter(io.Discard)
		rec := make([]string, t.NumCols())
		for i := range rows {
			for j, c := range t.Columns {
				rec[j] = fieldText(c, i)
			}
			if err := w.Write(rec); err != nil {
				b.Fatalf("Write: %v", err)
			}
		}
		w.Flush()
	}
}

// fieldText renders one value the way a program with no library would have to,
// which is one string per value.
func fieldText(c *array.Chunked, i int) string {
	if c.IsNull(i) {
		return ""
	}
	switch c.DType().Kind() {
	case dtype.Int64Kind:
		return strconv.FormatInt(c.Value[int64](i), 10)
	case dtype.Float64Kind:
		return strconv.FormatFloat(c.Value[float64](i), 'g', -1, 64)
	case dtype.BoolKind:
		return strconv.FormatBool(c.Bool(i))
	default:
		return string(c.Bytes(i))
	}
}

// BenchmarkReadWide is a hundred columns rather than four, where the work per
// row that is not the values themselves shows up.
func BenchmarkReadWide(b *testing.B) {
	benchRead(b, benchFile(wideFile()...), nil)
}

// BenchmarkReadWideColumns is the same file with four of its hundred columns
// asked for, which is the shape of nearly every query anyone runs over a file
// this wide. What is left is the cost of finding the fields, which is the part
// no reader can skip.
func BenchmarkReadWideColumns(b *testing.B) {
	benchRead(b, benchFile(wideFile()...), &csv.Options{
		Columns: []string{"int0", "int1", "int2", "int3"},
	})
}

// wideFile is the column list the two benchmarks above share.
func wideFile() []string {
	kinds := make([]string, 100)
	for i := range kinds {
		kinds[i] = "int"
	}
	return kinds
}
