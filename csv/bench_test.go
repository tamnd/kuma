package csv_test

import (
	"bytes"
	stdcsv "encoding/csv"
	"errors"
	"io"
	"math/rand/v2"
	"strconv"
	"testing"

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

// BenchmarkReadWide is a hundred columns rather than four, where the work per
// row that is not the values themselves shows up.
func BenchmarkReadWide(b *testing.B) {
	kinds := make([]string, 100)
	for i := range kinds {
		kinds[i] = "int"
	}
	benchRead(b, benchFile(kinds...), nil)
}
