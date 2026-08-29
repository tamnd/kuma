package ndjson_test

import (
	"bytes"
	"encoding/json/jsontext"
	json "encoding/json/v2"
	"errors"
	"io"
	"math/rand/v2"
	"strconv"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ndjson"
)

// benchRows is how many rows the benchmarks read. It is enough that the per
// row cost is what is being measured and small enough that generating the
// input is not most of the run.
const benchRows = 1 << 16

var (
	tableSink *ndjson.Table
	countSink int
	mapSink   map[string]any
)

// benchFile writes a file of benchRows lines with the given members, where a
// member is named by what it holds.
//
// The values are the same on every run, since a benchmark that reads different
// numbers each time is measuring the number generator as well.
func benchFile(kinds ...string) []byte {
	r := rand.New(rand.NewPCG(1, 2))
	syms := []string{"AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA"}

	var buf bytes.Buffer
	for range benchRows {
		buf.WriteByte('{')
		first := true
		for i, k := range kinds {
			// A gappy member is not written at all on the lines it is missing
			// from, which is what a file written a record at a time looks like
			// and is the shape this format has that a delimited one does not.
			if k == "gappy" && r.Uint32()%8 == 0 {
				continue
			}
			if !first {
				buf.WriteByte(',')
			}
			first = false

			buf.WriteByte('"')
			buf.WriteString(k)
			buf.WriteString(strconv.Itoa(i))
			buf.WriteString("\":")

			switch k {
			case "int", "gappy":
				buf.WriteString(strconv.FormatInt(int64(r.Uint32()), 10))
			case "float":
				buf.WriteString(strconv.FormatFloat(r.Float64()*1000, 'f', 4, 64))
			case "string":
				buf.WriteByte('"')
				buf.WriteString(syms[int(r.Uint32())%len(syms)])
				buf.WriteByte('"')
			case "bool":
				buf.WriteString(strconv.FormatBool(r.Uint32()%2 == 0))
			case "nested":
				buf.WriteString(`{"x":`)
				buf.WriteString(strconv.FormatInt(int64(r.Uint32()%1000), 10))
				buf.WriteString(`,"y":[1,2,3]}`)
			}
		}
		buf.WriteString("}\n")
	}
	return buf.Bytes()
}

// benchRead reads the same bytes over and over, so what is measured is the
// parse rather than the disk.
func benchRead(b *testing.B, in []byte, opts *ndjson.Options) {
	b.Helper()
	b.SetBytes(int64(len(in)))
	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		t, err := ndjson.Read(bytes.NewReader(in), opts)
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

// BenchmarkReadGappy is the file where one value in eight is not on the line at
// all, which is the case the reader has to fill in after the fact.
func BenchmarkReadGappy(b *testing.B) {
	benchRead(b, benchFile("gappy", "gappy", "gappy", "gappy"), nil)
}

// BenchmarkReadNested is the members holding an object and an array, which go
// into a string column as the text they arrived as.
func BenchmarkReadNested(b *testing.B) {
	benchRead(b, benchFile("nested", "nested", "int", "int"), nil)
}

// BenchmarkReadTyped is the same file read with the types given rather than
// worked out, which is what a program that already knows its own file should
// do and is the difference inference costs.
func BenchmarkReadTyped(b *testing.B) {
	benchRead(b, benchFile("string", "int", "float", "bool"), &ndjson.Options{
		Types: map[string]dtype.DataType{
			"string0": dtype.String,
			"int1":    dtype.Int64,
			"float2":  dtype.Float64,
			"bool3":   dtype.Bool,
		},
	})
}

// BenchmarkTokens is the floor: what walking the same file costs with nothing
// built out of it.
//
// The difference between this and BenchmarkRead is everything this package
// does. It is the number to watch, since no reader on top of this decoder can
// go faster than the decoder does.
func BenchmarkTokens(b *testing.B) {
	in := benchFile("string", "int", "float", "bool")
	b.SetBytes(int64(len(in)))
	b.ReportAllocs()

	for b.Loop() {
		d := jsontext.NewDecoder(bytes.NewReader(in))
		n := 0
		for {
			_, err := d.ReadToken()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				b.Fatalf("ReadToken: %v", err)
			}
			n++
		}
		countSink = n
	}
}

// BenchmarkMaps is what the same file costs read the way a program with no
// library reads it, which is one map and one boxed value per line.
//
// It is not a fair fight and it is not meant to be. It is what the alternative
// costs, and the gap is the reason to hold a file in columns.
func BenchmarkMaps(b *testing.B) {
	in := benchFile("string", "int", "float", "bool")
	b.SetBytes(int64(len(in)))
	b.ReportAllocs()

	for b.Loop() {
		d := jsontext.NewDecoder(bytes.NewReader(in))
		for {
			var m map[string]any
			err := json.UnmarshalDecode(d, &m)
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				b.Fatalf("UnmarshalDecode: %v", err)
			}
			mapSink = m
		}
	}
}

// benchTable reads a generated file into the table the write benchmarks write.
func benchTable(b *testing.B, kinds ...string) *ndjson.Table {
	b.Helper()
	t, err := ndjson.Read(bytes.NewReader(benchFile(kinds...)), nil)
	if err != nil {
		b.Fatalf("Read: %v", err)
	}
	return t
}

// benchWrite writes the same table over and over into a writer that throws it
// away, so what is measured is the formatting rather than a disk.
func benchWrite(b *testing.B, t *ndjson.Table, opts *ndjson.WriteOptions) {
	b.Helper()

	var n countingWriter
	if err := ndjson.Write(&n, t, opts); err != nil {
		b.Fatalf("Write: %v", err)
	}
	b.SetBytes(int64(n))
	b.ReportAllocs()
	b.ResetTimer()

	for b.Loop() {
		if err := ndjson.Write(io.Discard, t, opts); err != nil {
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

// BenchmarkWriteOmitNull is the gappy table written without the missing values
// in it, which is a smaller file and one branch more per value.
func BenchmarkWriteOmitNull(b *testing.B) {
	benchWrite(b, benchTable(b, "gappy", "gappy", "gappy", "gappy"),
		&ndjson.WriteOptions{OmitNull: true})
}

// BenchmarkWriteMaps is what the same table costs written the way a program
// with no library writes it, which is a map per row and a boxed value per
// column. It is the reason this package has a writer of its own.
func BenchmarkWriteMaps(b *testing.B) {
	t := benchTable(b, "string", "int", "float", "bool")
	names := t.Schema.Names()

	b.ReportAllocs()
	for b.Loop() {
		e := jsontext.NewEncoder(io.Discard)
		for i := range t.NumRows() {
			m := make(map[string]any, len(names))
			for j, c := range t.Columns {
				m[names[j]] = boxed(c, i)
			}
			if err := json.MarshalEncode(e, m); err != nil {
				b.Fatalf("MarshalEncode: %v", err)
			}
		}
	}
}

// boxed renders one value the way a program with no library would have to,
// which is one interface per value.
func boxed(c *array.Chunked, i int) any {
	if c.IsNull(i) {
		return nil
	}
	switch c.DType().Kind() {
	case dtype.Int64Kind:
		return c.Value[int64](i)
	case dtype.Float64Kind:
		return c.Value[float64](i)
	case dtype.BoolKind:
		return c.Bool(i)
	default:
		return string(c.Bytes(i))
	}
}

// BenchmarkReadWide is a hundred members rather than four, where the work per
// line that is not the values themselves shows up.
func BenchmarkReadWide(b *testing.B) {
	benchRead(b, benchFile(wideFile()...), nil)
}

// BenchmarkReadWideColumns is the same file with four of its hundred members
// asked for, which is the shape of nearly every query anyone runs over a file
// this wide. What is left is the cost of finding the members, which is the part
// no reader can skip.
func BenchmarkReadWideColumns(b *testing.B) {
	benchRead(b, benchFile(wideFile()...), &ndjson.Options{
		Columns: []string{"int0", "int1", "int2", "int3"},
	})
}

// wideFile is the member list the two benchmarks above share.
func wideFile() []string {
	kinds := make([]string, 100)
	for i := range kinds {
		kinds[i] = "int"
	}
	return kinds
}
