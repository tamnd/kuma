package ipc_test

import (
	"bytes"
	"errors"
	"strconv"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

// A dictionary encoded column is the one column that does not fit inside a
// batch. The indices are in it and the values are in a message somewhere in
// front of it, so what is checked here is the part that a round trip of a single
// batch never reaches: values arriving once and being read by every batch after
// them, a stream that sends new values half way through, and a file that is not
// allowed to.

// codeType is the column type all of these use. A country code is the case the
// encoding exists for, a few hundred distinct strings across however many rows.
var codeType = dtype.Dictionary{Index: dtype.Int32, Value: dtype.String}

// codeSchema is one dictionary encoded column on its own.
func codeSchema() dtype.Schema {
	return dtype.Schema{Fields: []dtype.Field{{Name: "code", Type: codeType, Nullable: true}}}
}

// mixedSchema puts an ordinary column on either side of it, which is where a
// writer that lost count of which column a dictionary belongs to would show up.
func mixedSchema() dtype.Schema {
	return dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64},
		{Name: "code", Type: codeType, Nullable: true},
		{Name: "price", Type: dtype.Float64},
	}}
}

// codes builds a dictionary encoded column out of the positions given and the
// values behind them. A position of minus one is a value that is missing.
func codes(t *testing.T, values *array.Array, at ...int32) *array.Array {
	t.Helper()
	b, err := array.NewBuilder(dtype.Int32)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	for _, i := range at {
		if i < 0 {
			b.AppendNull()
			continue
		}
		b.Append(i)
	}
	col, err := array.NewDictionary(b.Finish(), values)
	if err != nil {
		t.Fatalf("NewDictionary: %v", err)
	}
	return col
}

// wantCodes checks a dictionary encoded column against the values it should
// read as, with null for the rows that have none. It follows the indices rather
// than comparing the dictionaries, since two columns holding the same values
// under different indices are the same column.
func wantCodes(t *testing.T, col *array.Array, want ...string) {
	t.Helper()
	if !dtype.Equal(col.DType(), codeType) {
		t.Fatalf("the column came back as %s, want %s", col.DType(), codeType)
	}
	if col.Len() != len(want) {
		t.Fatalf("the column has %d values, want %d", col.Len(), len(want))
	}
	values := col.Dictionary()
	for i, w := range want {
		got := "null"
		if at := col.Index(i); at >= 0 && !values.IsNull(at) {
			got = string(values.Bytes(at))
		}
		if got != w {
			t.Errorf("value %d is %q, want %q", i, got, w)
		}
	}
}

// TestStreamDictionary writes two batches that share one dictionary and reads
// them back. The values go out once, in front of the first batch, and both
// batches read from that one array rather than from a copy each, which is the
// whole point of the encoding and is checked here by identity.
func TestStreamDictionary(t *testing.T) {
	values := array.OfStrings("GB", "JP", "US")
	s := codeSchema()
	stream := streamOf(t, s,
		ipc.Batch{Length: 4, Columns: []*array.Array{codes(t, values, 2, 0, -1, 1)}},
		ipc.Batch{Length: 2, Columns: []*array.Array{codes(t, values, 1, 1)}},
	)

	r, err := ipc.NewReader(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	if !r.Schema().Equal(s) {
		t.Fatalf("the stream carries the schema %v, want %v", r.Schema(), s)
	}

	var got []ipc.Batch
	for b := range r.All() {
		got = append(got, b)
	}
	if err := r.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("the stream came back with %d batches, want 2", len(got))
	}

	wantCodes(t, got[0].Columns[0], "US", "GB", "null", "JP")
	wantCodes(t, got[1].Columns[0], "JP", "JP")
	if got[0].Columns[0].Dictionary() != got[1].Columns[0].Dictionary() {
		t.Error("the two batches read from two dictionaries, want the one they were written with")
	}
}

// TestStreamDictionaryReplaced sends new values for a dictionary half way
// through a stream, which a stream is allowed to do. Everything after the new
// values reads from them and the batches in front are untouched, which is why
// the first batch is checked after the second has been read.
func TestStreamDictionaryReplaced(t *testing.T) {
	first := array.OfStrings("GB", "JP")
	second := array.OfStrings("US", "FR", "DE")
	s := codeSchema()
	stream := streamOf(t, s,
		ipc.Batch{Length: 2, Columns: []*array.Array{codes(t, first, 0, 1)}},
		ipc.Batch{Length: 3, Columns: []*array.Array{codes(t, second, 2, 0, 1)}},
	)

	r, err := ipc.NewReader(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	var got []ipc.Batch
	for b := range r.All() {
		got = append(got, b)
	}
	if err := r.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("the stream came back with %d batches, want 2", len(got))
	}

	wantCodes(t, got[0].Columns[0], "GB", "JP")
	wantCodes(t, got[1].Columns[0], "DE", "US", "FR")
}

// TestStreamDictionaryMixed reads a dictionary encoded column with an ordinary
// one on either side of it. The columns of a batch are written in order and the
// dictionaries are written in front of the batch, so a writer that matched the
// two up by counting would put the values of the middle column under the
// identifier of one of its neighbors.
func TestStreamDictionaryMixed(t *testing.T) {
	values := array.OfStrings("GB", "JP", "US")
	s := mixedSchema()
	stream := streamOf(t, s, ipc.Batch{Length: 3, Columns: []*array.Array{
		array.Of[int64](7, 8, 9),
		codes(t, values, 1, 2, 0),
		array.Of[float64](1.5, 2.5, 3.5),
	}})

	r, err := ipc.NewReader(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	if !r.Next() {
		t.Fatalf("Next read nothing: %v", r.Err())
	}
	b := r.Batch()
	if b.Length != 3 {
		t.Fatalf("the batch has %d rows, want 3", b.Length)
	}
	for i, want := range []int64{7, 8, 9} {
		if got := b.Columns[0].Value[int64](i); got != want {
			t.Errorf("id %d is %d, want %d", i, got, want)
		}
	}
	wantCodes(t, b.Columns[1], "JP", "US", "GB")
	for i, want := range []float64{1.5, 2.5, 3.5} {
		if got := b.Columns[2].Value[float64](i); got != want {
			t.Errorf("price %d is %v, want %v", i, got, want)
		}
	}
}

// TestStreamDictionaryTwoColumns writes two dictionary encoded columns holding
// different values. Each one gets an identifier of its own and each set of
// values arrives under the right one, which is the thing a single dictionary
// column cannot say anything about.
func TestStreamDictionaryTwoColumns(t *testing.T) {
	countries := array.OfStrings("GB", "JP", "US")
	sides := array.OfStrings("buy", "sell")
	s := dtype.Schema{Fields: []dtype.Field{
		{Name: "code", Type: codeType, Nullable: true},
		{Name: "side", Type: codeType, Nullable: true},
	}}
	stream := streamOf(t, s, ipc.Batch{Length: 3, Columns: []*array.Array{
		codes(t, countries, 0, 2, 1),
		codes(t, sides, 1, 1, 0),
	}})

	r, err := ipc.NewReader(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	if !r.Next() {
		t.Fatalf("Next read nothing: %v", r.Err())
	}
	b := r.Batch()
	wantCodes(t, b.Columns[0], "GB", "US", "JP")
	wantCodes(t, b.Columns[1], "sell", "sell", "buy")
}

// TestStreamDictionaryNulls reads a dictionary that holds a null of its own.
// The distinct values of a column include the missing one, so a null is allowed
// to be a value of the dictionary as well as a row of the column, and the two
// mean the same thing when they are read.
func TestStreamDictionaryNulls(t *testing.T) {
	values := buildStrings(t, []string{"GB", "", "US"}, 1)
	s := codeSchema()
	stream := streamOf(t, s, ipc.Batch{Length: 4, Columns: []*array.Array{
		codes(t, values, 0, 1, 2, -1),
	}})

	r, err := ipc.NewReader(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	if !r.Next() {
		t.Fatalf("Next read nothing: %v", r.Err())
	}
	col := r.Batch().Columns[0]
	wantCodes(t, col, "GB", "null", "US", "null")
	if got := col.Dictionary().NullCount(); got != 1 {
		t.Errorf("the dictionary came back with %d nulls, want 1", got)
	}
	if col.IsNull(1) {
		t.Error("row 1 is missing, want a row holding the missing value of the dictionary")
	}
}

// TestStreamDictionaryEmpty writes a batch of no rows against an empty
// dictionary, which is what the first batch of an empty result looks like. The
// values message still goes out, since the reader has to have something to bind
// the column to.
func TestStreamDictionaryEmpty(t *testing.T) {
	s := codeSchema()
	stream := streamOf(t, s, ipc.Batch{Columns: []*array.Array{
		codes(t, buildStrings(t, nil)),
	}})

	r, err := ipc.NewReader(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	if !r.Next() {
		t.Fatalf("Next read nothing: %v", r.Err())
	}
	col := r.Batch().Columns[0]
	wantCodes(t, col)
	if col.Dictionary().Len() != 0 {
		t.Errorf("the dictionary came back with %d values, want none", col.Dictionary().Len())
	}
}

// TestFileDictionary reads the batches of a file backwards. A file is read in
// whatever order the reader likes, so the dictionaries have to be in hand
// before any batch is, which is why they are read when the file is opened
// rather than when a batch that needs one is asked for.
func TestFileDictionary(t *testing.T) {
	values := array.OfStrings("GB", "JP", "US")
	s := codeSchema()
	file := fileOf(t, s,
		ipc.Batch{Length: 2, Columns: []*array.Array{codes(t, values, 0, 1)}},
		ipc.Batch{Length: 3, Columns: []*array.Array{codes(t, values, 2, 2, -1)}},
	)

	r := openFile(t, file)
	if r.NumBatches() != 2 {
		t.Fatalf("the file holds %d batches, want 2", r.NumBatches())
	}
	last, err := r.Batch(1)
	if err != nil {
		t.Fatalf("Batch(1): %v", err)
	}
	wantCodes(t, last.Columns[0], "US", "US", "null")

	first, err := r.Batch(0)
	if err != nil {
		t.Fatalf("Batch(0): %v", err)
	}
	wantCodes(t, first.Columns[0], "GB", "JP")
	if first.Columns[0].Dictionary() != last.Columns[0].Dictionary() {
		t.Error("the two batches read from two dictionaries, want the one the file holds")
	}
}

// TestFileDictionaryTwice writes two batches holding different values for one
// column. A stream would send the second lot and carry on, and a file cannot,
// since a batch read out of the middle of it would then mean something
// different depending on how much of the file had been read first.
func TestFileDictionaryTwice(t *testing.T) {
	first := array.OfStrings("GB", "JP")
	second := array.OfStrings("US", "FR")

	var buf bytes.Buffer
	w, err := ipc.NewFileWriter(&buf, codeSchema())
	if err != nil {
		t.Fatalf("NewFileWriter: %v", err)
	}
	if err = w.Write(ipc.Batch{Length: 2, Columns: []*array.Array{codes(t, first, 0, 1)}}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	err = w.Write(ipc.Batch{Length: 2, Columns: []*array.Array{codes(t, second, 0, 1)}})
	if !errors.Is(err, ipc.ErrUnsupported) {
		t.Fatalf("Write of a second dictionary = %v, want %v", err, ipc.ErrUnsupported)
	}

	// The refusal leaves the file where it was, so the batch that was written
	// is still readable and closing still produces a file.
	if err = w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	r := openFile(t, buf.Bytes())
	if r.NumBatches() != 1 {
		t.Fatalf("the file holds %d batches, want 1", r.NumBatches())
	}
	b, err := r.Batch(0)
	if err != nil {
		t.Fatalf("Batch(0): %v", err)
	}
	wantCodes(t, b.Columns[0], "GB", "JP")
}

// TestWriteDictionaryIndices hands a writer the indices of a dictionary encoded
// column rather than the column. The indices are a perfectly good integer array
// and writing them would produce a file whose values are nowhere in it, so both
// writers refuse rather than write half a column.
func TestWriteDictionaryIndices(t *testing.T) {
	col := codes(t, array.OfStrings("GB", "JP"), 0, 1)
	b := ipc.Batch{Length: 2, Columns: []*array.Array{col.Indices()}}

	var stream bytes.Buffer
	sw, err := ipc.NewWriter(&stream, codeSchema())
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err = sw.Write(b); !errors.Is(err, ipc.ErrBuffers) {
		t.Errorf("Write to a stream = %v, want %v", err, ipc.ErrBuffers)
	}

	var file bytes.Buffer
	fw, err := ipc.NewFileWriter(&file, codeSchema())
	if err != nil {
		t.Fatalf("NewFileWriter: %v", err)
	}
	if err := fw.Write(b); !errors.Is(err, ipc.ErrBuffers) {
		t.Errorf("Write to a file = %v, want %v", err, ipc.ErrBuffers)
	}
}

// TestDictionaryBatchIsIndices reads a record batch on its own against a schema
// that says the column is dictionary encoded. The message holds the indices and
// nothing else, so that is what comes back: the values are in another message
// and DecodeBatch is handed one message.
func TestDictionaryBatchIsIndices(t *testing.T) {
	s := codeSchema()
	msg, err := ipc.EncodeBatch(s, ipc.Batch{Length: 3, Columns: []*array.Array{
		codes(t, array.OfStrings("GB", "JP"), 0, 1, -1),
	}})
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}

	b, rest, err := ipc.DecodeBatch(s, msg)
	if err != nil {
		t.Fatalf("DecodeBatch: %v", err)
	}
	if len(rest) != 0 {
		t.Errorf("%d bytes left after the batch, want none", len(rest))
	}
	col := b.Columns[0]
	if !dtype.Equal(col.DType(), dtype.Int32) {
		t.Fatalf("the column came back as %s, want %s", col.DType(), dtype.Int32)
	}
	for i, want := range []int32{0, 1, 0} {
		if col.IsNull(i) {
			continue
		}
		if got := col.Value[int32](i); got != want {
			t.Errorf("index %d is %d, want %d", i, got, want)
		}
	}
	if !col.IsNull(2) {
		t.Error("index 2 came back as a value, want a null")
	}
}

// The shape a dictionary encoding is for: a few hundred distinct values across
// a few thousand rows, in a stream of several batches, since what the encoding
// buys is every batch after the first.
const (
	benchDictRows    = 4096
	benchDictValues  = 250
	benchDictBatches = 8
)

// benchStrings is the column written the ordinary way, with every row carrying
// its own copy of the text.
func benchDictStrings(b *testing.B) (dtype.Schema, []ipc.Batch) {
	b.Helper()
	s := dtype.Schema{Fields: []dtype.Field{{Name: "code", Type: dtype.String, Nullable: true}}}

	values := make([]string, benchDictRows)
	for i := range values {
		values[i] = "code number " + strconv.Itoa(i%benchDictValues)
	}
	batches := make([]ipc.Batch, benchDictBatches)
	for i := range batches {
		batches[i] = ipc.Batch{
			Length:  benchDictRows,
			Columns: []*array.Array{array.OfStrings(values...)},
		}
	}
	return s, batches
}

// benchDictCodes is the same column dictionary encoded, with the values shared
// by every batch the way reading a file gives them.
func benchDictCodes(b *testing.B) (dtype.Schema, []ipc.Batch) {
	b.Helper()
	s := dtype.Schema{Fields: []dtype.Field{{Name: "code", Type: codeType, Nullable: true}}}

	distinct := make([]string, benchDictValues)
	for i := range distinct {
		distinct[i] = "code number " + strconv.Itoa(i)
	}
	values := array.OfStrings(distinct...)

	at := make([]int32, benchDictRows)
	for i := range at {
		at[i] = int32(i % benchDictValues)
	}
	batches := make([]ipc.Batch, benchDictBatches)
	for i := range batches {
		col, err := array.NewDictionary(array.Of(at...), values)
		if err != nil {
			b.Fatalf("NewDictionary: %v", err)
		}
		batches[i] = ipc.Batch{Length: benchDictRows, Columns: []*array.Array{col}}
	}
	return s, batches
}

// benchDictCase is one form of the column, and the two of them are the same
// rows written two ways, which is what the numbers are compared across.
type benchDictCase struct {
	name    string
	schema  dtype.Schema
	batches []ipc.Batch
}

func benchDictCases(b *testing.B) []benchDictCase {
	b.Helper()
	text, plain := benchDictStrings(b)
	codes, coded := benchDictCodes(b)
	return []benchDictCase{
		{"strings", text, plain},
		{"dictionary", codes, coded},
	}
}

// BenchmarkDictionaryWrite writes the same rows both ways. The bytes of the
// stream are reported next to the time, since the encoding is there to make the
// stream smaller and a writer that spent longer producing fewer bytes would
// still be the one worth having.
func BenchmarkDictionaryWrite(b *testing.B) {
	for _, bb := range benchDictCases(b) {
		b.Run(bb.name, func(b *testing.B) {
			var buf bytes.Buffer
			b.ReportAllocs()
			for b.Loop() {
				buf.Reset()
				w, err := ipc.NewWriter(&buf, bb.schema)
				if err != nil {
					b.Fatal(err)
				}
				for _, batch := range bb.batches {
					if err := w.Write(batch); err != nil {
						b.Fatal(err)
					}
				}
				if err := w.Close(); err != nil {
					b.Fatal(err)
				}
			}
			b.SetBytes(int64(buf.Len()))
			b.ReportMetric(float64(buf.Len()), "bytes/stream")
		})
	}
}

// BenchmarkDictionaryRead reads the same rows back both ways. A dictionary
// encoded batch is the indices and a pointer to values that have already been
// read, so the work per batch is the work of an int32 column however long the
// strings behind it are.
func BenchmarkDictionaryRead(b *testing.B) {
	for _, bb := range benchDictCases(b) {
		b.Run(bb.name, func(b *testing.B) {
			var buf bytes.Buffer
			w, err := ipc.NewWriter(&buf, bb.schema)
			if err != nil {
				b.Fatal(err)
			}
			for _, batch := range bb.batches {
				if err := w.Write(batch); err != nil {
					b.Fatal(err)
				}
			}
			if err := w.Close(); err != nil {
				b.Fatal(err)
			}
			stream := buf.Bytes()

			b.SetBytes(int64(len(stream)))
			b.ReportAllocs()
			for b.Loop() {
				r, err := ipc.NewReader(bytes.NewReader(stream))
				if err != nil {
					b.Fatal(err)
				}
				for r.Next() {
					_ = r.Batch()
				}
				if err := r.Err(); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(len(stream)), "bytes/stream")
		})
	}
}
