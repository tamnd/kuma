package ipc_test

import (
	"bytes"
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

// A batch is the values of a schema, so what is checked here is the pair: a
// batch written against a schema and read back against the same schema is the
// same columns. The messages a writer would not write are in the internal test
// beside this one, and the agreement with another implementation is in the
// pyarrow one.

type batchCase struct {
	name   string
	schema dtype.Schema
	batch  ipc.Batch
}

// batchCases is one batch per shape the encoder has to deal with: the fixed
// width columns, the two that carry views, the one with no buffers at all, and
// the batches with nothing in them.
func batchCases(t *testing.T) []batchCase {
	t.Helper()

	mixed := dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64, Nullable: true},
		{Name: "price", Type: dtype.Float64},
		{Name: "live", Type: dtype.Bool, Nullable: true},
		{Name: "symbol", Type: dtype.String, Nullable: true},
		{Name: "payload", Type: dtype.Binary},
		{Name: "at", Type: dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}},
		{Name: "nothing", Type: dtype.Null, Nullable: true},
		{Name: "key", Type: dtype.FixedSizeBinary{ByteWidth: 3}},
	}}

	bools, err := array.NewBuilder(dtype.Bool)
	if err != nil {
		t.Fatalf("NewBuilder = %v", err)
	}
	bools.AppendBool(true)
	bools.AppendNull()
	bools.AppendBool(false)
	bools.AppendBool(true)

	empty := dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64, Nullable: true},
		{Name: "symbol", Type: dtype.String, Nullable: true},
	}}

	return []batchCase{
		{
			name:   "ints",
			schema: dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64, Nullable: true}}},
			batch: ipc.Batch{Length: 4, Columns: []*array.Array{
				buildInts(t, []int64{1, 2, 3, 4}, 1),
			}},
		},
		{
			name:   "mixed",
			schema: mixed,
			batch: ipc.Batch{Length: 4, Columns: []*array.Array{
				buildInts(t, []int64{10, 20, 30, 40}, 2),
				array.Of[float64](1.5, 2.5, 3.5, 4.5),
				bools.Finish(),
				buildStrings(t, []string{"a", "bb", "ccc", "dddd"}, 0, 3),
				buildBinary(t, [][]byte{{1}, {2, 3}, nil, {4, 5, 6}}),
				fixed(t, dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}, 4),
				array.NewNull(4),
				fixed(t, dtype.FixedSizeBinary{ByteWidth: 3}, 4),
			}},
		},
		{
			name:   "blocks",
			schema: dtype.Schema{Fields: []dtype.Field{{Name: "text", Type: dtype.String}}},
			batch: ipc.Batch{Length: blockRows, Columns: []*array.Array{
				manyStrings(t),
			}},
		},
		{
			name:   "empty",
			schema: empty,
			batch: ipc.Batch{Columns: []*array.Array{
				buildInts(t, nil),
				buildStrings(t, nil),
			}},
		},
		{
			name:   "no columns",
			schema: dtype.Schema{},
			batch:  ipc.Batch{Length: 7},
		},
	}
}

// blockRows is enough strings of enough length to fill more than one data
// block, which is the only way to write a variadic buffer count that is not
// one. A block is 32 kilobytes and these are 32 bytes each.
const blockRows = 3000

func manyStrings(t *testing.T) *array.Array {
	t.Helper()
	vals := make([]string, blockRows)
	for i := range vals {
		vals[i] = strings.Repeat("x", 22) + strconv.Itoa(1000000000+i)
	}
	return array.OfStrings(vals...)
}

func TestBatchRoundTrip(t *testing.T) {
	for _, c := range batchCases(t) {
		t.Run(c.name, func(t *testing.T) {
			msg, err := ipc.EncodeBatch(c.schema, c.batch)
			if err != nil {
				t.Fatalf("EncodeBatch: %v", err)
			}
			if len(msg)%8 != 0 {
				t.Errorf("the message and its body are %d bytes, which is not a multiple of 8", len(msg))
			}

			got, rest, err := ipc.DecodeBatch(c.schema, msg)
			if err != nil {
				t.Fatalf("DecodeBatch: %v", err)
			}
			if len(rest) != 0 {
				t.Errorf("%d bytes left after the batch, want none", len(rest))
			}
			if got.Length != c.batch.Length {
				t.Errorf("batch of %d rows, want %d", got.Length, c.batch.Length)
			}
			if len(got.Columns) != len(c.batch.Columns) {
				t.Fatalf("batch of %d columns, want %d", len(got.Columns), len(c.batch.Columns))
			}
			for i, want := range c.batch.Columns {
				equalArrays(t, got.Columns[i], want)
			}
		})
	}
}

// TestBatchBlocks checks that a column of strings long enough to fill several
// data blocks comes back as the same values. The count of them is the one thing
// in the metadata that a reader cannot work out for itself.
func TestBatchBlocks(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{{Name: "text", Type: dtype.String}}}
	col := manyStrings(t)

	l, err := ipc.Export(col)
	if err != nil {
		t.Fatalf("Export: %v", err)
	}
	if len(l.Buffers) < 4 {
		t.Fatalf("the column has %d buffers, want at least 4 so that it has more than one block",
			len(l.Buffers))
	}

	msg, err := ipc.EncodeBatch(s, ipc.Batch{Length: col.Len(), Columns: []*array.Array{col}})
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}
	got, _, err := ipc.DecodeBatch(s, msg)
	if err != nil {
		t.Fatalf("DecodeBatch: %v", err)
	}
	equalArrays(t, got.Columns[0], col)
}

// TestBatchSlice checks the batches that carry an offset, which is what slicing
// a chunk out of a longer one gives. The format has no offset, so the encoder
// has to cut the buffers down, and a validity bitmap that starts part way
// through a byte has to be shifted rather than sliced.
func TestBatchSlice(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64, Nullable: true},
		{Name: "symbol", Type: dtype.String, Nullable: true},
		{Name: "live", Type: dtype.Bool, Nullable: true},
	}}

	vals := make([]int64, 32)
	text := make([]string, 32)
	for i := range vals {
		vals[i] = int64(i)
		text[i] = "row " + strconv.Itoa(i)
	}
	bits, err := array.NewBuilder(dtype.Bool)
	if err != nil {
		t.Fatalf("NewBuilder = %v", err)
	}
	for i := range vals {
		if i%5 == 0 {
			bits.AppendNull()
			continue
		}
		bits.AppendBool(i%2 == 0)
	}

	ints := buildInts(t, vals, 3, 17)
	strs := buildStrings(t, text, 0, 31)
	bools := bits.Finish()

	// Offsets on and off a byte boundary, since the aligned one is a subslice
	// and the other is a copy with every bit moved.
	for _, r := range []struct{ lo, hi int }{{0, 32}, {8, 24}, {3, 19}, {31, 32}, {5, 5}} {
		t.Run(strconv.Itoa(r.lo)+"-"+strconv.Itoa(r.hi), func(t *testing.T) {
			cols := []*array.Array{
				ints.Slice(r.lo, r.hi),
				strs.Slice(r.lo, r.hi),
				bools.Slice(r.lo, r.hi),
			}
			msg, err := ipc.EncodeBatch(s, ipc.Batch{Length: r.hi - r.lo, Columns: cols})
			if err != nil {
				t.Fatalf("EncodeBatch: %v", err)
			}
			got, _, err := ipc.DecodeBatch(s, msg)
			if err != nil {
				t.Fatalf("DecodeBatch: %v", err)
			}
			for i, want := range cols {
				equalArrays(t, got.Columns[i], want)
			}
		})
	}
}

// TestBatchZeroCopy checks that the values of a decoded batch are the bytes the
// message was read from rather than a copy of them. A reader that copies is a
// reader that has doubled the cost of every file it opens.
func TestBatchZeroCopy(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64}}}
	col := benchInts()

	msg, err := ipc.EncodeBatch(s, ipc.Batch{Length: col.Len(), Columns: []*array.Array{col}})
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}
	got, _, err := ipc.DecodeBatch(s, msg)
	if err != nil {
		t.Fatalf("DecodeBatch: %v", err)
	}

	l, err := ipc.Export(got.Columns[0])
	if err != nil {
		t.Fatalf("Export: %v", err)
	}
	at := bytes.Index(msg, l.Buffers[1])
	if at < 0 {
		t.Fatal("the values of the decoded column are not in the message at all")
	}
	if !sameBytes(l.Buffers[1], msg[at:]) {
		t.Error("the values of the decoded column are a copy, want the message itself")
	}
}

// TestBatchStream checks that batches read back one after another, which is what
// a stream is once the framing is off the schema in front of it.
func TestBatchStream(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64, Nullable: true}}}
	want := [][]int64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}

	var stream []byte
	for _, vals := range want {
		msg, err := ipc.EncodeBatch(s, ipc.Batch{Length: len(vals), Columns: []*array.Array{
			buildInts(t, vals),
		}})
		if err != nil {
			t.Fatalf("EncodeBatch: %v", err)
		}
		stream = append(stream, msg...)
	}

	rest := stream
	for i, vals := range want {
		var got ipc.Batch
		var err error
		if got, rest, err = ipc.DecodeBatch(s, rest); err != nil {
			t.Fatalf("batch %d: DecodeBatch: %v", i, err)
		}
		equalArrays(t, got.Columns[0], buildInts(t, vals))
	}
	if len(rest) != 0 {
		t.Errorf("%d bytes left after three batches, want none", len(rest))
	}
}

func TestEncodeBatchError(t *testing.T) {
	ints := dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64}}}
	col := buildInts(t, []int64{1, 2, 3})

	for _, tt := range []struct {
		name   string
		schema dtype.Schema
		batch  ipc.Batch
		want   error
	}{
		{
			name:   "too few columns",
			schema: ints,
			batch:  ipc.Batch{Length: 3},
			want:   ipc.ErrBuffers,
		},
		{
			name:   "nil column",
			schema: ints,
			batch:  ipc.Batch{Length: 3, Columns: []*array.Array{nil}},
			want:   ipc.ErrBuffers,
		},
		{
			name:   "another type",
			schema: dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int32}}},
			batch:  ipc.Batch{Length: 3, Columns: []*array.Array{col}},
			want:   ipc.ErrBuffers,
		},
		{
			name:   "another length",
			schema: ints,
			batch:  ipc.Batch{Length: 2, Columns: []*array.Array{col}},
			want:   ipc.ErrBuffers,
		},
		{
			name:   "negative length",
			schema: dtype.Schema{},
			batch:  ipc.Batch{Length: -1},
			want:   ipc.ErrBuffers,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := ipc.EncodeBatch(tt.schema, tt.batch); !errors.Is(err, tt.want) {
				t.Errorf("EncodeBatch = %v, want %v", err, tt.want)
			}
		})
	}
}

func TestDecodeBatchError(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64}}}
	msg, err := ipc.EncodeBatch(s, ipc.Batch{Length: 3, Columns: []*array.Array{
		buildInts(t, []int64{1, 2, 3}),
	}})
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}
	schemaMsg, err := ipc.EncodeSchema(s)
	if err != nil {
		t.Fatalf("EncodeSchema: %v", err)
	}

	for _, tt := range []struct {
		name   string
		schema dtype.Schema
		msg    []byte
		want   error
	}{
		{"nothing", s, nil, ipc.ErrMessage},
		{"end of stream", s, []byte{0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0}, ipc.ErrMessage},
		{"a schema", s, schemaMsg, ipc.ErrMessage},
		{"cut short", s, msg[:len(msg)-8], ipc.ErrMessage},
		{"another schema", dtype.Schema{}, msg, ipc.ErrMessage},
		{
			name:   "a nested column",
			schema: dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.List{Elem: dtype.Int64}}}},
			msg:    msg,
			want:   ipc.ErrUnsupported,
		},
		{
			// A dictionary column decodes to its indices, so a schema saying the
			// column is one reads the int64 batch as int64 indices and is fine.
			// An index type that is not an integer is not a dictionary anything
			// could have written.
			name:   "a dictionary indexed by strings",
			schema: dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Dictionary{Index: dtype.String, Value: dtype.String}}}},
			msg:    msg,
			want:   ipc.ErrType,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, _, err := ipc.DecodeBatch(tt.schema, tt.msg); !errors.Is(err, tt.want) {
				t.Errorf("DecodeBatch = %v, want %v", err, tt.want)
			}
		})
	}
}

// TestDecodeBatchCorrupt changes one byte at a time and checks that nothing
// panics and that whatever comes back says which kind of wrong it is.
//
// The metadata is offsets and the buffer descriptions are numbers pointing into
// the body, so a single byte changed anywhere is either a message the reader has
// to refuse or a set of buffers the array package has to. Plenty of changes are
// still a readable batch, and that is fine: what is not fine is a panic or a
// read past the end of the body.
func TestDecodeBatchCorrupt(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64, Nullable: true},
		{Name: "symbol", Type: dtype.String, Nullable: true},
	}}
	msg, err := ipc.EncodeBatch(s, ipc.Batch{Length: 3, Columns: []*array.Array{
		buildInts(t, []int64{1, 2, 3}, 1),
		buildStrings(t, []string{"a", "bb", "ccc"}, 2),
	}})
	if err != nil {
		t.Fatalf("EncodeBatch: %v", err)
	}

	for i := range msg {
		for _, b := range []byte{0x00, 0x01, 0x7F, 0xFF} {
			bad := append([]byte(nil), msg...)
			bad[i] = b

			batch, _, err := ipc.DecodeBatch(s, bad)
			if err != nil {
				if !errors.Is(err, ipc.ErrMessage) && !errors.Is(err, ipc.ErrBuffers) &&
					!errors.Is(err, ipc.ErrUnsupported) {
					t.Fatalf("byte %d set to %#x: %v, want one of the errors this package returns", i, b, err)
				}
				continue
			}
			// A batch that was accepted has to be readable, since the whole
			// point of checking the buffers is that no read after this one has
			// to check anything.
			for _, col := range batch.Columns {
				for k := range col.Len() {
					if !col.IsNull(k) {
						valueAt(col, k)
					}
				}
			}
		}
	}
}

// benchBatch is a batch of the shape a file arrives in: a few columns of
// several thousand rows, which is what one call of this pair costs on the way
// through a reader or a writer.
func benchBatch() (dtype.Schema, ipc.Batch) {
	s := dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64},
		{Name: "count", Type: dtype.Int64},
		{Name: "price", Type: dtype.Float64},
		{Name: "symbol", Type: dtype.String},
	}}

	prices := make([]float64, benchRows)
	for i := range prices {
		prices[i] = float64(i) / 4
	}
	return s, ipc.Batch{Length: benchRows, Columns: []*array.Array{
		benchInts(),
		benchInts(),
		array.Of(prices...),
		benchStrings(),
	}}
}

func BenchmarkEncodeBatch(b *testing.B) {
	s, batch := benchBatch()
	b.ReportAllocs()
	for b.Loop() {
		if _, err := ipc.EncodeBatch(s, batch); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeBatch(b *testing.B) {
	s, batch := benchBatch()
	msg, err := ipc.EncodeBatch(s, batch)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, _, err := ipc.DecodeBatch(s, msg); err != nil {
			b.Fatal(err)
		}
	}
}
