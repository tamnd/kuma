package ipc

import (
	"bytes"
	"errors"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// The dictionary messages a kuma writer would not write, and the bookkeeping
// that decides which ones come out of one. What is checked from outside is that
// the values arrive; what is checked here is that they are sent once, that they
// go out under the identifier the schema gave the column, and that a message
// nobody can make sense of is refused rather than guessed at.

// dictCodes is the type the tests in here use, the same one the round trips
// outside use.
var dictCodes = dtype.Dictionary{Index: dtype.Int32, Value: dtype.String}

// dictStream is a schema of one dictionary encoded column.
func dictStream() dtype.Schema {
	return dtype.Schema{Fields: []dtype.Field{{Name: "code", Type: dictCodes, Nullable: true}}}
}

// dictColumn builds a dictionary encoded column out of positions into values.
func dictColumn(t *testing.T, values *array.Array, at ...int32) *array.Array {
	t.Helper()
	col, err := array.NewDictionary(array.Of(at...), values)
	if err != nil {
		t.Fatalf("NewDictionary: %v", err)
	}
	return col
}

// kinds is the messages of a stream in order, which is what says whether a
// dictionary went out once or every time.
func kinds(t *testing.T, stream []byte) []uint8 {
	t.Helper()
	var out []uint8
	r := bytes.NewReader(stream)
	for {
		msg, kind, err := readMessage(r)
		if err != nil {
			t.Fatalf("readMessage: %v", err)
		}
		if msg == nil {
			return out
		}
		out = append(out, kind)
	}
}

// TestDictionarySentOnce writes three batches sharing one dictionary and counts
// the messages. The values are the expensive part and sending them again per
// batch would still read correctly, which is why nothing outside this file
// would notice.
func TestDictionarySentOnce(t *testing.T) {
	values := array.OfStrings("GB", "JP", "US")
	var buf bytes.Buffer
	w, err := NewWriter(&buf, dictStream())
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	for range 3 {
		b := Batch{Length: 2, Columns: []*array.Array{dictColumn(t, values, 0, 2)}}
		if err := w.Write(b); err != nil {
			t.Fatalf("Write: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	want := []uint8{
		fbHeaderSchema,
		fbHeaderDictionaryBatch,
		fbHeaderRecordBatch,
		fbHeaderRecordBatch,
		fbHeaderRecordBatch,
	}
	if got := kinds(t, buf.Bytes()); !bytes.Equal(got, want) {
		t.Errorf("the stream holds the messages %v, want %v", got, want)
	}
}

// TestDictionarySentAgain writes batches holding different values, which is the
// case that has to send them again. The comparison is on the array rather than
// on what is in it, so a second dictionary holding the same strings is sent
// again too, which is the trade for not walking the values once per batch.
func TestDictionarySentAgain(t *testing.T) {
	first := array.OfStrings("GB", "JP")
	second := array.OfStrings("GB", "JP")

	var buf bytes.Buffer
	w, err := NewWriter(&buf, dictStream())
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	for _, values := range []*array.Array{first, second, second} {
		b := Batch{Length: 2, Columns: []*array.Array{dictColumn(t, values, 0, 1)}}
		if err := w.Write(b); err != nil {
			t.Fatalf("Write: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	want := []uint8{
		fbHeaderSchema,
		fbHeaderDictionaryBatch,
		fbHeaderRecordBatch,
		fbHeaderDictionaryBatch,
		fbHeaderRecordBatch,
		fbHeaderRecordBatch,
	}
	if got := kinds(t, buf.Bytes()); !bytes.Equal(got, want) {
		t.Errorf("the stream holds the messages %v, want %v", got, want)
	}
}

// TestDictionaryNotSent drops the values out of a stream and leaves the batch
// that reads from them. A reader that took a missing dictionary as an empty one
// would hand back a column of indices into nothing.
func TestDictionaryNotSent(t *testing.T) {
	values := array.OfStrings("GB", "JP")
	var buf bytes.Buffer
	w, err := NewWriter(&buf, dictStream())
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	if err = w.Write(Batch{Length: 2, Columns: []*array.Array{dictColumn(t, values, 0, 1)}}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err = w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	stream := drop(t, buf.Bytes(), fbHeaderDictionaryBatch)
	r, err := NewReader(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	if r.Next() {
		t.Fatal("Next read a batch whose dictionary never arrived")
	}
	if err := r.Err(); !errors.Is(err, ErrMessage) {
		t.Errorf("Err = %v, want %v", err, ErrMessage)
	}
}

// drop rebuilds a stream without the messages of one kind in it.
func drop(t *testing.T, stream []byte, kind uint8) []byte {
	t.Helper()
	var out []byte
	r := bytes.NewReader(stream)
	for {
		msg, got, err := readMessage(r)
		if err != nil {
			t.Fatalf("readMessage: %v", err)
		}
		if msg == nil {
			return append(out, streamEnd...)
		}
		if got != kind {
			out = append(out, msg...)
		}
	}
}

// TestDictionaryDelta reads a dictionary batch that says it holds values to add
// to the ones already sent rather than values to replace them with. Reading one
// means concatenating two arrays, which this package cannot do without importing
// the kernels, so it says so instead of dropping the values on the floor.
func TestDictionaryDelta(t *testing.T) {
	s := dictStream()
	msg, ids, err := encodeSchemaMessage(s)
	if err != nil {
		t.Fatalf("encodeSchemaMessage: %v", err)
	}
	values := array.OfStrings("GB", "JP")
	first, err := encodeDictionaryBatch(ids[0], values)
	if err != nil {
		t.Fatalf("encodeDictionaryBatch: %v", err)
	}

	var stream []byte
	stream = append(stream, msg...)
	stream = append(stream, first...)
	stream = append(stream, delta(t, ids[0], array.OfStrings("US"))...)
	stream = append(stream, streamEnd...)

	r, err := NewReader(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	if r.Next() {
		t.Fatal("Next read a batch out of a stream that holds none")
	}
	if err := r.Err(); !errors.Is(err, ErrUnsupported) {
		t.Errorf("Err = %v, want %v", err, ErrUnsupported)
	}
}

// delta writes the dictionary batch a writer here never writes, which is the
// one with isDelta set.
func delta(t *testing.T, id int64, values *array.Array) []byte {
	t.Helper()
	s := dictSchema(id, values.DType())
	nodes, body, variadic, err := batchBody(s, Batch{
		Length:  values.Len(),
		Columns: []*array.Array{values},
	})
	if err != nil {
		t.Fatalf("batchBody: %v", err)
	}

	var w fbBuilder
	data := recordBatch(&w, values.Len(), nodes, body.descs, variadic)
	w.startTable()
	w.slotInt(fbDictBatchID, id, -1)
	w.slotOffset(fbDictBatchData, data)
	w.slotBool(fbDictBatchDelta, true)
	batch := w.endTable()

	w.startTable()
	w.slotInt(fbMessageVersion, fbVersionV5, 0)
	w.slotUint8(fbMessageHeaderType, fbHeaderDictionaryBatch)
	w.slotOffset(fbMessageHeader, batch)
	w.slotInt(fbMessageBodyLength, int64(len(body.buf)), 0)
	return append(frame(w.finish(w.endTable())), body.buf...)
}

// TestDictionaryStrangeID reads a dictionary batch carrying an identifier no
// column of the schema reads from. Keeping it would be keeping values nothing
// can ever ask for, and the likelier reading is a stream whose schema and whose
// dictionaries do not belong to each other.
func TestDictionaryStrangeID(t *testing.T) {
	s := dictStream()
	msg, ids, err := encodeSchemaMessage(s)
	if err != nil {
		t.Fatalf("encodeSchemaMessage: %v", err)
	}

	values, err := encodeDictionaryBatch(ids[0]+7, array.OfStrings("GB", "JP"))
	if err != nil {
		t.Fatalf("encodeDictionaryBatch: %v", err)
	}

	var stream []byte
	stream = append(stream, msg...)
	stream = append(stream, values...)
	stream = append(stream, streamEnd...)

	r, err := NewReader(bytes.NewReader(stream))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}
	if r.Next() {
		t.Fatal("Next read a batch out of a stream that holds none")
	}
	if err := r.Err(); !errors.Is(err, ErrMessage) {
		t.Errorf("Err = %v, want %v", err, ErrMessage)
	}
}

// TestDictionarySharedID covers the schema kuma does not write and other
// writers do, where two columns read from one dictionary. That is worth doing
// when the columns hold the same kind of thing, and it is worth refusing when
// they do not, since the values can only be decoded one way.
func TestDictionarySharedID(t *testing.T) {
	strings := dtype.Field{Name: "a", Type: dictCodes, Nullable: true}
	numbers := dtype.Field{
		Name:     "b",
		Type:     dtype.Dictionary{Index: dtype.Int32, Value: dtype.Int64},
		Nullable: true,
	}

	shared := dtype.Schema{Fields: []dtype.Field{strings, {Name: "b", Type: dictCodes, Nullable: true}}}
	d, err := newDicts(shared, []int64{3, 3})
	if err != nil {
		t.Fatalf("newDicts of two columns sharing a dictionary: %v", err)
	}
	if len(d.types) != 1 {
		t.Errorf("two columns sharing a dictionary gave %d of them, want 1", len(d.types))
	}

	clash := dtype.Schema{Fields: []dtype.Field{strings, numbers}}
	if _, err := newDicts(clash, []int64{3, 3}); !errors.Is(err, ErrMessage) {
		t.Errorf("newDicts of a shared dictionary read two ways = %v, want %v", err, ErrMessage)
	}
}

// TestDictionaryNoID covers a schema message that says a column is dictionary
// encoded and carries no identifiers, which is a message this package could
// only have built by losing count.
func TestDictionaryNoID(t *testing.T) {
	if _, err := newDicts(dictStream(), nil); !errors.Is(err, ErrMessage) {
		t.Errorf("newDicts with no identifiers = %v, want %v", err, ErrMessage)
	}
}

// TestDictionaryIdentifiers checks that the identifiers a schema is written
// with are the ones read back out of it. They are the only thing tying a
// dictionary to the column that reads from it, and a writer that numbered the
// columns and a reader that counted them would agree with each other and with
// nobody else.
func TestDictionaryIdentifiers(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64},
		{Name: "code", Type: dictCodes, Nullable: true},
		{Name: "side", Type: dictCodes, Nullable: true},
	}}

	msg, ids, err := encodeSchemaMessage(s)
	if err != nil {
		t.Fatalf("encodeSchemaMessage: %v", err)
	}
	if len(ids) != len(s.Fields) {
		t.Fatalf("the schema was written with %d identifiers, want %d", len(ids), len(s.Fields))
	}
	if ids[1] == ids[2] {
		t.Errorf("both dictionary columns were written under identifier %d, want one each", ids[1])
	}

	body, _, err := unframe(msg)
	if err != nil {
		t.Fatalf("unframe: %v", err)
	}
	back, read, err := decodeSchemaMessage(body)
	if err != nil {
		t.Fatalf("decodeSchemaMessage: %v", err)
	}
	if !back.Equal(s) {
		t.Fatalf("the schema came back as %v, want %v", back, s)
	}
	for i := range s.Fields {
		if read[i] != ids[i] {
			t.Errorf("column %q was written under identifier %d and read as %d",
				s.Fields[i].Name, ids[i], read[i])
		}
	}
}
