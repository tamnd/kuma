package ipc

import (
	"fmt"
	"strconv"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// The Arrow IPC dictionary batch message, which is how the values of a
// dictionary encoded column travel.
//
// A dictionary encoded column is split in two on the wire. The record batch
// carries the indices, which are an integer column like any other, and the
// values are sent once in a message of their own and used by every batch after
// it. A file of ten thousand batches of country codes holds the two hundred and
// fifty strings once.
//
// The two halves are tied together by an identifier. The schema says that a
// column is dictionary encoded and gives the identifier of the dictionary it
// reads from, and a dictionary batch says which identifier it is carrying. The
// identifier is the only thing connecting them, so a reader has to keep the
// numbers the schema gave it rather than assume they count from zero, and a
// writer has to give the same number to the same column every time.
//
// A dictionary batch is a record batch with a number on it. What is inside it is
// one column of values, described exactly the way a column of a record batch is,
// which is why the two of them are encoded and decoded by the same code with
// different wrapping.
//
// The values may be sent again under an identifier that has already been used,
// which is a replacement, and every batch after it reads the new values. A
// stream allows that and a file does not: a file is read in whatever order the
// reader likes, so a dictionary that changes half way through would mean the
// answer depended on where the reader started. A delta, which is values to add
// to a dictionary already sent rather than values to put in its place, is
// refused for now.

// The field numbers of the DictionaryBatch table, in declaration order.
const (
	fbDictBatchID = iota
	fbDictBatchData
	fbDictBatchDelta
)

// dictSchema is the schema of the one column inside a dictionary batch. The
// name is only ever read out of an error, where it says which dictionary of a
// wide schema the trouble is in.
func dictSchema(id int64, t dtype.DataType) dtype.Schema {
	return dtype.Schema{Fields: []dtype.Field{{
		Name:     "dictionary " + strconv.FormatInt(id, 10),
		Type:     t,
		Nullable: true,
	}}}
}

// encodeDictionaryBatch returns the message carrying values under identifier id,
// with the values after it, framed the way a record batch is.
func encodeDictionaryBatch(id int64, values *array.Array) ([]byte, error) {
	if values == nil {
		return nil, fmt.Errorf("ipc: %w: dictionary %d has no values", ErrBuffers, id)
	}

	s := dictSchema(id, values.DType())
	nodes, body, variadic, err := batchBody(s, Batch{
		Length:  values.Len(),
		Columns: []*array.Array{values},
	})
	if err != nil {
		return nil, err
	}

	msg := encodeDictionaryMessage(id, values.Len(), nodes, body.descs, variadic, len(body.buf))
	if err := checkLength(len(msg), "a dictionary batch"); err != nil {
		return nil, err
	}
	return append(frame(msg), body.buf...), nil
}

// encodeDictionaryMessage writes the metadata on its own, which is the record
// batch table a record batch message would hold with the identifier wrapped
// around it.
//
// The delta flag is left at false, since these are the values of the dictionary
// rather than values to add to it.
func encodeDictionaryMessage(id int64, length int, nodes, buffers []span, variadic []int64, bodyLength int) []byte {
	var w fbBuilder
	data := recordBatch(&w, length, nodes, buffers, variadic)

	w.startTable()
	w.slotInt(fbDictBatchID, id, -1)
	w.slotOffset(fbDictBatchData, data)
	batch := w.endTable()

	w.startTable()
	w.slotInt(fbMessageVersion, fbVersionV5, 0)
	w.slotUint8(fbMessageHeaderType, fbHeaderDictionaryBatch)
	w.slotOffset(fbMessageHeader, batch)
	w.slotInt(fbMessageBodyLength, int64(bodyLength), 0)
	return w.finish(w.endTable())
}

// decodeDictionaryBatch reads one dictionary batch message and returns the
// identifier it carries, the values, and whatever follows the message.
//
// The types say what the values under each identifier are, since a dictionary
// batch on the wire is a length and some buffers and nothing that says what is
// in them. That is the same bargain a record batch makes with its schema, and
// the map is what a schema has instead for these.
func decodeDictionaryBatch(types map[int64]dtype.DataType, b []byte) (int64, *array.Array, []byte, error) {
	table, body, after, err := messageBody(b, fbHeaderDictionaryBatch)
	if err != nil {
		return 0, nil, nil, err
	}

	id, err := table.integer(fbDictBatchID, int64(0))
	if err != nil {
		return 0, nil, nil, err
	}
	delta, err := table.boolean(fbDictBatchDelta, false)
	if err != nil {
		return 0, nil, nil, err
	}
	if delta {
		return 0, nil, nil, fmt.Errorf("ipc: %w: dictionary %d arrives as a delta, which is values to add to the ones already sent, and joining the two is not written yet",
			ErrUnsupported, id)
	}

	t, ok := types[id]
	if !ok {
		return 0, nil, nil, fmt.Errorf("ipc: %w: a dictionary batch under identifier %d, which no column of the schema reads from",
			ErrMessage, id)
	}
	data, ok, err := table.table(fbDictBatchData)
	if err != nil {
		return 0, nil, nil, err
	}
	if !ok {
		return 0, nil, nil, fmt.Errorf("ipc: %w: dictionary %d says it holds values and holds nothing",
			ErrMessage, id)
	}

	batch, err := decodeBatch(dictSchema(id, t), data, body)
	if err != nil {
		return 0, nil, nil, err
	}
	return id, batch.Columns[0], after, nil
}

// dicts is the dictionaries of one stream or one file, as they arrive.
//
// It is nil for a schema with no dictionary encoded columns in it, which is
// nearly every schema, and the methods take that as the answer rather than as
// something to check for. A reader that never sees a dictionary allocates
// nothing and does no work per batch.
type dicts struct {
	// ids is the identifier of each column of the schema, straight out of the
	// message the schema came in. Only the entries whose column is dictionary
	// encoded mean anything.
	ids []int64

	// types is what the values under each identifier are, which is what a
	// dictionary batch needs and does not carry.
	types map[int64]dtype.DataType

	// values is what has arrived so far. A column whose identifier is not in
	// here yet is a batch that came before its dictionary.
	values map[int64]*array.Array
}

// newDicts reads the dictionary encoded columns out of a schema. The ids are
// the ones the schema message carried, one per field.
func newDicts(s dtype.Schema, ids []int64) (*dicts, error) {
	var d *dicts
	for i, f := range s.Fields {
		t, ok := f.Type.(dtype.Dictionary)
		if !ok {
			continue
		}
		if i >= len(ids) {
			return nil, fmt.Errorf("ipc: %w: column %q is dictionary encoded and has no identifier",
				ErrMessage, f.Name)
		}
		if d == nil {
			d = &dicts{
				ids:    ids,
				types:  make(map[int64]dtype.DataType),
				values: make(map[int64]*array.Array),
			}
		}

		// Two columns may read from one dictionary, which is what a table of
		// several columns of country codes should be doing. What they may not do
		// is disagree about what is in it.
		if was, shared := d.types[ids[i]]; shared && !dtype.Equal(was, t.Value) {
			return nil, fmt.Errorf("ipc: %w: dictionary %d holds the values of a %s column and of a %s column",
				ErrMessage, ids[i], was, t.Value)
		}
		d.types[ids[i]] = t.Value
	}
	return d, nil
}

// read takes one dictionary batch message and keeps the values in it. Values
// under an identifier that has already been used replace the ones there, which
// is what a stream is allowed to do.
func (d *dicts) read(b []byte) error {
	if d == nil {
		return fmt.Errorf("ipc: %w: a dictionary batch where no column of the schema is dictionary encoded",
			ErrMessage)
	}
	id, values, _, err := decodeDictionaryBatch(d.types, b)
	if err != nil {
		return err
	}
	d.values[id] = values
	return nil
}

// count is how many dictionaries have arrived, which is how a file catches a
// dictionary written twice: it knows from its footer how many were written.
func (d *dicts) count() int {
	if d == nil {
		return 0
	}
	return len(d.values)
}

// bind turns the index columns of a batch into the dictionary encoded columns
// the schema says they are. The columns are replaced in the batch it was given,
// since the batch has just been decoded and nobody else is holding it.
//
// Every index is checked against the dictionary here, by the array package, so
// after this no kernel reading the column has to wonder whether the value it is
// pointed at exists.
func (d *dicts) bind(s dtype.Schema, b *Batch) error {
	if d == nil {
		return nil
	}
	for i, f := range s.Fields {
		if _, ok := f.Type.(dtype.Dictionary); !ok {
			continue
		}
		values, ok := d.values[d.ids[i]]
		if !ok {
			return fmt.Errorf("ipc: %w: column %q reads from dictionary %d, which has not been sent",
				ErrMessage, f.Name, d.ids[i])
		}
		col, err := array.NewDictionary(b.Columns[i], values)
		if err != nil {
			return fmt.Errorf("ipc: %w: column %q: %w", ErrBuffers, f.Name, err)
		}
		b.Columns[i] = col
	}
	return nil
}

// dictWriter is the other half: which dictionary each column writes under and
// what it wrote last time.
//
// It is nil for a schema with no dictionary encoded columns, the same way dicts
// is, so a writer of ordinary columns carries nothing and does nothing per
// batch.
type dictWriter struct {
	ids []int64

	// last is the values written for each column so far, compared by identity
	// rather than by value. Two batches of one column share the dictionary they
	// were built with, which is a pointer comparison, and a writer that compared
	// the values instead would walk the whole dictionary once per batch to find
	// out what it already knew.
	last []*array.Array
}

// dictMessage is one dictionary batch about to go out: the message, the column
// it belongs to and the values in it, which the writer remembers once the bytes
// are really written.
type dictMessage struct {
	col    int
	values *array.Array
	msg    []byte
}

// newDictWriter gives the dictionary encoded columns of a schema somewhere to
// be remembered. The ids are the ones the schema message was written with.
func newDictWriter(s dtype.Schema, ids []int64) *dictWriter {
	for _, f := range s.Fields {
		if _, ok := f.Type.(dtype.Dictionary); ok {
			return &dictWriter{ids: ids, last: make([]*array.Array, len(s.Fields))}
		}
	}
	return nil
}

// messages is the dictionary batches that have to go out in front of a record
// batch: one for every column whose values have not been written yet, and one
// for every column holding different values from the ones that were.
//
// Nothing is remembered here. A message that is built and then not written,
// because something later in the same batch turned out to be wrong, would
// otherwise never be written at all.
func (d *dictWriter) messages(s dtype.Schema, b Batch) ([]dictMessage, error) {
	if d == nil {
		return nil, nil
	}
	if len(b.Columns) != len(s.Fields) {
		return nil, fmt.Errorf("ipc: %w: a batch of %d columns for a schema of %d fields",
			ErrBuffers, len(b.Columns), len(s.Fields))
	}

	var out []dictMessage
	for i, f := range s.Fields {
		if _, ok := f.Type.(dtype.Dictionary); !ok {
			continue
		}
		values, err := dictValues(f, b.Columns[i])
		if err != nil {
			return nil, err
		}
		if values == d.last[i] {
			continue
		}
		msg, err := encodeDictionaryBatch(d.ids[i], values)
		if err != nil {
			return nil, err
		}
		out = append(out, dictMessage{col: i, values: values, msg: msg})
	}
	return out, nil
}

// once is messages for a file, where a dictionary is written for the first time
// or not at all.
//
// The batches of a file are read in whatever order the reader wants them, and a
// dictionary that was replaced half way through would mean a column read
// differently depending on where the reader started. So the format has one
// dictionary per identifier, and a column that arrives with a second one is a
// caller who has to concatenate the batches first.
func (d *dictWriter) once(s dtype.Schema, b Batch) ([]dictMessage, error) {
	out, err := d.messages(s, b)
	if err != nil {
		return nil, err
	}
	for _, m := range out {
		if d.last[m.col] != nil {
			return nil, fmt.Errorf("ipc: %w: column %q arrives with a second dictionary, and a file holds one dictionary per column",
				ErrUnsupported, s.Fields[m.col].Name)
		}
	}
	return out, nil
}

// done says that a message really went out, so the next batch sharing those
// values writes nothing.
func (d *dictWriter) done(m dictMessage) { d.last[m.col] = m.values }

// dictValues is the values of a dictionary encoded column, which a writer has to
// have. A column holding only its indices is one whose values are somewhere the
// writer cannot reach, and writing the indices on their own would produce a file
// nothing can read.
func dictValues(f dtype.Field, col *array.Array) (*array.Array, error) {
	if col == nil {
		return nil, fmt.Errorf("ipc: %w: column %q is nil", ErrBuffers, f.Name)
	}
	values := col.Dictionary()
	if values == nil {
		return nil, fmt.Errorf("ipc: %w: column %q is the indices of a dictionary rather than the dictionary, so there are no values to write",
			ErrBuffers, f.Name)
	}
	return values, nil
}
