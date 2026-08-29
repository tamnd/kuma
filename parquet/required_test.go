package parquet_test

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"strconv"
	"testing"

	"github.com/tamnd/kuma/parquet"
)

// The required fields of a footer, by where they live in it.
//
// Nearly every field in parquet is optional, and the few that are not are the
// ones a writer has to write whatever they hold. A reader that finds one missing
// is entitled to refuse the whole footer before it has looked at anything in it,
// and the generated thrift readers the other implementations are built on do
// exactly that: the file opens nowhere, with an error about the protocol rather
// than about the field.
//
// That is a mistake this package cannot make against itself, since the reader
// here fills a struct in and a field that never arrived reads back as nought.
// So the check is against the format rather than against the reader, and this is
// the list, written out of parquet.thrift by hand.
//
// The key is the path of field numbers from the footer down, with the lists
// collapsed, so 4.1.3 is the metadata of a column chunk of a row group. The
// deprecated ones are in here too, deprecated meaning nothing should read it
// rather than nobody wants it.
var required = map[string][]int16{
	"":      {1, 2, 3, 4},             // FileMetaData: version, schema, num_rows, row_groups
	"2":     {4},                      // SchemaElement: name
	"4":     {1, 2, 3},                // RowGroup: columns, total_byte_size, num_rows
	"4.1":   {2},                      // ColumnChunk: file_offset
	"4.1.3": {1, 2, 3, 4, 5, 6, 7, 9}, // ColumnMetaData: type through total_compressed_size, data_page_offset
}

// TestWriteRequired checks that a footer this writer wrote holds every field the
// format says it has to.
//
// The table is the one with a column of every type in it, so between the columns
// it covers the values a writer is most likely to leave out by mistake: a nought
// that means something, an offset that happens to be at the start, a chunk with
// nothing missing in it.
func TestWriteRequired(t *testing.T) {
	for _, opts := range []*parquet.WriteOptions{nil, {RowGroupSize: 2, PageSize: 16}} {
		_, raw := writtenBytes(t, everyType(t), opts)
		checkRequired(t, raw[footerStart(t, raw):len(raw)-8])
	}
}

// TestWriteMetadataRequired checks the same of every footer in testdata written
// back out, which is the other way a required field goes missing: the reader
// keeps it, the writer has a condition on it, and the condition is false for
// some file nobody tried.
func TestWriteMetadataRequired(t *testing.T) {
	for _, name := range files(t) {
		t.Run(name, func(t *testing.T) {
			var buf bytes.Buffer
			if _, err := parquet.WriteMetadata(&buf, read(t, name)); err != nil {
				t.Fatalf("WriteMetadata: %v", err)
			}

			b := buf.Bytes()
			checkRequired(t, b[:len(b)-8])
		})
	}
}

// checkRequired walks a footer and fails the test for every required field that
// is not in it.
func checkRequired(t *testing.T, footer []byte) {
	t.Helper()

	w := &walker{buf: footer, seen: make(map[string]map[int16]bool)}
	if err := w.structure(""); err != nil {
		t.Fatalf("walking the footer: %v", err)
	}
	if w.pos != len(w.buf) {
		t.Fatalf("the footer is %d bytes and the walk read %d", len(w.buf), w.pos)
	}

	for path, fields := range required {
		got, ok := w.seen[path]
		if !ok {
			// A footer of no row groups has no chunks in it and nothing below
			// them, which is a file rather than a mistake.
			continue
		}
		for _, id := range fields {
			if !got[id] {
				t.Errorf("the structure at %q leaves out field %d, which is required",
					pathName(path), id)
			}
		}
	}
}

// pathName is a path of field numbers with the root named, so a failure reads as
// somewhere in a footer rather than as an empty string.
func pathName(path string) string {
	if path == "" {
		return "the footer"
	}
	return path
}

// under is the path of field id inside the structure at path.
func under(path string, id int16) string {
	if path == "" {
		return strconv.Itoa(int(id))
	}
	return path + "." + strconv.Itoa(int(id))
}

// walker reads enough of the thrift compact protocol to say which fields each
// structure of a footer holds.
//
// It is a second reader of the same bytes on purpose. The one in the package
// fills in structs and a field that was never written reads back as the nought
// of its type, which is exactly the mistake being looked for, so a test built on
// it could not see one.
type walker struct {
	buf []byte
	pos int

	// seen is which field numbers turned up at each path. A list of structures
	// puts its elements at the path of the field holding it, since the elements
	// of one list are all the same structure.
	seen map[string]map[int16]bool
}

// The type codes of the compact protocol, which are in the field headers and in
// the list headers.
const (
	thriftTrue = iota + 1
	thriftFalse
	thriftByte
	thriftInt16
	thriftInt32
	thriftInt64
	thriftDouble
	thriftBinary
	thriftList
	thriftSet
	thriftMap
	thriftStruct
)

// structure reads one structure and everything under it.
func (w *walker) structure(path string) error {
	if w.seen[path] == nil {
		w.seen[path] = make(map[int16]bool)
	}

	last := int16(0)
	for {
		head, err := w.byte()
		if err != nil {
			return err
		}
		if head == 0 {
			return nil
		}

		id := last + int16(head>>4)
		if head>>4 == 0 {
			n, err := w.zigzag()
			if err != nil {
				return err
			}
			id = int16(n)
		}
		last = id
		w.seen[path][id] = true

		if err := w.value(head&0xf, under(path, id)); err != nil {
			return err
		}
	}
}

// value reads one value of the given type, which for a structure means reading
// the whole of it.
func (w *walker) value(kind byte, path string) error {
	switch kind {
	case thriftTrue, thriftFalse:
		// The value of a boolean is in the field header and there is nothing
		// behind it.
		return nil

	case thriftByte:
		_, err := w.byte()
		return err

	case thriftInt16, thriftInt32, thriftInt64:
		_, err := w.zigzag()
		return err

	case thriftDouble:
		return w.skip(8)

	case thriftBinary:
		n, err := w.varint()
		if err != nil {
			return err
		}
		return w.skip(int(n))

	case thriftList, thriftSet:
		return w.list(path)

	case thriftMap:
		return w.mapping()

	case thriftStruct:
		return w.structure(path)

	default:
		return fmt.Errorf("a field of type %d at %d", kind, w.pos)
	}
}

// list reads a list or a set, whose elements are all of one type.
func (w *walker) list(path string) error {
	head, err := w.byte()
	if err != nil {
		return err
	}

	n := int64(head >> 4)
	if n == 15 {
		if n, err = w.varint(); err != nil {
			return err
		}
	}
	if n < 0 || n > int64(len(w.buf)) {
		return fmt.Errorf("a list of %d at %d", n, w.pos)
	}

	for range n {
		if err := w.value(head&0xf, path); err != nil {
			return err
		}
	}
	return nil
}

// mapping reads a map, which a footer has none of and which is here so that a
// footer holding one is walked rather than refused.
func (w *walker) mapping() error {
	n, err := w.varint()
	if err != nil {
		return err
	}
	if n == 0 {
		return nil
	}
	if n < 0 || n > int64(len(w.buf)) {
		return fmt.Errorf("a map of %d at %d", n, w.pos)
	}

	kinds, err := w.byte()
	if err != nil {
		return err
	}
	for range n {
		if err := w.value(kinds>>4, ""); err != nil {
			return err
		}
		if err := w.value(kinds&0xf, ""); err != nil {
			return err
		}
	}
	return nil
}

func (w *walker) byte() (byte, error) {
	if w.pos >= len(w.buf) {
		return 0, fmt.Errorf("the footer ends at %d", w.pos)
	}
	b := w.buf[w.pos]
	w.pos++
	return b, nil
}

func (w *walker) skip(n int) error {
	if n < 0 || n > len(w.buf)-w.pos {
		return fmt.Errorf("%d bytes at %d of a footer of %d", n, w.pos, len(w.buf))
	}
	w.pos += n
	return nil
}

func (w *walker) varint() (int64, error) {
	v, n := binary.Uvarint(w.buf[w.pos:])
	if n <= 0 || v > math.MaxInt64 {
		return 0, fmt.Errorf("a number at %d", w.pos)
	}
	w.pos += n
	return int64(v), nil
}

func (w *walker) zigzag() (int64, error) {
	v, err := w.varint()
	if err != nil {
		return 0, err
	}
	return int64(uint64(v)>>1) ^ -(v & 1), nil
}
