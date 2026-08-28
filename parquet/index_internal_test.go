package parquet

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"testing"

	"github.com/tamnd/kuma/dtype"
)

// The indexes no writer produces.
//
// A page index is read out of the file rather than out of the footer, on
// offsets the footer gave, and it is what a scan decides not to read a page on.
// So the cases worth writing are the ones a real file does not hold: an index
// that bounds more pages than it locates, one that locates them in the wrong
// order, one that counts more missing values than a page has rows, and one
// whose offset points somewhere the file does not go. The bytes are written by
// hand because nothing else writes them.

// columnIndex writes a column index holding the given pages.
func columnIndex(nulls []bool, lo, hi [][]byte, counts []int64) *builder {
	w := &builder{}

	w.field(1, thriftList).list(len(nulls), thriftTrue)
	for _, v := range nulls {
		t := thriftFalse
		if v {
			t = thriftTrue
		}
		w.raw(byte(t))
	}

	for id, values := range [][][]byte{lo, hi} {
		w.field(int16(id)+2, thriftList).list(len(values), thriftBinary)
		for _, v := range values {
			w.binary(string(v))
		}
	}

	w.field(4, thriftInt32).varint(int64(Ascending))
	if counts != nil {
		w.field(5, thriftList).list(len(counts), thriftInt64)
		for _, v := range counts {
			w.varint(v)
		}
	}
	return w.raw(0)
}

// offsetIndex writes an offset index locating a page at each of the given rows.
func offsetIndex(rows []int64) *builder {
	w := &builder{}

	w.field(1, thriftList).list(len(rows), thriftStruct)
	for i, row := range rows {
		w.structure(func() {
			w.field(1, thriftInt64).varint(int64(100 + i*50))
			w.field(2, thriftInt32).varint(50)
			w.field(3, thriftInt64).varint(row)
		})
	}
	return w.raw(0)
}

// four is a value of an int32 column as a bound holds it, which is the bytes a
// page would hold and not the four bytes of length a page writes in front of a
// string.
func four(v int32) []byte { return binary.LittleEndian.AppendUint32(nil, uint32(v)) }

// forged lays the two indexes out behind a file's magic and hands back the
// chunk pointing at them, which is what a footer would have said.
func forged(column, offset []byte) (*bytes.Reader, int64, *ColumnChunk) {
	buf := append([]byte("PAR1"), column...)
	c := &ColumnChunk{Meta: ColumnMeta{Path: []string{"n"}}}
	if len(column) > 0 {
		c.ColumnIndexOffset, c.ColumnIndexLength = 4, int32(len(column))
	}
	if len(offset) > 0 {
		c.OffsetIndexOffset, c.OffsetIndexLength = int64(len(buf)), int32(len(offset))
		buf = append(buf, offset...)
	}
	return bytes.NewReader(buf), int64(len(buf)), c
}

// intColumn is a column of int32, which is the column every forged index here is
// read against.
func intColumn() Column {
	return Column{
		Path:    []string{"n"},
		Element: SchemaElement{Name: "n", Type: Int32, Repetition: Optional},
		Type:    dtype.Int32,
		Order:   TypeDefinedOrder,
	}
}

// TestReadColumnIndexRefused is the indexes that contradict themselves.
//
// The three lists are parallel and are read as parallel, so one of them being
// short is not something to notice halfway through a scan. Neither is a list of
// something other than what the format says goes in it.
func TestReadColumnIndexRefused(t *testing.T) {
	pages := []bool{false, false}
	lo, hi := [][]byte{four(0), four(100)}, [][]byte{four(99), four(199)}

	short := &builder{}
	short.field(1, thriftList).list(2, thriftBinary).binary("a").binary("b").raw(0)

	cases := []struct {
		name  string
		bytes []byte
	}{
		{"fewer smallest values than pages", columnIndex(pages, lo[:1], hi, nil).b},
		{"fewer largest values than pages", columnIndex(pages, lo, hi[:1], nil).b},
		{"a count of missing values for one page of two", columnIndex(pages, lo, hi, []int64{0}).b},
		{"a list of strings where the null pages go", short.b},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			src, size, chunk := forged(c.bytes, nil)
			if _, err := ReadColumnIndex(src, size, chunk); !errors.Is(err, ErrFormat) {
				t.Errorf("got %v, want %v", err, ErrFormat)
			}
		})
	}
}

// TestReadOffsetIndexRefused is an offset index that is not one.
func TestReadOffsetIndexRefused(t *testing.T) {
	w := &builder{}
	w.field(1, thriftList).list(2, thriftBinary).binary("a").binary("b").raw(0)

	src, size, chunk := forged(nil, w.b)
	if _, err := ReadOffsetIndex(src, size, chunk); !errors.Is(err, ErrFormat) {
		t.Errorf("got %v, want %v", err, ErrFormat)
	}
}

// TestReadPageBoundsRefused is the pages a reader will not act on.
//
// Every one of these would have a scan read the wrong bytes or skip a page
// holding rows a filter wanted, which is the mistake that changes an answer
// without saying it did.
func TestReadPageBoundsRefused(t *testing.T) {
	two := []bool{false, false}
	lo, hi := [][]byte{four(0), four(100)}, [][]byte{four(99), four(199)}

	cases := []struct {
		name   string
		column []byte
		offset []byte
		rows   int64
	}{
		{
			"a column index and no offset index",
			columnIndex(two, lo, hi, nil).b, nil, 200,
		},
		{
			"more pages bounded than located",
			columnIndex(two, lo, hi, nil).b, offsetIndex([]int64{0}).b, 200,
		},
		{
			"a first page that does not start at the first row",
			columnIndex(two, lo, hi, nil).b, offsetIndex([]int64{10, 100}).b, 200,
		},
		{
			"pages in the wrong order",
			columnIndex([]bool{false, false, false}, append(lo, four(200)), append(hi, four(299)), nil).b,
			offsetIndex([]int64{0, 100, 50}).b, 200,
		},
		{
			"a page starting past the end of the row group",
			columnIndex(two, lo, hi, nil).b, offsetIndex([]int64{0, 500}).b, 200,
		},
		{
			"more missing values than the page holds",
			columnIndex(two, lo, hi, []int64{0, 300}).b, offsetIndex([]int64{0, 100}).b, 200,
		},
		{
			"a count of missing values below nought",
			columnIndex(two, lo, hi, []int64{-1, 0}).b, offsetIndex([]int64{0, 100}).b, 200,
		},
		{
			"a page missing every value and counting some of them",
			columnIndex([]bool{false, true}, lo, hi, []int64{0, 5}).b, offsetIndex([]int64{0, 100}).b, 200,
		},
		{
			"a bound too short to be a value",
			columnIndex(two, [][]byte{{1}, four(100)}, hi, nil).b, offsetIndex([]int64{0, 100}).b, 200,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			src, size, chunk := forged(c.column, c.offset)
			if _, err := ReadPageBounds(src, size, chunk, intColumn(), c.rows); !errors.Is(err, ErrFormat) {
				t.Errorf("got %v, want %v", err, ErrFormat)
			}
		})
	}
}

// TestReadPageBoundsEmpty is a chunk whose index bounds no pages at all, which
// is nothing to read and nothing to refuse.
func TestReadPageBoundsEmpty(t *testing.T) {
	src, size, chunk := forged(columnIndex(nil, nil, nil, nil).b, offsetIndex(nil).b)

	b, err := ReadPageBounds(src, size, chunk, intColumn(), 0)
	if err != nil {
		t.Fatalf("ReadPageBounds: %v", err)
	}
	if len(b) != 0 {
		t.Errorf("an index of no pages came back with %d", len(b))
	}
}

// TestReadPageBoundsFields reads past the parts of the two indexes this package
// has no use for, which is what lets a file written by a newer writer be read
// by this one.
//
// The column index carries two histograms of levels, which say how many values
// of a page are at each level and mean something only on a column of lists. The
// offset index carries the unencoded size of every page of a byte array column,
// which says what reading one would cost rather than what is in it.
func TestReadPageBoundsFields(t *testing.T) {
	x := columnIndex([]bool{false}, [][]byte{four(0)}, [][]byte{four(99)}, []int64{0})
	x.b = x.b[:len(x.b)-1]
	x.field(6, thriftList).list(2, thriftInt64).varint(0).varint(1).raw(0)

	o := &builder{}
	o.field(1, thriftList).list(1, thriftStruct)
	o.structure(func() {
		o.field(1, thriftInt64).varint(100)
		o.field(2, thriftInt32).varint(50)
		o.field(3, thriftInt64).varint(0)
		o.field(4, thriftBinary).binary("what a later writer wrote here")
	})
	o.field(2, thriftList).list(1, thriftInt64).varint(400).raw(0)

	src, size, chunk := forged(x.b, o.b)
	b, err := ReadPageBounds(src, size, chunk, intColumn(), 100)
	if err != nil {
		t.Fatalf("ReadPageBounds: %v", err)
	}
	if len(b) != 1 {
		t.Fatalf("%d pages, want 1", len(b))
	}
	if b[0].Values == nil || b[0].Values.Value[int32](0) != 0 || b[0].Values.Value[int32](1) != 99 {
		t.Errorf("the page came back bounded by %v, want 0 to 99", b[0].Values)
	}
	if b[0].Offset != 100 || b[0].CompressedSize != 50 {
		t.Errorf("the page is %d bytes at %d, want 50 at 100", b[0].CompressedSize, b[0].Offset)
	}
}

// TestReadPageBoundsNaN drops the bounds of a page bounded by a NaN, the same
// as a row group bounded by one.
//
// NaN compares false against everything, so a page bounded by one has no range
// for a filter to be kept out of. What is left is where the page is, and a scan
// that cannot rule it out reads it.
func TestReadPageBoundsNaN(t *testing.T) {
	nan := binary.LittleEndian.AppendUint32(nil, 0x7fc00000)
	c := intColumn()
	c.Element.Type, c.Type = Float, dtype.Float32

	src, size, chunk := forged(
		columnIndex([]bool{false}, [][]byte{nan}, [][]byte{four(0)}, nil).b,
		offsetIndex([]int64{0}).b,
	)

	b, err := ReadPageBounds(src, size, chunk, c, 100)
	if err != nil {
		t.Fatalf("ReadPageBounds: %v", err)
	}
	if len(b) != 1 || b[0].Values != nil {
		t.Errorf("a page whose smallest value is a NaN came back bounded")
	}
	if b[0].Count != 100 {
		t.Errorf("the page holds %d rows, want 100", b[0].Count)
	}
}

// gone is a file that stops answering, which is what a read of an index has to
// come back from, the indexes being the one part of the metadata that is not in
// the footer already.
type gone struct{}

var errGoneIndex = errors.New("the file stopped answering")

func (gone) ReadAt([]byte, int64) (int, error) { return 0, errGoneIndex }

// TestIndexBytesRefused is an index the file cannot hold and an index the file
// will not give up.
//
// The offsets are numbers out of a footer, so a claim that an index is four
// gigabytes at the end of a file of two hundred bytes is a claim to check
// before anything is allocated for it.
func TestIndexBytesRefused(t *testing.T) {
	x := columnIndex([]bool{false}, [][]byte{four(0)}, [][]byte{four(99)}, nil).b
	src, size, chunk := forged(x, offsetIndex([]int64{0}).b)

	t.Run("an index past the end of the file", func(t *testing.T) {
		c := *chunk
		c.ColumnIndexOffset = size + 1

		if _, err := ReadColumnIndex(src, size, &c); !errors.Is(err, ErrFormat) {
			t.Errorf("got %v, want %v", err, ErrFormat)
		}
	})

	t.Run("an index running past the end of the file", func(t *testing.T) {
		c := *chunk
		c.OffsetIndexLength = int32(size)

		if _, err := ReadOffsetIndex(src, size, &c); !errors.Is(err, ErrFormat) {
			t.Errorf("got %v, want %v", err, ErrFormat)
		}
	})

	t.Run("a file that stops answering", func(t *testing.T) {
		if _, err := ReadPageBounds(gone{}, size, chunk, intColumn(), 100); !errors.Is(err, errGoneIndex) {
			t.Errorf("got %v, want %v", err, errGoneIndex)
		}
	})

	// The offset index is read after the column index, so a file that gives up
	// the first and not the second is its own case.
	t.Run("a file that stops answering halfway", func(t *testing.T) {
		var half io.ReaderAt = &halting{at: src, ok: 1}

		if _, err := ReadPageBounds(half, size, chunk, intColumn(), 100); !errors.Is(err, errGoneIndex) {
			t.Errorf("got %v, want %v", err, errGoneIndex)
		}
	})
}

// halting is a file that answers a few reads and then stops.
type halting struct {
	at io.ReaderAt
	ok int
}

func (h *halting) ReadAt(p []byte, off int64) (int, error) {
	if h.ok <= 0 {
		return 0, errGoneIndex
	}
	h.ok--
	return h.at.ReadAt(p, off)
}
