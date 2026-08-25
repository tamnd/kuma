package csv

import (
	"fmt"
	"io"
	"os"
	"unicode"
	"unicode/utf8"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// flushAt is how many bytes pile up before a write goes out to the writer
// underneath. It is big enough that a row is never a system call on its own
// and small enough that writing a file the size of memory does not need
// another one beside it.
const flushAt = 1 << 16

// WriteOptions controls how a table is written. The zero WriteOptions is the
// useful default: comma separated, a header row, an empty field for a missing
// value, and floats written with the fewest digits that read back as the same
// value.
type WriteOptions struct {
	// Delimiter is what goes between fields. Zero means a comma.
	Delimiter rune

	// NoHeader leaves out the line of column names.
	NoHeader bool

	// Names, if given, is what to write on the header line instead of the
	// names in the schema. It must have one name for every column.
	Names []string

	// NullValue is what a missing value is written as. The default is an empty
	// field, which is what the reader reads back as missing.
	NullValue string

	// Precision is how many digits a float is written with. Zero means the
	// shortest text that reads back as the same value, which is what a file
	// that will be read again wants. A number above zero is that many digits
	// after the point, which is what a file that will be looked at wants. A
	// file that wants no digits after the point wants an integer column.
	Precision int

	// QuoteAll puts quotes around every field rather than around the ones that
	// need them. Some readers outside Go expect it and no reader minds it.
	QuoteAll bool

	// CRLF ends each line with a carriage return and a newline. The default is
	// a newline on every platform, because a file is data rather than text on
	// a screen.
	CRLF bool
}

// withDefaults returns the options with the zero values filled in.
func (o *WriteOptions) withDefaults() WriteOptions {
	out := WriteOptions{}
	if o != nil {
		out = *o
	}
	if out.Delimiter == 0 {
		out.Delimiter = ','
	}
	return out
}

// WriteFile writes the table to the file at path, creating it if it is not
// there and truncating it if it is.
func WriteFile(path string, t *Table, opts *WriteOptions) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	err = Write(f, t, opts)

	// The close is not thrown away here, the way it is after a read. A write
	// that returned nothing is still in the hands of the operating system, and
	// close is where a full disk is reported.
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	return nil
}

// Write writes the table as a comma separated file.
//
// The columns go out as they are stored, so a column of numbers is formatted
// into the output buffer and never becomes a column of strings on the way. A
// value is quoted when it holds the delimiter, a quote or a line ending, and
// left alone when it does not, which is the rule [encoding/csv] writes by and
// the rule this package reads back.
//
// A type with no text of its own, which today means the timestamps and the
// dates, is cast to a string first and written from that. That costs a copy of
// the column and it is the only case that does.
func Write(w io.Writer, t *Table, opts *WriteOptions) error {
	o := opts.withDefaults()
	if err := checkDelim(o.Delimiter, 0); err != nil {
		return err
	}

	wr := &writer{dst: w, opts: o}
	wr.setup()
	if err := wr.plan(t); err != nil {
		return err
	}
	return wr.write()
}

// writer is one run of Write. What the file looks like is worked out once, and
// then a row is an append per column.
type writer struct {
	dst  io.Writer
	opts WriteOptions

	cols []cursor
	rows int

	buf []byte
	tmp []byte

	// special says which bytes force a field to be quoted, delim, eol and null
	// are the bytes to write for a separator, a line ending and a missing
	// value, and header is the first line, already quoted.
	special    [256]bool
	quoteEmpty bool
	delim      []byte
	eol        []byte
	null       []byte
	header     []byte
}

// setup works out the bytes that go between and around the values.
func (w *writer) setup() {
	w.special['"'] = true
	w.special['\r'] = true
	w.special['\n'] = true

	// A delimiter outside ASCII is several bytes and every one of them is
	// marked. That quotes the odd field that did not need it and never leaves
	// one unquoted that did, which is the direction to be wrong in.
	w.delim = []byte(string(w.opts.Delimiter))
	for _, b := range w.delim {
		w.special[b] = true
	}

	w.eol = []byte("\n")
	if w.opts.CRLF {
		w.eol = []byte("\r\n")
	}
}

// plan checks that the table holds together and works out how to write each
// column.
func (w *writer) plan(t *Table) error {
	if len(t.Columns) != t.Schema.Len() {
		return fmt.Errorf("csv: %d columns for a schema with %d fields: %w",
			len(t.Columns), t.Schema.Len(), ErrTable)
	}
	if len(w.opts.Names) > 0 && len(w.opts.Names) != len(t.Columns) {
		return fmt.Errorf("csv: %d names for a table with %d columns: %w",
			len(w.opts.Names), len(t.Columns), ErrNames)
	}
	w.rows = t.NumRows()

	w.cols = make([]cursor, len(t.Columns))
	for i, data := range t.Columns {
		name := t.Schema.Fields[i].Name
		if data.Len() != w.rows {
			return fmt.Errorf("csv: column %q has %d rows and the first has %d: %w",
				name, data.Len(), w.rows, ErrTable)
		}

		emit := emitterFor(data.DType(), &w.opts)
		if emit == nil {
			// The cast knows how to turn more types into text than this
			// package does, and the ones it does not know about yet are the
			// ones it will learn first.
			text, err := kernel.Cast(data, dtype.String)
			if err != nil {
				return fmt.Errorf("csv: column %q: cannot write a %s column as text: %w",
					name, data.DType(), ErrUnsupportedType)
			}
			data, emit = text, emitBytes
		}
		w.cols[i] = newCursor(data, emit)
	}

	// A table of one column writes an empty field as a pair of quotes rather
	// than as nothing, because a line with nothing on it is a blank line and a
	// blank line is not a row at all to the reader. With a second column there
	// is a delimiter on the line and the question does not arise.
	w.quoteEmpty = len(t.Columns) == 1
	w.null = w.enquote([]byte(w.opts.NullValue), 0)

	if !w.opts.NoHeader {
		names := w.opts.Names
		if len(names) == 0 {
			names = t.Schema.Names()
		}
		w.header = w.headerLine(names)
	}
	return nil
}

// write puts out the header and then the rows.
func (w *writer) write() error {
	if len(w.cols) == 0 {
		// No columns is no file. A line of nothing would read back as one
		// column of empty values rather than as no columns at all.
		return nil
	}

	w.buf = append(w.buf, w.header...)
	for range w.rows {
		w.row()
		if len(w.buf) >= flushAt {
			if err := w.flush(); err != nil {
				return err
			}
		}
	}
	return w.flush()
}

// row appends one row of every column.
func (w *writer) row() {
	for i := range w.cols {
		if i > 0 {
			w.buf = append(w.buf, w.delim...)
		}

		c := &w.cols[i]
		if c.arr.IsNull(c.at) {
			w.buf = append(w.buf, w.null...)
		} else {
			n := len(w.buf)
			w.buf = c.emit(w.buf, c.arr, c.at)
			w.buf = w.enquote(w.buf, n)
		}
		c.step()
	}
	w.buf = append(w.buf, w.eol...)
}

// headerLine returns the line of column names.
func (w *writer) headerLine(names []string) []byte {
	var line []byte
	for i, name := range names {
		if i > 0 {
			line = append(line, w.delim...)
		}
		n := len(line)
		line = append(line, name...)
		line = w.enquote(line, n)
	}
	return append(line, w.eol...)
}

// flush hands what has been built to the writer underneath.
func (w *writer) flush() error {
	if len(w.buf) == 0 {
		return nil
	}
	if _, err := w.dst.Write(w.buf); err != nil {
		return err
	}
	w.buf = w.buf[:0]
	return nil
}

// enquote puts quotes around the field that starts at n in dst, if it needs
// them, and returns dst with the field rewritten.
//
// The field is already in the buffer by the time this looks at it, because the
// common answer is that nothing has to happen and the cheapest way to write a
// field that needs no quotes is to have written it. The rewrite copies the
// field out and back, which is the price of the rarer answer.
func (w *writer) enquote(dst []byte, n int) []byte {
	if !w.opts.QuoteAll && !w.needsQuotes(dst[n:]) {
		return dst
	}

	w.tmp = append(w.tmp[:0], dst[n:]...)
	dst = append(dst[:n], '"')
	for _, b := range w.tmp {
		if b == '"' {
			dst = append(dst, b)
		}
		dst = append(dst, b)
	}
	return append(dst, '"')
}

// needsQuotes reports whether a field has to be quoted to read back as itself.
//
// This is the rule [encoding/csv] writes by. A field holding a quote, a line
// ending or the delimiter needs them, and so does one that starts with a space,
// since a reader is allowed to trim that. So does the single field holding a
// backslash and a dot, which is what a Postgres COPY reads as the end of the
// data. An empty field needs them only in a table of one column, where the
// line would otherwise be blank and the row would be lost.
func (w *writer) needsQuotes(p []byte) bool {
	if len(p) == 0 {
		return w.quoteEmpty
	}
	if len(p) == 2 && p[0] == '\\' && p[1] == '.' {
		return true
	}
	for _, b := range p {
		if w.special[b] {
			return true
		}
	}
	r, _ := utf8.DecodeRune(p)
	return unicode.IsSpace(r)
}

// cursor walks one column a row at a time.
//
// Two columns do not have to be chunked the same way, since concatenating
// frames leaves every column with the chunk boundaries it arrived with, so a
// row is not one offset into all of them. A cursor per column makes stepping
// to the next row a pointer bump rather than the binary search that asking a
// chunked column for value i costs.
type cursor struct {
	data *array.Chunked
	emit emitter

	arr  *array.Array
	at   int
	next int
}

// newCursor returns a cursor on the first row of the column.
func newCursor(data *array.Chunked, emit emitter) cursor {
	c := cursor{data: data, emit: emit}
	c.seek()
	return c
}

// seek moves to the next chunk that has a row in it.
func (c *cursor) seek() {
	for c.next < c.data.NumChunks() {
		a := c.data.Chunk(c.next)
		c.next++
		if a.Len() > 0 {
			c.arr, c.at = a, 0
			return
		}
	}
	c.arr = nil
}

// step moves to the next row of the column.
func (c *cursor) step() {
	c.at++
	if c.at == c.arr.Len() {
		c.seek()
	}
}
