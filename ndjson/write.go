package ndjson

import (
	"encoding/json/jsontext"
	"fmt"
	"io"
	"os"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// flushAt is how many bytes pile up before a write goes out to the writer
// underneath. It is big enough that a line is never a system call on its own and
// small enough that writing a file the size of memory does not need another one
// beside it.
const flushAt = 1 << 16

// WriteOptions controls how a table is written. The zero WriteOptions is the
// useful default: one object to a line, the members named and ordered the way
// the schema is, null where a value is missing, and floats written with the
// fewest digits that read back as the same value.
type WriteOptions struct {
	// Names, if given, is what to call the members instead of what the schema
	// calls the columns. It must have one name for every column.
	Names []string

	// OmitNull leaves a member out of a line rather than writing it as null. It
	// makes a file with a lot of missing values a good deal smaller, and it
	// reads back the same, since a member that is not there and a member that is
	// null are both missing.
	OmitNull bool

	// Precision is how many digits a float is written with. Zero means the
	// shortest text that reads back as the same value, which is what a file that
	// will be read again wants. A number above zero is that many digits after
	// the point, which is what a file that will be looked at wants. A file that
	// wants no digits after the point wants an integer column.
	Precision int
}

// withDefaults returns the options with the zero values filled in.
func (o *WriteOptions) withDefaults() WriteOptions {
	if o == nil {
		return WriteOptions{}
	}
	return *o
}

// WriteFile writes the table to the file at path, creating it if it is not there
// and truncating it if it is.
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

// Write writes the table as newline delimited JSON, one object to a line.
//
// The columns go out as they are stored, so a column of numbers is formatted
// into the output buffer and never becomes a column of strings on the way. The
// members are in the order the schema has the columns, which is the order the
// reader here puts them back in.
//
// A type with no JSON of its own, which today means the timestamps and the
// dates, is cast to a string first and written from that. That costs a copy of
// the column and it is the only case that does.
func Write(w io.Writer, t *Table, opts *WriteOptions) error {
	wr := &writer{dst: w, opts: opts.withDefaults()}
	if err := wr.plan(t); err != nil {
		return err
	}
	return wr.write()
}

// writer is one run of Write. What a line looks like is worked out once, and
// then a row is an append per column.
type writer struct {
	dst  io.Writer
	opts WriteOptions

	names []string
	cols  []cursor
	rows  int
	buf   []byte
}

// plan checks that the table holds together and works out how to write each
// column.
func (w *writer) plan(t *Table) error {
	if len(t.Columns) != t.Schema.Len() {
		return fmt.Errorf("ndjson: %d columns for a schema with %d fields: %w",
			len(t.Columns), t.Schema.Len(), ErrTable)
	}
	if len(w.opts.Names) > 0 && len(w.opts.Names) != len(t.Columns) {
		return fmt.Errorf("ndjson: %d names for a table with %d columns: %w",
			len(w.opts.Names), len(t.Columns), ErrNames)
	}
	w.rows = t.NumRows()

	w.names = w.opts.Names
	if len(w.names) == 0 {
		w.names = t.Schema.Names()
	}

	w.cols = make([]cursor, len(t.Columns))
	for i, data := range t.Columns {
		name := w.names[i]
		if data.Len() != w.rows {
			return fmt.Errorf("ndjson: column %q has %d rows and the first has %d: %w",
				name, data.Len(), w.rows, ErrTable)
		}

		emit := emitterFor(data.DType(), &w.opts)
		if emit == nil {
			// The cast knows how to turn more types into text than this package
			// does, and the ones it does not know about yet are the ones it will
			// learn first.
			text, err := kernel.Cast(data, dtype.String)
			if err != nil {
				return fmt.Errorf("ndjson: column %q: cannot write a %s column as JSON: %w",
					name, data.DType(), ErrUnsupportedType)
			}
			data, emit = text, emitText
		}

		// The name and the colon after it are the same bytes on every line, so
		// they are quoted once here rather than a million times below.
		member, err := jsontext.AppendQuote(nil, name)
		if err != nil {
			return fmt.Errorf("ndjson: column %q: %w", name, err)
		}
		w.cols[i] = newCursor(data, emit, append(member, ':'))
	}
	return nil
}

// write puts out one line per row.
func (w *writer) write() error {
	if len(w.cols) == 0 {
		// No columns is no file. A line holding an empty object every time would
		// read back as no columns and no rows, so there is nothing the rows
		// could be written as that says what they were.
		return nil
	}

	for i := range w.rows {
		if err := w.row(i + 1); err != nil {
			return err
		}
		if len(w.buf) >= flushAt {
			if err := w.flush(); err != nil {
				return err
			}
		}
	}
	return w.flush()
}

// row appends one row of every column as one JSON object.
func (w *writer) row(n int) error {
	w.buf = append(w.buf, '{')

	first := true
	for i := range w.cols {
		c := &w.cols[i]
		null := c.arr.IsNull(c.at)
		if null && w.opts.OmitNull {
			c.step()
			continue
		}

		if !first {
			w.buf = append(w.buf, ',')
		}
		first = false
		w.buf = append(w.buf, c.name...)

		if null {
			w.buf = append(w.buf, "null"...)
		} else {
			p, err := c.emit(w.buf, c.arr, c.at)
			if err != nil {
				return fmt.Errorf("ndjson: column %q, row %d: %w", w.names[i], n, err)
			}
			w.buf = p
		}
		c.step()
	}

	w.buf = append(w.buf, '}', '\n')
	return nil
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

// cursor walks one column a row at a time.
//
// Two columns do not have to be chunked the same way, since concatenating frames
// leaves every column with the chunk boundaries it arrived with, so a row is not
// one offset into all of them. A cursor per column makes stepping to the next
// row a pointer bump rather than the binary search that asking a chunked column
// for value i costs.
type cursor struct {
	data *array.Chunked
	emit emitter
	name []byte // the quoted member name and the colon after it

	arr  *array.Array
	at   int
	next int
}

// newCursor returns a cursor on the first row of the column.
func newCursor(data *array.Chunked, emit emitter, name []byte) cursor {
	c := cursor{data: data, emit: emit, name: name}
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
