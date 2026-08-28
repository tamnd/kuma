package kuma

import (
	"fmt"
	"io"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/csv"
)

// ReadCSV reads a whole CSV file into a frame.
//
// A nil options is the useful default: comma separated, a header row, and the
// type of each column worked out from the first thousand rows. What it decides
// and how to say otherwise is on [csv.Options].
//
//	f, err := kuma.ReadCSV(r, nil)
//
// This reads everything. ScanCSV, which arrives with the lazy frame, reads a
// chunk at a time and never holds more than one of them, which is what a file
// larger than memory needs. Either way a column comes back in chunks, so
// nothing here asks for one allocation the size of the file.
//
// The frame is Dynamic because a file is not a Go type. What is in it was
// decided by whoever wrote the file, so the columns are asked for by name and
// read with [Frame.Series]. A typed frame comes from kumagen, which writes the
// struct out of a sample of the file at build time.
func ReadCSV(r io.Reader, opts *csv.Options) (*Frame[Dynamic], error) {
	t, err := csv.Read(r, opts)
	if err != nil {
		return nil, err
	}
	return frameOf(t)
}

// ReadCSVFile reads the file at path. It is [ReadCSV] over an open file.
func ReadCSVFile(path string, opts *csv.Options) (*Frame[Dynamic], error) {
	t, err := csv.ReadFile(path, opts)
	if err != nil {
		return nil, err
	}
	return frameOf(t)
}

// WriteCSV writes the frame as a comma separated file.
//
// A nil options is the useful default: comma separated, a header line of the
// column names, an empty field where a value is missing, and floats written
// with the fewest digits that read back as the same value. What else it can do
// is on [csv.WriteOptions].
//
//	err := f.WriteCSV(w, nil)
//
// Reading the result back gives the same frame, with one thing to watch: a
// value that is an empty string comes back as a missing value, because a file
// cannot tell the two apart. Set [csv.WriteOptions.NullValue] to something
// else when the difference matters.
func (f *Frame[S]) WriteCSV(w io.Writer, opts *csv.WriteOptions) error {
	return csv.Write(w, f.table(), opts)
}

// WriteCSVFile writes the frame to the file at path, creating it if it is not
// there and truncating it if it is. It is [Frame.WriteCSV] over that file.
func (f *Frame[S]) WriteCSVFile(path string, opts *csv.WriteOptions) error {
	return csv.WriteFile(path, f.table(), opts)
}

// table returns the frame as the schema and columns a file writer takes.
// Nothing is copied: a table is a view of the same columns.
func (f *Frame[S]) table() *array.Table {
	cols := make([]*array.Chunked, len(f.cols))
	for i, c := range f.cols {
		cols[i] = c.Data()
	}
	return &array.Table{Schema: f.schema, Columns: cols}
}

// frameOf turns the columns a reader produced into a frame.
func frameOf(t *array.Table) (*Frame[Dynamic], error) {
	cols := make([]Column, len(t.Columns))
	for i, data := range t.Columns {
		c, err := NewColumn(t.Schema.Fields[i].Name, data)
		if err != nil {
			return nil, fmt.Errorf("kuma: %w", err)
		}
		cols[i] = c
	}
	return NewFrame(cols...)
}
