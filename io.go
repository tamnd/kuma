package kuma

import (
	"fmt"
	"io"

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

// frameOf turns the columns a reader produced into a frame.
func frameOf(t *csv.Table) (*Frame[Dynamic], error) {
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
