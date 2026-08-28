package kuma

import (
	"io"

	"github.com/tamnd/kuma/parquet"
)

// ReadParquet reads a whole parquet file into a frame.
//
// The size is the size of the file. A parquet file keeps its schema in a footer
// at the end, so the reader has to know where the end is and cannot be handed a
// plain [io.Reader] the way [ReadCSV] can.
//
//	f, err := kuma.ReadParquet(r, size, nil)
//
// A nil options reads every column. Naming the columns is what makes this format
// worth using on a wide file: the values of a column sit together, so a frame of
// three columns out of two hundred reads three runs of pages and never touches
// the rest.
//
//	f, err := kuma.ReadParquet(r, size, &parquet.Options{Columns: []string{"id", "price"}})
//
// A column comes back as the type the file's schema names, whatever the file
// did to store it. Most writers put a dictionary in front of nearly every
// column, and [parquet.Options.Dictionary] is how to keep that encoding for the
// columns it pays off on.
//
// The frame is Dynamic for the same reason a CSV frame is. A file is not a Go
// type, so the columns are asked for by name and read with [Frame.Series]. A
// typed frame comes from kumagen.
func ReadParquet(r io.ReaderAt, size int64, opts *parquet.Options) (*Frame[Dynamic], error) {
	t, err := parquet.Read(r, size, opts)
	if err != nil {
		return nil, err
	}
	return frameOf(t)
}

// ReadParquetFile reads the file at path. It is [ReadParquet] over an open file,
// which is where the size comes from.
func ReadParquetFile(path string, opts *parquet.Options) (*Frame[Dynamic], error) {
	t, err := parquet.ReadFile(path, opts)
	if err != nil {
		return nil, err
	}
	return frameOf(t)
}
