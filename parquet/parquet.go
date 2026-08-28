// Package parquet reads and writes Apache Parquet files.
//
// A parquet file is columns of compressed pages with an index at the end
// saying where every one of them is. That index is the footer, and it is what
// makes the format worth reading: a query that wants three columns out of two
// hundred reads the footer, works out which pages hold those three, and never
// touches the rest of the file. Metadata is that footer, and ReadMetadata is
// how to get it.
//
// The footer is a Thrift structure. This package reads that itself rather than
// pulling in a generated Thrift reader, for the same reason the Arrow IPC
// metadata is read by hand next door: a footer is somebody else's bytes, every
// length in it is a claim rather than a promise, and a reader for this job
// needs to bounds check every read and refuse the claims it cannot satisfy.
//
// Metadata is enough to say what a file holds, how many rows it has, where
// each column chunk lives and what the writer said about its values, which is
// what a scan needs before it reads anything. Metadata.Schema turns the file's
// own schema into kuma types, and Metadata.Columns is the leaves of it with the
// levels a page decoder will need to put nulls and list boundaries back.
//
// ReadPages goes one level down, walking the pages of a column chunk and
// handing back each header with the bytes behind it. The bytes are the ones in
// the file, compressed and encoded as the writer left them, so nothing here
// turns a page into values yet.
//
// The decoders are what turns a page into values. RLEDecoder and
// BitPackedDecoder read the runs of small integers that nulls, list boundaries
// and dictionary indices are written as, which is the hybrid of repeated and
// packed runs parquet uses now and the plain packing it used before that.
// PlainDecoder reads the values themselves, written as they are, which is what
// every other encoding in the format is a way of not doing and what every one
// of them ends up at. DeltaDecoder reads the one that a column of integers is
// written in when a dictionary is not worth keeping, which is the differences
// between the values rather than the values.
//
// ColumnReader is where the two halves meet. A page keeps its levels and its
// values apart and only the rows that have a value are written down, so putting
// a column back together means walking the two together and dropping the values
// in around the nulls. ReadColumn does that for a whole column chunk and hands
// back an array. A chunk that was written as indices into a dictionary, which is
// most of a real file, comes back dictionary encoded rather than expanded, since
// that is the shape it was written in and the shape the kernels would rather
// have it in. What it reads so far is a flat column of plain, dictionary or
// delta encoded pages in a chunk that was not compressed, and anything else is
// refused by name rather than guessed at.
package parquet

import "errors"

// The errors this package returns. Use errors.Is rather than comparing, since
// every one of them arrives wrapped in what was being read at the time.
var (
	// ErrFormat is a file that is not a parquet file, or is one that has been
	// truncated or corrupted. It covers the magic at both ends, the footer
	// length, and everything the Thrift decoder refuses.
	ErrFormat = errors.New("bad parquet file")

	// ErrUnsupported is a file this package understands and cannot read yet.
	// An encrypted footer is the one that turns up in the wild.
	ErrUnsupported = errors.New("not supported yet")
)
