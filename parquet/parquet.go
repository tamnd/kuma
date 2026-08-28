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
// Nothing here decodes a page yet. Metadata is enough to say what a file holds,
// how many rows it has, where each column chunk lives and what the writer said
// about its values, which is what a scan needs before it reads anything.
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
