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
// between the values rather than the values. DeltaLengthDecoder and
// DeltaByteArrayDecoder are the same idea for a column of byte arrays: the
// first writes their lengths as differences and puts all the bytes behind them,
// and the second writes how much of each value the one in front of it already
// said, which is what turns a sorted column of keys into a few bytes a row.
//
// PlainEncoder is the same encoding the other way round and is where writing a
// file starts. It writes the values of a page into a buffer, one method per
// physical type and no conversions, because a value written at the wrong width
// is not a wrong value but a different value at every position after it. There
// is nothing in it that writes an int96, since no writer has produced one for
// years and a timestamp in twelve bytes with no zone and no unit is a mistake
// the format has replaced twice over.
//
// RLEEncoder is the levels and the dictionary indices written back out, and
// picking the runs is the whole of what it does: the same values are a legal
// file written as one run or as fifty, so what makes an encoder of this worth
// anything is that it writes the small one. A value that repeats eight times or
// more becomes a repeat and everything else is packed in groups of eight, which
// is what makes the levels of a column with no nulls three bytes however many
// rows it has. There is nothing that writes the encoding this one replaced,
// since that would be writing for a reader that stopped existing years ago.
//
// WritePage is the walk in ReadPages turned around, and it is the one place in
// the writer where a mistake does not look like a mistake. A page has no length
// in front of its header, so a reader finds the second page by reading the first
// header and adding up, which means a compressed size that is one byte out does
// not produce a page that is slightly wrong but a chunk where every page after
// it is nonsense that a reader has no way to tell from a file that was never
// parquet. So the header is checked against the body it was handed rather than
// taken, by the same rule a header read out of a file is checked by, and a page
// this writes is a page this package would accept. The checksum is the one field
// a caller does not fill in, since a caller that computes its own is a caller
// that can get it wrong.
//
// WriteMetadata is the footer, which is what turns encoded pages into a file.
// It writes the Thrift structure ReadMetadata reads, then how long it is, then
// the magic, which is the last of every parquet file. The field numbers are the
// whole of what makes that work: a footer written with the right numbers is read
// by anything and one written with the wrong numbers is read by nothing, so the
// reading and the writing of every structure are kept next to each other and
// every footer in testdata is round tripped through both. What a writer decides
// and a reader never does is which fields to leave out, since nearly everything
// in the format is optional and a field that is absent reads back as the absent
// value of its type.
//
// ColumnReader is where the two halves meet. A page keeps its levels and its
// values apart and only the rows that have a value are written down, so putting
// a column back together means walking the two together and dropping the values
// in around the nulls. ReadColumn does that for a whole column chunk and hands
// back an array. A chunk that was written as indices into a dictionary, which is
// most of a real file, comes back dictionary encoded rather than expanded, since
// that is the shape it was written in and the shape the kernels would rather
// have it in. What it reads so far is a flat column of plain, dictionary or
// delta encoded pages, and anything else is refused by name rather than guessed
// at.
//
// FileReader is the whole of it in one place. It reads the footer, holds it,
// and hands back the columns of one row group at a time. Which columns is what
// Project says, and a projection is the reason the format exists: a file of two
// hundred columns keeps each of them apart from the others, so a reader that
// wants three of them reads three runs of pages and never touches the rest.
// BytesRead is how a caller checks that, since a projection that quietly read
// the whole file would give the same answers at ten times the cost.
//
// Bounds is the other half of not reading a file. A writer usually writes the
// smallest and largest value of every column chunk into the footer, so a scan
// carrying a filter can ask a row group what its columns hold and skip the whole
// group without opening a page of it. What makes that worth care is that parquet
// spent years without saying how its values compare, so a file holds two pairs
// of bounds written by two different rules and only one of them is worth reading
// on most types. ReadBounds is that rule, and FileReader.Bounds is it applied to
// a row group.
//
// The page index is the same idea one level down. A writer that was asked for
// one writes the bounds of every page and the whereabouts of every page into two
// structures at the end of the file, so a scan that cannot skip a row group can
// still tell which of its pages a filter would keep. They are read a column at a
// time, because they are not in the footer and a scan filtering on one column of
// two hundred has no reason to read the index of the rest. ReadPageBounds is the
// two of them together and FileReader.PageBounds is it applied to a row group.
//
// BloomFilter answers for the values bounds cannot. A range says nothing useful
// about a column of identifiers scattered across a file, since every row group
// covers a range with the wanted value somewhere in the middle of it, and yet
// only one group holds it. A writer that was asked for a filter hashed every
// value of a chunk into a bitset at the end of the file, and a reader hashing
// the value it wants looks at the same bits: a bit that is clear means the chunk
// never held it. The other answer is a maybe rather than a yes, so a filter is
// something to skip on and never to answer on. ReadBloomFilter is one chunk's
// and FileReader.BloomFilter is it applied to a row group.
//
// Predicate is what turns all of that into an answer. It is one column compared
// against one value, which is what most of a filter on a scan is made of, and
// FileReader.RowGroups takes a list of them and gives back the row groups that
// may hold a matching row. Every group it leaves out is one whose statistics say
// it holds none, worked out from the footer and, where the writer wrote one,
// from a bloom filter. A group it returns may hold a matching row rather than
// does, and a file whose writer wrote no statistics gives back every group, so
// what comes out of it is where to look rather than what is there.
//
// Options.Filter is both halves together and is what a caller reading a file
// wants. It skips the row groups the statistics rule out and compares the rows
// of the ones it reads, so what comes back is the rows that pass. The columns a
// predicate names do not have to be columns the caller asked for, since
// filtering on a timestamp nobody wants in the result is the ordinary case, and
// a file with no statistics gives the same rows for the cost it always took.
// Nothing there can change an answer, only how much of the file it took.
//
// Decompressor is what undoes the compression of a page on the way through.
// Nearly every parquet file in the world is compressed and the codec is a
// property of a column chunk rather than of the file, so one file may hold a
// snappy column next to a gzip one next to one that was left alone. Those three
// are what is undone so far. Snappy is read by kuma/compress/snappy and gzip by
// the standard library, and a chunk written with a codec that is neither is
// refused by name.
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
