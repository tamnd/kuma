// Package ipc moves data between kuma and other Arrow implementations.
//
// The Arrow C data interface is how two libraries in one process hand each
// other a column without copying it. It is a pair of C structs, one describing
// the type and one holding the pointers, and the type half of it is a string:
// "l" is an int64, "tsu:UTC" is a microsecond timestamp in UTC, "+l" is a
// list. Format and Type are the two directions of that mapping, and
// EncodeMetadata and DecodeMetadata are the small binary blob that carries the
// key and value pairs attached to a field.
//
// Export and Import are the other half, the values. They work on a Layout,
// which is what the C struct holds once the pointers are Go slices: a length,
// an offset, a null count and the buffers. Neither of them copies a value.
// Import borrows the buffers it is given, which is what the C data interface is
// for, so the array it returns is only alive for as long as the memory behind
// those buffers is.
//
// The C structs themselves are CSchema and CArray, with ExportField,
// ExportArray, ImportField and ImportArray between them and kuma. Those need
// cgo and live behind a build tag, so that a program that only reads Parquet
// does not pay for a C toolchain and still cross compiles. Everything else
// here is pure Go and works on any platform, including the ones with no C
// toolchain at all.
//
// EncodeSchema and DecodeSchema are the other way two libraries hand each other
// a table, which is to write it down. A schema on the wire is an Arrow IPC
// message, meaning FlatBuffers, and this package reads and writes that itself
// rather than pulling in a generated reader, because a message off a socket is
// somebody else's bytes and the generated readers for this format are not
// bounds checked.
//
// EncodeBatch and DecodeBatch are the values that go with a schema. A record
// batch is a message describing where every buffer of every column sits,
// followed by the buffers themselves, and the schema it belongs to is something
// the reader is expected to have already. Nothing is copied on the way in, so
// the arrays a batch decodes into point at the bytes it was decoded from.
//
// Writer and Reader are those two messages put together into the Arrow IPC
// stream format, which is a schema, a record batch for every batch, and a
// marker saying there are no more. That is what a file of Arrow data is, and
// what one process sends another over a socket.
//
// Both interfaces are checked against pyarrow, in both directions, by the tests
// in testdata/pyarrow. The C one runs two libraries in one process and compares
// buffer addresses, and the message one passes files.
//
// What is not here yet: arrays of the nested types, dictionary batches, a
// compressed body, the Arrow IPC file format with its footer and block index,
// and the arrow-go bridge.
//
// Stability: tier 1, stable.
package ipc

import "errors"

// The errors this package returns. They are comparable with errors.Is, and the
// message carries the format string or the type that caused it, since a
// mismatch at this boundary is nearly always a producer and a consumer that
// disagree about one column out of two hundred.
var (
	// ErrFormat is returned when a format string is not one the C data
	// interface defines, or is one that names a type kuma has no equivalent
	// for, such as a union.
	ErrFormat = errors.New("bad format string")

	// ErrType is returned when a kuma type has no format string, which means
	// the type was not built by this module or holds a parameter that is not a
	// real type, such as a time32 counting nanoseconds.
	ErrType = errors.New("type has no format string")

	// ErrChildren is returned when the number of child fields does not match
	// what the format string needs. A list has one child and a map has one
	// child that is a struct of two.
	ErrChildren = errors.New("wrong number of children")

	// ErrMetadata is returned when a metadata blob is truncated, claims a
	// length that is not there, or is too large to be written.
	ErrMetadata = errors.New("bad metadata")

	// ErrBuffers is returned when the buffers of an array are not the ones its
	// type needs: the wrong number of them, one too short for the values it has
	// to hold, or offsets that point outside the data.
	ErrBuffers = errors.New("bad buffers")

	// ErrMessage is returned when a piece of Arrow IPC metadata is malformed:
	// an offset that points outside the message it lives in, a length that runs
	// past the end, a vtable that is not there. The metadata is FlatBuffers, and
	// a FlatBuffer arriving over a socket is somebody else's bytes, so every
	// number in it is checked before it is used.
	ErrMessage = errors.New("bad message")

	// ErrClosed is returned by a write to a stream whose end of stream marker
	// has already gone out. There is nowhere left in the stream to put a batch,
	// so this is a mistake in the calling code rather than anything the bytes
	// did.
	ErrClosed = errors.New("the stream is closed")

	// ErrUnsupported is returned for the shapes this package understands and
	// cannot build yet, which is the nested types, since the array package has
	// no nested arrays to put them in.
	ErrUnsupported = errors.New("not supported yet")
)
