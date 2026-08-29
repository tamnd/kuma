package ndjson

import (
	"encoding/base64"
	"encoding/json/jsontext"
	"math"
	"strconv"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// emitter appends value i of a to dst as the JSON it goes out as, quotes and
// all. It is the writing half of an [appender] and the two are meant to read
// back what the other wrote.
//
// It returns an error because a string is the one value that can turn out not to
// be writable: JSON is UTF-8 and a string column that arrived from somewhere
// that does not check might hold bytes that are not.
type emitter func(dst []byte, a *array.Array, i int) ([]byte, error)

// emitterFor returns the emitter for a column of type dt, or nil for a type that
// has no JSON of its own and has to be cast to a string first.
//
// The types with an emitter here are the ones a file is usually made of, and the
// point of writing them straight out is that a column of numbers becomes a file
// without becoming a column of strings on the way. Everything else, which today
// means the timestamps and the dates, goes through the cast, so the writer
// learns a type the moment [kernel.Cast] does.
func emitterFor(dt dtype.DataType, o *WriteOptions) emitter {
	switch dt.Kind() {
	case dtype.BoolKind:
		return emitBool
	case dtype.Int8Kind:
		return emitSigned[int8]()
	case dtype.Int16Kind:
		return emitSigned[int16]()
	case dtype.Int32Kind:
		return emitSigned[int32]()
	case dtype.Int64Kind:
		return emitSigned[int64]()
	case dtype.Uint8Kind:
		return emitUnsigned[uint8]()
	case dtype.Uint16Kind:
		return emitUnsigned[uint16]()
	case dtype.Uint32Kind:
		return emitUnsigned[uint32]()
	case dtype.Uint64Kind:
		return emitUnsigned[uint64]()
	case dtype.Float32Kind:
		return emitFloat[float32](32, o)
	case dtype.Float64Kind:
		return emitFloat[float64](64, o)
	case dtype.StringKind:
		return emitText
	case dtype.BinaryKind:
		return emitBinary
	default:
		return nil
	}
}

// emitBool writes the literals true and false.
func emitBool(dst []byte, a *array.Array, i int) ([]byte, error) {
	return strconv.AppendBool(dst, a.Bool(i)), nil
}

// emitSigned returns the emitter for a signed integer column.
//
// The values slice is looked up once per chunk rather than once per value.
// Looking it up checks the type and reinterprets the buffer, which is nothing
// beside reading a column and a lot beside formatting one number. Keeping that
// in the closure is safe for the same reason the cast kernel keeps its scratch
// buffer there: an emitter belongs to one writer, which is one goroutine, and an
// array does not change once it has been built.
//
// An int64 goes out as a JSON number and keeps every digit it had. A reader that
// holds every number as a float64, which is what a browser does, rounds the ones
// past two to the fifty third; this package reads its own files back exactly.
func emitSigned[T int8 | int16 | int32 | int64]() emitter {
	var last *array.Array
	var vals []T
	return func(dst []byte, a *array.Array, i int) ([]byte, error) {
		if a != last {
			last, vals = a, a.Values[T]()
		}
		return strconv.AppendInt(dst, int64(vals[i]), 10), nil
	}
}

// emitUnsigned returns the emitter for an unsigned integer column.
func emitUnsigned[T uint8 | uint16 | uint32 | uint64]() emitter {
	var last *array.Array
	var vals []T
	return func(dst []byte, a *array.Array, i int) ([]byte, error) {
		if a != last {
			last, vals = a, a.Values[T]()
		}
		return strconv.AppendUint(dst, uint64(vals[i]), 10), nil
	}
}

// emitFloat returns the emitter for a float column, formatted the way
// [WriteOptions.Precision] asked for.
//
// The default is the shortest text that reads back as the same value, which is
// what a file that will be read again wants. The width is the width of the
// column, so a float32 that only ever had seven digits does not print twenty of
// them.
//
// A NaN and the two infinities go out as null. JSON has no way to write them,
// the two other things a writer can do are stop or put something in the file
// that is not JSON, and null is the one of the three that leaves a file every
// reader can read.
func emitFloat[T float32 | float64](bits int, o *WriteOptions) emitter {
	prec := o.Precision

	var last *array.Array
	var vals []T
	return func(dst []byte, a *array.Array, i int) ([]byte, error) {
		if a != last {
			last, vals = a, a.Values[T]()
		}
		v := float64(vals[i])
		switch {
		case math.IsNaN(v) || math.IsInf(v, 0):
			return append(dst, "null"...), nil
		case prec > 0:
			return strconv.AppendFloat(dst, v, 'f', prec, bits), nil
		default:
			return jsontext.AppendFloat(dst, v, bits), nil
		}
	}
}

// emitText writes a string column as a JSON string.
//
// The escaping is [jsontext]'s, which is the shortest that reads back as the
// same string: a quote, a backslash and the control characters go out as
// escapes and everything else goes out as itself. Nothing is escaped for the
// sake of a browser, since a file is data rather than something to paste into a
// page.
func emitText(dst []byte, a *array.Array, i int) ([]byte, error) {
	return jsontext.AppendQuote(dst, a.Bytes(i))
}

// emitBinary writes a binary column as base64 in a JSON string, which is what
// [encoding/json] does with a []byte and what the reader here reads back.
func emitBinary(dst []byte, a *array.Array, i int) ([]byte, error) {
	dst = append(dst, '"')
	dst = base64.StdEncoding.AppendEncode(dst, a.Bytes(i))
	return append(dst, '"'), nil
}
