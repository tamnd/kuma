package csv

import (
	"strconv"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// emitter appends value i of a to dst as the field appears in the file, with
// no quotes around it. Whether a field has to be quoted is the same question
// for every column, so the writer asks it once rather than every emitter
// asking it for itself.
type emitter func(dst []byte, a *array.Array, i int) []byte

// emitterFor returns the emitter for a column of type dt, or nil for a type
// that has no text of its own and has to be cast to a string first.
//
// The types with an emitter here are the ones a file is usually made of, and
// the point of writing them straight out is that a column of numbers becomes
// a file without becoming a column of strings on the way. Everything else,
// which today means the timestamps and the dates, goes through the cast, so
// the writer learns a type the moment [kernel.Cast] does.
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
	case dtype.StringKind, dtype.BinaryKind:
		return emitBytes
	default:
		return nil
	}
}

// emitBool writes the words true and false, which are the two [strconv] reads
// back and two of the six the reader infers a boolean from.
func emitBool(dst []byte, a *array.Array, i int) []byte {
	return strconv.AppendBool(dst, a.Bool(i))
}

// emitSigned returns the emitter for a signed integer column.
//
// The values slice is looked up once per chunk rather than once per value.
// Looking it up checks the type and reinterprets the buffer, which is nothing
// beside reading a column and a lot beside formatting one number. Keeping that
// in the closure is safe for the same reason the cast kernel keeps its scratch
// buffer there: an emitter belongs to one writer, which is one goroutine, and
// an array does not change once it has been built.
func emitSigned[T int8 | int16 | int32 | int64]() emitter {
	var last *array.Array
	var vals []T
	return func(dst []byte, a *array.Array, i int) []byte {
		if a != last {
			last, vals = a, a.Values[T]()
		}
		return strconv.AppendInt(dst, int64(vals[i]), 10)
	}
}

// emitUnsigned returns the emitter for an unsigned integer column.
func emitUnsigned[T uint8 | uint16 | uint32 | uint64]() emitter {
	var last *array.Array
	var vals []T
	return func(dst []byte, a *array.Array, i int) []byte {
		if a != last {
			last, vals = a, a.Values[T]()
		}
		return strconv.AppendUint(dst, uint64(vals[i]), 10)
	}
}

// emitFloat returns the emitter for a float column, formatted the way
// [WriteOptions.Precision] asked for.
//
// The default is the shortest text that reads back as the same value, which is
// what a file that will be read again wants and what the cast to a string
// already does. The width is the width of the column, so a float32 that only
// ever had seven digits does not print twenty of them.
func emitFloat[T float32 | float64](bits int, o *WriteOptions) emitter {
	verb, prec := byte('g'), -1
	if o.Precision > 0 {
		verb, prec = 'f', o.Precision
	}

	var last *array.Array
	var vals []T
	return func(dst []byte, a *array.Array, i int) []byte {
		if a != last {
			last, vals = a, a.Values[T]()
		}
		return strconv.AppendFloat(dst, float64(vals[i]), verb, prec, bits)
	}
}

// emitBytes writes the bytes of a column that already holds text.
func emitBytes(dst []byte, a *array.Array, i int) []byte {
	return append(dst, a.Bytes(i)...)
}
