package csv

import (
	"errors"
	"fmt"
	"strconv"
	"unicode/utf8"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// errNotUTF8 is what a string column says about bytes that are not text. It is
// deliberately not exported: a caller asks whether an error is [ErrValue] and
// then reads the message.
var errNotUTF8 = errors.New("not valid UTF-8")

// appender puts one field into a column. It returns what the parse said and
// leaves saying which line and which column to the reader, which is the only
// one that knows.
type appender func(b *array.Builder, s string) error

// appenderFor returns the appender for a column of type dt.
//
// The type is worked out once per column rather than once per field, so the
// inner loop of the read is a call through one closure and not a switch over
// every type the library has. The parse is the same one [kernel.Cast] does
// from text: a value is read as the type that was asked for rather than as a
// number in general, so 3.9 is not an int64 and neither is 1e3.
func appenderFor(dt dtype.DataType) (appender, error) {
	switch dt.Kind() {
	case dtype.BoolKind:
		return appendBool, nil
	case dtype.Int8Kind:
		return appendSigned[int8](8), nil
	case dtype.Int16Kind:
		return appendSigned[int16](16), nil
	case dtype.Int32Kind:
		return appendSigned[int32](32), nil
	case dtype.Int64Kind:
		return appendSigned[int64](64), nil
	case dtype.Uint8Kind:
		return appendUnsigned[uint8](8), nil
	case dtype.Uint16Kind:
		return appendUnsigned[uint16](16), nil
	case dtype.Uint32Kind:
		return appendUnsigned[uint32](32), nil
	case dtype.Uint64Kind:
		return appendUnsigned[uint64](64), nil
	case dtype.Float32Kind:
		return appendFloat[float32](32), nil
	case dtype.Float64Kind:
		return appendFloat[float64](64), nil
	case dtype.StringKind:
		return appendText, nil
	case dtype.BinaryKind:
		return appendBytes, nil
	default:
		return nil, fmt.Errorf("cannot read text into a %s column: %w",
			dt, ErrUnsupportedType)
	}
}

// appendBool reads one of the words strconv knows.
func appendBool(b *array.Builder, s string) error {
	v, err := strconv.ParseBool(s)
	if err != nil {
		return cause(err)
	}
	b.AppendBool(v)
	return nil
}

// appendSigned returns the appender for a signed integer column of the given
// width. The width goes to strconv, so a value it accepts is a value that fits
// and the conversion below cannot lose anything.
func appendSigned[T int8 | int16 | int32 | int64](bits int) appender {
	return func(b *array.Builder, s string) error {
		v, err := strconv.ParseInt(s, 10, bits)
		if err != nil {
			return cause(err)
		}
		b.Append(T(v))
		return nil
	}
}

// appendUnsigned returns the appender for an unsigned integer column.
func appendUnsigned[T uint8 | uint16 | uint32 | uint64](bits int) appender {
	return func(b *array.Builder, s string) error {
		v, err := strconv.ParseUint(s, 10, bits)
		if err != nil {
			return cause(err)
		}
		b.Append(T(v))
		return nil
	}
}

// appendFloat returns the appender for a float column.
func appendFloat[T float32 | float64](bits int) appender {
	return func(b *array.Builder, s string) error {
		v, err := strconv.ParseFloat(s, bits)
		if err != nil {
			return cause(err)
		}
		b.Append(T(v))
		return nil
	}
}

// appendText adds a field to a string column, which is every field except the
// ones that are not text at all.
//
// A string column holds UTF-8 and nothing else, so a file in some other
// encoding is reported here rather than becoming a column that prints as
// nonsense. Read it as binary if the bytes are what is wanted.
func appendText(b *array.Builder, s string) error {
	if !utf8.ValidString(s) {
		return errNotUTF8
	}
	b.AppendString(s)
	return nil
}

// appendBytes adds a field to a binary column, where anything goes.
func appendBytes(b *array.Builder, s string) error {
	b.AppendString(s)
	return nil
}

// cause returns what a strconv error was really about, which is the syntax or
// range error inside it rather than the wrapper carrying the function name and
// the input. The input is already in the ValueError this ends up in.
func cause(err error) error {
	var ne *strconv.NumError
	if errors.As(err, &ne) {
		return ne.Err
	}
	return err
}
