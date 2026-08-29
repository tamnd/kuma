package ndjson

import (
	"bytes"
	"encoding/base64"
	"encoding/json/jsontext"
	"errors"
	"fmt"
	"math"
	"strconv"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// What a value of the wrong kind for its column says. A column has one type and
// JSON says what a value is, so these are about the file rather than about the
// text of one value, which is what the strconv errors are about.
var (
	errNotBool   = errors.New("not a boolean")
	errNotNumber = errors.New("not a number")
	errNotString = errors.New("not a string")
)

// appender puts the next value of a line into a column.
//
// It returns the JSON text of the value, and it only returns it when the value
// would not go in. Formatting a value costs an allocation, an error is about to
// be built anyway when there is one, and the values that go in are all of them
// but a handful.
//
// An appender reads exactly one value whether or not that value goes in, so a
// line with a bad value in the middle of it is still a line the reader can walk
// to the end of. That is what [Options.IgnoreParseErrors] needs to be able to
// carry on.
type appender func(b *array.Builder, d *jsontext.Decoder) (string, error)

// appenderFor returns the appender for a column of type dt.
//
// The type is worked out once per column rather than once per value, so the
// inner loop of the read is a call through one closure and not a switch over
// every type the library has.
//
// A value is read as the JSON type it is. A column that was declared in
// [Options.Types] also takes a quoted value and reads the text inside it, since
// a caller who writes the type has said what the column is, and a file that
// quotes its numbers is common. Inference never does that, so this only comes up
// for a column somebody named.
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
		return appendText(), nil
	case dtype.BinaryKind:
		return appendBinary(), nil
	default:
		return nil, fmt.Errorf("cannot read JSON into a %s column: %w",
			dt, ErrUnsupportedType)
	}
}

// appendBool reads a JSON boolean, or the text of one out of a quoted value.
func appendBool(b *array.Builder, d *jsontext.Decoder) (string, error) {
	switch k := d.PeekKind(); {
	case k == '"':
		s, text, err := quoted(d)
		if err != nil {
			return text, err
		}
		v, err := strconv.ParseBool(s)
		if err != nil {
			return text, cause(err)
		}
		b.AppendBool(v)
		return "", nil

	case k != 't' && k != 'f':
		return refuse(d, errNotBool)
	}

	tok, err := d.ReadToken()
	if err != nil {
		return "", err
	}
	b.AppendBool(tok.Bool())
	return "", nil
}

// appendSigned returns the appender for a signed integer column of the given
// width.
//
// A JSON number is a number of no width at all, so what fits is asked here
// rather than left to a conversion that would quietly wrap. A number with a
// point or an exponent in it is refused rather than rounded, which is what the
// CSV reader does with the same text and what a column of counts wants.
func appendSigned[T int8 | int16 | int32 | int64](bits int) appender {
	return func(b *array.Builder, d *jsontext.Decoder) (string, error) {
		if d.PeekKind() == '"' {
			s, text, err := quoted(d)
			if err != nil {
				return text, err
			}
			v, err := strconv.ParseInt(s, 10, bits)
			if err != nil {
				return text, cause(err)
			}
			b.Append(T(v))
			return "", nil
		}

		if d.PeekKind() != '0' {
			return refuse(d, errNotNumber)
		}
		tok, err := d.ReadToken()
		if err != nil {
			return "", err
		}
		v, err := tok.Int()
		if err != nil {
			return tok.String(), cause(err)
		}
		if int64(T(v)) != v {
			return tok.String(), strconv.ErrRange
		}
		b.Append(T(v))
		return "", nil
	}
}

// appendUnsigned returns the appender for an unsigned integer column.
func appendUnsigned[T uint8 | uint16 | uint32 | uint64](bits int) appender {
	return func(b *array.Builder, d *jsontext.Decoder) (string, error) {
		if d.PeekKind() == '"' {
			s, text, err := quoted(d)
			if err != nil {
				return text, err
			}
			v, err := strconv.ParseUint(s, 10, bits)
			if err != nil {
				return text, cause(err)
			}
			b.Append(T(v))
			return "", nil
		}

		if d.PeekKind() != '0' {
			return refuse(d, errNotNumber)
		}
		tok, err := d.ReadToken()
		if err != nil {
			return "", err
		}
		v, err := tok.Uint()
		if err != nil {
			return tok.String(), cause(err)
		}
		if uint64(T(v)) != v {
			return tok.String(), strconv.ErrRange
		}
		b.Append(T(v))
		return "", nil
	}
}

// appendFloat returns the appender for a float column.
//
// A float column takes any JSON number, since every one of them has a float that
// is either the value or the nearest there is to it. What it will not take is a
// value too large for the width of the column, which is only a float32 with a
// float64 in front of it, because a column of infinities is not what the file
// said.
func appendFloat[T float32 | float64](bits int) appender {
	return func(b *array.Builder, d *jsontext.Decoder) (string, error) {
		if d.PeekKind() == '"' {
			s, text, err := quoted(d)
			if err != nil {
				return text, err
			}
			v, err := strconv.ParseFloat(s, bits)
			if err != nil {
				return text, cause(err)
			}
			b.Append(T(v))
			return "", nil
		}

		if d.PeekKind() != '0' {
			return refuse(d, errNotNumber)
		}
		tok, err := d.ReadToken()
		if err != nil {
			return "", err
		}
		v, err := tok.Float()
		if err != nil {
			return tok.String(), cause(err)
		}
		if bits == 32 && (v < -math.MaxFloat32 || v > math.MaxFloat32) {
			return tok.String(), strconv.ErrRange
		}
		b.Append(T(v))
		return "", nil
	}
}

// appendText returns the appender for a string column, which is the column type
// that takes anything.
//
// A quoted value goes in as the text inside the quotes, and everything else goes
// in as the JSON it arrived as. That is what makes a string column the answer
// for the members that hold an object or an array and for the ones that hold two
// kinds of value on different lines: nothing is thrown away and the text is
// there to be picked apart later.
//
// Nothing here checks for UTF-8. The decoder underneath refuses bytes that are
// not, which is what the JSON specification says to do, so by this point the
// question has been asked and answered.
func appendText() appender {
	var buf []byte
	return func(b *array.Builder, d *jsontext.Decoder) (string, error) {
		v, err := d.ReadValue()
		if err != nil {
			return "", err
		}
		if v.Kind() != '"' {
			b.AppendBytes(v)
			return "", nil
		}

		// Nearly every string in a file has nothing in it to unescape, and one
		// that has not is what it says between the quotes. Checking is a pass
		// over bytes that are in cache; unquoting is a pass over them and a
		// copy into somewhere else.
		if bytes.IndexByte(v, '\\') < 0 {
			b.AppendBytes(v[1 : len(v)-1])
			return "", nil
		}
		if buf, err = jsontext.AppendUnquote(buf[:0], v); err != nil {
			return string(v), err
		}
		b.AppendBytes(buf)
		return "", nil
	}
}

// appendBinary returns the appender for a binary column, which reads the value
// as base64.
//
// That is the one convention JSON has for bytes and it is the one
// [encoding/json] follows for a []byte, so a column written by anything else
// arrives readable. A binary column is the way to hold bytes that are not text,
// and text in JSON is the only thing a quoted value can be.
func appendBinary() appender {
	var raw, buf []byte
	return func(b *array.Builder, d *jsontext.Decoder) (string, error) {
		v, err := d.ReadValue()
		if err != nil {
			return "", err
		}
		if v.Kind() != '"' {
			return string(v), errNotString
		}
		if raw, err = jsontext.AppendUnquote(raw[:0], v); err != nil {
			return string(v), err
		}
		if buf, err = base64.StdEncoding.AppendDecode(buf[:0], raw); err != nil {
			return string(v), err
		}
		b.AppendBytes(buf)
		return "", nil
	}
}

// refuse reads a value the column cannot hold and says what is wrong with it.
//
// The value is read rather than left where it is, so that a line with one bad
// value in the middle of it is still a line the reader can walk to the end of.
// That is what an object in a column of numbers takes: the object is one value
// and passing over it is passing over the whole of it.
func refuse(d *jsontext.Decoder, why error) (string, error) {
	v, err := d.ReadValue()
	if err != nil {
		return "", err
	}
	return string(v), why
}

// quoted reads a quoted value and returns the text inside it, along with the
// value as it appeared for the error that is about to be built.
//
// This is the permissive path, so it allocates where the rest of the reader does
// not. A file that quotes its numbers is a file that has already decided to be
// three times its own size, and the column it lands in was named by hand.
func quoted(d *jsontext.Decoder) (s, text string, err error) {
	v, err := d.ReadValue()
	if err != nil {
		return "", "", err
	}
	p, err := jsontext.AppendUnquote(nil, v)
	if err != nil {
		return "", string(v), err
	}
	return string(p), string(v), nil
}

// cause returns what a failed parse was really about, which is the syntax or
// range error inside it rather than the wrapper carrying the function name and
// the input. The input is already in the ValueError this ends up in.
func cause(err error) error {
	switch {
	case errors.Is(err, strconv.ErrRange):
		return strconv.ErrRange
	case errors.Is(err, strconv.ErrSyntax):
		return strconv.ErrSyntax
	default:
		return err
	}
}
