package kernel

import (
	"errors"
	"strconv"
	"unicode/utf8"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// toStored returns the converter for a cast whose destination is a number, or
// something stored as one.
func toStored(from, to dtype.DataType) converse {
	if dtype.IsString(from) {
		return parseNumber(to.Kind())
	}

	k := to.Kind()
	return func(b *array.Builder, a *array.Array, i int) *CastError {
		n := readNumber(a, i)
		if !appendNumber(b, k, n) {
			return &CastError{Value: n.String(), Err: strconv.ErrRange}
		}
		return nil
	}
}

// parseNumber returns the converter that reads text as a number of kind k.
//
// The text is parsed as the type that was asked for rather than as a number in
// general, so "3.9" is not an int64 and "1e3" is not one either. A caller who
// writes int64 has said what they expect the column to hold, and turning a
// number that is not one of those into one that is would be answering a
// different question.
func parseNumber(k dtype.Kind) converse {
	bits := numberBits(k)
	return func(b *array.Builder, a *array.Array, i int) *CastError {
		s := string(a.Bytes(i))

		var n number
		var err error
		switch {
		case k == dtype.Float32Kind || k == dtype.Float64Kind:
			n.k = numFloat
			n.f, err = strconv.ParseFloat(s, bits)
		case isUnsignedKind(k):
			n.k = numUint
			n.u, err = strconv.ParseUint(s, 10, bits)
		default:
			n.i, err = strconv.ParseInt(s, 10, bits)
		}
		if err != nil {
			return &CastError{Value: strconv.Quote(s), Err: cause(err)}
		}

		if !appendNumber(b, k, n) {
			// strconv was told the width it was parsing for, so a value it
			// accepted is a value that fits.
			panic("kernel: " + s + " parsed as a " + k.String() + " and then did not fit")
		}
		return nil
	}
}

// toBool returns the converter for a cast whose destination is a boolean.
func toBool(from dtype.DataType) converse {
	if dtype.IsString(from) {
		return func(b *array.Builder, a *array.Array, i int) *CastError {
			s := string(a.Bytes(i))
			v, err := strconv.ParseBool(s)
			if err != nil {
				return &CastError{Value: strconv.Quote(s), Err: cause(err)}
			}
			b.AppendBool(v)
			return nil
		}
	}

	// Zero is false and everything else is true, which is C, Python, SQL and
	// every spreadsheet. A NaN is not zero and so is true, which looks odd
	// written down and is what every one of those does as well.
	return func(b *array.Builder, a *array.Array, i int) *CastError {
		n := readNumber(a, i)
		switch n.k {
		case numInt:
			b.AppendBool(n.i != 0)
		case numUint:
			b.AppendBool(n.u != 0)
		default:
			b.AppendBool(n.f != 0)
		}
		return nil
	}
}

// toText returns the converter for a cast whose destination holds bytes.
func toText(from, to dtype.DataType) converse {
	if dtype.IsString(from) || dtype.IsBinary(from) {
		return copyBytes(from, to)
	}
	if from.Kind() == dtype.BoolKind {
		return func(b *array.Builder, a *array.Array, i int) *CastError {
			b.AppendString(strconv.FormatBool(a.Bool(i)))
			return nil
		}
	}

	// A float prints with the fewest digits that read back as the same value,
	// which is what a caller writing a column out to a file wants and what
	// reading it back in returns unchanged. The float32 case says 32 so that a
	// value that only ever had seven digits does not print twenty.
	bits := numberBits(from.Kind())

	// The scratch buffer is the difference between one allocation for the
	// column and one per row. AppendBytes copies what it is given, so nothing
	// downstream holds on to this, and a converter is used by one cast on one
	// goroutine, which is what makes keeping state in the closure safe.
	scratch := make([]byte, 0, 32)

	return func(b *array.Builder, a *array.Array, i int) *CastError {
		n := readNumber(a, i)
		switch n.k {
		case numInt:
			scratch = strconv.AppendInt(scratch[:0], n.i, 10)
		case numUint:
			scratch = strconv.AppendUint(scratch[:0], n.u, 10)
		default:
			scratch = strconv.AppendFloat(scratch[:0], n.f, 'g', -1, bits)
		}
		b.AppendBytes(scratch)
		return nil
	}
}

// copyBytes returns the converter for bytes to bytes, which is text to bytes,
// bytes to text, and either of those into or out of a fixed width column.
//
// Nothing is converted. What happens instead is that the destination gets to
// object: text has to be valid UTF-8 and a fixed width column has to be handed
// exactly its width. Both of those are per value, so both are things a row can
// fail at rather than things the pair of types decides in advance.
func copyBytes(from, to dtype.DataType) converse {
	checkUTF8 := dtype.IsString(to) && !dtype.IsString(from)

	width := 0
	if f, ok := to.(dtype.FixedSizeBinary); ok {
		width = int(f.ByteWidth)
	}

	return func(b *array.Builder, a *array.Array, i int) *CastError {
		p := a.Bytes(i)
		if checkUTF8 && !utf8.Valid(p) {
			return &CastError{Value: strconv.Quote(string(p)), Err: errNotUTF8}
		}
		if width > 0 && len(p) != width {
			return &CastError{
				Value: strconv.Quote(string(p)),
				Err:   errors.New("it is " + strconv.Itoa(len(p)) + " bytes"),
			}
		}
		b.AppendBytes(p)
		return nil
	}
}

var errNotUTF8 = errors.New("it is not valid UTF-8")

// numberBits returns the width in bits of a numeric kind, which is what the
// strconv functions want.
func numberBits(k dtype.Kind) int {
	switch k {
	case dtype.Int8Kind, dtype.Uint8Kind:
		return 8
	case dtype.Int16Kind, dtype.Uint16Kind:
		return 16
	case dtype.Int32Kind, dtype.Uint32Kind, dtype.Float32Kind:
		return 32
	default:
		return 64
	}
}

// isUnsignedKind reports whether a kind counts from zero up.
func isUnsignedKind(k dtype.Kind) bool {
	switch k {
	case dtype.Uint8Kind, dtype.Uint16Kind, dtype.Uint32Kind, dtype.Uint64Kind:
		return true
	default:
		return false
	}
}

// cause returns what strconv was really complaining about, since the wrapper it
// returns repeats the function name and the value that the error message here
// is about to print anyway.
//
// Everything passed to this came from strconv, which always returns a
// *strconv.NumError around one of ErrSyntax and ErrRange, so the unwrap always
// finds the reason.
func cause(err error) error { return errors.Unwrap(err) }
