package ipc

import (
	"fmt"
	"strconv"

	"github.com/tamnd/kuma/dtype"
)

// Format returns the Arrow C data interface format string for t.
//
// The string describes t and nothing about the values, so a nullable column and
// a non-nullable one of the same type give the same string. Nullability is a
// flag on the schema struct rather than part of the type, the same way it is a
// flag on dtype.Field.
//
// The nested types give the format string of the container only: a list of
// int64 is "+l", and the int64 travels separately as the child. That is how the
// C structs are laid out, with the children in their own array, and it is why
// there is no format string anywhere that names two types at once.
//
// A dictionary is the one type whose format string is not its own. The C data
// interface puts the index type in the format string and the value type in the
// dictionary member of the schema, so Format reports the format of the index,
// and an exporter has to fill in the dictionary member itself. Passing the
// result back to Type gives the index type back rather than the dictionary,
// which is correct: the string never held the value type to begin with.
func Format(t dtype.DataType) (string, error) {
	if t == nil {
		return "", fmt.Errorf("ipc: %w: nil type", ErrType)
	}

	// The types with no parameters, and the nested types whose parameters are
	// all in the children, are a lookup on the kind.
	if k := t.Kind(); int(k) < len(plainFormats) && plainFormats[k] != "" {
		return plainFormats[k], nil
	}

	switch x := t.(type) {
	case dtype.FixedSizeBinary:
		if x.ByteWidth < 0 {
			return "", noFormat(x)
		}
		return "w:" + strconv.Itoa(int(x.ByteWidth)), nil

	case dtype.FixedSizeList:
		if x.Len < 0 {
			return "", noFormat(x)
		}
		return "+w:" + strconv.Itoa(int(x.Len)), nil

	case dtype.Time32:
		// Seconds and milliseconds are the only units that fit in 32 bits, so
		// the other two are a literal somebody wrote rather than a type.
		switch x.Unit {
		case dtype.Second:
			return "tts", nil
		case dtype.Millisecond:
			return "ttm", nil
		default:
			return "", noFormat(x)
		}

	case dtype.Time64:
		switch x.Unit {
		case dtype.Microsecond:
			return "ttu", nil
		case dtype.Nanosecond:
			return "ttn", nil
		default:
			return "", noFormat(x)
		}

	case dtype.Timestamp:
		u, ok := unitLetter(x.Unit)
		if !ok {
			return "", noFormat(x)
		}
		// The colon is always there. An empty zone means naive local time and
		// is written as "tsu:", which is a different type from a timestamp in
		// UTC and has to stay different on the wire.
		return "ts" + u + ":" + x.Zone, nil

	case dtype.Duration:
		u, ok := unitLetter(x.Unit)
		if !ok {
			return "", noFormat(x)
		}
		return "tD" + u, nil

	case dtype.Interval:
		switch x.Unit {
		case dtype.YearMonth:
			return "tiM", nil
		case dtype.DayTime:
			return "tiD", nil
		case dtype.MonthDayNano:
			return "tin", nil
		default:
			return "", noFormat(x)
		}

	case dtype.Decimal128:
		// The bit width is left off at 128 because that is the default and
		// every reader understands the short form. The long form is only
		// needed for the widths that are not the default.
		return decimalFormat(x.Precision, x.Scale, 0), nil

	case dtype.Decimal256:
		return decimalFormat(x.Precision, x.Scale, 256), nil

	case dtype.Dictionary:
		if !dtype.IsInteger(x.Index) {
			return "", noFormat(x)
		}
		return Format(x.Index)
	}

	return "", noFormat(t)
}

// noFormat is the error for a type this package cannot name. It prints the
// type, since the caller is holding a whole schema and needs to know which
// column of it stopped the export.
func noFormat(t dtype.DataType) error {
	return fmt.Errorf("ipc: %w: %s", ErrType, t)
}

// plainFormats is the format string of every type whose format string does not
// depend on a parameter. It is indexed by kind, and an empty string means the
// type is one of the parameterized ones handled above.
var plainFormats = [...]string{
	dtype.NullKind:    "n",
	dtype.BoolKind:    "b",
	dtype.Int8Kind:    "c",
	dtype.Int16Kind:   "s",
	dtype.Int32Kind:   "i",
	dtype.Int64Kind:   "l",
	dtype.Uint8Kind:   "C",
	dtype.Uint16Kind:  "S",
	dtype.Uint32Kind:  "I",
	dtype.Uint64Kind:  "L",
	dtype.Float32Kind: "f",
	dtype.Float64Kind: "g",

	// kuma stores text and bytes as views, which is what "vu" and "vz" name.
	// The two offset based layouts below can be described and are what arrives
	// from a producer that has not adopted views, most of them at the time of
	// writing.
	dtype.StringKind:      "vu",
	dtype.BinaryKind:      "vz",
	dtype.LargeStringKind: "U",
	dtype.LargeBinaryKind: "Z",

	dtype.Date32Kind: "tdD",
	dtype.Date64Kind: "tdm",

	dtype.ListKind:      "+l",
	dtype.LargeListKind: "+L",
	dtype.StructKind:    "+s",
	dtype.MapKind:       "+m",
}

// unitLetter returns the letter Arrow uses for a time unit in a format string,
// which is not the same as the name the unit prints under: microseconds are
// "us" in a type name and "u" here.
func unitLetter(u dtype.TimeUnit) (string, bool) {
	switch u {
	case dtype.Second:
		return "s", true
	case dtype.Millisecond:
		return "m", true
	case dtype.Microsecond:
		return "u", true
	case dtype.Nanosecond:
		return "n", true
	}
	return "", false
}

// decimalFormat writes "d:precision,scale" and appends the bit width when bits
// is not zero.
func decimalFormat(precision, scale int32, bits int) string {
	s := "d:" + strconv.Itoa(int(precision)) + "," + strconv.Itoa(int(scale))
	if bits != 0 {
		s += "," + strconv.Itoa(bits)
	}
	return s
}
