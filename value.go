package kuma

import (
	"time"

	"github.com/tamnd/kuma/dtype"
)

// Value is the set of Go types a Series can be read as.
//
// The types are exact rather than approximate, so a named type such as
// type Price float64 is not a Value. That is deliberate: the mapping from a Go
// type to a column type is a switch on the type itself, and a named type would
// fall off the end of it. A column of prices is a Series[float64] whose field
// in your struct is a Price, and the conversion happens where the struct is
// read rather than inside the column.
type Value interface {
	bool |
		int8 | int16 | int32 | int64 |
		uint8 | uint16 | uint32 | uint64 |
		float32 | float64 |
		string | time.Time
}

// DTypeOf returns the column type that values of type T are stored as.
//
// A string is a String column, meaning the Arrow view layout, not the 64 bit
// offset layout. A time.Time is a Timestamp in nanoseconds, which is the unit
// a time.Time holds and so the only one that loses nothing.
//
// This is the type a new column gets. It is not the only type a T can be read
// out of: an int64 reads a timestamp or a duration column as well, since those
// are int64 values with a meaning attached, and CanRead is what says so.
func DTypeOf[T Value]() dtype.DataType {
	var zero T
	switch any(zero).(type) {
	case bool:
		return dtype.Bool
	case int8:
		return dtype.Int8
	case int16:
		return dtype.Int16
	case int32:
		return dtype.Int32
	case int64:
		return dtype.Int64
	case uint8:
		return dtype.Uint8
	case uint16:
		return dtype.Uint16
	case uint32:
		return dtype.Uint32
	case uint64:
		return dtype.Uint64
	case float32:
		return dtype.Float32
	case float64:
		return dtype.Float64
	case string:
		return dtype.String
	case time.Time:
		return dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "UTC"}
	default:
		// Value lists every case above, so this is unreachable unless a type is
		// added to the constraint and not to the switch.
		panic("kuma: no column type for this Go type")
	}
}

// CanRead reports whether a column of type dt can be read as a T.
//
// It is wider than DTypeOf in the places where two column types are the same
// values with a different meaning. A timestamp, a duration, a time of day and a
// date64 are all int64 columns, so all of them read as an int64, and a kernel
// that adds a number of days to a date is an integer kernel that should not
// have to copy the column first. What is refused is a reinterpretation that
// changes the width or the meaning of the bits, so a float64 column does not
// read as an int64.
//
// A time.Time reads a timestamp column of any unit and any zone.
func CanRead[T Value](dt dtype.DataType) bool {
	if dt == nil {
		return false
	}

	var zero T
	if _, ok := any(zero).(time.Time); ok {
		return dt.Kind() == dtype.TimestampKind
	}

	want := DTypeOf[T]()
	if dtype.Equal(want, dt) {
		return true
	}
	switch want.Kind() {
	case dtype.Int32Kind:
		return dt.Kind() == dtype.Date32Kind || dt.Kind() == dtype.Time32Kind
	case dtype.Int64Kind:
		switch dt.Kind() {
		case dtype.Date64Kind, dtype.Time64Kind, dtype.TimestampKind, dtype.DurationKind:
			return true
		default:
			return false
		}
	case dtype.StringKind:
		return dt.Kind() == dtype.BinaryKind
	default:
		return false
	}
}
