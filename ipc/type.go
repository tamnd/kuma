package ipc

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/tamnd/kuma/dtype"
)

// Type returns the kuma type a format string names.
//
// The children are the types of the members the format string does not name,
// already converted: one field for a list, one field per member for a struct,
// and for a map one field whose type is a two field struct of the key and the
// value. Everything else takes no children and ignores them. An importer walks
// the C structs from the leaves up, so by the time it asks about a list it
// already has the element type in hand, which is why this takes the children
// rather than trying to parse a tree out of a string that cannot describe one.
//
// The result is validated, so a producer that writes a decimal with a precision
// of ninety nine is rejected here rather than three operations later.
//
// Text and bytes are the one place where this is not the inverse of Format. The
// C data interface has three layouts for each, meaning 32 bit offsets, 64 bit
// offsets and views, and kuma stores exactly one of them. An import materializes
// whatever arrived into the view layout, so "u", "U" and "vu" all become
// dtype.String and "z", "Z" and "vz" all become dtype.Binary. A caller that
// needs to know which layout the incoming buffers are in has the format string
// in front of it and should read that rather than the type.
//
// A dictionary encoded column does not arrive as a format string. The format
// names the index type and the value type hangs off the dictionary member of
// the schema, so an importer calls Type twice and builds the dtype.Dictionary
// itself.
func Type(format string, children []dtype.Field) (dtype.DataType, error) {
	t, err := parseFormat(format, children)
	if err != nil {
		return nil, err
	}
	if err := dtype.Validate(t); err != nil {
		return nil, fmt.Errorf("ipc: format %q: %w", format, err)
	}
	return t, nil
}

func parseFormat(format string, children []dtype.Field) (dtype.DataType, error) {
	switch format {
	case "n":
		return dtype.Null, nil
	case "b":
		return dtype.Bool, nil
	case "c":
		return dtype.Int8, nil
	case "C":
		return dtype.Uint8, nil
	case "s":
		return dtype.Int16, nil
	case "S":
		return dtype.Uint16, nil
	case "i":
		return dtype.Int32, nil
	case "I":
		return dtype.Uint32, nil
	case "l":
		return dtype.Int64, nil
	case "L":
		return dtype.Uint64, nil
	case "f":
		return dtype.Float32, nil
	case "g":
		return dtype.Float64, nil

	// Every text layout is one kuma type and every byte layout is another. See
	// the note on Type for why.
	case "u", "U", "vu":
		return dtype.String, nil
	case "z", "Z", "vz":
		return dtype.Binary, nil

	case "tdD":
		return dtype.Date32, nil
	case "tdm":
		return dtype.Date64, nil

	case "tts":
		return dtype.Time32{Unit: dtype.Second}, nil
	case "ttm":
		return dtype.Time32{Unit: dtype.Millisecond}, nil
	case "ttu":
		return dtype.Time64{Unit: dtype.Microsecond}, nil
	case "ttn":
		return dtype.Time64{Unit: dtype.Nanosecond}, nil

	case "tDs":
		return dtype.Duration{Unit: dtype.Second}, nil
	case "tDm":
		return dtype.Duration{Unit: dtype.Millisecond}, nil
	case "tDu":
		return dtype.Duration{Unit: dtype.Microsecond}, nil
	case "tDn":
		return dtype.Duration{Unit: dtype.Nanosecond}, nil

	case "tiM":
		return dtype.Interval{Unit: dtype.YearMonth}, nil
	case "tiD":
		return dtype.Interval{Unit: dtype.DayTime}, nil
	case "tin":
		return dtype.Interval{Unit: dtype.MonthDayNano}, nil

	case "+l", "+L":
		elem, err := onlyChild(format, children)
		if err != nil {
			return nil, err
		}
		if format == "+L" {
			return dtype.LargeList{Elem: elem.Type}, nil
		}
		return dtype.List{Elem: elem.Type}, nil

	case "+s":
		// The children are the members, in order, with their names and their
		// nullability, which is everything a struct type is.
		return dtype.Struct{Fields: slices.Clone(children)}, nil

	case "+m":
		return mapType(format, children)

	// Arrow types kuma has no equivalent for. These say so by name, because
	// "bad format string" on a column of unions sends the reader looking for a
	// typo that is not there.
	case "e":
		return nil, fmt.Errorf("ipc: %w: float16 has no kuma type", ErrFormat)
	case "+r":
		return nil, fmt.Errorf("ipc: %w: run end encoding has no kuma type", ErrFormat)
	case "+vl", "+vL":
		return nil, fmt.Errorf("ipc: %w: list views have no kuma type", ErrFormat)
	}

	switch {
	case strings.HasPrefix(format, "w:"):
		width, err := parseInt32(format, format[len("w:"):])
		if err != nil {
			return nil, err
		}
		return dtype.FixedSizeBinary{ByteWidth: width}, nil

	case strings.HasPrefix(format, "+w:"):
		size, err := parseInt32(format, format[len("+w:"):])
		if err != nil {
			return nil, err
		}
		elem, err := onlyChild(format, children)
		if err != nil {
			return nil, err
		}
		return dtype.FixedSizeList{Elem: elem.Type, Len: size}, nil

	case strings.HasPrefix(format, "d:"):
		return decimalType(format)

	case strings.HasPrefix(format, "ts"):
		return timestampType(format)

	case strings.HasPrefix(format, "+ud:"), strings.HasPrefix(format, "+us:"):
		return nil, fmt.Errorf("ipc: %w: unions have no kuma type", ErrFormat)
	}

	return nil, fmt.Errorf("ipc: %w: %q", ErrFormat, format)
}

// timestampType parses "ts" followed by the unit letter, a colon and the zone.
// The zone runs to the end of the string rather than to the next colon, since a
// fixed offset zone is written "+01:00" and has one in it.
func timestampType(format string) (dtype.DataType, error) {
	rest, zone, ok := strings.Cut(format[len("ts"):], ":")
	if !ok || len(rest) != 1 {
		return nil, fmt.Errorf("ipc: %w: %q, want ts<s|m|u|n>:<zone>", ErrFormat, format)
	}
	unit, ok := letterUnit(rest)
	if !ok {
		return nil, fmt.Errorf("ipc: %w: %q has unknown time unit %q", ErrFormat, format, rest)
	}
	return dtype.Timestamp{Unit: unit, Zone: zone}, nil
}

// decimalType parses "d:precision,scale" and "d:precision,scale,bits". The bit
// width is 128 when it is left out, which is the only width most producers ever
// write.
func decimalType(format string) (dtype.DataType, error) {
	parts := strings.Split(format[len("d:"):], ",")
	if len(parts) != 2 && len(parts) != 3 {
		return nil, fmt.Errorf("ipc: %w: %q, want d:<precision>,<scale>[,<bits>]",
			ErrFormat, format)
	}
	precision, err := parseInt32(format, parts[0])
	if err != nil {
		return nil, err
	}
	scale, err := parseInt32(format, parts[1])
	if err != nil {
		return nil, err
	}

	bits := "128"
	if len(parts) == 3 {
		bits = parts[2]
	}
	switch bits {
	case "128":
		return dtype.Decimal128{Precision: precision, Scale: scale}, nil
	case "256":
		return dtype.Decimal256{Precision: precision, Scale: scale}, nil
	}
	return nil, fmt.Errorf("ipc: %w: %q has decimal width %q, want 128 or 256",
		ErrFormat, format, bits)
}

// mapType turns the one child of "+m" into a key and a value. The child is a
// struct of two fields, which is how a map is stored: a list of entries, each
// entry a key and a value side by side.
func mapType(format string, children []dtype.Field) (dtype.DataType, error) {
	entries, err := onlyChild(format, children)
	if err != nil {
		return nil, err
	}
	s, ok := entries.Type.(dtype.Struct)
	if !ok || len(s.Fields) != 2 {
		return nil, fmt.Errorf("ipc: %w: the child of %q must be a struct of a key and a value, have %v",
			ErrChildren, format, entries.Type)
	}
	return dtype.Map{Key: s.Fields[0].Type, Value: s.Fields[1].Type}, nil
}

func onlyChild(format string, children []dtype.Field) (dtype.Field, error) {
	if len(children) != 1 {
		return dtype.Field{}, fmt.Errorf("ipc: %w: %q takes one child, have %d",
			ErrChildren, format, len(children))
	}
	return children[0], nil
}

// parseInt32 reads one of the numbers inside a format string. Nothing in the C
// data interface is wider than an int32, and a width or a precision that does
// not fit is a producer writing something else entirely.
func parseInt32(format, s string) (int32, error) {
	n, err := strconv.ParseInt(s, 10, 32)
	if err != nil {
		return 0, fmt.Errorf("ipc: %w: %q has bad number %q", ErrFormat, format, s)
	}
	return int32(n), nil
}

// letterUnit is the inverse of unitLetter.
func letterUnit(s string) (dtype.TimeUnit, bool) {
	switch s {
	case "s":
		return dtype.Second, true
	case "m":
		return dtype.Millisecond, true
	case "u":
		return dtype.Microsecond, true
	case "n":
		return dtype.Nanosecond, true
	}
	return 0, false
}
