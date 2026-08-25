// Package dtype describes the types a column can hold.
//
// A DataType says what the values in a column are and, for the parameterized
// types, what the parameters are. It says nothing about whether the values are
// present. Missing values are a validity bitmap on the array and a Nullable
// flag on the Field, deliberately not part of the type, because the
// alternative is pandas, where a column of integers turns into a column of
// floats the moment one value goes missing.
//
// Types with no parameters are package level values:
//
//	dtype.Int64
//	dtype.String
//
// Types with parameters are ordinary structs, written as composite literals:
//
//	dtype.Timestamp{Unit: dtype.Microsecond, Zone: "Europe/London"}
//	dtype.List{Elem: dtype.Int64}
//	dtype.Decimal128{Precision: 18, Scale: 2}
//
// There are no constructor functions and nothing here panics. A literal can be
// written that does not describe a real type, such as a Time32 in nanoseconds
// or a decimal with a scale larger than its precision, and Validate is what
// reports that. The frame layer validates a schema once, when it is built,
// rather than every operation checking for itself.
//
// Every type reports a Kind, which is the type with its parameters removed.
// Kind is what a kernel dispatch table keys on. It is not what two column
// types should be compared with, because a timestamp in microseconds and a
// timestamp in nanoseconds are the same Kind and are not interchangeable. Use
// Equal for that.
//
// Stability: tier 1, stable.
package dtype

import "fmt"

// Kind is a type with its parameters removed.
//
// The zero Kind is InvalidKind rather than a real type, so that a Kind that
// was never set does not silently read as null or as bool.
type Kind uint8

// The kinds. Every DataType reports exactly one of these.
const (
	InvalidKind Kind = iota
	NullKind
	BoolKind
	Int8Kind
	Int16Kind
	Int32Kind
	Int64Kind
	Uint8Kind
	Uint16Kind
	Uint32Kind
	Uint64Kind
	Float32Kind
	Float64Kind
	StringKind
	BinaryKind
	LargeStringKind
	LargeBinaryKind
	FixedSizeBinaryKind
	Date32Kind
	Date64Kind
	Time32Kind
	Time64Kind
	TimestampKind
	DurationKind
	IntervalKind
	Decimal128Kind
	Decimal256Kind
	ListKind
	LargeListKind
	FixedSizeListKind
	StructKind
	MapKind
	DictionaryKind
)

var kindNames = [...]string{
	InvalidKind:         "invalid",
	NullKind:            "null",
	BoolKind:            "bool",
	Int8Kind:            "int8",
	Int16Kind:           "int16",
	Int32Kind:           "int32",
	Int64Kind:           "int64",
	Uint8Kind:           "uint8",
	Uint16Kind:          "uint16",
	Uint32Kind:          "uint32",
	Uint64Kind:          "uint64",
	Float32Kind:         "float32",
	Float64Kind:         "float64",
	StringKind:          "string",
	BinaryKind:          "binary",
	LargeStringKind:     "large_string",
	LargeBinaryKind:     "large_binary",
	FixedSizeBinaryKind: "fixed_size_binary",
	Date32Kind:          "date32",
	Date64Kind:          "date64",
	Time32Kind:          "time32",
	Time64Kind:          "time64",
	TimestampKind:       "timestamp",
	DurationKind:        "duration",
	IntervalKind:        "interval",
	Decimal128Kind:      "decimal128",
	Decimal256Kind:      "decimal256",
	ListKind:            "list",
	LargeListKind:       "large_list",
	FixedSizeListKind:   "fixed_size_list",
	StructKind:          "struct",
	MapKind:             "map",
	DictionaryKind:      "dictionary",
}

// String returns the kind's name, which is the type's name with the parameters
// left off. The kind of a microsecond timestamp prints as "timestamp".
func (k Kind) String() string {
	if int(k) >= len(kindNames) {
		return fmt.Sprintf("Kind(%d)", uint8(k))
	}
	return kindNames[k]
}

// DataType is the type of the values in one column.
//
// The interface is two methods because it needs runtime polymorphism and
// nothing else. Anything that varies by type, meaning width, coercion,
// casting and every kernel, is a function or a table outside the interface,
// so that adding one does not force every implementation to change.
type DataType interface {
	// Kind returns the type with its parameters removed.
	Kind() Kind

	// String returns the canonical name of the type, parameters included. It
	// is unique: two types print the same if and only if Equal reports them
	// equal, with the one exception that metadata attached to the fields of a
	// struct is not printed. That is what makes the name usable in an error
	// message and in a test assertion.
	String() string
}

// FixedWidth is implemented by the types whose values all take the same number
// of bits, which is the property that lets a kernel index into a buffer by
// multiplication rather than by following offsets.
//
// Bool is fixed width at one bit, not one byte. Anything sizing a buffer has
// to handle that, and it is the reason this reports bits rather than bytes.
//
// The variable width string and binary types are not here even though their
// view structs are a fixed sixteen bytes, because the bytes those views point
// at are not.
type FixedWidth interface {
	DataType

	// Bits returns the width of one value in bits.
	Bits() int
}

// fixed is every parameterless type whose values have a fixed width. There is
// exactly one value per kind, all of them package level, so two of them are
// the same type if and only if they are the same pointer.
type fixed struct {
	kind Kind
	name string
	bits int
}

// Kind returns the kind this type was built with.
func (t *fixed) Kind() Kind { return t.kind }

// String returns the type's name.
func (t *fixed) String() string { return t.name }

// Bits returns the width of one value in bits.
func (t *fixed) Bits() int { return t.bits }

// variable is every parameterless type whose values do not have a fixed width,
// plus null, which has no values at all.
type variable struct {
	kind Kind
	name string
}

// Kind returns the kind this type was built with.
func (t *variable) Kind() Kind { return t.kind }

// String returns the type's name.
func (t *variable) String() string { return t.name }

// The parameterless types. These are pointers to values with unexported
// fields, so the types themselves cannot be modified. Do not reassign the
// variables.
var (
	// Null is a column of nothing. Every value is missing and no storage is
	// allocated. It exists because a literal nil and an all missing column
	// read from a file both have to have a type, and because it is what makes
	// Coerce able to combine a missing value with anything.
	Null DataType = &variable{NullKind, "null"}

	// Bool is one bit per value, packed the same way a validity bitmap is.
	Bool DataType = &fixed{BoolKind, "bool", 1}

	Int8  DataType = &fixed{Int8Kind, "int8", 8}
	Int16 DataType = &fixed{Int16Kind, "int16", 16}
	Int32 DataType = &fixed{Int32Kind, "int32", 32}
	Int64 DataType = &fixed{Int64Kind, "int64", 64}

	Uint8  DataType = &fixed{Uint8Kind, "uint8", 8}
	Uint16 DataType = &fixed{Uint16Kind, "uint16", 16}
	Uint32 DataType = &fixed{Uint32Kind, "uint32", 32}
	Uint64 DataType = &fixed{Uint64Kind, "uint64", 64}

	Float32 DataType = &fixed{Float32Kind, "float32", 32}
	Float64 DataType = &fixed{Float64Kind, "float64", 64}

	// String and Binary use the Arrow variable size binary view layout, which
	// is the reasoning in document 02: a sixteen byte view per value holding
	// the length, a four byte inline prefix, and then either the rest of the
	// value inline for twelve bytes or fewer or a buffer index and offset for
	// longer ones.
	String DataType = &variable{StringKind, "string"}
	Binary DataType = &variable{BinaryKind, "binary"}

	// LargeString and LargeBinary are the classic offsets and data layout with
	// 64 bit offsets. They are kept for interoperability, since that is what
	// arrives over Arrow IPC, and converted at the boundary.
	LargeString DataType = &variable{LargeStringKind, "large_string"}
	LargeBinary DataType = &variable{LargeBinaryKind, "large_binary"}

	// Date32 is days since the Unix epoch and Date64 is milliseconds since it,
	// constrained to exact multiples of a day. Date32 is the one to use.
	Date32 DataType = &fixed{Date32Kind, "date32", 32}
	Date64 DataType = &fixed{Date64Kind, "date64", 64}
)

// Equal reports whether a and b are the same type, parameters included. Two
// nil types are equal and a nil is equal to nothing else.
//
// A timestamp in microseconds and a timestamp in nanoseconds are not equal even
// though they are the same Kind, and a list of int64 is not equal to a list of
// int32. This is the comparison to use before deciding that two columns can be
// concatenated or joined.
func Equal(a, b DataType) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	if a.Kind() != b.Kind() {
		return false
	}

	// The types that hold other types are compared field by field. They have to
	// be, because a struct type holds a slice, so the language would refuse to
	// compare it, and because the children need this same treatment one level
	// down.
	switch x := a.(type) {
	case List:
		y, ok := b.(List)
		return ok && Equal(x.Elem, y.Elem)
	case LargeList:
		y, ok := b.(LargeList)
		return ok && Equal(x.Elem, y.Elem)
	case FixedSizeList:
		y, ok := b.(FixedSizeList)
		return ok && x.Len == y.Len && Equal(x.Elem, y.Elem)
	case Struct:
		y, ok := b.(Struct)
		if !ok || len(x.Fields) != len(y.Fields) {
			return false
		}
		for i := range x.Fields {
			if !x.Fields[i].Equal(y.Fields[i]) {
				return false
			}
		}
		return true
	case Map:
		y, ok := b.(Map)
		return ok && Equal(x.Key, y.Key) && Equal(x.Value, y.Value)
	case Dictionary:
		y, ok := b.(Dictionary)
		return ok && Equal(x.Index, y.Index) && Equal(x.Value, y.Value)
	}

	// Everything else is a comparable struct or one of the package level
	// singletons, so the language already knows how to compare it.
	return a == b
}

// Bits returns the width of one value in bits and whether t has one. It is
// the function form of the FixedWidth interface, for the common case where the
// caller wants a number or a false rather than a type assertion.
func Bits(t DataType) (int, bool) {
	fw, ok := t.(FixedWidth)
	if !ok {
		return 0, false
	}
	return fw.Bits(), true
}

// IsSigned reports whether t is one of the signed integer types.
func IsSigned(t DataType) bool {
	switch kindOf(t) {
	case Int8Kind, Int16Kind, Int32Kind, Int64Kind:
		return true
	default:
		return false
	}
}

// IsUnsigned reports whether t is one of the unsigned integer types.
func IsUnsigned(t DataType) bool {
	switch kindOf(t) {
	case Uint8Kind, Uint16Kind, Uint32Kind, Uint64Kind:
		return true
	default:
		return false
	}
}

// IsInteger reports whether t is a signed or unsigned integer type. Bool is
// not an integer here, which is a deliberate difference from pandas, where a
// boolean column sums to an integer without anyone asking for it.
func IsInteger(t DataType) bool { return IsSigned(t) || IsUnsigned(t) }

// IsFloat reports whether t is one of the floating point types.
func IsFloat(t DataType) bool {
	k := kindOf(t)
	return k == Float32Kind || k == Float64Kind
}

// IsDecimal reports whether t is one of the fixed point decimal types.
func IsDecimal(t DataType) bool {
	k := kindOf(t)
	return k == Decimal128Kind || k == Decimal256Kind
}

// IsNumeric reports whether t is an integer, a float or a decimal.
func IsNumeric(t DataType) bool { return IsInteger(t) || IsFloat(t) || IsDecimal(t) }

// IsTemporal reports whether t is a date, a time, a timestamp, a duration or
// an interval.
func IsTemporal(t DataType) bool {
	switch kindOf(t) {
	case Date32Kind, Date64Kind, Time32Kind, Time64Kind,
		TimestampKind, DurationKind, IntervalKind:
		return true
	default:
		return false
	}
}

// IsString reports whether t holds text, meaning String or LargeString.
func IsString(t DataType) bool {
	k := kindOf(t)
	return k == StringKind || k == LargeStringKind
}

// IsBinary reports whether t holds opaque bytes, meaning Binary, LargeBinary
// or FixedSizeBinary.
func IsBinary(t DataType) bool {
	switch kindOf(t) {
	case BinaryKind, LargeBinaryKind, FixedSizeBinaryKind:
		return true
	default:
		return false
	}
}

// IsNested reports whether t has child types, meaning the value of one row is
// itself made of other values.
func IsNested(t DataType) bool {
	switch kindOf(t) {
	case ListKind, LargeListKind, FixedSizeListKind, StructKind, MapKind:
		return true
	default:
		return false
	}
}

// kindOf is Kind with a nil check, so that every predicate above can be called
// on a type that has not been set yet without the caller checking first.
func kindOf(t DataType) Kind {
	if t == nil {
		return InvalidKind
	}
	return t.Kind()
}
