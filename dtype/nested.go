package dtype

import (
	"strconv"
	"strings"
)

// FixedSizeBinary is ByteWidth bytes per value with no offsets, which is what a
// UUID, a hash or a fixed length identifier should be. It is the only binary
// type that is fixed width.
type FixedSizeBinary struct {
	ByteWidth int32
}

// Kind returns FixedSizeBinaryKind.
func (t FixedSizeBinary) Kind() Kind { return FixedSizeBinaryKind }

// String returns the canonical name, such as "fixed_size_binary[16]".
func (t FixedSizeBinary) String() string {
	return "fixed_size_binary[" + strconv.Itoa(int(t.ByteWidth)) + "]"
}

// Bits returns the width of one value in bits, which is eight times ByteWidth.
func (t FixedSizeBinary) Bits() int { return int(t.ByteWidth) * 8 }

// List is a variable length sequence of Elem per row, stored as 32 bit offsets
// into one child array. Every element of every row lives in the same child, so
// a kernel that does not care about the row boundaries can run over the child
// directly.
type List struct {
	Elem DataType
}

// Kind returns ListKind.
func (t List) Kind() Kind { return ListKind }

// String returns the canonical name, such as "list<int64>".
func (t List) String() string { return "list<" + typeName(t.Elem) + ">" }

// LargeList is List with 64 bit offsets, for a child array with more than two
// billion elements in total.
type LargeList struct {
	Elem DataType
}

// Kind returns LargeListKind.
func (t LargeList) Kind() Kind { return LargeListKind }

// String returns the canonical name, such as "large_list<int64>".
func (t LargeList) String() string { return "large_list<" + typeName(t.Elem) + ">" }

// FixedSizeList is exactly Len elements of Elem per row, with no offsets. It is
// how a column of coordinates or a column of small embeddings should be stored,
// since the offsets in a List would all be multiples of the same number.
type FixedSizeList struct {
	Elem DataType
	Len  int32
}

// Kind returns FixedSizeListKind.
func (t FixedSizeList) Kind() Kind { return FixedSizeListKind }

// String returns the canonical name, such as "fixed_size_list<float32>[3]".
func (t FixedSizeList) String() string {
	return "fixed_size_list<" + typeName(t.Elem) + ">[" + strconv.Itoa(int(t.Len)) + "]"
}

// Struct is a fixed set of named fields per row, each stored as its own child
// array. A struct column of three fields is three columns that share one
// validity bitmap, so selecting one field costs nothing.
type Struct struct {
	Fields []Field
}

// Kind returns StructKind.
func (t Struct) Kind() Kind { return StructKind }

// String returns the canonical name, such as
// "struct<a: int64 not null, b: string>".
//
// Nullability is printed because it is part of what makes two struct types
// different, and leaving it out would mean two types that are not equal print
// the same way.
func (t Struct) String() string {
	var sb strings.Builder
	sb.WriteString("struct<")
	for i, f := range t.Fields {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(f.String())
	}
	sb.WriteString(">")
	return sb.String()
}

// Field returns the field with the given name and whether it was found.
func (t Struct) Field(name string) (Field, bool) {
	for _, f := range t.Fields {
		if f.Name == name {
			return f, true
		}
	}
	return Field{}, false
}

// Map is a variable number of key and value pairs per row.
//
// It is stored as a list of two field structs, which is worth knowing because
// it means the keys of every row are one contiguous array. Keys are not
// deduplicated across rows and nothing here enforces that they are unique
// within a row, since checking that on every read would cost more than the type
// is worth.
type Map struct {
	Key   DataType
	Value DataType
}

// Kind returns MapKind.
func (t Map) Kind() Kind { return MapKind }

// String returns the canonical name, such as "map<string, int64>".
func (t Map) String() string {
	return "map<" + typeName(t.Key) + ", " + typeName(t.Value) + ">"
}

// Dictionary stores each distinct value once and one Index per row pointing at
// it, which is what a low cardinality string column wants: a million rows of
// twenty country names become a million int32 values and twenty strings.
//
// This is the equivalent of a pandas categorical, except that the encoding is
// storage rather than semantics. A dictionary of strings compares, sorts and
// groups exactly like a column of strings. Ordered categories, where the
// comparison follows the dictionary order rather than the value order, are a
// separate thing and are not this.
//
// Index must be an integer type. Uint32 is the usual choice, and a smaller one
// is worth it when the cardinality is known to be small.
type Dictionary struct {
	Index DataType
	Value DataType
}

// Kind returns DictionaryKind.
func (t Dictionary) Kind() Kind { return DictionaryKind }

// String returns the canonical name, such as "dictionary<uint32, string>".
func (t Dictionary) String() string {
	return "dictionary<" + typeName(t.Index) + ", " + typeName(t.Value) + ">"
}

// typeName is String with a nil check, so that printing a half built type in an
// error message says which part is missing instead of panicking.
func typeName(t DataType) string {
	if t == nil {
		return "<nil>"
	}
	return t.String()
}
