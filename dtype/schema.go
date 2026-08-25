package dtype

import (
	"fmt"
	"strings"
)

// KeyValue is one piece of metadata.
type KeyValue struct {
	Key   string
	Value string
}

// Metadata is a list of key and value pairs carried alongside a field or a
// schema.
//
// It is a slice rather than a map because it round trips through Arrow IPC and
// Parquet, both of which specify an ordered list, and because a map would
// reorder the pairs on every write and make two identical schemas produce two
// different files. Duplicate keys are not rejected, since the formats allow
// them, and Get returns the first.
//
// Nothing in kuma reads metadata. It is here so that a value written by another
// tool survives a round trip through a kuma program, and so that a caller can
// attach a unit or a description that their own code understands.
type Metadata []KeyValue

// Get returns the value for the first pair with the given key, and whether
// there was one.
func (m Metadata) Get(key string) (string, bool) {
	for _, kv := range m {
		if kv.Key == key {
			return kv.Value, true
		}
	}
	return "", false
}

// Equal reports whether m and other hold the same pairs in the same order.
func (m Metadata) Equal(other Metadata) bool {
	if len(m) != len(other) {
		return false
	}
	for i := range m {
		if m[i] != other[i] {
			return false
		}
	}
	return true
}

// Clone returns a copy that shares no storage with m.
func (m Metadata) Clone() Metadata {
	if m == nil {
		return nil
	}
	return append(Metadata(nil), m...)
}

// Field is one named column in a schema, or one named member of a struct type.
//
// Nullable is on the field rather than on the type. That is the split that
// keeps int64 meaning int64 whether or not a value happens to be missing, which
// is the thing pandas got wrong and spent a decade adding nullable dtypes to
// work around. A non-nullable field is a promise about the data, and the
// builders and readers check it.
type Field struct {
	Name     string
	Type     DataType
	Nullable bool
	Metadata Metadata
}

// String returns the field as it appears inside a printed schema, such as
// "price: float64 not null".
func (f Field) String() string {
	s := f.Name + ": " + typeName(f.Type)
	if !f.Nullable {
		s += " not null"
	}
	return s
}

// Equal reports whether f and other have the same name, type, nullability and
// metadata.
func (f Field) Equal(other Field) bool {
	return f.Name == other.Name &&
		f.Nullable == other.Nullable &&
		Equal(f.Type, other.Type) &&
		f.Metadata.Equal(other.Metadata)
}

// Schema is the ordered list of fields in a frame.
//
// The order is part of the schema. Two schemas with the same fields in a
// different order are different schemas, because column order is what a
// positional read of a CSV or a Parquet row group depends on, and because
// printing a frame has to put the columns somewhere.
//
// Field names are not required to be unique by the type itself, since a CSV
// with two columns called "id" is a real thing that has to be readable. Validate
// is what rejects duplicates, and the frame layer calls it.
type Schema struct {
	Fields   []Field
	Metadata Metadata
}

// Len returns the number of fields.
func (s Schema) Len() int { return len(s.Fields) }

// Index returns the position of the first field with the given name, or -1 if
// there is none.
func (s Schema) Index(name string) int {
	for i, f := range s.Fields {
		if f.Name == name {
			return i
		}
	}
	return -1
}

// Field returns the first field with the given name and whether there was one.
func (s Schema) Field(name string) (Field, bool) {
	if i := s.Index(name); i >= 0 {
		return s.Fields[i], true
	}
	return Field{}, false
}

// Names returns the field names in order.
func (s Schema) Names() []string {
	names := make([]string, len(s.Fields))
	for i, f := range s.Fields {
		names[i] = f.Name
	}
	return names
}

// Select returns a schema holding the named fields in the order given. It
// reports an error naming the first field that is not in s, along with the
// names that are, since the usual cause is a typo or a stale column name.
//
// Metadata on the schema is carried over. Selecting a subset of the columns
// does not change what the table as a whole is.
func (s Schema) Select(names ...string) (Schema, error) {
	out := Schema{Fields: make([]Field, 0, len(names)), Metadata: s.Metadata}
	for _, name := range names {
		i := s.Index(name)
		if i < 0 {
			return Schema{}, fmt.Errorf("dtype: no field named %q, have %s",
				name, strings.Join(s.Names(), ", "))
		}
		out.Fields = append(out.Fields, s.Fields[i])
	}
	return out, nil
}

// Equal reports whether s and other have the same fields in the same order with
// the same metadata.
func (s Schema) Equal(other Schema) bool {
	if len(s.Fields) != len(other.Fields) {
		return false
	}
	for i := range s.Fields {
		if !s.Fields[i].Equal(other.Fields[i]) {
			return false
		}
	}
	return s.Metadata.Equal(other.Metadata)
}

// String returns the schema on one line, such as
// "schema<id: int64 not null, name: string>".
func (s Schema) String() string {
	var sb strings.Builder
	sb.WriteString("schema<")
	for i, f := range s.Fields {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(f.String())
	}
	sb.WriteString(">")
	return sb.String()
}

// Clone returns a copy that shares no slice storage with s. The types
// themselves are not copied, since they are immutable.
func (s Schema) Clone() Schema {
	out := Schema{Metadata: s.Metadata.Clone()}
	if s.Fields != nil {
		out.Fields = make([]Field, len(s.Fields))
		copy(out.Fields, s.Fields)
		for i := range out.Fields {
			out.Fields[i].Metadata = out.Fields[i].Metadata.Clone()
		}
	}
	return out
}
