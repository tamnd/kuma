package parquet

import (
	"fmt"
	"slices"
	"strings"

	"github.com/tamnd/kuma/dtype"
)

// The schema, and what it means in kuma's types.
//
// Parquet writes its schema as a tree flattened into a list: the root, then its
// children, then theirs, with every node saying how many of the nodes after it
// belong to it. The leaves of that tree are the columns the file stores, and
// the groups above them are the structs, lists and maps those columns add up
// to. Tree puts the shape back, Columns is the leaves with the levels a decoder
// needs, and Schema is the whole thing in kuma's types.
//
// The mapping is not one to one in either direction. A parquet column is a
// physical type, which is one of eight ways to write bytes down, plus an
// annotation saying what those bytes mean. The annotation has been written two
// ways over the format's life: the converted type came first and had nowhere to
// put parameters, and the logical type replaced it and can carry them. Files in
// the wild have one, the other or both, so both are read here and the logical
// type wins where they disagree, which is what every other reader does.
//
// An annotation that does not go with the physical type under it is refused
// rather than ignored. A file that says a column is a four byte integer holding
// a string is a file that was written wrong, and reading it as an int32 would
// hand back a column of numbers that were meant to be words.

// maxSchemaDepth is how deeply the schema tree may nest.
//
// The format does not say, and a schema anybody wrote by hand is a handful of
// levels deep. The limit is here because the tree is rebuilt from child counts
// the file supplies, and a file claiming to nest a million deep would otherwise
// be a stack overflow rather than an error.
const maxSchemaDepth = 64

// Node is one node of the schema tree.
//
// A node with no children is a column and carries a physical type. A node with
// children is a group and carries none, and what it means depends on how it is
// annotated: a list, a map, or a struct when it is not annotated at all.
type Node struct {
	SchemaElement

	// Children are the fields of a group, in the order the file wrote them,
	// which is the order their columns appear in every row group.
	Children []Node
}

// Leaf reports whether n is a column rather than a group.
func (n *Node) Leaf() bool { return len(n.Children) == 0 }

// Tree returns the schema as a tree, starting at the root.
//
// The root is the file itself and its name is whatever the writer called the
// record type, usually schema or spark_schema. Its children are the columns of
// the file as somebody querying it would name them.
func (m *Metadata) Tree() (Node, error) {
	if len(m.Nodes) == 0 {
		return Node{}, fmt.Errorf("parquet: %w: the file has no schema", ErrFormat)
	}
	root, rest, err := readNode(m.Nodes, 0)
	if err != nil {
		return Node{}, err
	}
	if len(rest) != 0 {
		return Node{}, fmt.Errorf("parquet: %w: %d schema nodes belong to nothing", ErrFormat, len(rest))
	}
	return root, nil
}

// readNode takes the first element of flat together with the elements of
// everything below it, and returns what is left over.
func readNode(flat []SchemaElement, depth int) (Node, []SchemaElement, error) {
	if depth > maxSchemaDepth {
		return Node{}, nil, fmt.Errorf("parquet: %w: the schema nests more than %d deep",
			ErrFormat, maxSchemaDepth)
	}
	if len(flat) == 0 {
		return Node{}, nil, fmt.Errorf("parquet: %w: the schema ends before its last group is filled",
			ErrFormat)
	}

	n := Node{SchemaElement: flat[0]}
	flat = flat[1:]
	if n.NumChildren == 0 {
		return n, flat, nil
	}
	if n.NumChildren < 0 || int(n.NumChildren) > len(flat) {
		return Node{}, nil, fmt.Errorf("parquet: %w: %q claims %d fields and %d nodes follow it",
			ErrFormat, n.Name, n.NumChildren, len(flat))
	}

	n.Children = make([]Node, n.NumChildren)
	for i := range n.Children {
		child, rest, err := readNode(flat, depth+1)
		if err != nil {
			return Node{}, nil, err
		}
		n.Children[i], flat = child, rest
	}
	return n, flat, nil
}

// Column is a leaf of the schema, which is one column of values in the file.
type Column struct {
	// Path is the names from the root down to the leaf, without the root's
	// own. It is what a row group's chunks are keyed by and what a projection
	// names.
	Path []string

	// Element is the leaf itself: its physical type, its width when that type
	// is a fixed length byte array, and its annotation.
	Element SchemaElement

	// Type is what one value means in kuma's types. It is the type of a value
	// and not of the column: a leaf inside a repeated group is an int64 here
	// and a list of int64 in the schema the file adds up to.
	Type dtype.DataType

	// MaxDefinition is the definition level of a value that is present all the
	// way down its path, and MaxRepetition is the repetition level inside the
	// innermost repeated group above it.
	//
	// The two of them are how a flat run of values becomes nulls and list
	// boundaries again. A value whose definition level is below MaxDefinition
	// is missing, and which of the optional nodes on the path it is missing at
	// is how far below. A MaxRepetition of zero means the column has exactly
	// one value per row and the file writes no repetition levels for it at
	// all, which is the case for every column of a flat table.
	MaxDefinition int
	MaxRepetition int
}

// Name returns the column's path joined with dots, which is what the column is
// called everywhere outside the schema.
func (c *Column) Name() string { return strings.Join(c.Path, ".") }

// Columns returns the leaves of the schema in the order the row groups hold
// their chunks, which is the order they appear in the flattened schema.
func (m *Metadata) Columns() ([]Column, error) {
	root, err := m.Tree()
	if err != nil {
		return nil, err
	}

	var columns []Column
	var walk func(n *Node, path []string, definition, repetition int) error
	walk = func(n *Node, path []string, definition, repetition int) error {
		switch n.Repetition {
		case Optional:
			definition++
		case Repeated:
			definition++
			repetition++
		case Required, NoRepetition:
			// A field that is always there adds no level, and the root has no
			// repetition at all.
		}
		path = append(path, n.Name)

		if !n.Leaf() {
			for i := range n.Children {
				if err := walk(&n.Children[i], path, definition, repetition); err != nil {
					return err
				}
			}
			return nil
		}

		t, err := leafType(&n.SchemaElement)
		if err != nil {
			return err
		}
		columns = append(columns, Column{
			// The path is cloned because the append above writes into the same
			// array for every sibling at this level.
			Path:          slices.Clone(path),
			Element:       n.SchemaElement,
			Type:          t,
			MaxDefinition: definition,
			MaxRepetition: repetition,
		})
		return nil
	}

	for i := range root.Children {
		if err := walk(&root.Children[i], nil, 0, 0); err != nil {
			return nil, err
		}
	}
	return columns, nil
}

// Schema returns what the file holds, in kuma's types.
//
// The file's own metadata comes back on the schema, including the Arrow schema
// that pyarrow and Spark attach under ARROW:schema. That entry is left alone
// rather than read, so a type Arrow can hold and parquet cannot, a dictionary
// for instance, is still whatever parquet wrote it as.
//
// Two things the schema cannot carry are dropped on the way. A list element and
// a map value may be optional in parquet and kuma's list and map types hold a
// type rather than a field, so their nullability goes no further than the
// levels on the columns, which is where a decoder reads it from anyway.
func (m *Metadata) Schema() (dtype.Schema, error) {
	root, err := m.Tree()
	if err != nil {
		return dtype.Schema{}, err
	}
	fields, err := groupFields(&root)
	if err != nil {
		return dtype.Schema{}, err
	}

	s := dtype.Schema{Fields: fields, Metadata: make(dtype.Metadata, len(m.KeyValue))}
	for i, kv := range m.KeyValue {
		s.Metadata[i] = dtype.KeyValue{Key: kv.Key, Value: kv.Value}
	}
	return s, nil
}

// groupFields converts the children of a group.
func groupFields(n *Node) ([]dtype.Field, error) {
	fields := make([]dtype.Field, len(n.Children))
	for i := range n.Children {
		f, err := nodeField(&n.Children[i])
		if err != nil {
			return nil, err
		}
		fields[i] = f
	}
	return fields, nil
}

// nodeField converts one field of a group.
//
// A repeated field that is not the inside of a list or a map is a list of
// itself. That is how parquet wrote a repeated field before it had an
// annotation for one, and it is what a file converted from protobuf still looks
// like. The list is not nullable, because a repeated field with nothing in it
// is an empty list rather than a missing one.
func nodeField(n *Node) (dtype.Field, error) {
	t, err := nodeType(n)
	if err != nil {
		return dtype.Field{}, err
	}
	if n.Repetition == Repeated {
		return dtype.Field{Name: n.Name, Type: dtype.List{Elem: t}}, nil
	}
	return dtype.Field{Name: n.Name, Type: t, Nullable: n.Repetition == Optional}, nil
}

// nodeType converts a node without regard to whether it repeats, which is its
// parent's business rather than its own.
func nodeType(n *Node) (dtype.DataType, error) {
	if n.Leaf() {
		return leafType(&n.SchemaElement)
	}
	switch {
	case n.Logical.Kind == ListLogical,
		n.Logical.Kind == NoLogical && n.Converted == ConvertedList:
		return listType(n)
	case n.Logical.Kind == MapLogical,
		n.Logical.Kind == NoLogical &&
			(n.Converted == ConvertedMap || n.Converted == ConvertedMapKeyValue):
		return mapType(n)
	}

	fields, err := groupFields(n)
	if err != nil {
		return nil, err
	}
	return dtype.Struct{Fields: fields}, nil
}

// listType converts a group annotated as a list.
//
// The shape the format asks for has three levels: the annotated group, a
// repeated group inside it with one entry per element, and the element itself.
// Files written before that annotation existed have two, with the repeated node
// being the element. The rules for telling them apart are the ones in the
// format's backward compatibility notes, and they are guesses about the shape
// because there is nothing in an old file that says which it is.
func listType(n *Node) (dtype.DataType, error) {
	if len(n.Children) != 1 {
		return nil, fmt.Errorf("parquet: %w: the list %q has %d fields and a list has one",
			ErrFormat, n.Name, len(n.Children))
	}

	repeated := &n.Children[0]
	if repeated.Repetition != Repeated {
		return nil, fmt.Errorf("parquet: %w: the list %q holds %q, which is %s rather than repeated",
			ErrFormat, n.Name, repeated.Name, repeated.Repetition)
	}

	if twoLevelList(n, repeated) {
		t, err := nodeType(repeated)
		if err != nil {
			return nil, err
		}
		return dtype.List{Elem: t}, nil
	}

	// The element is a field of the repeated group and is converted as one, so
	// that an element which is itself a repeated field becomes a list the same
	// way it would anywhere else. Its name and its nullability are dropped,
	// since a list type has nowhere to put either.
	f, err := nodeField(&repeated.Children[0])
	if err != nil {
		return nil, err
	}
	return dtype.List{Elem: f.Type}, nil
}

// twoLevelList reports whether the repeated node inside a list is the element
// itself rather than a group wrapped around it.
//
// A repeated node that is not a group has nothing inside it to be the element.
// A repeated group with more than one field is a struct, since the three level
// shape puts exactly one thing in there. The two names are the ones the tools
// that wrote two level lists used, and a group called either of them is taken
// at its word.
func twoLevelList(list, repeated *Node) bool {
	switch {
	case repeated.Leaf(), len(repeated.Children) > 1:
		return true
	case repeated.Name == "array", repeated.Name == list.Name+"_tuple":
		return true
	}
	return false
}

// mapType converts a group annotated as a map.
//
// A map is an annotated group holding one repeated group of two fields, a key
// and a value. The repeated group is what carries the entries, so a map with
// two entries in a row is two values of every leaf under it and one row.
func mapType(n *Node) (dtype.DataType, error) {
	if len(n.Children) != 1 {
		return nil, fmt.Errorf("parquet: %w: the map %q has %d fields and a map has one",
			ErrFormat, n.Name, len(n.Children))
	}

	entries := &n.Children[0]
	if entries.Repetition != Repeated {
		return nil, fmt.Errorf("parquet: %w: the map %q holds %q, which is %s rather than repeated",
			ErrFormat, n.Name, entries.Name, entries.Repetition)
	}
	if len(entries.Children) == 1 {
		return nil, fmt.Errorf("parquet: %w: the map %q has a key and no value, which is a set",
			ErrUnsupported, n.Name)
	}
	if len(entries.Children) != 2 {
		return nil, fmt.Errorf(
			"parquet: %w: an entry of the map %q holds %d fields and an entry is a key and a value",
			ErrFormat, n.Name, len(entries.Children))
	}

	key, err := nodeField(&entries.Children[0])
	if err != nil {
		return nil, err
	}
	value, err := nodeField(&entries.Children[1])
	if err != nil {
		return nil, err
	}
	return dtype.Map{Key: key.Type, Value: value.Type}, nil
}

// leafType converts a column's physical type and annotation.
func leafType(e *SchemaElement) (dtype.DataType, error) {
	// The annotation a writer puts on a column it knows nothing about, which in
	// practice is a column whose values are all missing. The physical type
	// under it is whatever the writer felt like and means nothing.
	if e.Logical.Kind == UnknownLogical {
		return dtype.Null, nil
	}

	switch e.Type {
	case Boolean:
		return dtype.Bool, nil
	case Int32:
		return int32Type(e)
	case Int64:
		return int64Type(e)
	case Int96:
		// The timestamp Impala wrote before parquet had one, as a Julian day
		// number and a count of nanoseconds into that day. Nothing has written
		// one for years and files full of them are still being read. The zone
		// is left empty because the writers disagreed about whether the value
		// was UTC or the machine's local time, and guessing wrong moves every
		// value in the column by hours.
		return dtype.Timestamp{Unit: dtype.Nanosecond}, nil
	case Float:
		return dtype.Float32, nil
	case Double:
		return dtype.Float64, nil
	case ByteArray:
		return bytesType(e)
	case FixedLenByteArray:
		return fixedType(e)
	case NoType:
		return nil, fmt.Errorf("parquet: %w: %q has no physical type and no fields", ErrFormat, e.Name)
	}
	return nil, fmt.Errorf("parquet: %w: the column %q is a %s", ErrUnsupported, e.Name, e.Type)
}

// int32Type is a four byte integer, which is how parquet writes every integer
// narrower than that, a date, and a time of day in milliseconds.
func int32Type(e *SchemaElement) (dtype.DataType, error) {
	switch e.Logical.Kind {
	case IntegerLogical:
		return integerType(e, 32)
	case DateLogical:
		return dtype.Date32, nil
	case TimeLogical:
		if e.Logical.Unit != Millis {
			return nil, fmt.Errorf("parquet: %w: the time %q is in %s and an int32 holds milliseconds",
				ErrFormat, e.Name, e.Logical.Unit)
		}
		return dtype.Time32{Unit: dtype.Millisecond}, nil
	case DecimalLogical:
		return decimalType(e)
	case NoLogical:
	default:
		return nil, mismatch(e)
	}

	switch e.Converted {
	case ConvertedInt8:
		return dtype.Int8, nil
	case ConvertedInt16:
		return dtype.Int16, nil
	case ConvertedInt32:
		return dtype.Int32, nil
	case ConvertedUint8:
		return dtype.Uint8, nil
	case ConvertedUint16:
		return dtype.Uint16, nil
	case ConvertedUint32:
		return dtype.Uint32, nil
	case ConvertedDate:
		return dtype.Date32, nil
	case ConvertedTimeMillis:
		return dtype.Time32{Unit: dtype.Millisecond}, nil
	case ConvertedDecimal:
		return decimalType(e)
	case NoConverted:
		return dtype.Int32, nil
	default:
		return nil, mismatch(e)
	}
}

// int64Type is an eight byte integer, which is also every timestamp parquet has
// a logical type for.
func int64Type(e *SchemaElement) (dtype.DataType, error) {
	switch e.Logical.Kind {
	case IntegerLogical:
		return integerType(e, 64)
	case TimeLogical:
		u, err := timeUnit(e)
		if err != nil {
			return nil, err
		}
		if u == dtype.Millisecond {
			return nil, fmt.Errorf("parquet: %w: the time %q is in %s and an int64 holds micros or nanos",
				ErrFormat, e.Name, e.Logical.Unit)
		}
		return dtype.Time64{Unit: u}, nil
	case TimestampLogical:
		u, err := timeUnit(e)
		if err != nil {
			return nil, err
		}
		// A timestamp adjusted to UTC is an instant and one that is not is a
		// reading off a wall clock somewhere unstated. The zone says which,
		// and UTC is the only name parquet can write down.
		var zone string
		if e.Logical.UTC {
			zone = "UTC"
		}
		return dtype.Timestamp{Unit: u, Zone: zone}, nil
	case DecimalLogical:
		return decimalType(e)
	case NoLogical:
	default:
		return nil, mismatch(e)
	}

	switch e.Converted {
	case ConvertedInt64:
		return dtype.Int64, nil
	case ConvertedUint64:
		return dtype.Uint64, nil
	case ConvertedTimeMicros:
		return dtype.Time64{Unit: dtype.Microsecond}, nil
	case ConvertedTimestampMillis:
		// The converted timestamps are instants, which is the one thing the
		// logical type had to add a flag for because these two could not say
		// they were anything else.
		return dtype.Timestamp{Unit: dtype.Millisecond, Zone: "UTC"}, nil
	case ConvertedTimestampMicros:
		return dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}, nil
	case ConvertedDecimal:
		return decimalType(e)
	case NoConverted:
		return dtype.Int64, nil
	default:
		return nil, mismatch(e)
	}
}

// bytesType is a length and that many bytes, which is a string, an opaque blob,
// and a decimal too big for either integer.
func bytesType(e *SchemaElement) (dtype.DataType, error) {
	switch e.Logical.Kind {
	case StringLogical, EnumLogical, JSONLogical:
		// An enum is one of a fixed set of names and a JSON document is text
		// with a grammar. Both are strings to anything that reads them, and
		// what they are on top of that is not something kuma has a type for.
		return dtype.String, nil
	case BSONLogical:
		return dtype.Binary, nil
	case DecimalLogical:
		return decimalType(e)
	case NoLogical:
	default:
		return nil, mismatch(e)
	}

	switch e.Converted {
	case ConvertedUTF8, ConvertedEnum, ConvertedJSON:
		return dtype.String, nil
	case ConvertedBSON:
		return dtype.Binary, nil
	case ConvertedDecimal:
		return decimalType(e)
	case NoConverted:
		return dtype.Binary, nil
	default:
		return nil, mismatch(e)
	}
}

// fixedType is a run of bytes of a width the schema gives, which is a decimal,
// a UUID, an interval, and anything else somebody needed a fixed width for.
func fixedType(e *SchemaElement) (dtype.DataType, error) {
	if e.TypeLength <= 0 {
		return nil, fmt.Errorf("parquet: %w: the column %q is %d bytes wide",
			ErrFormat, e.Name, e.TypeLength)
	}

	switch e.Logical.Kind {
	case DecimalLogical:
		return decimalType(e)
	case UUIDLogical:
		if e.TypeLength != 16 {
			return nil, fmt.Errorf("parquet: %w: the uuid %q is %d bytes wide and a uuid is 16",
				ErrFormat, e.Name, e.TypeLength)
		}
		return dtype.FixedSizeBinary{ByteWidth: 16}, nil
	case Float16Logical:
		return nil, fmt.Errorf("parquet: %w: the column %q is a half precision float",
			ErrUnsupported, e.Name)
	case NoLogical:
	default:
		return nil, mismatch(e)
	}

	switch e.Converted {
	case ConvertedDecimal:
		return decimalType(e)
	case ConvertedInterval:
		if e.TypeLength != 12 {
			return nil, fmt.Errorf("parquet: %w: the interval %q is %d bytes wide and an interval is 12",
				ErrFormat, e.Name, e.TypeLength)
		}
		// Parquet counts months, days and milliseconds in twelve bytes. The
		// narrowest thing kuma has that holds all three counts nanoseconds
		// instead of milliseconds, so the decoder widens the last of them,
		// which loses nothing.
		return dtype.Interval{Unit: dtype.MonthDayNano}, nil
	case NoConverted:
		return dtype.FixedSizeBinary{ByteWidth: e.TypeLength}, nil
	default:
		return nil, mismatch(e)
	}
}

// integerType is a column annotated with a width and a sign.
//
// The annotation is what the column means and the physical type is only how it
// travels, so a uint8 is an int32 on disk with an annotation saying eight bits
// and unsigned. A width wider than the physical type would not fit and is the
// one combination refused here.
func integerType(e *SchemaElement, physical int8) (dtype.DataType, error) {
	if e.Logical.BitWidth > physical {
		return nil, fmt.Errorf("parquet: %w: the integer %q is %d bits wide and it is written in %d",
			ErrFormat, e.Name, e.Logical.BitWidth, physical)
	}

	switch e.Logical.BitWidth {
	case 8:
		if e.Logical.Signed {
			return dtype.Int8, nil
		}
		return dtype.Uint8, nil
	case 16:
		if e.Logical.Signed {
			return dtype.Int16, nil
		}
		return dtype.Uint16, nil
	case 32:
		if e.Logical.Signed {
			return dtype.Int32, nil
		}
		return dtype.Uint32, nil
	case 64:
		if e.Logical.Signed {
			return dtype.Int64, nil
		}
		return dtype.Uint64, nil
	}
	return nil, fmt.Errorf("parquet: %w: the integer %q is %d bits wide",
		ErrFormat, e.Name, e.Logical.BitWidth)
}

// decimalType is a decimal's precision and scale, which arrive on the logical
// type in a new file and on the schema element itself in an old one.
func decimalType(e *SchemaElement) (dtype.DataType, error) {
	precision, scale := e.Precision, e.Scale
	if e.Logical.Kind == DecimalLogical {
		precision, scale = e.Logical.Precision, e.Logical.Scale
	}

	switch {
	case precision < 1:
		return nil, fmt.Errorf("parquet: %w: the decimal %q keeps %d digits",
			ErrFormat, e.Name, precision)
	case precision <= dtype.MaxDecimal128Precision:
		return dtype.Decimal128{Precision: precision, Scale: scale}, nil
	case precision <= dtype.MaxDecimal256Precision:
		return dtype.Decimal256{Precision: precision, Scale: scale}, nil
	}
	return nil, fmt.Errorf("parquet: %w: the decimal %q keeps %d digits and the widest kuma has %d",
		ErrUnsupported, e.Name, precision, dtype.MaxDecimal256Precision)
}

// timeUnit converts the unit of a logical time or timestamp. Parquet has no
// unit coarser than a millisecond, so a kuma time in seconds never comes out of
// a file.
func timeUnit(e *SchemaElement) (dtype.TimeUnit, error) {
	switch e.Logical.Unit {
	case Millis:
		return dtype.Millisecond, nil
	case Micros:
		return dtype.Microsecond, nil
	case Nanos:
		return dtype.Nanosecond, nil
	default:
		return dtype.Second, fmt.Errorf("parquet: %w: the %s %q has no unit",
			ErrFormat, e.Logical.Kind, e.Name)
	}
}

// mismatch is a column whose annotation does not go with the physical type
// under it, which is a file saying two things that cannot both be true.
func mismatch(e *SchemaElement) error {
	if e.Logical.Kind != NoLogical {
		return fmt.Errorf("parquet: %w: the column %q is written as a %s and means %s",
			ErrFormat, e.Name, e.Type, e.Logical.Kind)
	}
	return fmt.Errorf("parquet: %w: the column %q is written as a %s and converts as %s",
		ErrFormat, e.Name, e.Type, e.Converted)
}
