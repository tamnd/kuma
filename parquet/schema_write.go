package parquet

import (
	"fmt"

	"github.com/tamnd/kuma/dtype"
)

// Writing a schema.
//
// This is schema.go the other way round: a kuma schema turned into the flat
// list of nodes a footer holds. It is the first thing a file writer does and
// the thing everything after it depends on, because the leaves of the schema in
// the order they come out of here are the order the row groups have to hold
// their chunks in, and a file whose schema and chunks disagree is a file that
// reads as somebody else's columns rather than as a broken one.
//
// Two things make it more than a table lookup. The first is that a group is not
// a node with children but a node followed by them, so writing a tree means
// appending the nodes in the order a reader takes them off, which is what every
// function here is doing when it appends before it recurses. The second is that
// parquet says the same thing twice: a column narrower than the type it travels
// in carries a logical annotation and the converted type the logical types
// replaced, and both go out, since the converted one is what a reader written
// before 2019 looks at and it costs a couple of bytes.
//
// What kuma can say and parquet cannot is refused rather than approximated. A
// duration is an int64 in a file with nothing to say it is a duration, a date64
// is a date the format has no width for, and an interval counts nanoseconds
// where parquet counts milliseconds. Writing any of those as the nearest thing
// gives a file that reads back as a different column, so each of them is an
// error naming the type and the reason. What is approximated is the other way
// round, where kuma has two types for one thing parquet has: a large string and
// a string are one byte array, a fixed size list and a list are one list, and a
// dictionary is an encoding rather than a type, so those are written as the
// thing parquet has and come back as it.

// The names the format gives the nodes that hold a list or a map together.
// Nothing reads them except a reader guessing at an old file, but a file whose
// nodes are named anything else is a file that looks wrong to anyone who opens
// it with another tool.
const (
	rootName     = "schema"
	elementName  = "element"
	keyValueName = "key_value"
	keyName      = "key"
	valueName    = "value"
)

// SetSchema sets the schema of the file to s, which is what Schema reads back.
//
// It fills in the nodes and the key value metadata and touches nothing else, so
// a caller building a footer sets the schema first and then adds the row groups
// that go with it. The order of the leaves is the order the columns appear in,
// which is the order every row group has to hold its chunks in.
//
// A type parquet has no way of writing is an error naming the column and the
// type. A type kuma has two of and parquet has one of is written as the one:
// large strings and large lists lose the large, a fixed size list becomes an
// ordinary list, and a dictionary is written as the type of its values, since
// in parquet a dictionary is a decision about a page rather than a type. Those
// come back as what they were written as rather than as what they were.
//
// Metadata on the schema is kept, since a footer has a place for it. Metadata
// on a field is dropped, since a footer has nowhere to put it. Nothing else is
// lost.
func (m *Metadata) SetSchema(s dtype.Schema) error {
	// The schema is checked here rather than trusted because everything below
	// reads a type without asking whether it means anything, and because a
	// schema with two columns of the same name produces a file whose columns
	// cannot be told apart by the only thing that names them.
	if err := s.Validate(); err != nil {
		return fmt.Errorf("parquet: %w", err)
	}

	// The root is the file itself. Its name is what the writer called the
	// record type and nothing reads it, so it is called what everything else
	// calls it.
	nodes := make([]SchemaElement, 1, len(s.Fields)+1)
	nodes[0] = group(rootName, Required, int32(len(s.Fields)))

	var err error
	for _, f := range s.Fields {
		if nodes, err = appendField(nodes, f, f.Name, 1); err != nil {
			return err
		}
	}

	m.Nodes = nodes
	m.KeyValue = nil
	if s.Metadata != nil {
		m.KeyValue = make([]KeyValue, len(s.Metadata))
		for i, kv := range s.Metadata {
			m.KeyValue[i] = KeyValue{Key: kv.Key, Value: kv.Value}
		}
	}
	return nil
}

// appendField appends the nodes of one field of a group.
//
// Whether a value may be missing is on the field in kuma and on the node in
// parquet, which is the whole of the difference between the two.
func appendField(nodes []SchemaElement, f dtype.Field, path string, depth int) ([]SchemaElement, error) {
	r := Required
	if f.Nullable {
		r = Optional
	}
	return appendType(nodes, f.Name, path, r, f.Type, depth)
}

// appendType appends the node for a type and the nodes for everything under it.
//
// The name is what the node is called in the file and the path is the way down
// to it from the top, which is the same thing for a column and is not for
// anything inside one. Only the errors use the path, and they use it because
// the nodes a list or a map is made of are named by the format rather than by
// the caller: a refusal saying that the column "element" is a duration names
// nothing a caller can go and look at, and one naming "history.list.element"
// does.
//
// The depth is how far down the parquet tree this node sits rather than how far
// down the kuma type, because a list is one type and three nodes. It is checked
// against what the reader will accept: kuma allows a type to nest deeper than
// parquet does, so a schema this wrote without asking could be one that no
// reader here can read back.
func appendType(nodes []SchemaElement, name, path string, r Repetition, t dtype.DataType,
	depth int) ([]SchemaElement, error) {
	if depth > maxSchemaDepth {
		return nil, fmt.Errorf("parquet: %w: %q is more than %d nodes deep, which is deeper than a file says",
			ErrUnsupported, path, maxSchemaDepth)
	}

	switch t := t.(type) {
	case dtype.List:
		return appendList(nodes, name, path, r, t.Elem, depth)
	case dtype.LargeList:
		return appendList(nodes, name, path, r, t.Elem, depth)
	case dtype.FixedSizeList:
		return appendList(nodes, name, path, r, t.Elem, depth)
	case dtype.Struct:
		return appendStruct(nodes, name, path, r, t.Fields, depth)
	case dtype.Map:
		return appendMap(nodes, name, path, r, t, depth)
	case dtype.Dictionary:
		// The values are what the column holds and the indices are how a page
		// wrote them down, which is a decision the page writer makes for any
		// column and not something the schema says.
		return appendType(nodes, name, path, r, t.Value, depth)
	}

	e, err := leafElement(name, path, r, t)
	if err != nil {
		return nil, err
	}
	return append(nodes, e), nil
}

// appendList appends a list as the three nodes the format asks for: the
// annotated group, a repeated group inside it with one entry per element, and
// the element.
//
// The element is optional because a list type in kuma says nothing about
// whether an element may be missing, and a reader gives back the same type
// either way. Optional is the shape every tool writes and the one that can hold
// a list with a missing element in it.
func appendList(nodes []SchemaElement, name, path string, r Repetition, elem dtype.DataType,
	depth int) ([]SchemaElement, error) {
	list := group(name, r, 1)
	list.Converted = ConvertedList
	list.Logical = LogicalType{Kind: ListLogical}

	nodes = append(nodes, list, group(listName, Repeated, 1))
	return appendType(nodes, elementName, path+"."+listName+"."+elementName, Optional, elem, depth+2)
}

// appendStruct appends a group with no annotation and the fields under it,
// which is what parquet calls a struct by not calling it anything.
func appendStruct(nodes []SchemaElement, name, path string, r Repetition, fields []dtype.Field,
	depth int) ([]SchemaElement, error) {
	if len(fields) == 0 {
		return nil, fmt.Errorf("parquet: %w: the struct %q has no fields, and a node with none is a column",
			ErrUnsupported, path)
	}

	nodes = append(nodes, group(name, r, int32(len(fields))))

	var err error
	for _, f := range fields {
		if nodes, err = appendField(nodes, f, path+"."+f.Name, depth+1); err != nil {
			return nil, err
		}
	}
	return nodes, nil
}

// appendMap appends a map as an annotated group holding one repeated group of a
// key and a value, which is how the format writes one.
//
// The key is required because a map with a missing key is not a map, and the
// value is optional for the same reason a list element is.
func appendMap(nodes []SchemaElement, name, path string, r Repetition, t dtype.Map,
	depth int) ([]SchemaElement, error) {
	entries := group(name, r, 1)
	entries.Converted = ConvertedMap
	entries.Logical = LogicalType{Kind: MapLogical}

	nodes = append(nodes, entries, group(keyValueName, Repeated, 2))

	inside := path + "." + keyValueName + "."
	nodes, err := appendType(nodes, keyName, inside+keyName, Required, t.Key, depth+2)
	if err != nil {
		return nil, err
	}
	return appendType(nodes, valueName, inside+valueName, Optional, t.Value, depth+2)
}

// group returns a node with children, which carries no physical type and no
// annotation until whoever wanted it puts one on.
func group(name string, r Repetition, children int32) SchemaElement {
	e := element(name, r)
	e.NumChildren = children
	return e
}

// element returns a node with nothing said about it.
//
// It exists because the values that mean a writer said nothing are not the zero
// of their types: a zero physical type is a boolean and a zero converted type
// is a string, so a node built by taking the zero value and filling in a field
// or two is a node claiming things nobody asked for.
func element(name string, r Repetition) SchemaElement {
	return SchemaElement{Name: name, Type: NoType, Repetition: r, Converted: NoConverted}
}

// leafElement returns the node for a column of values.
//
// The types that carry parameters are taken apart here, where the parameters
// are reachable, and everything else goes to the switch below on what kind of
// type it is. The two switches are what they are because a type in kuma is an
// interface: a timestamp is a struct with a unit in it and an int64 is a value
// nobody outside the package can name, so the first can be matched by its type
// and the second cannot.
func leafElement(name, path string, r Repetition, t dtype.DataType) (SchemaElement, error) {
	e := element(name, r)

	var err error
	switch t := t.(type) {
	case dtype.FixedSizeBinary:
		err = e.fixed(t, path)
	case dtype.Time32:
		err = e.time32(t, path)
	case dtype.Time64:
		e.time64(t)
	case dtype.Timestamp:
		err = e.timestamp(t, path)
	case dtype.Decimal128:
		e.decimal(t.Precision, t.Scale)
	case dtype.Decimal256:
		e.decimal(t.Precision, t.Scale)
	default:
		err = e.simple(t, path)
	}
	return e, err
}

// simple fills in a node for a type that has no parameters, which is every type
// that is one physical type and an annotation that says nothing else.
func (e *SchemaElement) simple(t dtype.DataType, path string) error {
	switch t.Kind() {
	case dtype.NullKind:
		// A column of nothing. The annotation is what says so and the physical
		// type under it means nothing, since every value is missing, so it is
		// the one the tools that write these use.
		e.Type = Int32
		e.Logical = LogicalType{Kind: UnknownLogical}

	case dtype.BoolKind:
		e.Type = Boolean

	// Parquet has two integer widths and kuma has eight, so six of them travel
	// in a wider type with an annotation saying how wide they really are. The
	// two that are exactly a parquet type carry no annotation at all, because
	// the format already says an unannotated int32 is a signed 32 bit integer.
	case dtype.Int8Kind:
		e.integer(Int32, 8, true, ConvertedInt8)
	case dtype.Int16Kind:
		e.integer(Int32, 16, true, ConvertedInt16)
	case dtype.Int32Kind:
		e.Type = Int32
	case dtype.Int64Kind:
		e.Type = Int64
	case dtype.Uint8Kind:
		e.integer(Int32, 8, false, ConvertedUint8)
	case dtype.Uint16Kind:
		e.integer(Int32, 16, false, ConvertedUint16)
	case dtype.Uint32Kind:
		e.integer(Int32, 32, false, ConvertedUint32)
	case dtype.Uint64Kind:
		e.integer(Int64, 64, false, ConvertedUint64)

	case dtype.Float32Kind:
		e.Type = Float
	case dtype.Float64Kind:
		e.Type = Double

	case dtype.StringKind, dtype.LargeStringKind:
		e.Type = ByteArray
		e.Converted = ConvertedUTF8
		e.Logical = LogicalType{Kind: StringLogical}
	case dtype.BinaryKind, dtype.LargeBinaryKind:
		// Bytes with nothing said about them, which is what an unannotated byte
		// array is.
		e.Type = ByteArray

	case dtype.Date32Kind:
		e.Type = Int32
		e.Converted = ConvertedDate
		e.Logical = LogicalType{Kind: DateLogical}

	default:
		// A duration, an interval, a date in milliseconds, and anything added
		// to kuma after this was written. The first three are types parquet
		// cannot hold without changing the values, which is a cast and not a
		// writer's decision.
		return fmt.Errorf("parquet: %w: the column %q is a %s, which the format has no type for",
			ErrUnsupported, path, t)
	}
	return nil
}

// fixed annotates a run of bytes of a width the schema gives.
//
// A width of nothing is a type kuma allows and the format does not, since a
// file says the width of one of these on the node and a reader refuses a node
// that says nought.
func (e *SchemaElement) fixed(t dtype.FixedSizeBinary, path string) error {
	if t.ByteWidth <= 0 {
		return fmt.Errorf("parquet: %w: the column %q holds values of %d bytes",
			ErrUnsupported, path, t.ByteWidth)
	}

	e.Type = FixedLenByteArray
	e.TypeLength = t.ByteWidth
	return nil
}

// integer annotates a column as an integer that is narrower than the type it
// travels in, or as one that is unsigned, which are the two things a physical
// type cannot say on its own.
//
// It says it twice. The logical type is what a reader written since 2019 looks
// at and the converted type is what everything older looks at, and they mean
// the same thing here, so writing both costs three bytes and reads everywhere.
func (e *SchemaElement) integer(physical Type, bits int8, signed bool, converted ConvertedType) {
	e.Type = physical
	e.Converted = converted
	e.Logical = LogicalType{Kind: IntegerLogical, BitWidth: bits, Signed: signed}
}

// time32 annotates a time of day held in four bytes.
//
// Parquet counts a time of day in milliseconds at the coarsest and kuma counts
// one in seconds at the coarsest, so the one unit that has nowhere to go is
// seconds. Multiplying by a thousand on the way out would write a column whose
// type is not the one the caller had, which is a decision for a cast rather
// than for a writer.
func (e *SchemaElement) time32(t dtype.Time32, path string) error {
	if t.Unit != dtype.Millisecond {
		return fmt.Errorf("parquet: %w: the time %q is in %s and the coarsest a file holds is milliseconds",
			ErrUnsupported, path, t.Unit)
	}

	e.Type = Int32
	e.Converted = ConvertedTimeMillis
	e.Logical = LogicalType{Kind: TimeLogical, Unit: Millis}
	return nil
}

// time64 annotates a time of day held in eight bytes, which is one in
// microseconds or in nanoseconds and nothing else.
func (e *SchemaElement) time64(t dtype.Time64) {
	e.Type = Int64
	if t.Unit == dtype.Microsecond {
		e.Converted = ConvertedTimeMicros
		e.Logical = LogicalType{Kind: TimeLogical, Unit: Micros}
		return
	}

	// The converted types were written before parquet had nanoseconds and none
	// of them says one, so a nanosecond column carries the logical type alone.
	// A reader that only knows the converted types reads it as an int64, which
	// is what it is.
	e.Logical = LogicalType{Kind: TimeLogical, Unit: Nanos}
}

// timestamp annotates an instant, or a reading off a wall clock that belongs to
// no particular place.
//
// Which of the two a column holds is the whole of what a zone means. Parquet says
// it with a flag rather than a name, so a column in any zone is written as
// adjusted to UTC, which is true of every one of them: an instant is an instant
// whatever clock it is displayed on. The name of the zone has nowhere to go and
// comes back as UTC.
func (e *SchemaElement) timestamp(t dtype.Timestamp, path string) error {
	unit, ok := logicalUnit(t.Unit)
	if !ok {
		return fmt.Errorf("parquet: %w: the timestamp %q is in %s and the coarsest a file holds is milliseconds",
			ErrUnsupported, path, t.Unit)
	}

	e.Type = Int64
	e.Logical = LogicalType{Kind: TimestampLogical, Unit: unit, UTC: t.Zone != ""}

	// The converted timestamps are instants and have no way to say they are
	// not, so they go on a column that has a zone and are left off one that
	// does not. Leaving one off tells a reader that only knows them that it is
	// looking at something it cannot read as a timestamp, which is better than
	// telling it something false.
	if t.Zone == "" {
		return nil
	}
	switch unit {
	case Millis:
		e.Converted = ConvertedTimestampMillis
	case Micros:
		e.Converted = ConvertedTimestampMicros
	default:
		// Nanoseconds, which the format added after it stopped adding
		// converted types.
	}
	return nil
}

// decimal annotates an exact number of a precision and a scale.
//
// The values are written as bytes rather than as an integer, which is what
// every writer does by default and what covers every precision with one rule.
// The width is the fewest bytes the precision fits in, so a decimal of nine
// digits is four bytes rather than sixteen.
func (e *SchemaElement) decimal(precision, scale int32) {
	e.Type = FixedLenByteArray
	e.TypeLength = decimalWidth(precision)
	e.Converted = ConvertedDecimal
	e.Logical = LogicalType{Kind: DecimalLogical, Precision: precision, Scale: scale}

	// The precision and the scale go on the node as well as on the logical
	// type. A file that only wrote the converted type had nowhere else to put
	// them, so that is where a reader that only knows the converted type looks.
	e.Precision = precision
	e.Scale = scale
}

// logicalUnit converts a kuma time unit into the parquet one, and says whether
// there is one. Parquet has nothing coarser than a millisecond, so a column in
// seconds is the one that has nowhere to go.
func logicalUnit(u dtype.TimeUnit) (TimeUnit, bool) {
	switch u {
	case dtype.Millisecond:
		return Millis, true
	case dtype.Microsecond:
		return Micros, true
	case dtype.Nanosecond:
		return Nanos, true
	default:
		return NoUnit, false
	}
}

// decimalDigits is how many digits fit in a signed integer of a width in bytes,
// indexed by that width. A signed integer of n bytes holds values below two to
// the power of eight n minus one, and the entry is how many powers of ten fit
// under that. The first entry is a width of no bytes and holds nothing.
var decimalDigits = [...]int32{
	0, 2, 4, 6, 9, 11, 14, 16, 18, 21, 23, 26, 28, 31, 33, 35, 38,
	40, 43, 45, 47, 50, 52, 55, 57, 59, 62, 64, 67, 69, 71, 74, 76,
}

// decimalWidth returns the fewest bytes a decimal of the given precision fits
// in. The widest kuma decimal keeps 76 digits, which is the last entry of the
// table, so there is always one that fits.
func decimalWidth(precision int32) int32 {
	width := int32(1)
	for width < int32(len(decimalDigits))-1 && decimalDigits[width] < precision {
		width++
	}
	return width
}
