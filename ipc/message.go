package ipc

import (
	"fmt"

	"github.com/tamnd/kuma/dtype"
)

// The Arrow IPC schema message.
//
// A schema on the wire is a FlatBuffers Message whose header is a Schema
// table, which holds a Field table per column. A Field holds a name, a
// nullable flag, a type, its own children and its own metadata, so a schema is
// a tree rather than a list and the nested types are the reason.
//
// The type itself is a union: one number saying which of the twenty six type
// tables it is, and the table. That is a different shape from the format
// string the C data interface uses for the same job, and this file is the
// second of the two mappings. They agree on what a kuma type is and disagree
// about how to write it down, which is the specification's decision rather
// than this package's.

// The numbers of the Type union, in the order Type.fbs declares them. Zero is
// the union being absent, which is a schema that does not say what its column
// holds and is refused.
const (
	fbTypeNone = iota
	fbTypeNull
	fbTypeInt
	fbTypeFloat
	fbTypeBinary
	fbTypeUtf8
	fbTypeBool
	fbTypeDecimal
	fbTypeDate
	fbTypeTime
	fbTypeTimestamp
	fbTypeInterval
	fbTypeList
	fbTypeStruct
	fbTypeUnion
	fbTypeFixedSizeBinary
	fbTypeFixedSizeList
	fbTypeMap
	fbTypeDuration
	fbTypeLargeBinary
	fbTypeLargeUtf8
	fbTypeLargeList
	fbTypeRunEndEncoded
	fbTypeBinaryView
	fbTypeUtf8View
	fbTypeListView
	fbTypeLargeListView
)

// typeNames is what each number in the union is called, for the error a column
// of a type kuma has no equivalent for produces. A reader who gets one is
// holding a file somebody else wrote and needs to know what is in it.
var typeNames = [...]string{
	fbTypeNone:            "nothing",
	fbTypeNull:            "null",
	fbTypeInt:             "int",
	fbTypeFloat:           "floating point",
	fbTypeBinary:          "binary",
	fbTypeUtf8:            "utf8",
	fbTypeBool:            "bool",
	fbTypeDecimal:         "decimal",
	fbTypeDate:            "date",
	fbTypeTime:            "time",
	fbTypeTimestamp:       "timestamp",
	fbTypeInterval:        "interval",
	fbTypeList:            "list",
	fbTypeStruct:          "struct",
	fbTypeUnion:           "union",
	fbTypeFixedSizeBinary: "fixed size binary",
	fbTypeFixedSizeList:   "fixed size list",
	fbTypeMap:             "map",
	fbTypeDuration:        "duration",
	fbTypeLargeBinary:     "large binary",
	fbTypeLargeUtf8:       "large utf8",
	fbTypeLargeList:       "large list",
	fbTypeRunEndEncoded:   "run end encoded",
	fbTypeBinaryView:      "binary view",
	fbTypeUtf8View:        "utf8 view",
	fbTypeListView:        "list view",
	fbTypeLargeListView:   "large list view",
}

// The header numbers of the Message union, the metadata version kuma writes
// and reads, and the time and date units as Schema.fbs numbers them.
const (
	fbHeaderNone = iota
	fbHeaderSchema
	fbHeaderDictionaryBatch
	fbHeaderRecordBatch
	fbHeaderTensor
	fbHeaderSparseTensor
)

// fbVersionV5 is metadata version V5, which is what every implementation since
// Arrow 1.0 writes. V4 is readable by the same readers and differs in how a
// union writes its buffers, which kuma has no unions to write.
const fbVersionV5 int16 = 4

// The unit enumerations. TimeUnit counts up from seconds, DateUnit has two
// values, IntervalUnit has three, and Precision names the three widths of a
// float. All of them are shorts in the schema, which is what the type says
// here so that a call site does not have to.
const (
	fbSecond int16 = iota
	fbMillisecond
	fbMicrosecond
	fbNanosecond
)

const (
	fbDay int16 = iota
	fbDateMillisecond
)

const (
	fbYearMonth int16 = iota
	fbDayTime
	fbMonthDayNano
)

const (
	fbHalf int16 = iota
	fbSingle
	fbDouble
)

// The field numbers of the tables this file writes and reads. They are the
// order the fields appear in the .fbs files, and a union counts as two: the
// number saying which member it is, then the member.
const (
	fbMessageVersion = iota
	fbMessageHeaderType
	fbMessageHeader
	fbMessageBodyLength
	fbMessageMetadata
)

const (
	fbSchemaEndianness = iota
	fbSchemaFields
	fbSchemaMetadata
	fbSchemaFeatures
)

const (
	fbFieldName = iota
	fbFieldNullable
	fbFieldTypeType
	fbFieldType
	fbFieldDictionary
	fbFieldChildren
	fbFieldMetadata
)

const (
	fbKeyValueKey = iota
	fbKeyValueValue
)

const (
	fbDictionaryID = iota
	fbDictionaryIndexType
	fbDictionaryOrdered
	fbDictionaryKind
)

// The parameters of the type tables, in declaration order. The tables with no
// parameters have no numbers because there is nothing in them.
const (
	fbIntWidth = iota
	fbIntSigned
)

const (
	fbDecimalPrecision = iota
	fbDecimalScale
	fbDecimalWidth
)

const (
	fbTimeUnit = iota
	fbTimeWidth
)

const (
	fbTimestampUnit = iota
	fbTimestampZone
)

// The tables with one parameter. The number is zero every time, and naming it
// is the difference between a reader knowing which field it is looking at and
// counting the tables in the .fbs file to find out.
const (
	fbDateUnit             = 0
	fbDurationUnit         = 0
	fbIntervalUnit         = 0
	fbFloatPrecision       = 0
	fbFixedSizeBinaryWidth = 0
	fbFixedSizeListLen     = 0
)

// EncodeSchema returns the Arrow IPC schema message for s.
//
// The result is an encapsulated message: the continuation bytes, the length,
// and the metadata padded out to eight. That is what pyarrow's schema.serialize
// writes and what its read_schema reads, so the bytes go straight into any
// other implementation without anything wrapped around them first.
//
// A type kuma can hold but Arrow cannot name, and there are none today, would
// be an error here rather than a column that quietly changes type.
func EncodeSchema(s dtype.Schema) ([]byte, error) {
	msg, err := encodeSchemaMessage(s)
	if err != nil {
		return nil, err
	}
	if err := checkLength(len(msg), "a schema"); err != nil {
		return nil, err
	}
	return frame(msg), nil
}

// encodeSchemaMessage writes the message on its own, without the framing. The
// stream and the file both hold one of these and put their own bytes around
// it.
func encodeSchemaMessage(s dtype.Schema) ([]byte, error) {
	sw := schemaWriter{types: make(map[string]typeRef)}
	schema, err := sw.schema(s)
	if err != nil {
		return nil, err
	}

	w := &sw.w
	w.startTable()
	w.slotInt(fbMessageVersion, fbVersionV5, 0)
	w.slotUint8(fbMessageHeaderType, fbHeaderSchema)
	w.slotOffset(fbMessageHeader, schema)
	return w.finish(w.endTable()), nil
}

// schemaWriter builds one schema message.
//
// It is a FlatBuffers builder and a note of the type tables written so far. A
// wide table is mostly the same handful of types over and over, and a table
// already in the buffer can be pointed at again rather than written again, so
// a hundred int64 columns cost one description of int64 and a hundred offsets
// to it. The key is the type as it prints, which names every parameter of it.
type schemaWriter struct {
	w     fbBuilder
	types map[string]typeRef
}

// typeRef is a type table already in the buffer: which member of the union it
// is, and where it starts.
type typeRef struct {
	kind int
	off  fbOffset
}

// schema writes the Schema table itself and returns where it starts. A schema
// message holds one of these and so does the footer of a file, and the table is
// the same table either way.
func (sw *schemaWriter) schema(s dtype.Schema) (fbOffset, error) {
	fields := make([]fbOffset, len(s.Fields))
	for i, f := range s.Fields {
		off, err := sw.field(f)
		if err != nil {
			return 0, err
		}
		fields[i] = off
	}
	metadata := encodeKeyValues(&sw.w, s.Metadata)

	// The vectors go down before the table that points at them, since nothing
	// can be pointed at before it exists.
	w := &sw.w
	fieldsVec := w.offsets(fields)
	metadataVec := fbOffset(0)
	if len(metadata) > 0 {
		metadataVec = w.offsets(metadata)
	}

	w.startTable()
	w.slotOffset(fbSchemaFields, fieldsVec)
	w.slotOffset(fbSchemaMetadata, metadataVec)
	return w.endTable(), nil
}

// DecodeSchema reads an encapsulated Arrow IPC schema message.
//
// The bytes are somebody else's, so everything in them is checked: the message
// has to be a schema, the version has to be one kuma reads, and every offset
// has to point inside the message. A column of a type kuma has no equivalent
// for is an error naming the type and the column, since that is what the
// person holding the file needs to know.
//
// Anything after the message is ignored, so a schema read out of the front of a
// stream does not have to be cut out of it first.
func DecodeSchema(b []byte) (dtype.Schema, error) {
	body, _, err := unframe(b)
	if err != nil {
		return dtype.Schema{}, err
	}
	if body == nil {
		return dtype.Schema{}, fmt.Errorf("ipc: %w: a schema of no bytes", ErrMessage)
	}
	return decodeSchemaMessage(body)
}

// decodeSchemaMessage reads the message on its own, which is what the stream
// reader has in hand once it has taken the framing off.
func decodeSchemaMessage(b []byte) (dtype.Schema, error) {
	msg, err := fbRoot(b)
	if err != nil {
		return dtype.Schema{}, err
	}
	version, err := msg.integer(fbMessageVersion, int16(0))
	if err != nil {
		return dtype.Schema{}, err
	}
	if version > fbVersionV5 {
		return dtype.Schema{}, fmt.Errorf("ipc: %w: metadata version %d, this reads up to V5",
			ErrUnsupported, version)
	}
	header, err := msg.uint8(fbMessageHeaderType, fbHeaderNone)
	if err != nil {
		return dtype.Schema{}, err
	}
	if header != fbHeaderSchema {
		return dtype.Schema{}, fmt.Errorf("ipc: %w: the message is a %s, want a schema",
			ErrMessage, headerName(header))
	}
	table, ok, err := msg.table(fbMessageHeader)
	if err != nil {
		return dtype.Schema{}, err
	}
	if !ok {
		return dtype.Schema{}, fmt.Errorf("ipc: %w: the message says it holds a schema and holds nothing",
			ErrMessage)
	}
	return newDecoder(b).schema(table)
}

// messageHeader reads the two fields of a message that a reader needs before it
// knows what the message is: what kind of thing the header holds, and how many
// bytes of body come after the metadata.
//
// It takes the metadata on its own rather than the encapsulated message, since a
// reader pulling one out of a stream has the metadata in hand and the body is
// the thing it is about to go and read.
//
// The version is not checked here. Whichever decoder the kind leads to checks
// it, and it is the one that knows which versions it can read.
func messageHeader(meta []byte) (kind uint8, body int64, err error) {
	root, err := fbRoot(meta)
	if err != nil {
		return 0, 0, err
	}
	kind, err = root.uint8(fbMessageHeaderType, fbHeaderNone)
	if err != nil {
		return 0, 0, err
	}
	body, err = root.integer(fbMessageBodyLength, int64(0))
	if err != nil {
		return 0, 0, err
	}
	if body < 0 {
		return 0, 0, fmt.Errorf("ipc: %w: a %s with a body of %d bytes",
			ErrMessage, headerName(kind), body)
	}
	return kind, body, nil
}

// headerName is what a message header number is called, so that a message that
// is not the one expected says what it is instead.
func headerName(h uint8) string {
	switch h {
	case fbHeaderSchema:
		return "schema"
	case fbHeaderDictionaryBatch:
		return "dictionary batch"
	case fbHeaderRecordBatch:
		return "record batch"
	case fbHeaderTensor:
		return "tensor"
	case fbHeaderSparseTensor:
		return "sparse tensor"
	default:
		return "message with no header"
	}
}

// decoder reads a schema out of one message, and counts what it has read.
//
// The counting is the whole reason it is a type. A field can hold fields, so a
// message can point a field at itself and describe a schema that is infinitely
// deep, or point a hundred fields at one field that holds a hundred, and both
// of those are a few hundred bytes that ask for the rest of the machine. A
// real schema costs bytes in the message for every field in it, so the reader
// gives itself a budget of what the message could honestly hold and stops when
// it runs out.
type decoder struct {
	left  int // fields it may still read
	depth int // how far down it is now
}

// maxDepth is how deeply one field may nest. A list of a struct of a list is
// three, and a schema that goes past this is generated rather than written.
const maxDepth = 64

// newDecoder gives a reader of a message the budget that message can pay for.
// A field costs at least the four bytes of the offset that points at it, and
// in practice a good deal more.
func newDecoder(buf []byte) *decoder {
	return &decoder{left: len(buf) / 4}
}

// decodeSchema reads the Schema table itself, which the file format reads
// straight out of its footer rather than out of a message.
func (d *decoder) schema(t fbTable) (dtype.Schema, error) {
	endian, err := t.integer(fbSchemaEndianness, int16(0))
	if err != nil {
		return dtype.Schema{}, err
	}
	if endian != 0 {
		// Every machine kuma runs on is little endian, and a big endian file
		// would need every buffer byte swapped rather than merely read.
		return dtype.Schema{}, fmt.Errorf("ipc: %w: the schema is big endian", ErrUnsupported)
	}

	vec, ok, err := t.vector(fbSchemaFields)
	if err != nil {
		return dtype.Schema{}, err
	}
	var s dtype.Schema
	if ok {
		s.Fields = make([]dtype.Field, vec.len())
		for i := range vec.len() {
			var child fbTable
			if child, err = vec.table(i); err != nil {
				return dtype.Schema{}, err
			}
			if s.Fields[i], err = d.field(child); err != nil {
				return dtype.Schema{}, err
			}
		}
	}
	if s.Metadata, err = decodeKeyValues(t, fbSchemaMetadata); err != nil {
		return dtype.Schema{}, err
	}
	return s, nil
}

// field writes one Field table and returns where it starts.
func (sw *schemaWriter) field(f dtype.Field) (fbOffset, error) {
	children := childFields(f.Type)
	offs := make([]fbOffset, len(children))
	for i, c := range children {
		var err error
		if offs[i], err = sw.field(c); err != nil {
			return 0, err
		}
	}

	// A dictionary column writes the type of its indices as its own type and
	// hangs the value type off the dictionary table, the same split the C data
	// interface makes for the same reason.
	value := f.Type
	var index dtype.DataType
	if d, isDict := f.Type.(dtype.Dictionary); isDict {
		value, index = d.Value, d.Index
		if !dtype.IsInteger(index) {
			return 0, noFormat(f.Type)
		}
	}

	kind, typ, err := sw.dataType(value)
	if err != nil {
		return 0, err
	}
	w := &sw.w
	metadata := encodeKeyValues(w, f.Metadata)
	name := fbOffset(0)
	if f.Name != "" {
		name = w.str(f.Name)
	}

	childVec := fbOffset(0)
	if len(offs) > 0 {
		childVec = w.offsets(offs)
	}
	metadataVec := fbOffset(0)
	if len(metadata) > 0 {
		metadataVec = w.offsets(metadata)
	}
	dict := fbOffset(0)
	if index != nil {
		dict, err = encodeDictionary(w, index)
		if err != nil {
			return 0, err
		}
	}

	w.startTable()
	w.slotOffset(fbFieldName, name)
	w.slotBool(fbFieldNullable, f.Nullable)
	w.slotUint8(fbFieldTypeType, uint8(kind))
	w.slotOffset(fbFieldType, typ)
	w.slotOffset(fbFieldDictionary, dict)
	w.slotOffset(fbFieldChildren, childVec)
	w.slotOffset(fbFieldMetadata, metadataVec)
	return w.endTable(), nil
}

// decodeField reads one Field table, and its children with it.
func (d *decoder) field(t fbTable) (dtype.Field, error) {
	if d.left <= 0 {
		return dtype.Field{}, fmt.Errorf("ipc: %w: more fields than the message has room for",
			ErrMessage)
	}
	if d.depth > maxDepth {
		return dtype.Field{}, fmt.Errorf("ipc: %w: a field nested more than %d deep",
			ErrMessage, maxDepth)
	}
	d.left--

	var f dtype.Field
	var err error
	if f.Name, err = t.str(fbFieldName); err != nil {
		return dtype.Field{}, err
	}
	if f.Nullable, err = t.boolean(fbFieldNullable, false); err != nil {
		return dtype.Field{}, err
	}

	children, err := d.children(t)
	if err != nil {
		return dtype.Field{}, err
	}
	if f.Type, err = decodeType(t, f.Name, children); err != nil {
		return dtype.Field{}, err
	}
	if f.Type, err = decodeDictionary(t, f.Type); err != nil {
		return dtype.Field{}, err
	}
	if f.Metadata, err = decodeKeyValues(t, fbFieldMetadata); err != nil {
		return dtype.Field{}, err
	}
	if err := dtype.Validate(f.Type); err != nil {
		return dtype.Field{}, fmt.Errorf("ipc: field %q: %w", f.Name, err)
	}
	return f, nil
}

// decodeChildren reads the children of a field, which are the members of a
// struct, the values of a list, or the entries of a map.
func (d *decoder) children(t fbTable) ([]dtype.Field, error) {
	vec, ok, err := t.vector(fbFieldChildren)
	if err != nil || !ok {
		return nil, err
	}
	children := make([]dtype.Field, vec.len())

	d.depth++
	defer func() { d.depth-- }()
	for i := range vec.len() {
		child, err := vec.table(i)
		if err != nil {
			return nil, err
		}
		if children[i], err = d.field(child); err != nil {
			return nil, err
		}
	}
	return children, nil
}

// encodeDictionary writes the DictionaryEncoding table of a dictionary column.
//
// The identifier is left at zero. It is what ties a column to the dictionary
// batch that carries its values, and there are no dictionary batches to tie it
// to yet, so writing an identifier that names nothing would be worse than
// writing the one every reader treats as the first.
func encodeDictionary(w *fbBuilder, index dtype.DataType) (fbOffset, error) {
	bits, ok := dtype.Bits(index)
	if !ok {
		return 0, noFormat(index)
	}
	w.startTable()
	w.slotInt(fbIntWidth, int32(bits), 0)
	w.slotBool(fbIntSigned, dtype.IsSigned(index))
	indexType := w.endTable()

	w.startTable()
	w.slotOffset(fbDictionaryIndexType, indexType)
	return w.endTable(), nil
}

// decodeDictionary wraps a type in a dictionary if the field says it is one.
// What the field itself holds is the indices, and the type read so far is the
// values.
func decodeDictionary(t fbTable, value dtype.DataType) (dtype.DataType, error) {
	d, ok, err := t.table(fbFieldDictionary)
	if err != nil || !ok {
		return value, err
	}
	index := dtype.Int32
	it, ok, err := d.table(fbDictionaryIndexType)
	if err != nil {
		return nil, err
	}
	if ok {
		if index, err = decodeInt(it); err != nil {
			return nil, err
		}
	}
	return dtype.Dictionary{Index: index, Value: value}, nil
}

// encodeKeyValues writes a KeyValue table per pair and returns where each one
// starts. The pairs keep the order they were given in, which is what makes a
// round trip through a file give back the metadata that went into it.
func encodeKeyValues(w *fbBuilder, m dtype.Metadata) []fbOffset {
	if len(m) == 0 {
		return nil
	}
	offs := make([]fbOffset, len(m))
	for i, kv := range m {
		key, value := w.str(kv.Key), w.str(kv.Value)
		w.startTable()
		w.slotOffset(fbKeyValueKey, key)
		w.slotOffset(fbKeyValueValue, value)
		offs[i] = w.endTable()
	}
	return offs
}

// dataType writes the type table of one field and returns which member of the
// union it is along with where it starts. A type that has been written already
// is pointed at again rather than written twice.
//
// Text and bytes are the one place this is not a translation of the type but a
// choice about it. There is one kuma layout for each and four Arrow ones, and
// what goes on the wire is the layout kuma holds, so a string column is
// written as a utf8 view. A reader that wants the classic layout can say so
// with dtype.LargeString, which is there for exactly this.
func (sw *schemaWriter) dataType(t dtype.DataType) (int, fbOffset, error) {
	if t == nil {
		return 0, 0, fmt.Errorf("ipc: %w: nil type", ErrType)
	}
	key := t.String()
	if ref, ok := sw.types[key]; ok {
		return ref.kind, ref.off, nil
	}
	kind, off, err := encodeType(&sw.w, t)
	if err != nil {
		return 0, 0, err
	}
	sw.types[key] = typeRef{kind: kind, off: off}
	return kind, off, nil
}

// encodeType writes one type table.
func encodeType(w *fbBuilder, t dtype.DataType) (int, fbOffset, error) {
	// The types whose table has nothing in it. The table still has to be
	// written, since a union member that is not there is a field with no type.
	if kind, ok := emptyTypes[t.Kind()]; ok {
		w.startTable()
		return kind, w.endTable(), nil
	}

	switch x := t.(type) {
	case dtype.FixedSizeBinary:
		if x.ByteWidth < 0 {
			return 0, 0, noFormat(x)
		}
		w.startTable()
		w.slotInt(fbFixedSizeBinaryWidth, x.ByteWidth, 0)
		return fbTypeFixedSizeBinary, w.endTable(), nil

	case dtype.FixedSizeList:
		if x.Len < 0 {
			return 0, 0, noFormat(x)
		}
		w.startTable()
		w.slotInt(fbFixedSizeListLen, x.Len, 0)
		return fbTypeFixedSizeList, w.endTable(), nil

	case dtype.Time32:
		// Seconds and milliseconds are the only units that fit in 32 bits, so
		// the other two are a literal somebody wrote rather than a type.
		if x.Unit != dtype.Second && x.Unit != dtype.Millisecond {
			return 0, 0, noFormat(x)
		}
		off, err := encodeTime(w, x.Unit, 32)
		return fbTypeTime, off, err

	case dtype.Time64:
		if x.Unit != dtype.Microsecond && x.Unit != dtype.Nanosecond {
			return 0, 0, noFormat(x)
		}
		off, err := encodeTime(w, x.Unit, 64)
		return fbTypeTime, off, err

	case dtype.Timestamp:
		unit, err := fbUnit(x.Unit, x)
		if err != nil {
			return 0, 0, err
		}
		// A timestamp with no zone is naive local time and is a different type
		// from one in UTC, so the empty zone is written as an absent field
		// rather than as an empty string, which is what every other
		// implementation does and what a reader compares against.
		zone := fbOffset(0)
		if x.Zone != "" {
			zone = w.str(x.Zone)
		}
		w.startTable()
		w.slotInt(fbTimestampUnit, unit, fbSecond)
		w.slotOffset(fbTimestampZone, zone)
		return fbTypeTimestamp, w.endTable(), nil

	case dtype.Duration:
		unit, err := fbUnit(x.Unit, x)
		if err != nil {
			return 0, 0, err
		}
		w.startTable()
		w.slotInt(fbDurationUnit, unit, fbMillisecond)
		return fbTypeDuration, w.endTable(), nil

	case dtype.Interval:
		var unit int16
		switch x.Unit {
		case dtype.YearMonth:
			unit = fbYearMonth
		case dtype.DayTime:
			unit = fbDayTime
		case dtype.MonthDayNano:
			unit = fbMonthDayNano
		default:
			return 0, 0, noFormat(x)
		}
		w.startTable()
		w.slotInt(fbIntervalUnit, unit, fbYearMonth)
		return fbTypeInterval, w.endTable(), nil

	case dtype.Decimal128:
		return fbTypeDecimal, encodeDecimal(w, x.Precision, x.Scale, 128), nil

	case dtype.Decimal256:
		return fbTypeDecimal, encodeDecimal(w, x.Precision, x.Scale, 256), nil

	case dtype.Map:
		// Keys sorted is left at false. It is a promise about the values rather
		// than a part of the type, and kuma does not sort the keys of a map, so
		// saying that they are sorted would be a claim about data this package
		// has not looked at.
		w.startTable()
		return fbTypeMap, w.endTable(), nil
	}

	// The two dates, which are one Arrow type with a unit rather than two
	// types. The unit is written even for milliseconds, where it is the
	// default, because a date whose unit is missing is a date64 and reading a
	// date32 as one would be off by a factor of a hundred million.
	if k := t.Kind(); k == dtype.Date32Kind || k == dtype.Date64Kind {
		unit := fbDay
		if k == dtype.Date64Kind {
			unit = fbDateMillisecond
		}
		w.startTable()
		w.slotInt(fbDateUnit, unit, -1)
		return fbTypeDate, w.endTable(), nil
	}

	// The integers and the two floats, which are the same table with different
	// numbers in it.
	if bits, ok := dtype.Bits(t); ok && dtype.IsInteger(t) {
		w.startTable()
		w.slotInt(fbIntWidth, int32(bits), 0)
		w.slotBool(fbIntSigned, dtype.IsSigned(t))
		return fbTypeInt, w.endTable(), nil
	}
	if k := t.Kind(); k == dtype.Float32Kind || k == dtype.Float64Kind {
		precision := fbSingle
		if k == dtype.Float64Kind {
			precision = fbDouble
		}
		w.startTable()
		w.slotInt(fbFloatPrecision, precision, fbHalf)
		return fbTypeFloat, w.endTable(), nil
	}

	return 0, 0, noFormat(t)
}

// emptyTypes are the kinds whose Arrow type table carries no parameters, so
// that writing one is the union number and nothing else. The layout choice for
// text and bytes is in the comment on dataType.
var emptyTypes = map[dtype.Kind]int{
	dtype.NullKind:        fbTypeNull,
	dtype.BoolKind:        fbTypeBool,
	dtype.StringKind:      fbTypeUtf8View,
	dtype.BinaryKind:      fbTypeBinaryView,
	dtype.LargeStringKind: fbTypeLargeUtf8,
	dtype.LargeBinaryKind: fbTypeLargeBinary,
	dtype.ListKind:        fbTypeList,
	dtype.LargeListKind:   fbTypeLargeList,
	dtype.StructKind:      fbTypeStruct,
}

// encodeTime writes a Time table. The unit and the width have to agree, and
// the caller has already checked that they do.
func encodeTime(w *fbBuilder, u dtype.TimeUnit, bits int32) (fbOffset, error) {
	unit, err := fbUnit(u, dtype.Time32{Unit: u})
	if err != nil {
		return 0, err
	}
	w.startTable()
	w.slotInt(fbTimeUnit, unit, fbMillisecond)
	w.slotInt(fbTimeWidth, bits, 32)
	return w.endTable(), nil
}

// encodeDecimal writes a Decimal table. The bit width is always written, even
// at the default of 128, because a decimal is the one type where guessing
// wrong changes how many bytes a value is.
func encodeDecimal(w *fbBuilder, precision, scale, bits int32) fbOffset {
	w.startTable()
	w.slotInt(fbDecimalPrecision, precision, 0)
	w.slotInt(fbDecimalScale, scale, 0)
	w.slotInt(fbDecimalWidth, bits, 0)
	return w.endTable()
}

// fbUnit is the Arrow number for a kuma time unit. The two enumerations happen
// to count in the same order, and this converts rather than casts so that they
// are free to stop doing so.
func fbUnit(u dtype.TimeUnit, t dtype.DataType) (int16, error) {
	switch u {
	case dtype.Second:
		return fbSecond, nil
	case dtype.Millisecond:
		return fbMillisecond, nil
	case dtype.Microsecond:
		return fbMicrosecond, nil
	case dtype.Nanosecond:
		return fbNanosecond, nil
	}
	return 0, noFormat(t)
}

// kumaUnit is the inverse of fbUnit. A number that is not one of the four is
// a producer writing a unit that does not exist.
func kumaUnit(u int16) (dtype.TimeUnit, error) {
	switch u {
	case fbSecond:
		return dtype.Second, nil
	case fbMillisecond:
		return dtype.Millisecond, nil
	case fbMicrosecond:
		return dtype.Microsecond, nil
	case fbNanosecond:
		return dtype.Nanosecond, nil
	}
	return 0, fmt.Errorf("ipc: %w: time unit %d, want 0 to 3", ErrMessage, u)
}

// decodeType reads the type union of a field.
//
// The children have already been read, since a list has to know its element
// type before it can be a list of anything. The name is only used in errors,
// where it is the whole point: a schema of two hundred columns that stops on
// one of them has to say which one.
func decodeType(t fbTable, name string, children []dtype.Field) (dtype.DataType, error) {
	kind, err := t.uint8(fbFieldTypeType, fbTypeNone)
	if err != nil {
		return nil, err
	}
	table, ok, err := t.table(fbFieldType)
	if err != nil {
		return nil, err
	}
	if !ok || kind == fbTypeNone {
		return nil, fmt.Errorf("ipc: %w: field %q has no type", ErrMessage, name)
	}

	switch int(kind) {
	case fbTypeNull:
		return dtype.Null, nil
	case fbTypeBool:
		return dtype.Bool, nil
	case fbTypeInt:
		return decodeInt(table)
	case fbTypeFloat:
		return decodeFloat(table)

	// Every text layout is one kuma type and every byte layout is another, the
	// same collapse Type makes for the format strings that name them.
	case fbTypeUtf8, fbTypeLargeUtf8, fbTypeUtf8View:
		return dtype.String, nil
	case fbTypeBinary, fbTypeLargeBinary, fbTypeBinaryView:
		return dtype.Binary, nil

	case fbTypeFixedSizeBinary:
		width, err := table.integer(fbFixedSizeBinaryWidth, int32(0))
		return dtype.FixedSizeBinary{ByteWidth: width}, err
	case fbTypeDecimal:
		return decodeDecimal(table)
	case fbTypeDate:
		return decodeDate(table)
	case fbTypeTime:
		return decodeTime(table)
	case fbTypeTimestamp:
		return decodeTimestamp(table)
	case fbTypeDuration:
		return decodeDuration(table)
	case fbTypeInterval:
		return decodeInterval(table)

	case fbTypeList, fbTypeLargeList:
		elem, err := onlyChild(typeNames[kind], children)
		if err != nil {
			return nil, err
		}
		if int(kind) == fbTypeLargeList {
			return dtype.LargeList{Elem: elem.Type}, nil
		}
		return dtype.List{Elem: elem.Type}, nil

	case fbTypeFixedSizeList:
		size, err := table.integer(fbFixedSizeListLen, int32(0))
		if err != nil {
			return nil, err
		}
		elem, err := onlyChild(typeNames[kind], children)
		if err != nil {
			return nil, err
		}
		return dtype.FixedSizeList{Elem: elem.Type, Len: size}, nil

	case fbTypeStruct:
		return dtype.Struct{Fields: children}, nil
	case fbTypeMap:
		return mapType(typeNames[kind], children)
	}

	if int(kind) < len(typeNames) {
		return nil, fmt.Errorf("ipc: %w: field %q is %s, which has no kuma type",
			ErrFormat, name, typeNames[kind])
	}
	return nil, fmt.Errorf("ipc: %w: field %q has type number %d, which is not one Arrow defines",
		ErrMessage, name, kind)
}

// decodeInt reads an Int table, which is the type of an integer column and
// also the type of the indices of a dictionary.
func decodeInt(t fbTable) (dtype.DataType, error) {
	bits, err := t.integer(fbIntWidth, int32(0))
	if err != nil {
		return nil, err
	}
	signed, err := t.boolean(fbIntSigned, false)
	if err != nil {
		return nil, err
	}
	switch bits {
	case 8:
		return pick(signed, dtype.Int8, dtype.Uint8), nil
	case 16:
		return pick(signed, dtype.Int16, dtype.Uint16), nil
	case 32:
		return pick(signed, dtype.Int32, dtype.Uint32), nil
	case 64:
		return pick(signed, dtype.Int64, dtype.Uint64), nil
	}
	return nil, fmt.Errorf("ipc: %w: an integer of %d bits, want 8, 16, 32 or 64", ErrMessage, bits)
}

// pick is the signed or the unsigned type of one width.
func pick(signed bool, s, u dtype.DataType) dtype.DataType {
	if signed {
		return s
	}
	return u
}

// decodeFloat reads a FloatingPoint table. Half precision is a real Arrow type
// and kuma has no equivalent, so it is named rather than refused as a number.
func decodeFloat(t fbTable) (dtype.DataType, error) {
	precision, err := t.integer(fbFloatPrecision, fbHalf)
	if err != nil {
		return nil, err
	}
	switch precision {
	case fbHalf:
		return nil, fmt.Errorf("ipc: %w: float16 has no kuma type", ErrFormat)
	case fbSingle:
		return dtype.Float32, nil
	case fbDouble:
		return dtype.Float64, nil
	}
	return nil, fmt.Errorf("ipc: %w: floating point precision %d, want 0, 1 or 2",
		ErrMessage, precision)
}

// decodeDecimal reads a Decimal table. The width is 128 when it is not there,
// which is what the schema itself says the default is.
func decodeDecimal(t fbTable) (dtype.DataType, error) {
	precision, err := t.integer(fbDecimalPrecision, int32(0))
	if err != nil {
		return nil, err
	}
	scale, err := t.integer(fbDecimalScale, int32(0))
	if err != nil {
		return nil, err
	}
	bits, err := t.integer(fbDecimalWidth, int32(128))
	if err != nil {
		return nil, err
	}
	switch bits {
	case 128:
		return dtype.Decimal128{Precision: precision, Scale: scale}, nil
	case 256:
		return dtype.Decimal256{Precision: precision, Scale: scale}, nil
	}
	return nil, fmt.Errorf("ipc: %w: a decimal of %d bits, want 128 or 256", ErrMessage, bits)
}

// decodeDate reads a Date table. The unit is milliseconds when it is not
// there, which is the wider of the two and the one nothing should be writing.
func decodeDate(t fbTable) (dtype.DataType, error) {
	unit, err := t.integer(fbDateUnit, fbDateMillisecond)
	if err != nil {
		return nil, err
	}
	switch unit {
	case fbDay:
		return dtype.Date32, nil
	case fbDateMillisecond:
		return dtype.Date64, nil
	}
	return nil, fmt.Errorf("ipc: %w: date unit %d, want 0 or 1", ErrMessage, unit)
}

// decodeTime reads a Time table. The width and the unit have to agree, since
// microseconds do not fit in 32 bits and seconds in 64 bits is a type nothing
// writes.
func decodeTime(t fbTable) (dtype.DataType, error) {
	raw, err := t.integer(fbTimeUnit, fbMillisecond)
	if err != nil {
		return nil, err
	}
	unit, err := kumaUnit(raw)
	if err != nil {
		return nil, err
	}
	bits, err := t.integer(fbTimeWidth, int32(32))
	if err != nil {
		return nil, err
	}
	switch bits {
	case 32:
		return dtype.Time32{Unit: unit}, nil
	case 64:
		return dtype.Time64{Unit: unit}, nil
	}
	return nil, fmt.Errorf("ipc: %w: a time of %d bits, want 32 or 64", ErrMessage, bits)
}

// decodeTimestamp reads a Timestamp table. An absent zone is naive local time
// and stays the empty string, which is a different type from UTC.
func decodeTimestamp(t fbTable) (dtype.DataType, error) {
	raw, err := t.integer(fbTimestampUnit, fbSecond)
	if err != nil {
		return nil, err
	}
	unit, err := kumaUnit(raw)
	if err != nil {
		return nil, err
	}
	zone, err := t.str(fbTimestampZone)
	if err != nil {
		return nil, err
	}
	return dtype.Timestamp{Unit: unit, Zone: zone}, nil
}

// decodeDuration reads a Duration table.
func decodeDuration(t fbTable) (dtype.DataType, error) {
	raw, err := t.integer(fbDurationUnit, fbMillisecond)
	if err != nil {
		return nil, err
	}
	unit, err := kumaUnit(raw)
	if err != nil {
		return nil, err
	}
	return dtype.Duration{Unit: unit}, nil
}

// decodeInterval reads an Interval table.
func decodeInterval(t fbTable) (dtype.DataType, error) {
	unit, err := t.integer(fbIntervalUnit, fbYearMonth)
	if err != nil {
		return nil, err
	}
	switch unit {
	case fbYearMonth:
		return dtype.Interval{Unit: dtype.YearMonth}, nil
	case fbDayTime:
		return dtype.Interval{Unit: dtype.DayTime}, nil
	case fbMonthDayNano:
		return dtype.Interval{Unit: dtype.MonthDayNano}, nil
	}
	return nil, fmt.Errorf("ipc: %w: interval unit %d, want 0, 1 or 2", ErrMessage, unit)
}

// decodeKeyValues reads a vector of KeyValue tables.
func decodeKeyValues(t fbTable, id int) (dtype.Metadata, error) {
	vec, ok, err := t.vector(id)
	if err != nil || !ok {
		return nil, err
	}
	m := make(dtype.Metadata, vec.len())
	for i := range vec.len() {
		kv, err := vec.table(i)
		if err != nil {
			return nil, err
		}
		if m[i].Key, err = kv.str(fbKeyValueKey); err != nil {
			return nil, err
		}
		if m[i].Value, err = kv.str(fbKeyValueValue); err != nil {
			return nil, err
		}
	}
	return m, nil
}
