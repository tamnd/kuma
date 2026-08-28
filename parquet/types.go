package parquet

import "fmt"

// The types and enumerations the parquet format describes a column with.
//
// Every one of them is a number on the wire, and a file may hold a number this
// package has never heard of, since the format grows and files outlive readers.
// So none of them refuse an unknown value here. They print it as a number and
// leave refusing it to whoever tries to read the column, which is the only
// place that knows whether it mattered.
//
// Absent is a value of its own rather than a zero. A group node has no physical
// type and the root of the schema has no repetition, and both of those are
// different from being a boolean or being required, so the zero value of each
// of these is the one the format never uses.

// The names that more than one of these tables uses. A logical type and the
// converted type it replaced call the same thing the same thing, Thrift calls
// two of them the same thing again, and every enumeration here has a value that
// is not one and that prints the same way.
const (
	noName   = "none"
	mapName  = "map"
	listName = "list"
)

// Type is the physical type of a column, which is how the values are written
// down rather than what they mean. A date and a signed integer are both Int32
// and it takes the logical type to tell them apart.
type Type int32

// The physical types.
const (
	NoType Type = iota - 1
	Boolean
	Int32
	Int64
	Int96
	Float
	Double
	ByteArray
	FixedLenByteArray
)

var typeNames = [...]string{
	Boolean:           "boolean",
	Int32:             "int32",
	Int64:             "int64",
	Int96:             "int96",
	Float:             "float",
	Double:            "double",
	ByteArray:         "byte_array",
	FixedLenByteArray: "fixed_len_byte_array",
}

// String returns the name the format gives the type, lowercased.
func (t Type) String() string {
	if t == NoType {
		return noName
	}
	if t < 0 || int(t) >= len(typeNames) {
		return fmt.Sprintf("type %d", int32(t))
	}
	return typeNames[t]
}

// Repetition says whether a column may be missing and whether it may repeat. It
// is how parquet writes down nesting and nullability at once: an optional field
// is a nullable column, and a repeated one is a list.
type Repetition int32

// The repetition types.
const (
	NoRepetition Repetition = iota - 1
	Required
	Optional
	Repeated
)

var repetitionNames = [...]string{
	Required: "required",
	Optional: "optional",
	Repeated: "repeated",
}

// String returns the name the format gives the repetition, lowercased.
func (r Repetition) String() string {
	if r == NoRepetition {
		return noName
	}
	if r < 0 || int(r) >= len(repetitionNames) {
		return fmt.Sprintf("repetition %d", int32(r))
	}
	return repetitionNames[r]
}

// Encoding is how the values of a page are written down. A column chunk lists
// every encoding it used, since the dictionary page and the data pages of one
// chunk are not encoded the same way.
type Encoding int32

// The encodings. Two of them are numbered out of order because the format
// replaced them: PlainDictionary became RLEDictionary and BitPacked became RLE,
// and files written before the change still say the old thing.
//
// NoEncoding is a page that did not say how it was encoded. The format calls
// that field required, so it is a file to refuse rather than a page to guess
// at, and it needs a value of its own because zero is a real encoding.
const (
	NoEncoding           Encoding = -1
	Plain                Encoding = 0
	PlainDictionary      Encoding = 2
	RLE                  Encoding = 3
	BitPacked            Encoding = 4
	DeltaBinaryPacked    Encoding = 5
	DeltaLengthByteArray Encoding = 6
	DeltaByteArray       Encoding = 7
	RLEDictionary        Encoding = 8
	ByteStreamSplit      Encoding = 9
)

var encodingNames = [...]string{
	Plain:                "plain",
	PlainDictionary:      "plain_dictionary",
	RLE:                  "rle",
	BitPacked:            "bit_packed",
	DeltaBinaryPacked:    "delta_binary_packed",
	DeltaLengthByteArray: "delta_length_byte_array",
	DeltaByteArray:       "delta_byte_array",
	RLEDictionary:        "rle_dictionary",
	ByteStreamSplit:      "byte_stream_split",
}

// String returns the name the format gives the encoding, lowercased.
func (e Encoding) String() string {
	if e == NoEncoding {
		return noName
	}
	if e < 0 || int(e) >= len(encodingNames) || encodingNames[e] == "" {
		return fmt.Sprintf("encoding %d", int32(e))
	}
	return encodingNames[e]
}

// PageKind is what a page holds. A column chunk is a run of pages, at most one
// of them a dictionary and the rest of them data.
type PageKind int32

// The page types. The two data pages are two versions of the same thing: the
// second one moved the levels out of the compressed part of the page so that a
// reader can work out how many rows a page holds without decompressing it.
const (
	DataPage PageKind = iota
	IndexPage
	DictionaryPage
	DataPageV2
)

var pageNames = [...]string{
	DataPage:       "data_page",
	IndexPage:      "index_page",
	DictionaryPage: "dictionary_page",
	DataPageV2:     "data_page_v2",
}

// String returns the name the format gives the page type, lowercased.
func (k PageKind) String() string {
	if k < 0 || int(k) >= len(pageNames) {
		return fmt.Sprintf("page type %d", int32(k))
	}
	return pageNames[k]
}

// Codec is how the pages of a column chunk are compressed. It is per chunk
// rather than per file, so one file may hold a snappy column and a zstd one.
type Codec int32

// The compression codecs.
const (
	Uncompressed Codec = iota
	Snappy
	Gzip
	LZO
	Brotli
	LZ4
	Zstd
	LZ4Raw
)

var codecNames = [...]string{
	Uncompressed: "uncompressed",
	Snappy:       "snappy",
	Gzip:         "gzip",
	LZO:          "lzo",
	Brotli:       "brotli",
	LZ4:          "lz4",
	Zstd:         "zstd",
	LZ4Raw:       "lz4_raw",
}

// String returns the name the format gives the codec, lowercased.
func (c Codec) String() string {
	if c < 0 || int(c) >= len(codecNames) {
		return fmt.Sprintf("codec %d", int32(c))
	}
	return codecNames[c]
}

// ConvertedType is what a physical type means, as parquet wrote it before
// logical types existed. A writer that wants to be read by everything writes
// both, so this is usually the same thing the logical type says and is the only
// thing an old file says at all.
type ConvertedType int32

// The converted types.
const (
	NoConverted ConvertedType = iota - 1
	ConvertedUTF8
	ConvertedMap
	ConvertedMapKeyValue
	ConvertedList
	ConvertedEnum
	ConvertedDecimal
	ConvertedDate
	ConvertedTimeMillis
	ConvertedTimeMicros
	ConvertedTimestampMillis
	ConvertedTimestampMicros
	ConvertedUint8
	ConvertedUint16
	ConvertedUint32
	ConvertedUint64
	ConvertedInt8
	ConvertedInt16
	ConvertedInt32
	ConvertedInt64
	ConvertedJSON
	ConvertedBSON
	ConvertedInterval
)

var convertedNames = [...]string{
	ConvertedUTF8:            "utf8",
	ConvertedMap:             mapName,
	ConvertedMapKeyValue:     "map_key_value",
	ConvertedList:            listName,
	ConvertedEnum:            "enum",
	ConvertedDecimal:         "decimal",
	ConvertedDate:            "date",
	ConvertedTimeMillis:      "time_millis",
	ConvertedTimeMicros:      "time_micros",
	ConvertedTimestampMillis: "timestamp_millis",
	ConvertedTimestampMicros: "timestamp_micros",
	ConvertedUint8:           "uint_8",
	ConvertedUint16:          "uint_16",
	ConvertedUint32:          "uint_32",
	ConvertedUint64:          "uint_64",
	ConvertedInt8:            "int_8",
	ConvertedInt16:           "int_16",
	ConvertedInt32:           "int_32",
	ConvertedInt64:           "int_64",
	ConvertedJSON:            "json",
	ConvertedBSON:            "bson",
	ConvertedInterval:        "interval",
}

// String returns the name the format gives the converted type, lowercased.
func (c ConvertedType) String() string {
	if c == NoConverted {
		return noName
	}
	if c < 0 || int(c) >= len(convertedNames) {
		return fmt.Sprintf("converted type %d", int32(c))
	}
	return convertedNames[c]
}

// LogicalKind is what a physical type means, as parquet writes it now. It is
// the same idea as a converted type with the parameters that one could not
// carry: a decimal says its precision here rather than in the schema element,
// and a timestamp says whether it is UTC.
type LogicalKind int32

// The logical types, in the order the union declares them. The numbers are this
// package's own rather than the format's, which are the field numbers of the
// union and have a gap in them where an interval type was reserved and never
// defined.
const (
	NoLogical LogicalKind = iota
	StringLogical
	MapLogical
	ListLogical
	EnumLogical
	DecimalLogical
	DateLogical
	TimeLogical
	TimestampLogical
	IntegerLogical
	UnknownLogical
	JSONLogical
	BSONLogical
	UUIDLogical
	Float16Logical
)

var logicalNames = [...]string{
	NoLogical:        noName,
	StringLogical:    "string",
	MapLogical:       mapName,
	ListLogical:      listName,
	EnumLogical:      "enum",
	DecimalLogical:   "decimal",
	DateLogical:      "date",
	TimeLogical:      "time",
	TimestampLogical: "timestamp",
	IntegerLogical:   "integer",
	UnknownLogical:   "unknown",
	JSONLogical:      "json",
	BSONLogical:      "bson",
	UUIDLogical:      "uuid",
	Float16Logical:   "float16",
}

// String returns the name the format gives the logical type, lowercased.
func (l LogicalKind) String() string {
	if l < 0 || int(l) >= len(logicalNames) {
		return fmt.Sprintf("logical type %d", int32(l))
	}
	return logicalNames[l]
}

// TimeUnit is how fine a time or a timestamp is. There is no second unit, so
// the coarsest a parquet timestamp gets is milliseconds.
type TimeUnit int32

// The time units.
const (
	NoUnit TimeUnit = iota
	Millis
	Micros
	Nanos
)

var unitNames = [...]string{
	NoUnit: noName,
	Millis: "millis",
	Micros: "micros",
	Nanos:  "nanos",
}

// String returns the name of the unit, lowercased.
func (u TimeUnit) String() string {
	if u < 0 || int(u) >= len(unitNames) {
		return fmt.Sprintf("unit %d", int32(u))
	}
	return unitNames[u]
}

// ColumnOrder is how the values of a column compare, which is the only thing
// that makes the smallest and largest value of a chunk mean anything.
//
// The format has one order and a way of saying it has none. A column the file
// gave TypeDefinedOrder is ordered the way the format defines for its type,
// which is signed for a signed integer, unsigned for an unsigned one, and byte
// by byte for a string. A column the file said nothing about is ordered however
// the writer felt like ordering it, and the bounds on its chunks are one
// writer's opinion rather than something a reader can act on.
type ColumnOrder int32

// The column orders. Undefined is the zero value because a file that says
// nothing is the case a reader has to handle, and because an order this package
// has never heard of is one it knows nothing about either.
const (
	UndefinedOrder ColumnOrder = iota
	TypeDefinedOrder
)

var orderNames = [...]string{
	UndefinedOrder:   noName,
	TypeDefinedOrder: "type_defined_order",
}

// String returns the name the format gives the order, lowercased.
func (o ColumnOrder) String() string {
	if o < 0 || int(o) >= len(orderNames) {
		return fmt.Sprintf("column order %d", int32(o))
	}
	return orderNames[o]
}
