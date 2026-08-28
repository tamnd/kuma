package parquet

import (
	"strconv"
	"strings"
	"testing"
)

// TestNames checks the name of every value of every enumeration, and of the
// values that are not one.
//
// The names are what a schema prints as, so they are checked against the names
// the format itself uses, lowercased. The number that is not a name matters as
// much: a file may hold one this package has never heard of, and printing it as
// a number is what lets an error message say which column was refused and why.
func TestNames(t *testing.T) {
	tests := []struct {
		name string
		got  string
		want string
	}{
		{"the absent type", NoType.String(), "none"},
		{"a boolean", Boolean.String(), "boolean"},
		{"an int32", Int32.String(), "int32"},
		{"an int64", Int64.String(), "int64"},
		{"an int96", Int96.String(), "int96"},
		{"a float", Float.String(), "float"},
		{"a double", Double.String(), "double"},
		{"a byte array", ByteArray.String(), "byte_array"},
		{"a fixed byte array", FixedLenByteArray.String(), "fixed_len_byte_array"},

		{"the absent repetition", NoRepetition.String(), "none"},
		{"required", Required.String(), "required"},
		{"optional", Optional.String(), "optional"},
		{"repeated", Repeated.String(), "repeated"},

		{"the absent encoding", NoEncoding.String(), "none"},
		{"plain", Plain.String(), "plain"},
		{"the old dictionary encoding", PlainDictionary.String(), "plain_dictionary"},
		{"run length", RLE.String(), "rle"},
		{"bit packed", BitPacked.String(), "bit_packed"},
		{"delta binary packed", DeltaBinaryPacked.String(), "delta_binary_packed"},
		{"delta length byte array", DeltaLengthByteArray.String(), "delta_length_byte_array"},
		{"delta byte array", DeltaByteArray.String(), "delta_byte_array"},
		{"the dictionary encoding", RLEDictionary.String(), "rle_dictionary"},
		{"byte stream split", ByteStreamSplit.String(), "byte_stream_split"},

		{"a data page", DataPage.String(), "data_page"},
		{"an index page", IndexPage.String(), "index_page"},
		{"a dictionary page", DictionaryPage.String(), "dictionary_page"},
		{"a second version data page", DataPageV2.String(), "data_page_v2"},

		{"no compression", Uncompressed.String(), "uncompressed"},
		{"snappy", Snappy.String(), "snappy"},
		{"gzip", Gzip.String(), "gzip"},
		{"lzo", LZO.String(), "lzo"},
		{"brotli", Brotli.String(), "brotli"},
		{"lz4", LZ4.String(), "lz4"},
		{"zstd", Zstd.String(), "zstd"},
		{"raw lz4", LZ4Raw.String(), "lz4_raw"},

		{"the absent converted type", NoConverted.String(), "none"},
		{"converted utf8", ConvertedUTF8.String(), "utf8"},
		{"converted decimal", ConvertedDecimal.String(), "decimal"},
		{"converted timestamp millis", ConvertedTimestampMillis.String(), "timestamp_millis"},
		{"converted uint64", ConvertedUint64.String(), "uint_64"},
		{"converted interval", ConvertedInterval.String(), "interval"},

		{"the absent logical type", NoLogical.String(), "none"},
		{"a string", StringLogical.String(), "string"},
		{"a decimal", DecimalLogical.String(), "decimal"},
		{"a timestamp", TimestampLogical.String(), "timestamp"},
		{"a uuid", UUIDLogical.String(), "uuid"},
		{"a float16", Float16Logical.String(), "float16"},

		{"the absent unit", NoUnit.String(), "none"},
		{"millis", Millis.String(), "millis"},
		{"micros", Micros.String(), "micros"},
		{"nanos", Nanos.String(), "nanos"},

		{"the absent column order", UndefinedOrder.String(), "none"},
		{"the order the format defines", TypeDefinedOrder.String(), "type_defined_order"},

		{"pages in no order", Unordered.String(), "unordered"},
		{"pages running up", Ascending.String(), "ascending"},
		{"pages running down", Descending.String(), "descending"},

		{"the stop type", thriftStop.String(), "stop"},
		{"a thrift struct", thriftStruct.String(), "struct"},
		{"a thrift map", thriftMap.String(), "map"},
	}

	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("%s prints as %q, want %q", tt.name, tt.got, tt.want)
		}
	}
}

// TestUnknownNames checks the values no version of the format has defined. They
// print as their number, since a reader that outlives the file it is reading
// has to be able to say what it did not understand.
func TestUnknownNames(t *testing.T) {
	tests := []struct {
		got  string
		want string
	}{
		{Type(-2).String(), "type -2"},
		{Type(99).String(), "type 99"},
		{Repetition(-2).String(), "repetition -2"},
		{Repetition(99).String(), "repetition 99"},
		{Encoding(-2).String(), "encoding -2"},
		{Encoding(1).String(), "encoding 1"},
		{Encoding(99).String(), "encoding 99"},
		{PageKind(-1).String(), "page type -1"},
		{PageKind(99).String(), "page type 99"},
		{Codec(-1).String(), "codec -1"},
		{Codec(99).String(), "codec 99"},
		{ConvertedType(-2).String(), "converted type -2"},
		{ConvertedType(99).String(), "converted type 99"},
		{LogicalKind(-1).String(), "logical type -1"},
		{LogicalKind(99).String(), "logical type 99"},
		{TimeUnit(-1).String(), "unit -1"},
		{TimeUnit(99).String(), "unit 99"},
		{ColumnOrder(-1).String(), "column order -1"},
		{ColumnOrder(99).String(), "column order 99"},
		{BoundaryOrder(-1).String(), "boundary order -1"},
		{BoundaryOrder(99).String(), "boundary order 99"},
		{thriftType(99).String(), "type 99"},
	}

	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("an unknown value prints as %q, want %q", tt.got, tt.want)
		}
	}
}

// TestEncodingOne is the one hole in the encodings. The format numbered them
// from zero and never used one, so a name table indexed by the number has a gap
// in the middle of it, and a gap read as a name is an empty string.
func TestEncodingOne(t *testing.T) {
	if got := Encoding(1).String(); got == "" || !strings.Contains(got, "1") {
		t.Fatalf("the encoding that does not exist prints as %q, want its number", got)
	}
}

// TestNamesAreNames checks that no name table has a hole in it where a defined
// value is, which is what happens when a value is added to the constants and
// not to the table behind them.
func TestNamesAreNames(t *testing.T) {
	tables := []struct {
		what  string
		names []string
		from  int
	}{
		{"type", typeNames[:], int(Boolean)},
		{"repetition", repetitionNames[:], int(Required)},
		{"page type", pageNames[:], int(DataPage)},
		{"codec", codecNames[:], int(Uncompressed)},
		{"converted type", convertedNames[:], int(ConvertedUTF8)},
		{"logical type", logicalNames[:], int(NoLogical)},
		{"unit", unitNames[:], int(NoUnit)},
		{"column order", orderNames[:], int(UndefinedOrder)},
		{"boundary order", boundaryNames[:], int(Unordered)},
		{"thrift type", thriftNames[:], int(thriftStop)},
	}

	for _, tt := range tables {
		for i := tt.from; i < len(tt.names); i++ {
			if tt.names[i] == "" {
				t.Errorf("%s %d has no name", tt.what, i)
			}
			if strings.TrimSpace(tt.names[i]) != tt.names[i] {
				t.Errorf("%s %d is named %q, want it trimmed", tt.what, i, tt.names[i])
			}
			if _, err := strconv.Atoi(tt.names[i]); err == nil {
				t.Errorf("%s %d is named %q, which is a number", tt.what, i, tt.names[i])
			}
		}
	}
}
