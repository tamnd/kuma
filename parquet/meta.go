package parquet

// What a parquet file says about itself.
//
// The footer is one Thrift structure holding the schema, a row group per
// horizontal slice of the file, and a column chunk per column of each of those.
// A column chunk says where its pages are, how many values it has, how it was
// encoded and compressed, and often what its smallest and largest values are,
// which between them are everything a scan needs to decide what not to read.
//
// The structures here are the ones the format defines, with the names Go would
// have given them and without the fields nothing has needed yet. A field the
// file did not write reads as the absent value of its type rather than as a
// zero, because a column with no statistics and a column whose values are all
// zero are not the same column.

// Metadata is the footer of a parquet file.
type Metadata struct {
	// Version is the format version the writer wrote, which is 1 for every
	// file in the wild and 2 for files using the newer page header.
	Version int32

	// Nodes is the schema as a flattened tree, the root first, each node
	// followed by its children. A node with no children is a column and a node
	// with children is a group. The format calls this the schema, and it is
	// called something else here because the schema of a file is a tree of
	// kuma types and this is the bytes it is built out of. Tree, Columns and
	// Schema are the three ways to read it.
	Nodes []SchemaElement

	// NumRows is how many rows the whole file holds.
	NumRows int64

	// RowGroups are the horizontal slices of the file, in the order they were
	// written, which is the order the rows are in.
	RowGroups []RowGroup

	// KeyValue is what the writer attached to the file. Arrow puts its own
	// schema in here under the key ARROW:schema, which is how a file written
	// from Arrow remembers a type parquet has no way to write down.
	KeyValue []KeyValue

	// CreatedBy is the writer's name and version, as free text. It is worth
	// keeping because the format has had bugs that are identified by which
	// writer produced the file and nothing else.
	CreatedBy string

	// Orders is how the values of each column compare, one entry per leaf of
	// the schema in the order Columns hands them back. It is empty in a file
	// that did not say, and a file that did not say has left the bounds on its
	// chunks meaning nothing, which is why Column carries the order along and
	// ReadBounds asks for it.
	Orders []ColumnOrder
}

// SchemaElement is one node of the schema tree.
//
// The tree is written flat: this node, then its children, then their children.
// A node with children is a group and has no physical type, and a node with
// none is a column and has one. The root is the only node with no repetition.
type SchemaElement struct {
	Name string

	// Type is how the values are written down, or NoType for a group.
	Type Type

	// TypeLength is how many bytes a value takes, for FixedLenByteArray and
	// for nothing else.
	TypeLength int32

	// Repetition says whether the field may be missing or may repeat.
	Repetition Repetition

	// NumChildren is how many of the nodes that follow belong to this one.
	NumChildren int32

	// Converted is what the physical type means, in the older of the two ways
	// parquet says so, or NoConverted when the file did not say.
	Converted ConvertedType

	// Logical is the same thing in the newer way, which carries the parameters
	// the converted type could not. Its kind is NoLogical when the file did
	// not say.
	Logical LogicalType

	// Scale and Precision belong to a decimal and are here rather than on the
	// logical type because a file that only writes converted types has nowhere
	// else to put them.
	Scale     int32
	Precision int32

	// FieldID is the identifier a schema language gave the field, or zero when
	// the file did not say. Nothing in parquet uses it.
	FieldID int32
}

// LogicalType is what a physical type means.
//
// It is a union in the format, so which of the fields below mean anything
// depends on Kind. A DecimalLogical has Scale and Precision, a TimeLogical and
// a TimestampLogical have Unit and UTC, an IntegerLogical has BitWidth and
// Signed, and the rest are the kind and nothing else.
type LogicalType struct {
	Kind LogicalKind

	Scale     int32
	Precision int32

	// UTC says the writer knew what instant the value stands for. A timestamp
	// that is not adjusted to UTC is a wall clock reading with no zone, which
	// is a different thing from a timestamp in UTC.
	UTC  bool
	Unit TimeUnit

	BitWidth int8
	Signed   bool
}

// RowGroup is one horizontal slice of a file.
//
// Every column has a chunk in every row group, so a row group is a set of rows
// that can be read without touching the rest of the file. This is the unit a
// scan skips: a filter that no row of a group can satisfy skips the group and
// never reads a page of it.
type RowGroup struct {
	Columns []ColumnChunk

	// TotalByteSize is the size of the values before compression, and
	// TotalCompressedSize is the number of bytes on disk.
	TotalByteSize       int64
	TotalCompressedSize int64

	NumRows int64

	// FileOffset is where the group starts, or zero for the files that leave
	// it out. The offsets on the column chunks are the ones to trust.
	FileOffset int64

	// Ordinal is which group this is, counting from zero.
	Ordinal int16
}

// ColumnChunk is one column of one row group.
type ColumnChunk struct {
	// FilePath is the file the chunk lives in, which is empty for every file
	// written this decade. It is how the format allowed a single logical file
	// to be spread across several, which nothing does any more.
	FilePath string

	// FileOffset is where the chunk's metadata is, in the files that repeat it
	// next to the data. It is zero in most files.
	FileOffset int64

	// Meta is the chunk itself: where its pages are and what is in them.
	Meta ColumnMeta

	// ColumnIndex and OffsetIndex are two more structures at the end of the
	// file, holding the smallest and largest value of every page and where
	// every page starts. They are what makes skipping work at page granularity
	// rather than at row group granularity. Both offsets are zero when the
	// writer did not produce them.
	ColumnIndexOffset int64
	ColumnIndexLength int32
	OffsetIndexOffset int64
	OffsetIndexLength int32
}

// ColumnMeta is where a column chunk's pages are and what is in them.
type ColumnMeta struct {
	Type Type

	// Encodings are every encoding used by any page of the chunk, which is
	// more than one whenever there is a dictionary, since the dictionary page
	// and the data pages are not encoded the same way.
	Encodings []Encoding

	// Path is the column's name, one element per level of the schema, so a
	// nested field is address.city rather than city.
	Path []string

	Codec Codec

	// NumValues counts the values in the chunk, which for a column inside a
	// repeated field is more than the number of rows.
	NumValues int64

	TotalUncompressedSize int64
	TotalCompressedSize   int64

	// DataPageOffset is where the first data page starts, and
	// DictionaryPageOffset is where the dictionary page starts, or zero when
	// the chunk has none. A chunk with a dictionary starts at the dictionary
	// page, which comes before the data pages.
	DataPageOffset       int64
	DictionaryPageOffset int64

	// IndexPageOffset is a feature the format defined and no writer produces.
	IndexPageOffset int64

	// Stats is what the writer said about the values, which is the other half
	// of skipping a row group.
	Stats Statistics

	// BloomFilterOffset is where the chunk's bloom filter is, or zero when it
	// has none. A bloom filter answers whether a value is definitely not in
	// the chunk, which is what makes an equality filter skip a group whose
	// range happens to contain the value.
	BloomFilterOffset int64
	BloomFilterLength int32
}

// Statistics is what a writer said about the values of a column chunk.
//
// The values are the raw bytes of the physical type, so an Int32 column's
// smallest value is four bytes and a ByteArray column's is the string itself.
// Reading them means knowing the type, which is why they are not decoded here.
//
// MinValue and MaxValue are the ones to use. Min and Max are the same idea from
// before the format pinned down how values are ordered, and are only safe to
// read for the types whose ordering nobody ever disagreed about. ReadBounds is
// what applies that rule and hands back the values decoded.
//
// A bound the file did not write is nil and one it wrote empty is not, which is
// how a column of strings whose smallest value is the empty string is told from
// a column that said nothing about itself.
type Statistics struct {
	MinValue []byte
	MaxValue []byte

	// MinExact and MaxExact say the bounds are values that are in the chunk
	// rather than bounds around them. A writer that truncates a long string to
	// keep the footer small says so this way.
	MinExact bool
	MaxExact bool

	Min []byte
	Max []byte

	// NullCount is how many values of the chunk are missing, and
	// DistinctCount is how many different ones there are. Both are only
	// meaningful when the flag next to them is set, since a file that says
	// nothing and a file that says zero are different files.
	NullCount     int64
	HasNullCount  bool
	DistinctCount int64
	HasDistinct   bool
}

// KeyValue is one entry of the metadata a writer attached to a file.
type KeyValue struct {
	Key   string
	Value string
}

// The field numbers below are the ones in parquet.thrift. They are what the
// format is: a reader that has the numbers right reads a file written by
// anything, and one that has them wrong reads nothing at all.

func (m *Metadata) read(r *reader) error {
	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			m.Version, err = r.int32(t)
		case 2:
			m.Nodes, err = structs(r, t, (*SchemaElement).read)
		case 3:
			m.NumRows, err = r.integer(t)
		case 4:
			m.RowGroups, err = structs(r, t, (*RowGroup).read)
		case 5:
			m.KeyValue, err = structs(r, t, (*KeyValue).read)
		case 6:
			m.CreatedBy, err = r.text(t)
		case 7:
			m.Orders, err = structs(r, t, (*ColumnOrder).read)
		default:
			err = r.skip(t)
		}
		return err
	})
}

// read fills in the order from the union in the file.
//
// One member is defined and it is the empty struct that says the order is the
// one the format defines for the type. Anything else is a member written by
// something newer than this, and a reader that took it for the one it knows
// would be acting on an order it cannot see, so it stays undefined.
func (o *ColumnOrder) read(r *reader) error {
	return r.fields(func(id int16, t thriftType) error {
		if id == 1 {
			*o = TypeDefinedOrder
		}
		return r.skip(t)
	})
}

func (e *SchemaElement) read(r *reader) error {
	*e = SchemaElement{Type: NoType, Repetition: NoRepetition, Converted: NoConverted}

	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			var v int32
			if v, err = r.int32(t); err == nil {
				e.Type = Type(v)
			}
		case 2:
			e.TypeLength, err = r.int32(t)
		case 3:
			var v int32
			if v, err = r.int32(t); err == nil {
				e.Repetition = Repetition(v)
			}
		case 4:
			e.Name, err = r.text(t)
		case 5:
			e.NumChildren, err = r.int32(t)
		case 6:
			var v int32
			if v, err = r.int32(t); err == nil {
				e.Converted = ConvertedType(v)
			}
		case 7:
			e.Scale, err = r.int32(t)
		case 8:
			e.Precision, err = r.int32(t)
		case 9:
			e.FieldID, err = r.int32(t)
		case 10:
			err = e.Logical.read(r, t)
		default:
			err = r.skip(t)
		}
		return err
	})
}

// read fills in the logical type from the union in the file.
//
// A union is a struct with one field set, and which field it is says what the
// type is. The ones with parameters read them and the ones without still have a
// struct of their own to read past, which is what makes an empty struct here
// two bytes rather than none.
func (l *LogicalType) read(r *reader, t thriftType) error {
	if t != thriftStruct {
		return r.skip(t)
	}

	return r.fields(func(id int16, t thriftType) error {
		switch id {
		case 1:
			l.Kind = StringLogical
		case 2:
			l.Kind = MapLogical
		case 3:
			l.Kind = ListLogical
		case 4:
			l.Kind = EnumLogical
		case 5:
			l.Kind = DecimalLogical
			return l.readDecimal(r, t)
		case 6:
			l.Kind = DateLogical
		case 7:
			l.Kind = TimeLogical
			return l.readTime(r, t)
		case 8:
			l.Kind = TimestampLogical
			return l.readTime(r, t)
		case 10:
			l.Kind = IntegerLogical
			return l.readInteger(r, t)
		case 11:
			l.Kind = UnknownLogical
		case 12:
			l.Kind = JSONLogical
		case 13:
			l.Kind = BSONLogical
		case 14:
			l.Kind = UUIDLogical
		case 15:
			l.Kind = Float16Logical
		}
		return r.skip(t)
	})
}

func (l *LogicalType) readDecimal(r *reader, t thriftType) error {
	if t != thriftStruct {
		return r.skip(t)
	}

	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			l.Scale, err = r.int32(t)
		case 2:
			l.Precision, err = r.int32(t)
		default:
			err = r.skip(t)
		}
		return err
	})
}

// readTime reads a time or a timestamp, which are the same two fields.
func (l *LogicalType) readTime(r *reader, t thriftType) error {
	if t != thriftStruct {
		return r.skip(t)
	}

	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			l.UTC, err = r.boolean(t)
		case 2:
			err = l.readUnit(r, t)
		default:
			err = r.skip(t)
		}
		return err
	})
}

// readUnit reads the unit, which is a union of three empty structs.
func (l *LogicalType) readUnit(r *reader, t thriftType) error {
	if t != thriftStruct {
		return r.skip(t)
	}

	return r.fields(func(id int16, t thriftType) error {
		switch id {
		case 1:
			l.Unit = Millis
		case 2:
			l.Unit = Micros
		case 3:
			l.Unit = Nanos
		}
		return r.skip(t)
	})
}

func (l *LogicalType) readInteger(r *reader, t thriftType) error {
	if t != thriftStruct {
		return r.skip(t)
	}

	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			l.BitWidth, err = r.int8(t)
		case 2:
			l.Signed, err = r.boolean(t)
		default:
			err = r.skip(t)
		}
		return err
	})
}

func (g *RowGroup) read(r *reader) error {
	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			g.Columns, err = structs(r, t, (*ColumnChunk).read)
		case 2:
			g.TotalByteSize, err = r.integer(t)
		case 3:
			g.NumRows, err = r.integer(t)
		case 5:
			g.FileOffset, err = r.integer(t)
		case 6:
			g.TotalCompressedSize, err = r.integer(t)
		case 7:
			g.Ordinal, err = r.int16(t)
		default:
			err = r.skip(t)
		}
		return err
	})
}

func (c *ColumnChunk) read(r *reader) error {
	c.Meta.Type = NoType

	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			c.FilePath, err = r.text(t)
		case 2:
			c.FileOffset, err = r.integer(t)
		case 3:
			err = c.Meta.read(r, t)
		case 4:
			c.OffsetIndexOffset, err = r.integer(t)
		case 5:
			c.OffsetIndexLength, err = r.int32(t)
		case 6:
			c.ColumnIndexOffset, err = r.integer(t)
		case 7:
			c.ColumnIndexLength, err = r.int32(t)
		default:
			err = r.skip(t)
		}
		return err
	})
}

func (m *ColumnMeta) read(r *reader, t thriftType) error {
	if t != thriftStruct {
		return r.skip(t)
	}

	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			var v int32
			if v, err = r.int32(t); err == nil {
				m.Type = Type(v)
			}
		case 2:
			m.Encodings, err = enums[Encoding](r, t)
		case 3:
			m.Path, err = texts(r, t)
		case 4:
			var v int32
			if v, err = r.int32(t); err == nil {
				m.Codec = Codec(v)
			}
		case 5:
			m.NumValues, err = r.integer(t)
		case 6:
			m.TotalUncompressedSize, err = r.integer(t)
		case 7:
			m.TotalCompressedSize, err = r.integer(t)
		case 9:
			m.DataPageOffset, err = r.integer(t)
		case 10:
			m.IndexPageOffset, err = r.integer(t)
		case 11:
			m.DictionaryPageOffset, err = r.integer(t)
		case 12:
			err = m.Stats.read(r, t)
		case 14:
			m.BloomFilterOffset, err = r.integer(t)
		case 15:
			m.BloomFilterLength, err = r.int32(t)
		default:
			err = r.skip(t)
		}
		return err
	})
}

func (s *Statistics) read(r *reader, t thriftType) error {
	if t != thriftStruct {
		return r.skip(t)
	}

	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			s.Max, err = r.bytes(t)
		case 2:
			s.Min, err = r.bytes(t)
		case 3:
			if s.NullCount, err = r.integer(t); err == nil {
				s.HasNullCount = true
			}
		case 4:
			if s.DistinctCount, err = r.integer(t); err == nil {
				s.HasDistinct = true
			}
		case 5:
			s.MaxValue, err = r.bytes(t)
		case 6:
			s.MinValue, err = r.bytes(t)
		case 7:
			s.MaxExact, err = r.boolean(t)
		case 8:
			s.MinExact, err = r.boolean(t)
		default:
			err = r.skip(t)
		}
		return err
	})
}

func (kv *KeyValue) read(r *reader) error {
	return r.fields(func(id int16, t thriftType) error {
		var err error
		switch id {
		case 1:
			kv.Key, err = r.text(t)
		case 2:
			kv.Value, err = r.text(t)
		default:
			err = r.skip(t)
		}
		return err
	})
}
