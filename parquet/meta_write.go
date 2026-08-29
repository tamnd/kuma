package parquet

// Writing what a parquet file says about itself.
//
// This is the read methods in meta.go turned around, field number for field
// number, and the numbers are the whole of it. A footer written with the right
// numbers is read by anything and one written with the wrong numbers is read by
// nothing, so the two halves are kept next to each other on purpose: a field
// added to one is a field to add to the other, and a test round trips every
// footer in testdata through both.
//
// What a writer has to decide, and a reader never does, is which fields to leave
// out. Nearly everything in the format is optional, and an optional field that
// is absent reads back as the absent value of its type, so a footer that writes
// every field it has is a larger footer that says the same thing. The rule here
// is that a field is written when the file it came from had it and left out when
// it did not.
//
// Saying that in code is where the care goes, because for most of these fields
// absent and empty are two different things. A statistic of nil is a writer that
// said nothing and one of empty bytes is a writer that said the smallest value
// is the empty string. A null count carries a flag beside it because a file that
// says nothing and a file that says nought are different files. And a list that
// is nil was never written while one that is empty was written empty, which is
// not hypothetical: pyarrow writes an empty repetition histogram on every flat
// column, so a writer testing these with len rather than nil fails to give back
// the footer it was handed.

// write writes the footer.
func (m *Metadata) write(w *writer) {
	w.int32Field(1, m.Version)
	writeStructs(w, 2, m.Nodes, (*SchemaElement).write)
	w.int64Field(3, m.NumRows)
	writeStructs(w, 4, m.RowGroups, (*RowGroup).write)
	if m.KeyValue != nil {
		writeStructs(w, 5, m.KeyValue, (*KeyValue).write)
	}
	if m.CreatedBy != "" {
		w.textField(6, m.CreatedBy)
	}
	if m.Orders != nil {
		writeStructs(w, 7, m.Orders, (*ColumnOrder).write)
	}
}

// write writes the order as the union the format defines.
//
// One member exists and it is an empty struct saying the values compare the way
// the format defines for their type. An order that is not that one is a member
// written by something newer than this package, which the reader could not keep
// and this cannot put back, so it goes out as a union with nothing set. That is
// what the file would have said if its writer had not known either.
func (o *ColumnOrder) write(w *writer) {
	if *o == TypeDefinedOrder {
		w.structure(1, func() {})
	}
}

func (e *SchemaElement) write(w *writer) {
	if e.Type != NoType {
		w.int32Field(1, int32(e.Type))
	}
	if e.TypeLength != 0 {
		w.int32Field(2, e.TypeLength)
	}
	if e.Repetition != NoRepetition {
		w.int32Field(3, int32(e.Repetition))
	}
	w.textField(4, e.Name)
	if e.NumChildren != 0 {
		w.int32Field(5, e.NumChildren)
	}
	if e.Converted != NoConverted {
		w.int32Field(6, int32(e.Converted))
	}
	if e.Scale != 0 {
		w.int32Field(7, e.Scale)
	}
	if e.Precision != 0 {
		w.int32Field(8, e.Precision)
	}
	if e.FieldID != 0 {
		w.int32Field(9, e.FieldID)
	}
	e.Logical.write(w, 10)
}

// write writes the logical type as field id, or writes nothing for a node that
// has none.
//
// A union is a struct with one field set and which field it is says what the
// type is, so the kinds with parameters write them inside their member and the
// ones without still write a member with nothing in it. That is why an empty
// logical type is two bytes rather than none.
func (l *LogicalType) write(w *writer, id int16) {
	member := l.Kind.member()
	if member == 0 {
		return
	}

	w.structure(id, func() {
		w.structure(member, func() {
			switch l.Kind {
			case DecimalLogical:
				w.int32Field(1, l.Scale)
				w.int32Field(2, l.Precision)
			case TimeLogical, TimestampLogical:
				w.boolField(1, l.UTC)
				l.writeUnit(w, 2)
			case IntegerLogical:
				w.int8Field(1, l.BitWidth)
				w.boolField(2, l.Signed)
			default:
				// The rest are the kind and nothing else.
			}
		})
	})
}

// writeUnit writes the unit of a time or a timestamp as field id.
//
// It is a union of three empty structs and this package numbers the units the
// way the union numbers its members, so a unit is its own field number. That is
// worth saying out loud because it is the one place in here where two sets of
// numbers are being relied on to agree.
func (l *LogicalType) writeUnit(w *writer, id int16) {
	if l.Unit == NoUnit {
		return
	}
	w.structure(id, func() {
		w.structure(int16(l.Unit), func() {})
	})
}

// member is the field number of the union member a kind is written as, or nought
// for a kind that has no member.
//
// The gap at nine is a member the format reserved for an interval type and never
// defined, which is why this package numbers the kinds itself rather than using
// the format's numbers for both.
func (l LogicalKind) member() int16 {
	switch l {
	case StringLogical:
		return 1
	case MapLogical:
		return 2
	case ListLogical:
		return 3
	case EnumLogical:
		return 4
	case DecimalLogical:
		return 5
	case DateLogical:
		return 6
	case TimeLogical:
		return 7
	case TimestampLogical:
		return 8
	case IntegerLogical:
		return 10
	case UnknownLogical:
		return 11
	case JSONLogical:
		return 12
	case BSONLogical:
		return 13
	case UUIDLogical:
		return 14
	case Float16Logical:
		return 15
	default:
		return 0
	}
}

func (g *RowGroup) write(w *writer) {
	writeStructs(w, 1, g.Columns, (*ColumnChunk).write)
	w.int64Field(2, g.TotalByteSize)
	w.int64Field(3, g.NumRows)
	if g.FileOffset != 0 {
		w.int64Field(5, g.FileOffset)
	}
	if g.TotalCompressedSize != 0 {
		w.int64Field(6, g.TotalCompressedSize)
	}
	if g.Ordinal != 0 {
		w.int16Field(7, g.Ordinal)
	}
}

func (c *ColumnChunk) write(w *writer) {
	if c.FilePath != "" {
		w.textField(1, c.FilePath)
	}
	if c.FileOffset != 0 {
		w.int64Field(2, c.FileOffset)
	}
	c.Meta.write(w, 3)
	if c.OffsetIndexOffset != 0 {
		w.int64Field(4, c.OffsetIndexOffset)
	}
	if c.OffsetIndexLength != 0 {
		w.int32Field(5, c.OffsetIndexLength)
	}
	if c.ColumnIndexOffset != 0 {
		w.int64Field(6, c.ColumnIndexOffset)
	}
	if c.ColumnIndexLength != 0 {
		w.int32Field(7, c.ColumnIndexLength)
	}
}

// write writes the chunk's own metadata as field id, which is where its pages
// are and what is in them.
func (m *ColumnMeta) write(w *writer, id int16) {
	w.structure(id, func() {
		w.int32Field(1, int32(m.Type))
		writeEnums(w, 2, m.Encodings)
		writeTexts(w, 3, m.Path)
		w.int32Field(4, int32(m.Codec))
		w.int64Field(5, m.NumValues)
		w.int64Field(6, m.TotalUncompressedSize)
		w.int64Field(7, m.TotalCompressedSize)
		w.int64Field(9, m.DataPageOffset)
		if m.IndexPageOffset != 0 {
			w.int64Field(10, m.IndexPageOffset)
		}
		if m.DictionaryPageOffset != 0 {
			w.int64Field(11, m.DictionaryPageOffset)
		}
		m.Stats.write(w, 12)
		if m.PageStats != nil {
			writeStructs(w, 13, m.PageStats, (*PageEncodingStats).write)
		}
		if m.BloomFilterOffset != 0 {
			w.int64Field(14, m.BloomFilterOffset)
		}
		if m.BloomFilterLength != 0 {
			w.int32Field(15, m.BloomFilterLength)
		}
		m.Sizes.write(w, 16)
	})
}

// write writes one page count. All three fields are required, so all three go
// out whatever they hold.
func (p *PageEncodingStats) write(w *writer) {
	w.int32Field(1, int32(p.Kind))
	w.int32Field(2, int32(p.Encoding))
	w.int32Field(3, p.Count)
}

// write writes the sizes as field id, or writes nothing for a chunk whose writer
// said nothing about them.
func (s *SizeStatistics) write(w *writer, id int16) {
	if !s.written() {
		return
	}

	w.structure(id, func() {
		if s.HasUnencodedBytes {
			w.int64Field(1, s.UnencodedBytes)
		}
		if s.RepetitionHistogram != nil {
			writeLongs(w, 2, s.RepetitionHistogram)
		}
		if s.DefinitionHistogram != nil {
			writeLongs(w, 3, s.DefinitionHistogram)
		}
	})
}

// write writes the statistics as field id, or writes nothing for a chunk whose
// writer said nothing about its values.
func (s *Statistics) write(w *writer, id int16) {
	if !s.written() {
		return
	}

	w.structure(id, func() {
		if s.Max != nil {
			w.bytesField(1, s.Max)
		}
		if s.Min != nil {
			w.bytesField(2, s.Min)
		}
		if s.HasNullCount {
			w.int64Field(3, s.NullCount)
		}
		if s.HasDistinct {
			w.int64Field(4, s.DistinctCount)
		}
		if s.MaxValue != nil {
			w.bytesField(5, s.MaxValue)
		}
		if s.MinValue != nil {
			w.bytesField(6, s.MinValue)
		}
		if s.MaxExact {
			w.boolField(7, true)
		}
		if s.MinExact {
			w.boolField(8, true)
		}
	})
}

// written says whether the writer of the file said anything at all about the
// values, which is what tells a chunk with no statistics from one whose
// statistics happen to be nought.
//
// The two exactness flags are not counted, since a flag saying a bound is a
// value that is in the chunk means nothing without the bound next to it.
func (s *Statistics) written() bool {
	return s.MinValue != nil || s.MaxValue != nil || s.Min != nil || s.Max != nil ||
		s.HasNullCount || s.HasDistinct
}

func (kv *KeyValue) write(w *writer) {
	w.textField(1, kv.Key)
	if kv.Value != "" {
		w.textField(2, kv.Value)
	}
}
