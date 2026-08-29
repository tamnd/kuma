package arrowgo

import (
	"fmt"

	"github.com/apache/arrow-go/v18/arrow"

	"github.com/tamnd/kuma/dtype"
)

// ImportType returns the kuma type for an arrow-go one.
//
// The two string layouts arrive as the same kuma type, since kuma stores every
// string as a view: utf8, large_utf8 and string_view all come back as
// [dtype.String], and the three binary layouts as [dtype.Binary]. What that
// costs at the point the values cross is in the package comment.
//
// A dictionary keeps its index and value types and loses its ordered flag,
// which kuma does not carry. Nothing reads that flag except a sort, and a sort
// in kuma is over the values rather than over the codes.
//
// A nested, union or run end encoded type is an error rather than an
// approximation of one. They are named in the message so that a schema holding
// one says which column stopped it.
func ImportType(t arrow.DataType) (dtype.DataType, error) {
	if t == nil {
		return nil, fmt.Errorf("arrowgo: nil arrow type")
	}

	switch t.ID() {
	case arrow.NULL:
		return dtype.Null, nil
	case arrow.BOOL:
		return dtype.Bool, nil
	case arrow.INT8:
		return dtype.Int8, nil
	case arrow.INT16:
		return dtype.Int16, nil
	case arrow.INT32:
		return dtype.Int32, nil
	case arrow.INT64:
		return dtype.Int64, nil
	case arrow.UINT8:
		return dtype.Uint8, nil
	case arrow.UINT16:
		return dtype.Uint16, nil
	case arrow.UINT32:
		return dtype.Uint32, nil
	case arrow.UINT64:
		return dtype.Uint64, nil
	case arrow.FLOAT32:
		return dtype.Float32, nil
	case arrow.FLOAT64:
		return dtype.Float64, nil
	case arrow.STRING, arrow.LARGE_STRING, arrow.STRING_VIEW:
		return dtype.String, nil
	case arrow.BINARY, arrow.LARGE_BINARY, arrow.BINARY_VIEW:
		return dtype.Binary, nil
	case arrow.DATE32:
		return dtype.Date32, nil
	case arrow.DATE64:
		return dtype.Date64, nil
	}

	// The parameterized types, which have to be reached through the concrete
	// type rather than the id because the parameters are what makes them
	// different from each other.
	switch x := t.(type) {
	case *arrow.Time32Type:
		return dtype.Time32{Unit: importUnit(x.Unit)}, nil
	case *arrow.Time64Type:
		return dtype.Time64{Unit: importUnit(x.Unit)}, nil
	case *arrow.TimestampType:
		return dtype.Timestamp{Unit: importUnit(x.Unit), Zone: x.TimeZone}, nil
	case *arrow.DurationType:
		return dtype.Duration{Unit: importUnit(x.Unit)}, nil
	case *arrow.MonthIntervalType:
		return dtype.Interval{Unit: dtype.YearMonth}, nil
	case *arrow.DayTimeIntervalType:
		return dtype.Interval{Unit: dtype.DayTime}, nil
	case *arrow.MonthDayNanoIntervalType:
		return dtype.Interval{Unit: dtype.MonthDayNano}, nil
	case *arrow.FixedSizeBinaryType:
		return dtype.FixedSizeBinary{ByteWidth: int32(x.ByteWidth)}, nil
	case *arrow.Decimal128Type:
		return dtype.Decimal128{Precision: x.Precision, Scale: x.Scale}, nil
	case *arrow.Decimal256Type:
		return dtype.Decimal256{Precision: x.Precision, Scale: x.Scale}, nil
	case *arrow.DictionaryType:
		return importDictionary(x)
	}
	return nil, fmt.Errorf("arrowgo: a %s column does not cross, "+
		"kuma holds no nested, union or run end encoded values yet", t)
}

// importDictionary is the one type whose parts have to be checked, since a
// dictionary of a nested type is a nested type with a level of indirection in
// front of it.
func importDictionary(t *arrow.DictionaryType) (dtype.DataType, error) {
	index, err := ImportType(t.IndexType)
	if err != nil {
		return nil, fmt.Errorf("the index of a %s: %w", t, err)
	}
	value, err := ImportType(t.ValueType)
	if err != nil {
		return nil, fmt.Errorf("the values of a %s: %w", t, err)
	}
	return dtype.Dictionary{Index: index, Value: value}, nil
}

// ExportType returns the arrow-go type for a kuma one.
//
// A string is a string_view and a binary is a binary_view, which is the layout
// kuma stores and so the one that crosses without copying. large_string and
// large_binary are the offset layouts kuma keeps for interoperability and they
// go back out as themselves.
func ExportType(t dtype.DataType) (arrow.DataType, error) {
	if t == nil {
		return nil, fmt.Errorf("arrowgo: nil kuma type")
	}

	switch x := t.(type) {
	case dtype.Time32:
		return &arrow.Time32Type{Unit: exportUnit(x.Unit)}, nil
	case dtype.Time64:
		return &arrow.Time64Type{Unit: exportUnit(x.Unit)}, nil
	case dtype.Timestamp:
		return &arrow.TimestampType{Unit: exportUnit(x.Unit), TimeZone: x.Zone}, nil
	case dtype.Duration:
		return &arrow.DurationType{Unit: exportUnit(x.Unit)}, nil
	case dtype.Interval:
		return exportInterval(x)
	case dtype.FixedSizeBinary:
		return &arrow.FixedSizeBinaryType{ByteWidth: int(x.ByteWidth)}, nil
	case dtype.Decimal128:
		return &arrow.Decimal128Type{Precision: x.Precision, Scale: x.Scale}, nil
	case dtype.Decimal256:
		return &arrow.Decimal256Type{Precision: x.Precision, Scale: x.Scale}, nil
	case dtype.Dictionary:
		return exportDictionary(x)
	}

	if out, ok := plainArrowTypes[t.Kind()]; ok {
		return out, nil
	}
	return nil, fmt.Errorf("arrowgo: a %s column does not cross, "+
		"kuma holds no nested values yet and so has none to hand over", t)
}

// plainArrowTypes are the kinds with nothing to carry, which are a lookup
// rather than a switch because there is one answer per kind and no work to do
// to reach it.
var plainArrowTypes = map[dtype.Kind]arrow.DataType{
	dtype.NullKind:        arrow.Null,
	dtype.BoolKind:        arrow.FixedWidthTypes.Boolean,
	dtype.Int8Kind:        arrow.PrimitiveTypes.Int8,
	dtype.Int16Kind:       arrow.PrimitiveTypes.Int16,
	dtype.Int32Kind:       arrow.PrimitiveTypes.Int32,
	dtype.Int64Kind:       arrow.PrimitiveTypes.Int64,
	dtype.Uint8Kind:       arrow.PrimitiveTypes.Uint8,
	dtype.Uint16Kind:      arrow.PrimitiveTypes.Uint16,
	dtype.Uint32Kind:      arrow.PrimitiveTypes.Uint32,
	dtype.Uint64Kind:      arrow.PrimitiveTypes.Uint64,
	dtype.Float32Kind:     arrow.PrimitiveTypes.Float32,
	dtype.Float64Kind:     arrow.PrimitiveTypes.Float64,
	dtype.StringKind:      arrow.BinaryTypes.StringView,
	dtype.BinaryKind:      arrow.BinaryTypes.BinaryView,
	dtype.LargeStringKind: arrow.BinaryTypes.LargeString,
	dtype.LargeBinaryKind: arrow.BinaryTypes.LargeBinary,
	dtype.Date32Kind:      arrow.FixedWidthTypes.Date32,
	dtype.Date64Kind:      arrow.FixedWidthTypes.Date64,
}

// exportInterval is its own function because arrow-go has three types where
// kuma has one with a unit on it.
func exportInterval(t dtype.Interval) (arrow.DataType, error) {
	switch t.Unit {
	case dtype.YearMonth:
		return arrow.FixedWidthTypes.MonthInterval, nil
	case dtype.DayTime:
		return arrow.FixedWidthTypes.DayTimeInterval, nil
	case dtype.MonthDayNano:
		return arrow.FixedWidthTypes.MonthDayNanoInterval, nil
	}
	return nil, fmt.Errorf("arrowgo: %s has no interval unit %d", t, t.Unit)
}

// exportDictionary carries both halves across, and is where an index type that
// arrow-go will not accept is reported.
func exportDictionary(t dtype.Dictionary) (arrow.DataType, error) {
	index, err := ExportType(t.Index)
	if err != nil {
		return nil, fmt.Errorf("the index of a %s: %w", t, err)
	}
	value, err := ExportType(t.Value)
	if err != nil {
		return nil, fmt.Errorf("the values of a %s: %w", t, err)
	}
	return &arrow.DictionaryType{IndexType: index, ValueType: value}, nil
}

// importUnit and exportUnit convert the time units, which are the same four in
// the same order on both sides. They are written out rather than converted by
// arithmetic so that a renumbering on either side is a compile error here
// instead of a column of timestamps that is off by a factor of a thousand.
func importUnit(u arrow.TimeUnit) dtype.TimeUnit {
	switch u {
	case arrow.Second:
		return dtype.Second
	case arrow.Millisecond:
		return dtype.Millisecond
	case arrow.Microsecond:
		return dtype.Microsecond
	case arrow.Nanosecond:
		return dtype.Nanosecond
	}
	// arrow-go has four units and a type carrying anything else would not have
	// come from its own constructors. Nanoseconds is the one that loses no
	// precision if it ever happens, and Validate will refuse the type anyway
	// when it is a unit the column is not allowed to count in.
	return dtype.Nanosecond
}

func exportUnit(u dtype.TimeUnit) arrow.TimeUnit {
	switch u {
	case dtype.Second:
		return arrow.Second
	case dtype.Millisecond:
		return arrow.Millisecond
	case dtype.Microsecond:
		return arrow.Microsecond
	case dtype.Nanosecond:
		return arrow.Nanosecond
	}
	return arrow.Nanosecond
}

// ImportField returns the kuma field for an arrow-go one, metadata included.
func ImportField(f arrow.Field) (dtype.Field, error) {
	t, err := ImportType(f.Type)
	if err != nil {
		return dtype.Field{}, fmt.Errorf("the column %q: %w", f.Name, err)
	}
	return dtype.Field{
		Name:     f.Name,
		Type:     t,
		Nullable: f.Nullable,
		Metadata: importMetadata(f.Metadata),
	}, nil
}

// ExportField returns the arrow-go field for a kuma one.
func ExportField(f dtype.Field) (arrow.Field, error) {
	t, err := ExportType(f.Type)
	if err != nil {
		return arrow.Field{}, fmt.Errorf("the column %q: %w", f.Name, err)
	}
	return arrow.Field{
		Name:     f.Name,
		Type:     t,
		Nullable: f.Nullable,
		Metadata: exportMetadata(f.Metadata),
	}, nil
}

// ImportSchema returns the kuma schema for an arrow-go one.
func ImportSchema(s *arrow.Schema) (dtype.Schema, error) {
	if s == nil {
		return dtype.Schema{}, fmt.Errorf("arrowgo: nil arrow schema")
	}

	fields := make([]dtype.Field, s.NumFields())
	for i := range fields {
		f, err := ImportField(s.Field(i))
		if err != nil {
			return dtype.Schema{}, err
		}
		fields[i] = f
	}
	return dtype.Schema{Fields: fields, Metadata: importMetadata(s.Metadata())}, nil
}

// ExportSchema returns the arrow-go schema for a kuma one.
func ExportSchema(s dtype.Schema) (*arrow.Schema, error) {
	fields := make([]arrow.Field, len(s.Fields))
	for i, f := range s.Fields {
		out, err := ExportField(f)
		if err != nil {
			return nil, err
		}
		fields[i] = out
	}
	md := exportMetadata(s.Metadata)
	return arrow.NewSchema(fields, &md), nil
}

// importMetadata and exportMetadata convert between a list of pairs and two
// parallel lists. Order is kept, since Arrow metadata is a list rather than a
// map and a reader is allowed to care.
func importMetadata(m arrow.Metadata) dtype.Metadata {
	if m.Len() == 0 {
		return nil
	}

	keys, values := m.Keys(), m.Values()
	out := make(dtype.Metadata, len(keys))
	for i := range keys {
		out[i] = dtype.KeyValue{Key: keys[i], Value: values[i]}
	}
	return out
}

func exportMetadata(m dtype.Metadata) arrow.Metadata {
	keys := make([]string, len(m))
	values := make([]string, len(m))
	for i, kv := range m {
		keys[i], values[i] = kv.Key, kv.Value
	}
	return arrow.NewMetadata(keys, values)
}
