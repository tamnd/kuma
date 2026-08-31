package kuma

import (
	"fmt"
	"reflect"
	"strings"
	"time"
	"unicode"

	"github.com/tamnd/kuma/dtype"
)

// Bind checks a frame against the struct S and returns the same frame with S as
// its schema.
//
//	f, err := kuma.ReadCSV(r)
//	typed, err := kuma.Bind[Trade](f)
//
// This is the bridge from the dynamic world to the typed one, and it is where
// the names in the struct are checked against the names in the data. After it
// returns, a handle written for Trade can be used on the frame and the compiler
// takes over. The shape of most programs is a short dynamic prologue that reads
// whatever arrived, one Bind, and then a long typed body.
//
// Every field of the struct has to have a column, of a type the field can be
// read out of. A column the struct does not mention is left alone and stays in
// the frame, since a file usually holds more than the part of it a program
// cares about.
//
// The column for a field is the kuma tag when there is one, and the field name
// in snake case when there is not, which is the convention most Go code that
// reads JSON already follows. A field tagged "-" is skipped, and so is an
// unexported one.
//
// Nothing is copied. The frame that comes back shares the columns of the one
// that went in.
func Bind[S any](f *Frame[Dynamic]) (*Frame[S], error) {
	fields, err := schemaOf[S]()
	if err != nil {
		return nil, err
	}

	for _, sf := range fields {
		i, ok := f.index[sf.column]
		if !ok {
			return nil, noColumn("Bind", sf.column, f.Names())
		}
		if dt := f.cols[i].DType(); !canReadType(sf.typ, dt) {
			return nil, wrongField("Bind", sf, dt)
		}
	}

	return &Frame[S]{cols: f.cols, schema: f.schema, rows: f.rows, index: f.index}, nil
}

// schemaField is one field of a schema struct and the column it names.
type schemaField struct {
	field  string
	column string
	typ    reflect.Type
}

// schemaOf returns the fields of the schema struct S, in the order they are
// written.
//
// A schema type that is not a struct is a mistake in the program rather than
// something the data did, but it is returned as an error rather than a panic
// because Bind already returns one and a caller reading a file has a place to
// put it.
func schemaOf[S any]() ([]schemaField, error) {
	rt := reflect.TypeFor[S]()
	if rt.Kind() != reflect.Struct {
		return nil, fmt.Errorf("kuma: the schema type %s is not a struct: %w", rt, ErrWrongType)
	}

	var out []schemaField
	for i := range rt.NumField() {
		f := rt.Field(i)
		if !f.IsExported() {
			continue
		}
		name := f.Tag.Get("kuma")
		if name == "-" {
			continue
		}
		if name == "" {
			name = ColumnName(f.Name)
		}
		out = append(out, schemaField{field: f.Name, column: name, typ: f.Type})
	}
	return out, nil
}

// canReadType reports whether a field of type rt can be read out of a column of
// type dt. It is [CanRead] asked about a type that is only known at runtime,
// and the two answer the same question the same way.
func canReadType(rt reflect.Type, dt dtype.DataType) bool {
	if dt == nil {
		return false
	}
	if rt == reflect.TypeFor[time.Time]() {
		return dt.Kind() == dtype.TimestampKind
	}

	switch rt.Kind() {
	case reflect.Bool:
		return dt.Kind() == dtype.BoolKind
	case reflect.Int8:
		return dt.Kind() == dtype.Int8Kind
	case reflect.Int16:
		return dt.Kind() == dtype.Int16Kind
	case reflect.Int32:
		switch dt.Kind() {
		case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
			return true
		default:
			return false
		}
	case reflect.Int64:
		switch dt.Kind() {
		case dtype.Int64Kind, dtype.Date64Kind, dtype.Time64Kind,
			dtype.TimestampKind, dtype.DurationKind:
			return true
		default:
			return false
		}
	case reflect.Uint8:
		return dt.Kind() == dtype.Uint8Kind
	case reflect.Uint16:
		return dt.Kind() == dtype.Uint16Kind
	case reflect.Uint32:
		return dt.Kind() == dtype.Uint32Kind
	case reflect.Uint64:
		return dt.Kind() == dtype.Uint64Kind
	case reflect.Float32:
		return dt.Kind() == dtype.Float32Kind
	case reflect.Float64:
		return dt.Kind() == dtype.Float64Kind
	case reflect.String:
		return dt.Kind() == dtype.StringKind || dt.Kind() == dtype.BinaryKind
	default:
		return false
	}
}

// ColumnName returns the column a struct field binds to when it carries no kuma
// tag, which is the field name in snake case.
//
// It is the same rule the rest of the Go world uses for JSON: OrderID becomes
// order_id and TS becomes ts. A run of capitals is one word, so HTTPCode is
// http_code rather than h_t_t_p_code, and a digit stays with the word it
// follows.
//
// It is exported because kumagen has to name the same columns [Bind] does, and
// because a program that writes its own schema out has a use for the rule.
func ColumnName(field string) string {
	var sb strings.Builder
	sb.Grow(len(field) + 4)

	rs := []rune(field)
	for i, r := range rs {
		if unicode.IsUpper(r) {
			// A capital starts a word unless it is inside a run of them, and
			// the last capital of a run starts one when a lower case letter
			// follows it, which is the D of OrderID against the C of HTTPCode.
			prevLower := i > 0 && !unicode.IsUpper(rs[i-1])
			nextLower := i+1 < len(rs) && unicode.IsLower(rs[i+1])
			if i > 0 && (prevLower || nextLower) {
				sb.WriteByte('_')
			}
			sb.WriteRune(unicode.ToLower(r))
			continue
		}
		sb.WriteRune(r)
	}
	return sb.String()
}
