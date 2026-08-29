package main

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/csv"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
	"github.com/tamnd/kuma/ndjson"
	"github.com/tamnd/kuma/parquet"
)

// readSchema returns the schema of the file at path, read as the format its
// extension names.
//
// Parquet and Arrow IPC both keep the schema in a footer, so those two are read
// from the end of the file and nothing else is touched however large the file
// is. A delimited or a line delimited file has no footer and no schema in it at
// all, so the first lines of one are read and the types are worked out from
// what is in them, which is a guess made from a sample and is why lines is a
// flag rather than a constant.
func readSchema(path string, lines int) (dtype.Schema, error) {
	f, err := os.Open(path)
	if err != nil {
		return dtype.Schema{}, err
	}
	defer f.Close()

	switch ext := strings.ToLower(filepath.Ext(path)); ext {
	case ".parquet":
		return footerSchema(f, parquetSchema)
	case ".arrow", ".ipc", ".feather":
		return footerSchema(f, arrowSchema)
	case ".arrows", ".stream":
		r, err := ipc.NewReader(f)
		if err != nil {
			return dtype.Schema{}, err
		}
		return r.Schema(), nil
	case ".csv":
		return sampleSchema(f, lines+1, ',')
	case ".tsv":
		return sampleSchema(f, lines+1, '\t')
	case ".ndjson", ".jsonl":
		return jsonSchema(f, lines)
	default:
		return dtype.Schema{}, fmt.Errorf(
			"no schema can be read out of a %q file, and there is a reader for "+
				".parquet, .arrow, .arrows, .csv, .tsv and .ndjson", ext)
	}
}

// footerSchema reads the schema of a format that keeps one at the end, which
// needs the size of the file as well as the file.
func footerSchema(f *os.File, read func(io.ReaderAt, int64) (dtype.Schema, error)) (dtype.Schema, error) {
	info, err := f.Stat()
	if err != nil {
		return dtype.Schema{}, err
	}
	return read(f, info.Size())
}

// parquetSchema and arrowSchema are the two footer readers, written out so that
// [footerSchema] takes a function of one shape rather than each caller repeating
// the stat.
func parquetSchema(r io.ReaderAt, size int64) (dtype.Schema, error) {
	pr, err := parquet.NewFileReader(r, size)
	if err != nil {
		return dtype.Schema{}, err
	}
	return pr.Schema(), nil
}

func arrowSchema(r io.ReaderAt, size int64) (dtype.Schema, error) {
	ar, err := ipc.NewFileReader(r, size)
	if err != nil {
		return dtype.Schema{}, err
	}
	return ar.Schema(), nil
}

// sampleSchema works out the columns of a delimited file from its first lines.
//
// The sample is read whole and then inferred over with InferRows negative, so
// every line that was read is looked at. Reading a thousand lines and then
// inferring over a hundred of them would be a sample of a sample.
func sampleSchema(r io.Reader, lines int, delim rune) (dtype.Schema, error) {
	sample, err := head(r, lines)
	if err != nil {
		return dtype.Schema{}, err
	}
	t, err := csv.Read(sample, &csv.Options{Delimiter: delim, InferRows: -1})
	if err != nil {
		return dtype.Schema{}, err
	}
	return t.Schema, nil
}

// jsonSchema works out the columns of a line delimited JSON file the same way.
func jsonSchema(r io.Reader, lines int) (dtype.Schema, error) {
	sample, err := head(r, lines)
	if err != nil {
		return dtype.Schema{}, err
	}
	t, err := ndjson.Read(sample, &ndjson.Options{InferRows: -1})
	if err != nil {
		return dtype.Schema{}, err
	}
	return t.Schema, nil
}

// head returns the first lines of r, or all of it when it has fewer than that.
//
// A file of text is sampled by lines rather than by bytes because half a line
// is not a row, and a reader handed one would either report it or quietly read
// a truncated last value as a real one.
func head(r io.Reader, lines int) (io.Reader, error) {
	br := bufio.NewReader(r)

	var buf bytes.Buffer
	for range lines {
		line, err := br.ReadBytes('\n')
		buf.Write(line)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, err
		}
	}
	return &buf, nil
}

// structField is one column of a schema written out as a Go field.
type structField struct {
	name   string // the field name
	typ    string // the Go type it is declared as
	column string // the column it binds to
	tag    bool   // whether the column has to be named in a tag
}

// structFields turns a schema into the fields of a Go struct.
//
// The type of a field is the one [kuma.Bind] reads out of that column and no
// other, so an int32 column is an int32 field rather than an int64 one. A bind
// compares the widths exactly, and a struct that has to be edited before it
// binds is worse than no struct.
func structFields(typeName string, s dtype.Schema) ([]structField, error) {
	if s.Len() == 0 {
		return nil, errors.New("the file has no columns")
	}

	out := make([]structField, 0, s.Len())
	seen := make(map[string]string, s.Len())
	for _, f := range s.Fields {
		typ, ok := goType(f.Type)
		if !ok {
			return nil, fmt.Errorf("the column %s is a %s, and a field can hold a "+
				"string, a bool, an integer, a float and a time.Time", f.Name, f.Type)
		}

		name, err := fieldName(f.Name)
		if err != nil {
			return nil, err
		}
		if other, dup := seen[name]; dup {
			return nil, fmt.Errorf("the columns %s and %s are both %s as a field "+
				"of %s, so one of them has to be renamed on the way in",
				other, f.Name, name, typeName)
		}
		seen[name] = f.Name

		out = append(out, structField{
			name:   name,
			typ:    typ,
			column: f.Name,
			tag:    kuma.ColumnName(name) != f.Name,
		})
	}
	return out, nil
}

// goType returns the Go type a field has to be declared as to be read out of a
// column of this type, and whether there is one.
//
// The wide string and binary types are not here. They are converted at the IPC
// boundary rather than read directly, so a file holding one has to be read
// before it can be bound and the struct for it is the struct for what it
// becomes.
func goType(dt dtype.DataType) (string, bool) {
	switch dt.Kind() {
	case dtype.BoolKind:
		return "bool", true
	case dtype.Int8Kind:
		return "int8", true
	case dtype.Int16Kind:
		return "int16", true
	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		return "int32", true
	case dtype.Int64Kind, dtype.Date64Kind, dtype.Time64Kind, dtype.DurationKind:
		return "int64", true
	case dtype.Uint8Kind:
		return "uint8", true
	case dtype.Uint16Kind:
		return "uint16", true
	case dtype.Uint32Kind:
		return "uint32", true
	case dtype.Uint64Kind:
		return "uint64", true
	case dtype.Float32Kind:
		return "float32", true
	case dtype.Float64Kind:
		return "float64", true
	case dtype.StringKind, dtype.BinaryKind:
		return "string", true
	case dtype.TimestampKind:
		// A timestamp reads into an int64 as well, and does so as whatever unit
		// the column counts in. A time.Time is what the unit is for.
		return "time.Time", true
	default:
		return "", false
	}
}

// initialisms are the words a Go field name spells in capitals.
//
// The list is short on purpose. Every entry only changes how a name looks, not
// what it binds to, because a run of capitals is one word to [kuma.ColumnName]
// and so id and ID both come back as id.
var initialisms = map[string]string{
	"api":  "API",
	"cpu":  "CPU",
	"css":  "CSS",
	"db":   "DB",
	"dns":  "DNS",
	"eof":  "EOF",
	"guid": "GUID",
	"html": "HTML",
	"http": "HTTP",
	"id":   "ID",
	"ip":   "IP",
	"json": "JSON",
	"rpc":  "RPC",
	"sql":  "SQL",
	"ssh":  "SSH",
	"tcp":  "TCP",
	"tls":  "TLS",
	"ts":   "TS",
	"ttl":  "TTL",
	"udp":  "UDP",
	"uri":  "URI",
	"url":  "URL",
	"utc":  "UTC",
	"uuid": "UUID",
	"xml":  "XML",
}

// fieldName returns the Go field name for a column.
//
// It is the reverse of [kuma.ColumnName] and it does not have to be exact. A
// name that does not come back as the column it was made from is written with a
// tag naming that column, which is what [structFields] checks, so the worst a
// surprising column name costs is a tag.
func fieldName(column string) (string, error) {
	var sb strings.Builder
	sb.Grow(len(column))

	for _, word := range strings.FieldsFunc(column, notWord) {
		if up, ok := initialisms[strings.ToLower(word)]; ok {
			sb.WriteString(up)
			continue
		}
		rs := []rune(word)
		sb.WriteRune(unicode.ToUpper(rs[0]))
		sb.WriteString(string(rs[1:]))
	}

	name := sb.String()
	first, _ := utf8.DecodeRuneInString(name)
	switch {
	case name == "":
		return "", fmt.Errorf("the column %q has no letter or digit in it to "+
			"make a field name out of", column)
	case unicode.IsDigit(first):
		// A field cannot start with a digit and a column can, so one that does
		// is given a word in front of it and then a tag says what it binds to.
		return "Col" + name, nil
	}
	return name, nil
}

// notWord reports whether r separates two words of a column name. A column is
// written with underscores by every writer that has an opinion, but a space and
// a dot both turn up in files exported from a spreadsheet.
func notWord(r rune) bool {
	return !unicode.IsLetter(r) && !unicode.IsDigit(r)
}
