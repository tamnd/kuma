package ipc_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

// TestPyarrowSchema checks the Arrow IPC schema message against another
// implementation of it.
//
// The C data interface has a cross check that runs two libraries in one process,
// because that is the only way to test a pointer handover. A message is
// simpler: it is bytes, so the two sides can pass files. Each side writes the
// schemas it can build and, alongside them, what it expects the other side to
// make of them, and each side checks the other's expectations against what it
// actually read. What the two of them agree on is the specification rather than
// kuma's reading of it.
//
// It skips when there is no python3 with pyarrow in it, since that is a large
// dependency to ask of somebody running the tests. CI installs it, so the check
// runs on every commit.
func TestPyarrowSchema(t *testing.T) {
	python := findPython(t)
	dir := t.TempDir()

	var manifest strings.Builder
	for _, c := range pyarrowCases {
		msg, err := ipc.EncodeSchema(c.schema)
		if err != nil {
			t.Fatalf("%s: EncodeSchema: %v", c.name, err)
		}
		if err := os.WriteFile(filepath.Join(dir, "go-"+c.name+".arrow"), msg, 0o600); err != nil {
			t.Fatal(err)
		}
		manifest.WriteString(c.name + "\t" + c.arrow + "\n")
	}
	if err := os.WriteFile(filepath.Join(dir, "go.txt"), []byte(manifest.String()), 0o600); err != nil {
		t.Fatal(err)
	}

	run := exec.Command(python, filepath.Join("testdata", "pyarrow", "schema.py"), dir)
	out, err := run.CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	t.Log(strings.TrimSpace(string(out)))

	checkBack(t, dir)
	checkPython(t, dir)
}

// checkBack reads the schemas pyarrow wrote back out and compares them with
// what kuma sent. A schema that survives the trip is one both sides agree on,
// field for field.
func checkBack(t *testing.T, dir string) {
	t.Helper()
	for _, c := range pyarrowCases {
		msg, err := os.ReadFile(filepath.Join(dir, "back-"+c.name+".arrow"))
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		got, err := ipc.DecodeSchema(msg)
		if err != nil {
			t.Errorf("%s: DecodeSchema of what pyarrow wrote: %v", c.name, err)
			continue
		}
		want := c.back
		if want.Fields == nil && want.Metadata == nil {
			want = c.schema
		}
		if !got.Equal(want) {
			t.Errorf("%s: came back from pyarrow as\n%v\nwant\n%v", c.name, got, want)
		}
	}
}

// checkPython reads the schemas pyarrow built itself and compares each one with
// the rendering pyarrow said kuma should read.
//
// These are the types kuma cannot write. The three text layouts it collapses,
// the ones it has no equivalent for at all, and the defaults it never leaves
// out are all here, since a reader that has only ever read its own writing has
// not been tested against anything.
func checkPython(t *testing.T, dir string) {
	t.Helper()
	manifest, err := os.ReadFile(filepath.Join(dir, "py.txt"))
	if err != nil {
		t.Fatal(err)
	}
	for line := range strings.Lines(strings.TrimSpace(string(manifest))) {
		name, want, ok := strings.Cut(strings.TrimSpace(line), "\t")
		if !ok {
			t.Fatalf("py.txt: %q is not a name and a rendering", line)
		}
		msg, err := os.ReadFile(filepath.Join(dir, "py-"+name+".arrow"))
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}

		s, err := ipc.DecodeSchema(msg)
		if want == "!error" {
			if err == nil {
				t.Errorf("%s: read as %v, want an error", name, s)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: DecodeSchema: %v", name, err)
			continue
		}
		if got := render(s); got != want {
			t.Errorf("%s: read as\n%s\nwant\n%s", name, got, want)
		}
	}
}

// render writes a schema the way both sides of the cross check write one: a
// field is a name, a type and whether it is nullable, its metadata follows it
// after semicolons, and the metadata of the schema itself is the last field
// under the name "@".
func render(s dtype.Schema) string {
	parts := make([]string, 0, len(s.Fields)+1)
	for _, f := range s.Fields {
		null := "notnull"
		if f.Nullable {
			null = "null"
		}
		parts = append(parts, f.Name+":"+f.Type.String()+":"+null+renderMetadata(f.Metadata))
	}
	if len(s.Metadata) > 0 {
		parts = append(parts, "@"+renderMetadata(s.Metadata))
	}
	return strings.Join(parts, "|")
}

func renderMetadata(m dtype.Metadata) string {
	var sb strings.Builder
	for _, kv := range m {
		sb.WriteString(";" + kv.Key + "=" + kv.Value)
	}
	return sb.String()
}

// findPython returns an interpreter that can import pyarrow, or skips.
func findPython(t *testing.T) string {
	t.Helper()
	for _, name := range []string{"python3", "python"} {
		path, err := exec.LookPath(name)
		if err != nil {
			continue
		}
		if err := exec.Command(path, "-c", "import pyarrow").Run(); err == nil {
			return path
		}
	}
	t.Skip("no python3 with pyarrow, which this needs to have somebody to disagree with")
	return ""
}

// pyarrowCases are the schemas kuma writes for pyarrow to read.
//
// arrow is what pyarrow should make of it, written in pyarrow's own vocabulary,
// and back is what kuma should read when pyarrow serializes it again. The two
// of them are what the cross check is: a claim about another implementation,
// written down before it runs.
var pyarrowCases = []struct {
	name   string
	schema dtype.Schema
	arrow  string
	back   dtype.Schema
}{
	{
		name: "primitives",
		schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "null", Type: dtype.Null, Nullable: true},
			{Name: "bool", Type: dtype.Bool},
			{Name: "int8", Type: dtype.Int8},
			{Name: "int16", Type: dtype.Int16},
			{Name: "int32", Type: dtype.Int32},
			{Name: "int64", Type: dtype.Int64, Nullable: true},
			{Name: "uint8", Type: dtype.Uint8},
			{Name: "uint16", Type: dtype.Uint16},
			{Name: "uint32", Type: dtype.Uint32},
			{Name: "uint64", Type: dtype.Uint64},
			{Name: "float32", Type: dtype.Float32},
			{Name: "float64", Type: dtype.Float64},
		}},
		arrow: "null:null:null|bool:bool:notnull|int8:int8:notnull|int16:int16:notnull|" +
			"int32:int32:notnull|int64:int64:null|uint8:uint8:notnull|uint16:uint16:notnull|" +
			"uint32:uint32:notnull|uint64:uint64:notnull|float32:float:notnull|" +
			"float64:double:notnull",
	},
	{
		name: "text",
		schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "string", Type: dtype.String},
			{Name: "binary", Type: dtype.Binary},
			{Name: "large string", Type: dtype.LargeString},
			{Name: "large binary", Type: dtype.LargeBinary},
			{Name: "fixed", Type: dtype.FixedSizeBinary{ByteWidth: 16}},
		}},
		arrow: "string:string_view:notnull|binary:binary_view:notnull|" +
			"large string:large_string:notnull|large binary:large_binary:notnull|" +
			"fixed:fixed_size_binary[16]:notnull",
		// The two large layouts come back as the one layout kuma has, which is
		// the same collapse the format strings make and the reason there is a
		// separate column for what comes back at all.
		back: dtype.Schema{Fields: []dtype.Field{
			{Name: "string", Type: dtype.String},
			{Name: "binary", Type: dtype.Binary},
			{Name: "large string", Type: dtype.String},
			{Name: "large binary", Type: dtype.Binary},
			{Name: "fixed", Type: dtype.FixedSizeBinary{ByteWidth: 16}},
		}},
	},
	{
		name: "temporal",
		schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "date32", Type: dtype.Date32},
			{Name: "date64", Type: dtype.Date64},
			{Name: "time32 s", Type: dtype.Time32{Unit: dtype.Second}},
			{Name: "time32 ms", Type: dtype.Time32{Unit: dtype.Millisecond}},
			{Name: "time64 us", Type: dtype.Time64{Unit: dtype.Microsecond}},
			{Name: "time64 ns", Type: dtype.Time64{Unit: dtype.Nanosecond}},
			{Name: "naive", Type: dtype.Timestamp{Unit: dtype.Microsecond}},
			{Name: "utc", Type: dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "UTC"}},
			{Name: "zoned", Type: dtype.Timestamp{Unit: dtype.Second, Zone: "Europe/London"}},
			{Name: "offset", Type: dtype.Timestamp{Unit: dtype.Millisecond, Zone: "+01:00"}},
			{Name: "duration", Type: dtype.Duration{Unit: dtype.Second}},
			{Name: "year month", Type: dtype.Interval{Unit: dtype.YearMonth}},
			{Name: "month day nano", Type: dtype.Interval{Unit: dtype.MonthDayNano}},
		}},
		arrow: "date32:date32[day]:notnull|date64:date64[ms]:notnull|" +
			"time32 s:time32[s]:notnull|time32 ms:time32[ms]:notnull|" +
			"time64 us:time64[us]:notnull|time64 ns:time64[ns]:notnull|" +
			"naive:timestamp[us]:notnull|utc:timestamp[ns, tz=UTC]:notnull|" +
			"zoned:timestamp[s, tz=Europe/London]:notnull|" +
			"offset:timestamp[ms, tz=+01:00]:notnull|duration:duration[s]:notnull|" +
			"year month:month_interval:notnull|month day nano:month_day_nano_interval:notnull",
	},
	{
		name: "decimals",
		schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "money", Type: dtype.Decimal128{Precision: 18, Scale: 2}},
			{Name: "no scale", Type: dtype.Decimal128{Precision: 38, Scale: 0}},
			{Name: "wide", Type: dtype.Decimal256{Precision: 60, Scale: 10}},
		}},
		arrow: "money:decimal128(18, 2):notnull|no scale:decimal128(38, 0):notnull|" +
			"wide:decimal256(60, 10):notnull",
	},
	{
		name: "nested",
		schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "list", Type: dtype.List{Elem: dtype.Int32}, Nullable: true},
			{Name: "large list", Type: dtype.LargeList{Elem: dtype.Int64}},
			{Name: "fixed list", Type: dtype.FixedSizeList{Elem: dtype.Float64, Len: 3}},
			{Name: "map", Type: dtype.Map{Key: dtype.String, Value: dtype.Int64}},
			{Name: "struct", Type: dtype.Struct{Fields: []dtype.Field{
				{Name: "x", Type: dtype.Float64},
				{Name: "y", Type: dtype.Float64, Nullable: true},
			}}},
		}},
		arrow: "list:list<item: int32>:null|large list:large_list<item: int64>:notnull|" +
			"fixed list:fixed_size_list<item: double>[3]:notnull|" +
			"map:map<string_view, int64>:notnull|" +
			"struct:struct<x: double not null, y: double>:notnull",
	},
	{
		name: "dictionary",
		schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "small", Type: dtype.Dictionary{Index: dtype.Int8, Value: dtype.String}},
			{Name: "wide", Type: dtype.Dictionary{Index: dtype.Int32, Value: dtype.String}},
		}},
		arrow: "small:dictionary<values=string_view, indices=int8, ordered=0>:notnull|" +
			"wide:dictionary<values=string_view, indices=int32, ordered=0>:notnull",
	},
	{
		name: "metadata",
		schema: dtype.Schema{
			Fields: []dtype.Field{
				{Name: "one", Type: dtype.Int64, Metadata: dtype.Metadata{
					{Key: "unit", Value: "meters"},
					{Key: "empty", Value: ""},
				}},
				{Name: "two", Type: dtype.Float64},
			},
			Metadata: dtype.Metadata{
				{Key: "written by", Value: "kuma"},
				{Key: "unicode", Value: "\u30af\u30de"},
			},
		},
		arrow: "one:int64:notnull;unit=meters;empty=|two:double:notnull|" +
			"@;written by=kuma;unicode=\u30af\u30de",
	},
}
