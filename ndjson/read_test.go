package ndjson

import (
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// read is Read over a string, which is what every test here has.
func read(t *testing.T, in string, opts *Options) *Table {
	t.Helper()
	tbl, err := Read(strings.NewReader(in), opts)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	return tbl
}

// show renders a table as one string per column, so a test can say what it
// expects in one line and the failure prints the whole column.
//
//	qty int64 [1 . 3]
func show(t *testing.T, tbl *Table) []string {
	t.Helper()
	out := make([]string, len(tbl.Columns))
	for i, c := range tbl.Columns {
		f := tbl.Schema.Fields[i]
		vals := make([]string, c.Len())
		for j := range vals {
			vals[j] = value(t, c, j)
		}
		out[i] = f.Name + " " + f.Type.String() + " [" + strings.Join(vals, " ") + "]"
	}
	return out
}

// value renders one value, with a dot for a missing one.
func value(t *testing.T, c *array.Chunked, i int) string {
	t.Helper()
	if c.IsNull(i) {
		return "."
	}
	switch c.DType().Kind() {
	case dtype.BoolKind:
		return strconv.FormatBool(c.Bool(i))
	case dtype.Int8Kind:
		return strconv.FormatInt(int64(c.Value[int8](i)), 10)
	case dtype.Int16Kind:
		return strconv.FormatInt(int64(c.Value[int16](i)), 10)
	case dtype.Int32Kind:
		return strconv.FormatInt(int64(c.Value[int32](i)), 10)
	case dtype.Int64Kind:
		return strconv.FormatInt(c.Value[int64](i), 10)
	case dtype.Uint8Kind:
		return strconv.FormatUint(uint64(c.Value[uint8](i)), 10)
	case dtype.Uint16Kind:
		return strconv.FormatUint(uint64(c.Value[uint16](i)), 10)
	case dtype.Uint32Kind:
		return strconv.FormatUint(uint64(c.Value[uint32](i)), 10)
	case dtype.Uint64Kind:
		return strconv.FormatUint(c.Value[uint64](i), 10)
	case dtype.Float32Kind:
		return strconv.FormatFloat(float64(c.Value[float32](i)), 'g', -1, 32)
	case dtype.Float64Kind:
		return strconv.FormatFloat(c.Value[float64](i), 'g', -1, 64)
	default:
		return string(c.Bytes(i))
	}
}

// want checks a table against one string per column.
func want(t *testing.T, tbl *Table, cols ...string) {
	t.Helper()
	got := show(t, tbl)
	if len(got) != len(cols) {
		t.Fatalf("got %d columns, want %d:\n%s", len(got), len(cols), strings.Join(got, "\n"))
	}
	for i := range cols {
		if got[i] != cols[i] {
			t.Errorf("column %d:\n got %s\nwant %s", i, got[i], cols[i])
		}
	}
}

func TestRead(t *testing.T) {
	cases := []struct {
		name string
		in   string
		opts *Options
		cols []string
	}{
		{
			name: "the columns are the members and the types are the values",
			in: `{"sym":"AAPL","qty":100,"px":1.5,"live":true}
{"sym":"MSFT","qty":200,"px":2,"live":false}
`,
			cols: []string{
				"sym string [AAPL MSFT]",
				"qty int64 [100 200]",
				"px float64 [1.5 2]",
				"live bool [true false]",
			},
		},
		{
			name: "a null is a missing value",
			in:   "{\"sym\":null,\"qty\":1}\n{\"sym\":\"MSFT\",\"qty\":null}\n",
			cols: []string{"sym string [. MSFT]", "qty int64 [1 .]"},
		},
		{
			name: "a member a line has not got is a missing value",
			in:   "{\"a\":1,\"b\":2}\n{\"a\":3}\n{\"b\":4}\n",
			cols: []string{"a int64 [1 3 .]", "b int64 [2 . 4]"},
		},
		{
			name: "the columns come out in the order they were first seen",
			in:   "{\"b\":1,\"a\":2}\n{\"c\":3,\"a\":4}\n",
			cols: []string{"b int64 [1 .]", "a int64 [2 4]", "c int64 [. 3]"},
		},
		{
			name: "an empty string is a value and not a missing one",
			in:   "{\"a\":\"\"}\n{\"a\":null}\n",
			cols: []string{"a string [ .]"},
		},
		{
			name: "an integer among floats is a float column",
			in:   "{\"px\":1}\n{\"px\":2.5}\n{\"px\":3}\n",
			cols: []string{"px float64 [1 2.5 3]"},
		},
		{
			name: "a number with an exponent is a float",
			in:   "{\"px\":1e3}\n{\"px\":2}\n",
			cols: []string{"px float64 [1000 2]"},
		},
		{
			name: "a number too large for an int64 is a float",
			in:   "{\"n\":99999999999999999999}\n{\"n\":1}\n",
			cols: []string{"n float64 [1e+20 1]"},
		},
		{
			name: "a quoted number is a string, which is what the file said",
			in:   "{\"n\":\"1\"}\n{\"n\":\"2\"}\n",
			cols: []string{"n string [1 2]"},
		},
		{
			name: "two kinds of value in one member is a string column",
			in:   "{\"x\":1}\n{\"x\":true}\n",
			cols: []string{"x string [1 true]"},
		},
		{
			name: "a member that was null everywhere is a string column",
			in:   "{\"a\":1,\"b\":null}\n{\"a\":2,\"b\":null}\n",
			cols: []string{"a int64 [1 2]", "b string [. .]"},
		},
		{
			name: "an object goes in as the text it arrived as",
			in:   "{\"a\":{\"b\":1,\"c\":[2,3]}}\n{\"a\":{}}\n",
			cols: []string{"a string [{\"b\":1,\"c\":[2,3]} {}]"},
		},
		{
			name: "and so does an array",
			in:   "{\"a\":[1,2]}\n{\"a\":[]}\n",
			cols: []string{"a string [[1,2] []]"},
		},
		{
			name: "escapes are read in the values and in the names",
			in:   "{\"a\\u0062\":\"one\\ttwo\",\"c\":\"\\u00e9\"}\n",
			cols: []string{"ab string [one\ttwo]", "c string [\u00e9]"},
		},
		{
			name: "blank lines are passed over",
			in:   "\n{\"a\":1}\n\n   \n{\"a\":2}\n\n",
			cols: []string{"a int64 [1 2]"},
		},
		{
			name: "a last line with no newline after it",
			in:   "{\"a\":1}\n{\"a\":2}",
			cols: []string{"a int64 [1 2]"},
		},
		{
			name: "carriage returns are line endings",
			in:   "{\"a\":1}\r\n{\"a\":2}\r\n",
			cols: []string{"a int64 [1 2]"},
		},
		{
			name: "whitespace inside a line is the decoder's business",
			in:   "  { \"a\" : 1 , \"b\" : [ 1 ] }  \n",
			cols: []string{"a int64 [1]", "b string [[ 1 ]]"},
		},
		{
			name: "columns names what to read and in which order",
			in:   "{\"a\":1,\"b\":2,\"c\":3}\n{\"a\":4,\"b\":5,\"c\":6}\n",
			opts: &Options{Columns: []string{"c", "a"}},
			cols: []string{"c int64 [3 6]", "a int64 [1 4]"},
		},
		{
			// The check for a repeated member is on the columns being read, so
			// a repeat of one that is not being read is passed over like any
			// other member nothing is reading.
			name: "the same member twice where nothing is reading it",
			in:   "{\"a\":1,\"b\":2,\"b\":3}\n",
			opts: &Options{Columns: []string{"a"}},
			cols: []string{"a int64 [1]"},
		},
		{
			name: "types name what a column holds",
			in:   "{\"a\":1,\"b\":2}\n{\"a\":3,\"b\":4}\n",
			opts: &Options{Types: map[string]dtype.DataType{
				"a": dtype.Int32, "b": dtype.Float32,
			}},
			cols: []string{"a int32 [1 3]", "b float32 [2 4]"},
		},
		{
			name: "a declared column reads a quoted value too",
			in:   "{\"a\":\"1\",\"b\":\"2.5\",\"c\":\"true\"}\n{\"a\":2,\"b\":3,\"c\":false}\n",
			opts: &Options{Types: map[string]dtype.DataType{
				"a": dtype.Int64, "b": dtype.Float64, "c": dtype.Bool,
			}},
			cols: []string{"a int64 [1 2]", "b float64 [2.5 3]", "c bool [true false]"},
		},
		{
			name: "a binary column is base64",
			in:   "{\"a\":\"aGVsbG8=\"}\n",
			opts: &Options{Types: map[string]dtype.DataType{"a": dtype.Binary}},
			cols: []string{"a binary [hello]"},
		},
		{
			name: "a value that will not parse can be dropped",
			in:   "{\"a\":1}\n{\"a\":\"x\"}\n{\"a\":3}\n",
			opts: &Options{
				Types:             map[string]dtype.DataType{"a": dtype.Int64},
				IgnoreParseErrors: true,
			},
			cols: []string{"a int64 [1 . 3]"},
		},
		{
			name: "a member the sample has not got can be dropped",
			in:   "{\"a\":1}\n{\"a\":2,\"b\":3}\n",
			opts: &Options{InferRows: 1, IgnoreUnknownFields: true},
			cols: []string{"a int64 [1 2]"},
		},
		{
			name: "the whole file is the sample when the rows are asked for",
			in:   "{\"a\":1}\n{\"a\":2,\"b\":3}\n",
			opts: &Options{InferRows: -1},
			cols: []string{"a int64 [1 2]", "b int64 [. 3]"},
		},
		{
			name: "a file longer than the sample",
			in:   "{\"a\":1}\n{\"a\":2}\n{\"a\":3}\n{\"a\":4}\n",
			opts: &Options{InferRows: 2},
			cols: []string{"a int64 [1 2 3 4]"},
		},
		{
			name: "one row per chunk is still one column",
			in:   "{\"a\":1}\n{\"a\":2}\n{\"a\":3}\n",
			opts: &Options{ChunkSize: 1},
			cols: []string{"a int64 [1 2 3]"},
		},
		{
			name: "a file of one line",
			in:   "{\"a\":1}\n",
			cols: []string{"a int64 [1]"},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			want(t, read(t, c.in, c.opts), c.cols...)
		})
	}
}

func TestReadChunks(t *testing.T) {
	var in strings.Builder
	for i := range 10 {
		in.WriteString("{\"a\":" + strconv.Itoa(i) + "}\n")
	}

	tbl := read(t, in.String(), &Options{ChunkSize: 4, InferRows: 2})
	if got := tbl.Columns[0].NumChunks(); got != 3 {
		t.Errorf("got %d chunks, want 3", got)
	}
	if got := tbl.NumRows(); got != 10 {
		t.Fatalf("got %d rows, want 10", got)
	}
	for i := range 10 {
		if got := tbl.Columns[0].Value[int64](i); got != int64(i) {
			t.Errorf("row %d: got %d", i, got)
		}
	}
}

func TestReadLongLine(t *testing.T) {
	// A line longer than the buffer the line reader holds, which is the one that
	// has to be gathered rather than handed out where it sits.
	long := strings.Repeat("x", 3*lineBuffer)
	in := "{\"a\":\"" + long + "\",\"b\":1}\n{\"a\":\"short\",\"b\":2}\n"

	tbl := read(t, in, nil)
	if got := tbl.Columns[0].Bytes(0); string(got) != long {
		t.Errorf("got %d bytes, want %d", len(got), len(long))
	}
	want(t, tbl, "a string ["+long+" short]", "b int64 [1 2]")
}

func TestReadErrors(t *testing.T) {
	cases := []struct {
		name string
		in   string
		opts *Options
		is   error
		msg  string
	}{
		{
			name: "nothing at all",
			in:   "",
			is:   ErrNoData,
		},
		{
			name: "nothing but blank lines",
			in:   "\n\n  \n",
			is:   ErrNoData,
		},
		{
			name: "objects with nothing in them",
			in:   "{}\n{}\n",
			is:   ErrNoData,
		},
		{
			name: "a line that is not JSON",
			in:   "{\"a\":}\n",
			is:   ErrSyntax,
			msg:  "ndjson: line 1: ",
		},
		{
			name: "a line that is JSON but not an object",
			in:   "[1,2]\n",
			is:   ErrSyntax,
			msg:  "ndjson: line 1: a JSON array where an object should be: malformed JSON",
		},
		{
			name: "a line that is a bare value",
			in:   "{\"a\":1}\n5\n",
			is:   ErrSyntax,
			msg:  "ndjson: line 2: a JSON number where an object should be: malformed JSON",
		},
		{
			name: "a line that stops in the middle",
			in:   "{\"a\":1\n",
			is:   ErrSyntax,
			msg:  "ndjson: line 1: the line stops in the middle of the object: malformed JSON",
		},
		{
			name: "more than one object on a line",
			in:   "{\"a\":1} {\"a\":2}\n",
			is:   ErrSyntax,
			msg:  "ndjson: line 1: more than one value on the line: malformed JSON",
		},
		{
			name: "the same member twice on a line",
			in:   "{\"a\":1,\"a\":2}\n",
			is:   ErrSyntax,
			msg:  "ndjson: line 1: the member \"a\" is on the line twice: malformed JSON",
		},
		{
			name: "the same member twice, the second time as null",
			in:   "{\"a\":1,\"a\":null}\n",
			is:   ErrSyntax,
			msg:  "ndjson: line 1: the member \"a\" is on the line twice: malformed JSON",
		},
		{
			name: "the same member twice with the escapes written differently",
			in:   "{\"a\":1,\"\\u0061\":2}\n",
			is:   ErrSyntax,
			msg:  "ndjson: line 1: the member \"a\" is on the line twice: malformed JSON",
		},
		{
			name: "a member the sample has not got",
			in:   "{\"a\":1}\n{\"a\":2,\"b\":3}\n",
			opts: &Options{InferRows: 1},
			is:   ErrUnknownField,
			msg:  "ndjson: line 2: the member \"b\" is not a column: unknown member",
		},
		{
			name: "a value of the wrong kind",
			in:   "{\"a\":1}\n{\"a\":true}\n",
			opts: &Options{InferRows: 1},
			is:   ErrValue,
			msg:  "ndjson: line 2, column \"a\": cannot read true as int64: not a number",
		},
		{
			name: "a number that is not an integer in an integer column",
			in:   "{\"a\":1}\n{\"a\":2.5}\n",
			opts: &Options{InferRows: 1},
			is:   strconv.ErrSyntax,
			msg:  "ndjson: line 2, column \"a\": cannot read 2.5 as int64: invalid syntax",
		},
		{
			name: "a number too large for the column",
			in:   "{\"a\":1}\n{\"a\":300}\n",
			opts: &Options{Types: map[string]dtype.DataType{"a": dtype.Int8}},
			is:   strconv.ErrRange,
			msg:  "ndjson: line 2, column \"a\": cannot read 300 as int8: value out of range",
		},
		{
			name: "a negative number in an unsigned column",
			in:   "{\"a\":-1}\n",
			opts: &Options{Types: map[string]dtype.DataType{"a": dtype.Uint8}},
			is:   ErrValue,
		},
		{
			name: "a float too large for a float32",
			in:   "{\"a\":1e300}\n",
			opts: &Options{Types: map[string]dtype.DataType{"a": dtype.Float32}},
			is:   strconv.ErrRange,
		},
		{
			name: "a quoted value that is not a number",
			in:   "{\"a\":\"x\"}\n",
			opts: &Options{Types: map[string]dtype.DataType{"a": dtype.Int64}},
			is:   strconv.ErrSyntax,
			msg:  "ndjson: line 1, column \"a\": cannot read \"x\" as int64: invalid syntax",
		},
		{
			name: "an object in a column of numbers",
			in:   "{\"a\":{\"b\":1}}\n",
			opts: &Options{Types: map[string]dtype.DataType{"a": dtype.Float64}},
			is:   ErrValue,
			msg:  "ndjson: line 1, column \"a\": cannot read {\"b\":1} as float64: not a number",
		},
		{
			name: "a value that is not base64",
			in:   "{\"a\":\"!\"}\n",
			opts: &Options{Types: map[string]dtype.DataType{"a": dtype.Binary}},
			is:   ErrValue,
		},
		{
			name: "a number in a binary column",
			in:   "{\"a\":1}\n",
			opts: &Options{Types: map[string]dtype.DataType{"a": dtype.Binary}},
			is:   ErrValue,
			msg:  "ndjson: line 1, column \"a\": cannot read 1 as binary: not a string",
		},
		{
			name: "a value of the wrong kind in a boolean column",
			in:   "{\"a\":1}\n",
			opts: &Options{Types: map[string]dtype.DataType{"a": dtype.Bool}},
			is:   ErrValue,
			msg:  "ndjson: line 1, column \"a\": cannot read 1 as bool: not a boolean",
		},
		{
			name: "a quoted value that is not a boolean",
			in:   "{\"a\":\"yes\"}\n",
			opts: &Options{Types: map[string]dtype.DataType{"a": dtype.Bool}},
			is:   strconv.ErrSyntax,
		},
		{
			name: "a type the reader cannot read JSON into",
			in:   "{\"a\":1}\n",
			opts: &Options{Types: map[string]dtype.DataType{"a": dtype.Timestamp{Unit: dtype.Second}}},
			is:   ErrUnsupportedType,
		},
		{
			name: "a column that is not in the file",
			in:   "{\"a\":1}\n",
			opts: &Options{Columns: []string{"b"}},
			is:   ErrNoColumn,
			msg:  "ndjson: no member named \"b\" in the sample: no such column",
		},
		{
			name: "a type for a column that is not in the file",
			in:   "{\"a\":1}\n",
			opts: &Options{Types: map[string]dtype.DataType{"b": dtype.Int64}},
			is:   ErrNoColumn,
		},
		{
			name: "the same column asked for twice",
			in:   "{\"a\":1}\n",
			opts: &Options{Columns: []string{"a", "a"}},
			is:   ErrNames,
			msg:  "ndjson: column \"a\" asked for twice: bad column names",
		},
		{
			name: "a number too large for a float64 in a column declared as one",
			in:   "{\"f\":1e999}\n",
			opts: &Options{Types: map[string]dtype.DataType{"f": dtype.Float64}},
			is:   strconv.ErrRange,
			msg:  "ndjson: line 1, column \"f\": cannot read 1e999 as float64: value out of range",
		},
		{
			name: "a quoted number too large for the column it was declared in",
			in:   "{\"n\":\"99999999999999999999\"}\n",
			opts: &Options{Types: map[string]dtype.DataType{"n": dtype.Int64}},
			is:   strconv.ErrRange,
		},
		{
			name: "a quoted value that is not a number at all",
			in:   "{\"f\":\"lots\"}\n",
			opts: &Options{Types: map[string]dtype.DataType{"f": dtype.Float64}},
			is:   strconv.ErrSyntax,
		},
		{
			name: "a quoted value in an unsigned column that is not one",
			in:   "{\"u\":\"-1\"}\n",
			opts: &Options{Types: map[string]dtype.DataType{"u": dtype.Uint64}},
			is:   strconv.ErrSyntax,
		},
		{
			name: "a value of the wrong kind in an unsigned column",
			in:   "{\"u\":true}\n",
			opts: &Options{Types: map[string]dtype.DataType{"u": dtype.Uint64}},
			is:   ErrValue,
			msg:  "ndjson: line 1, column \"u\": cannot read true as uint64: not a number",
		},
		{
			name: "a number too large for the unsigned column it was declared in",
			in:   "{\"u\":300}\n",
			opts: &Options{Types: map[string]dtype.DataType{"u": dtype.Uint8}},
			is:   strconv.ErrRange,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := Read(strings.NewReader(c.in), c.opts)
			if err == nil {
				t.Fatal("read a file that should not have been read")
			}
			if !errors.Is(err, c.is) {
				t.Errorf("got %v, want %v", err, c.is)
			}
			if c.msg == "" {
				return
			}
			if got := err.Error(); !strings.HasPrefix(got, c.msg) {
				t.Errorf("got %q, want it to start with %q", got, c.msg)
			}
		})
	}
}

// TestReadErrorLines checks that the line in an error is the line in the file,
// which is the whole point of carrying one.
func TestReadErrorLines(t *testing.T) {
	in := "\n{\"a\":1}\n\n\n{\"a\":true}\n"
	_, err := Read(strings.NewReader(in), &Options{InferRows: 1})

	var ve *ValueError
	if !errors.As(err, &ve) {
		t.Fatalf("got %v, want a *ValueError", err)
	}
	if ve.Line != 5 {
		t.Errorf("got line %d, want 5", ve.Line)
	}
	if ve.Column != "a" || ve.Value != "true" || ve.Type != "int64" {
		t.Errorf("got %+v", ve)
	}
}

// TestReadUnwrap checks that a *ValueError answers about both the package error
// and the one underneath it.
func TestReadUnwrap(t *testing.T) {
	_, err := Read(strings.NewReader("{\"a\":2.5}\n"),
		&Options{Types: map[string]dtype.DataType{"a": dtype.Int64}})

	for _, target := range []error{ErrValue, strconv.ErrSyntax} {
		if !errors.Is(err, target) {
			t.Errorf("got %v, want it to be %v", err, target)
		}
	}
	if errors.Is(err, ErrSyntax) {
		t.Error("a bad value is not a bad line")
	}

	_, err = Read(strings.NewReader("[]\n"), nil)
	if errors.Is(err, ErrValue) {
		t.Error("a bad line is not a bad value")
	}
}

func TestReadFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "trades.ndjson")
	in := "{\"sym\":\"AAPL\",\"qty\":100}\n{\"sym\":\"MSFT\",\"qty\":200}\n"
	if err := os.WriteFile(path, []byte(in), 0o600); err != nil {
		t.Fatal(err)
	}

	tbl, err := ReadFile(path, nil)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	want(t, tbl, "sym string [AAPL MSFT]", "qty int64 [100 200]")

	if _, err = ReadFile(filepath.Join(dir, "gone.ndjson"), nil); err == nil {
		t.Error("read a file that is not there")
	}

	bad := filepath.Join(dir, "bad.ndjson")
	if err = os.WriteFile(bad, []byte("[]\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = ReadFile(bad, nil)
	if !errors.Is(err, ErrSyntax) {
		t.Errorf("got %v, want %v", err, ErrSyntax)
	}
	if got := err.Error(); !strings.Contains(got, "bad.ndjson") {
		t.Errorf("got %q, want the name of the file in it", got)
	}
}

// failing is a reader that stops with an error part way through, which is the
// one thing a file cannot do.
type failing struct {
	in   string
	at   int
	fail error
}

func (f *failing) Read(p []byte) (int, error) {
	if f.at >= len(f.in) {
		return 0, f.fail
	}
	n := copy(p, f.in[f.at:])
	f.at += n
	return n, nil
}

func TestReadBroken(t *testing.T) {
	fail := errors.New("the disk went away")

	for _, c := range []struct {
		name string
		opts *Options
	}{
		{name: "while the sample is being read"},
		{name: "after it", opts: &Options{InferRows: 1}},
	} {
		t.Run(c.name, func(t *testing.T) {
			src := &failing{in: "{\"a\":1}\n{\"a\":2}\n", fail: fail}
			if _, err := Read(src, c.opts); !errors.Is(err, fail) {
				t.Errorf("got %v, want %v", err, fail)
			}
		})
	}
}

// TestReadErrorsAfterTheSample is every way a line can go wrong, on a line the
// sample did not reach.
//
// The reader walks a line twice on the way in, once to work out what the
// columns are and once to put the values in them, and after the sample only the
// second walk happens. The two report the same things, and this is the half
// that the tests above, which are all inside the sample, do not reach.
func TestReadErrorsAfterTheSample(t *testing.T) {
	types := func(name string, dt dtype.DataType) *Options {
		return &Options{InferRows: 1, Types: map[string]dtype.DataType{name: dt}}
	}

	cases := []struct {
		name string
		in   string
		opts *Options
		msg  string
	}{
		{
			name: "a line that is not an object",
			in:   "{\"a\":1}\n[1]\n",
			msg:  "ndjson: line 2: a JSON array where an object should be: malformed JSON",
		},
		{
			name: "a line that is not JSON at all",
			in:   "{\"a\":1}\n@\n",
		},
		{
			name: "a member name that is not a string this reader can read",
			in:   "{\"a\":1}\n{\"\\ud800\":2}\n",
		},
		{
			name: "a second value on the line",
			in:   "{\"a\":1}\n{\"a\":2} {\"a\":3}\n",
			msg:  "ndjson: line 2: more than one value on the line: malformed JSON",
		},
		{
			name: "junk after the object",
			in:   "{\"a\":1}\n{\"a\":2} @\n",
		},
		{
			name: "a null that is not spelled null",
			in:   "{\"a\":1}\n{\"a\":nul}\n",
		},
		{
			name: "a broken value where nothing is reading it",
			in:   "{\"a\":1,\"b\":2}\n{\"a\":2,\"b\":[1,]}\n",
			opts: &Options{InferRows: 1, Columns: []string{"a"}},
		},
		{
			name: "a broken value the column will not take either",
			in:   "{\"a\":1}\n{\"a\":[1,]}\n",
		},
		{
			name: "a broken string in a boolean column",
			in:   "{\"b\":true}\n{\"b\":\"\\ud800\"}\n",
		},
		{
			name: "a boolean that is not spelled out",
			in:   "{\"b\":true}\n{\"b\":tru}\n",
		},
		{
			name: "a broken string in a signed column",
			in:   "{\"n\":1}\n{\"n\":\"\\ud800\"}\n",
		},
		{
			name: "a broken number in a signed column",
			in:   "{\"n\":1}\n{\"n\":1.}\n",
		},
		{
			name: "a broken string in an unsigned column",
			in:   "{\"u\":1}\n{\"u\":\"\\ud800\"}\n",
			opts: types("u", dtype.Uint64),
		},
		{
			name: "a broken number in an unsigned column",
			in:   "{\"u\":1}\n{\"u\":1.}\n",
			opts: types("u", dtype.Uint64),
		},
		{
			name: "a broken string in a float column",
			in:   "{\"f\":1.5}\n{\"f\":\"\\ud800\"}\n",
		},
		{
			name: "a broken number in a float column",
			in:   "{\"f\":1.5}\n{\"f\":1.}\n",
		},
		{
			name: "a broken string in a string column",
			in:   "{\"s\":\"a\"}\n{\"s\":\"\\ud800\"}\n",
		},
		{
			name: "a broken string in a binary column",
			in:   "{\"bin\":\"aGk=\"}\n{\"bin\":\"\\ud800\"}\n",
			opts: types("bin", dtype.Binary),
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			opts := c.opts
			if opts == nil {
				opts = &Options{InferRows: 1}
			}

			_, err := Read(strings.NewReader(c.in), opts)
			if err == nil {
				t.Fatal("read a file that should not have been read")
			}
			if !errors.Is(err, ErrSyntax) {
				t.Errorf("got %v, want ErrSyntax", err)
			}
			if c.msg != "" && err.Error() != c.msg {
				t.Errorf("got %q, want %q", err.Error(), c.msg)
			}
			if !strings.HasPrefix(err.Error(), "ndjson: line 2: ") {
				t.Errorf("got %q, want it to say which line", err.Error())
			}
		})
	}
}

// TestReadQuotedNumbers is the permissive reading a declared column gets, which
// inference never asks for and which files that quote everything need.
func TestReadQuotedNumbers(t *testing.T) {
	in := "{\"i\":\"-1\",\"u\":\"1\",\"f\":\"1.5\",\"b\":\"true\"}\n" +
		"{\"i\":\"2\",\"u\":\"2\",\"f\":\"2\",\"b\":\"false\"}\n"

	tbl := read(t, in, &Options{Types: map[string]dtype.DataType{
		"i": dtype.Int32,
		"u": dtype.Uint16,
		"f": dtype.Float32,
		"b": dtype.Bool,
	}})
	want(t, tbl, "i int32 [-1 2]", "u uint16 [1 2]", "f float32 [1.5 2]", "b bool [true false]")
}

// TestReadFileEndingInSpaces is the file whose last line has nothing on it but
// whitespace and no newline after it, which is the one way the line reader can
// run out on a line it was going to hand over.
func TestReadFileEndingInSpaces(t *testing.T) {
	tbl := read(t, "{\"a\":1}\n{\"a\":2}\n   ", nil)
	want(t, tbl, "a int64 [1 2]")
}

// TestErrorTypes is the two error types on their own, since both are exported
// and a caller can build one as well as read one.
//
// The message and what it unwraps to are the whole of what they promise, and a
// ValueError with nothing wrapped is the case the reader never produces and the
// documented zero value allows.
func TestErrorTypes(t *testing.T) {
	line := &LineError{Line: 12, Err: ErrSyntax}
	if got := line.Error(); got != "ndjson: line 12: malformed JSON" {
		t.Errorf("got %q", got)
	}
	if !errors.Is(line, ErrSyntax) {
		t.Error("a LineError does not unwrap to what went wrong")
	}

	val := &ValueError{Line: 4, Column: "qty", Type: "int64", Value: "3.9", Err: strconv.ErrSyntax}
	want := "ndjson: line 4, column \"qty\": cannot read 3.9 as int64: invalid syntax"
	if got := val.Error(); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
	if !errors.Is(val, ErrValue) || !errors.Is(val, strconv.ErrSyntax) {
		t.Error("a ValueError does not unwrap to both of the things it is")
	}

	bare := &ValueError{Line: 4, Column: "qty", Type: "int64", Value: "3.9"}
	if got := bare.Error(); got != "ndjson: line 4, column \"qty\": cannot read 3.9 as int64" {
		t.Errorf("got %q", got)
	}
	if !errors.Is(bare, ErrValue) {
		t.Error("a ValueError with nothing wrapped is still a bad value")
	}
}
