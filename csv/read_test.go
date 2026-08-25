package csv

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
			name: "types are inferred from the values",
			in:   "sym,qty,px,live\nAAPL,100,1.5,true\nMSFT,200,2,false\n",
			cols: []string{
				"sym string [AAPL MSFT]",
				"qty int64 [100 200]",
				"px float64 [1.5 2]",
				"live bool [true false]",
			},
		},
		{
			name: "an empty field is missing in every column",
			in:   "sym,qty,px\n,1,1.5\nMSFT,,2.5\nGOOG,3,\n",
			cols: []string{
				"sym string [. MSFT GOOG]",
				"qty int64 [1 . 3]",
				"px float64 [1.5 2.5 .]",
			},
		},
		{
			name: "an integer among floats is a float column",
			in:   "px\n1\n2.5\n3\n",
			cols: []string{"px float64 [1 2.5 3]"},
		},
		{
			name: "anything else mixed is a string column",
			in:   "x\n1\ntrue\n",
			cols: []string{"x string [1 true]"},
		},
		{
			name: "ones and zeros are numbers rather than booleans",
			in:   "flag\n1\n0\n",
			cols: []string{"flag int64 [1 0]"},
		},
		{
			name: "a single letter is not a boolean",
			in:   "grade\nT\nF\n",
			cols: []string{"grade string [T F]"},
		},
		{
			name: "true and false are, in any case",
			in:   "live\nTRUE\nFalse\n",
			cols: []string{"live bool [true false]"},
		},
		{
			name: "a number too large for an int64 is a float",
			in:   "n\n99999999999999999999\n1\n",
			cols: []string{"n float64 [1e+20 1]"},
		},
		{
			name: "a column with nothing in it is a string column",
			in:   "a,b\n1,\n2,\n",
			cols: []string{"a int64 [1 2]", "b string [. .]"},
		},
		{
			name: "a header and no rows is a table with no rows",
			in:   "sym,qty\n",
			cols: []string{"sym string []", "qty string []"},
		},
		{
			name: "quotes hold delimiters, newlines and quotes",
			in:   "note,n\n\"a,b\",1\n\"line\none\",2\n\"say \"\"hi\"\"\",3\n",
			cols: []string{"note string [a,b line\none say \"hi\"]", "n int64 [1 2 3]"},
		},
		{
			name: "carriage returns are line endings",
			in:   "a,b\r\n1,2\r\n3,4\r\n",
			cols: []string{"a int64 [1 3]", "b int64 [2 4]"},
		},
		{
			name: "an empty header cell still gets a name",
			in:   "sym,,qty\nAAPL,x,1\n",
			cols: []string{"sym string [AAPL]", "column_2 string [x]", "qty int64 [1]"},
		},
		{
			name: "no header names the columns by position",
			in:   "AAPL,100\nMSFT,200\n",
			opts: &Options{NoHeader: true},
			cols: []string{"column_1 string [AAPL MSFT]", "column_2 int64 [100 200]"},
		},
		{
			name: "names replace the ones in the file",
			in:   "a,b\n1,2\n",
			opts: &Options{Names: []string{"sym", "qty"}},
			cols: []string{"sym int64 [1]", "qty int64 [2]"},
		},
		{
			name: "names supply the ones a headerless file has not got",
			in:   "1,2\n",
			opts: &Options{NoHeader: true, Names: []string{"sym", "qty"}},
			cols: []string{"sym int64 [1]", "qty int64 [2]"},
		},
		{
			name: "another delimiter",
			in:   "a;b\n1;2\n",
			opts: &Options{Delimiter: ';'},
			cols: []string{"a int64 [1]", "b int64 [2]"},
		},
		{
			name: "a tab, which is the other common one",
			in:   "a\tb\n1\t2\n",
			opts: &Options{Delimiter: '\t'},
			cols: []string{"a int64 [1]", "b int64 [2]"},
		},
		{
			name: "comments are skipped wherever they are",
			in:   "# written by hand\na,b\n1,2\n# and again\n3,4\n",
			opts: &Options{Comment: '#'},
			cols: []string{"a int64 [1 3]", "b int64 [2 4]"},
		},
		{
			name: "skip throws away the lines before the header",
			in:   "report of the day\nand a blank one below\n\na,b\n1,2\n",
			opts: &Options{Skip: 2},
			cols: []string{"a int64 [1]", "b int64 [2]"},
		},
		{
			name: "leading space can be trimmed",
			in:   "a, b\n1, 2\n",
			opts: &Options{TrimLeadingSpace: true},
			cols: []string{"a int64 [1]", "b int64 [2]"},
		},
		{
			name: "leading space is part of the value otherwise",
			in:   "a,b\n1, 2\n",
			cols: []string{"a int64 [1]", "b string [ 2]"},
		},
		{
			name: "null values say what missing looks like in this file",
			in:   "sym,qty\nNA,1\nMSFT,NA\n",
			opts: &Options{NullValues: []string{"NA"}},
			cols: []string{"sym string [. MSFT]", "qty int64 [1 .]"},
		},
		{
			name: "a null list without the empty string keeps empty strings",
			in:   "sym,n\n,1\nMSFT,2\n",
			opts: &Options{NullValues: []string{"NA"}},
			cols: []string{"sym string [ MSFT]", "n int64 [1 2]"},
		},
		{
			name: "a value that will not parse can be dropped instead",
			in:   "qty\n1\nlots\n3\n",
			opts: &Options{Types: map[string]dtype.DataType{"qty": dtype.Int64}, IgnoreParseErrors: true},
			cols: []string{"qty int64 [1 . 3]"},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			want(t, read(t, c.in, c.opts), c.cols...)
		})
	}
}

func TestReadTypes(t *testing.T) {
	in := "i8,i16,i32,i64,u8,u16,u32,u64,f32,f64,b,s,raw\n" +
		"-1,-2,-3,-4,1,2,3,4,1.5,2.5,t,hello,bytes\n"

	tbl := read(t, in, &Options{Types: map[string]dtype.DataType{
		"i8": dtype.Int8, "i16": dtype.Int16, "i32": dtype.Int32, "i64": dtype.Int64,
		"u8": dtype.Uint8, "u16": dtype.Uint16, "u32": dtype.Uint32, "u64": dtype.Uint64,
		"f32": dtype.Float32, "f64": dtype.Float64,
		"b": dtype.Bool, "s": dtype.String, "raw": dtype.Binary,
	}})

	want(t, tbl,
		"i8 int8 [-1]", "i16 int16 [-2]", "i32 int32 [-3]", "i64 int64 [-4]",
		"u8 uint8 [1]", "u16 uint16 [2]", "u32 uint32 [3]", "u64 uint64 [4]",
		"f32 float32 [1.5]", "f64 float64 [2.5]",
		"b bool [true]", "s string [hello]", "raw binary [bytes]",
	)
}

func TestReadTypesAreMoreWillingThanInference(t *testing.T) {
	// The single letters are a string when the reader has to guess and a
	// boolean when it has been told.
	tbl := read(t, "flag\nt\nf\n", &Options{Types: map[string]dtype.DataType{"flag": dtype.Bool}})
	want(t, tbl, "flag bool [true false]")
}

func TestReadTypesSkipsInference(t *testing.T) {
	// Every column is named, so nothing is held to look at. A file that is one
	// row longer than the sample would be would still come out whole.
	tbl := read(t, "a\n1\n2\n3\n", &Options{
		Types:     map[string]dtype.DataType{"a": dtype.Float64},
		InferRows: 1,
	})
	want(t, tbl, "a float64 [1 2 3]")
}

func TestReadInferRows(t *testing.T) {
	// The float is past the rows the reader looked at, so the column was
	// decided to be an integer column and the float does not fit in it.
	in := "px\n1\n2\n3\n4.5\n"
	_, err := Read(strings.NewReader(in), &Options{InferRows: 2})
	if !errors.Is(err, ErrValue) {
		t.Fatalf("got %v, want an ErrValue", err)
	}

	// Reading the whole file before deciding gets it right whatever the shape
	// of the file, at the cost of holding all of it.
	want(t, read(t, in, &Options{InferRows: -1}), "px float64 [1 2 3 4.5]")
}

func TestReadChunks(t *testing.T) {
	in := "a,b\n1,x\n2,y\n3,z\n4,w\n5,v\n"
	tbl := read(t, in, &Options{ChunkSize: 2})

	want(t, tbl, "a int64 [1 2 3 4 5]", "b string [x y z w v]")
	for i, c := range tbl.Columns {
		if got := c.NumChunks(); got != 3 {
			t.Errorf("column %d is in %d chunks, want 3", i, got)
		}
	}
}

func TestReadChunkExactlyFull(t *testing.T) {
	// The last row fills the chunk, so the flush at the end has nothing left
	// to do and must not add an empty chunk.
	tbl := read(t, "a\n1\n2\n3\n4\n", &Options{ChunkSize: 2})
	want(t, tbl, "a int64 [1 2 3 4]")
	if got := tbl.Columns[0].NumChunks(); got != 2 {
		t.Errorf("got %d chunks, want 2", got)
	}
}

func TestReadSchema(t *testing.T) {
	tbl := read(t, "sym,qty\nAAPL,1\nMSFT,\n", nil)

	fields := tbl.Schema.Fields
	if len(fields) != 2 {
		t.Fatalf("got %d fields, want 2", len(fields))
	}
	if fields[0].Nullable {
		t.Error("sym has nothing missing and is nullable")
	}
	if !fields[1].Nullable {
		t.Error("qty has a value missing and is not nullable")
	}
	if got := tbl.NumRows(); got != 2 {
		t.Errorf("got %d rows, want 2", got)
	}
	if got := tbl.NumCols(); got != 2 {
		t.Errorf("got %d columns, want 2", got)
	}
}

func TestReadValueError(t *testing.T) {
	// The sample stops before the bad row, so the column was decided to be an
	// integer column and then the file disagreed. Without the limit the reader
	// would have seen the whole file and called the column a string.
	in := "sym,qty\nAAPL,1\nMSFT,2\nGOOG,lots\n"
	_, err := Read(strings.NewReader(in), &Options{InferRows: 2})

	var ve *ValueError
	if !errors.As(err, &ve) {
		t.Fatalf("got %v, want a *ValueError", err)
	}
	if ve.Line != 4 {
		t.Errorf("got line %d, want 4", ve.Line)
	}
	if ve.Column != "qty" {
		t.Errorf("got column %q, want qty", ve.Column)
	}
	if ve.Value != "lots" {
		t.Errorf("got value %q, want lots", ve.Value)
	}
	if !errors.Is(err, ErrValue) {
		t.Error("does not unwrap to ErrValue")
	}
	if !errors.Is(err, strconv.ErrSyntax) {
		t.Error("does not unwrap to what strconv said")
	}

	msg := `csv: line 4, column "qty": cannot read "lots" as int64: invalid syntax`
	if err.Error() != msg {
		t.Errorf("got %s\nwant %s", err, msg)
	}
}

func TestReadValueErrorLineWithQuotedNewlines(t *testing.T) {
	// The row starts on line 3 and runs onto line 4, and the line reported is
	// the one the row starts on.
	in := "note,qty\na,1\n\"two\nlines\",lots\n"
	_, err := Read(strings.NewReader(in), &Options{
		Types: map[string]dtype.DataType{"qty": dtype.Int64},
	})

	var ve *ValueError
	if !errors.As(err, &ve) {
		t.Fatalf("got %v, want a *ValueError", err)
	}
	if ve.Line != 3 {
		t.Errorf("got line %d, want 3", ve.Line)
	}
}

func TestReadValueErrorInTheSample(t *testing.T) {
	// A value can only be wrong in the sample when the type was given, since
	// otherwise the sample is what decided the type. The line still has to be
	// right, and by then the csv reader is nowhere near that row.
	in := "qty\n1\n2\nlots\n"
	_, err := Read(strings.NewReader(in), &Options{
		Types: map[string]dtype.DataType{"qty": dtype.Int64},
	})

	var ve *ValueError
	if !errors.As(err, &ve) {
		t.Fatalf("got %v, want a *ValueError", err)
	}
	if ve.Line != 4 {
		t.Errorf("got line %d, want 4", ve.Line)
	}
}

func TestReadValueOutOfRange(t *testing.T) {
	_, err := Read(strings.NewReader("n\n300\n"), &Options{
		Types: map[string]dtype.DataType{"n": dtype.Uint8},
	})
	if !errors.Is(err, strconv.ErrRange) {
		t.Fatalf("got %v, want a range error", err)
	}
}

func TestReadNotUTF8(t *testing.T) {
	_, err := Read(strings.NewReader("s\n\xff\xfe\n"), nil)
	if !errors.Is(err, ErrValue) {
		t.Fatalf("got %v, want an ErrValue", err)
	}
	if !strings.Contains(err.Error(), "UTF-8") {
		t.Errorf("got %v, want it to say what is wrong", err)
	}

	// The same bytes are a binary column without complaint.
	tbl := read(t, "s\n\xff\xfe\n", &Options{Types: map[string]dtype.DataType{"s": dtype.Binary}})
	if got := tbl.Columns[0].Bytes(0); string(got) != "\xff\xfe" {
		t.Errorf("got %q, want the bytes back", got)
	}
}

func TestReadErrors(t *testing.T) {
	cases := []struct {
		name string
		in   string
		opts *Options
		is   error
	}{
		{
			name: "nothing at all",
			in:   "",
			is:   ErrNoData,
		},
		{
			name: "nothing but comments",
			in:   "# all of it\n",
			opts: &Options{Comment: '#'},
			is:   ErrNoData,
		},
		{
			name: "skipping past the end",
			in:   "a\n1\n",
			opts: &Options{Skip: 5},
			is:   ErrNoData,
		},
		{
			name: "two columns with the same name",
			in:   "a,a\n1,2\n",
			is:   ErrNames,
		},
		{
			name: "not enough names",
			in:   "a,b\n1,2\n",
			opts: &Options{Names: []string{"only"}},
			is:   ErrNames,
		},
		{
			name: "a type for a column that is not there",
			in:   "a\n1\n",
			opts: &Options{Types: map[string]dtype.DataType{"b": dtype.Int64}},
			is:   ErrNoColumn,
		},
		{
			name: "a type text cannot be read into",
			in:   "a\n1\n",
			opts: &Options{Types: map[string]dtype.DataType{"a": dtype.List{Elem: dtype.Int64}}},
			is:   ErrUnsupportedType,
		},
		{
			name: "a row with the wrong number of fields",
			in:   "a,b\n1,2\n3\n",
			is:   ErrFieldCount,
		},
		{
			name: "a quote where the format does not allow one",
			in:   "a\n\"b\"c\n",
			is:   ErrQuote,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := Read(strings.NewReader(c.in), c.opts)
			if !errors.Is(err, c.is) {
				t.Fatalf("got %v, want %v", err, c.is)
			}
		})
	}
}

func TestReadLazyQuotes(t *testing.T) {
	// The same input the strict reader refuses above. The field is still
	// quoted as far as the reader is concerned, so it runs to the end of the
	// line and the line ending is part of the value.
	tbl := read(t, "a\n\"b\"c\n", &Options{LazyQuotes: true})
	if got := string(tbl.Columns[0].Bytes(0)); got != "b\"c\n" {
		t.Errorf("got %q, want %q", got, "b\"c\n")
	}
}

func TestReadFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trades.csv")
	if err := os.WriteFile(path, []byte("sym,qty\nAAPL,100\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	tbl, err := ReadFile(path, nil)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	want(t, tbl, "sym string [AAPL]", "qty int64 [100]")
}

func TestReadFileErrors(t *testing.T) {
	dir := t.TempDir()

	if _, err := ReadFile(filepath.Join(dir, "nope.csv"), nil); !errors.Is(err, os.ErrNotExist) {
		t.Errorf("got %v, want a not exist error", err)
	}

	// A bad value names the file, since a program reading a directory of them
	// has to know which one to go and look at.
	path := filepath.Join(dir, "trades.csv")
	if err := os.WriteFile(path, []byte("qty\n1\nlots\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := ReadFile(path, &Options{Types: map[string]dtype.DataType{"qty": dtype.Int64}})
	if !errors.Is(err, ErrValue) {
		t.Fatalf("got %v, want an ErrValue", err)
	}
	if !strings.Contains(err.Error(), "trades.csv") {
		t.Errorf("got %v, want it to name the file", err)
	}
}

func TestReadLargeText(t *testing.T) {
	// The large offsets are an IPC encoding rather than a way to hold a
	// column, so there is no builder for one and asking for it here says so
	// rather than half working.
	_, err := Read(strings.NewReader("s\nhello\n"), &Options{
		Types: map[string]dtype.DataType{"s": dtype.LargeString},
	})
	if !errors.Is(err, ErrUnsupportedType) {
		t.Fatalf("got %v, want an unsupported type error", err)
	}
}

func TestReadManyRows(t *testing.T) {
	// More rows than the sample and more than one chunk, so the two phases of
	// the read meet in the middle and the row numbering has to survive it.
	const rows = 3000

	var sb strings.Builder
	sb.WriteString("i,s\n")
	for i := range rows {
		sb.WriteString(strconv.Itoa(i))
		sb.WriteString(",row")
		sb.WriteString(strconv.Itoa(i))
		sb.WriteByte('\n')
	}

	tbl := read(t, sb.String(), &Options{ChunkSize: 1000})
	if got := tbl.NumRows(); got != rows {
		t.Fatalf("got %d rows, want %d", got, rows)
	}
	for i := range rows {
		if got := tbl.Columns[0].Value[int64](i); got != int64(i) {
			t.Fatalf("row %d is %d", i, got)
		}
		if got := string(tbl.Columns[1].Bytes(i)); got != "row"+strconv.Itoa(i) {
			t.Fatalf("row %d is %q", i, got)
		}
	}
}

// errReader hands out some bytes and then fails, which is what a network read
// or a bad disk looks like from up here.
type errReader struct {
	data string
	err  error
}

func (r *errReader) Read(p []byte) (int, error) {
	if r.data == "" {
		return 0, r.err
	}
	n := copy(p, r.data)
	r.data = r.data[n:]
	return n, nil
}

func TestReadPassesOnReadErrors(t *testing.T) {
	boom := errors.New("boom")

	// Once while skipping, once while sampling, and once while reading the
	// rest of the file, since those are three different loops.
	cases := []struct {
		name string
		in   string
		opts *Options
	}{
		{name: "in the skip", in: "banner\n", opts: &Options{Skip: 2}},
		{name: "in the sample", in: "a\n1\n"},
		{
			name: "in the rest",
			in:   "a\n1\n",
			opts: &Options{Types: map[string]dtype.DataType{"a": dtype.Int64}},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := Read(&errReader{data: c.in, err: boom}, c.opts)
			if !errors.Is(err, boom) {
				t.Fatalf("got %v, want boom", err)
			}
		})
	}
}

func TestReadFieldCountAfterTheSample(t *testing.T) {
	// The types are given, so nothing is held back and the short row is found
	// by the loop that reads the rest of the file rather than by the sample.
	_, err := Read(strings.NewReader("a,b\n1,2\n3\n"), &Options{
		Types: map[string]dtype.DataType{"a": dtype.Int64, "b": dtype.Int64},
	})
	if !errors.Is(err, ErrFieldCount) {
		t.Fatalf("got %v, want a field count error", err)
	}
}

func TestReadBadValueOfEveryKind(t *testing.T) {
	cases := []struct {
		dt dtype.DataType
		in string
	}{
		{dtype.Bool, "maybe"},
		{dtype.Int64, "1.5"},
		{dtype.Uint64, "-1"},
		{dtype.Float64, "one and a half"},
	}

	for _, c := range cases {
		t.Run(c.dt.String(), func(t *testing.T) {
			_, err := Read(strings.NewReader("v\n"+c.in+"\n"), &Options{
				Types: map[string]dtype.DataType{"v": c.dt},
			})
			if !errors.Is(err, ErrValue) {
				t.Fatalf("got %v, want an ErrValue", err)
			}
		})
	}
}

func TestTableOfNothing(t *testing.T) {
	// A Table is an ordinary struct and a caller can hold an empty one, so it
	// answers rather than panicking.
	var tbl Table
	if tbl.NumRows() != 0 || tbl.NumCols() != 0 {
		t.Errorf("got %d by %d, want nothing", tbl.NumRows(), tbl.NumCols())
	}
}

func TestValueErrorWithoutACause(t *testing.T) {
	// The parse error is what a ValueError usually carries, and a caller
	// building one by hand does not have to.
	err := &ValueError{Line: 2, Column: "qty", Type: "int64", Value: "lots"}

	msg := `csv: line 2, column "qty": cannot read "lots" as int64`
	if err.Error() != msg {
		t.Errorf("got %s\nwant %s", err, msg)
	}
	if !errors.Is(err, ErrValue) {
		t.Error("does not unwrap to ErrValue")
	}
}
