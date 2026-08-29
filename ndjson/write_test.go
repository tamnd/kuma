package ndjson

import (
	"bytes"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// wrote returns what a table writes, which is what most of the tests here
// compare against.
func wrote(t *testing.T, tbl *Table, opts *WriteOptions) string {
	t.Helper()
	var buf bytes.Buffer
	if err := Write(&buf, tbl, opts); err != nil {
		t.Fatalf("Write: %v", err)
	}
	return buf.String()
}

// build returns a one chunk column of dt, for the types the reader does not
// produce and a test has to make by hand.
func build(t *testing.T, dt dtype.DataType, fill func(b *array.Builder)) *array.Chunked {
	t.Helper()
	b, err := array.NewBuilder(dt)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	fill(b)

	c, err := array.NewChunked(dt, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	return c
}

// newTable returns a table of the given columns, with the schema they describe.
func newTable(t *testing.T, names []string, cols ...*array.Chunked) *Table {
	t.Helper()
	if len(names) != len(cols) {
		t.Fatalf("%d names for %d columns", len(names), len(cols))
	}

	fields := make([]dtype.Field, len(cols))
	for i, c := range cols {
		fields[i] = dtype.Field{Name: names[i], Type: c.DType(), Nullable: c.NullCount() > 0}
	}
	return &Table{Schema: dtype.Schema{Fields: fields}, Columns: cols}
}

func TestWrite(t *testing.T) {
	tests := []struct {
		name string
		in   string
		read *Options
		opts *WriteOptions
		want string
	}{
		{
			name: "a file comes back as itself",
			in:   "{\"sym\":\"AAPL\",\"qty\":100,\"px\":1.5,\"live\":true}\n",
			want: "{\"sym\":\"AAPL\",\"qty\":100,\"px\":1.5,\"live\":true}\n",
		},
		{
			name: "the members come out in the order the schema has them",
			in:   "{\"b\":1,\"a\":2}\n{\"a\":3,\"b\":4}\n",
			want: "{\"b\":1,\"a\":2}\n{\"b\":4,\"a\":3}\n",
		},
		{
			name: "a missing value is null",
			in:   "{\"sym\":\"AAPL\"}\n{\"sym\":\"MSFT\",\"qty\":200}\n",
			want: "{\"sym\":\"AAPL\",\"qty\":null}\n{\"sym\":\"MSFT\",\"qty\":200}\n",
		},
		{
			name: "a missing value can be left out instead",
			in:   "{\"sym\":\"AAPL\"}\n{\"sym\":\"MSFT\",\"qty\":200}\n",
			opts: &WriteOptions{OmitNull: true},
			want: "{\"sym\":\"AAPL\"}\n{\"sym\":\"MSFT\",\"qty\":200}\n",
		},
		{
			name: "a row where everything is missing is an empty object",
			in:   "{\"a\":null,\"b\":null}\n{\"a\":1,\"b\":2}\n",
			read: &Options{Types: map[string]dtype.DataType{"a": dtype.Int64, "b": dtype.Int64}},
			opts: &WriteOptions{OmitNull: true},
			want: "{}\n{\"a\":1,\"b\":2}\n",
		},
		{
			name: "names replace the ones in the schema",
			in:   "{\"sym\":\"AAPL\",\"qty\":100}\n",
			opts: &WriteOptions{Names: []string{"symbol", "quantity"}},
			want: "{\"symbol\":\"AAPL\",\"quantity\":100}\n",
		},
		{
			name: "a name that needs escaping gets it",
			in:   "{\"a\\\"b\":1}\n",
			want: "{\"a\\\"b\":1}\n",
		},
		{
			name: "a quote in a value is escaped",
			in:   "{\"note\":\"say \\\"hi\\\"\"}\n",
			want: "{\"note\":\"say \\\"hi\\\"\"}\n",
		},
		{
			name: "a newline in a value is escaped",
			in:   "{\"note\":\"a\\nb\"}\n",
			want: "{\"note\":\"a\\nb\"}\n",
		},
		{
			name: "text outside ASCII goes out as itself",
			in:   "{\"note\":\"\\u3042\"}\n",
			want: "{\"note\":\"\u3042\"}\n",
		},
		{
			name: "a float keeps the digits it needs and no more",
			in:   "{\"px\":0.1}\n{\"px\":1e10}\n{\"px\":123456789.123}\n",
			want: "{\"px\":0.1}\n{\"px\":10000000000}\n{\"px\":123456789.123}\n",
		},
		{
			name: "floats can be written at a fixed precision",
			in:   "{\"px\":1.5}\n{\"px\":2.25}\n",
			want: "{\"px\":1.500}\n{\"px\":2.250}\n",
			opts: &WriteOptions{Precision: 3},
		},
		{
			name: "a nested object goes out as the object it was",
			in:   "{\"at\":{\"x\":1,\"y\":2}}\n",
			want: "{\"at\":\"{\\\"x\\\":1,\\\"y\\\":2}\"}\n",
		},
		{
			name: "an integer keeps every digit it had",
			in:   "{\"n\":9007199254740993}\n",
			want: "{\"n\":9007199254740993}\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tbl := read(t, tt.in, tt.read)
			if got := wrote(t, tbl, tt.opts); got != tt.want {
				t.Errorf("got\n%q\nwant\n%q", got, tt.want)
			}
		})
	}
}

func TestWriteEveryType(t *testing.T) {
	const in = "{\"i8\":-8,\"i16\":-16,\"i32\":-32,\"i64\":-64," +
		"\"u8\":8,\"u16\":16,\"u32\":32,\"u64\":64," +
		"\"f32\":1.5,\"f64\":2.5,\"b\":true,\"s\":\"hi\",\"bin\":\"Ynll\"}\n" +
		"{\"i8\":127,\"i16\":-32768,\"i32\":2147483647,\"i64\":-9223372036854775808," +
		"\"u8\":255,\"u16\":65535,\"u32\":4294967295,\"u64\":18446744073709551615," +
		"\"f32\":-0.25,\"f64\":1e+100,\"b\":false,\"s\":\"\",\"bin\":\"eA==\"}\n"

	tbl := read(t, in, &Options{Types: map[string]dtype.DataType{
		"i8":  dtype.Int8,
		"i16": dtype.Int16,
		"i32": dtype.Int32,
		"i64": dtype.Int64,
		"u8":  dtype.Uint8,
		"u16": dtype.Uint16,
		"u32": dtype.Uint32,
		"u64": dtype.Uint64,
		"f32": dtype.Float32,
		"f64": dtype.Float64,
		"b":   dtype.Bool,
		"s":   dtype.String,
		"bin": dtype.Binary,
	}})

	if got := wrote(t, tbl, nil); got != in {
		t.Errorf("got\n%s\nwant\n%s", got, in)
	}
}

// TestWriteNotANumber is the one value a float column can hold that JSON has no
// way to write.
func TestWriteNotANumber(t *testing.T) {
	c := build(t, dtype.Float64, func(b *array.Builder) {
		b.Append(math.NaN())
		b.Append(math.Inf(1))
		b.Append(math.Inf(-1))
		b.Append(1.5)
	})

	const want = "{\"px\":null}\n{\"px\":null}\n{\"px\":null}\n{\"px\":1.5}\n"
	if got := wrote(t, newTable(t, []string{"px"}, c), nil); got != want {
		t.Errorf("got\n%q\nwant\n%q", got, want)
	}
}

// TestWriteNotANumberAtPrecision is the same values on the other side of the
// branch, since a fixed precision is a different formatter.
func TestWriteNotANumberAtPrecision(t *testing.T) {
	c := build(t, dtype.Float32, func(b *array.Builder) {
		b.Append(float32(math.Inf(1)))
		b.Append(float32(1.5))
	})

	const want = "{\"px\":null}\n{\"px\":1.50}\n"
	if got := wrote(t, newTable(t, []string{"px"}, c), &WriteOptions{Precision: 2}); got != want {
		t.Errorf("got\n%q\nwant\n%q", got, want)
	}
}

// TestWriteChunks writes a table whose columns are chunked differently, which
// is what concatenating frames leaves behind and what the cursor per column is
// there for.
func TestWriteChunks(t *testing.T) {
	ints := func(vs ...int64) *array.Array {
		b, err := array.NewBuilder(dtype.Int64)
		if err != nil {
			t.Fatalf("NewBuilder: %v", err)
		}
		for _, v := range vs {
			b.Append(v)
		}
		return b.Finish()
	}
	text := func(vs ...string) *array.Array {
		b, err := array.NewBuilder(dtype.String)
		if err != nil {
			t.Fatalf("NewBuilder: %v", err)
		}
		for _, v := range vs {
			b.AppendString(v)
		}
		return b.Finish()
	}

	a, err := array.NewChunked(dtype.Int64, ints(1, 2, 3), ints(4, 5))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	b, err := array.NewChunked(dtype.String, text("a"), text("b", "c", "d", "e"))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	const want = "{\"n\":1,\"s\":\"a\"}\n{\"n\":2,\"s\":\"b\"}\n{\"n\":3,\"s\":\"c\"}\n" +
		"{\"n\":4,\"s\":\"d\"}\n{\"n\":5,\"s\":\"e\"}\n"
	if got := wrote(t, newTable(t, []string{"n", "s"}, a, b), nil); got != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}
}

// TestWriteCastsWhatItCannotFormat is the fallback: a type with no JSON of its
// own goes through the cast and is written from that.
func TestWriteCastsWhatItCannotFormat(t *testing.T) {
	dt := dtype.FixedSizeBinary{ByteWidth: 3}
	c := build(t, dt, func(b *array.Builder) {
		b.AppendBytes([]byte("abc"))
		b.AppendNull()
		b.AppendBytes([]byte("def"))
	})

	const want = "{\"code\":\"abc\"}\n{\"code\":null}\n{\"code\":\"def\"}\n"
	if got := wrote(t, newTable(t, []string{"code"}, c), nil); got != want {
		t.Errorf("got\n%q\nwant\n%q", got, want)
	}
}

// TestWriteUnsupportedType is the type the cast cannot turn into text either,
// which today is anything on the calendar.
func TestWriteUnsupportedType(t *testing.T) {
	c := build(t, dtype.Date32, func(b *array.Builder) {
		b.Append(int32(19000))
	})

	err := Write(&bytes.Buffer{}, newTable(t, []string{"day"}, c), nil)
	if !errors.Is(err, ErrUnsupportedType) {
		t.Fatalf("got %v, want ErrUnsupportedType", err)
	}
	if want := `ndjson: column "day": cannot write a date32 column as JSON`; !strings.HasPrefix(err.Error(), want) {
		t.Errorf("got %q, want it to start with %q", err.Error(), want)
	}
}

func TestWriteErrors(t *testing.T) {
	tbl := read(t, "{\"sym\":\"AAPL\",\"qty\":100}\n", nil)

	tests := []struct {
		name string
		tbl  *Table
		opts *WriteOptions
		want error
	}{
		{
			name: "the wrong number of names",
			tbl:  tbl,
			opts: &WriteOptions{Names: []string{"only one"}},
			want: ErrNames,
		},
		{
			name: "more fields than columns",
			tbl:  &Table{Schema: tbl.Schema, Columns: tbl.Columns[:1]},
			want: ErrTable,
		},
		{
			name: "columns of different lengths",
			tbl: &Table{
				Schema:  tbl.Schema,
				Columns: []*array.Chunked{tbl.Columns[0], tbl.Columns[1].Slice(0, 0)},
			},
			want: ErrTable,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := Write(&bytes.Buffer{}, tt.tbl, tt.opts)
			if !errors.Is(err, tt.want) {
				t.Errorf("got %v, want %v", err, tt.want)
			}
		})
	}
}

// failWriter fails after it has taken n bytes, which is how the tests reach
// both the write at the end and the ones in the middle of a long file.
type failWriter struct {
	n int
}

var errFull = errors.New("no room")

func (w *failWriter) Write(p []byte) (int, error) {
	if len(p) > w.n {
		return 0, errFull
	}
	w.n -= len(p)
	return len(p), nil
}

func TestWritePassesOnWriteErrors(t *testing.T) {
	tbl := read(t, "{\"sym\":\"AAPL\",\"qty\":100}\n", nil)
	if err := Write(&failWriter{}, tbl, nil); !errors.Is(err, errFull) {
		t.Errorf("got %v, want errFull", err)
	}

	// A file long enough to be handed over in pieces fails on one of the
	// pieces rather than at the end.
	var in strings.Builder
	for i := range 20000 {
		in.WriteString("{\"sym\":\"AAPL\",\"qty\":" + strconv.Itoa(i) + "}\n")
	}
	if err := Write(&failWriter{n: flushAt}, read(t, in.String(), nil), nil); !errors.Is(err, errFull) {
		t.Errorf("got %v, want errFull", err)
	}
}

// TestWriteOnTheFlushBoundary is the file whose last row is the one that fills
// the buffer, so the write at the end has nothing left to do.
func TestWriteOnTheFlushBoundary(t *testing.T) {
	var in strings.Builder
	for v := 10000; in.Len() < flushAt; v++ {
		in.WriteString("{\"n\":" + strconv.Itoa(v) + "}\n")
	}

	if got := wrote(t, read(t, in.String(), nil), nil); got != in.String() {
		t.Errorf("got %d bytes back, want %d", len(got), in.Len())
	}
}

func TestWriteNoColumns(t *testing.T) {
	if got := wrote(t, &Table{}, nil); got != "" {
		t.Errorf("got %q, want nothing", got)
	}
}

func TestWriteFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trades.ndjson")
	tbl := read(t, "{\"sym\":\"AAPL\",\"qty\":100}\n{\"sym\":\"MSFT\",\"qty\":200}\n", nil)

	if err := WriteFile(path, tbl, nil); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	want := "{\"sym\":\"AAPL\",\"qty\":100}\n{\"sym\":\"MSFT\",\"qty\":200}\n"
	if string(got) != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}
}

func TestWriteFileErrors(t *testing.T) {
	dir := t.TempDir()
	tbl := read(t, "{\"sym\":\"AAPL\",\"qty\":100}\n", nil)

	if err := WriteFile(filepath.Join(dir, "no", "such", "dir.ndjson"), tbl, nil); err == nil {
		t.Error("writing into a directory that is not there worked")
	}

	// The name of the file is in the error, since a program writing several of
	// them needs to know which one went wrong.
	path := filepath.Join(dir, "bad.ndjson")
	err := WriteFile(path, tbl, &WriteOptions{Names: []string{"only one"}})
	if !errors.Is(err, ErrNames) {
		t.Fatalf("got %v, want ErrNames", err)
	}
	if !strings.Contains(err.Error(), path) {
		t.Errorf("got %q, want the path in it", err.Error())
	}
}

// TestWriteRoundTrip writes a table of every kind of value and reads it back,
// which is the promise the writer is really making.
func TestWriteRoundTrip(t *testing.T) {
	const in = "{\"sym\":\"AAPL\",\"qty\":100,\"px\":1.5,\"live\":true,\"note\":\"a,b\"}\n" +
		"{\"sym\":\"MSFT\",\"qty\":null,\"px\":2.25,\"live\":false,\"note\":\"say \\\"hi\\\"\"}\n" +
		"{\"sym\":\"GOOG\",\"qty\":300,\"px\":null,\"live\":true,\"note\":\" leading\"}\n" +
		"{\"sym\":\"NVDA\",\"qty\":400,\"px\":4.5,\"live\":null,\"note\":\"two\\nlines\"}\n"

	tbl := read(t, in, nil)
	if got := wrote(t, tbl, nil); got != in {
		t.Fatalf("got\n%s\nwant\n%s", got, in)
	}

	back := read(t, wrote(t, tbl, nil), nil)
	got, exp := show(t, back), show(t, tbl)
	if len(got) != len(exp) {
		t.Fatalf("got %d columns back, want %d", len(got), len(exp))
	}
	for i := range got {
		if got[i] != exp[i] {
			t.Errorf("column %d came back as\n%s\nwant\n%s", i, got[i], exp[i])
		}
	}
}

// TestWriteRoundTripKeepsTheEmptyString is what NDJSON has over a delimited
// file: a missing value and an empty string are different things on the page,
// so both survive the trip.
func TestWriteRoundTripKeepsTheEmptyString(t *testing.T) {
	tbl := read(t, "{\"note\":\"\"}\n{\"note\":null}\n", nil)
	back := read(t, wrote(t, tbl, nil), nil)

	if back.Columns[0].IsNull(0) {
		t.Error("an empty string came back missing")
	}
	if !back.Columns[0].IsNull(1) {
		t.Error("a missing value came back as a value")
	}
}

// TestWriteInvalidUTF8 is the one thing a table can hold that has no JSON.
//
// JSON is UTF-8, and a string column that arrived from somewhere that does not
// check can hold bytes that are not. The writer finds out in one of two places,
// once for a member name, which is quoted a single time when the line is
// planned, and once for a value, which is quoted on every row.
func TestWriteInvalidUTF8(t *testing.T) {
	text := func(vals ...string) *array.Chunked {
		return build(t, dtype.String, func(b *array.Builder) {
			for _, v := range vals {
				b.AppendBytes([]byte(v))
			}
		})
	}

	tbl := newTable(t, []string{"s"}, text("ok"))
	err := Write(&bytes.Buffer{}, tbl, &WriteOptions{Names: []string{"\xff"}})
	if err == nil {
		t.Error("wrote a member name that is not UTF-8")
	}

	bad := newTable(t, []string{"s"}, text("ok", "\xff"))
	err = Write(&bytes.Buffer{}, bad, nil)
	if err == nil {
		t.Error("wrote a value that is not UTF-8")
	} else if !strings.Contains(err.Error(), "row 2") {
		t.Errorf("got %q, want it to say which row", err.Error())
	}
}
