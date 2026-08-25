package csv

import (
	"bytes"
	"errors"
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
			in:   "sym,qty,px,live\nAAPL,100,1.5,true\nMSFT,200,2.25,false\n",
			want: "sym,qty,px,live\nAAPL,100,1.5,true\nMSFT,200,2.25,false\n",
		},
		{
			name: "a missing value is an empty field",
			in:   "sym,qty\nAAPL,\nMSFT,200\n",
			want: "sym,qty\nAAPL,\nMSFT,200\n",
		},
		{
			name: "a missing value can be a word instead",
			in:   "sym,qty\nAAPL,\nMSFT,200\n",
			opts: &WriteOptions{NullValue: "NA"},
			want: "sym,qty\nAAPL,NA\nMSFT,200\n",
		},
		{
			name: "a missing value that needs quotes gets them",
			in:   "sym,qty\nAAPL,\n",
			opts: &WriteOptions{NullValue: "a,b"},
			want: "sym,qty\nAAPL,\"a,b\"\n",
		},
		{
			name: "the header can be left out",
			in:   "sym,qty\nAAPL,100\n",
			opts: &WriteOptions{NoHeader: true},
			want: "AAPL,100\n",
		},
		{
			name: "names replace the ones in the schema",
			in:   "sym,qty\nAAPL,100\n",
			opts: &WriteOptions{Names: []string{"symbol", "quantity"}},
			want: "symbol,quantity\nAAPL,100\n",
		},
		{
			name: "another delimiter",
			in:   "sym,qty\nAAPL,100\n",
			opts: &WriteOptions{Delimiter: ';'},
			want: "sym;qty\nAAPL;100\n",
		},
		{
			name: "a tab",
			in:   "sym,qty\nAAPL,100\n",
			opts: &WriteOptions{Delimiter: '\t'},
			want: "sym\tqty\nAAPL\t100\n",
		},
		{
			name: "a delimiter outside ASCII",
			in:   "sym,qty\nAAPL,100\n",
			opts: &WriteOptions{Delimiter: '\u00b6'},
			want: "sym\u00b6qty\nAAPL\u00b6100\n",
		},
		{
			name: "a value holding the delimiter is quoted",
			in:   "sym,note\nAAPL,\"a,b\"\n",
			want: "sym,note\nAAPL,\"a,b\"\n",
		},
		{
			name: "a quote in a value is doubled",
			in:   "sym,note\nAAPL,\"say \"\"hi\"\"\"\n",
			want: "sym,note\nAAPL,\"say \"\"hi\"\"\"\n",
		},
		{
			name: "a newline in a value is quoted",
			in:   "sym,note\nAAPL,\"a\nb\"\n",
			want: "sym,note\nAAPL,\"a\nb\"\n",
		},
		{
			name: "a value starting with a space is quoted",
			in:   "sym,note\nAAPL,\" x\"\n",
			want: "sym,note\nAAPL,\" x\"\n",
		},
		{
			name: "a space in the middle of a value is left alone",
			in:   "sym,note\nAAPL,New York\n",
			want: "sym,note\nAAPL,New York\n",
		},
		{
			name: "the end of data marker is quoted",
			in:   "sym,note\nAAPL,\\.\n",
			want: "sym,note\nAAPL,\"\\.\"\n",
		},
		{
			name: "a name that needs quotes gets them",
			in:   "\"a,b\",qty\n1,2\n",
			want: "\"a,b\",qty\n1,2\n",
		},
		{
			name: "every field quoted",
			in:   "sym,qty\nAAPL,100\n",
			opts: &WriteOptions{QuoteAll: true},
			want: "\"sym\",\"qty\"\n\"AAPL\",\"100\"\n",
		},
		{
			name: "a missing value is quoted too when everything is",
			in:   "sym,qty\nAAPL,\n",
			opts: &WriteOptions{QuoteAll: true},
			want: "\"sym\",\"qty\"\n\"AAPL\",\"\"\n",
		},
		{
			name: "windows line endings",
			in:   "sym,qty\nAAPL,100\n",
			opts: &WriteOptions{CRLF: true},
			want: "sym,qty\r\nAAPL,100\r\n",
		},
		{
			name: "a float keeps the digits it needs and no more",
			in:   "px\n0.1\n1e10\n123456789.123\n",
			want: "px\n0.1\n1e+10\n1.23456789123e+08\n",
		},
		{
			name: "floats can be written at a fixed precision",
			in:   "px\n1.5\n2.25\n",
			opts: &WriteOptions{Precision: 3},
			want: "px\n1.500\n2.250\n",
		},
		{
			name: "a header and no rows is a header and no rows",
			in:   "sym,qty\n",
			want: "sym,qty\n",
		},
		{
			name: "an empty value stays empty when there is a second column",
			in:   "sym,n\n,1\nMSFT,2\n",
			want: "sym,n\n,1\nMSFT,2\n",
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

// TestWriteOneColumn is the case a blank line would swallow. A row whose only
// value is missing has to be a pair of quotes, because a line with nothing on
// it is not a row.
func TestWriteOneColumn(t *testing.T) {
	tbl := read(t, "sym\nAAPL\n\"\"\nMSFT\n", nil)

	const want = "sym\nAAPL\n\"\"\nMSFT\n"
	if got := wrote(t, tbl, nil); got != want {
		t.Fatalf("got %q, want %q", got, want)
	}

	back := read(t, want, nil)
	if back.NumRows() != 3 {
		t.Errorf("got %d rows back, want 3", back.NumRows())
	}
	if !back.Columns[0].IsNull(1) {
		t.Error("the empty row came back as a value")
	}
}

func TestWriteEveryType(t *testing.T) {
	const in = "i8,i16,i32,i64,u8,u16,u32,u64,f32,f64,b,s,bin\n" +
		"-8,-16,-32,-64,8,16,32,64,1.5,2.5,true,hi,bye\n" +
		"127,-32768,2147483647,-9223372036854775808,255,65535,4294967295,18446744073709551615,-0.25,1e+100,false,,x\n"

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

	const want = "n,s\n1,a\n2,b\n3,c\n4,d\n5,e\n"
	if got := wrote(t, newTable(t, []string{"n", "s"}, a, b), nil); got != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}
}

// TestWriteCastsWhatItCannotFormat is the fallback: a type with no text of its
// own goes through the cast and is written from that.
func TestWriteCastsWhatItCannotFormat(t *testing.T) {
	dt := dtype.FixedSizeBinary{ByteWidth: 3}
	c := build(t, dt, func(b *array.Builder) {
		b.AppendBytes([]byte("abc"))
		b.AppendNull()
		b.AppendBytes([]byte("def"))
	})

	const want = "code\nabc\n\"\"\ndef\n"
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
	if want := `csv: column "day": cannot write a date32 column as text`; !strings.HasPrefix(err.Error(), want) {
		t.Errorf("got %q, want it to start with %q", err.Error(), want)
	}
}

func TestWriteErrors(t *testing.T) {
	tbl := read(t, "sym,qty\nAAPL,100\n", nil)

	tests := []struct {
		name string
		tbl  *Table
		opts *WriteOptions
		want error
	}{
		{
			name: "a quote cannot separate fields",
			tbl:  tbl,
			opts: &WriteOptions{Delimiter: '"'},
			want: ErrDelimiter,
		},
		{
			name: "neither can a newline",
			tbl:  tbl,
			opts: &WriteOptions{Delimiter: '\n'},
			want: ErrDelimiter,
		},
		{
			name: "the wrong number of names",
			tbl:  tbl,
			opts: &WriteOptions{Names: []string{"only one"}},
			want: ErrNames,
		},
		{
			name: "more columns than fields",
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
	tbl := read(t, "sym,qty\nAAPL,100\n", nil)
	if err := Write(&failWriter{}, tbl, nil); !errors.Is(err, errFull) {
		t.Errorf("got %v, want errFull", err)
	}

	// A file long enough to be handed over in pieces fails on one of the
	// pieces rather than at the end.
	var in strings.Builder
	in.WriteString("sym,qty\n")
	for i := range 20000 {
		in.WriteString("AAPL," + strconv.Itoa(i) + "\n")
	}
	if err := Write(&failWriter{n: flushAt}, read(t, in.String(), nil), nil); !errors.Is(err, errFull) {
		t.Errorf("got %v, want errFull", err)
	}
}

// TestWriteOnTheFlushBoundary is the file whose last row is the one that fills
// the buffer, so the write at the end has nothing left to do.
func TestWriteOnTheFlushBoundary(t *testing.T) {
	var in strings.Builder
	in.WriteString("n\n")
	for v := 10000; in.Len() < flushAt; v++ {
		in.WriteString(strconv.Itoa(v))
		in.WriteByte('\n')
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
	path := filepath.Join(t.TempDir(), "trades.csv")
	tbl := read(t, "sym,qty\nAAPL,100\nMSFT,200\n", nil)

	if err := WriteFile(path, tbl, nil); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if want := "sym,qty\nAAPL,100\nMSFT,200\n"; string(got) != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}
}

func TestWriteFileErrors(t *testing.T) {
	dir := t.TempDir()
	tbl := read(t, "sym,qty\nAAPL,100\n", nil)

	if err := WriteFile(filepath.Join(dir, "no", "such", "dir.csv"), tbl, nil); err == nil {
		t.Error("writing into a directory that is not there worked")
	}

	// The name of the file is in the error, since a program writing several of
	// them needs to know which one went wrong.
	path := filepath.Join(dir, "bad.csv")
	err := WriteFile(path, tbl, &WriteOptions{Delimiter: '"'})
	if !errors.Is(err, ErrDelimiter) {
		t.Fatalf("got %v, want ErrDelimiter", err)
	}
	if !strings.Contains(err.Error(), path) {
		t.Errorf("got %q, want the path in it", err.Error())
	}
}

// TestWriteRoundTrip writes a table of every kind of value and reads it back,
// which is the promise the writer is really making.
func TestWriteRoundTrip(t *testing.T) {
	const in = "sym,qty,px,live,note\n" +
		"AAPL,100,1.5,true,\"a,b\"\n" +
		"MSFT,,2.25,false,\"say \"\"hi\"\"\"\n" +
		"GOOG,300,,true,\" leading\"\n" +
		"NVDA,400,4.5,,\"two\nlines\"\n"

	tbl := read(t, in, nil)
	back := read(t, wrote(t, tbl, nil), nil)

	if got, want := show(t, back), show(t, tbl); len(got) != len(want) {
		t.Fatalf("got %d columns back, want %d", len(got), len(want))
	} else {
		for i := range got {
			if got[i] != want[i] {
				t.Errorf("column %d came back as\n%s\nwant\n%s", i, got[i], want[i])
			}
		}
	}
}

// TestWriteEmptyStringComesBackMissing is the one thing a round trip cannot
// promise, written down so that it changing is a test failing.
func TestWriteEmptyStringComesBackMissing(t *testing.T) {
	tbl := read(t, "sym,note\nAAPL,\"\"\n", &Options{NullValues: []string{}})
	if tbl.Columns[1].NullCount() != 0 {
		t.Fatal("the empty note was read as missing")
	}

	back := read(t, wrote(t, tbl, nil), nil)
	if !back.Columns[1].IsNull(0) {
		t.Error("an empty field came back as a value, which a file cannot say")
	}
}

func TestReadBadDelimiter(t *testing.T) {
	tests := []struct {
		name string
		opts *Options
	}{
		{"a quote", &Options{Delimiter: '"'}},
		{"a newline", &Options{Delimiter: '\n'}},
		{"a comment of a quote", &Options{Comment: '"'}},
		{"both the same", &Options{Delimiter: ';', Comment: ';'}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Read(strings.NewReader("a,b\n1,2\n"), tt.opts)
			if !errors.Is(err, ErrDelimiter) {
				t.Errorf("got %v, want ErrDelimiter", err)
			}
		})
	}
}
