package dataset

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// table builds a one column table of the given name and values, which is what
// the fake reader in these tests hands back for a file.
func table(name string, vals ...int64) *array.Table {
	c, err := array.NewChunked(dtype.Int64, array.Of(vals...))
	if err != nil {
		panic(err)
	}
	return &array.Table{
		Schema:  dtype.Schema{Fields: []dtype.Field{{Name: name, Type: dtype.Int64}}},
		Columns: []*array.Chunked{c},
	}
}

// files reads a dataset with a reader that hands back whatever the map holds
// for a path, so nothing has to be written to disk to test the read.
func files(m map[string]*array.Table) *ReadOptions {
	return &ReadOptions{Open: func(p string) (*array.Table, error) {
		t, ok := m[p]
		if !ok {
			return nil, fmt.Errorf("no such file: %s", p)
		}
		return t, nil
	}}
}

// read runs Read and fails the test if it did not work.
func read(t *testing.T, d *Dataset, opts *ReadOptions) *array.Table {
	t.Helper()
	out, err := Read(d, opts)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

// numbers renders a column of numbers, with a missing value written as a dash,
// which is short enough to compare a whole column against in one string. The
// type has to be named because that is how the values come out of a column.
func numbers[T array.Numeric](c *array.Chunked) string {
	var b strings.Builder
	for i := range c.Len() {
		if i > 0 {
			b.WriteByte(' ')
		}
		if c.IsNull(i) {
			b.WriteByte('-')
			continue
		}
		fmt.Fprintf(&b, "%v", c.Value[T](i))
	}
	return b.String()
}

// ints renders an int64 column, which is what most of these tests hold.
func ints(c *array.Chunked) string { return numbers[int64](c) }

// text renders a string column the same way.
func text(c *array.Chunked) string {
	var b strings.Builder
	for i := range c.Len() {
		if i > 0 {
			b.WriteByte(' ')
		}
		if c.IsNull(i) {
			b.WriteByte('-')
			continue
		}
		b.WriteString(string(c.Bytes(i)))
	}
	return b.String()
}

func TestRead(t *testing.T) {
	fsys := tree(
		"year=2024/month=01/part-0.parquet",
		"year=2024/month=02/part-0.parquet",
		"year=2025/month=01/part-0.parquet",
	)
	d := discover(t, fsys, nil)

	out := read(t, d, files(map[string]*array.Table{
		"year=2024/month=01/part-0.parquet": table("qty", 1, 2),
		"year=2024/month=02/part-0.parquet": table("qty", 3),
		"year=2025/month=01/part-0.parquet": table("qty", 4, 5, 6),
	}))

	if got := out.Schema.String(); got != "schema<qty: int64 not null, year: int64 not null, month: string not null>" {
		t.Errorf("schema %s", got)
	}
	if out.NumRows() != 6 {
		t.Errorf("read %d rows, want 6", out.NumRows())
	}
	if got := ints(out.Columns[0]); got != "1 2 3 4 5 6" {
		t.Errorf("qty is %q", got)
	}
	if got := ints(out.Columns[1]); got != "2024 2024 2024 2025 2025 2025" {
		t.Errorf("year is %q", got)
	}
	if got := text(out.Columns[2]); got != "01 01 02 01 01 01" {
		t.Errorf("month is %q", got)
	}

	// Nothing was copied to join the files, so the chunks the files arrived in
	// are the chunks that came back.
	if got := out.Columns[0].NumChunks(); got != 3 {
		t.Errorf("the qty column has %d chunks, want one per file", got)
	}
	if got := out.Columns[1].NumChunks(); got != 3 {
		t.Errorf("the year column has %d chunks, want one per file", got)
	}
}

func TestReadOmitPartitions(t *testing.T) {
	d := discover(t, tree("year=2024/part-0.parquet"), nil)
	opts := files(map[string]*array.Table{"year=2024/part-0.parquet": table("qty", 1, 2)})
	opts.OmitPartitions = true

	out := read(t, d, opts)
	if out.NumCols() != 1 {
		t.Fatalf("read %d columns, want only the one the file holds", out.NumCols())
	}
	if got := out.Schema.String(); got != "schema<qty: int64 not null>" {
		t.Errorf("schema %s", got)
	}
}

func TestReadUnpartitioned(t *testing.T) {
	d := discover(t, tree("a.parquet", "b.parquet"), nil)

	out := read(t, d, files(map[string]*array.Table{
		"a.parquet": table("qty", 1),
		"b.parquet": table("qty", 2),
	}))
	if got := ints(out.Columns[0]); got != "1 2" {
		t.Errorf("qty is %q", got)
	}
	if out.NumCols() != 1 {
		t.Errorf("read %d columns, want the one the files hold", out.NumCols())
	}
}

func TestReadEmptyFile(t *testing.T) {
	d := discover(t, tree("year=2024/a.parquet", "year=2025/b.parquet"), nil)

	out := read(t, d, files(map[string]*array.Table{
		"year=2024/a.parquet": table("qty"),
		"year=2025/b.parquet": table("qty", 7),
	}))
	if got := ints(out.Columns[0]); got != "7" {
		t.Errorf("qty is %q", got)
	}
	if got := ints(out.Columns[1]); got != "2025" {
		t.Errorf("year is %q, want the empty file to have added nothing", got)
	}
	if got := out.Columns[0].Len(); got != out.Columns[1].Len() {
		t.Errorf("the columns are %d and %d long", got, out.Columns[1].Len())
	}
}

func TestReadNullPartition(t *testing.T) {
	fsys := tree(
		"region=us/part-0.parquet",
		"region=__HIVE_DEFAULT_PARTITION__/part-0.parquet",
	)
	d := discover(t, fsys, nil)

	out := read(t, d, files(map[string]*array.Table{
		"region=us/part-0.parquet":                         table("qty", 1),
		"region=__HIVE_DEFAULT_PARTITION__/part-0.parquet": table("qty", 2),
	}))
	if got := text(out.Columns[1]); got != "- us" {
		t.Errorf("region is %q, want the default partition to read as missing", got)
	}
	if got := out.Columns[1].NullCount(); got != 1 {
		t.Errorf("the region column counts %d nulls, want 1", got)
	}
}

func TestReadTypes(t *testing.T) {
	fsys := tree("n=7/f=1.5/flag=true/name=ab/part-0.parquet")

	cases := []struct {
		name string
		dt   dtype.DataType
		show func(*array.Chunked) string
	}{
		{"int8", dtype.Int8, numbers[int8]},
		{"int16", dtype.Int16, numbers[int16]},
		{"int32", dtype.Int32, numbers[int32]},
		{"int64", dtype.Int64, numbers[int64]},
		{"uint8", dtype.Uint8, numbers[uint8]},
		{"uint16", dtype.Uint16, numbers[uint16]},
		{"uint32", dtype.Uint32, numbers[uint32]},
		{"uint64", dtype.Uint64, numbers[uint64]},
		{"float32", dtype.Float32, numbers[float32]},
		{"float64", dtype.Float64, numbers[float64]},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d := discover(t, fsys, &Options{Types: map[string]dtype.DataType{"n": c.dt}})
			out := read(t, d, files(map[string]*array.Table{
				"n=7/f=1.5/flag=true/name=ab/part-0.parquet": table("qty", 1, 2),
			}))
			if got := c.show(out.Columns[out.Schema.Index("n")]); got != "7 7" {
				t.Errorf("n is %q, want 7 7", got)
			}
		})
	}
}

func TestReadFloatBoolAndBytes(t *testing.T) {
	fsys := tree("f=1.5/flag=true/name=ab/part-0.parquet")
	d := discover(t, fsys, &Options{Types: map[string]dtype.DataType{
		"f":    dtype.Float32,
		"flag": dtype.Bool,
		"name": dtype.Binary,
	}})
	out := read(t, d, files(map[string]*array.Table{
		"f=1.5/flag=true/name=ab/part-0.parquet": table("qty", 1, 2),
	}))

	if got := numbers[float32](out.Columns[out.Schema.Index("f")]); got != "1.5 1.5" {
		t.Errorf("f is %q, want 1.5 1.5", got)
	}
	flag := out.Columns[out.Schema.Index("flag")]
	if !flag.Bool(0) || !flag.Bool(1) {
		t.Error("flag is not true the whole way down")
	}
	if got := text(out.Columns[out.Schema.Index("name")]); got != "ab ab" {
		t.Errorf("name is %q", got)
	}

	// A declared column reads the permissive way, so a leading zero that
	// inference would have called text is a number here.
	d = discover(t, tree("n=007/part-0.parquet"), &Options{
		Types: map[string]dtype.DataType{"n": dtype.Int64},
	})
	out = read(t, d, files(map[string]*array.Table{"n=007/part-0.parquet": table("qty", 1)}))
	if got := ints(out.Columns[1]); got != "7" {
		t.Errorf("n is %q, want 7", got)
	}
}

func TestReadFloat64Partition(t *testing.T) {
	d := discover(t, tree("f=1.5/part-0.parquet", "f=2/part-0.parquet"), nil)
	if got := d.Schema.Fields[0].Type; !dtype.Equal(got, dtype.Float64) {
		t.Fatalf("f is %s, want float64", got)
	}

	out := read(t, d, files(map[string]*array.Table{
		"f=1.5/part-0.parquet": table("qty", 1),
		"f=2/part-0.parquet":   table("qty", 2),
	}))
	if got := numbers[float64](out.Columns[1]); got != "1.5 2" {
		t.Errorf("f is %q, want 1.5 2", got)
	}
}

func TestReadNoReader(t *testing.T) {
	d := discover(t, tree("year=2024/part-0.parquet"), nil)

	if _, err := Read(d, nil); !errors.Is(err, ErrOpen) {
		t.Errorf("got %v, want ErrOpen", err)
	}
	if _, err := Read(d, &ReadOptions{}); !errors.Is(err, ErrOpen) {
		t.Errorf("got %v, want ErrOpen", err)
	}
	if _, err := Read(&Dataset{}, files(nil)); !errors.Is(err, ErrNoData) {
		t.Errorf("got %v, want ErrNoData", err)
	}
}

func TestReadFileError(t *testing.T) {
	d := discover(t, tree("year=2024/part-0.parquet"), nil)
	fail := errors.New("the disk went away")

	_, err := Read(d, &ReadOptions{Open: func(string) (*array.Table, error) {
		return nil, fail
	}})
	if !errors.Is(err, fail) {
		t.Fatalf("got %v, want the error the reader returned", err)
	}
	want := "dataset: year=2024/part-0.parquet: the disk went away"
	if err.Error() != want {
		t.Errorf("got %q, want %q", err.Error(), want)
	}
}

func TestReadSchemaErrors(t *testing.T) {
	two := tree("year=2024/a.parquet", "year=2025/b.parquet")

	t.Run("two files with different columns", func(t *testing.T) {
		d := discover(t, two, nil)
		_, err := Read(d, files(map[string]*array.Table{
			"year=2024/a.parquet": table("qty", 1),
			"year=2025/b.parquet": table("amount", 2),
		}))
		if !errors.Is(err, ErrSchema) {
			t.Fatalf("got %v, want ErrSchema", err)
		}
		for _, want := range []string{"year=2024/a.parquet", "year=2025/b.parquet", "qty", "amount"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("the message does not say %q: %s", want, err)
			}
		}
	})

	t.Run("a file holding a partition column", func(t *testing.T) {
		d := discover(t, two, nil)
		_, err := Read(d, files(map[string]*array.Table{
			"year=2024/a.parquet": table("year", 1),
		}))
		if !errors.Is(err, ErrSchema) {
			t.Fatalf("got %v, want ErrSchema", err)
		}
		want := `dataset: year=2024/a.parquet holds a column called "year" and so does its path: ` +
			`the files do not share a schema`
		if err.Error() != want {
			t.Errorf("got %q, want %q", err.Error(), want)
		}
	})

	t.Run("a table whose columns and schema disagree", func(t *testing.T) {
		d := discover(t, two, nil)
		broken := table("qty", 1)
		broken.Columns = nil

		_, err := Read(d, files(map[string]*array.Table{"year=2024/a.parquet": broken}))
		if !errors.Is(err, ErrSchema) {
			t.Fatalf("got %v, want ErrSchema", err)
		}
		want := "dataset: year=2024/a.parquet: 0 columns for a schema of 1 fields: " +
			"the files do not share a schema"
		if err.Error() != want {
			t.Errorf("got %q, want %q", err.Error(), want)
		}
	})

	t.Run("a file with fewer values than the schema has columns", func(t *testing.T) {
		d := discover(t, two, nil)
		d.Files[0].Values = nil

		_, err := Read(d, files(map[string]*array.Table{"year=2024/a.parquet": table("qty", 1)}))
		if !errors.Is(err, ErrSchema) {
			t.Fatalf("got %v, want ErrSchema", err)
		}
		want := "dataset: year=2024/a.parquet: 0 partition values for 1 partition columns: " +
			"the files do not share a schema"
		if err.Error() != want {
			t.Errorf("got %q, want %q", err.Error(), want)
		}
	})
}

func TestReadValueErrors(t *testing.T) {
	cases := []struct {
		name string
		path string
		dt   dtype.DataType
		want error
		msg  string
	}{
		{
			name: "a word in a number column",
			path: "n=march/part-0.parquet",
			dt:   dtype.Int64,
			want: strconv.ErrSyntax,
			msg: "dataset: n=march/part-0.parquet: cannot read partition n=march as int64: " +
				"invalid syntax",
		},
		{
			name: "a number too big for the column",
			path: "n=300/part-0.parquet",
			dt:   dtype.Int8,
			want: strconv.ErrRange,
			msg: "dataset: n=300/part-0.parquet: cannot read partition n=300 as int8: " +
				"value out of range",
		},
		{
			name: "a negative number in an unsigned column",
			path: "n=-1/part-0.parquet",
			dt:   dtype.Uint32,
			want: strconv.ErrSyntax,
		},
		{
			name: "a word in a float column",
			path: "n=march/part-0.parquet",
			dt:   dtype.Float64,
			want: strconv.ErrSyntax,
		},
		{
			name: "a word in a boolean column",
			path: "n=march/part-0.parquet",
			dt:   dtype.Bool,
			want: strconv.ErrSyntax,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d := discover(t, tree(c.path), &Options{
				Types: map[string]dtype.DataType{"n": c.dt},
			})
			_, err := Read(d, files(map[string]*array.Table{c.path: table("qty", 1)}))

			if !errors.Is(err, ErrValue) || !errors.Is(err, c.want) {
				t.Fatalf("got %v, want ErrValue and %v", err, c.want)
			}
			var val *ValueError
			if !errors.As(err, &val) {
				t.Fatalf("got %T, want a *ValueError", err)
			}
			if val.Column != "n" {
				t.Errorf("column %q, want n", val.Column)
			}
			if c.msg != "" && err.Error() != c.msg {
				t.Errorf("got %q, want %q", err.Error(), c.msg)
			}
		})
	}
}

func TestReadValueErrorWithoutACause(t *testing.T) {
	err := &ValueError{Path: "p", Column: "n", Value: "march", Type: "int64"}
	if got := err.Error(); got != "dataset: p: cannot read partition n=march as int64" {
		t.Errorf("got %q", got)
	}
	if !errors.Is(err, ErrValue) {
		t.Error("a ValueError with no cause does not unwrap to ErrValue")
	}
}

func TestReadUnbuildableColumn(t *testing.T) {
	// A type that never went through the check in Discover, which is what a
	// caller who built the Dataset by hand has, fails in the read rather than
	// panicking in the builder.
	d := discover(t, tree("n=1/part-0.parquet"), nil)
	d.Schema.Fields[0].Type = dtype.Date32

	_, err := Read(d, files(map[string]*array.Table{"n=1/part-0.parquet": table("qty", 1)}))
	if !errors.Is(err, ErrUnsupportedType) {
		t.Fatalf("got %v, want ErrUnsupportedType", err)
	}
	want := "dataset: n=1/part-0.parquet: cannot read partition n=1 as date32: " +
		"unsupported partition type"
	if err.Error() != want {
		t.Errorf("got %q, want %q", err.Error(), want)
	}
}

func TestReadColumnWithNoBuilder(t *testing.T) {
	// A large_string column is converted at the IPC boundary rather than built,
	// so there is no builder for one. Discover turns the type away, and a
	// Dataset built by hand finds out here.
	d := discover(t, tree("n=1/part-0.parquet"), nil)
	d.Schema.Fields[0].Type = dtype.LargeString

	_, err := Read(d, files(map[string]*array.Table{"n=1/part-0.parquet": table("qty", 1)}))
	if !errors.Is(err, ErrValue) {
		t.Fatalf("got %v, want ErrValue", err)
	}
	want := "dataset: n=1/part-0.parquet: cannot read partition n=1 as large_string: " +
		"array: a large_string column is converted at the IPC boundary, store it as a string"
	if err.Error() != want {
		t.Errorf("got %q, want %q", err.Error(), want)
	}
}

func TestReadColumnThatIsNotWhatItSaysItIs(t *testing.T) {
	// A reader that hands back a table whose schema and columns disagree is a
	// bug in the reader, and the message says which file it came from.
	lying := table("qty", 1)
	lying.Schema.Fields[0].Type = dtype.Float64

	d := discover(t, tree("year=2024/part-0.parquet"), nil)
	_, err := Read(d, files(map[string]*array.Table{"year=2024/part-0.parquet": lying}))
	if err == nil {
		t.Fatal("read a table whose column is not the type its schema gave")
	}
	want := "dataset: array: chunk 0 is a int64 column, want float64"
	if err.Error() != want {
		t.Errorf("got %q, want %q", err.Error(), want)
	}
}

func TestCause(t *testing.T) {
	// Every parse in this package returns a *strconv.NumError, so the message
	// carried into a ValueError is the syntax or the range error inside it and
	// not the wrapper repeating the input. Anything else is passed through.
	err := cause(&strconv.NumError{Func: "ParseInt", Num: "march", Err: strconv.ErrSyntax})
	if !errors.Is(err, strconv.ErrSyntax) || err.Error() != "invalid syntax" {
		t.Errorf("got %q, want the error inside", err)
	}

	plain := errors.New("something else")
	if got := cause(plain); got != plain {
		t.Errorf("got %v, want the error it was given", got)
	}
}
