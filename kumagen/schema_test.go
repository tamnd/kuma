package main

import (
	"bytes"
	"errors"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
	"github.com/tamnd/kuma/parquet"
)

// dataFile writes a file of the given name into a directory of its own and
// returns the path, which is what -from is given.
func dataFile(t *testing.T, name, data string) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// fromFile runs the tool over a data file and returns what it wrote to standard
// output, which is where -o - puts it.
func fromFile(t *testing.T, path, typeName string, args ...string) string {
	t.Helper()

	var out bytes.Buffer
	all := append([]string{"-from", path, "-type", typeName, "-o", "-",
		"-package", "trades"}, args...)
	if err := run(all, &out); err != nil {
		t.Fatalf("run: %v", err)
	}
	return out.String()
}

const csvTrades = "symbol,price,qty,is_open\n" +
	"AAPL,1.5,100,true\n" +
	"MSFT,2.5,50,false\n"

func TestRunFromCSV(t *testing.T) {
	got := fromFile(t, dataFile(t, "trades.csv", csvTrades), "Trade")

	want := `package trades

// Trade is a row of trades.csv.
//
// The fields came from the columns of that file and the types are the ones a
// bind reads out of them. Edit it: drop the columns this program does not
// want, rename a field, and run go generate to write the handles again.
//
//go:generate kumagen -type Trade
type Trade struct {
	Symbol string
	Price  float64
	Qty    int64
	IsOpen bool
}
`
	if got != want {
		t.Errorf("run wrote\n%s\nand it should have written\n%s", got, want)
	}
}

func TestRunFromTSV(t *testing.T) {
	tsv := strings.ReplaceAll(csvTrades, ",", "\t")

	got := fromFile(t, dataFile(t, "trades.tsv", tsv), "Trade")
	if !strings.Contains(got, "Symbol string") || !strings.Contains(got, "Qty    int64") {
		t.Errorf("run wrote\n%s\nand a tab separated file has the same columns", got)
	}
}

func TestRunFromNDJSON(t *testing.T) {
	const lines = `{"symbol":"AAPL","price":1.5,"qty":100}` + "\n" +
		`{"symbol":"MSFT","price":2.5,"qty":50}` + "\n"

	got := fromFile(t, dataFile(t, "trades.ndjson", lines), "Trade")
	if !strings.Contains(got, "type Trade struct") ||
		!strings.Contains(got, "Price  float64") {
		t.Errorf("run wrote\n%s", got)
	}
}

// TestRunFromParquet is the one that matters most, since a Parquet file is
// where a schema is written down rather than guessed at.
func TestRunFromParquet(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trades.parquet")
	if err := parquet.WriteFile(path, tradesTable(t), nil); err != nil {
		t.Fatalf("parquet.WriteFile: %v", err)
	}

	got := fromFile(t, path, "Trade")
	for _, want := range []string{
		"import \"time\"", "Symbol string", "Qty    int32", "Price  float64",
		"TS     time.Time",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("run wrote\n%s\nand it is missing %q", got, want)
		}
	}
}

// TestRunFromArrow covers the IPC file format, which keeps its schema in a
// footer the same way.
func TestRunFromArrow(t *testing.T) {
	got := fromFile(t, arrowFile(t, "trades.arrow"), "Trade")
	if !strings.Contains(got, "Qty    int32") {
		t.Errorf("run wrote\n%s\nand the width of a column is the file's to say", got)
	}
}

// TestRunFromArrowStream covers the stream, which has no footer at all and
// carries its schema in front of the batches.
func TestRunFromArrowStream(t *testing.T) {
	got := fromFile(t, arrowFile(t, "trades.arrows"), "Trade")
	if !strings.Contains(got, "Qty    int32") {
		t.Errorf("run wrote\n%s", got)
	}
}

// TestRunFromNamesTheColumnWhenTheFieldDoesNot covers the tag, which is written
// for a column whose name does not come back from the field name.
func TestRunFromNamesTheColumnWhenTheFieldDoesNot(t *testing.T) {
	const odd = "Order ID,2024 total,weird..name,plain\n1,2.5,a,true\n"

	got := fromFile(t, dataFile(t, "odd.csv", odd), "Odd")
	for _, want := range []string{
		"OrderID      int64   `kuma:\"Order ID\"`",
		"Col2024Total float64 `kuma:\"2024 total\"`",
		"WeirdName    string  `kuma:\"weird..name\"`",
		"Plain        bool\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("run wrote\n%s\nand it is missing %s", got, want)
		}
	}
}

// TestRunFromWritesTheDefaultFile is the shape a program actually uses, which
// is a file next to the package rather than standard output.
func TestRunFromWritesTheDefaultFile(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "market")
	if err := os.Mkdir(dir, 0o750); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "order_books.csv")
	if err := os.WriteFile(path, []byte(csvTrades), 0o600); err != nil {
		t.Fatal(err)
	}

	if err := run([]string{"-from", path, "-type", "OrderBook", "-dir", dir},
		&bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}

	// The file is named after the type in snake case, the way the handles are,
	// and the package is the name of the directory it went into.
	got := read(t, filepath.Join(dir, "order_book.go"))
	if !strings.HasPrefix(got, "package "+filepath.Base(dir)+"\n") {
		t.Errorf("run wrote\n%s\nand the package should be the directory", got)
	}
	if !strings.Contains(got, "//go:generate kumagen -type OrderBook") {
		t.Errorf("run wrote\n%s\nand it should say how to write the handles", got)
	}
}

// TestRunFromTakesThePackageFromTheDirectory covers writing into a package that
// is already there, which is the usual case and the one where the name of the
// directory is not the answer.
func TestRunFromTakesThePackageFromTheDirectory(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, "doc.go"), "package market\n")
	write(t, filepath.Join(dir, "aaa_test.go"), "package market_test\n")

	path := filepath.Join(dir, "trades.csv")
	if err := os.WriteFile(path, []byte(csvTrades), 0o600); err != nil {
		t.Fatal(err)
	}

	var out bytes.Buffer
	if err := run([]string{"-from", path, "-type", "Trade", "-dir", dir, "-o", "-"},
		&out); err != nil {
		t.Fatalf("run: %v", err)
	}
	if !strings.HasPrefix(out.String(), "package market\n") {
		t.Errorf("run wrote\n%s\nand the test only package is not the one to join",
			out.String())
	}
}

// TestRunFromLeavesAFileThatIsThere is the guard on the one file kumagen writes
// that a person is meant to edit.
func TestRunFromLeavesAFileThatIsThere(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "trades.csv")
	if err := os.WriteFile(path, []byte(csvTrades), 0o600); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(dir, "trade.go"), "package trades\n\n// mine\n")

	args := []string{"-from", path, "-type", "Trade", "-dir", dir}
	err := run(args, &bytes.Buffer{})
	if err == nil {
		t.Fatal("run replaced a file that was already there")
	}
	if !strings.Contains(err.Error(), "pass -f") {
		t.Errorf("run says %q, and it should say how to replace it", err)
	}
	if got := read(t, filepath.Join(dir, "trade.go")); !strings.Contains(got, "// mine") {
		t.Errorf("trade.go holds %q, and the run that failed should not have touched it", got)
	}

	if err := run(append(args, "-f"), &bytes.Buffer{}); err != nil {
		t.Fatalf("run -f: %v", err)
	}
	if got := read(t, filepath.Join(dir, "trade.go")); strings.Contains(got, "// mine") {
		t.Errorf("trade.go holds %q, and -f should have replaced it", got)
	}
}

// TestRunFromSamplesTheLinesItIsTold covers -lines, which is the difference
// between a column of numbers and a column that turns out to hold text further
// down.
func TestRunFromSamplesTheLinesItIsTold(t *testing.T) {
	var sb strings.Builder
	sb.WriteString("note\n")
	for i := range 20 {
		if i == 15 {
			sb.WriteString("late\n")
			continue
		}
		sb.WriteString("1\n")
	}
	path := dataFile(t, "notes.csv", sb.String())

	// The first ten lines are all numbers, so ten lines call the column one.
	if got := fromFile(t, path, "Note", "-lines", "10"); !strings.Contains(got, "Note int64") {
		t.Errorf("ten lines wrote\n%s\nand every one of them held a number", got)
	}
	if got := fromFile(t, path, "Note", "-lines", "100"); !strings.Contains(got, "Note string") {
		t.Errorf("a hundred lines wrote\n%s\nand line sixteen is not a number", got)
	}
}

func TestRunFromErrors(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want string
	}{
		{
			name: "a format with no schema in it",
			args: []string{"-from", dataFile(t, "trades.orc", "no"), "-type", "T"},
			want: `no schema can be read out of a ".orc" file`,
		},
		{
			name: "a file that is not what it says it is",
			args: []string{"-from", dataFile(t, "trades.parquet", "not a parquet file"), "-type", "T"},
			want: "trades.parquet",
		},
		{
			name: "two types out of one file",
			args: []string{"-from", dataFile(t, "trades.csv", csvTrades), "-type", "A,B"},
			want: "a data file holds one schema",
		},
		{
			name: "a sample of no lines",
			args: []string{"-from", dataFile(t, "trades.csv", csvTrades), "-type", "T", "-lines", "0"},
			want: "fewer than one line",
		},
		{
			name: "a package name that is not one",
			args: []string{"-from", dataFile(t, "trades.csv", csvTrades), "-type", "T",
				"-package", "not a package"},
			want: `"not a package" is not a package name`,
		},
		{
			name: "a file with nothing in it",
			args: []string{"-from", dataFile(t, "empty.csv", ""), "-type", "T"},
			want: "no data",
		},
		{
			name: "a directory where a file should be",
			args: []string{"-from", dataDir(t, "trades.csv"), "-type", "T"},
			want: "trades.csv",
		},
		{
			name: "an arrow file that is not one",
			args: []string{"-from", dataFile(t, "trades.arrow", "not an arrow file"), "-type", "T"},
			want: "trades.arrow",
		},
		{
			name: "an arrow stream that is not one",
			args: []string{"-from", dataFile(t, "trades.arrows", "not a stream"), "-type", "T"},
			want: "trades.arrows",
		},
		{
			name: "a line that is not JSON",
			args: []string{"-from", dataFile(t, "trades.ndjson", "{not json\n"), "-type", "T"},
			want: "trades.ndjson",
		},
		{
			name: "a column with no name to make a field of",
			args: []string{"-from", dataFile(t, "dots.csv", "...\n1\n"), "-type", "T"},
			want: "no letter or digit in it",
		},
		{
			name: "a column of a type no field holds",
			args: []string{"-from", listArrowFile(t), "-type", "T"},
			want: "and a field can hold",
		},
		{
			name: "a file of no columns",
			args: []string{"-from", emptyArrowFile(t), "-type", "T"},
			want: "no columns",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := run(append(c.args, "-o", "-"), &bytes.Buffer{})
			if err == nil {
				t.Fatal("run said nothing")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("run says %q, and it should mention %q", err, c.want)
			}
		})
	}
}

// TestRunFromAFileThatIsNotThere is its own test rather than a row of the table
// above, because what a missing file is called is up to the operating system
// and only the wrapped error is the same everywhere.
func TestRunFromAFileThatIsNotThere(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gone.csv")

	err := run([]string{"-from", path, "-type", "T", "-o", "-"}, &bytes.Buffer{})
	if !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("run says %v, and a file that is not there is a not exist error", err)
	}
	if err != nil && !strings.Contains(err.Error(), "gone.csv") {
		t.Errorf("run says %q, and it should name the file it was given", err)
	}
}

// TestRunFromWithNoPackageToJoin covers the directory that holds no Go file and
// is not named like a package either, which is the one case that has to be told.
func TestRunFromWithNoPackageToJoin(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "not-a-package")
	if err := os.Mkdir(dir, 0o750); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "trades.csv")
	if err := os.WriteFile(path, []byte(csvTrades), 0o600); err != nil {
		t.Fatal(err)
	}

	err := run([]string{"-from", path, "-type", "Trade", "-dir", dir, "-o", "-"},
		&bytes.Buffer{})
	if err == nil {
		t.Fatal("run made a package name out of not-a-package")
	}
	if !strings.Contains(err.Error(), "pass -package") {
		t.Errorf("run says %q, and it should say what to pass", err)
	}
}

// TestRunFromReportsAWriteThatFails is the -o - path with nothing on the other
// end of it.
func TestRunFromReportsAWriteThatFails(t *testing.T) {
	err := run([]string{"-from", dataFile(t, "trades.csv", csvTrades),
		"-type", "Trade", "-o", "-", "-package", "trades"}, brokenPipe{})
	if err == nil {
		t.Fatal("run wrote to a writer that refuses everything and said nothing")
	}
	if !strings.Contains(err.Error(), "broken pipe") {
		t.Errorf("run says %q", err)
	}
}

// TestRunFromCannotWriteThere covers a path that is not a file, which is the
// open failing for a reason that is not the file being there already.
func TestRunFromCannotWriteThere(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "trades.csv")
	if err := os.WriteFile(path, []byte(csvTrades), 0o600); err != nil {
		t.Fatal(err)
	}

	err := run([]string{"-from", path, "-type", "Trade", "-dir", dir, "-o", dir},
		&bytes.Buffer{})
	if err == nil {
		t.Fatal("run wrote a file over a directory")
	}
	if strings.Contains(err.Error(), "pass -f") {
		t.Errorf("run says %q, and -f would not help with a directory", err)
	}
}

func TestGoType(t *testing.T) {
	cases := []struct {
		dt   dtype.DataType
		want string
	}{
		{dtype.Bool, "bool"},
		{dtype.Int8, "int8"},
		{dtype.Int16, "int16"},
		{dtype.Int32, "int32"},
		{dtype.Int64, "int64"},
		{dtype.Uint8, "uint8"},
		{dtype.Uint16, "uint16"},
		{dtype.Uint32, "uint32"},
		{dtype.Uint64, "uint64"},
		{dtype.Float32, "float32"},
		{dtype.Float64, "float64"},
		{dtype.String, "string"},
		{dtype.Binary, "string"},
		{dtype.Date32, "int32"},
		{dtype.Date64, "int64"},
		{dtype.Timestamp{Unit: dtype.Microsecond}, "time.Time"},
		{dtype.Duration{Unit: dtype.Second}, "int64"},
		{dtype.LargeString, ""},
		{dtype.List{Elem: dtype.Int64}, ""},
	}

	for _, c := range cases {
		got, ok := goType(c.dt)
		if ok != (c.want != "") {
			t.Errorf("goType(%s) said %v, want %v", c.dt, ok, c.want != "")
		}
		if got != c.want {
			t.Errorf("goType(%s) is %q, want %q", c.dt, got, c.want)
		}
	}
}

func TestFieldName(t *testing.T) {
	cases := map[string]string{
		"symbol":      "Symbol",
		"order_id":    "OrderID",
		"user_id":     "UserID",
		"url":         "URL",
		"http_code":   "HTTPCode",
		"ts":          "TS",
		"Order ID":    "OrderID",
		"weird..name": "WeirdName",
		"2024_total":  "Col2024Total",
		"a":           "A",
	}

	for column, want := range cases {
		got, err := fieldName(column)
		if err != nil {
			t.Errorf("fieldName(%q): %v", column, err)
			continue
		}
		if got != want {
			t.Errorf("fieldName(%q) is %q, want %q", column, got, want)
		}
	}

	if _, err := fieldName("..."); err == nil {
		t.Error("a column of nothing but dots made a field name")
	}
}

// TestStructFieldsRefusesTwoColumnsThatAreOneField covers the collision, which
// a tag cannot fix because the two fields would still have the one name.
func TestStructFieldsRefusesTwoColumnsThatAreOneField(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{
		{Name: "order id", Type: dtype.Int64},
		{Name: "order_id", Type: dtype.Int64},
	}}

	_, err := structFields("Order", s)
	if err == nil {
		t.Fatal("two columns became the one field")
	}
	if !strings.Contains(err.Error(), "order id") || !strings.Contains(err.Error(), "order_id") {
		t.Errorf("structFields says %q, and it should name both columns", err)
	}
}

// TestHeadTakesWholeLines checks that a sample is lines rather than bytes, and
// that a file with fewer lines than were asked for is read to the end of it.
func TestHeadTakesWholeLines(t *testing.T) {
	cases := []struct {
		in    string
		lines int
		want  string
	}{
		{"a\nb\nc\n", 2, "a\nb\n"},
		{"a\nb\nc\n", 9, "a\nb\nc\n"},
		{"a\nb\nc", 9, "a\nb\nc"},
		{"", 3, ""},
	}

	for _, c := range cases {
		r, err := head(strings.NewReader(c.in), c.lines)
		if err != nil {
			t.Fatalf("head(%q, %d): %v", c.in, c.lines, err)
		}
		b, err := io.ReadAll(r)
		if err != nil {
			t.Fatal(err)
		}
		if got := string(b); got != c.want {
			t.Errorf("head(%q, %d) is %q, want %q", c.in, c.lines, got, c.want)
		}
	}
}

// TestHeadPassesOnAReadThatFails covers a sample of a file that stops being
// readable part way through it, which is the disk rather than the format.
func TestHeadPassesOnAReadThatFails(t *testing.T) {
	if _, err := head(failingReader{}, 4); err == nil {
		t.Error("head read a file that will not be read")
	}
}

// failingReader is a file that reports an error rather than reaching the end.
type failingReader struct{}

func (failingReader) Read([]byte) (int, error) { return 0, errTornDisk }

var errTornDisk = errors.New("the disk gave up")

// tradesTable is a table of every type the struct writer has an opinion about,
// which is what the Parquet and Arrow tests are written out of.
func tradesTable(t *testing.T) *array.Table {
	t.Helper()

	stamp := dtype.Timestamp{Unit: dtype.Microsecond}
	b, err := array.NewBuilder(stamp)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	b.AppendValues([]int64{time.Unix(0, 0).UnixMicro(), time.Unix(1, 0).UnixMicro()})

	sym := array.OfStrings("AAPL", "MSFT")
	qty := array.Of[int32](100, 50)
	price := array.Of(1.5, 2.5)
	ts := b.Finish()

	fields := []dtype.Field{
		{Name: "symbol", Type: dtype.String},
		{Name: "qty", Type: dtype.Int32},
		{Name: "price", Type: dtype.Float64},
		{Name: "ts", Type: stamp},
	}
	arrays := []*array.Array{sym, qty, price, ts}

	cols := make([]*array.Chunked, len(fields))
	for i := range fields {
		c, err := array.NewChunked(fields[i].Type, arrays[i])
		if err != nil {
			t.Fatalf("NewChunked: %v", err)
		}
		cols[i] = c
	}
	return &array.Table{Schema: dtype.Schema{Fields: fields}, Columns: cols}
}

// arrowFile writes the trades table as an IPC file or an IPC stream, whichever
// the name asks for, and returns the path.
func arrowFile(t *testing.T, name string) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), name)
	f := create(t, path)
	defer f.Close()

	table := tradesTable(t)
	batch := ipc.Batch{
		Length:  table.NumRows(),
		Columns: make([]*array.Array, len(table.Columns)),
	}
	for i, c := range table.Columns {
		batch.Columns[i] = c.Chunks()[0]
	}

	if strings.HasSuffix(name, ".arrows") {
		w, err := ipc.NewWriter(f, table.Schema)
		if err != nil {
			t.Fatal(err)
		}
		if err := w.Write(batch); err != nil {
			t.Fatal(err)
		}
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}
		return path
	}

	w, err := ipc.NewFileWriter(f, table.Schema)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Write(batch); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

// emptyArrowFile writes an IPC file of no columns at all, which is the one way
// to hand the struct writer a schema with nothing in it.
func emptyArrowFile(t *testing.T) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "empty.arrow")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	w, err := ipc.NewFileWriter(f, dtype.Schema{})
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

// listArrowFile writes an IPC file holding a column of a type that no field can
// be read out of, which is what the struct writer turns away.
func listArrowFile(t *testing.T) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "lists.arrow")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	schema := dtype.Schema{Fields: []dtype.Field{
		{Name: "tags", Type: dtype.List{Elem: dtype.Int64}},
	}}
	w, err := ipc.NewFileWriter(f, schema)
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

// dataDir makes a directory with the name of a data file, which is the mistake
// of pointing -from at the tree rather than at one file in it.
func dataDir(t *testing.T, name string) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), name)
	if err := os.Mkdir(path, 0o750); err != nil {
		t.Fatal(err)
	}
	return path
}

// create makes a file and fails the test rather than returning an error, so
// that the writers above have one err in them and it is the one that matters.
func create(t *testing.T, path string) *os.File {
	t.Helper()

	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// TestTheWrittenStructBinds is the test that matters for -from, and it is the
// only one here that runs the go tool.
//
// A CSV file goes in, kumagen writes the struct for it, kumagen writes the
// handles for that struct, and a program reads the same file and binds it. If
// the types were a width out or a column name did not come back from a field
// name the bind would fail, which is the whole thing this mode has to get
// right.
func TestTheWrittenStructBinds(t *testing.T) {
	if testing.Short() {
		t.Skip("the go tool takes a few seconds")
	}
	gotool, err := exec.LookPath("go")
	if err != nil {
		t.Skipf("no go tool to build with: %v", err)
	}

	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatalf("finding the module root: %v", err)
	}

	dir := filepath.Join(t.TempDir(), "trades")
	if mkerr := os.Mkdir(dir, 0o750); mkerr != nil {
		t.Fatal(mkerr)
	}
	data := filepath.Join(dir, "trades.csv")
	if werr := os.WriteFile(data, []byte(bindable), 0o600); werr != nil {
		t.Fatal(werr)
	}

	if rerr := run([]string{"-from", data, "-type", "Trade", "-dir", dir},
		&bytes.Buffer{}); rerr != nil {
		t.Fatalf("run -from: %v", rerr)
	}
	if rerr := run([]string{"-dir", dir, "-type", "Trade"}, &bytes.Buffer{}); rerr != nil {
		t.Fatalf("run: %v", rerr)
	}

	write(t, filepath.Join(dir, "go.mod"), "module trades\n\ngo 1.27.0\n\n"+
		"require github.com/tamnd/kuma v0.0.0\n\n"+
		"replace github.com/tamnd/kuma => "+filepath.ToSlash(root)+"\n")
	if mkerr := os.Mkdir(filepath.Join(dir, "use"), 0o750); mkerr != nil {
		t.Fatal(mkerr)
	}
	write(t, filepath.Join(dir, "use", "main.go"), bindTheStruct)

	cmd := exec.Command(gotool, "run", "./use")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOPROXY=off", "GOFLAGS=")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go run: %v\n%s", err, out)
	}
	if got := strings.TrimSpace(string(out)); got != "[AAPL]" {
		t.Errorf("the written struct filtered to %q, and the answer is [AAPL]", got)
	}
}

// bindable is a file with a column of each type the struct writer has an
// opinion about, and a column whose name has to be carried in a tag.
const bindable = "symbol,price,qty,is_open,Order ID\n" +
	"AAPL,189.5,100,true,7\n" +
	"MSFT,411.2,50,false,8\n"

// bindTheStruct is the program TestTheWrittenStructBinds builds. Nothing in it
// was written by hand for this file: the struct, the handles and the frame all
// came from the same CSV.
const bindTheStruct = `package main

import (
	"fmt"

	"trades"

	"github.com/tamnd/kuma"
)

func main() {
	f, err := kuma.ReadCSVFile("trades.csv", nil)
	if err != nil {
		panic(err)
	}

	typed, err := kuma.Bind[trades.Trade](f)
	if err != nil {
		panic(err)
	}

	t := trades.TradeCols
	cheap, err := typed.Filter(t.Price.Lt(200).And(t.Qty.Gt(50)).And(t.IsOpen))
	if err != nil {
		panic(err)
	}

	symbols, err := t.Symbol.Series(cheap)
	if err != nil {
		panic(err)
	}
	fmt.Println(symbols.Values())
}
`
