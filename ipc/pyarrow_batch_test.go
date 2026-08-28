package ipc_test

import (
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

// TestPyarrowBatch checks the record batch message against another
// implementation of it.
//
// The schema check next to this one passes types across. This passes values,
// which is the half where a wrong answer is silent: a buffer that starts eight
// bytes late still reads as numbers. Each side writes the batches it can build
// along with the values it says are in them, and each side reads the other's and
// says what it found.
//
// Every message here is one batch on its own, which is what pyarrow's serialize
// and read_record_batch pass around. The stream that puts a schema in front of
// several of them is checked next door.
//
// It skips when there is no python3 with pyarrow in it, since that is a large
// dependency to ask of somebody running the tests. CI installs it, so the check
// runs on every commit.
func TestPyarrowBatch(t *testing.T) {
	python := findPython(t)
	dir := t.TempDir()
	cases := batchPyarrowCases(t)

	var manifest strings.Builder
	for _, c := range cases {
		msg, err := ipc.EncodeSchema(c.schema)
		if err != nil {
			t.Fatalf("%s: EncodeSchema: %v", c.name, err)
		}
		writeMessage(t, dir, "go-"+c.name+"-schema.arrow", msg)

		rendered := make([]string, len(c.batches))
		for i, b := range c.batches {
			msg, err := ipc.EncodeBatch(c.schema, b)
			if err != nil {
				t.Fatalf("%s: EncodeBatch: %v", c.name, err)
			}
			writeMessage(t, dir, fmt.Sprintf("go-%s-%d.arrow", c.name, i), msg)
			rendered[i] = renderBatch(c.schema, b)
		}
		fmt.Fprintf(&manifest, "%s\t%d\t%s\n", c.name, len(c.batches), strings.Join(rendered, "|"))
	}
	if err := os.WriteFile(filepath.Join(dir, "go.txt"), []byte(manifest.String()), 0o600); err != nil {
		t.Fatal(err)
	}

	run := exec.Command(python, filepath.Join("testdata", "pyarrow", "batch.py"), dir)
	out, err := run.CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	t.Log(strings.TrimSpace(string(out)))

	checkBackBatches(t, dir, cases)
	checkPythonBatches(t, dir)
}

// checkBackBatches reads the batches pyarrow wrote back out and compares them
// with the ones kuma sent. What survives the trip is a batch both sides agree
// on, value for value, and the way back is pyarrow's own writer rather than a
// copy of the bytes it was handed.
func checkBackBatches(t *testing.T, dir string, cases []pyarrowBatchCase) {
	t.Helper()
	for _, c := range cases {
		for i, want := range c.batches {
			msg, err := os.ReadFile(filepath.Join(dir, fmt.Sprintf("back-%s-%d.arrow", c.name, i)))
			if err != nil {
				t.Errorf("%s: %v", c.name, err)
				continue
			}
			got, _, err := ipc.DecodeBatch(c.schema, msg)
			if err != nil {
				t.Errorf("%s batch %d: DecodeBatch of what pyarrow wrote: %v", c.name, i, err)
				continue
			}
			if got.Length != want.Length {
				t.Errorf("%s batch %d: came back with %d rows, want %d",
					c.name, i, got.Length, want.Length)
				continue
			}
			for k, col := range want.Columns {
				equalArrays(t, got.Columns[k], col)
			}
		}
	}
}

// checkPythonBatches reads the batches pyarrow built itself and compares each
// one with the values pyarrow says are in it.
//
// These are the layouts kuma does not write. Every other implementation stores
// text as offsets into one buffer rather than as views, so this is the side of
// the check that reads what the rest of the world actually sends.
func checkPythonBatches(t *testing.T, dir string) {
	t.Helper()
	manifest, err := os.ReadFile(filepath.Join(dir, "py.txt"))
	if err != nil {
		t.Fatal(err)
	}

	for line := range strings.Lines(strings.TrimSpace(string(manifest))) {
		fields := strings.Split(strings.TrimSpace(line), "\t")
		if len(fields) != 3 {
			t.Fatalf("py.txt: %q is not a name, a count and a rendering", line)
		}
		name, want := fields[0], fields[2]
		count, err := strconv.Atoi(fields[1])
		if err != nil {
			t.Fatalf("py.txt: %s: %v", name, err)
		}

		msg, err := os.ReadFile(filepath.Join(dir, "py-"+name+"-schema.arrow"))
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		s, err := ipc.DecodeSchema(msg)
		if err != nil {
			t.Errorf("%s: DecodeSchema: %v", name, err)
			continue
		}

		got := make([]string, 0, count)
		for i := range count {
			msg, err := os.ReadFile(filepath.Join(dir, fmt.Sprintf("py-%s-%d.arrow", name, i)))
			if err != nil {
				t.Errorf("%s: %v", name, err)
				break
			}
			b, rest, err := ipc.DecodeBatch(s, msg)
			if err != nil {
				t.Errorf("%s batch %d: DecodeBatch: %v", name, i, err)
				break
			}
			if len(rest) != 0 {
				t.Errorf("%s batch %d: %d bytes left after the batch, want none", name, i, len(rest))
			}
			got = append(got, renderBatch(s, b))
		}
		if line := strings.Join(got, "|"); line != want {
			t.Errorf("%s: read as\n%s\nwant\n%s", name, line, want)
		}
	}
}

func writeMessage(t *testing.T, dir, name string, msg []byte) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), msg, 0o600); err != nil {
		t.Fatal(err)
	}
}

// renderBatch writes the values of a batch the way both sides of the cross check
// write them: a column is its name, an equals sign and its values, columns are
// separated by semicolons and values by commas, and a missing value is null.
//
// Values only. The types are what the schema check is for, and a rendering that
// carried them would fail there first.
func renderBatch(s dtype.Schema, b ipc.Batch) string {
	parts := make([]string, len(s.Fields))
	for i, f := range s.Fields {
		col := b.Columns[i]
		vals := make([]string, col.Len())
		for k := range col.Len() {
			vals[k] = renderValue(col, k)
		}
		parts[i] = f.Name + "=" + strings.Join(vals, ",")
	}
	return strings.Join(parts, ";")
}

// renderValue writes one value as the two sides agree to write it: bytes as
// hex, a timestamp or a date as the number it is stored as rather than the day
// it means, and a float as the shortest text that reads back as itself.
func renderValue(a *array.Array, i int) string {
	if a.IsNull(i) {
		return "null"
	}
	switch a.DType().Kind() {
	case dtype.BoolKind:
		return strconv.FormatBool(a.Bool(i))
	case dtype.StringKind:
		return string(a.Bytes(i))
	case dtype.BinaryKind, dtype.FixedSizeBinaryKind:
		return hex.EncodeToString(a.Bytes(i))
	case dtype.Float64Kind:
		return strconv.FormatFloat(a.Value[float64](i), 'g', -1, 64)
	case dtype.Int32Kind, dtype.Date32Kind:
		return strconv.FormatInt(int64(a.Value[int32](i)), 10)
	default:
		return strconv.FormatInt(a.Value[int64](i), 10)
	}
}

// The plumbing the stream and the file cross checks share. Both of them write
// one of these cases per file along with a manifest saying what is in it, hand
// the directory to a script, and read back what the script wrote. The only
// difference between the two is what one of those files is, so that is a pair of
// functions rather than two copies of the rest.
type (
	crossWriter func(t *testing.T, path string, s dtype.Schema, batches []ipc.Batch)
	crossReader func(t *testing.T, path string) ([]ipc.Batch, dtype.Schema)
)

// writeCases writes a file per case and the manifest, which is one line per case
// holding the name and the rendering of every batch in it joined by a bar.
func writeCases(t *testing.T, dir, ext string, cases []pyarrowBatchCase, write crossWriter) {
	t.Helper()

	var manifest strings.Builder
	for _, c := range cases {
		rendered := make([]string, len(c.batches))
		for i, b := range c.batches {
			rendered[i] = renderBatch(c.schema, b)
		}
		write(t, filepath.Join(dir, "go-"+c.name+ext), c.schema, c.batches)
		fmt.Fprintf(&manifest, "%s\t%s\n", c.name, strings.Join(rendered, "|"))
	}
	// One with no batches in it, which the batch cases cannot cover since a case
	// with no batches has no message to write.
	write(t, filepath.Join(dir, "go-none"+ext),
		dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64}}}, nil)
	fmt.Fprint(&manifest, "none\t\n")

	if err := os.WriteFile(filepath.Join(dir, "go.txt"), []byte(manifest.String()), 0o600); err != nil {
		t.Fatal(err)
	}
}

// checkBackWhole reads what kuma wrote and pyarrow wrote back out with its own
// writer, and compares it with what kuma sent. The schema in these is pyarrow's
// spelling of the one kuma wrote, so this is also where a type that survives one
// trip and not two would show up.
func checkBackWhole(t *testing.T, dir, ext string, cases []pyarrowBatchCase, read crossReader) {
	t.Helper()
	for _, c := range cases {
		got, s := read(t, filepath.Join(dir, "back-"+c.name+ext))
		if got == nil {
			continue
		}
		if len(got) != len(c.batches) {
			t.Errorf("%s: came back with %d batches, want %d", c.name, len(got), len(c.batches))
			continue
		}
		if !s.Equal(c.schema) {
			t.Errorf("%s: came back with the schema %v, want %v", c.name, s, c.schema)
		}
		for i, want := range c.batches {
			if got[i].Length != want.Length {
				t.Errorf("%s batch %d: came back with %d rows, want %d",
					c.name, i, got[i].Length, want.Length)
				continue
			}
			for k, col := range want.Columns {
				equalArrays(t, got[i].Columns[k], col)
			}
		}
	}
}

// checkPythonWhole reads what pyarrow built itself and compares each one with the
// values pyarrow says are in it. These are the layouts kuma does not write,
// arriving the way they arrive from anywhere else.
func checkPythonWhole(t *testing.T, dir, ext string, read crossReader) {
	t.Helper()
	manifest, err := os.ReadFile(filepath.Join(dir, "py.txt"))
	if err != nil {
		t.Fatal(err)
	}

	for line := range strings.Lines(string(manifest)) {
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			continue
		}
		name, want, ok := strings.Cut(line, "\t")
		if !ok {
			t.Fatalf("py.txt: %q is not a name and a rendering", line)
		}

		batches, s := read(t, filepath.Join(dir, "py-"+name+ext))
		if batches == nil && want != "" {
			continue
		}
		got := make([]string, len(batches))
		for i, b := range batches {
			got[i] = renderBatch(s, b)
		}
		if joined := strings.Join(got, "|"); joined != want {
			t.Errorf("%s: read as\n%s\nwant\n%s", name, joined, want)
		}
	}
}

type pyarrowBatchCase struct {
	name    string
	schema  dtype.Schema
	batches []ipc.Batch
}

// batchPyarrowCases are the batches kuma writes for pyarrow to read. Between
// them they cover every buffer layout it can write: the fixed width ones, the
// bits of a bool column, the views of a text column with one data block and
// with several, and the column that has no buffers at all.
func batchPyarrowCases(t *testing.T) []pyarrowBatchCase {
	t.Helper()

	flat := dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64, Nullable: true},
		{Name: "price", Type: dtype.Float64},
		{Name: "live", Type: dtype.Bool, Nullable: true},
		{Name: "at", Type: dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}},
		{Name: "day", Type: dtype.Date32},
	}}
	text := dtype.Schema{Fields: []dtype.Field{
		{Name: "symbol", Type: dtype.String, Nullable: true},
		{Name: "payload", Type: dtype.Binary, Nullable: true},
		{Name: "key", Type: dtype.FixedSizeBinary{ByteWidth: 3}},
	}}

	return []pyarrowBatchCase{
		{
			name:   "flat",
			schema: flat,
			batches: []ipc.Batch{
				{Length: 4, Columns: []*array.Array{
					buildInts(t, []int64{10, 20, 30, 40}, 1),
					array.Of[float64](1.5, -2.25, 0.75, 12345.5),
					buildBools(t, []bool{true, false, true, true}, 2),
					fixed(t, dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}, 4),
					fixed(t, dtype.Date32, 4),
				}},
				{Length: 2, Columns: []*array.Array{
					buildInts(t, []int64{50, 60}),
					array.Of[float64](7.5, 8.25),
					buildBools(t, []bool{false, false}),
					fixed(t, dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}, 2),
					fixed(t, dtype.Date32, 2),
				}},
			},
		},
		{
			name:   "text",
			schema: text,
			batches: []ipc.Batch{{Length: 4, Columns: []*array.Array{
				buildStrings(t, []string{"a", "bb", "ccc", "dddd"}, 1),
				buildBinary(t, [][]byte{{1}, {2, 3}, {}, {4, 5, 6}}),
				fixed(t, dtype.FixedSizeBinary{ByteWidth: 3}, 4),
			}}},
		},
		{
			name:   "blocks",
			schema: dtype.Schema{Fields: []dtype.Field{{Name: "text", Type: dtype.String}}},
			batches: []ipc.Batch{{Length: blockRows, Columns: []*array.Array{
				manyStrings(t),
			}}},
		},
		{
			name: "nothing",
			schema: dtype.Schema{Fields: []dtype.Field{
				{Name: "nothing", Type: dtype.Null, Nullable: true},
				{Name: "id", Type: dtype.Int64},
			}},
			batches: []ipc.Batch{{Length: 3, Columns: []*array.Array{
				array.NewNull(3),
				buildInts(t, []int64{1, 2, 3}),
			}}},
		},
		{
			name:   "empty",
			schema: text,
			batches: []ipc.Batch{{Columns: []*array.Array{
				buildStrings(t, nil),
				buildBinary(t, nil),
				fixed(t, dtype.FixedSizeBinary{ByteWidth: 3}, 0),
			}}},
		},
	}
}

// buildBools builds a bool column, which is the one type the builders next to
// this one do not cover and the only one whose values are bits.
func buildBools(t *testing.T, vals []bool, nulls ...int) *array.Array {
	t.Helper()
	b, err := array.NewBuilder(dtype.Bool)
	if err != nil {
		t.Fatalf("NewBuilder = %v", err)
	}
	for i, v := range vals {
		if isNull(i, nulls) {
			b.AppendNull()
			continue
		}
		b.AppendBool(v)
	}
	return b.Finish()
}
