package ipc_test

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

// TestPyarrowStream checks the stream format against another implementation of
// it.
//
// The message checks next to this one pass one message at a time, which leaves
// out the part that makes a stream a stream: the schema in front, the batches
// after it, and the marker saying there are no more. A reader written against
// the same misreading as the writer would round trip perfectly and produce a
// file nobody else can open, so the only useful check is somebody else's reader.
//
// Each side writes whole streams and says what is in them, and each side reads
// the other's. pyarrow also writes kuma's streams back out with its own writer,
// so what comes back has been through both.
//
// It skips when there is no python3 with pyarrow in it, since that is a large
// dependency to ask of somebody running the tests. CI installs it, so the check
// runs on every commit.
func TestPyarrowStream(t *testing.T) {
	python := findPython(t)
	dir := t.TempDir()
	cases := batchPyarrowCases(t)

	var manifest strings.Builder
	for _, c := range cases {
		rendered := make([]string, len(c.batches))
		for i, b := range c.batches {
			rendered[i] = renderBatch(c.schema, b)
		}
		writeStream(t, filepath.Join(dir, "go-"+c.name+".arrows"), c.schema, c.batches)
		fmt.Fprintf(&manifest, "%s\t%s\n", c.name, strings.Join(rendered, "|"))
	}
	// A stream with no batches in it, which the batch cases cannot cover since
	// a case with no batches has no message to write.
	writeStream(t, filepath.Join(dir, "go-none.arrows"),
		dtype.Schema{Fields: []dtype.Field{{Name: "id", Type: dtype.Int64}}}, nil)
	fmt.Fprintf(&manifest, "none\t\n")

	if err := os.WriteFile(filepath.Join(dir, "go.txt"), []byte(manifest.String()), 0o600); err != nil {
		t.Fatal(err)
	}

	run := exec.Command(python, filepath.Join("testdata", "pyarrow", "stream.py"), dir)
	// This is the one cross check whose script imports another, so without this
	// python would leave a __pycache__ behind in the source tree.
	run.Env = append(os.Environ(), "PYTHONDONTWRITEBYTECODE=1")
	out, err := run.CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	t.Log(strings.TrimSpace(string(out)))

	checkBackStreams(t, dir, cases)
	checkPythonStreams(t, dir)
}

// writeStream writes one whole stream to a file.
func writeStream(t *testing.T, path string, s dtype.Schema, batches []ipc.Batch) {
	t.Helper()
	var buf bytes.Buffer
	w, err := ipc.NewWriter(&buf, s)
	if err != nil {
		t.Fatalf("%s: NewWriter: %v", filepath.Base(path), err)
	}
	for i, b := range batches {
		if err := w.Write(b); err != nil {
			t.Fatalf("%s: Write batch %d: %v", filepath.Base(path), i, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("%s: Close: %v", filepath.Base(path), err)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
}

// checkBackStreams reads the streams pyarrow wrote back out with its own writer
// and compares them with what kuma sent. The schema in these is pyarrow's
// spelling of the one kuma wrote, so this is also where a type that survives one
// trip and not two would show up.
func checkBackStreams(t *testing.T, dir string, cases []pyarrowBatchCase) {
	t.Helper()
	for _, c := range cases {
		got, s := readStream(t, filepath.Join(dir, "back-"+c.name+".arrows"))
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

// checkPythonStreams reads the streams pyarrow built itself and compares each
// one with the values pyarrow says are in it. These are the layouts kuma does
// not write, arriving the way they arrive from anywhere else.
func checkPythonStreams(t *testing.T, dir string) {
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

		batches, s := readStream(t, filepath.Join(dir, "py-"+name+".arrows"))
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

// readStream reads a whole stream out of a file. It reports the error itself and
// gives back nothing, since one unreadable stream should not stop the rest of
// them from being checked.
func readStream(t *testing.T, path string) ([]ipc.Batch, dtype.Schema) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Error(err)
		return nil, dtype.Schema{}
	}
	defer f.Close()

	r, err := ipc.NewReader(f)
	if err != nil {
		t.Errorf("%s: NewReader: %v", filepath.Base(path), err)
		return nil, dtype.Schema{}
	}
	batches := []ipc.Batch{}
	for b := range r.All() {
		batches = append(batches, b)
	}
	if err := r.Err(); err != nil {
		t.Errorf("%s: %v", filepath.Base(path), err)
		return nil, dtype.Schema{}
	}
	return batches, r.Schema()
}
