package ipc_test

import (
	"bytes"
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
// the other's. What kuma writes is then written back out by pyarrow's own
// writer, so the batches that come back have been through both.
//
// It skips when there is no python3 with pyarrow in it, since that is a large
// dependency to ask of somebody running the tests. CI installs it, so the check
// runs on every commit.
func TestPyarrowStream(t *testing.T) {
	python := findPython(t)
	dir := t.TempDir()
	cases := batchPyarrowCases(t)
	writeCases(t, dir, ".arrows", cases, writeStream)

	run := exec.Command(python, filepath.Join("testdata", "pyarrow", "stream.py"), dir)
	// The script imports the batch check's, so without this python would leave a
	// __pycache__ behind in the source tree.
	run.Env = append(os.Environ(), "PYTHONDONTWRITEBYTECODE=1")
	out, err := run.CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	t.Log(strings.TrimSpace(string(out)))

	checkBackWhole(t, dir, ".arrows", cases, readStream)
	checkPythonWhole(t, dir, ".arrows", readStream)
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
