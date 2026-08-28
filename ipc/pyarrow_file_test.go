package ipc_test

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

// TestPyarrowFile checks the file format against another implementation of it.
//
// The stream check next to this one passes whole streams, which leaves out what
// a file adds: the magic at both ends and a footer saying where every batch is.
// A footer whose blocks are a few bytes out reads perfectly in the writer that
// produced it and nowhere else, so what is being asked here is whether pyarrow
// can open a file kuma wrote and pull a batch out of the middle of it.
//
// Both sides read the batches backwards, which is the order that has to go
// through the footer rather than through the batch in front.
//
// It skips when there is no python3 with pyarrow in it, since that is a large
// dependency to ask of somebody running the tests. CI installs it, so the check
// runs on every commit.
func TestPyarrowFile(t *testing.T) {
	python := findPython(t)
	dir := t.TempDir()
	cases := append(batchPyarrowCases(t), dictPyarrowCases(t)...)
	writeCases(t, dir, ".arrow", cases, writeFile)

	run := exec.Command(python, filepath.Join("testdata", "pyarrow", "file.py"), dir)
	// The script imports the other two, so without this python would leave a
	// __pycache__ behind in the source tree.
	run.Env = append(os.Environ(), "PYTHONDONTWRITEBYTECODE=1")
	out, err := run.CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	t.Log(strings.TrimSpace(string(out)))

	checkBackWhole(t, dir, ".arrow", cases, readFile)
	checkPythonWhole(t, dir, ".arrow", readFile)
}

// writeFile writes one whole file.
func writeFile(t *testing.T, path string, s dtype.Schema, batches []ipc.Batch) {
	t.Helper()
	var buf bytes.Buffer
	w, err := ipc.NewFileWriter(&buf, s)
	if err != nil {
		t.Fatalf("%s: NewFileWriter: %v", filepath.Base(path), err)
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

// readFile reads every batch out of a file, last one first, and gives them back
// in order. It reports the error itself and gives back nothing, since one
// unreadable file should not stop the rest of them from being checked.
func readFile(t *testing.T, path string) ([]ipc.Batch, dtype.Schema) {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Error(err)
		return nil, dtype.Schema{}
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		t.Error(err)
		return nil, dtype.Schema{}
	}
	r, err := ipc.NewFileReader(f, info.Size())
	if err != nil {
		t.Errorf("%s: NewFileReader: %v", filepath.Base(path), err)
		return nil, dtype.Schema{}
	}

	batches := []ipc.Batch{}
	for i := r.NumBatches() - 1; i >= 0; i-- {
		b, err := r.Batch(i)
		if err != nil {
			t.Errorf("%s: Batch(%d): %v", filepath.Base(path), i, err)
			return nil, dtype.Schema{}
		}
		batches = append(batches, b)
	}
	slices.Reverse(batches)
	return batches, r.Schema()
}
