//go:build cgo

package ipc_test

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestPyarrow checks the C data interface against another implementation of it.
//
// Everything else in this package tests kuma against kuma, which cannot say
// whether the format string kuma writes for a timestamp is the one the rest of
// the world reads as a timestamp, or whether the memory it hands over is the
// memory another library ends up reading. Only a second implementation in the
// same process can say that, so this builds testdata/pyarrow into a shared
// library and lets pyarrow drive it. What the two of them agree on is the
// specification rather than kuma's reading of it.
//
// It skips when there is no python3 with pyarrow in it, since that is a large
// dependency to ask of somebody running the tests. CI installs it, so the
// check runs on every commit.
func TestPyarrow(t *testing.T) {
	python := findPython(t)
	dir := t.TempDir()
	lib := filepath.Join(dir, "crosscheck"+libraryExt())

	build := exec.Command("go", "build", "-buildmode=c-shared", "-o", lib, "./testdata/pyarrow")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building the cross check library: %v\n%s", err, out)
	}

	run := exec.Command(python, filepath.Join("testdata", "pyarrow", "crosscheck.py"), lib)
	out, err := run.CombinedOutput()
	if err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
	t.Log(strings.TrimSpace(string(out)))
}

// findPython returns an interpreter that can import pyarrow, or skips.
func findPython(t *testing.T) string {
	t.Helper()
	for _, name := range []string{"python3", "python"} {
		path, err := exec.LookPath(name)
		if err != nil {
			continue
		}
		if err := exec.Command(path, "-c", "import pyarrow").Run(); err == nil {
			return path
		}
	}
	t.Skip("no python3 with pyarrow, which this needs to have somebody to disagree with")
	return ""
}

func libraryExt() string {
	switch runtime.GOOS {
	case "windows":
		return ".dll"
	case "darwin":
		return ".dylib"
	default:
		return ".so"
	}
}
