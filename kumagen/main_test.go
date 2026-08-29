package main

import (
	"bytes"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestRunWritesTheDefaultFiles is the way go generate calls the tool: a
// directory, the types in it, and no say in where the files go.
func TestRunWritesTheDefaultFiles(t *testing.T) {
	dir := copyPackage(t, "testdata/trades")

	var out bytes.Buffer
	if err := run([]string{"-dir", dir, "-type", "Trade,Bar"}, &out); err != nil {
		t.Fatalf("run: %v", err)
	}
	if out.Len() > 0 {
		t.Errorf("run printed %q, and a run that worked says nothing", out.String())
	}

	for _, name := range []string{"trade", "bar"} {
		got := read(t, filepath.Join(dir, name+"_kuma.go"))
		want := read(t, filepath.Join("testdata", "trades", name+"_kuma.golden"))
		if got != want {
			t.Errorf("%s_kuma.go holds\n%s\nand it should hold\n%s", name, got, want)
		}
	}
}

// TestRunNamesTheFileInSnakeCase checks the default name for a type that is
// more than one word, which is where the two ways of writing it differ.
func TestRunNamesTheFileInSnakeCase(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, "book.go"), "package book\n\n"+
		"type OrderBook struct {\n\tPrice float64 `kuma:\"price\"`\n}\n")

	if err := run([]string{"-dir", dir, "-type", "OrderBook"}, &bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "order_book_kuma.go")); err != nil {
		t.Errorf("run did not write order_book_kuma.go: %v", err)
	}
}

// TestRunWritesToAFile covers -o, which is for a program that keeps its
// generated code somewhere of its own.
func TestRunWritesToAFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "handles.go")

	if err := run([]string{"-dir", "testdata/trades", "-type", "Trade", "-o", path},
		&bytes.Buffer{}); err != nil {
		t.Fatalf("run: %v", err)
	}

	got := read(t, path)
	if want := read(t, "testdata/trades/trade_kuma.golden"); got != want {
		t.Errorf("%s holds\n%s\nand it should hold\n%s", path, got, want)
	}
}

// TestRunWritesToStandardOutput is -o -, which is how a person looks at the
// output before letting the tool near their package.
func TestRunWritesToStandardOutput(t *testing.T) {
	var out bytes.Buffer
	if err := run([]string{"-dir", "testdata/trades", "-type", "Trade,Bar", "-o", "-"},
		&out); err != nil {
		t.Fatalf("run: %v", err)
	}

	want := read(t, "testdata/trades/trade_kuma.golden") +
		read(t, "testdata/trades/bar_kuma.golden")
	if out.String() != want {
		t.Errorf("run printed\n%s\nand it should have printed\n%s", out.String(), want)
	}
}

// TestRunPrintsUsage checks that -h explains itself and is not an error.
func TestRunPrintsUsage(t *testing.T) {
	var out bytes.Buffer
	if err := run([]string{"-h"}, &out); err != nil {
		t.Fatalf("run -h: %v", err)
	}

	for _, want := range []string{"kumagen [-dir", "-type", "-dir", "-o"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("the usage does not mention %q:\n%s", want, out.String())
		}
	}
}

// TestRunErrors is everything a command line can get wrong.
func TestRunErrors(t *testing.T) {
	cases := []struct {
		name string
		args []string
		says string
	}{
		{
			name: "no type at all",
			args: []string{"-dir", "testdata/trades"},
			says: "no type to write handles for",
		},
		{
			name: "a list with an empty name in it",
			args: []string{"-dir", "testdata/trades", "-type", "Trade,"},
			says: "empty name",
		},
		{
			name: "two types into the one file",
			args: []string{"-dir", "testdata/trades", "-type", "Trade,Bar", "-o", "both.go"},
			says: "2 types cannot go in the one file both.go",
		},
		{
			name: "a flag that is not a flag",
			args: []string{"-columns", "trades"},
			says: "not defined",
		},
		{
			name: "a directory that is not there",
			args: []string{"-dir", "testdata/nowhere", "-type", "Trade"},
			says: "nowhere",
		},
		{
			name: "a type that is not there",
			args: []string{"-dir", "testdata/trades", "-type", "Missing"},
			says: "no type called Missing",
		},
		{
			name: "a file that cannot be written",
			args: []string{"-dir", "testdata/trades", "-type", "Trade", "-o", "nowhere/x.go"},
			says: "nowhere",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var out bytes.Buffer
			err := run(c.args, &out)
			if err == nil {
				t.Fatalf("run(%q) worked and should not have, and printed %q",
					c.args, out.String())
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Errorf("run(%q) says %q, and it should mention %q", c.args, err, c.says)
			}
		})
	}
}

// TestGeneratedCodeCompiles is the test that matters, and it is the only one
// here that runs the go tool.
//
// The generated file is put in a module of its own next to the struct it was
// written for, with a program that binds a frame to that struct and filters it
// with the handles. If the names or the types were wrong the program would not
// compile, and if the columns kumagen named were not the ones Bind looks for it
// would not run.
func TestGeneratedCodeCompiles(t *testing.T) {
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

	dir := copyPackage(t, "testdata/trades")
	if rerr := run([]string{"-dir", dir, "-type", "Trade"}, &bytes.Buffer{}); rerr != nil {
		t.Fatalf("run: %v", rerr)
	}
	write(t, filepath.Join(dir, "go.mod"), "module trades\n\ngo 1.27.0\n\n"+
		"require github.com/tamnd/kuma v0.0.0\n\n"+
		"replace github.com/tamnd/kuma => "+filepath.ToSlash(root)+"\n")
	if mkerr := os.Mkdir(filepath.Join(dir, "use"), 0o755); mkerr != nil {
		t.Fatalf("making the program directory: %v", mkerr)
	}
	write(t, filepath.Join(dir, "use", "main.go"), useTheHandles)

	cmd := exec.Command(gotool, "run", "./use")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "GOPROXY=off", "GOFLAGS=")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go run: %v\n%s", err, out)
	}
	if got := strings.TrimSpace(string(out)); got != "[AAPL]" {
		t.Errorf("the generated handles filtered to %q, and the answer is [AAPL]", got)
	}
}

// useTheHandles is the program TestGeneratedCodeCompiles builds. It is written
// the way a person would write it, which is the point of the whole tool.
const useTheHandles = `package main

import (
	"fmt"
	"time"

	"trades"

	"github.com/tamnd/kuma"
)

func main() {
	f, err := kuma.NewFrame(
		kuma.NewSeries("symbol", "AAPL", "MSFT").Column(),
		kuma.NewSeries("price", 189.5, 411.2).Column(),
		kuma.NewSeries("qty", int64(100), int64(50)).Column(),
		kuma.NewSeries("filled", true, false).Column(),
		kuma.NewSeries("ts", time.Now(), time.Now()).Column(),
		kuma.NewSeries("order_id", int64(7), int64(8)).Column(),
		kuma.NewSeries("seen", time.Now(), time.Now()).Column(),
	)
	if err != nil {
		panic(err)
	}

	typed, err := kuma.Bind[trades.Trade](f)
	if err != nil {
		panic(err)
	}

	t := trades.TradeCols
	cheap, err := typed.Filter(t.Price.Lt(200).And(t.Qty.Gt(50)).And(t.Filled))
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

// TestRunReportsAWriteThatFails covers the -o - path when whatever is reading
// the output has gone away, which is a broken pipe rather than anything the
// caller did wrong.
func TestRunReportsAWriteThatFails(t *testing.T) {
	runs := map[string][]string{
		"the generated file": {"-dir", "testdata/trades", "-type", "Trade", "-o", "-"},
		"the usage":          {"-h"},
	}

	for name, args := range runs {
		t.Run(name, func(t *testing.T) {
			err := run(args, brokenPipe{})
			if err == nil {
				t.Fatal("run wrote to a writer that refuses everything and said nothing")
			}
			if !strings.Contains(err.Error(), "broken pipe") {
				t.Errorf("run says %q, and it should pass on what the write said", err)
			}
		})
	}
}

// brokenPipe is standard output with nothing on the other end of it.
type brokenPipe struct{}

func (brokenPipe) Write([]byte) (int, error) { return 0, errors.New("broken pipe") }

// copyPackage copies the Go files of a package into a directory of its own, so
// that a test can write into it without touching what is checked in.
func copyPackage(t *testing.T, dir string) string {
	t.Helper()

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("reading %s: %v", dir, err)
	}

	out := t.TempDir()
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		write(t, filepath.Join(out, e.Name()), read(t, filepath.Join(dir, e.Name())))
	}
	return out
}

// read returns the whole of a file, or fails the test.
func read(t *testing.T, path string) string {
	t.Helper()

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	return string(b)
}
