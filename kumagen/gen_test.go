package main

import (
	"bytes"
	"flag"
	"go/ast"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var update = flag.Bool("update", false, "rewrite the golden files with what kumagen wrote")

// TestGenerate writes the handles for the two structs in the trades package and
// compares them with the files checked in next to it. Run with -update to
// rewrite those after a deliberate change to the output.
func TestGenerate(t *testing.T) {
	p := load(t, "testdata/trades")

	for _, name := range []string{"Trade", "Bar", "Quote"} {
		t.Run(name, func(t *testing.T) {
			got, err := generate(p, name)
			if err != nil {
				t.Fatalf("generate(%s): %v", name, err)
			}

			golden := filepath.Join("testdata", "trades", strings.ToLower(name)+"_kuma.golden")
			if *update {
				if werr := os.WriteFile(golden, got, 0o644); werr != nil {
					t.Fatalf("writing %s: %v", golden, werr)
				}
			}
			want, err := os.ReadFile(golden)
			if err != nil {
				t.Fatalf("reading %s: %v", golden, err)
			}
			if !bytes.Equal(got, want) {
				t.Errorf("generate(%s) wrote\n%s\nand %s holds\n%s", name, got, golden, want)
			}
		})
	}
}

// TestGenerateIsDeterministic checks the property the whole tool rests on. A
// package that has not changed has to produce the same bytes, or running
// kumagen in CI and looking for a diff would report a change that is not one.
func TestGenerateIsDeterministic(t *testing.T) {
	first, err := generate(load(t, "testdata/trades"), "Trade")
	if err != nil {
		t.Fatalf("the first generate: %v", err)
	}
	second, err := generate(load(t, "testdata/trades"), "Trade")
	if err != nil {
		t.Fatalf("the second generate: %v", err)
	}
	if !bytes.Equal(first, second) {
		t.Errorf("two runs over the same package wrote different files:\n%s\nand\n%s",
			first, second)
	}
}

// TestGenerateErrors is one struct per reason there is to refuse one.
func TestGenerateErrors(t *testing.T) {
	cases := []struct {
		name string
		dir  string
		typ  string
		says string
	}{
		{
			name: "a type that is not there",
			typ:  "Missing",
			says: "no type called Missing",
		},
		{
			name: "a type that is not a struct",
			typ:  "NotAStruct",
			says: "NotAStruct is a []int rather than a struct",
		},
		{
			name: "a type with type parameters",
			typ:  "Generic",
			says: "Generic has type parameters",
		},
		{
			name: "a field of a width with no handle",
			typ:  "Narrow",
			says: "field Qty of Narrow has type int32",
		},
		{
			name: "a field that is another struct",
			typ:  "Nested",
			says: "field Inner of Nested has type Narrow",
		},
		{
			name: "a field that is a pointer",
			typ:  "Pointed",
			says: "field Price of Pointed has type *float64",
		},
		{
			name: "an embedded type that is not a column",
			typ:  "Embedded",
			says: "field Generic of Embedded has type Generic[int]",
		},
		{
			name: "a struct with every field left out",
			typ:  "Hidden",
			says: "Hidden has no field that names a column",
		},
		{
			name: "an embedded type with two type arguments",
			typ:  "EmbeddedPair",
			says: "field Pair of EmbeddedPair has type Pair[int64, string]",
		},
		{
			name: "an embedded pointer",
			typ:  "EmbeddedPointer",
			says: "field Narrow of EmbeddedPointer has type *Narrow",
		},
		{
			name: "a type declared in two files",
			typ:  "Twice",
			says: "Twice is declared twice, in bad.go and in twice.go",
		},
		{
			name: "a struct in an external test package",
			dir:  "testdata/externaltest",
			typ:  "Quote",
			says: "Quote is declared in quotes_test",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			dir := c.dir
			if dir == "" {
				dir = "testdata/bad"
			}

			src, err := generate(load(t, dir), c.typ)
			if err == nil {
				t.Fatalf("generate(%s) wrote a file and should not have:\n%s", c.typ, src)
			}
			if !strings.Contains(err.Error(), c.says) {
				t.Errorf("generate(%s) says %q, and it should mention %q", c.typ, err, c.says)
			}
		})
	}
}

// TestLoadPackageErrors covers the two ways a directory can fail to be a
// package. Both are written out in a temporary directory rather than kept under
// testdata, since a file that does not parse would fail the gofmt check the
// repository runs over everything.
func TestLoadPackageErrors(t *testing.T) {
	t.Run("a directory that is not there", func(t *testing.T) {
		if _, err := loadPackage(filepath.Join(t.TempDir(), "nowhere")); err == nil {
			t.Fatal("loadPackage read a directory that does not exist")
		}
	})

	t.Run("a directory with no Go file in it", func(t *testing.T) {
		dir := t.TempDir()
		write(t, filepath.Join(dir, "notes.txt"), "this is not Go")

		_, err := loadPackage(dir)
		if err == nil {
			t.Fatal("loadPackage found a package in a directory of text")
		}
		if !strings.Contains(err.Error(), "no Go file") {
			t.Errorf("loadPackage says %q, and it should mention that there is no Go file", err)
		}
	})

	t.Run("a file that does not parse", func(t *testing.T) {
		dir := t.TempDir()
		write(t, filepath.Join(dir, "half.go"), "package half\n\ntype Trade struct {\n")

		_, err := loadPackage(dir)
		if err == nil {
			t.Fatal("loadPackage parsed a file that is half written")
		}
		if !strings.Contains(err.Error(), "half.go") {
			t.Errorf("loadPackage says %q, and it should name the file", err)
		}
	})
}

// TestEmbeddedFieldWithNoName is the guard in fieldNames, which the Go syntax
// for an embedded field says cannot happen, so the field is built here rather
// than parsed out of a file.
func TestEmbeddedFieldWithNoName(t *testing.T) {
	f := &ast.Field{Type: &ast.ArrayType{Elt: ast.NewIdent("int64")}}

	names, err := fieldNames(f)
	if err == nil {
		t.Fatalf("fieldNames called an embedded slice %q", names)
	}
	if !strings.Contains(err.Error(), "[]int64") {
		t.Errorf("fieldNames says %q, and it should name the type", err)
	}
}

// TestHandleKindNeedsTheTimePackage checks that the time of time.Time is the
// package rather than any other name that happens to be spelled that way.
func TestHandleKindNeedsTheTimePackage(t *testing.T) {
	dir := t.TempDir()
	write(t, filepath.Join(dir, "clock.go"), "package clock\n\n"+
		"import time \"github.com/tamnd/kuma/kumagen/testdata/nottime\"\n\n"+
		"type Tick struct {\n\tAt time.Time `kuma:\"at\"`\n}\n")

	_, err := generate(load(t, dir), "Tick")
	if err == nil {
		t.Fatal("generate wrote a handle for a Time that is not the time package")
	}
	if !strings.Contains(err.Error(), "field At of Tick has type time.Time") {
		t.Errorf("generate says %q, and it should name the field and the type", err)
	}
}

// TestLiteralsThatWillNotUnquote covers the two guards over strconv.Unquote.
// The parser has already read every tag and every import path as a string
// literal, so neither can fail for a package that was parsed, and the way to
// reach them is to build the literal here.
func TestLiteralsThatWillNotUnquote(t *testing.T) {
	bad := &ast.BasicLit{Kind: token.STRING, Value: `"\q"`}

	if got := tagName(bad); got != "" {
		t.Errorf("tagName read %q out of a tag that is not a string", got)
	}

	f := &ast.File{Imports: []*ast.ImportSpec{{Path: bad}}}
	if got := importsOf(f); len(got) != 0 {
		t.Errorf("importsOf found %v in a file whose import path is not a string", got)
	}
}

// load parses a package or fails the test.
func load(t *testing.T, dir string) *pkg {
	t.Helper()

	p, err := loadPackage(dir)
	if err != nil {
		t.Fatalf("loadPackage(%s): %v", dir, err)
	}
	return p
}

// write puts a file where a test wants one.
func write(t *testing.T, path, content string) {
	t.Helper()

	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("writing %s: %v", path, err)
	}
}
