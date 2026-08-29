// Kumagen writes the column handles for a Go struct, so that a query names a
// column with a field selector the compiler checks rather than with a string.
//
// Usage:
//
//	kumagen [-dir directory] [-o file] -type Name[,Name...]
//	kumagen -from file [-dir directory] [-o file] [-lines n] [-package name] [-f] -type Name
//
// The first form writes the handles for a struct that is already written. The
// second writes the struct itself, taking the columns from a data file, for the
// case where the data came first.
//
// Given this struct in a package:
//
//	//go:generate kumagen -type Trade
//
//	type Trade struct {
//		Symbol string    `kuma:"symbol"`
//		Price  float64   `kuma:"price"`
//		Qty    int64     `kuma:"qty"`
//		TS     time.Time `kuma:"ts"`
//	}
//
// kumagen writes trade_kuma.go next to it, holding one variable:
//
//	var TradeCols = struct {
//		Symbol kuma.StrCol[Trade]
//		Price  kuma.F64Col[Trade]
//		Qty    kuma.I64Col[Trade]
//		TS     kuma.TimeCol[Trade]
//	}{
//		Symbol: kuma.NewStrCol[Trade]("symbol"),
//		Price:  kuma.NewF64Col[Trade]("price"),
//		Qty:    kuma.NewI64Col[Trade]("qty"),
//		TS:     kuma.NewTimeCol[Trade]("ts"),
//	}
//
// After that a query is an ordinary Go expression over ordinary Go values, and
// renaming a column is a rename in the struct followed by go generate, with the
// compiler pointing at every line that has to change:
//
//	t := TradeCols
//	dear, err := trades.Filter(t.Price.Gt(100).And(t.Symbol.Ne("MSFT")))
//
// The column a field names is its kuma tag, or the field name in snake case
// when it has no tag, which is the rule [kuma.Bind] follows when it checks a
// frame against the same struct. A field tagged "-" is left out and so is an
// unexported one.
//
// # Writing the struct from a file
//
// A program that starts from data rather than from a type asks for the struct
// instead:
//
//	kumagen -from trades.parquet -type Trade
//
// That reads the schema of the file, writes trade.go holding a Trade with a
// field for each column, and puts a go:generate line above it so that the
// handles follow from a go generate. The types are the ones a [kuma.Bind] reads
// out of those columns and no others, so an int32 column is an int32 field, and
// a column whose name is not the snake case of its field is named in a kuma
// tag.
//
// Parquet and Arrow IPC keep their schema in a footer, so those two are read
// from the end of the file whatever its size. A CSV, TSV or NDJSON file has no
// schema in it, so the first thousand lines are read and the types are worked
// out from those, which -lines changes.
//
// The struct that comes out is a starting point and not an output. Drop the
// columns the program has no use for, rename what reads badly, add the methods
// it needs. Nothing rewrites it, and a second run of the same command says the
// file is there rather than replacing it, unless -f says to.
//
// The flags are:
//
//	-type Name[,Name...]
//		the struct types to write handles for, one file each, or the one
//		struct to write when -from is given. Required.
//	-dir directory
//		the directory holding the package to read, and to write into. The
//		default is the working directory, which is where go generate runs
//		the tool.
//	-o file
//		the file to write, or - for standard output. The default is
//		<type>_kuma.go in the same directory, with the type name in snake
//		case, so Trade gives trade_kuma.go, and <type>.go under -from.
//		Only one type may be written to a named file.
//	-from file
//		a data file to take the columns from, instead of reading a struct
//		out of the package. A .parquet, .arrow, .arrows, .csv, .tsv or
//		.ndjson file.
//	-lines n
//		how many lines of a CSV, TSV or NDJSON file to look at before
//		deciding what the columns hold. The default is 1000. It says
//		nothing about Parquet or Arrow, which are read from the footer.
//	-package name
//		the package clause to write under -from. The default is the
//		package the directory already holds, or the name of the directory
//		when it holds none.
//	-f
//		replace the file -from writes when it is already there.
//
// The output is deterministic, so regenerating a package that has not changed
// rewrites the same bytes, and running kumagen in CI and looking for a diff is
// a way to catch a struct that was edited without the handles being written
// again.
package main

import (
	"errors"
	"flag"
	"fmt"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/tamnd/kuma"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "kumagen:", err)
		os.Exit(1)
	}
}

// usageText is the first half of what -h prints, the flags being the rest.
const usageText = `kumagen writes the column handles for a Go struct, and the
struct itself when it is given a data file to read the columns out of.

Usage:
	kumagen [-dir directory] [-o file] -type Name[,Name...]
	kumagen -from file [-dir directory] [-o file] [-lines n] [-package name] [-f] -type Name

Flags:
`

// run is main without the process, so that a test can call it and read what it
// wrote. Everything it prints on purpose goes to out, and an error is returned
// rather than printed.
func run(args []string, out io.Writer) error {
	fs := flag.NewFlagSet("kumagen", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	types := fs.String("type", "", "the struct types to write handles for, separated by commas")
	dir := fs.String("dir", ".", "the directory holding the package to read")
	file := fs.String("o", "", "the file to write, or - for standard output")
	from := fs.String("from", "", "a data file to write the struct for, instead of reading one")
	pkgName := fs.String("package", "", "the package the written struct belongs to")
	lines := fs.Int("lines", 1000, "how many lines of a text file to look at")
	force := fs.Bool("f", false, "overwrite the file -from writes when it is already there")

	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return printUsage(out, fs)
		}
		return err
	}

	names, err := typeNames(*types)
	if err != nil {
		return err
	}
	if *file != "" && *file != "-" && len(names) > 1 {
		return fmt.Errorf("%d types cannot go in the one file %s, "+
			"drop -o and each is written to its own", len(names), *file)
	}

	if *from != "" {
		return fromData(out, *from, names, *dir, *file, *pkgName, *lines, *force)
	}
	return handles(out, names, *dir, *file)
}

// fromData writes the struct for the columns of a data file, which is what
// -from asks for.
func fromData(out io.Writer, from string, names []string,
	dir, file, pkgName string, lines int, force bool) error {
	if len(names) > 1 {
		return fmt.Errorf("%d types cannot come out of the one file %s, "+
			"a data file holds one schema and so is one struct", len(names), from)
	}
	if lines < 1 {
		return fmt.Errorf("-lines is %d, and a schema cannot be read from "+
			"fewer than one line", lines)
	}

	src, err := structSource(from, names[0], dir, pkgName, lines)
	if err != nil {
		return err
	}
	return writeSource(out, structPath(dir, file, names[0]), src, force)
}

// handles writes the column handles for structs that are already written,
// which is what a go generate line asks for.
func handles(out io.Writer, names []string, dir, file string) error {
	pkg, err := loadPackage(dir)
	if err != nil {
		return err
	}

	for _, name := range names {
		src, err := generate(pkg, name)
		if err != nil {
			return err
		}
		if file == "-" {
			if _, err := out.Write(src); err != nil {
				return err
			}
			continue
		}
		path := file
		if path == "" {
			path = filepath.Join(dir, kuma.ColumnName(name)+"_kuma.go")
		}
		if err := os.WriteFile(path, src, 0o644); err != nil {
			return err
		}
	}
	return nil
}

// structSource returns the source of the file declaring the struct for the
// columns of a data file.
func structSource(from, name, dir, pkgName string, lines int) ([]byte, error) {
	schema, err := readSchema(from, lines)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", from, err)
	}

	fields, err := structFields(name, schema)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", from, err)
	}

	pkgName, err = packageOf(dir, pkgName)
	if err != nil {
		return nil, err
	}
	return renderStruct(pkgName, name, filepath.Base(from), fields)
}

// structPath is the file the struct goes in, which is the type in snake case
// unless -o named one.
func structPath(dir, file, name string) string {
	if file != "" {
		return file
	}
	return filepath.Join(dir, kuma.ColumnName(name)+".go")
}

// writeSource puts the source in the named file, or on out when the name is -.
//
// A file that is already there is left alone unless force says otherwise, since
// this one is written once and then edited, and a second run of the same
// command is nearly always a repeat rather than a request to throw the edits
// away.
func writeSource(out io.Writer, path string, src []byte, force bool) error {
	if path == "-" {
		_, err := out.Write(src)
		return err
	}

	flags := os.O_WRONLY | os.O_CREATE | os.O_EXCL
	if force {
		flags = os.O_WRONLY | os.O_CREATE | os.O_TRUNC
	}
	f, err := os.OpenFile(path, flags, 0o644)
	if err != nil {
		if errors.Is(err, os.ErrExist) {
			return fmt.Errorf("%s is already there, and it is a file to edit "+
				"rather than one to write again, so pass -f to replace it", path)
		}
		return err
	}
	if _, err := f.Write(src); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// packageOf works out the package clause for a file written into dir.
//
// The package the directory already holds is the answer when it holds one. An
// empty directory is named after itself, the way go mod init does it, and a
// directory whose name is not an identifier has to be told.
func packageOf(dir, given string) (string, error) {
	if given != "" {
		if !token.IsIdentifier(given) {
			return "", fmt.Errorf("%q is not a package name", given)
		}
		return given, nil
	}

	if p, err := loadPackage(dir); err == nil {
		for _, f := range p.files {
			if !strings.HasSuffix(f.Name.Name, "_test") {
				return f.Name.Name, nil
			}
		}
	}

	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}
	if base := filepath.Base(abs); token.IsIdentifier(base) {
		return base, nil
	}
	return "", fmt.Errorf("there is no Go file in %s to take a package name "+
		"from and the directory is not named like one, so pass -package", dir)
}

// printUsage writes what -h asks for, which is the text above and then the
// flags, since a flag set already knows how to describe itself.
func printUsage(out io.Writer, fs *flag.FlagSet) error {
	if _, err := fmt.Fprint(out, usageText); err != nil {
		return err
	}
	fs.SetOutput(out)
	fs.PrintDefaults()
	return nil
}

// typeNames splits the -type flag, which is one name or several separated by
// commas.
func typeNames(list string) ([]string, error) {
	if strings.TrimSpace(list) == "" {
		return nil, errors.New("no type to write handles for, pass -type Name")
	}

	names := strings.Split(list, ",")
	for i, name := range names {
		names[i] = strings.TrimSpace(name)
		if names[i] == "" {
			return nil, fmt.Errorf("%q has an empty name in it, "+
				"the types are separated by commas", list)
		}
	}
	return names, nil
}
