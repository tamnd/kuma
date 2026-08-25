// Kumagen writes the column handles for a Go struct, so that a query names a
// column with a field selector the compiler checks rather than with a string.
//
// Usage:
//
//	kumagen [-dir directory] [-o file] -type Name[,Name...]
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
// The flags are:
//
//	-type Name[,Name...]
//		the struct types to write handles for, one file each. Required.
//	-dir directory
//		the directory holding the package to read. The default is the
//		working directory, which is where go generate runs the tool.
//	-o file
//		the file to write, or - for standard output. The default is
//		<type>_kuma.go in the same directory, with the type name in snake
//		case, so Trade gives trade_kuma.go. Only one type may be written
//		to a named file.
//
// The output is deterministic, so regenerating a package that has not changed
// rewrites the same bytes, and running kumagen in CI and looking for a diff is
// a way to catch a struct that was edited without the handles being written
// again.
//
// This first version reads a Go struct. Reading the schema out of a data file,
// so that the struct itself comes from the data, is the next version.
package main

import (
	"errors"
	"flag"
	"fmt"
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
const usageText = `kumagen writes the column handles for a Go struct.

Usage:
	kumagen [-dir directory] [-o file] -type Name[,Name...]

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

	pkg, err := loadPackage(*dir)
	if err != nil {
		return err
	}

	for _, name := range names {
		src, err := generate(pkg, name)
		if err != nil {
			return err
		}
		if *file == "-" {
			if _, err := out.Write(src); err != nil {
				return err
			}
			continue
		}
		path := *file
		if path == "" {
			path = filepath.Join(*dir, kuma.ColumnName(name)+"_kuma.go")
		}
		if err := os.WriteFile(path, src, 0o644); err != nil {
			return err
		}
	}
	return nil
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
