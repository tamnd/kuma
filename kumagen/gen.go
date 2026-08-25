package main

import (
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"

	"github.com/tamnd/kuma"
)

// pkg is a parsed Go package, which is all kumagen needs to know about the
// program it is writing a file into.
type pkg struct {
	dir   string
	fset  *token.FileSet
	files []*ast.File
}

// loadPackage parses every Go file in dir.
//
// The files are read in the order the directory lists them, which os.ReadDir
// sorts by name, so two runs over the same directory see the same thing.
//
// A file is parsed whatever its build tags say, since the only way to settle a
// build tag is to ask the go tool what this platform builds. A type
// declared twice under two tags is reported rather than picked between, which
// is the only place the difference shows.
func loadPackage(dir string) (*pkg, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	p := &pkg{dir: dir, fset: token.NewFileSet()}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		f, err := parser.ParseFile(p.fset, filepath.Join(dir, e.Name()), nil,
			parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		p.files = append(p.files, f)
	}
	if len(p.files) == 0 {
		return nil, fmt.Errorf("no Go file in %s to read a struct out of", dir)
	}
	return p, nil
}

// field is one field of a schema struct and the handle it turns into.
type field struct {
	name   string // the field, which is the name in the generated struct
	column string // the column it binds to
	kind   string // the handle family, the Str of StrCol and NewStrCol
}

// generate returns the source of the file holding the handles for the struct
// called name.
func generate(p *pkg, name string) ([]byte, error) {
	file, st, err := p.findStruct(name)
	if err != nil {
		return nil, err
	}

	pkgName := file.Name.Name
	if strings.HasSuffix(pkgName, "_test") {
		return nil, fmt.Errorf("%s is declared in %s, which is a test only package "+
			"that an ordinary file cannot be part of", name, pkgName)
	}

	fields, err := fieldsOf(file, name, st)
	if err != nil {
		return nil, err
	}
	if len(fields) == 0 {
		return nil, fmt.Errorf("%s has no field that names a column, "+
			"an unexported field and a field tagged \"-\" are both left out", name)
	}
	return render(pkgName, name, fields)
}

// findStruct returns the file declaring the struct type called name, and the
// struct itself.
func (p *pkg) findStruct(name string) (*ast.File, *ast.StructType, error) {
	var (
		file *ast.File
		spec *ast.TypeSpec
	)
	for _, f := range p.files {
		for _, d := range f.Decls {
			gd, ok := d.(*ast.GenDecl)
			if !ok || gd.Tok != token.TYPE {
				continue
			}
			for _, s := range gd.Specs {
				ts, ok := s.(*ast.TypeSpec)
				if !ok || ts.Name.Name != name {
					continue
				}
				if spec != nil {
					return nil, nil, fmt.Errorf("%s is declared twice, in %s and in %s",
						name, p.filename(file), p.filename(f))
				}
				file, spec = f, ts
			}
		}
	}

	switch {
	case spec == nil:
		return nil, nil, fmt.Errorf("no type called %s in %s", name, p.dir)
	case spec.TypeParams != nil:
		return nil, nil, fmt.Errorf("%s has type parameters, and a schema is a "+
			"plain struct with a field for each column", name)
	}
	st, ok := spec.Type.(*ast.StructType)
	if !ok {
		return nil, nil, fmt.Errorf("%s is a %s rather than a struct",
			name, types.ExprString(spec.Type))
	}
	return file, st, nil
}

// filename returns the name of the file f was parsed from, for an error that
// has to say where something is.
func (p *pkg) filename(f *ast.File) string {
	return filepath.Base(p.fset.Position(f.Package).Filename)
}

// fieldsOf returns the handles the fields of st turn into, in the order they
// are written.
//
// The rules are the ones [kuma.Bind] follows, because a handle that named a
// different column than the bind checked would be worse than no handle at all.
// The column is the kuma tag or the field name in snake case, a field tagged
// "-" is left out, and so is an unexported one.
func fieldsOf(file *ast.File, typeName string, st *ast.StructType) ([]field, error) {
	imports := importsOf(file)

	var out []field
	for _, f := range st.Fields.List {
		names, err := fieldNames(f)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", typeName, err)
		}
		for _, name := range names {
			if !ast.IsExported(name) {
				continue
			}
			column := tagName(f.Tag)
			if column == "-" {
				continue
			}
			if column == "" {
				column = kuma.ColumnName(name)
			}

			kind, ok := handleKind(f.Type, imports)
			if !ok {
				return nil, fmt.Errorf("field %s of %s has type %s, and there is a "+
					"handle for a string, a float64, an int64, a bool and a time.Time",
					name, typeName, types.ExprString(f.Type))
			}
			out = append(out, field{name: name, column: column, kind: kind})
		}
	}
	return out, nil
}

// fieldNames returns the names a field declares, which is more than one for
// fields written together and the name of the type for an embedded one.
func fieldNames(f *ast.Field) ([]string, error) {
	if len(f.Names) > 0 {
		names := make([]string, len(f.Names))
		for i, n := range f.Names {
			names[i] = n.Name
		}
		return names, nil
	}

	// An embedded field is named after its type, which is what reflect calls it
	// and so what a bind looks for. The name is the type on its own, without the
	// package it came from, the pointer it may be behind or the type arguments
	// it may have been given.
	switch t := f.Type.(type) {
	case *ast.Ident:
		return []string{t.Name}, nil
	case *ast.SelectorExpr:
		return []string{t.Sel.Name}, nil
	case *ast.StarExpr:
		return fieldNames(&ast.Field{Type: t.X})
	case *ast.IndexExpr:
		return fieldNames(&ast.Field{Type: t.X})
	case *ast.IndexListExpr:
		return fieldNames(&ast.Field{Type: t.X})
	}
	return nil, fmt.Errorf("the embedded %s has no field name to use",
		types.ExprString(f.Type))
}

// tagName returns the kuma tag of a field, or the empty string when it has no
// tag or the tag says nothing about kuma.
func tagName(tag *ast.BasicLit) string {
	if tag == nil {
		return ""
	}
	// The parser has already refused a tag that is not a string literal, so a tag
	// that will not unquote has been built by hand rather than read out of a file.
	value, err := strconv.Unquote(tag.Value)
	if err != nil {
		return ""
	}
	// reflect.StructTag rather than a hand written split, so that the tag is
	// read exactly the way the bind reads it.
	return reflect.StructTag(value).Get("kuma")
}

// handleKind returns the family of handle a field of the given type binds to,
// and whether there is one.
//
// A field of another width, an int32 or a float32, binds at run time but has no
// handle yet, and reporting that is better than writing a file that does not
// compile.
func handleKind(expr ast.Expr, imports map[string]string) (string, bool) {
	switch t := expr.(type) {
	case *ast.Ident:
		switch t.Name {
		case "string":
			return "Str", true
		case "float64":
			return "F64", true
		case "int64":
			return "I64", true
		case "bool":
			return "Bool", true
		}
	case *ast.SelectorExpr:
		x, ok := t.X.(*ast.Ident)
		if ok && t.Sel.Name == "Time" && imports[x.Name] == "time" {
			return "Time", true
		}
	}
	return "", false
}

// importsOf returns what each name a file's imports bind stands for, so that
// the time of time.Time is known to be the time package rather than a variable
// or another package imported under that name.
func importsOf(f *ast.File) map[string]string {
	out := make(map[string]string, len(f.Imports))
	for _, imp := range f.Imports {
		// The path of an import is a string literal the parser has read, the same
		// as a tag, so one that will not unquote did not come from a file.
		path, err := strconv.Unquote(imp.Path.Value)
		if err != nil {
			continue
		}
		name := path[strings.LastIndex(path, "/")+1:]
		if imp.Name != nil {
			name = imp.Name.Name
		}
		out[name] = path
	}
	return out
}

// render writes the file out.
//
// The result goes through [format.Source], so the struct and the literal come
// back aligned and a change here cannot produce a file that gofmt would want to
// rewrite.
func render(pkgName, typeName string, fields []field) ([]byte, error) {
	var sb strings.Builder
	sb.WriteString("// Code generated by kumagen. DO NOT EDIT.\n\n")
	fmt.Fprintf(&sb, "package %s\n\n", pkgName)
	sb.WriteString("import \"github.com/tamnd/kuma\"\n\n")

	fmt.Fprintf(&sb, "// %sCols is the set of column handles for [%s].\n", typeName, typeName)
	sb.WriteString("//\n")
	sb.WriteString("// A handle carries the name of a column and the type it holds, so a query\n")
	sb.WriteString("// written with one is checked by the compiler rather than when it runs.\n")

	fmt.Fprintf(&sb, "var %sCols = struct {\n", typeName)
	for _, f := range fields {
		fmt.Fprintf(&sb, "%s kuma.%sCol[%s]\n", f.name, f.kind, typeName)
	}
	sb.WriteString("}{\n")
	for _, f := range fields {
		fmt.Fprintf(&sb, "%s: kuma.New%sCol[%s](%q),\n", f.name, f.kind, typeName, f.column)
	}
	sb.WriteString("}\n")

	src, err := format.Source([]byte(sb.String()))
	if err != nil {
		// Everything written above is a name the parser has already accepted, so
		// a file that does not parse is a mistake in this function.
		return nil, fmt.Errorf("kumagen wrote a file that does not parse, "+
			"which is a bug in kumagen: %w", err)
	}
	return src, nil
}
