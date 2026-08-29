package dataset

import (
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/tamnd/kuma/dtype"
)

// Discover walks the tree under root and returns what is in it.
//
// Nothing is opened and nothing is read. What comes back is the file list and
// what the paths said about each file, which is enough to narrow the list with
// [Dataset.Select] before any of it is read.
//
//	d, err := dataset.Discover("orders", nil)
//
// A nil options is the useful default: every file whose name does not start
// with a dot or an underscore, with the type of each partition column worked
// out from the values across the whole tree.
//
// The paths on the files are root joined to the rest, using the separator this
// operating system uses, so one can go straight to os.Open or to
// parquet.ReadFile.
func Discover(root string, opts *Options) (*Dataset, error) {
	d, err := walk(os.DirFS(root), opts)
	if err != nil {
		return nil, err
	}

	d.Root = root
	for i := range d.Files {
		d.Files[i].Path = filepath.Join(root, filepath.FromSlash(d.Files[i].Path))
	}
	return d, nil
}

// DiscoverFS is [Discover] over an [fs.FS] rather than over a directory.
//
// The paths on the files are the slash separated paths within fsys, which is
// what fsys.Open takes, so [ReadOptions.Open] gets a path it can use and this
// package still does not have to know where the bytes are. It is what a test
// uses with [testing/fstest.MapFS], and what a reader for a tree that is not on
// this machine uses.
func DiscoverFS(fsys fs.FS, opts *Options) (*Dataset, error) {
	return walk(fsys, opts)
}

// walk is the whole of the discovery, over whichever [fs.FS] the two entry
// points handed it.
func walk(fsys fs.FS, opts *Options) (*Dataset, error) {
	o := opts.withDefaults()

	var (
		d    Dataset
		keys []string
		cols [][]Value
	)
	err := fs.WalkDir(fsys, ".", func(p string, e fs.DirEntry, err error) error {
		switch {
		case err != nil:
			return err
		case p == ".":
			return nil
		case !o.Hidden && skipped(path.Base(p)):
			if e.IsDir() {
				return fs.SkipDir
			}
			return nil
		case e.IsDir():
			return nil
		case o.Extension != "" && path.Ext(p) != o.Extension:
			return nil
		}

		got, vals, err := parse(p)
		if err != nil {
			return fmt.Errorf("dataset: %s: %w", p, err)
		}
		if len(d.Files) == 0 {
			keys, cols = got, make([][]Value, len(got))
		} else if !slices.Equal(got, keys) {
			return fmt.Errorf("dataset: %s is partitioned by %s and %s is partitioned by %s: %w",
				p, list(got), d.Files[0].Path, list(keys), ErrLayout)
		}

		for i, v := range vals {
			cols[i] = append(cols[i], v)
		}
		d.Files = append(d.Files, File{Path: p, Values: vals})
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(d.Files) == 0 {
		return nil, fmt.Errorf("dataset: nothing to read: %w", ErrNoData)
	}

	if d.Schema, err = schema(keys, cols, &o); err != nil {
		return nil, err
	}
	return &d, nil
}

// schema works out the type of each partition column, which is what
// [Options.Types] said or what the values across the whole tree infer to.
func schema(keys []string, cols [][]Value, o *Options) (dtype.Schema, error) {
	for name := range o.Types {
		if !slices.Contains(keys, name) {
			return dtype.Schema{}, fmt.Errorf(
				"dataset: %q is not partitioned on: %w", name, ErrNoColumn)
		}
	}

	fields := make([]dtype.Field, len(keys))
	for i, name := range keys {
		dt, ok := o.Types[name]
		if !ok {
			dt = infer(cols[i])
		} else if !readable(dt) {
			return dtype.Schema{}, fmt.Errorf(
				"dataset: partition %q: cannot read a path into a %s column: %w",
				name, dt, ErrUnsupportedType)
		}

		null := slices.ContainsFunc(cols[i], func(v Value) bool { return v.Null })
		fields[i] = dtype.Field{Name: name, Type: dt, Nullable: null}
	}
	return dtype.Schema{Fields: fields}, nil
}

// skipped says whether a file or directory of this name is one of the things a
// writer leaves behind rather than one holding rows.
//
// _SUCCESS, _committed_ and _temporary come out of Spark and Hadoop, and the
// .crc files sit beside every part file. None of them is data and all of them
// are in the tree.
func skipped(name string) bool {
	return strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_")
}

// list renders a set of partition keys for a message, with nothing at all
// written as a phrase rather than as an empty pair of brackets.
func list(keys []string) string {
	if len(keys) == 0 {
		return "nothing"
	}
	return strings.Join(keys, "/")
}
