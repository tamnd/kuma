package kuma

import (
	"fmt"
	"path/filepath"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/csv"
	"github.com/tamnd/kuma/dataset"
	"github.com/tamnd/kuma/ndjson"
	"github.com/tamnd/kuma/parquet"
)

// ReadDataset reads a tree of partitioned files under root into one frame.
//
// A dataset is a directory whose subdirectories are named key=value, which is
// the layout Hive wrote and every engine since has read. The directory names
// are data, so a tree of orders under year=2024/month=03 comes back with a year
// column and a month column that are in no file.
//
//	f, err := kuma.ReadDataset("orders", nil)
//
// The format is chosen by the extension: .parquet, .csv, .tsv, .ndjson and
// .jsonl. A tree holding anything else is an error rather than a guess, and a
// tree holding two formats at once is one too, since the first file decides.
//
// This reads every file. Reading part of a tree is the reason the layout exists
// and it is two steps: [dataset.Discover] to find out what is there, which does
// not open anything, [dataset.Dataset.Select] to narrow it, and
// [ReadDatasetFiles] to read what is left.
func ReadDataset(root string, opts *dataset.Options) (*Frame[Dynamic], error) {
	d, err := dataset.Discover(root, opts)
	if err != nil {
		return nil, err
	}
	return ReadDatasetFiles(d)
}

// ReadDatasetFiles reads the files of a dataset that has already been found,
// which is how to read part of a tree.
//
//	d, err := dataset.Discover("orders", nil)
//	march := d.Select(func(f dataset.File) bool {
//		return d.Value(f, "month").Text == "03"
//	})
//	f, err := kuma.ReadDatasetFiles(march)
//
// Nothing outside the file list is opened, so a year narrowed to a month reads
// one twelfth of the tree. The format is chosen by the extension, the same as
// in [ReadDataset].
//
// Reading a format this does not know, or one it knows with options of your
// own, is [dataset.Read] with a [dataset.ReadOptions.Open] that calls whatever
// reader you like.
func ReadDatasetFiles(d *dataset.Dataset) (*Frame[Dynamic], error) {
	t, err := dataset.Read(d, &dataset.ReadOptions{Open: openByExtension})
	if err != nil {
		return nil, err
	}
	return frameOf(t)
}

// openByExtension reads one file with the reader its name calls for.
//
// The extension is all there is to go on. Sniffing the bytes would tell parquet
// from the rest, since it starts with a magic number, but CSV and NDJSON are
// both text and telling them apart means reading a line and guessing, which is
// a worse answer than an error. A caller who knows better passes their own
// [dataset.ReadOptions.Open].
func openByExtension(path string) (*array.Table, error) {
	switch ext := filepath.Ext(path); ext {
	case ".parquet":
		return parquet.ReadFile(path, nil)
	case ".csv":
		return csv.ReadFile(path, nil)
	case ".tsv":
		return csv.ReadFile(path, &csv.Options{Delimiter: '\t'})
	case ".ndjson", ".jsonl":
		return ndjson.ReadFile(path, nil)
	default:
		return nil, fmt.Errorf(
			"kuma: %s: no reader for a %q file, pass dataset.ReadOptions.Open", path, ext)
	}
}
