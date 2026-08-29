// Package dataset reads a tree of files as one table.
//
// A dataset is a directory whose subdirectories are named key=value, which is
// the layout Hive wrote and every engine since has read. The directory names
// are data: a file under year=2024/month=03 holds the rows for that month and
// the file itself does not have a year column or a month column in it, because
// the path already said.
//
//	orders/
//	  year=2024/month=01/part-0.parquet
//	  year=2024/month=02/part-0.parquet
//	  year=2025/month=01/part-0.parquet
//
// [Discover] walks the tree and works out what is in it. [Read] turns that into
// one table, with the partition columns filled in from the paths. Between the
// two is where a dataset earns its keep: the partition values are known before
// a single file is opened, so a query for one month can throw away eleven
// twelfths of the tree without reading any of it.
//
// This package does not know about any file format. [ReadOptions.Open] reads one
// file, so parquet, CSV and NDJSON all work and so does anything else. That is
// also what keeps this package underneath kuma in the import graph, alongside
// the format packages rather than on top of them. kuma.ReadDataset is the
// convenient version that picks the reader by the file extension.
//
// # Partition values
//
// A path segment is a partition when it reads as key=value. The key is the
// column name and the value is the text in the path, percent decoded, since
// that is how a writer puts a space or a slash into a directory name.
//
// The type of a partition column is inferred from the values across the whole
// dataset, and the types it will infer are int64, float64 and string. A value
// is a number only when formatting the number back gives exactly the text that
// was in the path. That is a stricter rule than [strconv.ParseInt] and it is
// the rule that matters here, because a partition value has to name the
// directory it came from. Reading month=01 as the number 1 loses the file: 01
// and 1 are different directories and only one of them exists. So 01 is a
// string, and so are +1, 1.50, 1e3 and NaN, while 1, -1, 0 and 1.5 are numbers.
//
// Dates and timestamps are not inferred, the same as in the csv and ndjson
// readers, so a partition on a date reads as a string until the text to
// timestamp casts arrive in milestone 8.
//
// [DefaultPartition] is the directory name Hive and Spark write for a null, and
// it reads back as a missing value.
//
// # What is not a dataset
//
// Every directory between the root and a data file has to be a key=value
// segment, and every file has to have the same keys in the same order.
// Anything else is [ErrLayout] rather than a guess, because a tree that is
// partitioned two ways is two datasets and reading it as one would quietly put
// the rows of one under the columns of the other.
//
// Files whose name starts with a dot or an underscore are skipped. That is not
// a general rule about hidden files, it is that _SUCCESS and .part.crc are what
// the writers leave behind and neither of them holds rows.
//
// # The metadata
//
// A [Dataset] is the whole description of the tree: which files, which
// partition values, which types. Nothing is read until [Read] is called, and
// [Dataset.Select] narrows the file list before that happens.
//
// It is kept as a value rather than folded into the read for the sake of the
// lazy engine, which will hang it on a scan node and push a filter on a
// partition column into [Dataset.Select]. Nothing does that yet. What is here
// works on its own and is the same description that will be used then.
//
// Stability: tier 1, stable.
package dataset

import (
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// DefaultPartition is the directory name Hive and Spark write when a partition
// value is null, since a null has no text and every directory has a name.
//
// A partition value that is this string reads back as a missing value rather
// than as itself, so a column partitioned on a nullable key comes back with the
// nulls it went in with.
const DefaultPartition = "__HIVE_DEFAULT_PARTITION__"

// Options controls what [Discover] counts as part of the dataset. The zero
// Options is the useful default: every file that is not one of the writers'
// leftovers, with the partition types inferred from the values.
type Options struct {
	// Types names the type of a partition column instead of inferring it. A
	// name that is not a partition key is an error, since it is almost always a
	// typo.
	//
	// The types that can be named are bool, the signed and unsigned integers,
	// the floats and string. A declared column reads its values the permissive
	// way, so 01 is the number 1 and a column of dates can be declared a string
	// without waiting for inference to agree.
	Types map[string]dtype.DataType

	// Extension, if given, is the extension a file has to have to be counted,
	// written with the dot, as in ".parquet". It is for the tree that has a
	// manifest or a schema file sitting next to the data.
	Extension string

	// Hidden counts the files and directories whose name starts with a dot or
	// an underscore, which are skipped by default because they are what the
	// writers leave behind rather than rows.
	Hidden bool
}

// withDefaults returns the options with the zero values filled in, so the rest
// of the package can read the fields without asking what zero meant each time.
func (o *Options) withDefaults() Options {
	if o == nil {
		return Options{}
	}
	return *o
}

// ReadOptions controls how [Read] turns a dataset into a table.
type ReadOptions struct {
	// Open reads one file and returns its rows. It is called once per file, in
	// the order [Dataset.Files] has them, and it is given [File.Path] unchanged.
	//
	// This is the whole of what this package knows about file formats, which is
	// nothing. A dataset of parquet files passes a function that calls
	// parquet.ReadFile, and one of CSV files passes one that calls csv.ReadFile.
	//
	// Open is required. There is no default, because a default would be a guess
	// about the format from the extension, and this package would rather be told.
	Open func(path string) (*array.Table, error)

	// OmitPartitions leaves the partition columns out of the table and returns
	// only what the files hold.
	//
	// It is for the caller who wants the rows and already knows which partition
	// they came from, which is the case when reading one partition at a time.
	OmitPartitions bool
}

// Dataset is a tree of files and what their paths said about them.
//
// It is the whole description: nothing has been read when one of these comes
// back from [Discover], and reading it is [Read]. Narrowing it to the files
// worth reading is [Dataset.Select], and doing that before [Read] is the reason
// this format is worth using at all.
type Dataset struct {
	// Root is the directory the dataset was found under, as it was given to
	// [Discover]. It is empty for a dataset found with [DiscoverFS].
	Root string

	// Schema is the partition columns, in the order the path segments have
	// them, which is the order they are appended to a table in.
	//
	// It is empty for an unpartitioned tree, meaning one whose files sit
	// directly in the root. That is a dataset too and reading it is reading the
	// files.
	Schema dtype.Schema

	// Files is every file in the dataset, in the order the walk found them,
	// which is sorted by path.
	Files []File
}

// File is one file of a dataset and the partition values its path gave it.
type File struct {
	// Path is the path to hand to [ReadOptions.Open]. For a dataset found with
	// [Discover] it is the root joined to the rest and uses the separator this
	// operating system uses, so it can go straight to os.Open. For one found
	// with [DiscoverFS] it is the slash separated path within that [fs.FS].
	Path string

	// Values is the partition value for each field of [Dataset.Schema], in the
	// same order. It is nil for an unpartitioned dataset.
	Values []Value
}

// Value is one partition value, as the text that was in the path.
//
// It is text rather than the type of the column because the type is a property
// of the dataset and the text is a property of the directory. Keeping the text
// is what lets a caller rebuild the path, and what lets [Options.Types] change
// the type without walking the tree again.
type Value struct {
	// Text is the value from the path, percent decoded. It is empty when the
	// value is missing, which is not the same as a value that really is empty.
	Text string

	// Null says the directory was named [DefaultPartition], which is how a
	// writer says the value is missing.
	Null bool
}

// String returns the value as it would read in a message, which is the text, or
// the word null for a missing one.
func (v Value) String() string {
	if v.Null {
		return "null"
	}
	return v.Text
}

// Len returns the number of files in the dataset.
//
// There is no NumRows to go with it. How many rows a dataset holds is in the
// files and finding it out means opening them, and what a dataset knows without
// opening anything is what the paths said.
func (d *Dataset) Len() int { return len(d.Files) }

// Select returns the dataset holding only the files keep returned true for,
// sharing the schema with this one.
//
// This is partition pruning, and doing it before [Read] is the whole point of
// the layout. The values are in hand from the paths, so a year of orders can be
// narrowed to one month without opening a file.
//
//	march := d.Select(func(f dataset.File) bool {
//		return d.Value(f, "month").Text == "03"
//	})
//
// A dataset that keeps no files is not an error here. It is a table with no
// rows, and [Read] says so.
func (d *Dataset) Select(keep func(File) bool) *Dataset {
	out := &Dataset{Root: d.Root, Schema: d.Schema}
	for _, f := range d.Files {
		if keep(f) {
			out.Files = append(out.Files, f)
		}
	}
	return out
}

// index returns the position of the partition column called name, or -1.
func (d *Dataset) index(name string) int {
	for i, f := range d.Schema.Fields {
		if f.Name == name {
			return i
		}
	}
	return -1
}

// Value returns the file's value for the partition column called name.
//
// A name the dataset has not got returns the zero Value, which reads as a
// missing value. That is what makes [Dataset.Select] safe to write without
// checking first: a predicate on a column that is not there keeps nothing,
// which is what a filter on a column of nothing should do.
func (d *Dataset) Value(f File, name string) Value {
	i := d.index(name)
	if i < 0 || i >= len(f.Values) {
		return Value{Null: true}
	}
	return f.Values[i]
}
