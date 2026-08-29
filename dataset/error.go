package dataset

import "errors"

// The errors this package returns, all comparable with [errors.Is].
var (
	// ErrNoData is returned when there is no file to read at all, either
	// because the tree is empty or because every file in it was skipped.
	ErrNoData = errors.New("no data")

	// ErrLayout is returned when the tree is not one Hive dataset: a directory
	// above a data file that is not named key=value, or two files whose paths
	// name different partition keys.
	//
	// It is an error rather than a guess because a tree partitioned two ways is
	// two datasets, and reading it as one would put the rows of the one under
	// the columns of the other without saying so.
	ErrLayout = errors.New("not a Hive partitioned dataset")

	// ErrNoColumn is returned when [Options.Types] names something that is not
	// a partition key of this dataset.
	ErrNoColumn = errors.New("no such partition column")

	// ErrUnsupportedType is returned when [Options.Types] names a type a
	// partition value cannot be read into, such as a list or a struct.
	ErrUnsupportedType = errors.New("unsupported partition type")

	// ErrSchema is returned when the files do not fit together: two files whose
	// columns or types differ, or a file holding a column with the same name as
	// a partition column.
	ErrSchema = errors.New("the files do not share a schema")

	// ErrOpen is returned when [ReadOptions.Open] is nil, since there is no
	// default and no way to guess one.
	ErrOpen = errors.New("no reader for the files")

	// ErrValue is what a partition value that will not go into its column
	// unwraps to. The error itself is a [*ValueError], which says which file
	// and which column.
	ErrValue = errors.New("bad partition value")
)

// ValueError says that one partition value could not be read as the type of its
// column.
//
// It carries the path as well as the value, because a dataset is a great many
// files and knowing which directory is wrong is the difference between a fix
// and a search.
//
//	dataset: orders/year=2024/month=ab/part-0.parquet: cannot read partition month=ab as int64: invalid syntax
type ValueError struct {
	// Path is the file whose path held the value.
	Path string

	// Column is the partition key, which is the name of the column.
	Column string

	// Value is the text that was in the directory name.
	Value string

	// Type is the type the value was being read as.
	Type string

	// Err is what the parse said, usually [strconv.ErrSyntax] or
	// [strconv.ErrRange].
	Err error
}

// Error returns the message described on ValueError.
func (e *ValueError) Error() string {
	msg := "dataset: " + e.Path + ": cannot read partition " +
		e.Column + "=" + e.Value + " as " + e.Type
	if e.Err != nil {
		msg += ": " + e.Err.Error()
	}
	return msg
}

// Unwrap returns the errors this wraps, so that both errors.Is(err,
// dataset.ErrValue) and errors.Is(err, strconv.ErrSyntax) answer about the same
// error.
func (e *ValueError) Unwrap() []error {
	if e.Err == nil {
		return []error{ErrValue}
	}
	return []error{ErrValue, e.Err}
}

// FileError says that one file of the dataset could not be read, and which one.
//
// [ReadOptions.Open] returns whatever the format package it called returned, so
// this is a wrapper that puts the path in front of it rather than a new error
// about anything.
type FileError struct {
	// Path is the file, as it was handed to [ReadOptions.Open].
	Path string

	// Err is what reading it said.
	Err error
}

// Error returns the message described on FileError.
func (e *FileError) Error() string { return "dataset: " + e.Path + ": " + e.Err.Error() }

// Unwrap returns what reading the file said, so that asking about an error the
// format package defines works through this.
func (e *FileError) Unwrap() error { return e.Err }
