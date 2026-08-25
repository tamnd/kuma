package csv

import (
	"strconv"

	"github.com/tamnd/kuma/dtype"
)

// inferred is what a column looks like so far, as a small lattice that only
// ever moves in one direction: nothing, then a type, then string.
//
// It is not dtype.Coerce. Coercion between two column types is a strict
// question and refuses to combine an int64 and a float64, which is right when
// two real columns are being concatenated and the caller should say which one
// wins. Reading a file is the opposite situation. The values are text and
// nothing has been decided yet, so the job is to find the narrowest type that
// holds all of them, and there is always one, because string holds everything.
type inferred int

const (
	inferNothing inferred = iota // no value seen yet
	inferBool
	inferInt
	inferFloat
	inferString
)

// dtype returns the column type an inference ended at.
//
// A column with nothing in the sample is a string column. There is nothing to
// say what it holds, and string is the type that can take whatever turns up
// further down the file, which a null column could not.
func (i inferred) dtype() dtype.DataType {
	switch i {
	case inferBool:
		return dtype.Bool
	case inferInt:
		return dtype.Int64
	case inferFloat:
		return dtype.Float64
	default:
		return dtype.String
	}
}

// merge returns the type that holds both what has been seen and one more
// value.
//
// An integer among floats is a float, since every integer in the file has a
// float that reads back the same and the column has to be one type. Anything
// else that disagrees is a string.
func (i inferred) merge(next inferred) inferred {
	switch {
	case i == inferNothing:
		return next
	case next == inferNothing || i == next:
		return i
	case i == inferInt && next == inferFloat, i == inferFloat && next == inferInt:
		return inferFloat
	default:
		return inferString
	}
}

// inferValue returns the narrowest type that holds one field.
//
// The order is the point. An integer is tried first, so a column of ones and
// zeros is a column of numbers rather than a column of booleans, which is what
// [strconv.ParseBool] would have said. A float is tried next, which is what
// catches the values that are too large for an int64 as well as the ones with
// a point in them. Only the words true and false are a boolean, and everything
// left is a string.
func inferValue(s string) inferred {
	if _, err := strconv.ParseInt(s, 10, 64); err == nil {
		return inferInt
	}
	if _, err := strconv.ParseFloat(s, 64); err == nil {
		return inferFloat
	}
	if isBoolWord(s) {
		return inferBool
	}
	return inferString
}

// isBoolWord reports whether s is one of the words this reader will infer as a
// boolean.
//
// This is narrower than what a cast accepts on purpose. A cast is told the
// column is a boolean and t means true; inference has to guess, and a column
// of single letters is far more likely to be a code than a boolean.
//
// The list is exactly the spellings [strconv.ParseBool] takes, minus the ones
// that are a single letter or a digit. Any wider and inference would decide a
// column is a boolean and then the parse would refuse a value in it, which is
// the one way this can be wrong that the caller has no answer to.
func isBoolWord(s string) bool {
	switch s {
	case "true", "TRUE", "True", "false", "FALSE", "False":
		return true
	}
	return false
}

// infer works out the type of each of cols columns from the sample rows.
func infer(sample [][]string, cols int, opts *Options) []inferred {
	out := make([]inferred, cols)
	for _, row := range sample {
		for i, s := range row {
			if i >= cols {
				break
			}
			if out[i] == inferString {
				// Already at the bottom of the lattice, so there is nothing
				// left for this column to learn.
				continue
			}
			if opts.isNull(s) {
				continue
			}
			out[i] = out[i].merge(inferValue(s))
		}
	}
	return out
}
