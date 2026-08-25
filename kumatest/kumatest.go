// Package kumatest compares frames in a test and prints what differs.
//
// A failing test has to answer two questions: what is wrong, and where. A test
// that prints both frames answers neither as soon as they are longer than a
// screen, so what these functions print is the cells that differ rather than
// the data around them.
//
//	kumatest.EqualFrames(t, got, want, nil)
//
// reports a difference like this:
//
//	frames differ in 2 of 4 rows
//
//	  row | column | got    | want
//	------+--------+--------+-------
//	    1 | price  | 150.25 | 150.5
//	    3 | symbol | null   | GOOG
//
// The values are written the way a printed frame writes them, so a timestamp
// is a date rather than an integer and a string that ends in a space is quoted.
//
// # What equal means
//
// Two frames are equal when they hold the same columns in the same order, with
// the same names and the same types, the same number of rows, and the same
// value in every cell. A missing value equals a missing value and equals
// nothing else, including the zero value of its column, which is the
// distinction the whole library is built to keep.
//
// The type is part of it. A column of int32 and a column of int64 holding the
// same numbers are not equal, because a function that returns the wrong width
// has a bug whether or not this frame happens to fit in both. A cast on the way
// in is how a test says it does not care.
//
// Two floating point numbers are equal when they are the same number, which is
// a comparison that fails a test over the last bit of a sum. [Options.Fraction]
// and [Options.Margin] are how much difference to allow, and the two together
// cover a value that is large and a value that is near zero. A NaN equals a NaN
// by default, since a column of them is an answer as much as any other, and
// [Options.NaNsDiffer] says otherwise.
//
// # Random data
//
// [Random] builds a frame of random values for a test that needs data rather
// than particular data, which is mostly a benchmark or a property test. The
// same options give the same frame on every run and on every machine, since a
// test that fails one run in ten and cannot be run again is worse than no test.
package kumatest

import "github.com/tamnd/kuma"

// TB is the part of [testing.TB] the comparisons use. A *testing.T, a
// *testing.B and a *testing.F all satisfy it.
//
// It is an interface here rather than testing.TB itself because testing.TB has
// an unexported method and so cannot be implemented anywhere else, and the
// tests for this package have to be able to read what a failure printed.
type TB interface {
	Helper()
	Errorf(format string, args ...any)
}

// Options says how strict a comparison is and how much of a difference to
// print. A nil *Options means the zero value, which is an exact comparison
// printing the first ten cells that differ.
type Options struct {
	// Fraction and Margin are how far apart two floating point numbers may be
	// and still count as equal. They are an allowance each, and a pair of
	// values is equal when it is within either: no further apart than Fraction
	// times the smaller of the two in magnitude, or no further apart than
	// Margin. Both are zero by default, which asks for the same number.
	//
	// Fraction is the one for a computed value, since the error in a sum grows
	// with the size of what went into it. Margin is the one for a value near
	// zero, where a relative allowance is almost no allowance at all. A test
	// over data that spans both wants both.
	Fraction float64
	Margin   float64

	// NaNsDiffer says that a NaN is not equal to a NaN, which is what
	// comparing two of them in Go says.
	//
	// The default is the other way round. A column of NaNs is an answer like
	// any other, and a test that computes one and compares it with the answer
	// it expected is asking whether the same thing came out rather than asking
	// about the bits.
	NaNsDiffer bool

	// MaxCells is how many differing cells to print. Zero means ten and a
	// negative number means all of them. The counts in the first line are of
	// everything that differs, so a report that shows ten still says how many
	// there were.
	MaxCells int

	// Print is how a value is rendered, and it is the options a printed frame
	// takes. It is worth setting for [kuma.PrintOptions.MaxWidth], when the
	// difference between two long strings is off the end of a normal cell.
	Print *kuma.PrintOptions
}

// defaultCells is how many differing cells a report shows when MaxCells is not
// set. It is ten because that is a screen, and because a difference that takes
// more than ten cells to see is one where the eleventh was not going to help.
const defaultCells = 10

// withDefaults returns the options with the zero fields filled in, so that the
// rest of the package can read them without asking whether they were set.
func (o *Options) withDefaults() Options {
	var out Options
	if o != nil {
		out = *o
	}
	if out.MaxCells == 0 {
		out.MaxCells = defaultCells
	}
	return out
}
