package ndjson

import (
	"encoding/json/jsontext"

	"github.com/tamnd/kuma/dtype"
)

// inferred is what a column looks like so far, as a small lattice that only
// ever moves in one direction: nothing, then a type, then string.
//
// It is not dtype.Coerce. Coercion between two column types is a strict
// question and refuses to combine an int64 and a float64, which is right when
// two real columns are being concatenated and the caller should say which one
// wins. Reading a file is the opposite situation. Nothing has been decided yet,
// so the job is to find the narrowest type that holds every value that turned
// up, and there is always one, because string holds everything.
type inferred int

const (
	inferNothing inferred = iota // no value seen yet, or nothing but nulls
	inferBool
	inferInt
	inferFloat
	inferString
)

// dtype returns the column type an inference ended at.
//
// A column that was null on every line of the sample is a string column. There
// is nothing to say what it holds, and string is the type that can take whatever
// turns up further down the file, which a null column could not.
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

// merge returns the type that holds both what has been seen and one more value.
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

// inferValue reads the next value out of d and returns the narrowest type that
// holds it.
//
// Almost all of this is reading rather than guessing, which is what a JSON file
// gives that a delimited one cannot: a quoted value is a string and a bare true
// is a boolean, and neither is a question. The one thing left to decide is what
// kind of number a number is, and that is decided the way the CSV reader decides
// it, by trying an int64 first, so that a column of whole numbers is a column of
// integers and one with a point or an exponent anywhere in it is a column of
// floats.
//
// A number too large for a float64, which JSON allows and which nothing this
// package holds can carry, is a string. That is the same answer the CSV reader
// gives the same digits, and it keeps the promise inference is really making:
// the type it settles on is a type every value in the file goes into.
//
// A nested object or array is a string column holding the text it arrived as,
// which is what the package documentation says about them and all there is to do
// with them until list and struct columns arrive.
func inferValue(d *jsontext.Decoder) (inferred, error) {
	if k := d.PeekKind(); k == '{' || k == '[' {
		return inferString, d.SkipValue()
	}

	tok, err := d.ReadToken()
	if err != nil {
		return inferNothing, err
	}
	switch tok.Kind() {
	case 'n':
		return inferNothing, nil
	case 't', 'f':
		return inferBool, nil
	case '"':
		return inferString, nil
	default:
		if _, err := tok.Int(); err == nil {
			return inferInt, nil
		}
		if _, err := tok.Float(); err == nil {
			return inferFloat, nil
		}
		return inferString, nil
	}
}
