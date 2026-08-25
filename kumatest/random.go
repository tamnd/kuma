package kumatest

import (
	"fmt"
	"math/rand/v2"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// RandomOptions says what a random frame holds.
//
// The zero value is ten rows of four columns, one int64, one float64, one
// string and one bool, with no missing values. That is a frame to try something
// on rather than a frame to prove something about, which is what most of the
// calls to this want.
type RandomOptions struct {
	// Rows is how many rows to make. Zero means ten. It panics on a negative.
	Rows int

	// Types is the type of each column, and it says how many there are. Empty
	// means int64, float64, string and bool.
	//
	// The types that can be made are the ones an array can hold a value of
	// today: the booleans, the integers, the floats, strings and binary, the
	// dates, the times, timestamps and durations. A decimal, an interval or a
	// nested type panics, since a caller who asked for one wants that column
	// and not a substitute.
	Types []dtype.DataType

	// Names is what to call each column. A column with no name here is called
	// column_1, column_2 and so on, which is what the CSV reader calls a
	// column that arrived without a header. More names than types panics.
	Names []string

	// Nulls is the fraction of values that are missing, from zero to one. It
	// is zero by default. A number outside that range panics.
	//
	// It is worth setting for anything that will one day be handed a real
	// file. Most of the bugs in a column library live in the branch that runs
	// when a value is not there.
	Nulls float64

	// Seed says which frame you get. The same seed gives the same frame on
	// every run and on every machine.
	//
	// Zero is a seed like any other, so a test that says nothing about it
	// still gets the same data every time it runs. A test that fails once in
	// ten runs and cannot be run again is worse than no test at all.
	Seed uint64
}

// The defaults, and the numbers the values are drawn from.
//
// The ranges are chosen to be readable in a printed frame rather than to cover
// the width of the type. A column of numbers between plus and minus a million
// can be looked at, and one of numbers around nine quintillion cannot.
const (
	defaultRows  = 10
	numberSpread = 1_000_001
	floatSpread  = 1000
	epochDay2020 = 18262 // days from the Unix epoch to the first of January 2020
	daysToDraw   = 365 * 5
	secondsInDay = 86400
	daysInYear   = 365
)

// defaultTypes is what a frame holds when the caller did not say. It is one
// column of each of the four types most data is made of.
func defaultTypes() []dtype.DataType {
	return []dtype.DataType{dtype.Int64, dtype.Float64, dtype.String, dtype.Bool}
}

// Random returns a frame of random values, for a test that needs data rather
// than particular data. A nil *RandomOptions means the defaults.
//
// Everything about the frame comes from the options, so two calls with the same
// options return the same frame, on any machine and in any order. The columns
// are made one after another out of a single stream of random numbers, so
// changing the type of the first column changes the values in the second. That
// is worth knowing when a golden file is written against one of these.
//
// It panics rather than returning an error, since everything it can object to
// is something written in the test rather than something that happened while it
// ran.
func Random(o *RandomOptions) *kuma.Frame[kuma.Dynamic] {
	opts := o.withDefaults()
	r := rand.New(rand.NewPCG(opts.Seed, ^opts.Seed))

	cols := make([]kuma.Column, len(opts.Types))
	for i, dt := range opts.Types {
		if !canMake(dt) {
			panic(fmt.Sprintf("kumatest: Random cannot make a %s column yet", dt))
		}

		b := must(array.NewBuilder(dt))
		for range opts.Rows {
			if dt.Kind() == dtype.NullKind || r.Float64() < opts.Nulls {
				b.AppendNull()
				continue
			}
			appendRandom(b, dt, r)
		}
		data := must(array.NewChunked(dt, b.Finish()))
		cols[i] = must(kuma.NewColumn(opts.Names[i], data))
	}
	return must(kuma.NewFrame(cols...))
}

// withDefaults returns the options with the zero fields filled in and the ones
// that make no sense turned away.
func (o *RandomOptions) withDefaults() RandomOptions {
	var out RandomOptions
	if o != nil {
		out = *o
	}
	if out.Rows == 0 {
		out.Rows = defaultRows
	}
	if out.Rows < 0 {
		panic(fmt.Sprintf("kumatest: Random of %d rows", out.Rows))
	}
	if out.Nulls < 0 || out.Nulls > 1 {
		panic(fmt.Sprintf("kumatest: Random with %v of the values missing", out.Nulls))
	}
	if len(out.Types) == 0 {
		out.Types = defaultTypes()
	}
	if len(out.Names) > len(out.Types) {
		panic(fmt.Sprintf("kumatest: Random with %d names for %d columns",
			len(out.Names), len(out.Types)))
	}

	names := make([]string, len(out.Types))
	copy(names, out.Names)
	for i, n := range names {
		if n == "" {
			names[i] = fmt.Sprintf("column_%d", i+1)
		}
	}
	out.Names = names
	return out
}

// appendRandom puts one value on the end of a column being built.
func appendRandom(b *array.Builder, dt dtype.DataType, r *rand.Rand) {
	// The types that carry a parameter come first, since the parameter is what
	// says which numbers are a sensible value and a kind has had it taken off.
	switch t := dt.(type) {
	case dtype.Time32:
		b.Append(int32(r.Int64N(perDay(t.Unit))))
		return
	case dtype.Time64:
		b.Append(r.Int64N(perDay(t.Unit)))
		return
	case dtype.Timestamp:
		b.Append(instant(t.Unit, r))
		return
	case dtype.Duration:
		b.Append(r.Int64N(perDay(t.Unit)))
		return
	}

	switch dt.Kind() {
	case dtype.BoolKind:
		b.AppendBool(r.IntN(2) == 0)
	case dtype.Int8Kind:
		b.Append(int8(r.IntN(256) - 128))
	case dtype.Int16Kind:
		b.Append(int16(r.IntN(65536) - 32768))
	case dtype.Int32Kind:
		b.Append(int32(r.IntN(numberSpread) - numberSpread/2))
	case dtype.Int64Kind:
		b.Append(int64(r.IntN(numberSpread) - numberSpread/2))
	case dtype.Uint8Kind:
		b.Append(uint8(r.IntN(256)))
	case dtype.Uint16Kind:
		b.Append(uint16(r.IntN(65536)))
	case dtype.Uint32Kind:
		b.Append(uint32(r.IntN(numberSpread)))
	case dtype.Uint64Kind:
		b.Append(uint64(r.IntN(numberSpread)))
	case dtype.Float32Kind:
		b.Append(float32(hundredths(r)))
	case dtype.Float64Kind:
		b.Append(hundredths(r))
	case dtype.StringKind:
		b.AppendString(word(r))
	case dtype.BinaryKind:
		b.AppendBytes([]byte(word(r)))
	case dtype.Date32Kind:
		b.Append(int32(epochDay2020 + r.IntN(daysToDraw)))
	case dtype.Date64Kind:
		b.Append(int64(epochDay2020+r.IntN(daysToDraw)) * secondsInDay * 1000)
	default:
		// canMake has already turned away everything that does not have a case
		// here, so this is only reachable by adding a type to that list and
		// forgetting this switch.
		panic(fmt.Sprintf("kumatest: no random value for a %s column", dt))
	}
}

// hundredths returns a number between plus and minus floatSpread with two
// decimal places, which is what a price looks like and what a column of them
// reads as.
//
// It is drawn as a whole number and divided rather than scaled from a fraction,
// because a multiply followed by an add may be fused into one operation on a
// machine that has the instruction and not on one that does not, and the last
// bit of the result then depends on which machine ran the test. A single
// division is a single rounding everywhere.
func hundredths(r *rand.Rand) float64 {
	return float64(r.Int64N(2*floatSpread*100+1)-floatSpread*100) / 100
}

// word returns a run of letters, usually short enough to live inside a string
// view and now and then long enough not to.
//
// Both lengths matter. A string column stores a value of twelve bytes or fewer
// in the view itself and anything longer in a buffer beside it, and a test run
// over none of the second kind is a test that has not been near half the code.
func word(r *rand.Rand) string {
	n := 3 + r.IntN(6)
	if r.IntN(8) == 0 {
		n = 16 + r.IntN(24)
	}

	letters := make([]byte, n)
	for i := range letters {
		letters[i] = byte('a' + r.IntN(26))
	}
	return string(letters)
}

// instant returns a timestamp somewhere in 2020, in the unit asked for.
//
// It is a whole number of seconds in every unit, because a frame of timestamps
// is usually looked at rather than computed with, and a table of round times is
// easier to read than one of times to the nanosecond.
func instant(u dtype.TimeUnit, r *rand.Rand) int64 {
	scale := perSecond(u)
	return (epochDay2020*secondsInDay + int64(r.IntN(daysInYear*secondsInDay))) * scale
}

// perSecond is how many of a unit make a second.
func perSecond(u dtype.TimeUnit) int64 {
	switch u {
	case dtype.Millisecond:
		return 1e3
	case dtype.Microsecond:
		return 1e6
	case dtype.Nanosecond:
		return 1e9
	default:
		return 1
	}
}

// perDay is how many of a unit make a day, which is the range a time of day is
// drawn from and a span long enough to be an interesting duration.
func perDay(u dtype.TimeUnit) int64 { return secondsInDay * perSecond(u) }

// canMake reports whether Random knows how to make a column of this type.
func canMake(t dtype.DataType) bool {
	switch t.Kind() {
	case dtype.NullKind, dtype.BoolKind,
		dtype.Int8Kind, dtype.Int16Kind, dtype.Int32Kind, dtype.Int64Kind,
		dtype.Uint8Kind, dtype.Uint16Kind, dtype.Uint32Kind, dtype.Uint64Kind,
		dtype.Float32Kind, dtype.Float64Kind,
		dtype.StringKind, dtype.BinaryKind,
		dtype.Date32Kind, dtype.Date64Kind, dtype.Time32Kind, dtype.Time64Kind,
		dtype.TimestampKind, dtype.DurationKind:
		return true
	default:
		return false
	}
}

// must is for the errors that cannot happen here. Every column is built to the
// same length out of a type that has already been checked, so there is nothing
// left for a builder or a frame to object to, and an error from one of them is
// a bug in this package rather than in the test that called it.
func must[T any](v T, err error) T {
	if err != nil {
		panic("kumatest: " + err.Error())
	}
	return v
}
