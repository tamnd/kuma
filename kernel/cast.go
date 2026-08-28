package kernel

import (
	"fmt"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// Cast returns a column holding the values of c in the type to.
//
// A value that will not fit is an error and the whole cast fails. Casting an
// int64 column to int8 when one row holds 400, or a string column to float64
// when one row holds "n/a", stops and says which row it was. That is the right
// default because the alternative silently changes data, and a column with one
// bad row in it is nearly always a mistake somewhere upstream rather than
// something to paper over.
//
// [TryCast] is the same cast with that one decision reversed.
//
// A cast between two types that mean nothing to each other is an error before
// any value is read, so a plan fails while it is being built rather than partway
// through the second file. [dtype.CanCast] is the same question asked on its
// own.
//
// The conversions are the obvious ones. A number becomes another number, a
// boolean, or its decimal text. A boolean becomes zero or one, or "true" and
// "false". Text is parsed into a number or a boolean. Bytes become text once
// they are checked for being valid UTF-8, and text becomes bytes for free. A
// temporal column and a number of the same width convert into each other by
// reinterpreting the stored count, which is what turns a timestamp into the
// microseconds it is made of and back.
//
// A float becomes an integer by throwing the fraction away, the way a Go
// conversion does. It is the value that has to fit, not the fraction, so 3.9
// becomes 3 and 300.0 does not become an int8. NaN and the infinities fit
// nowhere and are always the error case.
//
// Text is parsed as the destination and not as a number in general. Casting
// "3.9" to int64 fails rather than producing 3, because the caller who wrote
// int64 said what they expected the file to contain.
//
// A null in becomes a null out. A cast to the null type throws every value away,
// which is allowed because it takes saying so.
//
// A dictionary encoded column casts its dictionary and not its rows, so a
// million rows over two hundred and fifty country codes parse two hundred and
// fifty numbers. A cast to a plain type expands the result and a cast to another
// dictionary type keeps the encoding. A value in the dictionary that will not
// cast fails the column only when a row points at it, since a writer that
// carried a value over from another row group has not put anything wrong in the
// rows.
//
// Not yet: the calendar side of the temporal types, meaning a change of unit, a
// formatted date and a parsed one. The decimals and the intervals are not here
// either. Encoding a plain column as a dictionary is not here either, since
// finding the distinct values is the grouping machinery rather than a value at a
// time. Each of those is an error saying as much rather than a wrong answer.
func Cast(c *array.Chunked, to dtype.DataType) (*array.Chunked, error) {
	return cast(c, to, false)
}

// TryCast is [Cast] with a value that does not fit becoming a null.
//
// This is the cast to reach for when the data is known to be dirty and the plan
// is to count the nulls afterwards, which is a real thing to want when the
// alternative is a file of a million rows failing on the one that says "N/A".
// It is Polars with strict off, and SQL's TRY_CAST.
//
// It still fails on a pair of types that mean nothing to each other, since no
// per row answer would make that cast into a sensible one.
func TryCast(c *array.Chunked, to dtype.DataType) (*array.Chunked, error) {
	return cast(c, to, true)
}

// CastError says which row of a cast failed and why.
//
// The row is counted across the whole column rather than within a chunk,
// because chunk boundaries are an accident of how the data was read and nobody
// looking for the bad row in a file cares where they fell.
type CastError struct {
	Row   int            // which value, counted from the start of the column
	From  dtype.DataType // the type of the column
	To    dtype.DataType // the type asked for
	Value string         // the value that would not fit, printed
	Err   error          // what a parser said, or nil when nothing was parsed
}

func (e *CastError) Error() string {
	msg := fmt.Sprintf("kernel: cannot cast %s to %s: row %d is %s",
		e.From, e.To, e.Row, e.Value)
	if e.Err != nil {
		msg += ": " + e.Err.Error()
	}
	return msg
}

// Unwrap returns the parse error, so that errors.Is finds strconv.ErrRange
// through a cast that failed on a number that was too big for its type.
func (e *CastError) Unwrap() error { return e.Err }

// cast is the whole of Cast and TryCast. The loose flag is the one thing the
// two of them disagree about.
func cast(c *array.Chunked, to dtype.DataType, loose bool) (*array.Chunked, error) {
	if c == nil {
		panic("kernel: cast of a nil column")
	}
	if to == nil {
		panic("kernel: cast to a nil type")
	}

	from := c.DType()
	if dtype.Equal(from, to) {
		return c, nil
	}
	if !dtype.CanCast(from, to) {
		return nil, fmt.Errorf("kernel: cannot cast %s to %s, the two have nothing to say to each other", from, to)
	}

	// A dictionary encoded column casts its values rather than its rows, which
	// is a different loop and a different set of decisions about what a bad
	// value means.
	if d, ok := from.(dtype.Dictionary); ok {
		return castDictionary(c, d, to, loose)
	}
	conv, err := converter(from, to)
	if err != nil {
		return nil, err
	}

	// One chunk out per chunk in. A cast is a value at a time and changes
	// nothing about where the rows are, so the caller keeps whatever chunking
	// they had, and with it whatever the chunking was for.
	b, err := array.NewBuilder(to)
	if err != nil {
		return nil, fmt.Errorf("kernel: cannot cast to %s: %w", to, err)
	}

	row := 0
	chunks := make([]*array.Array, 0, c.NumChunks())
	for _, a := range c.Chunks() {
		b.Grow(a.Len())
		for i := range a.Len() {
			if a.IsNull(i) {
				b.AppendNull()
				row++
				continue
			}
			if err := conv(b, a, i); err != nil {
				if !loose {
					err.Row = row
					err.From, err.To = from, to
					return nil, err
				}
				b.AppendNull()
			}
			row++
		}
		chunks = append(chunks, b.Finish())
	}

	// The builder was made for this dtype, so what it built is of this dtype.
	return chunked(to, chunks...), nil
}

// converse is one value of a cast. It reads value i of a, which is known to be
// present, and appends the converted value to b.
//
// The error is a *CastError with only the parts this level knows filled in. The
// row and the two types are added by the caller, which is the level that knows
// them, so that a converter is a function of one value and nothing else.
type converse func(b *array.Builder, a *array.Array, i int) *CastError

// converter returns the function that converts one value, or an error if the
// pair is one this package has not learned yet.
//
// The pair has already been through dtype.CanCast and been called meaningful,
// so everything refused here is refused for being unwritten rather than for
// being nonsense.
func converter(from, to dtype.DataType) (converse, error) {
	if to.Kind() == dtype.NullKind || from.Kind() == dtype.NullKind {
		// Every value of the source is thrown away, or every value of the
		// result is missing. Either way nothing is read and nothing is
		// converted, and the null branch in the loop above never reaches here
		// for a Null source because a Null column has no valid values.
		return func(b *array.Builder, _ *array.Array, _ int) *CastError {
			b.AppendNull()
			return nil
		}, nil
	}

	if !known(from) || !known(to) || isCalendar(from, to) {
		return nil, notYet(from, to)
	}

	switch {
	case isStored(to):
		return toStored(from, to), nil
	case to.Kind() == dtype.BoolKind:
		return toBool(from), nil
	default:
		return toText(from, to), nil
	}
}

// known reports whether this package can read and write values of this type at
// all.
//
// The large offset string types are missing because an array never holds one,
// since they are converted at the Arrow boundary. The decimals, the intervals
// and the nested types are missing because the conversions are written and this
// one is not.
func known(t dtype.DataType) bool {
	switch t.Kind() {
	case dtype.NullKind, dtype.BoolKind,
		dtype.StringKind, dtype.BinaryKind, dtype.FixedSizeBinaryKind:
		return true
	default:
		return isStored(t)
	}
}

// isCalendar reports whether a cast between these two is a question about the
// calendar rather than about the stored count.
//
// A temporal type and a number of the same width are two readings of the same
// bytes, and converting between them is reinterpreting them. Anything else with
// a temporal on one side is arithmetic: a change of unit rescales, a cast to
// text formats a date, and a cast from text parses one. Those are worth doing
// properly and are not done here.
func isCalendar(from, to dtype.DataType) bool {
	if dtype.IsTemporal(from) && !dtype.IsNumeric(to) {
		return true
	}
	return dtype.IsTemporal(to) && !dtype.IsNumeric(from)
}

// isStored reports whether a type is one whose values this package can read and
// write as a plain number, which is every numeric type plus the temporal types
// that are a single count.
//
// The temporal types are in here because a cast to or from a number is a
// reinterpretation of the stored count and nothing else. What is not in here is
// the calendar, which is a change of unit, a formatted date or a parsed one.
func isStored(t dtype.DataType) bool {
	switch t.Kind() {
	case dtype.Int8Kind, dtype.Int16Kind, dtype.Int32Kind, dtype.Int64Kind,
		dtype.Uint8Kind, dtype.Uint16Kind, dtype.Uint32Kind, dtype.Uint64Kind,
		dtype.Float32Kind, dtype.Float64Kind,
		dtype.Date32Kind, dtype.Date64Kind, dtype.Time32Kind, dtype.Time64Kind,
		dtype.TimestampKind, dtype.DurationKind:
		return true
	default:
		return false
	}
}

// notYet is the error for a cast that is meaningful and unwritten.
func notYet(from, to dtype.DataType) error {
	return fmt.Errorf("kernel: casting %s to %s is not implemented yet", from, to)
}
