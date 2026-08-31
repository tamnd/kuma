package plan

import (
	"fmt"
	"time"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// LiteralColumn builds the one value column that a value written in a query
// becomes.
//
// A column of one value is what the kernels take as the other side of an
// operation over a whole column, so there is no scalar path here to keep in
// step with the column path.
//
// The hint is the type of the column the literal is being used with, and it
// decides what the literal becomes. An integer literal takes the column's own
// integer type, so comparing a uint32 column against 0 does not turn it into an
// int64 column. What is refused is refused by [dtype.CoerceLiteral], which is
// where the rule about a float literal against an integer column lives.
//
// It is [LiteralTypeAgainst] with the value as well as the type, and the two
// answer the same question for the same reasons, which is why they are here
// together. The engine calls it to evaluate a literal and the optimizer calls
// it to work one out at plan time.
func LiteralColumn(v any, hint dtype.DataType) (*array.Chunked, error) {
	if t, ok := v.(time.Time); ok {
		// A time is the one value with no column type of its own to coerce, so
		// what it becomes is a rule rather than a coercion.
		return timeLiteral(t, TimeLiteralType(hint))
	}

	want, err := LiteralTypeAgainst(v, hint)
	if err != nil {
		return nil, err
	}

	dt, err := LiteralTypeAgainst(v, nil)
	if err != nil {
		return nil, err
	}

	c, err := litColumn(dt, v)
	if err != nil {
		return nil, err
	}
	if dtype.Equal(want, dt) {
		return c, nil
	}
	// The cast is the conversion and nothing more. A value that does not fit
	// the column has already been turned away by [LiteralTypeAgainst], which is
	// the check that used to be this one and now happens while the plan is
	// being built.
	return kernel.Cast(c, want)
}

// timeLiteral builds a wall clock time as a count in the column's own unit.
//
// This is a separate path because a time.Time is not a number and the cast
// kernel does not rescale a timestamp yet. Doing it here also means the literal
// lands in the column's unit exactly or not at all: a time with nanoseconds in
// it against a column of seconds is an error rather than a silent truncation,
// for the same reason 1.5 against an integer column is.
func timeLiteral(t time.Time, ts dtype.Timestamp) (*array.Chunked, error) {
	var count int64
	switch ts.Unit {
	case dtype.Second:
		count = t.Unix()
	case dtype.Millisecond:
		count = t.UnixMilli()
	case dtype.Microsecond:
		count = t.UnixMicro()
	default:
		count = t.UnixNano()
	}
	if !t.Equal(timeOfCount(count, ts.Unit)) {
		return nil, fmt.Errorf("kuma: the time %s does not fit a %s column exactly, truncate it first",
			t.Format(time.RFC3339Nano), ts)
	}
	return litColumn(ts, count)
}

// timeOfCount is timeLiteral's conversion the other way, which is how the
// conversion is checked for having lost anything.
func timeOfCount(count int64, unit dtype.TimeUnit) time.Time {
	switch unit {
	case dtype.Second:
		return time.Unix(count, 0)
	case dtype.Millisecond:
		return time.UnixMilli(count)
	case dtype.Microsecond:
		return time.UnixMicro(count)
	default:
		return time.Unix(0, count)
	}
}

// litColumn builds a column holding the single value v, which is what a literal
// written in a query becomes.
func litColumn(dt dtype.DataType, v any) (*array.Chunked, error) {
	b, err := array.NewBuilder(dt)
	if err != nil {
		return nil, fmt.Errorf("kuma: %w", err)
	}
	appendLiteral(b, v)

	c, err := array.NewChunked(dt, b.Finish())
	if err != nil {
		return nil, fmt.Errorf("kuma: %w", err)
	}
	return c, nil
}

// appendLiteral writes one value of whatever type it turns out to be.
//
// A type that is not one of these has already been refused by [LiteralType],
// which is asked first, or turned into a count by [timeLiteral], so getting
// here with one is a mistake in this file rather than something a caller can
// write.
func appendLiteral(b *array.Builder, v any) {
	switch v := v.(type) {
	case nil:
		b.AppendNull()
	case bool:
		b.AppendBool(v)
	case int:
		b.Append(int64(v))
	case int8:
		b.Append(v)
	case int16:
		b.Append(v)
	case int32:
		b.Append(v)
	case int64:
		b.Append(v)
	case uint:
		b.Append(uint64(v))
	case uint8:
		b.Append(v)
	case uint16:
		b.Append(v)
	case uint32:
		b.Append(v)
	case uint64:
		b.Append(v)
	case float32:
		b.Append(v)
	case float64:
		b.Append(v)
	case string:
		b.AppendString(v)
	case []byte:
		b.AppendBytes(v)
	default:
		panic(fmt.Sprintf("kuma: no way to write a literal of type %T", v))
	}
}
