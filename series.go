package kuma

import (
	"fmt"
	"time"
	"unsafe"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/bitmap"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// Series is one named column, read as a Go type.
//
// The values live in a chunked array underneath, which is where the memory and
// the nulls are. A Series is the typed view of that: it knows the column is
// stored as an int64 and that you want to read it as an int64, and it checked
// that once when it was made rather than on every value.
//
// A Series is immutable and cheap to copy. Slice, Head and Tail return a new
// one over the same memory.
//
// The zero Series is not usable. Use NewSeries or SeriesFrom, or take one out
// of a Frame.
type Series[T Value] struct {
	name string
	data *array.Chunked

	// read is how a value comes back out, decided by T alone, since the column
	// type was checked against T when the series was made.
	read readAs

	// nanos is how many nanoseconds one unit of the column is, which is what
	// turns a value into a time.Time. It is one for every column that is not a
	// timestamp, where it is not used at all.
	nanos int64
}

// readAs is the shape of a read, which is a property of T rather than of the
// column, because a column that could not be read as a T was rejected before
// the series existed.
type readAs uint8

const (
	readBool readAs = iota
	readInt8
	readInt16
	readInt32
	readInt64
	readUint8
	readUint16
	readUint32
	readUint64
	readFloat32
	readFloat64
	readString
	readTime
)

// NewSeries returns a series of the given values, with no nulls, of the column
// type that matches T.
//
// It is for tests, examples and small literal columns. Loading data goes
// through a reader, which builds the arrays and hands them over.
func NewSeries[T Value](name string, values ...T) Series[T] {
	dt := DTypeOf[T]()

	b, err := array.NewBuilder(dt)
	if err != nil {
		// DTypeOf returns a type every builder accepts.
		panic("kuma: " + err.Error())
	}
	b.Grow(len(values))
	appendValues(b, values)

	c, err := array.NewChunked(dt, b.Finish())
	if err != nil {
		panic("kuma: " + err.Error())
	}

	s, err := SeriesFrom[T](name, c)
	if err != nil {
		panic("kuma: " + err.Error())
	}
	return s
}

// SeriesFrom returns a series over data, read as a T.
//
// It reports an error unless the column can be read as a T, which is what
// CanRead answers. Nothing is copied.
func SeriesFrom[T Value](name string, data *array.Chunked) (Series[T], error) {
	if data == nil {
		return Series[T]{}, fmt.Errorf("kuma: no values for column %q: %w", name, ErrNoValues)
	}
	if !CanRead[T](data.DType()) {
		var zero T
		return Series[T]{}, fmt.Errorf("kuma: column %q is a %s column, which does not read as a %T: %w",
			name, data.DType(), zero, ErrWrongType)
	}

	s := Series[T]{name: name, data: data, read: readAsOf[T](), nanos: 1}
	if ts, ok := data.DType().(dtype.Timestamp); ok {
		s.nanos = nanosPerUnit(ts.Unit)
	}
	return s, nil
}

// Name returns the name of the column.
func (s Series[T]) Name() string { return s.name }

// DType returns the type the values are stored as, which is not always the type
// they are read as. A Series[int64] over a timestamp column reads int64 values
// out of a column whose DType is a timestamp.
func (s Series[T]) DType() dtype.DataType { return s.data.DType() }

// Len returns how many values the column holds.
func (s Series[T]) Len() int { return s.data.Len() }

// NullCount returns how many of them are missing.
func (s Series[T]) NullCount() int { return s.data.NullCount() }

// IsNull reports whether value i is missing. It panics if i is out of range.
func (s Series[T]) IsNull(i int) bool { return s.data.IsNull(i) }

// IsValid reports whether value i is present. It panics if i is out of range.
func (s Series[T]) IsValid(i int) bool { return s.data.IsValid(i) }

// Data returns the values underneath, which is the door out to array and to
// hand written kernels. It is a supported door rather than an accident.
func (s Series[T]) Data() *array.Chunked { return s.data }

// Rename returns the same column under a different name.
func (s Series[T]) Rename(name string) Series[T] {
	s.name = name
	return s
}

// Column returns the series as an untyped column, which is what a Frame holds.
func (s Series[T]) Column() Column {
	return Column{name: s.name, data: s.data}
}

// String returns a short description of the column, for a log line or a test
// failure. It is not the values.
func (s Series[T]) String() string {
	return fmt.Sprintf("kuma.Series[%s]{%q, len %d, nulls %d}",
		s.DType(), s.name, s.Len(), s.NullCount())
}

// Slice returns the values from i up to but not including j, as a series. It
// panics unless 0 <= i <= j <= Len.
//
// It shares the memory it came from and it is constant time, give or take the
// null count over the range.
func (s Series[T]) Slice(i, j int) Series[T] {
	s.data = s.data.Slice(i, j)
	return s
}

// Head returns the first n values, or all of them if there are fewer than n. A
// negative n means all but the last n.
func (s Series[T]) Head(n int) Series[T] { return s.Slice(0, headEnd(n, s.Len())) }

// Tail returns the last n values, or all of them if there are fewer than n. A
// negative n means all but the first n.
func (s Series[T]) Tail(n int) Series[T] { return s.Slice(tailStart(n, s.Len()), s.Len()) }

// Take returns the values at the given positions, in the order given.
//
// This is the operation everything that reorders rows is made of. A sort
// produces the positions and takes them, a join produces the positions and
// takes them, and this is where the values actually move.
//
// A position below zero gives a null, which is what an outer join does with a
// row that matched nothing. A position at or past the length panics, the same
// way indexing a slice does.
//
// Unlike Slice this copies, since the values it wants are scattered through the
// column and a scattering is not something a slice header can describe.
func (s Series[T]) Take(idx []int) Series[T] {
	checkPositions(idx, s.Len())
	s.data = kernel.Take(s.data, idx)
	return s
}

// Filter returns the values that mask selects, in the order they were in.
//
// A null in the mask selects nothing, since a row that nobody can say belongs
// in the result does not go in the result. It reports an error if the mask is
// not the same length as the column.
func (s Series[T]) Filter(mask Series[bool]) (Series[T], error) {
	if mask.Len() != s.Len() {
		return Series[T]{}, fmt.Errorf("kuma: a mask of %d values for a column of %d: %w",
			mask.Len(), s.Len(), ErrLength)
	}
	s.data = kernel.Filter(s.data, mask.data)
	return s, nil
}

// Cast returns the column in the type to, read as a U.
//
// The two type arguments are doing different jobs. The argument to is what the
// values are stored as, so it is the one that says microseconds or twelve
// bytes, and U is how they are read back, so a cast to a timestamp comes back
// as a Series[int64] or a Series[time.Time] depending on what the caller means
// to do next.
//
// A value that will not fit is an error naming the row it was in. TryCast is
// the same cast with that decision reversed, and [kernel.Cast] documents what
// converts into what.
func (s Series[T]) Cast[U Value](to dtype.DataType) (Series[U], error) {
	data, err := kernel.Cast(s.data, to)
	if err != nil {
		return Series[U]{}, err
	}
	return SeriesFrom[U](s.name, data)
}

// TryCast is Cast with a value that will not fit becoming a null.
func (s Series[T]) TryCast[U Value](to dtype.DataType) (Series[U], error) {
	data, err := kernel.TryCast(s.data, to)
	if err != nil {
		return Series[U]{}, err
	}
	return SeriesFrom[U](s.name, data)
}

// Value returns value i. It panics if i is out of range.
//
// A missing value reads as the zero value of T, which is why IsNull exists. The
// alternative is an ok return on every read, and a kernel that has already
// checked the null count does not want to pay for one.
//
// A string is the column's own bytes rather than a copy of them, which is what
// makes reading a string column allocation free. It is safe because a column is
// immutable, and it means a string kept from a large column keeps that column's
// memory alive, so copy it with strings.Clone if you are keeping a handful of
// values out of a large file.
func (s Series[T]) Value(i int) T {
	switch s.read {
	case readBool:
		return as[T](s.data.Bool(i))
	case readInt8:
		return as[T](s.data.Value[int8](i))
	case readInt16:
		return as[T](s.data.Value[int16](i))
	case readInt32:
		return as[T](s.data.Value[int32](i))
	case readInt64:
		return as[T](s.data.Value[int64](i))
	case readUint8:
		return as[T](s.data.Value[uint8](i))
	case readUint16:
		return as[T](s.data.Value[uint16](i))
	case readUint32:
		return as[T](s.data.Value[uint32](i))
	case readUint64:
		return as[T](s.data.Value[uint64](i))
	case readFloat32:
		return as[T](s.data.Value[float32](i))
	case readFloat64:
		return as[T](s.data.Value[float64](i))
	case readString:
		p := s.data.Bytes(i)
		return as[T](unsafe.String(unsafe.SliceData(p), len(p)))
	case readTime:
		return as[T](time.Unix(0, s.data.Value[int64](i)*s.nanos).UTC())
	default:
		panic("kuma: no way to read this column")
	}
}

// Values returns every value as one Go slice.
//
// For a column of numbers held in one chunk, which is what a column that has
// been through a filter or a select is, this is the memory itself: no copy, no
// allocation, and 64 byte aligned. A column in several chunks is copied into
// one slice, and so is a column of strings or of times, since neither of those
// is stored as a Go value.
//
// The result of the no copy case must not be modified. A Series is immutable
// and this is the one place that promise is left to the caller.
func (s Series[T]) Values() []T {
	chunks := s.data.Chunks()
	if s.read < readString && len(chunks) == 1 {
		return numbers[T](chunks[0])
	}

	out := make([]T, s.data.Len())
	if s.read < readString {
		// Numbers still come out a chunk at a time rather than a value at a
		// time, since each chunk is already a slice of exactly this type.
		n := 0
		for _, a := range chunks {
			n += copy(out[n:], numbers[T](a))
		}
		return out
	}
	for i := range out {
		out[i] = s.Value(i)
	}
	return out
}

// Validity returns the bitmap saying which values are present, or nil when none
// are missing.
//
// It is only meaningful for a column held in one chunk, and it reports whether
// that is the case. A column in several chunks has one bitmap per chunk, and
// the way to reach those is through Data.
func (s Series[T]) Validity() (*bitmap.Bitmap, bool) {
	chunks := s.data.Chunks()
	switch len(chunks) {
	case 0:
		return nil, true
	case 1:
		return chunks[0].Validity(), true
	default:
		return nil, false
	}
}

// numbers returns a's values as a slice of T, which is the same memory read
// under the Go type the series was asked for. The two have the same width,
// because CanRead said so before the series existed.
func numbers[T Value](a *array.Array) []T {
	var zero T
	switch any(zero).(type) {
	case int8:
		return reinterpret[T](a.Values[int8]())
	case int16:
		return reinterpret[T](a.Values[int16]())
	case int32:
		return reinterpret[T](a.Values[int32]())
	case int64:
		return reinterpret[T](a.Values[int64]())
	case uint8:
		return reinterpret[T](a.Values[uint8]())
	case uint16:
		return reinterpret[T](a.Values[uint16]())
	case uint32:
		return reinterpret[T](a.Values[uint32]())
	case uint64:
		return reinterpret[T](a.Values[uint64]())
	case float32:
		return reinterpret[T](a.Values[float32]())
	case float64:
		return reinterpret[T](a.Values[float64]())
	default:
		// A bool column is bits rather than values and never gets here, and
		// neither does a string or a time. Values is what decides.
		panic("kuma: not a column of numbers")
	}
}

// appendValues adds every value in vs to b, which is a builder for the column
// type that matches T.
func appendValues[T Value](b *array.Builder, vs []T) {
	for _, v := range vs {
		switch p := any(&v).(type) {
		case *bool:
			b.AppendBool(*p)
		case *int8:
			b.Append(*p)
		case *int16:
			b.Append(*p)
		case *int32:
			b.Append(*p)
		case *int64:
			b.Append(*p)
		case *uint8:
			b.Append(*p)
		case *uint16:
			b.Append(*p)
		case *uint32:
			b.Append(*p)
		case *uint64:
			b.Append(*p)
		case *float32:
			b.Append(*p)
		case *float64:
			b.Append(*p)
		case *string:
			b.AppendString(*p)
		case *time.Time:
			b.Append(p.UnixNano())
		default:
			panic("kuma: no way to append this Go type")
		}
	}
}

// as reinterprets v as a T, where the caller knows the two are the same type.
//
// Go has no way to say that in the type system, and the honest alternatives are
// worse: any(v).(T) allocates on every call for a value that does not fit in a
// word, which for a column read one value at a time is an allocation per row.
// Every caller here is inside a switch that has already established which type
// T is, so this is a move rather than a conversion.
func as[T Value, V Value](v V) T {
	return *(*T)(unsafe.Pointer(&v))
}

// reinterpret returns vs as a slice of T, which is the same memory under the
// same width. Same argument as as.
func reinterpret[T Value, V Value](vs []V) []T {
	return unsafe.Slice((*T)(unsafe.Pointer(unsafe.SliceData(vs))), len(vs))
}

// readAsOf returns how a value of type T comes out of a column.
func readAsOf[T Value]() readAs {
	var zero T
	switch any(zero).(type) {
	case bool:
		return readBool
	case int8:
		return readInt8
	case int16:
		return readInt16
	case int32:
		return readInt32
	case int64:
		return readInt64
	case uint8:
		return readUint8
	case uint16:
		return readUint16
	case uint32:
		return readUint32
	case uint64:
		return readUint64
	case float32:
		return readFloat32
	case float64:
		return readFloat64
	case string:
		return readString
	case time.Time:
		return readTime
	default:
		panic("kuma: no way to read this Go type")
	}
}

// nanosPerUnit returns how many nanoseconds one unit of a timestamp column is.
func nanosPerUnit(unit dtype.TimeUnit) int64 {
	switch unit {
	case dtype.Second:
		return int64(time.Second)
	case dtype.Millisecond:
		return int64(time.Millisecond)
	case dtype.Microsecond:
		return int64(time.Microsecond)
	case dtype.Nanosecond:
		return 1
	default:
		return 1
	}
}

// headEnd returns where the range Head asks for ends. A negative n counts from
// the end, which is what pandas and Polars both do and what a caller who wants
// "all but the last one" reaches for.
func headEnd(n, length int) int {
	if n < 0 {
		return max(0, length+n)
	}
	return min(n, length)
}

// checkPositions panics if any of the positions is past the end of a column of
// n values. A position below zero is not an error, it is the way to ask for a
// null.
//
// The kernel checks this too, and the check here is what makes the message name
// the frame rather than the column, and what makes a take out of a frame with
// no columns fail the same way as a take out of a frame with some.
func checkPositions(idx []int, n int) {
	for _, i := range idx {
		if i >= n {
			panic(fmt.Sprintf("kuma: Take position %d out of range, there are %d values", i, n))
		}
	}
}

// tailStart returns where the range Tail asks for begins.
func tailStart(n, length int) int {
	if n < 0 {
		return min(-n, length)
	}
	return max(0, length-n)
}
