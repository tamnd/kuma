// Command crosscheck is a shared library that checks kuma against pyarrow
// across the Arrow C data interface.
//
// Two libraries in one process is the only way to test this interface for
// real. A file format can be checked against a file somebody else wrote, and an
// ABI cannot: the whole contract is about pointers into memory that only exists
// while both sides are running. So this is built into a shared library and
// crosscheck.py next to it loads it, hands pyarrow arrays to it and takes kuma
// arrays back.
//
// It runs in both directions. A kuma array goes out through the C structs,
// pyarrow imports it, and pyarrow sends it straight back, which this compares
// against the array it started as. A pyarrow array comes in, kuma imports it
// and exports it again without touching it, and pyarrow compares. Every case
// checks that the buffer addresses on both sides are the same addresses, since
// values that match prove the data was read correctly and addresses that match
// prove it was not copied.
//
// This is a main package under testdata because it is only ever built by
// TestPyarrow, with go build -buildmode=c-shared. The strings it returns are
// allocated with malloc and never freed, which is what a process that runs a
// few dozen cases and exits can afford.
package main

import "C"

import (
	"bytes"
	"fmt"
	"unsafe"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

func main() {}

// A case is one kuma array and the name pyarrow prints for its type. The name
// is the point of half of them: it says that the format string kuma writes is
// the type another implementation thinks it is, which nothing inside kuma can
// check.
type testCase struct {
	name  string
	kind  string
	array *array.Array
}

var cases = []testCase{
	{"null", "null", array.NewNull(5)},
	{"bool", "bool", array.OfBools(true, false, true, true, false)},
	{"int8", "int8", array.Of[int8](1, -2, 3)},
	{"int16", "int16", array.Of[int16](1, -2, 3)},
	{"int32", "int32", array.Of[int32](1, -2, 3)},
	{"int64", "int64", array.Of[int64](1, -2, 3)},
	{"uint8", "uint8", array.Of[uint8](1, 2, 3)},
	{"uint16", "uint16", array.Of[uint16](1, 2, 3)},
	{"uint32", "uint32", array.Of[uint32](1, 2, 3)},
	{"uint64", "uint64", array.Of[uint64](1, 2, 3)},
	{"float32", "float", array.Of[float32](1.5, -2.5)},
	{"float64", "double", array.Of(1.5, -2.5, 3.25)},
	{"string", "string_view", array.OfStrings("a", "bb", "a string too long to live inside a view")},
	{"inline string", "string_view", array.OfStrings("a", "bb", "ccc")},
	{"empty string", "string_view", array.OfStrings()},
	{"binary", "binary_view", binaryArray()},
	{"nulls", "int64", intsWithNulls()},
	{"string nulls", "string_view", stringsWithNulls()},
	{"sliced", "int64", array.Of[int64](10, 20, 30, 40, 50).Slice(1, 4)},
	{"timestamp", "timestamp[us, tz=UTC]", fixed(dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}, 3)},
	{"naive timestamp", "timestamp[ns]", fixed(dtype.Timestamp{Unit: dtype.Nanosecond}, 3)},
	{"date32", "date32[day]", fixed(dtype.Date32, 4)},
	{"date64", "date64[ms]", fixed(dtype.Date64, 2)},
	{"time32", "time32[ms]", fixed(dtype.Time32{Unit: dtype.Millisecond}, 3)},
	{"time64", "time64[us]", fixed(dtype.Time64{Unit: dtype.Microsecond}, 3)},
	{"duration", "duration[s]", fixed(dtype.Duration{Unit: dtype.Second}, 3)},
	{"interval", "month_day_nano_interval", fixed(dtype.Interval{Unit: dtype.MonthDayNano}, 2)},
	{"decimal128", "decimal128(18, 2)", fixed(dtype.Decimal128{Precision: 18, Scale: 2}, 2)},
	{"fixed size binary", "fixed_size_binary[4]", fixed(dtype.FixedSizeBinary{ByteWidth: 4}, 3)},
}

// live is what a pass through import is holding on to. The Python side releases
// it when it has finished with what came back out, since kuma's own buffers are
// the ones pyarrow lent it.
var live []*ipc.Imported

//export kuma_case_count
func kuma_case_count() C.int { return C.int(len(cases)) }

//export kuma_case_name
func kuma_case_name(i C.int) *C.char { return C.CString(cases[i].name) }

//export kuma_case_kind
func kuma_case_kind(i C.int) *C.char { return C.CString(cases[i].kind) }

// kuma_case_export fills the two structs with case i and returns an error
// string, or nothing at all when it worked.
//
//export kuma_case_export
func kuma_case_export(i C.int, schema, values unsafe.Pointer) *C.char {
	c := cases[i]
	field := dtype.Field{Name: c.name, Type: c.array.DType(), Nullable: true}
	if err := ipc.ExportField(field, (*ipc.CSchema)(schema)); err != nil {
		return C.CString(err.Error())
	}
	if err := ipc.ExportArray(c.array, (*ipc.CArray)(values)); err != nil {
		ipc.ReleaseSchema((*ipc.CSchema)(schema))
		return C.CString(err.Error())
	}
	return nil
}

// kuma_case_verify imports the structs and compares what arrived against the
// array case i started as. This is the end of the round trip that went out
// through pyarrow, so a difference here is a value that changed hands twice and
// came back wrong.
//
//export kuma_case_verify
func kuma_case_verify(i C.int, schema, values unsafe.Pointer) *C.char {
	imported, err := ipc.ImportArray((*ipc.CSchema)(schema), (*ipc.CArray)(values))
	if err != nil {
		return C.CString(err.Error())
	}
	defer imported.Release()

	// The field name is not checked. An array pyarrow exports carries the name
	// of its type rather than of a column, since an array on its own has no
	// column to be named after.
	if err := sameValues(imported.Array, cases[i].array); err != nil {
		return C.CString(err.Error())
	}
	return nil
}

// kuma_pass_through imports an array pyarrow exported and exports it straight
// back out, without copying or reading a value. What comes out the far side is
// pyarrow's own memory in kuma's idea of the layout, which for a column of text
// means views over the blocks it arrived in.
//
//export kuma_pass_through
func kuma_pass_through(schema, values, outSchema, outValues unsafe.Pointer) *C.char {
	imported, err := ipc.ImportArray((*ipc.CSchema)(schema), (*ipc.CArray)(values))
	if err != nil {
		return C.CString(err.Error())
	}
	live = append(live, imported)

	field := imported.Field
	field.Type = imported.Array.DType()
	if err := ipc.ExportField(field, (*ipc.CSchema)(outSchema)); err != nil {
		return C.CString(err.Error())
	}
	if err := ipc.ExportArray(imported.Array, (*ipc.CArray)(outValues)); err != nil {
		ipc.ReleaseSchema((*ipc.CSchema)(outSchema))
		return C.CString(err.Error())
	}
	return nil
}

// kuma_release_live hands back everything a pass through is holding.
//
//export kuma_release_live
func kuma_release_live() {
	for _, i := range live {
		i.Release()
	}
	live = nil
}

// sameValues compares two arrays value by value. It is what the test helpers in
// the ipc package do, written again here because a main package under testdata
// cannot import them.
func sameValues(got, want *array.Array) error {
	if !dtype.Equal(got.DType(), want.DType()) {
		return fmt.Errorf("type %s, want %s", got.DType(), want.DType())
	}
	if got.Len() != want.Len() {
		return fmt.Errorf("length %d, want %d", got.Len(), want.Len())
	}
	if got.NullCount() != want.NullCount() {
		return fmt.Errorf("%d nulls, want %d", got.NullCount(), want.NullCount())
	}
	for i := range want.Len() {
		if got.IsNull(i) != want.IsNull(i) {
			return fmt.Errorf("value %d is null %v, want %v", i, got.IsNull(i), want.IsNull(i))
		}
		if want.IsNull(i) {
			continue
		}
		if a, b := valueAt(got, i), valueAt(want, i); !bytes.Equal(a, b) {
			return fmt.Errorf("value %d is %x, want %x", i, a, b)
		}
	}
	return nil
}

// valueAt returns the bytes of one value, whatever the type means by them.
func valueAt(a *array.Array, i int) []byte {
	switch a.DType().Kind() {
	case dtype.BoolKind:
		if a.Bool(i) {
			return []byte{1}
		}
		return []byte{0}
	case dtype.StringKind, dtype.BinaryKind:
		return a.Bytes(i)
	default:
		bits, _ := dtype.Bits(a.DType())
		w := bits / 8
		k := (a.Offset() + i) * w
		return a.Buffer().Bytes()[k : k+w]
	}
}

func intsWithNulls() *array.Array {
	b, err := array.NewBuilder(dtype.Int64)
	if err != nil {
		panic(err)
	}
	for i := range 6 {
		if i%2 == 0 {
			b.AppendNull()
			continue
		}
		b.Append(int64(i))
	}
	return b.Finish()
}

func stringsWithNulls() *array.Array {
	b, err := array.NewBuilder(dtype.String)
	if err != nil {
		panic(err)
	}
	b.AppendNull()
	b.AppendString("a string too long to live inside a view")
	b.AppendString("short")
	b.AppendNull()
	return b.Finish()
}

func binaryArray() *array.Array {
	b, err := array.NewBuilder(dtype.Binary)
	if err != nil {
		panic(err)
	}
	b.AppendBytes([]byte{0, 1, 2})
	b.AppendBytes(bytes.Repeat([]byte{7}, 40))
	return b.Finish()
}

// fixed builds an array of a type whose values are bytes nobody here reads. The
// bytes are a counter, since what is being checked is that they arrive.
func fixed(t dtype.DataType, n int) *array.Array {
	bits, ok := dtype.Bits(t)
	if !ok {
		panic("no width for " + t.String())
	}
	buf := buffer.New(n * bits / 8)
	for i := range buf.Bytes() {
		buf.Bytes()[i] = byte(i + 1)
	}
	a, err := array.New(t, n, buf, nil)
	if err != nil {
		panic(err)
	}
	return a
}
