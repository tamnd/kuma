package array_test

import (
	"reflect"
	"strconv"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// This is the M0 exit criterion: every dtype round trips through construction,
// slice, append and read. One table, one case per type, and a check at the end
// that no kind was left out of it.

// roundTrip is one type and the three things a test has to know about it: how
// to put value i into a builder, how to read it back out of a column, and what
// value i is supposed to be.
type roundTrip struct {
	name   string
	dt     dtype.DataType
	append func(b *array.Builder, i int)
	value  func(a *array.Array, i int) any
	want   func(i int) any
}

func numeric[T array.Numeric](dt dtype.DataType, name string, at func(i int) T) roundTrip {
	return roundTrip{
		name:   name,
		dt:     dt,
		append: func(b *array.Builder, i int) { b.Append(at(i)) },
		value:  func(a *array.Array, i int) any { return a.Value[T](i) },
		want:   func(i int) any { return at(i) },
	}
}

// blob is a type whose values are bytes rather than numbers, meaning the
// strings, the fixed width binaries, the decimals and the intervals.
func blob(dt dtype.DataType, name string, at func(i int) []byte) roundTrip {
	return roundTrip{
		name:   name,
		dt:     dt,
		append: func(b *array.Builder, i int) { b.AppendBytes(at(i)) },
		value:  func(a *array.Array, i int) any { return a.Bytes(i) },
		want:   func(i int) any { return at(i) },
	}
}

// pad returns i as a value of w bytes, little endian, which is how a decimal
// and an interval are both laid out.
func pad(w, i int) []byte {
	p := make([]byte, w)
	for k := range min(w, 8) {
		p[k] = byte(i >> (8 * k))
	}
	return p
}

func roundTrips() []roundTrip {
	return []roundTrip{
		{
			name:   "null",
			dt:     dtype.Null,
			append: func(b *array.Builder, _ int) { b.AppendNull() },
		},
		{
			name:   "bool",
			dt:     dtype.Bool,
			append: func(b *array.Builder, i int) { b.AppendBool(i%3 == 0) },
			value:  func(a *array.Array, i int) any { return a.Bool(i) },
			want:   func(i int) any { return i%3 == 0 },
		},

		numeric(dtype.Int8, "int8", func(i int) int8 { return int8(i - 40) }),
		numeric(dtype.Int16, "int16", func(i int) int16 { return int16(i * 300) }),
		numeric(dtype.Int32, "int32", func(i int) int32 { return int32(i * 70000) }),
		numeric(dtype.Int64, "int64", func(i int) int64 { return int64(i) * 5e9 }),
		numeric(dtype.Uint8, "uint8", func(i int) uint8 { return uint8(i + 100) }),
		numeric(dtype.Uint16, "uint16", func(i int) uint16 { return uint16(i * 900) }),
		numeric(dtype.Uint32, "uint32", func(i int) uint32 { return uint32(i) * 4e6 }),
		numeric(dtype.Uint64, "uint64", func(i int) uint64 { return uint64(i) * 9e15 }),
		numeric(dtype.Float32, "float32", func(i int) float32 { return float32(i) / 8 }),
		numeric(dtype.Float64, "float64", func(i int) float64 { return float64(i) / 3 }),

		numeric(dtype.Date32, "date32", func(i int) int32 { return int32(20000 + i) }),
		numeric(dtype.Date64, "date64", func(i int) int64 { return int64(20000+i) * 86400000 }),
		numeric(dtype.Time32{Unit: dtype.Millisecond}, "time32", func(i int) int32 { return int32(i * 1000) }),
		numeric(dtype.Time64{Unit: dtype.Nanosecond}, "time64", func(i int) int64 { return int64(i) * 1e9 }),
		numeric(dtype.Timestamp{Unit: dtype.Microsecond, Zone: "UTC"}, "timestamp",
			func(i int) int64 { return 1700000000000000 + int64(i) }),
		numeric(dtype.Duration{Unit: dtype.Second}, "duration", func(i int) int64 { return int64(i) * 60 }),

		{
			name:   "string",
			dt:     dtype.String,
			append: func(b *array.Builder, i int) { b.AppendString(text(i)) },
			value:  func(a *array.Array, i int) any { return string(a.Bytes(i)) },
			want:   func(i int) any { return text(i) },
		},
		blob(dtype.Binary, "binary", func(i int) []byte { return []byte(text(i)) }),
		blob(dtype.FixedSizeBinary{ByteWidth: 5}, "fixed_size_binary", func(i int) []byte { return pad(5, i) }),

		blob(dtype.Decimal128{Precision: 18, Scale: 2}, "decimal128", func(i int) []byte { return pad(16, i) }),
		blob(dtype.Decimal256{Precision: 40, Scale: 4}, "decimal256", func(i int) []byte { return pad(32, i) }),

		blob(dtype.Interval{Unit: dtype.YearMonth}, "interval_year_month", func(i int) []byte { return pad(4, i) }),
		blob(dtype.Interval{Unit: dtype.DayTime}, "interval_day_time", func(i int) []byte { return pad(8, i) }),
		blob(dtype.Interval{Unit: dtype.MonthDayNano}, "interval_month_day_nano", func(i int) []byte { return pad(16, i) }),
	}
}

// text returns value i of a string column, long enough at every third value to
// be held outside its view rather than inside it.
func text(i int) string {
	if i%3 == 0 {
		return "a value too long to live inside its own view, number " + strconv.Itoa(i)
	}
	return strconv.Itoa(i)
}

const (
	roundTripLen  = 70
	roundTripNull = 7 // every seventh value is missing
)

func TestRoundTrip(t *testing.T) {
	for _, rt := range roundTrips() {
		t.Run(rt.name, func(t *testing.T) {
			b, err := array.NewBuilder(rt.dt)
			if err != nil {
				t.Fatalf("NewBuilder(%s): %v", rt.dt, err)
			}
			b.Grow(roundTripLen)
			for i := range roundTripLen {
				if i%roundTripNull == 0 {
					b.AppendNull()
					continue
				}
				rt.append(b, i)
			}

			a := b.Finish()
			if !dtype.Equal(a.DType(), rt.dt) {
				t.Fatalf("DType() = %s, want %s", a.DType(), rt.dt)
			}
			checkRoundTrip(t, rt, a, 0)

			// The same values through a slice, which reads them at an offset
			// into the buffers rather than from the start.
			for _, cut := range [][2]int{{0, roundTripLen}, {1, 2}, {9, 64}, {roundTripLen, roundTripLen}} {
				s := a.Slice(cut[0], cut[1])
				if s.Len() != cut[1]-cut[0] {
					t.Fatalf("Slice(%d, %d) has %d values", cut[0], cut[1], s.Len())
				}
				checkRoundTrip(t, rt, s, cut[0])
				checkRoundTrip(t, rt, s.Clone(), cut[0])
			}
		})
	}
}

// checkRoundTrip checks a against the model, where value i of a is value from+i
// of the column that was built.
func checkRoundTrip(t *testing.T, rt roundTrip, a *array.Array, from int) {
	t.Helper()

	nulls := 0
	for i := range a.Len() {
		want := (from + i) % roundTripNull
		if rt.dt.Kind() == dtype.NullKind {
			want = 0 // a null column is missing all the way along
		}
		if a.IsNull(i) != (want == 0) {
			t.Fatalf("from %d, IsNull(%d) = %v", from, i, a.IsNull(i))
		}
		if a.IsNull(i) {
			nulls++
			continue
		}
		if got := rt.value(a, i); !reflect.DeepEqual(got, rt.want(from+i)) {
			t.Fatalf("from %d, value %d is %v, want %v", from, i, got, rt.want(from+i))
		}
	}
	if a.NullCount() != nulls {
		t.Fatalf("from %d, NullCount() = %d, want %d", from, a.NullCount(), nulls)
	}
}

// TestRoundTripCoversEveryKind is what keeps the table above honest. A dtype
// added to kuma is either in it or in the list of kinds that are not stored in
// an Array yet, and adding one without deciding which fails here.
func TestRoundTripCoversEveryKind(t *testing.T) {
	notYet := map[dtype.Kind]string{
		dtype.LargeStringKind:   "converted to a string at the IPC boundary",
		dtype.LargeBinaryKind:   "converted to a binary at the IPC boundary",
		dtype.ListKind:          "nested, which is M8",
		dtype.LargeListKind:     "nested, which is M8",
		dtype.FixedSizeListKind: "nested, which is M8",
		dtype.StructKind:        "nested, which is M8",
		dtype.MapKind:           "nested, which is M8",
		dtype.DictionaryKind:    "dictionary encoding, which is M8",
	}

	covered := map[dtype.Kind]bool{}
	for _, rt := range roundTrips() {
		covered[rt.dt.Kind()] = true
	}

	for k := dtype.InvalidKind + 1; k <= dtype.DictionaryKind; k++ {
		if covered[k] {
			continue
		}
		if _, ok := notYet[k]; !ok {
			t.Errorf("the %s kind is neither in the round trip table nor in the list of kinds that are not stored yet", k)
		}
	}
}
