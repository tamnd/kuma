package kuma_test

import (
	"errors"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

func TestNewSeries(t *testing.T) {
	t.Run("int64", func(t *testing.T) {
		s := kuma.NewSeries("qty", int64(1), 2, 3)

		if s.Name() != "qty" {
			t.Errorf("Name() = %q, want %q", s.Name(), "qty")
		}
		if s.Len() != 3 || s.NullCount() != 0 {
			t.Errorf("%s, want 3 values and no nulls", s)
		}
		if !dtype.Equal(s.DType(), dtype.Int64) {
			t.Errorf("DType() = %s, want int64", s.DType())
		}
		for i, want := range []int64{1, 2, 3} {
			if got := s.Value(i); got != want {
				t.Errorf("Value(%d) = %d, want %d", i, got, want)
			}
			if !s.IsValid(i) || s.IsNull(i) {
				t.Errorf("value %d is missing, want it present", i)
			}
		}
	})

	t.Run("float64", func(t *testing.T) {
		s := kuma.NewSeries("price", 1.5, 2.25)
		if got := s.Values(); got[0] != 1.5 || got[1] != 2.25 {
			t.Errorf("Values() = %v", got)
		}
	})

	t.Run("bool", func(t *testing.T) {
		s := kuma.NewSeries("hit", true, false, true, true)
		for i, want := range []bool{true, false, true, true} {
			if got := s.Value(i); got != want {
				t.Errorf("Value(%d) = %v, want %v", i, got, want)
			}
		}
	})

	t.Run("string", func(t *testing.T) {
		s := kuma.NewSeries("symbol", "AAPL", "MSFT", "a name too long to sit inside a view")
		want := []string{"AAPL", "MSFT", "a name too long to sit inside a view"}
		for i, w := range want {
			if got := s.Value(i); got != w {
				t.Errorf("Value(%d) = %q, want %q", i, got, w)
			}
		}
		if got := s.Values(); len(got) != 3 || got[2] != want[2] {
			t.Errorf("Values() = %q", got)
		}
	})

	t.Run("time", func(t *testing.T) {
		first := time.Date(2026, 8, 25, 9, 30, 0, 0, time.UTC)
		s := kuma.NewSeries("ts", first, first.Add(time.Minute))

		if !dtype.Equal(s.DType(), dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "UTC"}) {
			t.Errorf("DType() = %s", s.DType())
		}
		if got := s.Value(0); !got.Equal(first) {
			t.Errorf("Value(0) = %s, want %s", got, first)
		}
		if got := s.Value(1); !got.Equal(first.Add(time.Minute)) {
			t.Errorf("Value(1) = %s", got)
		}
	})

	t.Run("empty", func(t *testing.T) {
		s := kuma.NewSeries[int64]("qty")
		if s.Len() != 0 {
			t.Errorf("%s, want no values", s)
		}
		if got := s.Values(); len(got) != 0 {
			t.Errorf("Values() = %v, want nothing", got)
		}
	})

	t.Run("every other numeric type", func(t *testing.T) {
		checkNumeric(t, int8(-1), 2)
		checkNumeric(t, int16(-300), 4)
		checkNumeric(t, int32(-70000), 8)
		checkNumeric(t, uint8(1), 200)
		checkNumeric(t, uint16(1), 40000)
		checkNumeric(t, uint32(1), 3000000000)
		checkNumeric(t, uint64(1), 1<<40)
		checkNumeric(t, float32(1.5), -2.5)
	})
}

// checkNumeric builds a two value column of whatever type it is given and reads
// it back, which is the whole of what a Series has to do for a number.
func checkNumeric[T kuma.Value](t *testing.T, a, b T) {
	t.Helper()

	s := kuma.NewSeries("v", a, b)
	if got := s.Value(0); got != a {
		t.Errorf("Value(0) = %v, want %v", got, a)
	}
	if got := s.Value(1); got != b {
		t.Errorf("Value(1) = %v, want %v", got, b)
	}
	if got := s.Values(); len(got) != 2 || got[0] != a || got[1] != b {
		t.Errorf("Values() = %v, want [%v %v]", got, a, b)
	}
}

// nullableInts returns a column of the given length holding 0, 1, 2 and so on
// with every third value missing, in one chunk or in several.
func nullableInts(t *testing.T, length int, chunks ...int) kuma.Series[int64] {
	t.Helper()
	if len(chunks) == 0 {
		chunks = []int{length}
	}

	var (
		arrays []*array.Array
		next   int64
	)
	for _, n := range chunks {
		b, err := array.NewBuilder(dtype.Int64)
		if err != nil {
			t.Fatalf("NewBuilder: %v", err)
		}
		for range n {
			if next%3 == 0 {
				b.AppendNull()
			} else {
				b.Append(next)
			}
			next++
		}
		arrays = append(arrays, b.Finish())
	}

	c, err := array.NewChunked(dtype.Int64, arrays...)
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	s, err := kuma.SeriesFrom[int64]("qty", c)
	if err != nil {
		t.Fatalf("SeriesFrom: %v", err)
	}
	return s
}

func TestSeriesNulls(t *testing.T) {
	s := nullableInts(t, 10)

	if s.NullCount() != 4 {
		t.Fatalf("NullCount() = %d, want 4", s.NullCount())
	}
	for i := range s.Len() {
		want := int64(i)
		if i%3 == 0 {
			want = 0
			if !s.IsNull(i) {
				t.Errorf("value %d is present, want it missing", i)
			}
		}
		if got := s.Value(i); got != want {
			t.Errorf("Value(%d) = %d, want %d", i, got, want)
		}
	}

	valid, ok := s.Validity()
	if !ok {
		t.Fatal("Validity() gave up on a column in one chunk")
	}
	if valid == nil {
		t.Fatal("Validity() = nil on a column with nulls")
	}
	for i := range s.Len() {
		if valid.Get(i) != s.IsValid(i) {
			t.Errorf("the bitmap and IsValid disagree at %d", i)
		}
	}
}

// TestSeriesValuesSharesMemory is the promise in document 04: the values of a
// column held in one chunk are the memory itself, not a copy of it.
func TestSeriesValuesSharesMemory(t *testing.T) {
	s := nullableInts(t, 64)

	raw := s.Data().Chunk(0).Values[int64]()
	got := s.Values()
	if len(got) != len(raw) {
		t.Fatalf("Values() has %d values, the chunk has %d", len(got), len(raw))
	}
	if &got[0] != &raw[0] {
		t.Error("Values() copied a column held in one chunk")
	}

	if n := testing.AllocsPerRun(100, func() { valuesSink = s.Values() }); n != 0 {
		t.Errorf("Values() allocated %v times, want none", n)
	}
}

func TestSeriesValuesAcrossChunks(t *testing.T) {
	s := nullableInts(t, 12, 5, 3, 4)

	if s.Data().NumChunks() != 3 {
		t.Fatalf("the column has %d chunks, want 3", s.Data().NumChunks())
	}
	got := s.Values()
	if len(got) != 12 {
		t.Fatalf("Values() has %d values, want 12", len(got))
	}
	for i, v := range got {
		want := int64(i)
		if i%3 == 0 {
			want = 0
		}
		if v != want {
			t.Errorf("Values()[%d] = %d, want %d", i, v, want)
		}
	}

	if _, ok := s.Validity(); ok {
		t.Error("Validity() answered for a column in three chunks, want it to say no")
	}
}

// TestSeriesValueAllocations is why Value reinterprets rather than asserting.
// Reading one value at a time is what a row oriented loop does, and it has to
// cost nothing beyond the read.
func TestSeriesValueAllocations(t *testing.T) {
	ints := kuma.NewSeries("qty", int64(1), 2, 3)
	strs := kuma.NewSeries("symbol", "AAPL", "a name too long to sit inside a view")
	times := kuma.NewSeries("ts", time.Now().UTC())

	tests := []struct {
		name string
		read func()
	}{
		{"int64", func() { intSink = ints.Value(1) }},
		{"string", func() { stringSink = strs.Value(1) }},
		{"time", func() { timeSink = times.Value(0) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if n := testing.AllocsPerRun(100, tt.read); n != 0 {
				t.Errorf("Value allocated %v times, want none", n)
			}
		})
	}
}

func TestSeriesSlice(t *testing.T) {
	s := nullableInts(t, 10)

	for i := range s.Len() + 1 {
		for j := i; j <= s.Len(); j++ {
			cut := s.Slice(i, j)
			if cut.Len() != j-i {
				t.Fatalf("Slice(%d, %d).Len() = %d, want %d", i, j, cut.Len(), j-i)
			}
			if cut.Name() != s.Name() {
				t.Fatalf("Slice(%d, %d) is called %q", i, j, cut.Name())
			}
			for k := range cut.Len() {
				if cut.IsNull(k) != s.IsNull(i+k) || cut.Value(k) != s.Value(i+k) {
					t.Fatalf("Slice(%d, %d) disagrees with the column at %d", i, j, k)
				}
			}
		}
	}
}

func TestSeriesHeadAndTail(t *testing.T) {
	s := kuma.NewSeries("qty", int64(0), 1, 2, 3, 4)

	tests := []struct {
		name string
		got  kuma.Series[int64]
		want []int64
	}{
		{"head 2", s.Head(2), []int64{0, 1}},
		{"head 0", s.Head(0), nil},
		{"head past the end", s.Head(99), []int64{0, 1, 2, 3, 4}},
		{"head all but the last two", s.Head(-2), []int64{0, 1, 2}},
		{"head all but more than there is", s.Head(-99), nil},
		{"tail 2", s.Tail(2), []int64{3, 4}},
		{"tail 0", s.Tail(0), nil},
		{"tail past the end", s.Tail(99), []int64{0, 1, 2, 3, 4}},
		{"tail all but the first two", s.Tail(-2), []int64{2, 3, 4}},
		{"tail all but more than there is", s.Tail(-99), nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.got.Values()
			if len(got) != len(tt.want) {
				t.Fatalf("got %v, want %v", got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("got %v, want %v", got, tt.want)
				}
			}
		})
	}
}

func TestSeriesRenameAndColumn(t *testing.T) {
	s := kuma.NewSeries("qty", int64(1), 2)

	other := s.Rename("quantity")
	if other.Name() != "quantity" {
		t.Errorf("Rename gave %q", other.Name())
	}
	if s.Name() != "qty" {
		t.Errorf("Rename changed the column it was called on, which is now %q", s.Name())
	}
	if other.Data() != s.Data() {
		t.Error("Rename copied the values")
	}

	c := s.Column()
	if c.Name() != "qty" || c.Data() != s.Data() {
		t.Errorf("Column() = %s", c)
	}
}

func TestSeriesTimeUnits(t *testing.T) {
	when := time.Date(2026, 8, 25, 9, 30, 0, 0, time.UTC)

	tests := []struct {
		name string
		unit dtype.TimeUnit
		div  int64
	}{
		{"seconds", dtype.Second, int64(time.Second)},
		{"milliseconds", dtype.Millisecond, int64(time.Millisecond)},
		{"microseconds", dtype.Microsecond, int64(time.Microsecond)},
		{"nanoseconds", dtype.Nanosecond, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dt := dtype.Timestamp{Unit: tt.unit, Zone: "UTC"}
			b, err := array.NewBuilder(dt)
			if err != nil {
				t.Fatalf("NewBuilder: %v", err)
			}
			b.Append(when.UnixNano() / tt.div)

			c, err := array.NewChunked(dt, b.Finish())
			if err != nil {
				t.Fatalf("NewChunked: %v", err)
			}

			// The same column reads as a time and as the int64 underneath it,
			// which is the point of CanRead being wider than DTypeOf.
			s, err := kuma.SeriesFrom[time.Time]("ts", c)
			if err != nil {
				t.Fatalf("SeriesFrom: %v", err)
			}
			if got := s.Value(0); !got.Equal(when) {
				t.Errorf("Value(0) = %s, want %s", got, when)
			}

			raw, err := kuma.SeriesFrom[int64]("ts", c)
			if err != nil {
				t.Fatalf("SeriesFrom[int64]: %v", err)
			}
			if got, want := raw.Value(0), when.UnixNano()/tt.div; got != want {
				t.Errorf("the int64 view has %d, want %d", got, want)
			}
		})
	}
}

func TestSeriesFromErrors(t *testing.T) {
	c, err := array.NewChunked(dtype.Float64, array.Of[float64](1.5))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	t.Run("the wrong type", func(t *testing.T) {
		_, err := kuma.SeriesFrom[int64]("price", c)
		if !errors.Is(err, kuma.ErrWrongType) {
			t.Fatalf("error is %v, want it to be ErrWrongType", err)
		}
		for _, want := range []string{"price", "float64", "int64"} {
			if !strings.Contains(err.Error(), want) {
				t.Errorf("error is %q, want it to mention %q", err, want)
			}
		}
	})

	t.Run("no values", func(t *testing.T) {
		_, err := kuma.SeriesFrom[int64]("qty", nil)
		if !errors.Is(err, kuma.ErrNoValues) {
			t.Fatalf("error is %v, want it to be ErrNoValues", err)
		}
	})
}

// TestSeriesValidityOfAnEmptyColumn covers the case where there is nothing to
// have a bitmap for, which is a column that has been sliced down to nothing.
func TestSeriesValidityOfAnEmptyColumn(t *testing.T) {
	s := nullableInts(t, 10).Slice(3, 3)

	valid, ok := s.Validity()
	if !ok || valid != nil {
		t.Errorf("Validity() = %v, %v, want nil and true", valid, ok)
	}
	if got := s.Values(); len(got) != 0 {
		t.Errorf("Values() = %v, want nothing", got)
	}
}

func TestSeriesPanics(t *testing.T) {
	s := kuma.NewSeries("qty", int64(1), 2, 3)

	tests := []struct {
		name string
		fn   func()
		want string
	}{
		{"index too high", func() { intSink = s.Value(3) }, "index out of range"},
		{"index negative", func() { intSink = s.Value(-1) }, "index out of range"},
		{"slice past the end", func() { s.Slice(0, 4) }, "of a column of 3 values"},
		{"slice backwards", func() { s.Slice(2, 1) }, "of a column of 3 values"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("did not panic")
				}
				msg, ok := r.(string)
				if !ok {
					t.Fatalf("panicked with %T, want a string", r)
				}
				if !strings.Contains(msg, tt.want) {
					t.Errorf("panicked with %q, want it to mention %q", msg, tt.want)
				}
			}()
			tt.fn()
		})
	}
}

func TestSeriesTake(t *testing.T) {
	s := nullableInts(t, 10, 4, 6)

	idx := []int{9, 0, 3, 3, -1, 1}
	got := s.Take(idx)
	if got.Len() != len(idx) {
		t.Fatalf("Take gave %d values, want %d", got.Len(), len(idx))
	}
	if got.Name() != s.Name() {
		t.Errorf("Take gave a column called %q, want %q", got.Name(), s.Name())
	}

	for k, i := range idx {
		null := i < 0 || s.IsNull(i)
		if got.IsNull(k) != null {
			t.Fatalf("value %d, taken from %d, is null %v, want %v", k, i, got.IsNull(k), null)
		}
		if !null && got.Value(k) != s.Value(i) {
			t.Errorf("value %d is %d, want %d", k, got.Value(k), s.Value(i))
		}
	}
	if s.Len() != 10 {
		t.Error("Take changed the column it was called on")
	}
}

func TestSeriesTakeStrings(t *testing.T) {
	s := kuma.NewSeries("symbol", "AAPL", "MSFT", "NVDA")

	got := s.Take([]int{2, 2, 0})
	if want := []string{"NVDA", "NVDA", "AAPL"}; !slices.Equal(got.Values(), want) {
		t.Errorf("Take gave %v, want %v", got.Values(), want)
	}
}

func TestSeriesFilter(t *testing.T) {
	s := kuma.NewSeries("qty", int64(10), 20, 30, 40)
	mask := kuma.NewSeries("keep", true, false, true, false)

	got, err := s.Filter(mask)
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if want := []int64{10, 30}; !slices.Equal(got.Values(), want) {
		t.Errorf("Filter gave %v, want %v", got.Values(), want)
	}
	if got.Name() != "qty" {
		t.Errorf("Filter gave a column called %q", got.Name())
	}

	if _, err := s.Filter(kuma.NewSeries("keep", true, false)); !errors.Is(err, kuma.ErrLength) {
		t.Errorf("a short mask gave %v, want an ErrLength", err)
	}
}

func TestSeriesTakePanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("Take past the end did not panic")
		}
	}()
	kuma.NewSeries("qty", int64(1), 2, 3).Take([]int{3})
}

func TestSeriesCast(t *testing.T) {
	s := kuma.NewSeries("qty", int32(1), int32(2), int32(3))

	got, err := s.Cast[int64](dtype.Int64)
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	if got.Name() != "qty" {
		t.Errorf("Cast gave a column called %q, want %q", got.Name(), "qty")
	}
	if want := []int64{1, 2, 3}; !slices.Equal(got.Values(), want) {
		t.Errorf("Cast gave %v, want %v", got.Values(), want)
	}
	if !dtype.Equal(got.DType(), dtype.Int64) {
		t.Errorf("Cast gave a %s column, want int64", got.DType())
	}
}

// TestSeriesCastReadAs is the reason Cast takes two types. One says what the
// values are stored as and the other says how they are read back, and a
// timestamp is the case where the two are genuinely different questions.
func TestSeriesCastReadAs(t *testing.T) {
	s := kuma.NewSeries("at", int64(1700000000), int64(1700000001))

	got, err := s.Cast[time.Time](dtype.Timestamp{Unit: dtype.Second})
	if err != nil {
		t.Fatalf("Cast: %v", err)
	}
	if want := time.Unix(1700000000, 0).UTC(); !got.Value(0).Equal(want) {
		t.Errorf("value 0 is %v, want %v", got.Value(0), want)
	}
}

func TestSeriesCastDoesNotFit(t *testing.T) {
	s := kuma.NewSeries("qty", int64(1), int64(400))

	if _, err := s.Cast[int8](dtype.Int8); err == nil {
		t.Fatal("Cast of 400 into an int8 succeeded")
	}

	got, err := s.TryCast[int8](dtype.Int8)
	if err != nil {
		t.Fatalf("TryCast: %v", err)
	}
	if got.NullCount() != 1 || !got.IsNull(1) {
		t.Errorf("TryCast gave %d nulls, want the second value missing", got.NullCount())
	}
}

// TestSeriesCastWrongReadType is the mistake the two type arguments make
// possible, and it has to be an error rather than a wrong answer.
func TestSeriesCastWrongReadType(t *testing.T) {
	s := kuma.NewSeries("qty", int32(1))

	if _, err := s.Cast[int8](dtype.Int64); err == nil {
		t.Fatal("reading an int64 column as int8 succeeded")
	}
	if _, err := s.TryCast[int8](dtype.Int64); err == nil {
		t.Fatal("reading an int64 column as int8 succeeded")
	}
}
