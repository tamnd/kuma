package kumatest_test

import (
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kumatest"
)

// randomTypes is every type [kumatest.Random] can make, one of each and one per
// unit for the ones that take a unit.
func randomTypes() []dtype.DataType {
	return []dtype.DataType{
		dtype.Null, dtype.Bool,
		dtype.Int8, dtype.Int16, dtype.Int32, dtype.Int64,
		dtype.Uint8, dtype.Uint16, dtype.Uint32, dtype.Uint64,
		dtype.Float32, dtype.Float64,
		dtype.String, dtype.Binary,
		dtype.Date32, dtype.Date64,
		dtype.Time32{Unit: dtype.Second},
		dtype.Time32{Unit: dtype.Millisecond},
		dtype.Time64{Unit: dtype.Microsecond},
		dtype.Time64{Unit: dtype.Nanosecond},
		dtype.Timestamp{Unit: dtype.Second},
		dtype.Timestamp{Unit: dtype.Millisecond},
		dtype.Timestamp{Unit: dtype.Microsecond},
		dtype.Timestamp{Unit: dtype.Nanosecond},
		dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "Europe/London"},
		dtype.Duration{Unit: dtype.Second},
		dtype.Duration{Unit: dtype.Nanosecond},
	}
}

func TestRandomWithNoOptions(t *testing.T) {
	f := kumatest.Random(nil)

	if rows, cols := f.Shape(); rows != 10 || cols != 4 {
		t.Errorf("the default frame is %d rows x %d cols, want 10 x 4", rows, cols)
	}
	want := []string{"column_1", "column_2", "column_3", "column_4"}
	if names := f.Names(); !slices.Equal(names, want) {
		t.Errorf("the default columns are %v, want %v", names, want)
	}
	types := []dtype.DataType{dtype.Int64, dtype.Float64, dtype.String, dtype.Bool}
	for i, dt := range types {
		if got := f.ColumnAt(i).DType(); !dtype.Equal(got, dt) {
			t.Errorf("column %d is %s, want %s", i, got, dt)
		}
		if n := f.ColumnAt(i).Data().NullCount(); n != 0 {
			t.Errorf("column %d holds %d missing values, and none were asked for", i, n)
		}
	}
}

// TestRandomIsTheSameFrameEveryTime is the property the whole thing rests on. A
// benchmark that runs over different data on every run is comparing numbers
// that were never comparable.
func TestRandomIsTheSameFrameEveryTime(t *testing.T) {
	opts := &kumatest.RandomOptions{Rows: 100, Nulls: 0.2, Seed: 7}

	first := kumatest.Random(opts)
	second := kumatest.Random(opts)
	if d := kumatest.DiffFrames(first, second, nil); d != "" {
		t.Errorf("two calls with the same options gave different frames:\n%s", d)
	}

	other := *opts
	other.Seed = 8
	if d := kumatest.DiffFrames(first, kumatest.Random(&other), nil); d == "" {
		t.Error("two seeds gave the same frame")
	}
}

// TestRandomIsTheSameFrameOnEveryMachine pins a small frame down to its text,
// which is the part a test on another machine or another Go release could lose
// without anything else noticing.
func TestRandomIsTheSameFrameOnEveryMachine(t *testing.T) {
	f := kumatest.Random(&kumatest.RandomOptions{
		Rows:  3,
		Types: []dtype.DataType{dtype.Int64, dtype.String, dtype.Bool},
		Names: []string{"qty", "symbol", "filled"},
		Seed:  1,
	})

	const want = `kuma.Frame[kuma.Dynamic] 3 rows x 3 cols

     qty | symbol                          | filled
   int64 | string                          | bool
---------+---------------------------------+-------
   85520 | lnyhyiguqesyjpcldyroeoqqhcaaaki | true
  468274 | ifsoyg                          | true
  117310 | gxkvjbsc                        | false`

	if got := f.Render(nil); got != want {
		t.Errorf("the frame is\n%s\nand it should be\n%s", got, want)
	}
}

func TestRandomTakesTheNamesItIsGiven(t *testing.T) {
	f := kumatest.Random(&kumatest.RandomOptions{
		Types: []dtype.DataType{dtype.Int64, dtype.Int64, dtype.Int64},
		Names: []string{"first", "", "third"},
	})

	want := []string{"first", "column_2", "third"}
	if names := f.Names(); !slices.Equal(names, want) {
		t.Errorf("the columns are %v, want %v", names, want)
	}
}

// TestRandomLeavesRoughlyTheAskedForFractionOut checks the missing values, since
// a frame with none of them exercises none of the code that matters.
func TestRandomLeavesRoughlyTheAskedForFractionOut(t *testing.T) {
	const rows = 10000

	for _, nulls := range []float64{0, 0.25, 1} {
		f := kumatest.Random(&kumatest.RandomOptions{
			Rows:  rows,
			Types: []dtype.DataType{dtype.Int64},
			Nulls: nulls,
		})

		got := float64(f.ColumnAt(0).Data().NullCount()) / rows
		if math.Abs(got-nulls) > 0.02 {
			t.Errorf("%v of the values are missing, want about %v", got, nulls)
		}
	}
}

// TestRandomFillsEveryColumnItSaysItCan is the check that a type in the list of
// what can be made is a type that comes out with values in it.
func TestRandomFillsEveryColumnItSaysItCan(t *testing.T) {
	types := randomTypes()
	f := kumatest.Random(&kumatest.RandomOptions{Rows: 200, Types: types, Seed: 3})

	for i, dt := range types {
		col := f.ColumnAt(i)
		if !dtype.Equal(col.DType(), dt) {
			t.Errorf("column %d is %s, want %s", i, col.DType(), dt)
		}
		if col.Len() != 200 {
			t.Errorf("column %s is %d rows, want 200", dt, col.Len())
		}

		want := 0
		if dt.Kind() == dtype.NullKind {
			// A null column is missing values and nothing else, by definition.
			want = 200
		}
		if n := col.Data().NullCount(); n != want {
			t.Errorf("column %s holds %d missing values, want %d", dt, n, want)
		}
	}

	// Rendering is the cheapest way to ask every column for every value, and it
	// catches a value that is outside what its type allows.
	if s := f.Render(&kuma.PrintOptions{MaxRows: 200}); strings.Count(s, "\n") < 200 {
		t.Errorf("the frame printed as %d lines, want a line per row", strings.Count(s, "\n"))
	}
}

// TestRandomMakesBothKindsOfString covers the split in the string layout, where
// a short value lives in the view and a long one lives in a buffer beside it.
func TestRandomMakesBothKindsOfString(t *testing.T) {
	f := kumatest.Random(&kumatest.RandomOptions{
		Rows:  200,
		Types: []dtype.DataType{dtype.String},
		Seed:  4,
	})
	s, err := f.Series[string]("column_1")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}

	var short, long int
	for _, v := range s.Values() {
		if len(v) > 12 {
			long++
			continue
		}
		short++
	}
	if short == 0 || long == 0 {
		t.Errorf("%d values fit in a view and %d did not, and a test wants both",
			short, long)
	}
}

// TestRandomDrawsBothWaysOnABoolColumn checks that a column of booleans is not
// all one value, which is the sort of thing that goes unnoticed for a year.
func TestRandomDrawsBothWaysOnABoolColumn(t *testing.T) {
	f := kumatest.Random(&kumatest.RandomOptions{
		Rows:  1000,
		Types: []dtype.DataType{dtype.Bool},
		Seed:  5,
	})
	s, err := f.Series[bool]("column_1")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}

	var yes int
	for _, v := range s.Values() {
		if v {
			yes++
		}
	}
	if yes < 400 || yes > 600 {
		t.Errorf("%d of 1000 booleans are true, want something near half", yes)
	}
}

// TestRandomTurnsAwayWhatItCannotMake checks the panics, which are all about a
// mistake in the test that called it rather than about the data.
func TestRandomTurnsAwayWhatItCannotMake(t *testing.T) {
	cases := []struct {
		name string
		opts *kumatest.RandomOptions
		says string
	}{
		{
			name: "a negative number of rows",
			opts: &kumatest.RandomOptions{Rows: -1},
			says: "Random of -1 rows",
		},
		{
			name: "more missing values than there are",
			opts: &kumatest.RandomOptions{Nulls: 1.5},
			says: "with 1.5 of the values missing",
		},
		{
			name: "a negative fraction missing",
			opts: &kumatest.RandomOptions{Nulls: -0.5},
			says: "with -0.5 of the values missing",
		},
		{
			name: "more names than columns",
			opts: &kumatest.RandomOptions{
				Types: []dtype.DataType{dtype.Int64},
				Names: []string{"a", "b"},
			},
			says: "with 2 names for 1 columns",
		},
		{
			name: "a type it cannot make yet",
			opts: &kumatest.RandomOptions{
				Types: []dtype.DataType{dtype.List{Elem: dtype.Int64}},
			},
			says: "cannot make a list<int64> column yet",
		},
		{
			name: "a decimal, which an array cannot hold yet",
			opts: &kumatest.RandomOptions{
				Types: []dtype.DataType{dtype.Decimal128{Precision: 10, Scale: 2}},
			},
			says: "cannot make a decimal128(10, 2) column yet",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Fatal("it did not panic")
				}
				msg, ok := r.(string)
				if !ok {
					t.Fatalf("it panicked with %#v, want a string", r)
				}
				if !strings.HasPrefix(msg, "kumatest: ") {
					t.Errorf("the panic is %q, and it should say which package it came from", msg)
				}
				if !strings.Contains(msg, c.says) {
					t.Errorf("the panic is %q, and it should mention %q", msg, c.says)
				}
			}()

			kumatest.Random(c.opts)
		})
	}
}

// TestRandomOfNoRows is the edge a benchmark hits when it divides its work up
// and one of the pieces is empty.
func TestRandomOfNoRows(t *testing.T) {
	f := kumatest.Random(&kumatest.RandomOptions{Rows: 0, Types: randomTypes()})
	if f.NumRows() != defaultRowsInTest {
		t.Errorf("Rows 0 gave %d rows, want the default", f.NumRows())
	}

	one := kumatest.Random(&kumatest.RandomOptions{
		Rows:  1,
		Types: []dtype.DataType{dtype.Int64},
	})
	if one.NumRows() != 1 {
		t.Errorf("Rows 1 gave %d rows", one.NumRows())
	}
}

// defaultRowsInTest is the number of rows a frame has when the options do not
// say, written out here so that the test reads as the promise the documentation
// makes rather than as the constant the code uses.
const defaultRowsInTest = 10
