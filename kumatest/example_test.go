package kumatest_test

import (
	"fmt"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kumatest"
)

// printer is a [kumatest.TB] that prints instead of failing, so that these
// examples can show what a failing test would put in the log. A real test
// passes its own *testing.T.
type printer struct{}

func (printer) Helper() {}

func (printer) Errorf(format string, args ...any) {
	fmt.Printf(format+"\n", args...)
}

// This example compares two frames the way a test would, and prints what the
// test log would hold when they differ.
func Example() {
	got := frame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "GOOG", "AMZN").Column(),
		kuma.NewSeries("price", 150.25, 100.0, 200.0, 300.0).Column(),
	)
	want := frame(
		kuma.NewSeries("symbol", "AAPL", "MSFT", "GOOG", "META").Column(),
		kuma.NewSeries("price", 150.5, 100.0, 200.0, 300.0).Column(),
	)

	kumatest.EqualFrames(printer{}, got, want, nil)

	// Output:
	// frames differ in 2 of 4 rows
	//
	//   row | column | got    | want
	// ------+--------+--------+------
	//     0 | price  | 150.25 | 150.5
	//     3 | symbol | AMZN   | META
}

// This example allows for the rounding in a computed column, which is what
// [kumatest.Options.Fraction] is for. The two prices here are the same number
// to a part in a million and different in the last bit, which is the difference
// a sum of floating point values arrives at.
func ExampleOptions() {
	tenth, fifth := 0.1, 0.2
	got := kuma.NewSeries("price", tenth+fifth)
	want := kuma.NewSeries("price", 0.3)

	fmt.Println(kumatest.DiffSeries(got, want, nil))
	fmt.Println("equal to a part in a billion:",
		kumatest.DiffSeries(got, want, &kumatest.Options{Fraction: 1e-9}) == "")

	// Output:
	// series differ in 1 of 1 rows
	//
	//   row | got                 | want
	// ------+---------------------+-----
	//     0 | 0.30000000000000004 | 0.3
	// equal to a part in a billion: true
}

// This example builds a frame to run something on, rather than a frame to
// prove something about. The same options give the same frame every time.
func ExampleRandom() {
	f := kumatest.Random(&kumatest.RandomOptions{
		Rows:  4,
		Types: []dtype.DataType{dtype.String, dtype.Float64, dtype.Int64},
		Names: []string{"symbol", "price", "qty"},
		Nulls: 0.1,
		Seed:  2,
	})

	fmt.Println(f)

	// Output:
	// kuma.Frame[kuma.Dynamic] 4 rows x 3 cols
	//
	//   symbol           |   price |     qty
	//   string           | float64 |   int64
	// -------------------+---------+--------
	//   kipdofjfprafdrec | -545.01 |    null
	//   txxyqr           | -287.66 |  404700
	//   iyqm             | -457.49 |  191080
	//   vpekp            |   572.3 | -326703
}

// This example asks for the difference as text rather than reporting it, which
// is what a test that wants to say something else about it does.
func ExampleDiffColumns() {
	got := kuma.NewSeries("qty", int64(1), 2, 3).Column()
	want := kuma.NewSeries("qty", int32(1), 2, 3).Column()

	fmt.Println(kumatest.DiffColumns(got, want, nil))

	// Output:
	// columns differ
	//   the type is int64 where it should be int32
}
