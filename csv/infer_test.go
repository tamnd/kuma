package csv

import "testing"

func TestInferValue(t *testing.T) {
	cases := []struct {
		in   string
		want inferred
	}{
		{"1", inferInt},
		{"-1", inferInt},
		{"+1", inferInt},
		{"007", inferInt},
		{"9223372036854775807", inferInt},
		{"9223372036854775808", inferFloat}, // one past an int64
		{"1.5", inferFloat},
		{"1e3", inferFloat},
		{"-0.0", inferFloat},
		{"NaN", inferFloat},
		{"Inf", inferFloat},
		{"true", inferBool},
		{"FALSE", inferBool},
		{"False", inferBool},
		{"fAlsE", inferString}, // not a spelling strconv takes
		{"t", inferString},
		{"F", inferString},
		{"AAPL", inferString},
		{"1,5", inferString},
		{" 1", inferString}, // a space is part of the value
		{"0x10", inferString},
	}

	for _, c := range cases {
		if got := inferValue(c.in); got != c.want {
			t.Errorf("inferValue(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestInferMerge(t *testing.T) {
	cases := []struct {
		a, b inferred
		want inferred
	}{
		{inferNothing, inferInt, inferInt},
		{inferInt, inferNothing, inferInt},
		{inferInt, inferInt, inferInt},
		{inferInt, inferFloat, inferFloat},
		{inferFloat, inferInt, inferFloat},
		{inferBool, inferInt, inferString},
		{inferBool, inferString, inferString},
		{inferString, inferInt, inferString},
		{inferNothing, inferNothing, inferNothing},
	}

	for _, c := range cases {
		if got := c.a.merge(c.b); got != c.want {
			t.Errorf("%v merge %v = %v, want %v", c.a, c.b, got, c.want)
		}
	}
}

func TestInferDType(t *testing.T) {
	// Nothing seen is a string column, since a string can hold whatever turns
	// up in the rows the sample did not reach.
	for _, i := range []inferred{inferNothing, inferString} {
		if got := i.dtype().String(); got != "string" {
			t.Errorf("%v is a %s column, want string", i, got)
		}
	}
}

func TestInferIgnoresShortRows(t *testing.T) {
	// A row cannot be short by the time the reader gets here, since the csv
	// reader insists every record is the same length, but inference should not
	// be the thing that falls over if that ever changes.
	opts := (&Options{}).withDefaults()
	got := infer([][]string{{"1", "2", "3"}, {"4"}}, []int{0, 1}, &opts)

	if len(got) != 2 || got[0] != inferInt || got[1] != inferInt {
		t.Errorf("got %v, want two integer columns", got)
	}
}
