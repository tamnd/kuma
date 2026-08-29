package ndjson

import (
	"encoding/json/jsontext"
	"strings"
	"testing"
)

func TestInferValue(t *testing.T) {
	cases := []struct {
		in   string
		want inferred
	}{
		{"1", inferInt},
		{"-1", inferInt},
		{"9223372036854775807", inferInt},
		{"9223372036854775808", inferFloat}, // one past an int64
		{"1.5", inferFloat},
		{"1e3", inferFloat},
		{"1e999", inferString},  // no float64 holds it
		{"-1e999", inferString}, // nor this
		{"1e-999", inferFloat},  // this one is zero, which is a float
		{"1.0", inferFloat},     // a point means a float, whatever follows it
		{"-0.0", inferFloat},
		{"true", inferBool},
		{"false", inferBool},
		{"null", inferNothing},
		{`"AAPL"`, inferString},
		{`"1"`, inferString}, // JSON says this is text, so it stays text
		{`""`, inferString},
		{`{"a":1}`, inferString},
		{"[1,2]", inferString},
		{"[]", inferString},
	}

	for _, c := range cases {
		d := jsontext.NewDecoder(strings.NewReader(c.in))
		got, err := inferValue(d)
		if err != nil {
			t.Errorf("inferValue(%q): %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("inferValue(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

// TestInferValueReadsExactlyOne checks the other half of what inferValue owes
// the caller, which is that the decoder is left on the next value whatever the
// one it read turned out to be. A nested object is the case that could get this
// wrong.
func TestInferValueReadsExactlyOne(t *testing.T) {
	d := jsontext.NewDecoder(strings.NewReader(`[{"a":[1,{"b":2}]},"next"]`))
	if _, err := d.ReadToken(); err != nil {
		t.Fatalf("ReadToken: %v", err)
	}
	if _, err := inferValue(d); err != nil {
		t.Fatalf("inferValue: %v", err)
	}

	got, err := d.ReadValue()
	if err != nil {
		t.Fatalf("ReadValue: %v", err)
	}
	if string(got) != `"next"` {
		t.Errorf("the decoder is at %s, want the value after the object", got)
	}
}

func TestInferValueErrors(t *testing.T) {
	for _, in := range []string{"", "tru", `{"a":`, "[1,"} {
		d := jsontext.NewDecoder(strings.NewReader(in))
		if _, err := inferValue(d); err == nil {
			t.Errorf("inferValue(%q) worked, want an error", in)
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
	// up in the lines the sample did not reach.
	for _, i := range []inferred{inferNothing, inferString} {
		if got := i.dtype().String(); got != "string" {
			t.Errorf("%v is a %s column, want string", i, got)
		}
	}
}
