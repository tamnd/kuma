package dataset

import (
	"errors"
	"slices"
	"testing"

	"github.com/tamnd/kuma/dtype"
)

func TestParse(t *testing.T) {
	cases := []struct {
		name string
		rel  string
		keys []string
		vals []Value
	}{
		{
			name: "a file in the root has no partitions",
			rel:  "part-0.parquet",
		},
		{
			name: "one directory is one partition",
			rel:  "year=2024/part-0.parquet",
			keys: []string{"year"},
			vals: []Value{{Text: "2024"}},
		},
		{
			name: "the directories are the columns in order",
			rel:  "year=2024/month=03/day=17/part-0.parquet",
			keys: []string{"year", "month", "day"},
			vals: []Value{{Text: "2024"}, {Text: "03"}, {Text: "17"}},
		},
		{
			name: "an empty value is an empty value",
			rel:  "note=/part-0.parquet",
			keys: []string{"note"},
			vals: []Value{{Text: ""}},
		},
		{
			name: "the writers' name for a null reads as one",
			rel:  "region=__HIVE_DEFAULT_PARTITION__/part-0.parquet",
			keys: []string{"region"},
			vals: []Value{{Null: true}},
		},
		{
			name: "a value can hold an equals sign",
			rel:  "expr=a=b/part-0.parquet",
			keys: []string{"expr"},
			vals: []Value{{Text: "a=b"}},
		},
		{
			name: "percent escapes are decoded",
			rel:  "city=New%20York/part-0.parquet",
			keys: []string{"city"},
			vals: []Value{{Text: "New York"}},
		},
		{
			name: "a key can be escaped too",
			rel:  "the%20key=1/part-0.parquet",
			keys: []string{"the key"},
			vals: []Value{{Text: "1"}},
		},
		{
			name: "a percent that is not an escape is left alone",
			rel:  "share=100%/part-0.parquet",
			keys: []string{"share"},
			vals: []Value{{Text: "100%"}},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			keys, vals, err := parse(c.rel)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(keys, c.keys) {
				t.Errorf("keys %q, want %q", keys, c.keys)
			}
			if !slices.Equal(vals, c.vals) {
				t.Errorf("values %v, want %v", vals, c.vals)
			}
		})
	}
}

func TestParseNotADataset(t *testing.T) {
	for _, rel := range []string{
		"2024/part-0.parquet",
		"year=2024/march/part-0.parquet",
		"=2024/part-0.parquet",
	} {
		if _, _, err := parse(rel); !errors.Is(err, ErrLayout) {
			t.Errorf("parse(%q) = %v, want ErrLayout", rel, err)
		}
	}
}

func TestValueString(t *testing.T) {
	if got := (Value{Text: "03"}).String(); got != "03" {
		t.Errorf("got %q, want 03", got)
	}
	if got := (Value{Null: true}).String(); got != "null" {
		t.Errorf("got %q, want null", got)
	}
}

func TestNumeric(t *testing.T) {
	cases := []struct {
		text string
		want dtype.DataType
	}{
		{"1", dtype.Int64},
		{"0", dtype.Int64},
		{"-1", dtype.Int64},
		{"2024", dtype.Int64},
		{"1.5", dtype.Float64},
		{"-0.25", dtype.Float64},
		{"", dtype.String},
		{"01", dtype.String},
		{"+1", dtype.String},
		{"-0", dtype.Float64},
		{"1.50", dtype.String},
		{"-Inf", dtype.String},
		{"1e3", dtype.String},
		{"0x10", dtype.String},
		{"1_000", dtype.String},
		{"march", dtype.String},
		{"Inf", dtype.String},
		{"NaN", dtype.String},
		{"99999999999999999999", dtype.String},
	}

	for _, c := range cases {
		if got := numeric(c.text); !dtype.Equal(got, c.want) {
			t.Errorf("numeric(%q) = %s, want %s", c.text, got, c.want)
		}
	}
}

func TestInfer(t *testing.T) {
	text := func(vals ...string) []Value {
		out := make([]Value, len(vals))
		for i, v := range vals {
			out[i] = Value{Text: v}
		}
		return out
	}
	null := Value{Null: true}

	cases := []struct {
		name string
		vals []Value
		want dtype.DataType
	}{
		{"whole numbers", text("1", "2", "3"), dtype.Int64},
		{"one of them with a point", text("1", "2.5"), dtype.Float64},
		{"one of them a word", text("1", "march"), dtype.String},
		{"a word after a point", text("2.5", "march"), dtype.String},
		{"a leading zero holds the column to text", text("01", "02"), dtype.String},
		{"nulls do not decide anything", []Value{null, {Text: "1"}}, dtype.Int64},
		{"nothing but nulls is text", []Value{null, null}, dtype.String},
		{"no values at all is text", nil, dtype.String},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := infer(c.vals); !dtype.Equal(got, c.want) {
				t.Errorf("got %s, want %s", got, c.want)
			}
		})
	}
}

// FuzzParse checks that no path makes the parser panic and that what it returns
// hangs together, which is one key per value and never one without the other.
func FuzzParse(f *testing.F) {
	for _, seed := range []string{
		"part-0.parquet",
		"year=2024/month=03/part-0.parquet",
		"region=__HIVE_DEFAULT_PARTITION__/f",
		"city=New%20York/f",
		"a=%/f",
		"=/f",
		"//f",
		"a=b=c/f",
	} {
		f.Add(seed)
	}

	f.Fuzz(func(t *testing.T, rel string) {
		keys, vals, err := parse(rel)
		if err != nil {
			if !errors.Is(err, ErrLayout) {
				t.Errorf("parse(%q) = %v, want ErrLayout", rel, err)
			}
			if keys != nil || vals != nil {
				t.Errorf("parse(%q) returned %q and %v with an error", rel, keys, vals)
			}
			return
		}
		if len(keys) != len(vals) {
			t.Errorf("parse(%q) returned %d keys for %d values", rel, len(keys), len(vals))
		}
		for _, k := range keys {
			if k == "" {
				t.Errorf("parse(%q) returned an empty key", rel)
			}
		}
		for _, v := range vals {
			if v.Null && v.Text != "" {
				t.Errorf("parse(%q) returned a null holding %q", rel, v.Text)
			}
		}
	})
}
