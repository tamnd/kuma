package kernel_test

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// TestAndOr is the whole of Kleene's table, written out. Nine rows for and,
// nine for or, and nothing in the implementation is allowed to be cleverer than
// this table.
func TestAndOr(t *testing.T) {
	three := []any{true, false, nil}

	tests := []struct {
		name string
		fn   func(a, b *array.Chunked) (*array.Chunked, error)
		want []any
	}{
		{
			name: "and",
			fn:   kernel.And,
			want: []any{
				true, false, nil,
				false, false, false,
				nil, false, nil,
			},
		},
		{
			name: "or",
			fn:   kernel.Or,
			want: []any{
				true, true, true,
				true, false, nil,
				true, nil, nil,
			},
		},
	}

	// Every pair of the three values, in the order the tables above are
	// written: the left value varies slowest.
	var left, right []any
	for _, x := range three {
		for _, y := range three {
			left = append(left, x)
			right = append(right, y)
		}
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, b := col(t, dtype.Bool, left), col(t, dtype.Bool, right)
			got, err := tt.fn(a, b)
			if err != nil {
				t.Fatalf("%s: %v", tt.name, err)
			}
			have := answers(t, got)
			if !same(have, tt.want) {
				t.Fatalf("%s is %v, want %v", tt.name, have, tt.want)
			}
			for i := range left {
				t.Logf("%v %s %v is %v", left[i], tt.name, right[i], have[i])
			}
		})
	}
}

func TestNot(t *testing.T) {
	c := col(t, dtype.Bool, []any{true, false, nil}, []any{true})

	got, err := kernel.Not(c)
	if err != nil {
		t.Fatalf("Not: %v", err)
	}
	if have, want := answers(t, got), []any{false, true, nil, false}; !same(have, want) {
		t.Errorf("the negation is %v, want %v", have, want)
	}
}

// TestAndShortCircuitsWithALiteral is the case that lets a predicate be turned
// off without building a column of falses.
func TestAndShortCircuits(t *testing.T) {
	c := col(t, dtype.Bool, []any{true, false, nil})
	off := col(t, dtype.Bool, []any{false})
	on := col(t, dtype.Bool, []any{true})

	got, err := kernel.And(c, off)
	if err != nil {
		t.Fatalf("And: %v", err)
	}
	if have, want := answers(t, got), []any{false, false, false}; !same(have, want) {
		t.Errorf("and false is %v, want %v", have, want)
	}

	got, err = kernel.And(on, c)
	if err != nil {
		t.Fatalf("And: %v", err)
	}
	if have, want := answers(t, got), []any{true, false, nil}; !same(have, want) {
		t.Errorf("true and is %v, want %v", have, want)
	}
}

func TestOrShortCircuits(t *testing.T) {
	c := col(t, dtype.Bool, []any{true, false, nil})
	on := col(t, dtype.Bool, []any{true})

	got, err := kernel.Or(c, on)
	if err != nil {
		t.Fatalf("Or: %v", err)
	}
	if have, want := answers(t, got), []any{true, true, true}; !same(have, want) {
		t.Errorf("or true is %v, want %v", have, want)
	}
}

// TestLogicOnAColumnOfNothing is the column an empty file gives, where every
// value is missing and the type says so.
func TestLogicOnAColumnOfNothing(t *testing.T) {
	none := col(t, dtype.Null, []any{nil, nil, nil})
	c := col(t, dtype.Bool, []any{true, false, nil})

	got, err := kernel.And(c, none)
	if err != nil {
		t.Fatalf("And: %v", err)
	}
	if have, want := answers(t, got), []any{nil, false, nil}; !same(have, want) {
		t.Errorf("and nothing is %v, want %v", have, want)
	}

	got, err = kernel.Not(none)
	if err != nil {
		t.Fatalf("Not: %v", err)
	}
	if have, want := answers(t, got), []any{nil, nil, nil}; !same(have, want) {
		t.Errorf("not nothing is %v, want %v", have, want)
	}
}

func TestLogicAcrossChunks(t *testing.T) {
	a := col(t, dtype.Bool, []any{true}, []any{false, true}, nil, []any{true, true})
	b := col(t, dtype.Bool, []any{true, true}, []any{false, false, true})

	got, err := kernel.And(a, b)
	if err != nil {
		t.Fatalf("And: %v", err)
	}
	if have, want := answers(t, got), []any{true, false, false, false, true}; !same(have, want) {
		t.Errorf("the conjunction is %v, want %v", have, want)
	}
}

func TestLogicOnAColumnThatIsNotACondition(t *testing.T) {
	c := col(t, dtype.Int64, []any{int64(1)})
	ok := col(t, dtype.Bool, []any{true})

	tests := []struct {
		name string
		err  error
	}{
		{"and on the left", firstError(kernel.And(c, ok))},
		{"and on the right", firstError(kernel.And(ok, c))},
		{"or", firstError(kernel.Or(c, ok))},
		{"not", firstError(kernel.Not(c))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.err == nil {
				t.Fatal("an int64 column was taken as a condition")
			}
			if !strings.Contains(tt.err.Error(), "not a condition") {
				t.Errorf("the error is %q, which does not say what is wrong", tt.err)
			}
		})
	}
}

// firstError throws the column away and keeps the error, so that a table of
// calls can be written in one line each.
func firstError(_ *array.Chunked, err error) error { return err }

func TestLogicUnalignedLengths(t *testing.T) {
	a := col(t, dtype.Bool, []any{true, false, true})
	b := col(t, dtype.Bool, []any{true, false})

	defer func() {
		if recover() == nil {
			t.Fatal("two columns of different lengths did not panic")
		}
	}()
	_, _ = kernel.And(a, b)
}

func TestLogicNilColumn(t *testing.T) {
	tests := []struct {
		name string
		call func()
	}{
		{"and", func() { _, _ = kernel.And(nil, nil) }},
		{"or", func() { _, _ = kernel.Or(nil, nil) }},
		{"not", func() { _, _ = kernel.Not(nil) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("a nil column did not panic")
				}
			}()
			tt.call()
		})
	}
}
