package kuma_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
)

// TestColumnErrorSuggestions is the table for the part of the message people
// read: which name it offers when the one they typed is not there. The rule is
// that a suggestion has to be close, since a wrong suggestion is worse than
// none at all.
func TestColumnErrorSuggestions(t *testing.T) {
	have := []string{"symbol", "price", "qty", "side", "ts"}

	tests := []struct {
		name string
		want string
	}{
		{"sym", "symbol"},
		{"symbo", "symbol"},
		{"symbolq", "symbol"},
		{"smybol", "symbol"},
		{"SYMBOL", "symbol"},
		{"Price", "price"},
		{"prices", "price"},
		{"qt", "qty"},
		{"sid", "side"},
		{"t", "ts"},
		{"volume", ""},
		{"", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := &kuma.ColumnError{Op: "Select", Name: tt.name, Have: have}
			msg := err.Error()

			const marker = "  did you mean: "
			got := ""
			if k := strings.Index(msg, marker); k >= 0 {
				got = strings.TrimSuffix(msg[k+len(marker):], "?")
			}
			if got != tt.want {
				t.Errorf("the message suggests %q, want %q\n%s", got, tt.want, msg)
			}
		})
	}
}

func TestColumnErrorMessage(t *testing.T) {
	tests := []struct {
		name string
		err  *kuma.ColumnError
		want string
	}{
		{
			name: "with a suggestion",
			err:  &kuma.ColumnError{Op: "Select", Name: "prices", Have: []string{"symbol", "price"}},
			want: "kuma: column \"prices\" not found in Select\n  available: symbol, price\n  did you mean: price?",
		},
		{
			name: "with nothing close",
			err:  &kuma.ColumnError{Op: "Drop", Name: "volume", Have: []string{"symbol", "price"}},
			want: "kuma: column \"volume\" not found in Drop\n  available: symbol, price",
		},
		{
			name: "with no operation",
			err:  &kuma.ColumnError{Name: "volume", Have: []string{"price"}},
			want: "kuma: column \"volume\" not found\n  available: price",
		},
		{
			name: "on a frame with no columns",
			err:  &kuma.ColumnError{Op: "Column", Name: "price"},
			want: "kuma: column \"price\" not found in Column\n  the frame has no columns",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.err.Error(); got != tt.want {
				t.Errorf("Error() =\n%s\nwant\n%s", got, tt.want)
			}
			if !errors.Is(tt.err, kuma.ErrNoColumn) {
				t.Error("the error is not an ErrNoColumn")
			}
		})
	}
}

// TestErrorsAreDistinct is worth having because errors.Is on a sentinel that
// was accidentally assigned from another sentinel would be true for both, and
// every test that checks an error would still pass.
func TestErrorsAreDistinct(t *testing.T) {
	all := []error{
		kuma.ErrNoColumn,
		kuma.ErrDuplicateColumn,
		kuma.ErrWrongType,
		kuma.ErrLength,
		kuma.ErrNoValues,
	}

	for i, a := range all {
		for j, b := range all {
			if i != j && errors.Is(a, b) {
				t.Errorf("%v and %v are the same error", a, b)
			}
		}
	}
}
