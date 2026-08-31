package plan

// The edit distance behind the did you mean line, tested on its own. It is
// worth its own file because the version that runs keeps three rows of the
// matrix and rotates them, and gives up early, and neither of those shows up in
// the message when it goes wrong. It shows up as a suggestion quietly not being
// offered.

import (
	"strings"
	"testing"
)

// TestDistanceCounts is the table of answers worked out by hand, which is what
// says the recurrence is the one intended rather than merely the one the
// reference below also has.
func TestDistanceCounts(t *testing.T) {
	tests := []struct {
		a, b string
		want int
	}{
		{"", "", 0},
		{"a", "", 1},
		{"", "ab", 2},
		{"abc", "abc", 0},
		{"ab", "ba", 1},
		{"abc", "acb", 1},
		{"price", "prcie", 1},
		{"symbol", "smybol", 1},
		{"side", "sdie", 1},
		{"qty", "qyt", 1},
		{"ts", "st", 1},
		{"price", "irpce", 2},
		{"price", "prices", 1},
		{"kitten", "sitting", 3},

		// Two letters swapped across a stretch that also changed is the one
		// place the restricted count and the unrestricted one part company.
		// Turning "ca" into "abc" is a swap and an insertion, which the
		// unrestricted count charges 2 for and this charges 3, because it will
		// not edit between a pair and then swap it.
		{"ca", "abc", 3},
	}

	for _, tt := range tests {
		name := tt.a + " to " + tt.b
		t.Run(name, func(t *testing.T) {
			if got := distance(tt.a, tt.b, 99); got != tt.want {
				t.Errorf("distance(%q, %q) = %d, want %d", tt.a, tt.b, got, tt.want)
			}
			if got := distance(tt.b, tt.a, 99); got != tt.want {
				t.Errorf("distance(%q, %q) = %d, want %d, which is not the same both ways round", tt.b, tt.a, got, tt.want)
			}
		})
	}
}

// TestDistanceMatchesTheFullMatrix runs every pair of short strings over a
// three letter alphabet past a reference that keeps the whole matrix and never
// gives up, and asks for the same answer at every limit.
//
// The two properties this holds are the ones the running version could lose
// without any test of the message noticing. Rotating three rows has to leave
// the same numbers in them the full matrix would have, and giving up early has
// to be giving up only when the answer really is at least the limit, so what
// comes back is the answer or the limit and never a third thing.
func TestDistanceMatchesTheFullMatrix(t *testing.T) {
	words := shortWords("abc", 4)
	for _, a := range words {
		for _, b := range words {
			want := fullMatrix(a, b)
			for limit := range 6 {
				if got := distance(a, b, limit); got != min(want, limit) {
					t.Fatalf("distance(%q, %q, %d) = %d, want %d", a, b, limit, got, min(want, limit))
				}
			}
			if got := distance(a, b, 99); got != want {
				t.Fatalf("distance(%q, %q, 99) = %d, want %d", a, b, got, want)
			}
		}
	}
}

// shortWords returns every string of up to n letters over the given alphabet,
// starting with the empty one.
func shortWords(alphabet string, n int) []string {
	words := []string{""}
	level := []string{""}
	for range n {
		var next []string
		for _, w := range level {
			for _, c := range alphabet {
				next = append(next, w+string(c))
			}
		}
		words = append(words, next...)
		level = next
	}
	return words
}

// fullMatrix is the same edit distance written the obvious way, keeping every
// row and giving up on nothing.
//
// It is here to be read rather than to be fast, since the point of comparing
// against it is that it is short enough to check by eye.
func fullMatrix(a, b string) int {
	d := make([][]int, len(a)+1)
	for i := range d {
		d[i] = make([]int, len(b)+1)
		d[i][0] = i
	}
	for j := range d[0] {
		d[0][j] = j
	}

	for i := 1; i <= len(a); i++ {
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
			if i > 1 && j > 1 && a[i-1] == b[j-2] && a[i-2] == b[j-1] {
				d[i][j] = min(d[i][j], d[i-2][j-2]+1)
			}
		}
	}
	return d[len(a)][len(b)]
}

// BenchmarkDistance is one name against one column name of about the length
// people give columns, which is the unit the search for a suggestion repeats
// once per column of the frame.
func BenchmarkDistance(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		intSink = distance("prcie", "price", 1)
	}
}

// BenchmarkDistanceGivesUp is the same against a name it is nowhere near, which
// is what most columns of a wide frame are and so is the cost that actually
// adds up.
func BenchmarkDistanceGivesUp(b *testing.B) {
	name := strings.Repeat("ab", 8)
	b.ReportAllocs()
	for b.Loop() {
		intSink = distance("prcie", name, 1)
	}
}

// intSink keeps the benchmarks above from being optimized away.
var intSink int
