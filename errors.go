package kuma

import (
	"errors"
	"fmt"
	"strings"
)

// The errors this package returns. They are comparable with errors.Is, and the
// error values themselves carry the detail: which column, which operation, and
// what the frame actually holds.
//
// Nothing here panics across an API boundary. The exceptions are the ones Go
// itself makes: an index out of range panics, the same way indexing a slice
// does, because a program that reads past the end of a column has a bug in it
// rather than a condition to handle.
var (
	// ErrNoColumn is returned when a column name is not in the frame. The error
	// lists the names that are, and suggests one if the name looks like a typo.
	ErrNoColumn = errors.New("no such column")

	// ErrDuplicateColumn is returned when two columns in the same frame have
	// the same name.
	ErrDuplicateColumn = errors.New("duplicate column")

	// ErrWrongType is returned when a column is read as a Go type it is not
	// stored as, such as reading a float64 column as an int64.
	ErrWrongType = errors.New("wrong type")

	// ErrLength is returned when the columns of a frame are not all the same
	// length.
	ErrLength = errors.New("columns of different length")

	// ErrNoValues is returned when a column is built with nothing underneath
	// it.
	ErrNoValues = errors.New("no values")
)

// ColumnError says that a column was asked for and is not there.
//
// It prints on several lines on purpose. A missing column name is the most
// common thing that goes wrong in day to day work, and the fastest way to fix
// it is to see what the frame does hold and to be told which of those names is
// one letter away from the one that was typed.
//
//	kuma: column "sym" not found in Select
//	  available: symbol, price, qty, side
//	  did you mean: symbol?
type ColumnError struct {
	// Op is the operation that was running, such as "Select".
	Op string

	// Name is the column that was asked for.
	Name string

	// Have is the names the frame holds, in order.
	Have []string
}

// Error returns the message described on ColumnError.
func (e *ColumnError) Error() string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "kuma: column %q not found", e.Name)
	if e.Op != "" {
		fmt.Fprintf(&sb, " in %s", e.Op)
	}
	if len(e.Have) == 0 {
		sb.WriteString("\n  the frame has no columns")
		return sb.String()
	}

	fmt.Fprintf(&sb, "\n  available: %s", strings.Join(e.Have, ", "))
	if near, ok := nearest(e.Name, e.Have); ok {
		fmt.Fprintf(&sb, "\n  did you mean: %s?", near)
	}
	return sb.String()
}

// Unwrap returns ErrNoColumn, so that errors.Is(err, kuma.ErrNoColumn) is the
// way to ask whether a name was wrong.
func (e *ColumnError) Unwrap() error { return ErrNoColumn }

// noColumn returns the error for a name that is not in names.
func noColumn(op, name string, names []string) error {
	return &ColumnError{Op: op, Name: name, Have: names}
}

// nearest returns the name in have that is closest to name, if one is close
// enough to be worth suggesting.
//
// Close enough means one of three things. The same name in another case, which
// is the most common miss of all. An edit distance of at most a third of the
// length of the name and at most three whatever the length, which covers a
// transposition, a missing letter and an extra one. Or the name being the start
// of one that is there, which is what makes "sym" suggest "symbol" even though
// three edits apart is nowhere near close for a name that short.
//
// Everything else gets no suggestion. A wrong suggestion is worse than none,
// because it sends the reader looking at the wrong column, so "volume" does not
// get offered "qty" on the grounds that it is the least bad of a bad set.
func nearest(name string, have []string) (string, bool) {
	for _, h := range have {
		if strings.EqualFold(h, name) {
			// Only the case is wrong, so there is nothing closer to find.
			return h, true
		}
	}
	if name == "" {
		return "", false
	}

	limit := min(max(len(name)/3, 1), 3)
	best, bestDist := "", limit+1
	prefix := ""
	for _, h := range have {
		if d := distance(name, h, bestDist); d < bestDist {
			best, bestDist = h, d
		}
		if prefix == "" && len(h) > len(name) && strings.EqualFold(h[:len(name)], name) {
			prefix = h
		}
	}

	if best != "" {
		return best, true
	}
	return prefix, prefix != ""
}

// distance returns the Levenshtein distance between a and b, giving up and
// returning limit as soon as it is clear the answer is at least that, since the
// caller only cares about short distances.
func distance(a, b string, limit int) int {
	if abs(len(a)-len(b)) > limit {
		return limit
	}

	// One row of the matrix, holding the distance from the prefix of a to the
	// prefix of b. The full matrix is not needed because each cell depends only
	// on the row above and the cell to its left.
	prev := make([]int, len(b)+1)
	curr := make([]int, len(b)+1)
	for j := range prev {
		prev[j] = j
	}

	for i := 1; i <= len(a); i++ {
		curr[0] = i
		row := curr[0]
		for j := 1; j <= len(b); j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			curr[j] = min(prev[j]+1, curr[j-1]+1, prev[j-1]+cost)
			row = min(row, curr[j])
		}
		if row >= limit {
			return limit
		}
		prev, curr = curr, prev
	}
	return min(prev[len(b)], limit)
}

// abs returns the absolute value of n.
func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
}
