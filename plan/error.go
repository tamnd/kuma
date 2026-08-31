package plan

import (
	"errors"
	"fmt"
	"strings"
)

// The errors a plan reports. They say kuma rather than plan, because a plan is
// something the kuma package built on the caller's behalf and naming a package
// they never imported would send them looking in the wrong place. They are the
// same values the kuma package exports, so a name that is wrong when the plan
// is checked and a name that is wrong when a frame is read give the same error
// to the same errors.Is.
var (
	// ErrNoColumn is returned when a column name is not in the schema. The
	// error lists the names that are, and suggests one if the name looks like a
	// typo.
	ErrNoColumn = errors.New("no such column")

	// ErrWrongType is returned when a value is used where its type has no
	// meaning, such as a literal no column can hold or an expression that is
	// asked to be a condition and is not.
	ErrWrongType = errors.New("wrong type")

	// ErrDuplicateColumn is returned when an operator would produce two columns
	// of the same name, which is a frame nobody can read a column out of by
	// name.
	ErrDuplicateColumn = errors.New("duplicate column")
)

// errNoPlan is what every entry point says about a nil node. A plan is built by
// the kuma package rather than written out by hand, so reaching it means a
// query was thrown away half built and the caller kept going, and the message
// is aimed at whoever has to work that out.
var errNoPlan = errors.New("kuma: a plan with no operator in it")

// errNoRun is what [Profile] says about a measure with no operator in it. A
// measure is filled in by the engine as a query runs, so an empty one means a
// profile was asked for of a query that never ran.
var errNoRun = errors.New("kuma: a profile of a query that has not run")

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
	// Op is the operation that was running, such as "Select", or the operator
	// the name was written in when the plan was being checked rather than run,
	// such as "Filter". It is empty when neither is known, which is what
	// checking an expression on its own against a schema gives.
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
// length of the name and at most three whatever the length, where two letters
// the wrong way round count as one edit rather than two. Or the name being the
// start of one that is there, which is what makes "sym" suggest "symbol" even
// though three edits apart is nowhere near close for a name that short.
//
// Counting a swap as one edit is what makes "prcie" suggest "price". Two
// letters the wrong way round is the typo people make most often after getting
// the case wrong, and a plain edit distance charges two substitutions for it,
// which is over the limit for every name shorter than six letters. Raising the
// limit to let it in would have let in every unrelated name that happens to be
// two edits away as well.
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

// distance returns the edit distance between a and b, counting two letters the
// wrong way round as one edit rather than as the two substitutions a plain
// Levenshtein distance charges for it. It gives up and returns limit as soon as
// it is clear the answer is at least that, since the caller only cares about
// short distances.
//
// The swap is counted the restricted way, meaning a pair of letters is swapped
// at most once and a stretch of text is never edited and then swapped again.
// That is the cheaper of the two ways to count it and it gives the same answer
// for anything anyone types by mistake, since the two only part company on
// strings mangled in several overlapping ways.
func distance(a, b string, limit int) int {
	if abs(len(a)-len(b)) > limit {
		return limit
	}

	// Three rows of the matrix, holding the distance from a prefix of a to a
	// prefix of b. The full matrix is not needed because each cell depends only
	// on the row above, the cell to its left, and for the swap the row above
	// that one.
	prev2 := make([]int, len(b)+1)
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
			if i > 1 && j > 1 && a[i-1] == b[j-2] && a[i-2] == b[j-1] {
				curr[j] = min(curr[j], prev2[j-2]+1)
			}
			row = min(row, curr[j])
		}
		// A row that is all at the limit means nothing below it can be under the
		// limit either. The smallest number in a row never falls as the rows go
		// down and never climbs by more than one, so the row two above the next
		// one is at least one short of the limit, and the swap, which is the only
		// thing reaching back that far, adds one to it.
		if row >= limit {
			return limit
		}
		prev2, prev, curr = prev, curr, prev2
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
