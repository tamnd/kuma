package plan

// Printing a plan for a person to read. There are two of those. One is the
// plan as a tree, which is the query written out. The other is the query as
// written set beside the query that will run, which is the only way a caller
// can see what the optimizer did for them.

import (
	"strings"
)

// Tree returns the plan as text, one operator per line with the inputs indented
// under it.
//
// It is [Node.String] for a whole plan rather than for one operator, and it is
// the shape both pandas and Polars print, so a reader who has seen one of those
// does not have to learn another. The two inputs of a join are both indented
// under it, the left one first.
//
// A nil plan is the empty string rather than an error, since this is for
// reading and a caller who is printing a plan they have not built yet is better
// served by nothing than by a panic.
func (n *Node) Tree() string {
	var sb strings.Builder
	writeTree(&sb, n, 0)
	return sb.String()
}

// writeTree is [Node.Tree] for one operator and everything under it. The
// newline goes before the line rather than after it so that the text does not
// end in one, which is what a caller comparing two plans wants.
func writeTree(sb *strings.Builder, n *Node, depth int) {
	if n == nil {
		return
	}

	if depth > 0 {
		sb.WriteByte('\n')
		sb.WriteString(strings.Repeat("  ", depth))
	}
	sb.WriteString(n.String())

	writeTree(sb, n.l, depth+1)
	writeTree(sb, n.r, depth+1)
}

// Explain returns the query as written, the query that will run, and the passes
// that made the difference between the two.
//
// This is a shipping feature rather than a debugging one. A pushdown is worth
// having because it reads two columns instead of forty, and a caller who cannot
// see whether it happened has to take that on trust. So the format is part of
// what this package promises: three sections, each headed by a line of plain
// English with no punctuation, the plans indented under their heading by two
// spaces, and a blank line between sections. A query no pass changes has two
// sections rather than three, because printing the same plan twice is not
// telling anybody anything.
//
// It reads:
//
//	the query as written
//	  Limit 20
//	    Project symbol, price
//	      Filter (price > 100)
//	        Sort by qty desc
//	          Scan trades/*.parquet
//
//	the query that runs
//	  Project symbol, price
//	    Limit 20
//	      Sort by qty desc
//	        Filter (price > 100)
//	          Scan trades/*.parquet
//
//	changed by predicate pushdown and slice pushdown
//
// The plan is checked before the passes run, so a query with a mistake in it
// reports the mistake as it was written rather than as some pass left it, the
// same way [github.com/tamnd/kuma.LazyFrame.Collect] does. A pass is named when
// it changed the plan at least once, which is not quite the same as it having
// made the difference you are looking at, since the passes run over each other
// until they settle and one of them can undo what another did.
//
// What is not in it yet is a row count per operator, which needs statistics on
// the source, and a wall time per operator, which needs the query to have run
// and is a profile rather than an explain.
func Explain(n *Node, passes ...Pass) (string, error) {
	if n == nil {
		return "", errNoPlan
	}
	if _, err := n.Schema(); err != nil {
		return "", err
	}

	out, changed, err := optimize(n, passes)
	if err != nil {
		return "", err
	}

	var sb strings.Builder
	sb.WriteString("the query as written\n")
	sb.WriteString(indented(n.Tree()))
	if len(changed) == 0 {
		sb.WriteString("\n\nnothing the optimizer does changes it\n")
		return sb.String(), nil
	}

	sb.WriteString("\n\nthe query that runs\n")
	sb.WriteString(indented(out.Tree()))
	sb.WriteString("\n\nchanged by ")
	sb.WriteString(listed(changed))
	sb.WriteByte('\n')
	return sb.String(), nil
}

// indented moves every line of the text in by two spaces, which is how a plan
// is set under the heading that says which plan it is.
func indented(s string) string {
	return "  " + strings.ReplaceAll(s, "\n", "\n  ")
}

// listed joins the names the way a sentence would rather than the way a list
// would, since the line it goes on is read and not parsed.
func listed(names []string) string {
	switch len(names) {
	case 0:
		return ""
	case 1:
		return names[0]
	case 2:
		return names[0] + " and " + names[1]
	}
	return strings.Join(names[:len(names)-1], ", ") + " and " + names[len(names)-1]
}
