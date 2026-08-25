package kumatest

import (
	"strings"
	"unicode/utf8"
)

// table lays a few columns of text out under a header, the way a printed frame
// lays its values out, so that a difference and the frame it came from read the
// same way.
//
// Widths are counted in runes, which is right for the text a name and a value
// normally are and wrong for the scripts where one character takes two cells on
// a screen. That is the same trade the printer makes, and for the same reason.
type table struct {
	head  []string
	right []bool
	rows  [][]string
}

// newTable starts a table with these headings. The first column is a row
// number and is the one that reads better against the right hand edge, since
// the numbers under it line up. The rest hold a name or a value each and are
// left where a reader's eye already is.
func newTable(head ...string) *table {
	t := &table{head: head, right: make([]bool, len(head))}
	t.right[0] = true
	return t
}

// add puts a line at the bottom. It takes one cell per heading.
func (t *table) add(cells ...string) {
	t.rows = append(t.rows, cells)
}

// String draws the table.
func (t *table) String() string {
	width := make([]int, len(t.head))
	for i, h := range t.head {
		width[i] = utf8.RuneCountInString(h)
	}
	for _, r := range t.rows {
		for i, c := range r {
			if w := utf8.RuneCountInString(c); w > width[i] {
				width[i] = w
			}
		}
	}

	var sb strings.Builder
	t.line(&sb, t.head, width)
	sb.WriteString("\n")
	t.rule(&sb, width)
	for _, r := range t.rows {
		sb.WriteString("\n")
		t.line(&sb, r, width)
	}
	return sb.String()
}

// line writes one line of cells, each at the width of its column.
//
// The last cell is not padded on the right. That would leave trailing
// whitespace on the line, which some diff somewhere will complain about and
// somebody will then have to explain.
func (t *table) line(sb *strings.Builder, cells []string, width []int) {
	sb.WriteString("  ")
	for i, c := range cells {
		if i > 0 {
			sb.WriteString(" | ")
		}
		pad := width[i] - utf8.RuneCountInString(c)
		if t.right[i] {
			sb.WriteString(strings.Repeat(" ", pad))
			sb.WriteString(c)
			continue
		}
		sb.WriteString(c)
		if i < len(cells)-1 {
			sb.WriteString(strings.Repeat(" ", pad))
		}
	}
}

// rule writes the line of dashes under the header, with a plus wherever there
// is a bar above it.
func (t *table) rule(sb *strings.Builder, width []int) {
	sb.WriteString("--")
	for i, w := range width {
		if i > 0 {
			sb.WriteString("-+-")
		}
		sb.WriteString(strings.Repeat("-", w))
	}
}
