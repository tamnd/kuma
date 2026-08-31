package plan

// Printing a plan that has already run, with what each operator cost written
// next to it. This is [Explain] with the numbers on, and the numbers are the
// half of the question an explain cannot answer: the plan says the filter runs
// before the join, and the profile says whether that was where the time went.

import (
	"strconv"
	"strings"
	"time"
)

// A Measure is one operator of a plan that has run, and what running it cost.
//
// It is a tree rather than a map from node to cost because the same node can
// stand in a plan twice, which is what a self join is, and two runs of it are
// two different costs. The shape of a Measure is the shape of the plan it came
// from, so [Measure.Input] is in the order the operator reads its inputs and a
// join has two of them, the left one first.
//
// Took is this operator and everything under it, which is the thing a clock can
// actually be read for. The time an operator spent on its own is that less the
// same for its inputs, and it is what [Profile] prints, since an operator that
// is slow because its input is slow is not the operator to look at.
type Measure struct {
	// Node is the operator this was measured over.
	Node *Node

	// Took is how long the operator and everything under it took.
	Took time.Duration

	// Rows is how many rows the operator produced. How many it read is the
	// same number on the operators under it, which is why it is not here
	// twice.
	Rows int64

	// Input is the same for the operators this one reads, in the order it
	// reads them.
	Input []Measure
}

// own is the time the operator spent that was not spent by its inputs.
//
// It is clamped at zero. A clock read either side of a call that took no time
// at all can come back in the wrong order on some machines, and a profile that
// says an operator took less than nothing is a profile nobody believes the rest
// of.
func (m Measure) own() time.Duration {
	own := m.Took
	for _, in := range m.Input {
		own -= in.Took
	}
	if own < 0 {
		return 0
	}
	return own
}

// Profile is [Explain] for a query that has run, with the time each operator
// spent and the rows it produced written next to it.
//
// The plan is the query as it was written and the measure is the query that
// ran, which is what the passes turned the first into. It reads:
//
//	the query as written
//	  Limit 20
//	    Project symbol, price
//	      Filter (price > 100)
//	        Scan trades/*.parquet
//
//	the query that ran
//	  Project symbol, price                         10.0us   2 rows
//	    Limit 20                                    10.0us   2 rows
//	      Filter (price > 100)                      50.0us   3 rows
//	        Scan trades/*.parquet [symbol, price]   30.0us   4 rows
//
//	changed by slice pushdown and projection pushdown
//	ran in 100us
//
// The time on a line is what that operator spent and not what its inputs spent,
// so the lines add up to the total on the last one and the largest of them is
// the operator to go and look at. The rows on a line are the rows it produced,
// which are also the rows the operator above it read, so a line and the line
// under it are a row count in and a row count out and there is no need to print
// either twice. A join has two lines under it and reads both.
//
// A query no pass changed is printed once, the same rule [Explain] follows, and
// the heading says it ran rather than that it was written.
//
// What is not in it yet is the bytes a scan read, which the sources do not
// report. It goes on the scan line when they do.
func Profile(n *Node, ran Measure, passes ...Pass) (string, error) {
	if n == nil {
		return "", errNoPlan
	}
	if ran.Node == nil {
		return "", errNoRun
	}
	if _, err := n.Schema(); err != nil {
		return "", err
	}

	_, changed, err := optimize(n, passes)
	if err != nil {
		return "", err
	}

	var sb strings.Builder
	if len(changed) > 0 {
		sb.WriteString("the query as written\n")
		sb.WriteString(indented(n.Tree()))
		sb.WriteString("\n\n")
	}

	sb.WriteString("the query that ran\n")
	sb.WriteString(indented(measureTree(ran)))
	sb.WriteString("\n\n")

	if len(changed) == 0 {
		sb.WriteString("nothing the optimizer does changes it\n")
	} else {
		sb.WriteString("changed by ")
		sb.WriteString(listed(changed))
		sb.WriteByte('\n')
	}
	sb.WriteString("ran in ")
	sb.WriteString(took(ran.Took))
	sb.WriteByte('\n')
	return sb.String(), nil
}

// measureTree is [Node.Tree] with the numbers on, which means the lines have to
// be built before any of them can be written: the columns line up against the
// longest operator and the longest time, and neither is known until the last
// line is in hand.
func measureTree(m Measure) string {
	var rows []profileRow
	collect(&rows, m, 0)

	var wide, clock int
	for _, r := range rows {
		wide = max(wide, len(r.op))
		clock = max(clock, len(r.took))
	}

	var sb strings.Builder
	for i, r := range rows {
		if i > 0 {
			sb.WriteByte('\n')
		}
		sb.WriteString(r.op)
		sb.WriteString(strings.Repeat(" ", wide-len(r.op)+3))
		sb.WriteString(strings.Repeat(" ", clock-len(r.took)))
		sb.WriteString(r.took)
		sb.WriteString("   ")
		sb.WriteString(r.rows)
	}
	return sb.String()
}

// A profileRow is one line of a profile before the columns are lined up.
type profileRow struct {
	op   string
	took string
	rows string
}

// collect walks the measures the way [Node.Tree] walks a plan, the operator
// first and its inputs indented under it.
func collect(rows *[]profileRow, m Measure, depth int) {
	*rows = append(*rows, profileRow{
		op:   strings.Repeat("  ", depth) + m.Node.String(),
		took: took(m.own()),
		rows: count(m.Rows),
	})
	for _, in := range m.Input {
		collect(rows, in, depth+1)
	}
}

// took is a duration as this package prints one, which is three figures and a
// unit. It is not [time.Duration.String], which writes a microsecond with the
// micro sign, and everything printed here is ASCII.
func took(d time.Duration) string {
	switch {
	case d < time.Microsecond:
		return strconv.FormatInt(int64(d), 10) + "ns"
	case d < time.Millisecond:
		return figures(float64(d)/float64(time.Microsecond)) + "us"
	case d < time.Second:
		return figures(float64(d)/float64(time.Millisecond)) + "ms"
	default:
		return figures(d.Seconds()) + "s"
	}
}

// figures is a number to three of them, which is as much as a wall time read
// once is worth writing down.
func figures(v float64) string {
	switch {
	case v < 10:
		return strconv.FormatFloat(v, 'f', 2, 64)
	case v < 100:
		return strconv.FormatFloat(v, 'f', 1, 64)
	default:
		return strconv.FormatFloat(v, 'f', 0, 64)
	}
}

// count is a row count with the word after it, which is "row" once and "rows"
// every other time, including none.
func count(n int64) string {
	if n == 1 {
		return "1 row"
	}
	return strconv.FormatInt(n, 10) + " rows"
}
