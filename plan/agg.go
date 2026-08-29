package plan

import (
	"fmt"

	"github.com/tamnd/kuma/kernel"
)

// AggFunc is what an aggregation works out about a group.
//
// The set is closed and this is the whole of it. A caller cannot add one,
// because an aggregation the kernels and the planner cannot see inside is an
// aggregation neither of them can do anything with. User defined aggregations
// will arrive as their own thing, with the cost written on the label.
type AggFunc uint8

// The aggregations, which are the ones the eager frame has.
const (
	AggSum AggFunc = iota
	AggMean
	AggMin
	AggMax
	AggCount
	AggSize
	AggFirst
	AggLast
	AggVar
	AggStd
	AggMedian
	AggQuantile
	AggNUnique
)

// String returns what the aggregation is called, which is the name a caller
// wrote and the name an error about it should use.
func (f AggFunc) String() string {
	switch f {
	case AggSum:
		return "Sum"
	case AggMean:
		return "Mean"
	case AggMin:
		return "Min"
	case AggMax:
		return "Max"
	case AggCount:
		return "Count"
	case AggSize:
		return "Size"
	case AggFirst:
		return "First"
	case AggLast:
		return "Last"
	case AggVar:
		return "Var"
	case AggStd:
		return "Std"
	case AggMedian:
		return "Median"
	case AggQuantile:
		return "Quantile"
	default:
		return "NUnique"
	}
}

// Agg is one thing to work out about each group: what to read, what to do to
// it, and what to call the answer.
//
// What it reads is an expression rather than a column, so the sum of the
// product of two columns is one aggregation over one expression rather than a
// projection followed by a sum. That is the difference between this and the
// eager [github.com/tamnd/kuma.Aggregation], which names a column because there
// is no plan there to fold the arithmetic into.
//
// The fields are exported and this is plain data, unlike [Expr] and [Node].
// Nothing here is shared between plans and there is no combination of fields
// that has to be prevented, so a pass writing one out field by field is the
// easiest thing to read.
type Agg struct {
	// Func is the aggregation to work out.
	Func AggFunc

	// Expr is what to work it out over, and is nil for an [AggSize], which
	// counts rows and reads nothing.
	Expr *Expr

	// As is what to call the result column. When it is empty the name is the
	// expression's own text, or the word size for an [AggSize].
	As string

	// DDoF is the divisor adjustment of [AggVar] and [AggStd], and is ignored
	// by every other aggregation. One is the sample variance and zero is the
	// population variance.
	DDoF int

	// Q and How are the arguments of [AggQuantile], and are ignored by every
	// other aggregation. Q is how far through the sorted values to look, and
	// How says what to do when that lands between two of them.
	Q   float64
	How kernel.Interpolation
}

// Name returns what the result column will be called.
func (a Agg) Name() string {
	switch {
	case a.As != "":
		return a.As
	case a.Func == AggSize:
		return "size"
	case a.Expr != nil:
		return a.Expr.String()
	default:
		return ""
	}
}

// String returns the aggregation as it would be written, which is what an
// explain shows and what an error about it should say.
func (a Agg) String() string {
	s := a.Func.String() + "("
	if a.Expr != nil {
		s += a.Expr.String()
	}
	switch a.Func {
	case AggVar, AggStd:
		s = fmt.Sprintf("%s, %d", s, a.DDoF)
	case AggQuantile:
		s = fmt.Sprintf("%s, %v, %s", s, a.Q, a.How)
	case AggSum, AggMean, AggMin, AggMax, AggCount, AggSize, AggFirst, AggLast,
		AggMedian, AggNUnique:
	}
	s += ")"
	if a.As != "" {
		s += " as " + a.As
	}
	return s
}
