// Package kuma is a columnar dataframe engine.
//
// This is the top of the library, and it is where the two types most programs
// use live. A [Frame] is a table: an ordered list of named columns, all of the
// same length. A [Series] is one of those columns read as a Go type, so a
// Series[float64] over a column of prices hands out a []float64 that is the
// memory itself rather than a copy of it.
//
// # Getting started
//
//	prices, err := kuma.NewFrame(
//		kuma.NewSeries("symbol", "AAPL", "MSFT", "AAPL").Column(),
//		kuma.NewSeries("price", 189.5, 411.2, 190.1).Column(),
//	)
//
//	f, err := prices.Select("price")
//	s, err := f.Series[float64]("price")
//	for _, v := range s.Values() {
//		// v is a float64 read straight out of the column.
//	}
//
// # Grouping
//
// [Frame.GroupBy] divides the rows up and hands back a [GroupedFrame], which
// holds the division so that asking it several questions costs one grouping
// rather than several:
//
//	g, err := prices.GroupBy("symbol")
//	totals, err := g.Agg(
//		kuma.Sum("qty").As("total"),
//		kuma.Mean("price").As("avg"),
//		kuma.Size(),
//	)
//
// A missing key is a group of its own rather than a row that disappears, and
// the groups come out in the order they first appear, which is deterministic
// without being sorted. Sort the result when the order matters.
//
// # Joining
//
// [Frame.Join] puts two frames together on the columns they share, in all seven
// of the ways SQL can, and [Frame.InnerJoin] and [Frame.LeftJoin] are the two
// that come up often enough to have names:
//
//	got, err := trades.InnerJoin(sectors, "symbol")
//
// A missing key matches nothing, including another missing key, which is what
// SQL says and what keeps a join from gluing together every row whose field was
// left blank. Output order is the left frame's row order, with the matches of
// one left row in the right frame's order, and an outer join puts the unmatched
// right rows at the end. Use [On] when the two sides call the key different
// things.
//
// # Stacking
//
// [Concat] puts frames on top of each other and [HStack] puts them side by
// side:
//
//	week, err := kuma.Concat(monday, tuesday, wednesday)
//
// Neither copies anything. A column is stored as a list of chunks, so stacking
// two frames puts the two lists together and the values stay where they are,
// which is why reading a directory of files and concatenating them costs about
// what reading them cost and nothing more. [ConcatUnion] is the version for
// frames that do not hold the same columns, and it is the only one of the three
// that has to build anything, being the nulls that stand in for a column a
// frame does not have.
//
// # Types and schemas
//
// A frame carries its schema as a type parameter. [Dynamic] is the schema type
// meaning not known at compile time, which is what reading an arbitrary file
// gives you, and it is what the string based methods here are for. The typed
// column handles that make a wrong column name a compile error are generated
// from a Go struct, and they are described in docs/03-typed-api.md.
//
// The Go type a column is read as is not always the type it is stored as. A
// timestamp, a duration, a date and a time of day are all int64 values with a
// meaning attached, so all of them read as an int64 without copying anything,
// and a timestamp also reads as a [time.Time]. [CanRead] is what says which
// pairings are allowed and [Value] is the set of Go types involved.
//
// # Everything is immutable
//
// No operation changes the frame it was called on. Select, Drop, Slice and the
// rest return a new frame that shares the columns it did not change, so they
// cost a slice header rather than a copy of the data, and the same frame can be
// handed to several goroutines at once.
//
// # There is no index
//
// The one pandas has is the source of most of the surprising behavior in that
// library, where two frames silently align themselves by label in the middle of
// an expression. Joins here take explicit keys and nothing aligns itself behind
// your back.
//
// # Errors
//
// A wrong column name is an error, and the error says what the frame does hold
// and which of those names is one letter away from the one that was typed. The
// sentinels are comparable with [errors.Is]. An index out of range panics, the
// way indexing a slice does, because that is a bug in the program rather than
// something the data did.
//
// Stability: tier 1, stable. After 1.0 this package follows the Go 1
// compatibility promise. Before 1.0 it will break whenever breaking it is the
// right call, and that is what the leading zero in the version is for.
package kuma

// Version is the current version of the library.
//
// It is a placeholder until there is something to version.
const Version = "0.0.0-dev"
