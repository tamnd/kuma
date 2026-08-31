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
// # Filtering and expressions
//
// A query is written against column handles rather than strings. A handle is a
// name and a Go type, so [F64] names a column of float64 values and the methods
// on it build a condition:
//
//	dear, err := prices.Filter(kuma.F64("price").Gt(150))
//
// The comparisons are Eq, Ne, Lt, Le, Gt and Ge, the arithmetic is Add, Sub,
// Mul, Div and Mod, and each of them has an Expr version that takes another
// expression instead of a literal, so Gt(150) compares against a number and
// GtExpr(cost) compares against a column. And, Or and Not put conditions
// together, IsNull and IsNotNull ask about the holes, and [Frame.Eval] and
// [Frame.WithExpr] work an expression out as a column rather than as a filter:
//
//	f, err := prices.WithExpr("notional", kuma.F64("price").MulExpr(kuma.F64("qty")))
//
// A literal takes the type of the column it is used with, so comparing a uint32
// column against 0 leaves it a uint32 column, and a literal that cannot be used
// with the column is an error rather than a rounding. A row where either side
// is missing gives a missing answer rather than a false, so [Frame.Filter]
// drops it: a row nobody can say belongs in the result does not go in it, and a
// condition and its negation do not add up to the frame. [Frame.FilterMask] is
// the version that takes a mask that is already worked out.
//
// [Bind] checks a frame against a Go struct and gives back the same frame with
// that struct as its schema, after which a handle written for the struct works
// on it and a handle written for anything else does not compile. [Dyn] is the
// handle for a column whose type is only known when the file is read.
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
// # Writing a query and running it later
//
// [Frame.Lazy] gives a [LazyFrame], which is a query that has been written
// down and not worked out yet:
//
//	out, err := prices.Lazy().
//		Filter(kuma.F64("price").Gt(150)).
//		SortDesc("price").
//		Head(20).
//		Collect(ctx)
//
// Nothing runs until Collect, and that is the whole point. The query is known
// before any of it happens, so a column that is not there is an error before
// the first file is opened, [LazyFrame.Schema] says what the result will hold
// without reading a row, and the optimizer passes get to see what the last step
// asked for before deciding what the first one has to read. What comes back is
// what the same query written out by hand gives, because the same kernels do
// the work.
//
// The steps are Filter, Select, With, Drop, GroupBy, Join, Sort, Head and
// Slice, and a group by and a join are written the way the eager ones are:
//
//	totals, err := prices.Lazy().GroupBy("symbol").Agg(kuma.Sum("qty")).Collect(ctx)
//	both, err := trades.Lazy().InnerJoin(sectors.Lazy(), "symbol").Collect(ctx)
//
// A distinct and an explode are what the engine is being taught next, and
// asking for one of those is an error that says so rather than a wrong answer.
// [LazyFrame.Plan] is the plan as it stands, [LazyFrame.Validate] is the check
// on its own, and printing a query prints the tree of operators it built.
//
// # Reading a file
//
// [ReadCSV] and [ReadCSVFile] read a comma separated file into a frame, working
// out what each column holds from the first thousand rows:
//
//	f, err := kuma.ReadCSVFile("trades.csv", nil)
//
// What it decides and how to say otherwise is on [csv.Options], which is where
// the delimiter, the header, the types, the values that mean nothing is there
// and the rest of it live. The frame is [Dynamic], because a file is not a Go
// type and what is in it was decided by whoever wrote it.
//
// This reads the whole file. ScanCSV, which is the lazy frame's own way in and
// is not written yet, reads a chunk at a time and never holds more than one of
// them, which is what a file larger than memory needs.
//
// [Frame.WriteCSV] and [Frame.WriteCSVFile] go the other way, and
// [csv.WriteOptions] is where the delimiter, the header, what a missing value
// looks like and how many digits a float gets live:
//
//	err := f.WriteCSVFile("out.csv", nil)
//
// A frame written and read back is the frame that went in, except that a value
// that was an empty string comes back missing, since a file cannot tell an
// empty field from an absent one. Write a null value of your own when that
// difference matters.
//
// # Looking at a frame
//
// A frame prints as a table, so fmt.Println is a real way to find out what a
// query did:
//
//	kuma.Frame[kuma.Dynamic] 3 rows x 3 cols
//
//	  sym    |   qty |      px
//	  string | int64 | float64
//	---------+-------+--------
//	  AAPL   |   100 |   182.5
//	  MSFT   |  null |   411.2
//	  GOOG   |   300 |   141.8
//
// The types are in the header and a missing value shows as null, which is not
// the same cell as an empty string and does not look like one. Ten rows and
// twelve columns are shown by default and the rest are a line of dots in the
// middle. [Frame.Render] takes a [PrintOptions] when a different amount is
// wanted, and MaxRows of -1 means the whole thing.
//
// Numbers print at the shortest text that reads back as the same number rather
// than rounded to a fixed number of digits, so two values that differ in the
// last place look different. A string that begins or ends in a space is quoted
// for the same reason, that space being invisible in a table where every cell
// is padded with spaces anyway. [Series] and [Column] print the same way, as a
// table of one column.
//
// # Missing values
//
// A missing value is a null, which is a bit in a bitmap beside the data rather
// than a value chosen out of the range of the type. That is why there is no
// NaN standing in for a missing float here and no integer column turning into a
// float column the moment a value goes missing, both of which pandas does.
//
// [Frame.IsNull] and [Frame.IsNotNull] give a frame of boolean columns saying
// where the holes are, [Column.NullMask] and [Series.ValidMask] do the same for
// one column, and a mask goes straight back into [Frame.Filter].
// [Frame.FillNull] puts a value where nothing was, [Frame.DropNulls] takes out
// the rows that are missing something, and [Frame.KeepAtLeast] is the same with
// the rule relaxed to a count. None of them looks at a value one at a time: a
// mask is a copy of the bitmap the column already carries, so it costs a byte
// per eight rows.
//
//	clean, err := prices.DropNulls("price")
//	filled, err := prices.FillNull("price", 0.0)
//
// # Types and schemas
//
// A frame carries its schema as a type parameter. [Dynamic] is the schema type
// meaning not known at compile time, which is what reading an arbitrary file
// gives you, and it is what the string based methods here are for. [Bind] is
// the way from there to a frame with a Go struct as its schema, and after it a
// handle written for another schema is a compile error rather than a wrong
// answer. The handles are [NewF64Col] and the rest, or the light [F64] and the
// rest for a Dynamic frame, and kumagen writes them out of a tagged struct so
// that a program does not have to. Document 03 has the design.
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
// # Testing
//
// A test that compares two frames should print what differs rather than both
// frames, and that is what kuma/kumatest is for. It reports the cells that are
// not the same, in the same text a printed frame would show them in, with an
// allowance for floating point values that were computed rather than typed. It
// also builds a frame of random values for a benchmark or a property test.
//
// Stability: tier 1, stable. After 1.0 this package follows the Go 1
// compatibility promise. Before 1.0 it will break whenever breaking it is the
// right call, and that is what the leading zero in the version is for.
package kuma

// Version is the current version of the library.
//
// It is a placeholder until there is something to version.
const Version = "0.0.0-dev"
