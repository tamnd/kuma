// Package kernel is the compute layer, where a column turns into another
// column.
//
// Everything here works on arrays rather than on frames. A kernel is handed the
// values, does one job over all of them, and hands back new values, so the
// layer above can decide what a row is and this layer can concentrate on doing
// the same thing a million times without stopping to ask.
//
// # The rules
//
// SIMD is an optimization and never a dependency. Everything in this package
// has a plain Go implementation that is built into every binary, on every
// platform, with no build tag. A vectorized version, when it arrives, is
// checked against the plain one, and when the two disagree the plain one is
// right by definition.
//
// No exported signature names a type from the simd packages. Kernels take and
// return arrays and slices. That is what keeps an experimental package that has
// already broken once from being something a caller of this library has to know
// about.
//
// A kernel panics for a mistake about types and lengths, the same way indexing
// a slice panics, since handing a gather a column it cannot read is a bug in
// the program rather than something the data did. Nothing here returns an error
// yet, and if something does it will be for a condition the data can cause.
//
// # What is here
//
// [Take], which reads values out of a column at the positions given, and
// [Filter], which keeps the values a boolean mask selects. Everything that
// reorders or drops rows goes through one of those two, so joins, sorts, limits
// and predicates all come back here in the end. A gather over a dictionary
// encoded column moves the indices and leaves the values where they are, and a
// gather over a list column moves the elements in one pass rather than taking
// the rows apart and putting them back together one at a time, so the two
// encodings that hold more than one thing per row cost about what the values
// underneath them cost.
//
// [Cast], which turns the values of a column into another type, [Fits], which
// is the range check the cast makes asked about one value before there is a
// column to put it in, and [SortIndex], which works out the order rows go in
// and leaves the moving to Take.
//
// [Compare], which is the six comparisons, [Arith], which is the five
// arithmetic operators, and [And], [Or] and [Not], which are the three valued
// logic that a column with holes in it needs. These are the three that read two
// columns at once, so a column of one value on either side is that value
// against every row of the other, which is how a comparison against a literal
// is written without building a column of copies of it.
//
// [GroupBy], which divides rows up by the values of some key columns, and the
// aggregations that run over what it produces: [Sum], [Mean], [Count], [Size],
// [Min], [Max], [First], [Last], [Var], [Std], [Median], [Quantile] and
// [NUnique]. An aggregation over a whole column is an aggregation over
// [OneGroup], so there is one of each rather than two.
//
// [DistinctIndex], which is the same division of the rows with the answer a
// drop duplicates wants taken out of it, which is the first row of each set of
// equal ones. Like a sort it returns positions and leaves the moving to Take.
//
// Beside the kernels that can turn a column away are the rules that say the
// same thing about a type on its own: [ArithType], [CompareType],
// [IsCondition], [HasOrder], [GroupKeyType], [SumType], [MeanType],
// [MinMaxType], [VarType], [StdType], [MedianType], [QuantileType] and
// [NUniqueType]. They are here so that a query can be checked before it is run,
// against nothing but the types of the columns, and each of them is either the
// rule its kernel uses or a copy held to it by a test.
//
// [Join], which works out which rows of two tables go together, in all seven of
// the ways SQL can. Like a sort it returns positions rather than a table, so
// building the result is Take's job and a caller who only wants to know what
// matched does not pay to build one.
//
// [IsNull] and [IsNotNull], which turn what is missing into a boolean column,
// [FillNull], which puts a value where nothing was, and [KeepIndex], which is
// the positions of the rows that have enough of their values to be worth
// keeping. The first two are a copy of a bitmap and the last returns positions,
// so the only one of the three that writes a value per row is the fill.
//
// These are the reference implementations and they are not the fast ones. They
// append a value at a time, which is the version that is obviously right when
// read next to the definition of what a gather is. The one that writes a run of
// values into a buffer in one go, and the vectorized one after that, are both
// checked against what is here.
//
// Stability: tier 2, evolving.
//
// Document 11 has this package at tier 3, on the grounds that it will one day
// be full of build tagged files calling an unstable package. The exported
// surface never names any of that, which is the whole point of the second rule
// above, so the churn stays inside the package and the tier says what a caller
// can rely on rather than what the implementation is made of.
package kernel
