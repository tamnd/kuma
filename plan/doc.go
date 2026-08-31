// Package plan is the logical plan a query is compiled into, and the
// expressions the plan is written in.
//
// An expression here is untyped. [Expr] is a step over columns named by string,
// which is what an optimizer pass rewrites and what a plan is serialized from.
// The typed handles in the kuma package are a layer over this one: kuma.F64Col
// carries the schema in a type parameter so that the compiler can turn away an
// expression written against the wrong frame, and what it hands to the engine
// underneath is an [Expr].
//
// A plan is a tree of [Node], one node per operator, with a scan at the leaves
// and the rows flowing up. It says what the query asks for and not how to work
// it out, so a filter over a scan is a filter over a scan whether the engine
// ends up reading the whole file or skipping most of it. Turning the first into
// the second is what the optimizer passes do, and each of them takes a plan and
// returns a plan rather than changing the one it was given.
//
// A plan is checked before it is run. [TypeOf] says what an expression comes
// out as over a given schema, or what is wrong with it, and the rules it
// follows are the kernels' own rather than a second set written out here. That
// is what lets a query be turned away while it is still being built, with the
// column that was misspelled and the cast that would fix it, rather than
// partway through the second file.
//
// [Node.Schema] is the same thing for a whole plan. Each operator asks its
// input what it holds, checks its own expressions against that, and says what
// it leaves behind, so asking the last operator checks every operator under it
// and answers what the query produces. [Node.Validate] is that walk without the
// answer, for a caller who only wants to know whether the query is right.
//
// [Optimize] is the other half of what a plan is for. It runs a list of passes
// over a plan until none of them finds anything left to do, and each [Pass] is
// a name and a rewrite that takes a plan and returns a plan. [Passes] is the
// set that runs over every query. [PushPredicate] moves each filter as far
// down as the query allows, so that a row is thrown away before the join that
// would have paired it and the sort that would have ordered it. Then
// [PushProjection] works out which columns each step of a query actually reads
// and writes that into the scan at the bottom, so that a query over two columns
// of a file of forty reads two. The predicate goes first because a filter that
// has moved down changes which columns the steps above it need.
//
// The two rules that everything else here depends on are that an expression
// never changes once it has been built, and that two expressions that say the
// same thing are the same [Expr]. Together they make finding a repeated
// subexpression a pointer comparison, which is what the optimizer wants, and
// they make an expression safe to share between goroutines, which is what an
// engine that runs a plan on more than one core wants. See intern.go for how
// that is arranged and what it costs.
//
// Stability: tier 2, evolving. The API may change in any minor release.
// Programs should use the kuma package rather than importing this directly.
package plan
