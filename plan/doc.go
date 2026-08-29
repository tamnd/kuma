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
