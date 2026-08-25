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
// and predicates all come back here in the end.
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
