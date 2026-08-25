// Package quotes_test looks like an external test package, which an ordinary
// generated file cannot be part of.
package quotes_test

// Quote is a fine struct in a package that cannot hold the file kumagen would
// write next to it.
type Quote struct {
	Bid float64 `kuma:"bid"`
}
