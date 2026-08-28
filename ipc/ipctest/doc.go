// Package ipctest is an Arrow C data interface producer that is not kuma.
//
// A test of the export side needs a pair of C structs to export into, and a
// test of the import side needs buffers that kuma did not allocate and a
// release callback that kuma does not control. This package is both. It
// allocates the structs with malloc, fills them from Go byte slices, and counts
// the times a consumer hands one back, which is the part of the contract that
// nothing on the Go side of the boundary can check.
//
// It exists as its own package because a Go test file cannot use cgo, so the C
// side of a test of that interface has to live somewhere else. It is public for
// the same reason kumatest is: anyone writing a producer or a consumer of their
// own needs the same two or three things, and writing them again is how two
// implementations end up testing different contracts.
//
// Everything here needs cgo. Built without it, the package is empty, so that a
// program that cross compiles for a target with no C toolchain still builds
// every package in the module.
//
// Stability: tier 2, evolving. It hands out raw C pointers, and what those
// mean is settled by the specification rather than by this package, so the
// helpers around them are free to change shape in a minor release.
package ipctest
