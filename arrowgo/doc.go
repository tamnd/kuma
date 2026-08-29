// Package arrowgo converts columns between kuma and arrow-go.
//
// It exists so that being incomplete is survivable. Anything kuma cannot do
// yet can be handed to a library that can, and the answer handed back, without
// serializing anything and in most cases without copying anything either. The
// C Data Interface in the ipc package does the same job for other processes and
// other languages; this does it for Go code in the same process, where going
// through C would be a pointless round trip through cgo.
//
// # The boundary rule
//
// This converts at the boundary and nothing else. It does not call arrow-go's
// compute kernels and it never will. Those kernels are the thing kuma exists to
// provide, so building on them would make the project a wrapper around the
// library it is meant to replace, and their coverage is thin enough that it
// would be a wrapper that is blocked constantly. Document 02 says so in the
// list of decisions already made, and this package is where that decision is
// enforced rather than restated.
//
// # Why it is a separate module
//
// The kuma module depends on the standard library and nothing else, and that is
// worth keeping. A program that wants the bridge asks for it:
//
//	go get github.com/tamnd/kuma/arrowgo
//
// and a program that does not gets no arrow-go in its build and nothing in its
// go.sum. The cost is one more module to tag, which is paid once.
//
// # What is shared and what is copied
//
// Import means arrow-go to kuma and Export means kuma to arrow-go, which is the
// same naming the C Data Interface uses in package ipc.
//
// The layouts are the same on both sides, so a fixed width column, a validity
// bitmap and a string view column all cross by handing over the bytes. Nothing
// is copied and nothing is scanned, whatever the length of the column, and
// ImportArray on a column of a hundred million float64 values is a handful of
// allocations for the wrappers. The tests check this by comparing the addresses
// of the underlying bytes rather than by comparing values, since equal values
// are what a copy gives you too.
//
// Two shapes do not line up and are converted, which copies:
//
//   - An offset layout string or binary column, which is what arrow-go builds
//     by default, becomes the view layout kuma stores. Building the arrow-go
//     side with a StringViewBuilder or a BinaryViewBuilder avoids this.
//   - A large_string or large_binary column becomes the same view layout, for
//     the reason [array.NewStrings] gives: a view has no global offset to
//     overflow, so nothing is lost.
//
// Going the other way, [ExportArray] writes the view layout, which arrow-go
// reads as arrow.STRING_VIEW and arrow.BINARY_VIEW. Anything that insists on
// offsets has to be converted on the arrow-go side.
//
// # Memory
//
// kuma has no reference counting, so nothing here retains or releases. An
// arrow-go array that has been imported must be kept alive by the caller for as
// long as the kuma array is used, since arrow-go will hand its buffers back to
// its allocator when the last release lands and the kuma array is still
// pointing at them. With the default Go allocator this cannot bite, because the
// buffers are ordinary Go memory that the collector keeps for as long as
// anything refers to it. With a CGO allocator or a checked allocator it can, so
// hold the record until you are done with the column.
//
// Exported arrays are the safe direction. They are backed by Go memory that
// kuma owns and the collector keeps, so releasing one is a no-op that frees
// nothing early.
//
// # What does not cross
//
// The nested types, meaning list, struct, map and their variants, along with
// the union types and run end encoding. kuma's array package does not hold them
// yet, so there is nothing to convert them into. They are reported by name
// rather than skipped, so a record holding one fails at the boundary instead of
// arriving with a column quietly missing.
//
// Stability: tier 2. The conversions are settled, the surface may still grow.
package arrowgo
