// Package bad holds a struct for each reason kumagen has to refuse one. It is
// under testdata because it does not compile, on purpose: Twice is declared in
// two files, which is what the tests want to see reported.
package bad

// Narrow has a field of a width there is no handle for yet.
type Narrow struct {
	Qty int32 `kuma:"qty"`
}

// Nested has a field that is another struct.
type Nested struct {
	Inner Narrow `kuma:"inner"`
}

// Pointed has a field that is a pointer, which no column holds.
type Pointed struct {
	Price *float64 `kuma:"price"`
}

// Generic has type parameters, and a schema does not.
type Generic[T any] struct {
	Value T `kuma:"value"`
}

// NotAStruct is a named slice.
type NotAStruct []int

// Hidden has nothing kumagen is allowed to name.
type Hidden struct {
	Venue string `kuma:"-"`
	note  string
}

// Embedded embeds a type that is not a column.
type Embedded struct {
	Generic[int]
}

// Twice is also declared in twice.go.
type Twice struct {
	Qty int64 `kuma:"qty"`
}

// Pair is a type with two parameters, for EmbeddedPair to embed.
type Pair[A, B any] struct {
	Left  A
	Right B
}

// EmbeddedPair embeds a type with two type arguments.
type EmbeddedPair struct {
	Pair[int64, string]
}

// EmbeddedPointer embeds a pointer, which is named after the type it points at.
type EmbeddedPointer struct {
	*Narrow
}
