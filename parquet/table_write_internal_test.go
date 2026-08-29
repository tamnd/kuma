package parquet

import "testing"

// TestMustLevelsRefused asks the level encoder for something it refuses.
//
// Neither of these can happen while writing a file: the width comes from the
// schema this package just wrote and the levels come from counting the nulls of
// a column against it. So a failure there is a bug in the writer rather than
// anything a caller did, and the panic is how it is found in the test that
// caused it rather than several pages later as a file that will not open.
func TestMustLevelsRefused(t *testing.T) {
	t.Run("a width no encoding has", func(t *testing.T) {
		defer wantPanic(t)

		var e RLEEncoder
		mustLevels(&e, -1, nil)
	})

	t.Run("a level too wide for the width", func(t *testing.T) {
		defer wantPanic(t)

		var e RLEEncoder
		mustLevels(&e, 1, []int32{2})
	})
}

// TestLeavesRefused walks a schema that is not a schema.
//
// The writer walks the nodes it has just written itself, so this is the same
// kind of bug as the one above and is caught the same way. A file has at least a
// root, and a metadata with no nodes in it at all is one nothing built.
func TestLeavesRefused(t *testing.T) {
	defer wantPanic(t)

	leaves(&Metadata{})
}
