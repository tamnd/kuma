package ipc_test

import (
	"errors"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

// TestTypeLayouts covers the format strings kuma never writes and every other
// Arrow implementation does. The offset based text and byte layouts are the
// ones that actually arrive: pyarrow writes "u" for a string column unless it
// is asked for views.
func TestTypeLayouts(t *testing.T) {
	tests := []struct {
		format string
		want   dtype.DataType
	}{
		{"u", dtype.String},
		{"U", dtype.String},
		{"vu", dtype.String},
		{"z", dtype.Binary},
		{"Z", dtype.Binary},
		{"vz", dtype.Binary},

		// The short and the long spelling of a 128 bit decimal are the same
		// type. Both are in the wild.
		{"d:18,2", dtype.Decimal128{Precision: 18, Scale: 2}},
		{"d:18,2,128", dtype.Decimal128{Precision: 18, Scale: 2}},
	}

	for _, tt := range tests {
		got, err := ipc.Type(tt.format, nil)
		if err != nil {
			t.Errorf("Type(%q) = %v", tt.format, err)
			continue
		}
		if !dtype.Equal(got, tt.want) {
			t.Errorf("Type(%q) = %s, want %s", tt.format, got, tt.want)
		}
	}
}

func TestTypeErrors(t *testing.T) {
	item := []dtype.Field{{Name: "item", Type: dtype.Int64, Nullable: true}}

	// want is the sentinel the call has to report, or nil for the cases that
	// fail validation, since the rules those break belong to dtype and have no
	// sentinel of their own.
	tests := []struct {
		name     string
		format   string
		children []dtype.Field
		want     error
	}{
		{name: "empty", format: "", want: ipc.ErrFormat},
		{name: "not a format string", format: "int64", want: ipc.ErrFormat},
		{name: "unknown letter", format: "q", want: ipc.ErrFormat},
		{name: "case matters", format: "V", want: ipc.ErrFormat},

		// Arrow types with no kuma equivalent. These are rejected rather than
		// approximated, since silently reading a union as its first branch
		// would lose data and say nothing.
		{name: "float16", format: "e", want: ipc.ErrFormat},
		{name: "dense union", format: "+ud:0,1", want: ipc.ErrFormat},
		{name: "sparse union", format: "+us:0,1", want: ipc.ErrFormat},
		{name: "run end encoded", format: "+r", want: ipc.ErrFormat},
		{name: "list view", format: "+vl", want: ipc.ErrFormat},
		{name: "large list view", format: "+vL", want: ipc.ErrFormat},

		{name: "timestamp with no colon", format: "tsu", want: ipc.ErrFormat},
		{name: "timestamp with no unit", format: "ts:UTC", want: ipc.ErrFormat},
		{name: "timestamp with a bad unit", format: "tsq:UTC", want: ipc.ErrFormat},
		{name: "timestamp with a long unit", format: "tsus:UTC", want: ipc.ErrFormat},
		{name: "time with a bad unit", format: "ttq", want: ipc.ErrFormat},

		{name: "decimal with one number", format: "d:18", want: ipc.ErrFormat},
		{name: "decimal with four", format: "d:18,2,128,0", want: ipc.ErrFormat},
		{name: "decimal with a word", format: "d:eighteen,2", want: ipc.ErrFormat},
		{name: "decimal with a word for a scale", format: "d:18,two", want: ipc.ErrFormat},
		{name: "decimal at an odd width", format: "d:18,2,64", want: ipc.ErrFormat},
		{name: "decimal with no numbers", format: "d:", want: ipc.ErrFormat},
		{name: "binary with no width", format: "w:", want: ipc.ErrFormat},
		{name: "binary with a word", format: "w:wide", want: ipc.ErrFormat},
		{name: "binary too wide for an int32", format: "w:4294967296", want: ipc.ErrFormat},
		{name: "list with no length", format: "+w:", children: item, want: ipc.ErrFormat},

		// The count of children is part of what the format string means, so a
		// list arriving with two of them is a disagreement worth stopping for
		// rather than a first child worth guessing at.
		{name: "list with no child", format: "+l", want: ipc.ErrChildren},
		{name: "list with two children", format: "+l", children: []dtype.Field{
			{Name: "item", Type: dtype.Int64, Nullable: true},
			{Name: "second", Type: dtype.Int64, Nullable: true},
		}, want: ipc.ErrChildren},
		{name: "large list with no child", format: "+L", want: ipc.ErrChildren},
		{name: "fixed size list with no child", format: "+w:3", want: ipc.ErrChildren},
		{name: "map with no child", format: "+m", want: ipc.ErrChildren},
		{name: "map of a leaf", format: "+m", children: item, want: ipc.ErrChildren},
		{
			name:     "map of a one field struct",
			format:   "+m",
			children: []dtype.Field{{Name: "entries", Type: dtype.Struct{Fields: []dtype.Field{{Name: "key", Type: dtype.String}}}}},
			want:     ipc.ErrChildren,
		},

		// Validation is part of the answer. A precision of ninety nine parses
		// and is not a decimal128, and finding that out here is cheaper than
		// finding it out in a kernel.
		{name: "decimal too precise", format: "d:99,2"},
		{name: "decimal with too much scale", format: "d:4,9"},
		{name: "negative binary width", format: "w:-1"},
		{name: "negative list length", format: "+w:-1", children: item},
		{name: "struct with an unnamed field", format: "+s", children: []dtype.Field{{Type: dtype.Int64}}},
		{name: "struct with two of one name", format: "+s", children: []dtype.Field{
			{Name: "a", Type: dtype.Int64},
			{Name: "a", Type: dtype.Int64},
		}},
		{name: "list of nothing", format: "+l", children: []dtype.Field{{Name: "item"}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ipc.Type(tt.format, tt.children)
			if err == nil {
				t.Fatalf("Type(%q) = %s, want an error", tt.format, got)
			}
			if got != nil {
				t.Errorf("Type(%q) = %s with an error, want nil", tt.format, got)
			}
			if tt.want != nil && !errors.Is(err, tt.want) {
				t.Errorf("Type(%q) = %v, want %v", tt.format, err, tt.want)
			}
		})
	}
}

// TestTypeIgnoresSpareChildren pins down the leaf case. A schema arriving with
// children on an int64 is odd, and refusing it would mean rejecting a whole
// table over a field nothing reads.
func TestTypeIgnoresSpareChildren(t *testing.T) {
	got, err := ipc.Type("l", []dtype.Field{{Name: "item", Type: dtype.Int64}})
	if err != nil {
		t.Fatalf("Type = %v", err)
	}
	if !dtype.Equal(got, dtype.Int64) {
		t.Errorf("Type = %s, want int64", got)
	}
}

// TestTypeCopiesChildren checks that a struct type does not share the slice it
// was built from. A type is immutable, and one that changes when the caller
// reuses a scratch buffer is not.
func TestTypeCopiesChildren(t *testing.T) {
	children := []dtype.Field{
		{Name: "a", Type: dtype.Int64},
		{Name: "b", Type: dtype.String, Nullable: true},
	}
	got, err := ipc.Type("+s", children)
	if err != nil {
		t.Fatalf("Type = %v", err)
	}
	children[0] = dtype.Field{Name: "c", Type: dtype.Float64}

	want := dtype.Struct{Fields: []dtype.Field{
		{Name: "a", Type: dtype.Int64},
		{Name: "b", Type: dtype.String, Nullable: true},
	}}
	if !dtype.Equal(got, want) {
		t.Errorf("Type = %s, want %s", got, want)
	}
}
