package array

import (
	"testing"

	"github.com/tamnd/kuma/dtype"
)

// The tests here are inside the package because they are about the table that
// maps a dtype to the way its values are laid out. Nothing outside can name a
// layout, and the table is what decides whether reading a column as a Go type
// is allowed at all, so it is worth checking directly rather than through the
// panic messages it produces.

func TestLayoutString(t *testing.T) {
	tests := []struct {
		p    layout
		want string
	}{
		{layoutInt8, "int8"},
		{layoutInt16, "int16"},
		{layoutInt32, "int32"},
		{layoutInt64, "int64"},
		{layoutUint8, "uint8"},
		{layoutUint16, "uint16"},
		{layoutUint32, "uint32"},
		{layoutUint64, "uint64"},
		{layoutFloat32, "float32"},
		{layoutFloat64, "float64"},
		{layoutNone, "no Go type"},
	}

	for _, tt := range tests {
		if got := tt.p.String(); got != tt.want {
			t.Errorf("layout(%d).String() = %q, want %q", tt.p, got, tt.want)
		}
	}
}

// TestGoLayout checks every type the Numeric constraint allows, since a type
// missing from that switch is a column that cannot be read at all.
func TestGoLayout(t *testing.T) {
	tests := []struct {
		got  layout
		want layout
	}{
		{goLayout[int8](), layoutInt8},
		{goLayout[int16](), layoutInt16},
		{goLayout[int32](), layoutInt32},
		{goLayout[int64](), layoutInt64},
		{goLayout[uint8](), layoutUint8},
		{goLayout[uint16](), layoutUint16},
		{goLayout[uint32](), layoutUint32},
		{goLayout[uint64](), layoutUint64},
		{goLayout[float32](), layoutFloat32},
		{goLayout[float64](), layoutFloat64},
	}

	for _, tt := range tests {
		if tt.got != tt.want {
			t.Errorf("goLayout gave %s, want %s", tt.got, tt.want)
		}
	}
}

// TestLayoutDType checks that the two directions agree. A dtype that Of picks
// for a Go type has to be one that Values will then hand back as that same Go
// type, or Of builds columns nothing can read.
func TestLayoutDType(t *testing.T) {
	for _, p := range []layout{
		layoutInt8, layoutInt16, layoutInt32, layoutInt64,
		layoutUint8, layoutUint16, layoutUint32, layoutUint64,
		layoutFloat32, layoutFloat64,
	} {
		dt := layoutDType(p)
		got, ok := dtypeLayout(dt)
		if !ok {
			t.Errorf("layoutDType(%s) = %s, which has no layout", p, dt)
			continue
		}
		if got != p {
			t.Errorf("layoutDType(%s) = %s, which reads back as %s", p, dt, got)
		}
		if bits, _ := dtype.Bits(dt); bits/8 != byteWidth(dt) {
			t.Errorf("%s is %d bits and %d bytes", dt, bits, byteWidth(dt))
		}
	}
}

func TestLayoutDTypeUnknown(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("layoutDType(layoutNone) did not panic")
		}
	}()
	layoutDType(layoutNone)
}

// TestByteWidthPanics covers the two types that have no answer. Neither is
// reachable through the exported API, since New refuses both before it gets
// this far, but byteWidth is what Clone sizes its copy with and a wrong answer
// there is a copy of the wrong length.
func TestByteWidthPanics(t *testing.T) {
	tests := []struct {
		name string
		dt   dtype.DataType
	}{
		{"variable width", dtype.String},
		{"sub byte", nibble{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("byteWidth(%s) did not panic", tt.dt)
				}
			}()
			byteWidth(tt.dt)
		})
	}
}

// nibble claims to be four bits wide, which no dtype in kuma is. Bool is the
// only type narrower than a byte and everything handles it before it reaches
// the checks this reaches, but dtype.DataType is an interface and this is what
// someone implementing it themselves would run into.
type nibble struct{}

func (nibble) Kind() dtype.Kind { return dtype.InvalidKind }
func (nibble) String() string   { return "nibble" }
func (nibble) Bits() int        { return 4 }
