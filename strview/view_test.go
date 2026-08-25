package strview_test

import (
	"bytes"
	"math"
	"testing"

	"github.com/tamnd/kuma/strview"
)

func TestSizes(t *testing.T) {
	// These three numbers are the layout. Arrow specifies them, so changing one
	// here does not change the format, it breaks compatibility with every other
	// Arrow implementation there is.
	if strview.Size != 16 || strview.MaxInline != 12 || strview.PrefixLen != 4 {
		t.Fatalf("layout is Size %d MaxInline %d PrefixLen %d, want 16, 12 and 4",
			strview.Size, strview.MaxInline, strview.PrefixLen)
	}
	if got := len(strview.View{}); got != strview.Size {
		t.Fatalf("a View is %d bytes, want %d", got, strview.Size)
	}
}

func TestMakeInline(t *testing.T) {
	for _, want := range []string{"", "a", "kuma", "exactly12345"} {
		v := strview.MakeInline([]byte(want))
		if !v.IsInline() {
			t.Errorf("MakeInline(%q) is not inline", want)
		}
		if v.Len() != len(want) {
			t.Errorf("MakeInline(%q).Len() = %d", want, v.Len())
		}
		if got := string(v.Inline()); got != want {
			t.Errorf("MakeInline(%q).Inline() = %q", want, got)
		}

		// Everything past the value is zero, which is what lets equality be a
		// comparison of the raw sixteen bytes.
		if tail := v[4+len(want):]; !bytes.Equal(tail, make([]byte, len(tail))) {
			t.Errorf("MakeInline(%q) left %v in the padding", want, tail)
		}
	}
}

func TestMakeInlineZeroValue(t *testing.T) {
	// A slice of views that has been sized but not written reads as empty
	// values rather than as anything surprising, which matters because that is
	// what Grow leaves behind.
	var zero strview.View
	if zero != strview.MakeInline(nil) {
		t.Fatalf("the zero View is %v, want the view of an empty value", zero[:])
	}
	if zero.Len() != 0 || !zero.IsInline() || len(zero.Inline()) != 0 {
		t.Fatalf("the zero View reads as Len %d IsInline %v", zero.Len(), zero.IsInline())
	}
}

func TestMakeInlineTooLong(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("MakeInline of a 13 byte value did not panic")
		}
	}()
	strview.MakeInline([]byte("thirteen char"))
}

func TestMakeRef(t *testing.T) {
	value := []byte("a value that does not fit inline")
	v := strview.MakeRef(value, 3, 128)

	if v.IsInline() {
		t.Error("MakeRef produced an inline view")
	}
	if v.Len() != len(value) {
		t.Errorf("Len() = %d, want %d", v.Len(), len(value))
	}
	if v.Block() != 3 {
		t.Errorf("Block() = %d, want 3", v.Block())
	}
	if v.Offset() != 128 {
		t.Errorf("Offset() = %d, want 128", v.Offset())
	}
	if got, want := v.Prefix(), [4]byte{'a', ' ', 'v', 'a'}; got != want {
		t.Errorf("Prefix() = %q, want %q", got[:], want[:])
	}
}

func TestMakeRefShortValue(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("MakeRef of a value that fits inline did not panic")
		}
	}()
	strview.MakeRef([]byte("kuma"), 0, 0)
}

func TestMakeRefNegative(t *testing.T) {
	for _, c := range []struct {
		name          string
		block, offset int
	}{
		{"block", -1, 0},
		{"offset", 0, -1},
	} {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("MakeRef with a negative %s did not panic", c.name)
				}
			}()
			strview.MakeRef([]byte("a value that does not fit inline"), c.block, c.offset)
		})
	}
}

func TestMakeRefOutOfRange(t *testing.T) {
	// The layout gives the block number and the offset a signed 32 bit field
	// each. Truncating to fit would produce a view that describes a value which
	// is not there, and nothing downstream could tell.
	if math.MaxInt == math.MaxInt32 {
		t.Skip("an int on this platform cannot hold a number past MaxValue")
	}
	tooBig := int(int64(strview.MaxValue) + 1)

	for _, c := range []struct {
		name          string
		block, offset int
	}{
		{"block", tooBig, 0},
		{"offset", 0, tooBig},
	} {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("MakeRef with a %s past MaxValue did not panic", c.name)
				}
			}()
			strview.MakeRef([]byte("a value that does not fit inline"), c.block, c.offset)
		})
	}
}

func TestPrefixIsZeroPadded(t *testing.T) {
	// A short value pads its prefix with zeroes, which is what makes comparing
	// prefixes give the same answer as comparing values. "ab" sorts before
	// "abc" because its third prefix byte is zero and theirs is a c.
	ab := strview.MakeInline([]byte("ab")).Prefix()
	abc := strview.MakeInline([]byte("abc")).Prefix()
	if ab != [4]byte{'a', 'b', 0, 0} {
		t.Errorf("prefix of %q is %q", "ab", ab[:])
	}
	if bytes.Compare(ab[:], abc[:]) >= 0 {
		t.Errorf("prefix %q does not sort before %q", ab[:], abc[:])
	}
}

func TestInlineOnALongView(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("Inline on a long view did not panic")
		}
	}()
	v := strview.MakeRef([]byte("a value that does not fit inline"), 0, 0)
	_ = v.Inline()
}

func TestInlineAliasesTheView(t *testing.T) {
	// Inline hands back the view's own bytes rather than a copy, which is what
	// keeps reading a short value free. A caller that writes to the result is
	// out of contract, and this test is only here to pin down that it is an
	// alias and not a copy.
	v := strview.MakeInline([]byte("kuma"))
	v.Inline()[0] = 'p'
	if got := string(v.Inline()); got != "puma" {
		t.Errorf("Inline returned a copy: the view still reads %q", got)
	}
}

func TestString(t *testing.T) {
	for _, c := range []struct {
		view strview.View
		want string
	}{
		{strview.MakeInline([]byte("kuma")), `inline("kuma")`},
		{strview.MakeInline(nil), `inline("")`},
		{
			strview.MakeRef([]byte("a value that does not fit inline"), 2, 64),
			`ref(len 32, block 2, offset 64, prefix "a va")`,
		},
	} {
		if got := c.view.String(); got != c.want {
			t.Errorf("String() = %s, want %s", got, c.want)
		}
	}
}

// TestForeignLengths covers the views this package did not make. The length is
// a signed field on the wire, so a message from another process can claim a
// negative one, and the accessors have to report what it claims rather than
// pretend. Refusing it is NewData's job, which TestNewDataRejects covers.
func TestForeignLengths(t *testing.T) {
	var v strview.View
	v[3] = 0x80 // the sign bit of a little endian int32

	if v.Len() >= 0 {
		t.Errorf("Len() = %d for a view whose length field is negative", v.Len())
	}
	if !v.IsInline() {
		t.Error("a view claiming a negative length is not inline, so nothing would range check it")
	}
	if got, want := v.String(), `inline("")`; got != want {
		t.Errorf("String() = %s, want %s", got, want)
	}
}
