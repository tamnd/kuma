package ipc_test

import (
	"bytes"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

// FuzzType checks that reading a format string and writing it back gives the
// same string, for every string that reads as a type at all.
//
// The pair is not quite an inverse, since three text layouts read as one kuma
// type, so the property is stated one step along: whatever Type accepts,
// Format writes, and reading that gives the same type again. A format string
// that survives one trip and not two would be one this package can import and
// then fail to export, which is the bug worth catching.
//
// Format strings are short and structured, so the fuzzer finds the interesting
// ones quickly: a decimal with a comma in the wrong place, a timestamp with a
// zone that is itself a colon, a width that overflows an int32.
func FuzzType(f *testing.F) {
	for _, tt := range mappings {
		f.Add(tt.format)
	}
	f.Add("u")
	f.Add("d:18,2,128")
	f.Add("ts")
	f.Add("+w:")
	f.Add("")

	f.Fuzz(func(t *testing.T, format string) {
		// No children, so the format strings that need one fail here. The
		// nested types are covered by the table, and what is being fuzzed is
		// the parsing of the string rather than the assembly of a tree.
		first, err := ipc.Type(format, nil)
		if err != nil {
			return
		}

		written, err := ipc.Format(first)
		if err != nil {
			t.Fatalf("Type(%q) = %s, which Format refuses: %v", format, first, err)
		}
		second, err := ipc.Type(written, nil)
		if err != nil {
			t.Fatalf("Type(%q) = %s, written as %q, which Type refuses: %v",
				format, first, written, err)
		}
		if !dtype.Equal(first, second) {
			t.Fatalf("Type(%q) = %s, written as %q, which reads as %s",
				format, first, written, second)
		}
		if again, err := ipc.Format(second); err != nil || again != written {
			t.Fatalf("Format(%s) = %q, %v, want %q", second, again, err, written)
		}
	})
}

// FuzzMetadata checks that a blob this package accepts is one it writes back
// byte for byte.
//
// The blob is the one thing here that arrives as untrusted bytes from another
// process, so the first thing being asked is whether a truncated or a hostile
// one is refused rather than read past. The second is that the encoding is
// canonical, since a decoder that accepts two spellings of the same metadata
// would round trip through kuma and come out different.
func FuzzMetadata(f *testing.F) {
	seeds := []dtype.Metadata{
		nil,
		{{Key: "unit", Value: "metres"}},
		{{Key: "", Value: ""}},
		{{Key: "a", Value: "1"}, {Key: "a", Value: "2"}},
	}
	for _, m := range seeds {
		b, err := ipc.EncodeMetadata(m)
		if err != nil {
			f.Fatalf("EncodeMetadata = %v", err)
		}
		f.Add(b)
	}
	f.Add([]byte{1, 0, 0, 0})
	f.Add([]byte{255, 255, 255, 255})

	f.Fuzz(func(t *testing.T, blob []byte) {
		m, err := ipc.DecodeMetadata(blob)
		if err != nil {
			return
		}

		written, err := ipc.EncodeMetadata(m)
		if err != nil {
			t.Fatalf("DecodeMetadata(% x) = %v, which EncodeMetadata refuses: %v", blob, m, err)
		}
		if len(m) > 0 && !bytes.Equal(blob, written) {
			t.Fatalf("DecodeMetadata(% x) = %v, written back as % x", blob, m, written)
		}

		again, err := ipc.DecodeMetadata(written)
		if err != nil {
			t.Fatalf("EncodeMetadata(%v) = % x, which DecodeMetadata refuses: %v", m, written, err)
		}
		if !again.Equal(m) {
			t.Fatalf("round trip of %v gave %v", m, again)
		}
	})
}
