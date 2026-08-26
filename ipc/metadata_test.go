package ipc_test

import (
	"encoding/binary"
	"errors"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ipc"
)

func TestMetadataRoundTrip(t *testing.T) {
	tests := []struct {
		name string
		meta dtype.Metadata
	}{
		{"nil", nil},
		{"empty", dtype.Metadata{}},
		{"one pair", dtype.Metadata{{Key: "unit", Value: "metres"}}},
		{"several", dtype.Metadata{
			{Key: "unit", Value: "metres"},
			{Key: "source", Value: "the shipping system"},
		}},

		// Order is part of the value, and duplicate keys are allowed by the
		// format, so both have to survive.
		{"duplicate keys", dtype.Metadata{
			{Key: "tag", Value: "one"},
			{Key: "tag", Value: "two"},
		}},
		{"empty key and value", dtype.Metadata{{Key: "", Value: ""}}},

		// Nothing here is required to be text, and a value with a null byte in
		// it is exactly the case a C string would lose.
		{"bytes", dtype.Metadata{{Key: "raw", Value: "a\x00b\xff"}}},
		{"not ascii", dtype.Metadata{{Key: "unit", Value: "\u00b5s"}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b, err := ipc.EncodeMetadata(tt.meta)
			if err != nil {
				t.Fatalf("EncodeMetadata = %v", err)
			}
			got, err := ipc.DecodeMetadata(b)
			if err != nil {
				t.Fatalf("DecodeMetadata = %v", err)
			}
			if !got.Equal(tt.meta) {
				t.Errorf("round trip = %v, want %v", got, tt.meta)
			}
		})
	}
}

// TestEncodeMetadataEmpty pins down the one case with a choice in it. Nothing
// to say is a null pointer rather than a blob holding a zero.
func TestEncodeMetadataEmpty(t *testing.T) {
	for _, m := range []dtype.Metadata{nil, {}} {
		b, err := ipc.EncodeMetadata(m)
		if err != nil {
			t.Fatalf("EncodeMetadata = %v", err)
		}
		if b != nil {
			t.Errorf("EncodeMetadata(%v) = %v, want nil", m, b)
		}
	}
}

// TestMetadataLayout checks the bytes rather than the round trip. A reader on
// the other side of this boundary is a C library that will not be reading
// through DecodeMetadata, so a pair of functions that agree with each other and
// with nothing else would pass every other test in this file.
func TestMetadataLayout(t *testing.T) {
	b, err := ipc.EncodeMetadata(dtype.Metadata{{Key: "ab", Value: "cde"}})
	if err != nil {
		t.Fatalf("EncodeMetadata = %v", err)
	}

	want := make([]byte, 0, 17)
	want = binary.NativeEndian.AppendUint32(want, 1)
	want = binary.NativeEndian.AppendUint32(want, 2)
	want = append(want, "ab"...)
	want = binary.NativeEndian.AppendUint32(want, 3)
	want = append(want, "cde"...)

	if string(b) != string(want) {
		t.Errorf("EncodeMetadata = % x, want % x", b, want)
	}
}

// TestDecodeMetadataCopies checks that the strings do not point into the blob.
// In a real import the blob is memory the producer owns and frees, so a
// metadata value that shares it would read as something else, or as nothing,
// once the release callback has run.
func TestDecodeMetadataCopies(t *testing.T) {
	b, err := ipc.EncodeMetadata(dtype.Metadata{{Key: "unit", Value: "metres"}})
	if err != nil {
		t.Fatalf("EncodeMetadata = %v", err)
	}
	got, err := ipc.DecodeMetadata(b)
	if err != nil {
		t.Fatalf("DecodeMetadata = %v", err)
	}
	for i := range b {
		b[i] = 0
	}

	want := dtype.Metadata{{Key: "unit", Value: "metres"}}
	if !got.Equal(want) {
		t.Errorf("after overwriting the blob = %v, want %v", got, want)
	}
}

func TestDecodeMetadataErrors(t *testing.T) {
	full, err := ipc.EncodeMetadata(dtype.Metadata{{Key: "unit", Value: "metres"}})
	if err != nil {
		t.Fatalf("EncodeMetadata = %v", err)
	}

	count := func(n int32, rest ...byte) []byte {
		return append(binary.NativeEndian.AppendUint32(nil, uint32(n)), rest...)
	}

	// pair writes a blob claiming n pairs and holding one, with the lengths
	// given rather than the lengths the strings have, so that a length can be
	// wrong on purpose.
	pair := func(n int32, key string, valueLen int32, value string) []byte {
		b := count(n)
		b = binary.NativeEndian.AppendUint32(b, uint32(len(key)))
		b = append(b, key...)
		b = binary.NativeEndian.AppendUint32(b, uint32(valueLen))
		return append(b, value...)
	}

	tests := []struct {
		name string
		blob []byte
	}{
		{"short count", []byte{1, 0}},
		{"negative count", count(-1)},
		{"count with nothing after it", count(1)},

		// A blob claiming more pairs than could fit is refused before any
		// memory is asked for. If this ever regresses the test does not fail,
		// it stops responding, which is the point.
		{"count larger than the blob", count(1 << 30)},

		{"truncated key length", append(count(1), 0, 0)},
		{"key longer than the blob", append(count(1), binary.NativeEndian.AppendUint32(nil, 99)...)},
		{"key with no value", append(count(1), append(binary.NativeEndian.AppendUint32(nil, 2), "ab"...)...)},

		// Past the cheap check above, so these reach the pair by pair reading.
		{"value longer than the blob", pair(1, "ab", 99, "")},
		{"negative value length", pair(1, "ab", -1, "cde")},
		{"a second pair that is not there", pair(2, "abcdef", 6, "ghijkl")},

		{"truncated in the middle", full[:len(full)-1]},
		{"one byte too many", append(full, 0)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ipc.DecodeMetadata(tt.blob)
			if err == nil {
				t.Fatalf("DecodeMetadata = %v, want an error", got)
			}
			if got != nil {
				t.Errorf("DecodeMetadata = %v with an error, want nil", got)
			}
			if !errors.Is(err, ipc.ErrMetadata) {
				t.Errorf("DecodeMetadata = %v, want ErrMetadata", err)
			}
		})
	}
}
