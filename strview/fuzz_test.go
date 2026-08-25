package strview_test

import (
	"bytes"
	"testing"

	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/strview"
)

// This file was written before the implementation, because the failure mode of
// this layout is not a crash. Getting the short against long discriminant wrong
// hands back the wrong bytes, and the wrong bytes look like data. Nobody
// notices until somebody runs a real dataset through it and a join comes back
// with a row count that is off by a few thousand.
//
// So the tests come first, and they are written against a model rather than
// against the implementation. FuzzBuilder holds every value that went in and
// demands the same bytes back. FuzzData goes the other way and feeds arbitrary
// bytes in as views, which is what an Arrow IPC message from another process
// is, and demands that anything the validator accepted can then be read
// without a panic and without lying.

// carve splits data into values, using each value's first byte as the length of
// the one after it. Lengths run to nineteen so that the twelve byte boundary,
// which is the only interesting number in this layout, is crossed constantly.
func carve(data []byte) [][]byte {
	var out [][]byte
	for i := 0; i < len(data); {
		n := int(data[i]) % 20
		i++
		if i+n > len(data) {
			n = len(data) - i
		}
		out = append(out, data[i:i+n])
		i += n
	}
	return out
}

func FuzzBuilder(f *testing.F) {
	f.Add([]byte("\x04kuma"))
	f.Add([]byte("\x00\x00\x00"))
	f.Add([]byte("\x0cexactly12345\x0dthirteen chars"))
	f.Add(bytes.Repeat([]byte("\x13abcdefghijklmnopqrs"), 8))

	f.Fuzz(func(t *testing.T, data []byte) {
		values := carve(data)

		var b strview.Builder
		for _, v := range values {
			b.Append(v)
		}
		if b.Len() != len(values) {
			t.Fatalf("Builder.Len is %d after %d appends", b.Len(), len(values))
		}

		d := b.Finish()
		if d.Len() != len(values) {
			t.Fatalf("Data.Len is %d, want %d", d.Len(), len(values))
		}

		for i, want := range values {
			if got := d.At(i); !bytes.Equal(got, want) {
				t.Fatalf("At(%d) = %q, want %q", i, got, want)
			}

			v := d.View(i)
			if v.Len() != len(want) {
				t.Fatalf("View(%d).Len() = %d, want %d", i, v.Len(), len(want))
			}
			if v.IsInline() != (len(want) <= strview.MaxInline) {
				t.Fatalf("View(%d).IsInline() = %v for a value of %d bytes",
					i, v.IsInline(), len(want))
			}

			// The prefix is the first four bytes of the value, zero padded, and
			// it is in the same place whether the value is inline or not. That
			// is the whole reason this layout is worth having, so it is checked
			// for every value rather than for the long ones.
			var prefix [strview.PrefixLen]byte
			copy(prefix[:], want)
			if got := v.Prefix(); got != prefix {
				t.Fatalf("View(%d).Prefix() = %q, want %q", i, got[:], prefix[:])
			}

			// An inline value has to be byte for byte what MakeInline produces,
			// padding included. Two equal short values are then equal as raw
			// sixteen byte views, which is a comparison worth being able to
			// make.
			if v.IsInline() && v != strview.MakeInline(want) {
				t.Fatalf("View(%d) is %q, want %q",
					i, v[:], strview.MakeInline(want))
			}
		}

		// Comparisons agree with the bytes package, which is the only
		// definition of right there is here.
		for i := range min(len(values), 24) {
			for j := range min(len(values), 24) {
				if got, want := d.Equal(i, j), bytes.Equal(values[i], values[j]); got != want {
					t.Fatalf("Equal(%d, %d) = %v for %q and %q", i, j, got, values[i], values[j])
				}
				got := d.Compare(i, j)
				want := bytes.Compare(values[i], values[j])
				if sign(got) != sign(want) {
					t.Fatalf("Compare(%d, %d) = %d, want the sign of %d for %q and %q",
						i, j, got, want, values[i], values[j])
				}
				if got, want := d.EqualValue(i, values[j]), bytes.Equal(values[i], values[j]); got != want {
					t.Fatalf("EqualValue(%d, %q) = %v", i, values[j], got)
				}
			}
		}

		// A validator that rejects what the builder produced would be a
		// validator nothing could ever satisfy.
		if _, err := strview.NewData(d.Views(), d.Blocks()); err != nil {
			t.Fatalf("NewData rejected what Builder produced: %v", err)
		}

		// Build hands its memory to the Data, so it has to leave the builder
		// empty. A builder that came back still holding those views would let
		// the next Append write into a column that has already been handed out.
		if b.Len() != 0 {
			t.Fatalf("Builder.Len is %d after Build", b.Len())
		}
		b.Reset()
		if b.Len() != 0 {
			t.Fatalf("Builder.Len is %d after Reset", b.Len())
		}
		b.Append([]byte("after the reset"))
		again := b.Finish()
		if got := string(again.At(0)); got != "after the reset" {
			t.Fatalf("after Reset the builder holds %q", got)
		}

		// And the column built before is still what it was, which is the
		// property that handing the memory over is only safe because of.
		for i, want := range values {
			if got := d.At(i); !bytes.Equal(got, want) {
				t.Fatalf("At(%d) = %q after the builder was used again, want %q", i, got, want)
			}
		}
	})
}

// FuzzData feeds arbitrary bytes in as views, which is what a column arriving
// over Arrow IPC from another process amounts to. Nothing here says the data
// makes sense. What it says is that NewData either refuses it or hands back
// something that can be read, and that reading it never panics, never returns
// more bytes than the view claims, and never disagrees with itself about the
// order of two values.
func FuzzData(f *testing.F) {
	f.Add([]byte{4, 0, 0, 0, 'k', 'u', 'm', 'a'}, []byte("data block"))
	f.Add([]byte{20, 0, 0, 0, 'k', 'u', 'm', 'a', 0, 0, 0, 0, 0, 0, 0, 0}, []byte("kuma"))
	f.Add(bytes.Repeat([]byte{13, 0, 0, 0}, 8), bytes.Repeat([]byte("x"), 40))

	f.Fuzz(func(t *testing.T, raw, block []byte) {
		views := make([]strview.View, len(raw)/strview.Size)
		for i := range views {
			copy(views[i][:], raw[i*strview.Size:])
		}

		blocks := []*buffer.Buffer{buffer.Wrap(block)}
		d, err := strview.NewData(views, blocks)
		if err != nil {
			// Refusing is a fine answer for arbitrary bytes, and it is the
			// answer for most of them.
			return
		}

		for i := range d.Len() {
			got := d.At(i)
			if len(got) != d.View(i).Len() {
				t.Fatalf("At(%d) returned %d bytes for a view claiming %d",
					i, len(got), d.View(i).Len())
			}
			if !d.EqualValue(i, got) {
				t.Fatalf("EqualValue(%d, At(%d)) is false", i, i)
			}
			if !d.Equal(i, i) {
				t.Fatalf("Equal(%d, %d) is false", i, i)
			}
			if d.Compare(i, i) != 0 {
				t.Fatalf("Compare(%d, %d) = %d", i, i, d.Compare(i, i))
			}
		}

		// Order is a total order or it is not an order. Anything that passed
		// validation has to agree with the bytes it resolves to, both ways
		// round.
		for i := range min(d.Len(), 16) {
			for j := range min(d.Len(), 16) {
				a, b := d.At(i), d.At(j)
				if sign(d.Compare(i, j)) != sign(bytes.Compare(a, b)) {
					t.Fatalf("Compare(%d, %d) = %d but the values are %q and %q",
						i, j, d.Compare(i, j), a, b)
				}
				if d.Compare(i, j) != -d.Compare(j, i) && sign(d.Compare(i, j)) != -sign(d.Compare(j, i)) {
					t.Fatalf("Compare(%d, %d) and Compare(%d, %d) disagree", i, j, j, i)
				}
			}
		}
	})
}

// sign reduces a comparison result to -1, 0 or 1, since the contract is the
// sign and not the number.
func sign(n int) int {
	switch {
	case n < 0:
		return -1
	case n > 0:
		return 1
	}
	return 0
}
