package parquet_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/parquet"
)

// lengthsOf writes values the way a writer of DELTA_LENGTH_BYTE_ARRAY does,
// which is a block of the lengths as differences and then all the bytes.
func lengthsOf(values ...string) []byte {
	sizes := make([]int64, len(values))
	for i, v := range values {
		sizes[i] = int64(len(v))
	}
	return append(deltaOf(128, 4, sizes...), []byte(strings.Join(values, ""))...)
}

// prefixedOf writes values the way a writer of DELTA_BYTE_ARRAY does, which is
// a block of how much of each value the one in front of it already said and then
// the rest of each of them in the other encoding.
//
// Any number of shared bytes up to the whole of the value in front is legal, and
// a writer that shares none of them has written the other encoding with a block
// of noughts in front. This shares as much as there is, since that is what a
// writer worth using does and what makes the values in a test worth reading.
func prefixedOf(values ...string) []byte {
	shared := make([]int64, len(values))
	rest := make([]string, len(values))
	for i, v := range values {
		if i > 0 {
			for shared[i] < int64(min(len(v), len(values[i-1]))) &&
				v[shared[i]] == values[i-1][shared[i]] {
				shared[i]++
			}
		}
		rest[i] = v[shared[i]:]
	}
	return append(deltaOf(128, 4, shared...), lengthsOf(rest...)...)
}

// drain takes everything a decoder has, one value at a time, so that a decoder
// which loses its place between calls is caught along with one that hands back
// the wrong bytes.
func drain(t *testing.T, d interface{ Read([][]byte) (int, error) }) []string {
	t.Helper()

	var got []string
	one := make([][]byte, 1)
	for {
		n, err := d.Read(one)
		if errors.Is(err, io.EOF) {
			return got
		}
		if err != nil {
			t.Fatalf("Read: %v", err)
		}
		if n != 1 {
			t.Fatalf("Read wrote %d values into a buffer of one", n)
		}
		got = append(got, string(one[0]))
	}
}

func TestDeltaLengthDecoder(t *testing.T) {
	for _, tc := range []struct {
		name   string
		values []string
	}{
		{"a run of values", []string{"one", "two", "three", "four"}},
		{"values of nothing", []string{"", "", ""}},
		{"holes in a run", []string{"one", "", "three", "", ""}},
		{"one value", []string{"only"}},
		{"no values at all", nil},
		{"more than a block", counted()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var d parquet.DeltaLengthDecoder
			if err := d.Reset(lengthsOf(tc.values...)); err != nil {
				t.Fatalf("Reset: %v", err)
			}
			if d.Len() != len(tc.values) {
				t.Fatalf("Len is %d, want %d", d.Len(), len(tc.values))
			}
			if got := drain(t, &d); !slices.Equal(got, tc.values) {
				t.Fatalf("got %q, want %q", got, tc.values)
			}
		})
	}
}

func TestDeltaByteArrayDecoder(t *testing.T) {
	for _, tc := range []struct {
		name   string
		values []string
	}{
		{"a shared prefix", []string{"a/b/one", "a/b/two", "a/b/three"}},
		{"nothing shared", []string{"one", "two", "three"}},
		{"the same value over and over", []string{"repeated", "repeated", "repeated"}},
		{"values that get shorter", []string{"abcdef", "abcd", "ab", "a", ""}},
		{"values of nothing", []string{"", "", ""}},
		{"one value", []string{"only"}},
		{"no values at all", nil},
		{"more than a block", counted()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var d parquet.DeltaByteArrayDecoder
			if err := d.Reset(prefixedOf(tc.values...)); err != nil {
				t.Fatalf("Reset: %v", err)
			}
			if d.Len() != len(tc.values) {
				t.Fatalf("Len is %d, want %d", d.Len(), len(tc.values))
			}
			if got := drain(t, &d); !slices.Equal(got, tc.values) {
				t.Fatalf("got %q, want %q", got, tc.values)
			}
		})
	}
}

// counted is a run of values sharing a prefix that grows and shrinks. There are
// enough of them to need several blocks of differences, which is what a page of
// one block would never say anything about.
func counted() []string {
	values := make([]string, 300)
	for i := range values {
		values[i] = fmt.Sprintf("prefix/%d/%s", i%7, strings.Repeat("x", i%23))
	}
	return values
}

// TestDeltaBytesRead reads several values at a time rather than one, and then
// reads a second page through the same decoder.
func TestDeltaBytesRead(t *testing.T) {
	want := counted()

	var lengths parquet.DeltaLengthDecoder
	var prefixed parquet.DeltaByteArrayDecoder
	if err := lengths.Reset(lengthsOf(want...)); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if err := prefixed.Reset(prefixedOf(want...)); err != nil {
		t.Fatalf("Reset: %v", err)
	}

	for _, d := range []interface {
		Read([][]byte) (int, error)
		Len() int
	}{&lengths, &prefixed} {
		buf := make([][]byte, 7)
		for i := 0; i < len(want); i += 7 {
			n, err := d.Read(buf)
			if err != nil {
				t.Fatalf("Read at %d: %v", i, err)
			}
			if n != min(7, len(want)-i) {
				t.Fatalf("Read at %d wrote %d values", i, n)
			}
			for j := range n {
				if got := string(buf[j]); got != want[i+j] {
					t.Fatalf("value %d: got %q, want %q", i+j, got, want[i+j])
				}
			}
		}
		if d.Len() != 0 {
			t.Fatalf("%d values are left over", d.Len())
		}
	}

	// The same decoders again on a page of their own encoding, since a scan
	// reads a thousand pages of a column through one of each.
	if err := lengths.Reset(lengthsOf("second", "page")); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if got := drain(t, &lengths); !slices.Equal(got, []string{"second", "page"}) {
		t.Fatalf("got %q after Reset", got)
	}
	if err := prefixed.Reset(prefixedOf("second", "page")); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	if got := drain(t, &prefixed); !slices.Equal(got, []string{"second", "page"}) {
		t.Fatalf("got %q after Reset", got)
	}
}

func TestDeltaLengthDecoderRefused(t *testing.T) {
	for _, tc := range []struct {
		name string
		data []byte
	}{
		{"nothing at all", nil},
		{"a header that stops", []byte{0x80, 0x01}},
		{"a block of no values", deltaOf(0, 1)},
		{"a length of fewer than no bytes", deltaOf(128, 4, -1)},
		{
			name: "bytes that run out",
			data: append(deltaOf(128, 4, 3, 3), []byte("onetw")...),
		},
		{
			name: "the first length past the end of the page",
			data: deltaOf(128, 4, 1),
		},
		{
			// A lengths block that says it holds more than it wrote, so the
			// block runs out before the count in its header does.
			name: "a block shorter than its count",
			data: deltaOf(128, 4, 1, 2, 3)[:6],
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var d parquet.DeltaLengthDecoder
			err := d.Reset(tc.data)
			if !errors.Is(err, parquet.ErrFormat) {
				t.Fatalf("Reset: got %v, want a format error", err)
			}
			if d.Len() != 0 {
				t.Fatalf("a refused page left %d values behind", d.Len())
			}
			if _, err := d.Read(make([][]byte, 4)); !errors.Is(err, io.EOF) {
				t.Fatalf("Read: got %v, want EOF", err)
			}
		})
	}
}

func TestDeltaByteArrayDecoderRefused(t *testing.T) {
	// Values written by hand rather than by prefixedOf, since what is wrong
	// with each of them is something a writer would not do.
	shared := func(sizes []int64, rest ...string) []byte {
		return append(deltaOf(128, 4, sizes...), lengthsOf(rest...)...)
	}

	for _, tc := range []struct {
		name string
		data []byte
	}{
		{"nothing at all", nil},
		{"a header that stops", []byte{0x80, 0x01}},
		{"shared lengths and nothing behind them", deltaOf(128, 4, 0, 0)},
		{
			// A block of shared lengths that says it holds more than it wrote,
			// so it runs out before the count in its header does.
			name: "shared lengths shorter than their count",
			data: deltaOf(128, 4, 0, 1, 2)[:6],
		},
		{"more values than shared lengths", shared([]int64{0}, "one", "two")},
		{"fewer values than shared lengths", shared([]int64{0, 0}, "one")},
		{"a first value that shares bytes", shared([]int64{1}, "one")},
		{"sharing more than there is", shared([]int64{0, 4}, "ab", "cd")},
		{"sharing fewer than no bytes", shared([]int64{0, -1}, "ab", "cd")},
		{"a suffix longer than the bytes behind it", shared([]int64{0}, "one")[:len(shared([]int64{0}, "one"))-1]},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var d parquet.DeltaByteArrayDecoder
			err := d.Reset(tc.data)
			if !errors.Is(err, parquet.ErrFormat) {
				t.Fatalf("Reset: got %v, want a format error", err)
			}
			if d.Len() != 0 {
				t.Fatalf("a refused page left %d values behind", d.Len())
			}
			if _, err := d.Read(make([][]byte, 4)); !errors.Is(err, io.EOF) {
				t.Fatalf("Read: got %v, want EOF", err)
			}
		})
	}
}

// TestReadColumnStrings reads the columns of a file pyarrow wrote in the two
// encodings, which is the check that matters: the shapes here are what a real
// writer produces rather than what this package's own tests can build.
func TestReadColumnStrings(t *testing.T) {
	t.Run("a shared prefix", func(t *testing.T) {
		stringColumn(t, "key", func(i int) string {
			return fmt.Sprintf("customer/2026/08/%06d", i)
		})
	})

	t.Run("lengths of their own", func(t *testing.T) {
		stringColumn(t, "word", func(i int) string {
			return strings.Repeat("a", i%17) + strconv.Itoa(i%97)
		})
	})

	// Bytes rather than a string, and every thirteenth value is empty, which is
	// a value of no bytes rather than a missing one.
	t.Run("values of nothing", func(t *testing.T) {
		stringColumn(t, "blob", func(i int) string {
			if i%13 == 0 {
				return ""
			}
			return string(bytes.Repeat([]byte{byte(i % 256)}, i%11))
		})
	})

	// The far end of sharing a prefix, where every value shares the whole of
	// the one in front and writes nothing of its own.
	t.Run("the same value every time", func(t *testing.T) {
		stringColumn(t, "same", func(int) string { return "the same string every time" })
	})

	// The format allows the shared prefix encoding for a fixed width column and
	// not the other one, so this is the column that tells the two apart.
	t.Run("a fixed width", func(t *testing.T) {
		stringColumn(t, "fixed", func(i int) string { return fmt.Sprintf("%08d", i) })
	})

	// A page holds a value for a row that has one and nothing for a row that
	// does not, so what a value shares is shared with the value in front of it
	// rather than with the row in front of it.
	t.Run("with nulls", func(t *testing.T) {
		a := readColumn(t, "strings.parquet", "maybe")
		if a.Len() != 1000 || a.NullCount() != 143 {
			t.Fatalf("%d values, %d of them null", a.Len(), a.NullCount())
		}
		for i := range a.Len() {
			if i%7 == 0 {
				if !a.IsNull(i) {
					t.Fatalf("value %d is there and should be missing", i)
				}
				continue
			}
			if got, want := string(a.Bytes(i)), fmt.Sprintf("row-%d", i); got != want {
				t.Fatalf("value %d: got %q, want %q", i, got, want)
			}
		}
	})
}

// stringColumn reads a column of strings.parquet and checks every value of it.
func stringColumn(t *testing.T, column string, at func(i int) string) {
	t.Helper()

	a := readColumn(t, "strings.parquet", column)
	if a.Len() != 1000 || a.NullCount() != 0 {
		t.Fatalf("%d values, %d of them null", a.Len(), a.NullCount())
	}
	for i := range a.Len() {
		if got := string(a.Bytes(i)); got != at(i) {
			t.Fatalf("value %d: got %q, want %q", i, got, at(i))
		}
	}
}

// TestColumnReaderStringPages hands the reader pages written by hand, which is
// where the shapes a real writer does not produce come from.
func TestColumnReaderStringPages(t *testing.T) {
	c := parquet.Column{
		Element:       parquet.SchemaElement{Name: "s", Type: parquet.ByteArray},
		Type:          dtype.String,
		MaxDefinition: 1,
	}

	t.Run("with a hole in it", func(t *testing.T) {
		r := readerOf(t, c)
		// A bit packed run of one group of eight levels holding 1, 0, 1.
		page := bytesPage(parquet.DeltaByteArray, []byte{0x03, 0x05}, 3,
			prefixedOf("a/one", "a/two"))
		if err := r.Page(page); err != nil {
			t.Fatalf("Page: %v", err)
		}

		a := finish(t, r)
		if a.Len() != 3 || a.NullCount() != 1 || !a.IsNull(1) {
			t.Fatalf("%d values, %d of them null, and the second is null: %v",
				a.Len(), a.NullCount(), !a.IsNull(1))
		}
		if got := string(a.Bytes(0)) + " " + string(a.Bytes(2)); got != "a/one a/two" {
			t.Fatalf("got %q", got)
		}
	})

	t.Run("a page of nothing but nulls", func(t *testing.T) {
		r := readerOf(t, c)
		if err := r.Page(bytesPage(parquet.DeltaLengthByteArray, []byte{0x04, 0x00}, 2, nil)); err != nil {
			t.Fatalf("Page: %v", err)
		}

		a := finish(t, r)
		if a.Len() != 2 || a.NullCount() != 2 {
			t.Fatalf("%d values, %d of them null", a.Len(), a.NullCount())
		}
	})

	t.Run("a required column", func(t *testing.T) {
		required := c
		required.MaxDefinition = 0
		r := readerOf(t, required)

		page := bytesPage(parquet.DeltaLengthByteArray, nil, 3, lengthsOf("one", "two", "three"))
		page.Data = page.Data[4:]
		if err := r.Page(page); err != nil {
			t.Fatalf("Page: %v", err)
		}

		a := finish(t, r)
		if a.Len() != 3 || a.NullCount() != 0 {
			t.Fatalf("%d values, %d of them null", a.Len(), a.NullCount())
		}
		for i, want := range []string{"one", "two", "three"} {
			if got := string(a.Bytes(i)); got != want {
				t.Fatalf("value %d: got %q, want %q", i, got, want)
			}
		}
	})
}

// bytesPage builds a page of the first version, whose levels sit inside the
// body behind four bytes of their length.
func bytesPage(e parquet.Encoding, levels []byte, values int32, encoded []byte) parquet.Page {
	body := binary.LittleEndian.AppendUint32(nil, uint32(len(levels)))
	body = append(body, levels...)
	body = append(body, encoded...)
	return parquet.Page{
		PageHeader: parquet.PageHeader{
			Kind: parquet.DataPage, Encoding: e,
			DefinitionEncoding: parquet.RLE, NumValues: values,
		},
		Data: body,
	}
}

func TestColumnReaderRefusedDeltaBytes(t *testing.T) {
	strs := parquet.Column{
		Element: parquet.SchemaElement{Name: "s", Type: parquet.ByteArray},
		Type:    dtype.String,
	}
	fixed := parquet.Column{
		Element: parquet.SchemaElement{Name: "f", Type: parquet.FixedLenByteArray, TypeLength: 4},
		Type:    dtype.FixedSizeBinary{ByteWidth: 4},
	}
	numbers := parquet.Column{
		Element: parquet.SchemaElement{Name: "n", Type: parquet.Int32},
		Type:    dtype.Int32,
	}

	for _, tc := range []struct {
		name   string
		column parquet.Column
		page   parquet.Page
		want   error
	}{
		{
			// The lengths encoding is written for the byte arrays that carry
			// their own length and no other type.
			name: "lengths on a column of numbers", column: numbers, want: parquet.ErrFormat,
			page: bytesPage(parquet.DeltaLengthByteArray, nil, 1, lengthsOf("one")),
		},
		{
			// A fixed width column has no lengths of its own, so this is a page
			// that contradicts the schema rather than one to read.
			name: "lengths on a fixed width column", column: fixed, want: parquet.ErrFormat,
			page: bytesPage(parquet.DeltaLengthByteArray, nil, 1, lengthsOf("abcd")),
		},
		{
			name: "shared prefixes on a column of numbers", column: numbers, want: parquet.ErrFormat,
			page: bytesPage(parquet.DeltaByteArray, nil, 1, prefixedOf("one")),
		},
		{
			// The shared prefix encoding is allowed for a fixed width column,
			// so what is wrong here is the page rather than the pairing.
			name: "a fixed width page that stops", column: fixed, want: parquet.ErrFormat,
			page: bytesPage(parquet.DeltaByteArray, nil, 1, []byte{0x80, 0x01}),
		},
		{
			name: "a page of lengths that stops", column: strs, want: parquet.ErrFormat,
			page: bytesPage(parquet.DeltaLengthByteArray, nil, 2, lengthsOf("one")),
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := readerOf(t, tc.column)
			tc.page.Data = tc.page.Data[4:]

			err := r.Page(tc.page)
			if !errors.Is(err, tc.want) {
				t.Fatalf("Page: got %v, want %v", err, tc.want)
			}
		})
	}
}

func BenchmarkReadColumnStrings(b *testing.B) {
	for _, column := range []string{"key", "word"} {
		b.Run(column, func(b *testing.B) {
			file, chunk, c := chunkOf(&testing.T{}, "strings.parquet", column)
			r := bytes.NewReader(file)

			b.SetBytes(chunk.Meta.TotalCompressedSize)
			b.ReportAllocs()
			for b.Loop() {
				if _, err := parquet.ReadColumn(r, int64(len(file)), chunk, c); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
