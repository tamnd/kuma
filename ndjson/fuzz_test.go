package ndjson

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
)

// FuzzRead checks that whatever the input, a read either fails or produces a
// table whose columns line up with each other and with its schema.
//
// The row count is checked against the lines the file has, which is the whole
// of what the format promises: one object to a line, one row to an object.
func FuzzRead(f *testing.F) {
	f.Add("{\"a\":1,\"b\":2}\n", uint8(4))
	f.Add("{\"sym\":\"AAPL\",\"qty\":1,\"px\":1.5}\n{\"sym\":\"MSFT\",\"px\":2}\n", uint8(1))
	f.Add("{\"a\":{\"b\":[1,2]}}\n", uint8(2))
	f.Add("{\"a\":true,\"b\":null}\n\n{\"a\":false}\n", uint8(8))

	f.Fuzz(func(t *testing.T, in string, chunk uint8) {
		tbl, err := Read(strings.NewReader(in), &Options{
			ChunkSize:           int(chunk%8) + 1,
			IgnoreUnknownFields: true,
		})
		if err != nil {
			return
		}

		if len(tbl.Columns) != len(tbl.Schema.Fields) {
			t.Fatalf("%d columns and %d fields", len(tbl.Columns), len(tbl.Schema.Fields))
		}
		for i, c := range tbl.Columns {
			if c.Len() != tbl.NumRows() {
				t.Fatalf("column %d is %d rows, want %d", i, c.Len(), tbl.NumRows())
			}
			if f := tbl.Schema.Fields[i]; f.Nullable != (c.NullCount() > 0) {
				t.Fatalf("column %d says nullable %v and has %d nulls", i, f.Nullable, c.NullCount())
			}
		}

		if want := countLines(in); want != tbl.NumRows() {
			t.Fatalf("read %d rows from %q, want %d", tbl.NumRows(), in, want)
		}
	})
}

// countLines is how many rows a file that reads without an error has, which is
// its lines with the blank ones left out. A file whose last line has no newline
// after it still ends in a line.
func countLines(in string) int {
	n := 0
	for line := range strings.Lines(in) {
		if strings.Trim(line, " \t\r\n") != "" {
			n++
		}
	}
	return n
}

// FuzzWrite checks the round trip: a table written out and read back in is the
// table that went in.
//
// The types are handed to the read on the way back rather than inferred again,
// since the question here is whether the values survived the file and not
// whether a column of ones and zeros still looks like a column of numbers the
// second time around.
func FuzzWrite(f *testing.F) {
	f.Add("{\"a\":1,\"b\":2}\n", uint8(0))
	f.Add("{\"sym\":\"AAPL\",\"qty\":100}\n{\"sym\":\"MSFT\",\"qty\":null}\n", uint8(1))
	f.Add("{\"a\":\"b\\nc\"}\n{\"a\":\"\"}\n", uint8(2))
	f.Add("{\"px\":1.5}\n{\"px\":-0.0}\n", uint8(3))

	f.Fuzz(func(t *testing.T, in string, mode uint8) {
		tbl, err := Read(strings.NewReader(in), nil)
		if err != nil {
			return
		}

		names := tbl.Schema.Names()
		wopts := &WriteOptions{}
		switch mode % 3 {
		case 1:
			// A column that is missing on every line is written on no line at
			// all, so the file has nothing left to say the column was there.
			// That is what the option asks for rather than something going
			// wrong, so there is no round trip to check.
			for _, c := range tbl.Columns {
				if c.NullCount() == c.Len() {
					return
				}
			}
			wopts.OmitNull = true
		case 2:
			names = make([]string, tbl.NumCols())
			for i := range names {
				names[i] = "c" + string(rune('a'+i%26))
			}
			if len(names) > 26 {
				return // two columns would end up called the same thing
			}
			wopts.Names = names
		}

		ropts := &Options{
			InferRows: -1,
			Types:     make(map[string]dtype.DataType, tbl.NumCols()),
		}
		for i, name := range names {
			ropts.Types[name] = tbl.Columns[i].DType()
		}

		var buf bytes.Buffer
		if err = Write(&buf, tbl, wopts); err != nil {
			t.Fatalf("Write: %v", err)
		}
		text := buf.String()

		back, err := Read(&buf, ropts)
		if err != nil {
			t.Fatalf("reading back %q: %v", text, err)
		}
		if back.NumCols() != tbl.NumCols() || back.NumRows() != tbl.NumRows() {
			t.Fatalf("wrote %d by %d as %q and read back %d by %d",
				tbl.NumRows(), tbl.NumCols(), text, back.NumRows(), back.NumCols())
		}

		for i := range tbl.Columns {
			if got, want := back.Schema.Fields[i].Name, names[i]; got != want {
				t.Fatalf("column %d came back called %q, want %q", i, got, want)
			}
			for j := range tbl.NumRows() {
				want, got := tbl.Columns[i], back.Columns[i]
				if want.IsNull(j) != got.IsNull(j) || value(t, want, j) != value(t, got, j) {
					t.Fatalf("row %d of column %d was %q and came back %q, in %q",
						j, i, value(t, want, j), value(t, got, j), text)
				}
			}
		}
	})
}

// FuzzInfer checks the promise the inference makes: a type worked out from
// every line of the file is a type every line of the file parses into.
func FuzzInfer(f *testing.F) {
	f.Add("{\"a\":1,\"b\":2}\n{\"a\":1.5,\"b\":\"x\"}\n")
	f.Add("{\"n\":9223372036854775808}\n{\"n\":1}\n")
	f.Add("{\"b\":true}\n{\"b\":false}\n")
	f.Add("{\"n\":1e999}\n{\"n\":1}\n")

	f.Fuzz(func(t *testing.T, in string) {
		_, err := Read(strings.NewReader(in), &Options{InferRows: -1})
		if errors.Is(err, ErrValue) {
			t.Fatalf("inferred a type the file does not fit: %v", err)
		}
	})
}
