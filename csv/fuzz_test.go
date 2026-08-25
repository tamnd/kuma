package csv

import (
	"bytes"
	stdcsv "encoding/csv"
	"errors"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// FuzzRead checks that whatever the input, a read either fails or produces a
// table whose columns line up with each other and with its schema.
//
// The row count is checked against what encoding/csv itself read, which is the
// one thing here that is not this package's own opinion.
func FuzzRead(f *testing.F) {
	f.Add("a,b\n1,2\n", uint8(4))
	f.Add("sym,qty,px\nAAPL,1,1.5\nMSFT,,2\n", uint8(1))
	f.Add("a\n\"b\nc\"\n", uint8(2))
	f.Add("a,b\ntrue,\n", uint8(8))

	f.Fuzz(func(t *testing.T, in string, chunk uint8) {
		tbl, err := Read(strings.NewReader(in), &Options{ChunkSize: int(chunk%8) + 1})
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

		recs, err := stdcsv.NewReader(strings.NewReader(in)).ReadAll()
		if err != nil {
			return
		}
		if want := len(recs) - 1; want != tbl.NumRows() {
			t.Fatalf("read %d rows, want %d", tbl.NumRows(), want)
		}
	})
}

// FuzzWrite checks the round trip: a table written out and read back in is the
// table that went in.
//
// The types are handed to the read on the way back rather than inferred again,
// since the question here is whether the values survived the file and not
// whether a column of ones and zeros still looks like a column of numbers the
// second time around.
func FuzzWrite(f *testing.F) {
	f.Add("a,b\n1,2\n", uint8(0))
	f.Add("sym,qty\nAAPL,100\nMSFT,\n", uint8(1))
	f.Add("a\n\"b\nc\"\n", uint8(2))
	f.Add("px\n1.5\n-0.0\n", uint8(3))

	f.Fuzz(func(t *testing.T, in string, mode uint8) {
		tbl, err := Read(strings.NewReader(in), nil)
		if err != nil {
			return
		}

		wopts := &WriteOptions{}
		ropts := &Options{Types: make(map[string]dtype.DataType, tbl.NumCols())}
		switch mode % 4 {
		case 1:
			wopts.QuoteAll = true
		case 2:
			wopts.CRLF = true
		case 3:
			wopts.Delimiter, ropts.Delimiter = ';', ';'
		}
		for i, field := range tbl.Schema.Fields {
			ropts.Types[field.Name] = tbl.Columns[i].DType()
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
			if got, want := back.Schema.Fields[i].Name, tbl.Schema.Fields[i].Name; got != want {
				t.Fatalf("column %d came back called %q, want %q", i, got, want)
			}
			for j := range tbl.NumRows() {
				if !sameValue(t, tbl.Columns[i], back.Columns[i], j) {
					t.Fatalf("row %d of column %d was %q and came back %q, in %q",
						j, i, value(t, tbl.Columns[i], j), value(t, back.Columns[i], j), text)
				}
			}
		}
	})
}

// sameValue reports whether value i came back as itself, allowing for the two
// things a file cannot carry.
//
// An empty string comes back missing, because a field with nothing in it is the
// only way a file has of saying either. A carriage return before a newline
// comes back as the newline alone, because a reader cannot tell one inside a
// value from the line ending it has to normalize.
func sameValue(t *testing.T, want, got *array.Chunked, i int) bool {
	t.Helper()
	if want.IsNull(i) {
		return got.IsNull(i)
	}

	w := value(t, want, i)
	if w == "" {
		return got.IsNull(i)
	}
	if got.IsNull(i) {
		return false
	}
	return strings.ReplaceAll(w, "\r\n", "\n") == value(t, got, i)
}

// FuzzInfer checks the promise the inference makes: a type worked out from
// every row of the file is a type every row of the file parses into.
//
// The exception is text that is not UTF-8, where inference says string because
// there is nothing else to say and the string column then refuses the bytes.
// That is the reader reporting the file rather than the two disagreeing.
func FuzzInfer(f *testing.F) {
	f.Add("a,b\n1,2\n1.5,x\n")
	f.Add("n\n9223372036854775808\n1\n")
	f.Add("b\ntrue\nfalse\n")

	f.Fuzz(func(t *testing.T, in string) {
		_, err := Read(strings.NewReader(in), &Options{InferRows: -1})
		if errors.Is(err, ErrValue) && !errors.Is(err, errNotUTF8) {
			t.Fatalf("inferred a type the file does not fit: %v", err)
		}
	})
}
