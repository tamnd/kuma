package csv

import (
	stdcsv "encoding/csv"
	"errors"
	"strings"
	"testing"
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
