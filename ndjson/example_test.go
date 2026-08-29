package ndjson_test

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ndjson"
)

func ExampleRead() {
	in := `{"sym":"AAPL","qty":100,"px":182.5,"live":true}
{"sym":"MSFT","qty":null,"px":411.2,"live":false}
{"sym":"GOOG","qty":300,"px":null,"live":true}
`

	t, err := ndjson.Read(strings.NewReader(in), nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(t.Schema)
	fmt.Println(t.NumRows(), "rows")

	// Nothing is missing from sym and something is missing from qty, which is
	// what the schema says above and what the column carries below.
	fmt.Println(t.Columns[1])
	// Output:
	// schema<sym: string not null, qty: int64, px: float64, live: bool not null>
	// 3 rows
	// array.Chunked{int64, len 3, nulls 1, chunks 1}
}

func ExampleRead_missingMembers() {
	// A member that is not on a line is missing on that line, which is the same
	// thing as writing it as null. Files written a record at a time usually
	// leave out what they have nothing to say about.
	in := `{"sym":"AAPL","qty":100}
{"sym":"MSFT"}
{"sym":"GOOG","qty":300}
`

	t, err := ndjson.Read(strings.NewReader(in), nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(t.Schema)
	fmt.Println(t.Columns[1].NullCount(), "missing")
	// Output:
	// schema<sym: string not null, qty: int64>
	// 1 missing
}

func ExampleRead_types() {
	// The file writes a zip code, which is a name for a place rather than a
	// number, and the leading zero would be gone the moment it became one.
	// Naming the type is how a column stops being guessed at.
	in := `{"zip":"02134","pop":12345}
{"zip":"90210","pop":21345}
`

	t, err := ndjson.Read(strings.NewReader(in), &ndjson.Options{
		Types: map[string]dtype.DataType{"pop": dtype.Int32},
	})
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(t.Schema)
	// Output:
	// schema<zip: string not null, pop: int32 not null>
}

func ExampleRead_columns() {
	// A file of trades with a lot in it, where the question being asked is
	// about two of the members. The rest are never parsed and never stored.
	in := `{"ts":"2026-01-02T09:30:00Z","sym":"AAPL","qty":100,"px":182.5,"venue":"XNAS"}
{"ts":"2026-01-02T09:30:01Z","sym":"MSFT","qty":200,"px":411.2,"venue":"XNAS"}
`

	t, err := ndjson.Read(strings.NewReader(in), &ndjson.Options{
		Columns: []string{"sym", "qty"},
	})
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(t.Schema)
	fmt.Println(t.NumCols(), "columns of", 5)
	// Output:
	// schema<sym: string not null, qty: int64 not null>
	// 2 columns of 5
}

func ExampleRead_nested() {
	// A member holding an object or an array is a string column of the JSON it
	// arrived as, which keeps everything the file said until there is a list
	// column to put it in.
	in := `{"sym":"AAPL","tags":["tech","large"]}
{"sym":"MSFT","tags":[]}
`

	t, err := ndjson.Read(strings.NewReader(in), nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(t.Schema)
	fmt.Printf("%s\n", t.Columns[1].Bytes(0))
	// Output:
	// schema<sym: string not null, tags: string not null>
	// ["tech","large"]
}

func ExampleWrite() {
	in := `{"sym":"AAPL","qty":100,"px":182.5}
{"sym":"MSFT","qty":null,"px":411.2}
`

	t, err := ndjson.Read(strings.NewReader(in), nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	// The table goes back out as it came in. A missing quantity is null again,
	// since that is what the file said in the first place.
	if err := ndjson.Write(os.Stdout, t, nil); err != nil {
		fmt.Println(err)
	}
	// Output:
	// {"sym":"AAPL","qty":100,"px":182.5}
	// {"sym":"MSFT","qty":null,"px":411.2}
}

func ExampleWrite_options() {
	in := `{"sym":"AAPL","qty":100,"px":182.5}
{"sym":"MSFT","qty":null,"px":411.2}
`

	t, err := ndjson.Read(strings.NewReader(in), nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	err = ndjson.Write(os.Stdout, t, &ndjson.WriteOptions{
		Names:     []string{"symbol", "quantity", "price"},
		OmitNull:  true,
		Precision: 2,
	})
	if err != nil {
		fmt.Println(err)
	}
	// Output:
	// {"symbol":"AAPL","quantity":100,"price":182.50}
	// {"symbol":"MSFT","price":411.20}
}

func ExampleValueError() {
	in := `{"qty":100}
{"qty":"lots"}
`

	_, err := ndjson.Read(strings.NewReader(in), &ndjson.Options{
		Types: map[string]dtype.DataType{"qty": dtype.Int64},
	})
	fmt.Println(err)
	fmt.Println(errors.Is(err, ndjson.ErrValue))

	var ve *ndjson.ValueError
	if errors.As(err, &ve) {
		fmt.Println("line", ve.Line, "of column", ve.Column)
	}
	// Output:
	// ndjson: line 2, column "qty": cannot read "lots" as int64: invalid syntax
	// true
	// line 2 of column qty
}

func ExampleLineError() {
	in := `{"qty":100}
{"qty":
`

	_, err := ndjson.Read(strings.NewReader(in), nil)
	fmt.Println(err)
	fmt.Println(errors.Is(err, ndjson.ErrSyntax))

	var le *ndjson.LineError
	if errors.As(err, &le) {
		fmt.Println("line", le.Line)
	}
	// Output:
	// ndjson: line 2: the line stops in the middle of the object: malformed JSON
	// true
	// line 2
}
