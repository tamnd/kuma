package csv_test

import (
	"errors"
	"fmt"
	"strings"

	"github.com/tamnd/kuma/csv"
	"github.com/tamnd/kuma/dtype"
)

func ExampleRead() {
	in := `sym,qty,px,live
AAPL,100,182.5,true
MSFT,,411.2,false
GOOG,300,,true
`

	t, err := csv.Read(strings.NewReader(in), nil)
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

func ExampleRead_types() {
	// The file writes a zip code, which is a name for a place rather than a
	// number, and the leading zero is part of it. Naming the type is how a
	// column stops being guessed at.
	in := `zip,pop
02134,12345
90210,21345
`

	t, err := csv.Read(strings.NewReader(in), &csv.Options{
		Types: map[string]dtype.DataType{"zip": dtype.String},
	})
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(t.Schema)
	for i := range t.NumRows() {
		fmt.Printf("%s ", t.Columns[0].Bytes(i))
	}
	fmt.Println()
	// Output:
	// schema<zip: string not null, pop: int64 not null>
	// 02134 90210
}

func ExampleRead_noHeader() {
	in := "AAPL,100\nMSFT,200\n"

	t, err := csv.Read(strings.NewReader(in), &csv.Options{
		NoHeader: true,
		Names:    []string{"sym", "qty"},
	})
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println(t.Schema)
	// Output:
	// schema<sym: string not null, qty: int64 not null>
}

func ExampleValueError() {
	in := `qty
100
lots
`

	_, err := csv.Read(strings.NewReader(in), &csv.Options{
		Types: map[string]dtype.DataType{"qty": dtype.Int64},
	})
	fmt.Println(err)
	fmt.Println(errors.Is(err, csv.ErrValue))

	var ve *csv.ValueError
	if errors.As(err, &ve) {
		fmt.Println("line", ve.Line, "of column", ve.Column)
	}
	// Output:
	// csv: line 3, column "qty": cannot read "lots" as int64: invalid syntax
	// true
	// line 3 of column qty
}
