package arrowgo_test

import (
	"fmt"

	"github.com/apache/arrow-go/v18/arrow"
	arrowarray "github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/memory"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/arrowgo"
	"github.com/tamnd/kuma/dtype"
)

// This is the way in: something in the arrow-go ecosystem hands over a record
// batch, and kuma reads it without the values being copied.
func ExampleImportRecordBatch() {
	rec := trades()
	defer rec.Release()

	table, err := arrowgo.ImportRecordBatch(rec)
	if err != nil {
		fmt.Println(err)
		return
	}

	symbol, price := table.Columns[0], table.Columns[1]
	for i := range table.NumRows() {
		if price.IsNull(i) {
			fmt.Printf("%s no price\n", symbol.Bytes(i))
			continue
		}
		fmt.Printf("%s %.2f\n", symbol.Bytes(i), price.Value[float64](i))
	}

	// Output:
	// AAPL 1.50
	// MSFT no price
	// GOOG 3.50
}

// And this is the way out: a kuma table goes to anything that reads arrow-go,
// which is a Parquet writer, a Flight server or a database driver.
func ExampleExportRecordBatch() {
	symbol, err := array.NewBuilder(dtype.String)
	if err != nil {
		fmt.Println(err)
		return
	}
	symbol.AppendString("AAPL")
	symbol.AppendString("MSFT")

	price, err := array.NewBuilder(dtype.Float64)
	if err != nil {
		fmt.Println(err)
		return
	}
	price.AppendValues([]float64{1.5, 2.5})

	symbols, err := array.NewChunked(dtype.String, symbol.Finish())
	if err != nil {
		fmt.Println(err)
		return
	}
	prices, err := array.NewChunked(dtype.Float64, price.Finish())
	if err != nil {
		fmt.Println(err)
		return
	}

	table := &array.Table{
		Schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "symbol", Type: dtype.String},
			{Name: "price", Type: dtype.Float64, Nullable: true},
		}},
		Columns: []*array.Chunked{symbols, prices},
	}

	rec, err := arrowgo.ExportRecordBatch(table)
	if err != nil {
		fmt.Println(err)
		return
	}
	defer rec.Release()

	fmt.Println(rec.NumRows(), "rows of", rec.Schema().Field(0).Type)

	// Output:
	// 2 rows of string_view
}

// Types cross on their own, which is what a reader wants when it is deciding
// whether it can read a file at all before it opens one.
func ExampleImportSchema() {
	in := arrow.NewSchema([]arrow.Field{
		{Name: "symbol", Type: arrow.BinaryTypes.String, Nullable: true},
		{Name: "ts", Type: &arrow.TimestampType{Unit: arrow.Microsecond, TimeZone: "UTC"}},
	}, nil)

	schema, err := arrowgo.ImportSchema(in)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, f := range schema.Fields {
		fmt.Println(f.Name, f.Type)
	}

	// Output:
	// symbol string
	// ts timestamp[us, tz=UTC]
}

// trades builds the record batch the first example reads, which is the sort of
// thing a Parquet reader or a Flight client hands over.
func trades() arrow.RecordBatch {
	schema := arrow.NewSchema([]arrow.Field{
		{Name: "symbol", Type: arrow.BinaryTypes.StringView},
		{Name: "price", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
	}, nil)

	b := arrowarray.NewRecordBuilder(memory.DefaultAllocator, schema)
	defer b.Release()

	symbol := b.Field(0).(*arrowarray.StringViewBuilder)
	symbol.AppendValues([]string{"AAPL", "MSFT", "GOOG"}, nil)

	price := b.Field(1).(*arrowarray.Float64Builder)
	price.AppendValues([]float64{1.5, 0, 3.5}, []bool{true, false, true})

	return b.NewRecordBatch()
}
