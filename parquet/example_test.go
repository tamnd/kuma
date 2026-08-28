package parquet_test

import (
	"fmt"
	"log"

	"github.com/tamnd/kuma/parquet"
)

// Reading a whole file.
func ExampleReadFile() {
	t, err := parquet.ReadFile("testdata/chunks.parquet", nil)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(t.NumRows(), "rows and", t.NumCols(), "columns")
	for i, f := range t.Schema.Fields {
		fmt.Printf("%s is a %s column in %d chunks\n", f.Name, f.Type, t.Columns[i].NumChunks())
	}
	// Output:
	// 6 rows and 2 columns
	// code is a string column in 2 chunks
	// n is a int64 column in 2 chunks
}

// Reading two columns of a wide file and leaving the rest of it on disk.
func ExampleReadFile_projection() {
	t, err := parquet.ReadFile("testdata/alltypes.parquet", &parquet.Options{
		Columns: []string{"name", "total"},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(t.Schema)
	// Output:
	// schema<name: string, total: int64>
}

// Keeping the encoding of a column the file wrote as indices into a dictionary,
// which is worth doing for a column that repeats and not for one that does not.
func ExampleReadFile_dictionary() {
	t, err := parquet.ReadFile("testdata/chunks.parquet", &parquet.Options{
		Columns:    []string{"code"},
		Dictionary: true,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(t.Columns[0].DType())
	// Output:
	// dictionary<int32, string>
}
