package parquet_test

import (
	"bytes"
	"fmt"
	"log"
	"os"

	"github.com/tamnd/kuma/kernel"
	"github.com/tamnd/kuma/parquet"
)

// openTestdata makes a reader for one of the files the tests are run against.
func openTestdata(name string) *parquet.FileReader {
	buf, err := os.ReadFile("testdata/" + name)
	if err != nil {
		log.Fatal(err)
	}
	r, err := parquet.NewFileReader(bytes.NewReader(buf), int64(len(buf)))
	if err != nil {
		log.Fatal(err)
	}
	return r
}

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

// Reading the rows that pass a filter, which skips the row groups the footer
// rules out and compares the rows of the ones it reads.
func ExampleReadFile_filter() {
	// The file holds twelve rows in three row groups with n running from nought
	// to eleven, so only the last group can hold a row of eight or more and the
	// other two are never opened. The filter names a column the caller did not
	// ask for, which is the ordinary case, so n is read to compare the rows
	// against and left out of the table.
	t, err := parquet.ReadFile("testdata/stats.parquet", &parquet.Options{
		Columns: []string{"word"},
		Filter: []parquet.Predicate{
			parquet.Where("n", kernel.OpGe, int64(8)),
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(t.NumRows(), "rows of", t.Schema)
	fmt.Println(text(t.Columns[0]))
	// Output:
	// 4 rows of schema<word: string not null>
	// [zulu yankee victor sierra]
}

// Reading the row groups a filter cannot rule out and leaving the rest of the
// file alone.
func ExampleFileReader_RowGroups() {
	r := openTestdata("stats.parquet")

	// The file holds twelve rows in three row groups with n running from nought
	// to eleven, so two of the three cannot hold a row of eight or more and the
	// footer already says which.
	groups, err := r.RowGroups(parquet.Where("n", kernel.OpGe, int64(8)))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("reading", len(groups), "of", r.NumRowGroups(), "row groups")

	for _, g := range groups {
		b, err := r.RowGroup(g)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("row group", g, "holds", b.Length, "rows")
	}
	// Output:
	// reading 1 of 3 row groups
	// row group 2 holds 4 rows
}

// Ruling out a value that is inside the range of every row group, which is what
// a writer writes a bloom filter for.
func ExampleFileReader_RowGroups_bloom() {
	r := openTestdata("bloom.parquet")

	// The identifiers go up in sevens, so 1004 sits between two of them: inside
	// the bounds of the first group and not in the file.
	for _, id := range []int64{1007, 1004} {
		groups, err := r.RowGroups(parquet.Where("id", kernel.OpEq, id))
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(id, "may be in row groups", groups)
	}
	// Output:
	// 1007 may be in row groups [0]
	// 1004 may be in row groups []
}
