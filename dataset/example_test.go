package dataset_test

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dataset"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ndjson"
)

// orders writes a small partitioned tree and returns the directory it is in,
// which is what the rest of these examples read.
func orders() string {
	root, err := os.MkdirTemp("", "orders")
	if err != nil {
		panic(err)
	}

	rows := map[string]string{
		"year=2024/month=01": `{"sym":"AAPL","qty":100}` + "\n",
		"year=2024/month=02": `{"sym":"MSFT","qty":50}` + "\n" + `{"sym":"GOOG","qty":25}` + "\n",
		"year=2025/month=01": `{"sym":"AAPL","qty":300}` + "\n",
	}
	for dir, data := range rows {
		dir = filepath.Join(root, filepath.FromSlash(dir))
		if err = os.MkdirAll(dir, 0o750); err != nil {
			panic(err)
		}
		if err = os.WriteFile(filepath.Join(dir, "part-0.ndjson"), []byte(data), 0o600); err != nil {
			panic(err)
		}
	}
	return root
}

func ExampleDiscover() {
	root := orders()
	defer os.RemoveAll(root)

	d, err := dataset.Discover(root, nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Nothing has been opened. All of this came out of the paths.
	fmt.Println(d.Len(), "files")
	fmt.Println(d.Schema)
	for _, f := range d.Files {
		fmt.Println(d.Value(f, "year"), d.Value(f, "month"))
	}
	// Output:
	// 3 files
	// schema<year: int64 not null, month: string not null>
	// 2024 01
	// 2024 02
	// 2025 01
}

func ExampleRead() {
	root := orders()
	defer os.RemoveAll(root)

	d, err := dataset.Discover(root, nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	t, err := dataset.Read(d, &dataset.ReadOptions{
		Open: func(path string) (*array.Table, error) {
			return ndjson.ReadFile(path, nil)
		},
	})
	if err != nil {
		fmt.Println(err)
		return
	}

	// The year and the month are columns now, filled in from the paths, and no
	// file holds either of them.
	fmt.Println(t.Schema)
	fmt.Println(t.NumRows(), "rows")
	// Output:
	// schema<sym: string not null, qty: int64 not null, year: int64 not null, month: string not null>
	// 4 rows
}

func ExampleDataset_Select() {
	root := orders()
	defer os.RemoveAll(root)

	d, err := dataset.Discover(root, nil)
	if err != nil {
		fmt.Println(err)
		return
	}

	// This is the whole point of the layout. The values are in hand from the
	// paths, so the files of the other months are never opened.
	january := d.Select(func(f dataset.File) bool {
		return d.Value(f, "month").Text == "01"
	})
	fmt.Println(january.Len(), "of", d.Len(), "files")

	t, err := dataset.Read(january, &dataset.ReadOptions{
		Open: func(path string) (*array.Table, error) {
			return ndjson.ReadFile(path, nil)
		},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(t.NumRows(), "rows")
	// Output:
	// 2 of 3 files
	// 2 rows
}

func ExampleDiscover_types() {
	root := orders()
	defer os.RemoveAll(root)

	// The month is 01, and 01 is not what the number 1 prints as, so inference
	// leaves it as text rather than lose the name of the directory. A caller
	// who wants the number says so, and then the leading zero is allowed.
	d, err := dataset.Discover(root, &dataset.Options{
		Types: map[string]dtype.DataType{"month": dtype.Int8},
	})
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(d.Schema)
	// Output:
	// schema<year: int64 not null, month: int8 not null>
}
