package kuma_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/csv"
	"github.com/tamnd/kuma/dataset"
	"github.com/tamnd/kuma/parquet"
)

// writeTree writes a file at each of the paths, which are slash separated and
// relative to a new temporary directory, and returns that directory.
func writeTree(t *testing.T, files map[string]string) string {
	t.Helper()

	root := t.TempDir()
	for rel, data := range files {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o750); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func TestReadDataset(t *testing.T) {
	root := writeTree(t, map[string]string{
		"year=2024/month=01/part-0.ndjson": `{"sym":"AAPL","qty":100}` + "\n",
		"year=2024/month=02/part-0.ndjson": `{"sym":"MSFT","qty":50}` + "\n" + `{"sym":"GOOG","qty":25}` + "\n",
		"year=2025/month=01/part-0.ndjson": `{"sym":"AAPL","qty":300}` + "\n",
	})

	f, err := kuma.ReadDataset(root, nil)
	if err != nil {
		t.Fatalf("ReadDataset: %v", err)
	}

	if got, want := f.Names(), []string{"sym", "qty", "year", "month"}; !equalStrings(got, want) {
		t.Errorf("got names %v, want %v", got, want)
	}
	if rows, cols := f.Shape(); rows != 4 || cols != 4 {
		t.Errorf("got %d rows by %d cols, want 4 by 4", rows, cols)
	}

	// The year is in no file. It came out of the directory names.
	year, err := f.Series[int64]("year")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	for i, want := range []int64{2024, 2024, 2024, 2025} {
		if got := year.Value(i); got != want {
			t.Errorf("row %d is from %d, want %d", i, got, want)
		}
	}

	month, err := f.Series[string]("month")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := month.Value(3); got != "01" {
		t.Errorf("the last month is %q, want 01", got)
	}
}

func TestReadDatasetFiles(t *testing.T) {
	root := writeTree(t, map[string]string{
		"month=01/part-0.csv": "sym,qty\nAAPL,100\n",
		"month=02/part-0.csv": "sym,qty\nMSFT,50\nGOOG,25\n",
		"month=03/part-0.csv": "sym,qty\nNVDA,10\n",
	})

	d, err := dataset.Discover(root, nil)
	if err != nil {
		t.Fatalf("Discover: %v", err)
	}
	february := d.Select(func(f dataset.File) bool {
		return d.Value(f, "month").Text == "02"
	})

	f, err := kuma.ReadDatasetFiles(february)
	if err != nil {
		t.Fatalf("ReadDatasetFiles: %v", err)
	}
	if got := f.NumRows(); got != 2 {
		t.Errorf("read %d rows, want the 2 in February", got)
	}
	if got, want := f.Names(), []string{"sym", "qty", "month"}; !equalStrings(got, want) {
		t.Errorf("got names %v, want %v", got, want)
	}
}

func TestReadDatasetFormats(t *testing.T) {
	cases := []struct {
		name string
		file string
		data string
	}{
		{"csv", "part-0.csv", "sym,qty\nAAPL,100\n"},
		{"tsv", "part-0.tsv", "sym\tqty\nAAPL\t100\n"},
		{"ndjson", "part-0.ndjson", `{"sym":"AAPL","qty":100}` + "\n"},
		{"jsonl", "part-0.jsonl", `{"sym":"AAPL","qty":100}` + "\n"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			root := writeTree(t, map[string]string{"month=01/" + c.file: c.data})

			f, err := kuma.ReadDataset(root, nil)
			if err != nil {
				t.Fatalf("ReadDataset: %v", err)
			}
			if got, want := f.Names(), []string{"sym", "qty", "month"}; !equalStrings(got, want) {
				t.Errorf("got names %v, want %v", got, want)
			}
			if got := f.NumRows(); got != 1 {
				t.Errorf("read %d rows, want 1", got)
			}
		})
	}
}

func TestReadDatasetParquet(t *testing.T) {
	root := t.TempDir()
	dir := filepath.Join(root, "month=01")
	if err := os.MkdirAll(dir, 0o750); err != nil {
		t.Fatal(err)
	}

	src, err := csv.Read(strings.NewReader("sym,qty\nAAPL,100\nMSFT,50\n"), nil)
	if err != nil {
		t.Fatalf("csv.Read: %v", err)
	}
	if err = parquet.WriteFile(filepath.Join(dir, "part-0.parquet"), src, nil); err != nil {
		t.Fatalf("parquet.WriteFile: %v", err)
	}

	f, err := kuma.ReadDataset(root, nil)
	if err != nil {
		t.Fatalf("ReadDataset: %v", err)
	}
	if got, want := f.Names(), []string{"sym", "qty", "month"}; !equalStrings(got, want) {
		t.Errorf("got names %v, want %v", got, want)
	}
	if got := f.NumRows(); got != 2 {
		t.Errorf("read %d rows, want 2", got)
	}
}

func TestReadDatasetErrors(t *testing.T) {
	t.Run("a tree that is not there", func(t *testing.T) {
		if _, err := kuma.ReadDataset(filepath.Join(t.TempDir(), "gone"), nil); err == nil {
			t.Error("read a directory that is not there")
		}
	})

	t.Run("a format with no reader", func(t *testing.T) {
		root := writeTree(t, map[string]string{"month=01/part-0.orc": "not a format this knows"})

		_, err := kuma.ReadDataset(root, nil)
		if err == nil {
			t.Fatal("read a format that has no reader")
		}
		if !strings.Contains(err.Error(), `no reader for a ".orc" file`) {
			t.Errorf("got %q, want it to say what it could not read", err)
		}
		if !strings.Contains(err.Error(), "part-0.orc") {
			t.Errorf("got %q, want it to say which file", err)
		}
	})

	t.Run("a file with no extension at all", func(t *testing.T) {
		root := writeTree(t, map[string]string{"month=01/part-0": "sym,qty\nAAPL,100\n"})

		if _, err := kuma.ReadDataset(root, nil); err == nil {
			t.Error("read a file with nothing to say what it is")
		}
	})

	t.Run("a tree partitioned two ways", func(t *testing.T) {
		root := writeTree(t, map[string]string{
			"month=01/part-0.csv":         "sym\nAAPL\n",
			"year=2024/month=01/part.csv": "sym\nMSFT\n",
		})

		if _, err := kuma.ReadDataset(root, nil); err == nil {
			t.Error("read two datasets as one")
		}
	})
}
