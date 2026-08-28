package kuma_test

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/parquet"
)

// parquetFile is one of the files the parquet package tests read. They are
// there rather than here because that package is where a reader is checked
// against files somebody else wrote, and a second copy of a binary file is a
// second thing to keep in step.
func parquetFile(tb testing.TB, name string) string {
	tb.Helper()
	return filepath.Join("parquet", "testdata", name)
}

func TestReadParquetFile(t *testing.T) {
	f, err := kuma.ReadParquetFile(parquetFile(t, "chunks.parquet"), nil)
	if err != nil {
		t.Fatalf("ReadParquetFile: %v", err)
	}

	if got, want := f.Names(), []string{"code", "n"}; !equalStrings(got, want) {
		t.Errorf("got names %v, want %v", got, want)
	}
	if rows, cols := f.Shape(); rows != 6 || cols != 2 {
		t.Errorf("got %d rows by %d cols, want 6 by 2", rows, cols)
	}

	// The file wrote both of these as indices into a dictionary, which is what
	// pyarrow does to nearly every column. A caller asking for the values gets
	// the values, so a series is the column and not the encoding.
	code, err := f.Series[string]("code")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := code.Value(0); got != "GB" {
		t.Errorf("got code %q, want GB", got)
	}

	n, err := f.Series[int64]("n")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := n.Value(5); got != 5 {
		t.Errorf("got n %d, want 5", got)
	}
}

func TestReadParquet(t *testing.T) {
	b, err := os.ReadFile(parquetFile(t, "dictionary.parquet"))
	if err != nil {
		t.Fatal(err)
	}

	f, err := kuma.ReadParquet(bytes.NewReader(b), int64(len(b)), nil)
	if err != nil {
		t.Fatalf("ReadParquet: %v", err)
	}
	if got := f.NumRows(); got != 1000 {
		t.Errorf("got %d rows, want 1000", got)
	}

	// The nulls came through, which is what a frame filtering them out reads.
	kept, err := f.DropNulls()
	if err != nil {
		t.Fatalf("DropNulls: %v", err)
	}
	if got := kept.NumRows(); got != 857 {
		t.Errorf("got %d rows after dropping the nulls, want 857", got)
	}
}

func TestReadParquetProjection(t *testing.T) {
	// The file holds a decimal column, which is not assembled yet, so naming
	// the columns is the difference between a frame and an error.
	path := parquetFile(t, "alltypes.parquet")
	f, err := kuma.ReadParquetFile(path, &parquet.Options{Columns: []string{"name", "total"}})
	if err != nil {
		t.Fatalf("ReadParquetFile: %v", err)
	}

	if got := f.Schema().String(); got != "schema<name: string, total: int64 not null>" {
		t.Errorf("got %s", got)
	}
	if _, err := kuma.ReadParquetFile(path, nil); err == nil {
		t.Error("reading the whole file worked, so the projection proves nothing")
	}
}

func TestReadParquetDictionary(t *testing.T) {
	f, err := kuma.ReadParquetFile(parquetFile(t, "chunks.parquet"), &parquet.Options{Dictionary: true})
	if err != nil {
		t.Fatalf("ReadParquetFile: %v", err)
	}

	if got := f.Schema().String(); got != "schema<code: dictionary<int32, string> not null, n: dictionary<int32, int64> not null>" {
		t.Errorf("got %s", got)
	}

	// A group by reads through the encoding, so an encoded frame answers the
	// same question as a decoded one.
	g, err := f.GroupBy("code")
	if err != nil {
		t.Fatalf("GroupBy: %v", err)
	}
	if got := g.NumGroups(); got != 5 {
		t.Errorf("got %d groups, want 5", got)
	}
}

func TestReadParquetErrors(t *testing.T) {
	if _, err := kuma.ReadParquetFile(filepath.Join(t.TempDir(), "gone.parquet"), nil); err == nil {
		t.Error("reading a file that is not there worked")
	}

	junk := []byte("this is not a parquet file at all")
	if _, err := kuma.ReadParquet(bytes.NewReader(junk), int64(len(junk)), nil); err == nil {
		t.Error("reading a file of junk worked")
	}

	_, err := kuma.ReadParquetFile(parquetFile(t, "chunks.parquet"), &parquet.Options{
		Columns: []string{"code", "nope"},
	})
	if err == nil || !strings.Contains(err.Error(), `"nope"`) {
		t.Errorf("got %v, want an error naming nope", err)
	}

	// A frame cannot hold two columns of the same name, which is refused where
	// it is found rather than where it was asked for.
	if _, err := kuma.ReadParquetFile(parquetFile(t, "chunks.parquet"), &parquet.Options{
		Columns: []string{"n", "n"},
	}); err == nil {
		t.Error("a frame with two columns called n worked")
	}
}
