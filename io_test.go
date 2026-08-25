package kuma_test

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/csv"
	"github.com/tamnd/kuma/dtype"
)

func TestReadCSV(t *testing.T) {
	in := "sym,qty,px,live\nAAPL,100,1.5,true\nMSFT,,2.5,false\n"

	f, err := kuma.ReadCSV(strings.NewReader(in), nil)
	if err != nil {
		t.Fatalf("ReadCSV: %v", err)
	}

	if got, want := f.Names(), []string{"sym", "qty", "px", "live"}; !equalStrings(got, want) {
		t.Errorf("got names %v, want %v", got, want)
	}
	if rows, cols := f.Shape(); rows != 2 || cols != 4 {
		t.Errorf("got %d rows by %d cols, want 2 by 4", rows, cols)
	}

	// The columns come back as themselves rather than as text, so a series is
	// the values and not a parse waiting to happen.
	qty, err := f.Series[int64]("qty")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if !qty.IsNull(1) {
		t.Error("the missing quantity is not missing")
	}
	if got := qty.Value(0); got != 100 {
		t.Errorf("got qty %d, want 100", got)
	}

	sym, err := f.Series[string]("sym")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := sym.Value(1); got != "MSFT" {
		t.Errorf("got sym %q, want MSFT", got)
	}
}

func TestReadCSVOptions(t *testing.T) {
	in := "AAPL|100\nMSFT|200\n"

	f, err := kuma.ReadCSV(strings.NewReader(in), &csv.Options{
		Delimiter: '|',
		NoHeader:  true,
		Names:     []string{"sym", "qty"},
		Types:     map[string]dtype.DataType{"qty": dtype.Int32},
	})
	if err != nil {
		t.Fatalf("ReadCSV: %v", err)
	}

	if got := f.Schema().String(); got != "schema<sym: string not null, qty: int32 not null>" {
		t.Errorf("got %s", got)
	}
}

func TestReadCSVErrors(t *testing.T) {
	if _, err := kuma.ReadCSV(strings.NewReader(""), nil); !errors.Is(err, csv.ErrNoData) {
		t.Errorf("got %v, want ErrNoData", err)
	}

	// A file whose header repeats a name would be a frame with two columns of
	// the same name, which is refused where it is found rather than here.
	_, err := kuma.ReadCSV(strings.NewReader("a,a\n1,2\n"), nil)
	if !errors.Is(err, csv.ErrNames) {
		t.Errorf("got %v, want ErrNames", err)
	}
}

func TestReadCSVFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trades.csv")
	if err := os.WriteFile(path, []byte("sym,qty\nAAPL,100\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	f, err := kuma.ReadCSVFile(path, nil)
	if err != nil {
		t.Fatalf("ReadCSVFile: %v", err)
	}
	if got := f.NumRows(); got != 1 {
		t.Errorf("got %d rows, want 1", got)
	}

	if _, err := kuma.ReadCSVFile(filepath.Join(t.TempDir(), "gone.csv"), nil); err == nil {
		t.Error("reading a file that is not there worked")
	}
}

func TestReadCSVIsAFrameLikeAnyOther(t *testing.T) {
	in := "sym,qty\nAAPL,100\nMSFT,\nGOOG,300\n"

	f, err := kuma.ReadCSV(strings.NewReader(in), nil)
	if err != nil {
		t.Fatalf("ReadCSV: %v", err)
	}

	kept, err := f.DropNulls()
	if err != nil {
		t.Fatalf("DropNulls: %v", err)
	}
	if got := kept.NumRows(); got != 2 {
		t.Errorf("got %d rows, want 2", got)
	}
}

func TestWriteCSV(t *testing.T) {
	f, err := kuma.ReadCSV(strings.NewReader("sym,qty\nAAPL,100\nMSFT,\n"), nil)
	if err != nil {
		t.Fatalf("ReadCSV: %v", err)
	}

	var buf strings.Builder
	if err := f.WriteCSV(&buf, nil); err != nil {
		t.Fatalf("WriteCSV: %v", err)
	}
	if want := "sym,qty\nAAPL,100\nMSFT,\n"; buf.String() != want {
		t.Errorf("got\n%s\nwant\n%s", buf.String(), want)
	}
}

func TestWriteCSVOptions(t *testing.T) {
	f, err := kuma.NewFrame(
		kuma.NewSeries("sym", "AAPL", "MSFT").Column(),
		kuma.NewSeries("px", 1.5, 2.0).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	var buf strings.Builder
	err = f.WriteCSV(&buf, &csv.WriteOptions{
		Delimiter: ';',
		NoHeader:  true,
		Precision: 2,
	})
	if err != nil {
		t.Fatalf("WriteCSV: %v", err)
	}
	if want := "AAPL;1.50\nMSFT;2.00\n"; buf.String() != want {
		t.Errorf("got\n%s\nwant\n%s", buf.String(), want)
	}
}

func TestWriteCSVErrors(t *testing.T) {
	f, err := kuma.NewFrame(kuma.NewSeries("sym", "AAPL").Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	err = f.WriteCSV(&strings.Builder{}, &csv.WriteOptions{Delimiter: '\n'})
	if !errors.Is(err, csv.ErrDelimiter) {
		t.Errorf("got %v, want ErrDelimiter", err)
	}
}

func TestWriteCSVFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trades.csv")
	f, err := kuma.ReadCSV(strings.NewReader("sym,qty\nAAPL,100\n"), nil)
	if err != nil {
		t.Fatalf("ReadCSV: %v", err)
	}

	if err = f.WriteCSVFile(path, nil); err != nil {
		t.Fatalf("WriteCSVFile: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if want := "sym,qty\nAAPL,100\n"; string(got) != want {
		t.Errorf("got %q, want %q", got, want)
	}

	if err := f.WriteCSVFile(filepath.Join(path, "under", "a", "file.csv"), nil); err == nil {
		t.Error("writing under a file that is not a directory worked")
	}
}

// TestWriteCSVRoundTrip writes a frame out and reads it back, which is the
// promise the pair of them makes.
func TestWriteCSVRoundTrip(t *testing.T) {
	f, err := kuma.ReadCSV(strings.NewReader(
		"sym,qty,px,live\nAAPL,100,1.5,true\nMSFT,,2.25,false\nGOOG,300,,true\n"), nil)
	if err != nil {
		t.Fatalf("ReadCSV: %v", err)
	}

	var buf strings.Builder
	if err = f.WriteCSV(&buf, nil); err != nil {
		t.Fatalf("WriteCSV: %v", err)
	}
	back, err := kuma.ReadCSV(strings.NewReader(buf.String()), nil)
	if err != nil {
		t.Fatalf("ReadCSV: %v", err)
	}

	if got, want := back.Schema().String(), f.Schema().String(); got != want {
		t.Errorf("got %s, want %s", got, want)
	}
	if got, want := back.String(), f.String(); got != want {
		t.Errorf("got\n%s\nwant\n%s", got, want)
	}

	qty, err := back.Series[int64]("qty")
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	if got := qty.DropNulls().Values(); len(got) != 2 || got[0] != 100 || got[1] != 300 {
		t.Errorf("got %v, want [100 300]", got)
	}
}

// equalStrings reports whether two name lists are the same.
func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
