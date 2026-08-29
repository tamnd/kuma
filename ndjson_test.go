package kuma_test

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/ndjson"
)

func TestReadNDJSON(t *testing.T) {
	in := `{"sym":"AAPL","qty":100,"px":1.5,"live":true}
{"sym":"MSFT","qty":null,"px":2.5,"live":false}
`

	f, err := kuma.ReadNDJSON(strings.NewReader(in), nil)
	if err != nil {
		t.Fatalf("ReadNDJSON: %v", err)
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

func TestReadNDJSONOptions(t *testing.T) {
	in := `{"sym":"AAPL","qty":100,"venue":"XNAS"}
{"sym":"MSFT","qty":200,"venue":"XNAS"}
`

	f, err := kuma.ReadNDJSON(strings.NewReader(in), &ndjson.Options{
		Columns: []string{"qty", "sym"},
		Types:   map[string]dtype.DataType{"qty": dtype.Int32},
	})
	if err != nil {
		t.Fatalf("ReadNDJSON: %v", err)
	}

	if got := f.Schema().String(); got != "schema<qty: int32 not null, sym: string not null>" {
		t.Errorf("got %s", got)
	}
}

func TestReadNDJSONErrors(t *testing.T) {
	if _, err := kuma.ReadNDJSON(strings.NewReader(""), nil); !errors.Is(err, ndjson.ErrNoData) {
		t.Errorf("got %v, want ErrNoData", err)
	}

	_, err := kuma.ReadNDJSON(strings.NewReader("{\"a\":1}\nnot json\n"), nil)
	if !errors.Is(err, ndjson.ErrSyntax) {
		t.Errorf("got %v, want ErrSyntax", err)
	}
}

func TestReadNDJSONFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trades.ndjson")
	if err := os.WriteFile(path, []byte("{\"sym\":\"AAPL\",\"qty\":100}\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	f, err := kuma.ReadNDJSONFile(path, nil)
	if err != nil {
		t.Fatalf("ReadNDJSONFile: %v", err)
	}
	if got := f.NumRows(); got != 1 {
		t.Errorf("got %d rows, want 1", got)
	}

	if _, err := kuma.ReadNDJSONFile(filepath.Join(t.TempDir(), "gone.ndjson"), nil); err == nil {
		t.Error("reading a file that is not there worked")
	}
}

func TestReadNDJSONIsAFrameLikeAnyOther(t *testing.T) {
	in := `{"sym":"AAPL","qty":100}
{"sym":"MSFT"}
{"sym":"GOOG","qty":300}
`

	f, err := kuma.ReadNDJSON(strings.NewReader(in), nil)
	if err != nil {
		t.Fatalf("ReadNDJSON: %v", err)
	}

	kept, err := f.DropNulls()
	if err != nil {
		t.Fatalf("DropNulls: %v", err)
	}
	if got := kept.NumRows(); got != 2 {
		t.Errorf("got %d rows, want 2", got)
	}
}

func TestWriteNDJSON(t *testing.T) {
	const in = "{\"sym\":\"AAPL\",\"qty\":100}\n{\"sym\":\"MSFT\",\"qty\":null}\n"

	f, err := kuma.ReadNDJSON(strings.NewReader(in), nil)
	if err != nil {
		t.Fatalf("ReadNDJSON: %v", err)
	}

	var buf strings.Builder
	if err := f.WriteNDJSON(&buf, nil); err != nil {
		t.Fatalf("WriteNDJSON: %v", err)
	}
	if buf.String() != in {
		t.Errorf("got\n%s\nwant\n%s", buf.String(), in)
	}
}

func TestWriteNDJSONOptions(t *testing.T) {
	f, err := kuma.NewFrame(
		kuma.NewSeries("sym", "AAPL", "MSFT").Column(),
		kuma.NewSeries("px", 1.5, 2.0).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	var buf strings.Builder
	err = f.WriteNDJSON(&buf, &ndjson.WriteOptions{
		Names:     []string{"symbol", "price"},
		Precision: 2,
	})
	if err != nil {
		t.Fatalf("WriteNDJSON: %v", err)
	}
	want := "{\"symbol\":\"AAPL\",\"price\":1.50}\n{\"symbol\":\"MSFT\",\"price\":2.00}\n"
	if buf.String() != want {
		t.Errorf("got\n%s\nwant\n%s", buf.String(), want)
	}
}

func TestWriteNDJSONErrors(t *testing.T) {
	f, err := kuma.NewFrame(kuma.NewSeries("sym", "AAPL").Column())
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	err = f.WriteNDJSON(&strings.Builder{}, &ndjson.WriteOptions{Names: []string{"a", "b"}})
	if !errors.Is(err, ndjson.ErrNames) {
		t.Errorf("got %v, want ErrNames", err)
	}
}

func TestWriteNDJSONFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trades.ndjson")
	f, err := kuma.ReadNDJSON(strings.NewReader("{\"sym\":\"AAPL\",\"qty\":100}\n"), nil)
	if err != nil {
		t.Fatalf("ReadNDJSON: %v", err)
	}

	if err = f.WriteNDJSONFile(path, nil); err != nil {
		t.Fatalf("WriteNDJSONFile: %v", err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	if want := "{\"sym\":\"AAPL\",\"qty\":100}\n"; string(got) != want {
		t.Errorf("got %q, want %q", got, want)
	}

	if err := f.WriteNDJSONFile(filepath.Join(path, "under", "a", "file.ndjson"), nil); err == nil {
		t.Error("writing under a file that is not a directory worked")
	}
}

// TestWriteNDJSONRoundTrip writes a frame out and reads it back, which is the
// promise the pair of them makes.
func TestWriteNDJSONRoundTrip(t *testing.T) {
	in := `{"sym":"AAPL","qty":100,"px":1.5,"live":true}
{"sym":"MSFT","qty":null,"px":2.25,"live":false}
{"sym":"GOOG","qty":300,"px":null,"live":true}
`

	f, err := kuma.ReadNDJSON(strings.NewReader(in), nil)
	if err != nil {
		t.Fatalf("ReadNDJSON: %v", err)
	}

	var buf strings.Builder
	if err = f.WriteNDJSON(&buf, nil); err != nil {
		t.Fatalf("WriteNDJSON: %v", err)
	}
	back, err := kuma.ReadNDJSON(strings.NewReader(buf.String()), nil)
	if err != nil {
		t.Fatalf("ReadNDJSON: %v", err)
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

// TestNDJSONFromCSV is the pair of readers meeting: a file that came in as one
// format goes out as the other, which is most of what anyone wants a converter
// for.
func TestNDJSONFromCSV(t *testing.T) {
	f, err := kuma.ReadCSV(strings.NewReader("sym,qty\nAAPL,100\nMSFT,\n"), nil)
	if err != nil {
		t.Fatalf("ReadCSV: %v", err)
	}

	var buf strings.Builder
	if err := f.WriteNDJSON(&buf, nil); err != nil {
		t.Fatalf("WriteNDJSON: %v", err)
	}

	want := "{\"sym\":\"AAPL\",\"qty\":100}\n{\"sym\":\"MSFT\",\"qty\":null}\n"
	if buf.String() != want {
		t.Errorf("got\n%s\nwant\n%s", buf.String(), want)
	}
}
