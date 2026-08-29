package dataset

import (
	"fmt"
	"strconv"
	"testing"
	"testing/fstest"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// year builds a filesystem of one file per day of a year, which is the shape a
// dataset written daily has after a year of it.
func year() fstest.MapFS {
	fsys := make(fstest.MapFS, 365)
	for month := 1; month <= 12; month++ {
		for day := 1; day <= 30; day++ {
			p := fmt.Sprintf("year=2024/month=%02d/day=%02d/part-0.parquet", month, day)
			fsys[p] = &fstest.MapFile{Data: []byte("rows")}
		}
	}
	return fsys
}

func BenchmarkParse(b *testing.B) {
	cases := []struct {
		name string
		rel  string
	}{
		{"Flat", "part-0.parquet"},
		{"One", "year=2024/part-0.parquet"},
		{"Three", "year=2024/month=03/day=17/part-0.parquet"},
		{"Escaped", "city=New%20York/day=17/part-0.parquet"},
	}

	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, _, err := parse(c.rel); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkInfer(b *testing.B) {
	vals := make([]Value, 365)
	for i := range vals {
		vals[i] = Value{Text: strconv.Itoa(i)}
	}

	b.ReportAllocs()
	for b.Loop() {
		if got := infer(vals); !dtype.Equal(got, dtype.Int64) {
			b.Fatalf("inferred %s", got)
		}
	}
}

func BenchmarkDiscover(b *testing.B) {
	fsys := year()

	b.ReportAllocs()
	for b.Loop() {
		d, err := DiscoverFS(fsys, nil)
		if err != nil {
			b.Fatal(err)
		}
		if d.Len() != 360 {
			b.Fatalf("found %d files", d.Len())
		}
	}
}

func BenchmarkSelect(b *testing.B) {
	d, err := DiscoverFS(year(), nil)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	for b.Loop() {
		got := d.Select(func(f File) bool { return d.Value(f, "month").Text == "03" })
		if got.Len() != 30 {
			b.Fatalf("kept %d files", got.Len())
		}
	}
}

func BenchmarkRepeat(b *testing.B) {
	const rows = 65536

	cases := []struct {
		name string
		dt   dtype.DataType
		text string
	}{
		{"Int64", dtype.Int64, "2024"},
		{"Int32", dtype.Int32, "2024"},
		{"Float64", dtype.Float64, "1.5"},
		{"Bool", dtype.Bool, "true"},
		{"String", dtype.String, "2024-03-17"},
	}

	for _, c := range cases {
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				a, err := repeat(c.dt, Value{Text: c.text}, rows)
				if err != nil {
					b.Fatal(err)
				}
				if a.Len() != rows {
					b.Fatalf("built %d rows", a.Len())
				}
			}
		})
	}
}

func BenchmarkRepeatNull(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		if _, err := repeat(dtype.Int64, Value{Null: true}, 65536); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRead(b *testing.B) {
	fsys := make(fstest.MapFS, 12)
	for month := 1; month <= 12; month++ {
		fsys[fmt.Sprintf("year=2024/month=%02d/part-0.parquet", month)] = &fstest.MapFile{}
	}
	d, err := DiscoverFS(fsys, nil)
	if err != nil {
		b.Fatal(err)
	}

	// One table stands in for every file, since what is being measured is what
	// this package does with them and not the reading of a format.
	const rows = 8192
	vals := make([]int64, rows)
	for i := range vals {
		vals[i] = int64(i)
	}
	c, err := array.NewChunked(dtype.Int64, array.Of(vals...))
	if err != nil {
		b.Fatal(err)
	}
	one := &array.Table{
		Schema:  dtype.Schema{Fields: []dtype.Field{{Name: "qty", Type: dtype.Int64}}},
		Columns: []*array.Chunked{c},
	}
	opts := &ReadOptions{Open: func(string) (*array.Table, error) { return one, nil }}

	b.SetBytes(int64(len(d.Files)) * rows * 8)
	b.ReportAllocs()
	for b.Loop() {
		t, err := Read(d, opts)
		if err != nil {
			b.Fatal(err)
		}
		if t.NumRows() != len(d.Files)*rows {
			b.Fatalf("read %d rows", t.NumRows())
		}
	}
}

func BenchmarkReadOmitPartitions(b *testing.B) {
	fsys := make(fstest.MapFS, 12)
	for month := 1; month <= 12; month++ {
		fsys[fmt.Sprintf("year=2024/month=%02d/part-0.parquet", month)] = &fstest.MapFile{}
	}
	d, err := DiscoverFS(fsys, nil)
	if err != nil {
		b.Fatal(err)
	}

	const rows = 8192
	vals := make([]int64, rows)
	c, err := array.NewChunked(dtype.Int64, array.Of(vals...))
	if err != nil {
		b.Fatal(err)
	}
	one := &array.Table{
		Schema:  dtype.Schema{Fields: []dtype.Field{{Name: "qty", Type: dtype.Int64}}},
		Columns: []*array.Chunked{c},
	}
	opts := &ReadOptions{
		Open:           func(string) (*array.Table, error) { return one, nil },
		OmitPartitions: true,
	}

	b.SetBytes(int64(len(d.Files)) * rows * 8)
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Read(d, opts); err != nil {
			b.Fatal(err)
		}
	}
}
