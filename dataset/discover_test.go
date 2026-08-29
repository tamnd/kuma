package dataset

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"testing/fstest"

	"github.com/tamnd/kuma/dtype"
)

// tree builds a filesystem holding a file at each of the paths, since what is
// in the files does not matter to discovery.
func tree(paths ...string) fstest.MapFS {
	fsys := make(fstest.MapFS, len(paths))
	for _, p := range paths {
		fsys[p] = &fstest.MapFile{Data: []byte(p)}
	}
	return fsys
}

// discover runs DiscoverFS and fails the test if it did not work.
func discover(t *testing.T, fsys fs.FS, opts *Options) *Dataset {
	t.Helper()
	d, err := DiscoverFS(fsys, opts)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// paths is the file list of a dataset, which is what most of these compare.
func paths(d *Dataset) []string {
	out := make([]string, len(d.Files))
	for i, f := range d.Files {
		out[i] = f.Path
	}
	return out
}

func TestDiscover(t *testing.T) {
	fsys := tree(
		"year=2024/month=01/part-0.parquet",
		"year=2024/month=02/part-0.parquet",
		"year=2025/month=01/part-0.parquet",
		"year=2025/month=01/part-1.parquet",
	)
	d := discover(t, fsys, nil)

	if d.Len() != 4 {
		t.Fatalf("found %d files, want 4", d.Len())
	}
	if got := d.Schema.String(); got != "schema<year: int64 not null, month: string not null>" {
		t.Errorf("schema %s", got)
	}
	if d.Root != "" {
		t.Errorf("root %q, want it empty for a dataset found in an fs.FS", d.Root)
	}

	// The walk is sorted, so the files come back in path order.
	want := []string{
		"year=2024/month=01/part-0.parquet",
		"year=2024/month=02/part-0.parquet",
		"year=2025/month=01/part-0.parquet",
		"year=2025/month=01/part-1.parquet",
	}
	if got := paths(d); !slices.Equal(got, want) {
		t.Errorf("files %q, want %q", got, want)
	}

	if got := d.Value(d.Files[2], "year"); got.Text != "2025" {
		t.Errorf("year of the third file is %v, want 2025", got)
	}
	if got := d.Value(d.Files[2], "month"); got.Text != "01" {
		t.Errorf("month of the third file is %v, want 01", got)
	}
	if got := d.Value(d.Files[0], "day"); !got.Null {
		t.Errorf("a column that is not there gave %v, want a missing value", got)
	}
}

func TestDiscoverUnpartitioned(t *testing.T) {
	d := discover(t, tree("a.parquet", "b.parquet"), nil)

	if d.Schema.Len() != 0 {
		t.Errorf("schema %s, want no partition columns", d.Schema)
	}
	for _, f := range d.Files {
		if f.Values != nil {
			t.Errorf("%s has values %v, want none", f.Path, f.Values)
		}
	}
}

func TestDiscoverSkipsTheLeftovers(t *testing.T) {
	fsys := tree(
		"_SUCCESS",
		".hidden",
		"year=2024/_committed_1",
		"year=2024/.part-0.parquet.crc",
		"year=2024/part-0.parquet",
		"_temporary/year=2024/part-0.parquet",
	)

	d := discover(t, fsys, nil)
	want := []string{"year=2024/part-0.parquet"}
	if got := paths(d); !slices.Equal(got, want) {
		t.Errorf("files %q, want %q", got, want)
	}

	// With Hidden the tree stops being a dataset, since _SUCCESS sits in the
	// root and has no partition directories above it.
	if _, err := DiscoverFS(fsys, &Options{Hidden: true}); !errors.Is(err, ErrLayout) {
		t.Errorf("counting the hidden files gave %v, want ErrLayout", err)
	}

	// Where they are all under the partitions, counting them works and finds
	// them.
	hidden := tree("year=2024/_SUCCESS", "year=2024/part-0.parquet")
	d = discover(t, hidden, &Options{Hidden: true})
	if d.Len() != 2 {
		t.Errorf("found %d files, want both of them", d.Len())
	}
}

func TestDiscoverExtension(t *testing.T) {
	fsys := tree(
		"year=2024/part-0.parquet",
		"year=2024/schema.json",
		"year=2024/part-1.parquet",
	)

	d := discover(t, fsys, &Options{Extension: ".parquet"})
	want := []string{"year=2024/part-0.parquet", "year=2024/part-1.parquet"}
	if got := paths(d); !slices.Equal(got, want) {
		t.Errorf("files %q, want %q", got, want)
	}

	if _, err := DiscoverFS(fsys, &Options{Extension: ".orc"}); !errors.Is(err, ErrNoData) {
		t.Errorf("an extension nothing has gave %v, want ErrNoData", err)
	}
}

func TestDiscoverTypes(t *testing.T) {
	fsys := tree(
		"year=2024/flag=true/part-0.parquet",
		"year=2025/flag=false/part-0.parquet",
	)

	d := discover(t, fsys, &Options{Types: map[string]dtype.DataType{
		"year": dtype.Uint16,
		"flag": dtype.Bool,
	}})
	if got := d.Schema.String(); got != "schema<year: uint16 not null, flag: bool not null>" {
		t.Errorf("schema %s", got)
	}

	// Without them the booleans are text, since inference does not read them.
	d = discover(t, fsys, nil)
	if got := d.Schema.String(); got != "schema<year: int64 not null, flag: string not null>" {
		t.Errorf("schema %s", got)
	}
}

func TestDiscoverNullable(t *testing.T) {
	fsys := tree(
		"region=us/part-0.parquet",
		"region=__HIVE_DEFAULT_PARTITION__/part-0.parquet",
	)

	d := discover(t, fsys, nil)
	if !d.Schema.Fields[0].Nullable {
		t.Error("a column with a default partition in it is not nullable")
	}

	d = discover(t, tree("region=us/part-0.parquet"), nil)
	if d.Schema.Fields[0].Nullable {
		t.Error("a column with every value in it is nullable")
	}
}

func TestDiscoverErrors(t *testing.T) {
	cases := []struct {
		name string
		fsys fs.FS
		opts *Options
		want error
		msg  string
	}{
		{
			name: "an empty tree",
			fsys: tree(),
			want: ErrNoData,
		},
		{
			name: "a directory that is not key=value",
			fsys: tree("2024/part-0.parquet"),
			want: ErrLayout,
			msg:  `dataset: 2024/part-0.parquet: "2024" is not a key=value directory: not a Hive partitioned dataset`,
		},
		{
			name: "two files partitioned differently",
			fsys: tree("year=2024/part-0.parquet", "month=01/part-0.parquet"),
			want: ErrLayout,
		},
		{
			name: "one file partitioned deeper than another",
			fsys: tree("year=2024/month=01/part-0.parquet", "year=2024/part-0.parquet"),
			want: ErrLayout,
		},
		{
			name: "a file in the root beside a partitioned one",
			fsys: tree("part-0.parquet", "year=2024/part-0.parquet"),
			want: ErrLayout,
		},
		{
			name: "a type for a column that is not there",
			fsys: tree("year=2024/part-0.parquet"),
			opts: &Options{Types: map[string]dtype.DataType{"month": dtype.Int64}},
			want: ErrNoColumn,
			msg:  `dataset: "month" is not partitioned on: no such partition column`,
		},
		{
			name: "a type a path cannot be read into",
			fsys: tree("day=1/part-0.parquet"),
			opts: &Options{Types: map[string]dtype.DataType{"day": dtype.Date32}},
			want: ErrUnsupportedType,
			msg:  `dataset: partition "day": cannot read a path into a date32 column: unsupported partition type`,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := DiscoverFS(c.fsys, c.opts)
			if !errors.Is(err, c.want) {
				t.Fatalf("got %v, want %v", err, c.want)
			}
			if c.msg != "" && err.Error() != c.msg {
				t.Errorf("got %q, want %q", err.Error(), c.msg)
			}
		})
	}
}

func TestDiscoverLayoutMessageNamesBothFiles(t *testing.T) {
	fsys := tree("year=2024/part-0.parquet", "year=2025/month=01/part-0.parquet")

	_, err := DiscoverFS(fsys, nil)
	want := "dataset: year=2025/month=01/part-0.parquet is partitioned by year/month " +
		"and year=2024/part-0.parquet is partitioned by year: not a Hive partitioned dataset"
	if err == nil || err.Error() != want {
		t.Errorf("got %v, want %q", err, want)
	}

	// A file in the root has no keys at all, which the message has a word for
	// rather than an empty pair of brackets.
	_, err = DiscoverFS(tree("year=2024/part-0.parquet", "z.parquet"), nil)
	if err == nil || !errors.Is(err, ErrLayout) {
		t.Fatalf("got %v, want ErrLayout", err)
	}
	if got := err.Error(); got != "dataset: z.parquet is partitioned by nothing "+
		"and year=2024/part-0.parquet is partitioned by year: not a Hive partitioned dataset" {
		t.Errorf("got %q", got)
	}
}

func TestDiscoverOnDisk(t *testing.T) {
	root := t.TempDir()
	for _, rel := range []string{
		filepath.Join("year=2024", "month=01", "part-0.parquet"),
		filepath.Join("year=2024", "month=02", "part-0.parquet"),
	} {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o750); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte("rows"), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	d, err := Discover(root, nil)
	if err != nil {
		t.Fatal(err)
	}
	if d.Root != root {
		t.Errorf("root %q, want %q", d.Root, root)
	}
	if d.Len() != 2 {
		t.Fatalf("found %d files, want 2", d.Len())
	}

	// The paths are the ones this operating system uses, so they open.
	for _, f := range d.Files {
		if _, err = os.ReadFile(f.Path); err != nil {
			t.Errorf("cannot open the path discovery returned: %v", err)
		}
		if want := filepath.Join(root, "year=2024"); filepath.Dir(filepath.Dir(f.Path)) != want {
			t.Errorf("path %q, want it under %q", f.Path, want)
		}
	}

	if _, err = Discover(filepath.Join(root, "gone"), nil); err == nil {
		t.Error("discovered a directory that is not there")
	}
}

func TestSelect(t *testing.T) {
	fsys := tree(
		"year=2024/month=01/part-0.parquet",
		"year=2024/month=02/part-0.parquet",
		"year=2025/month=01/part-0.parquet",
	)
	d := discover(t, fsys, nil)

	january := d.Select(func(f File) bool { return d.Value(f, "month").Text == "01" })
	want := []string{"year=2024/month=01/part-0.parquet", "year=2025/month=01/part-0.parquet"}
	if got := paths(january); !slices.Equal(got, want) {
		t.Errorf("files %q, want %q", got, want)
	}
	if !january.Schema.Equal(d.Schema) {
		t.Errorf("schema %s, want the one it was narrowed from", january.Schema)
	}
	if january.Root != d.Root {
		t.Errorf("root %q, want %q", january.Root, d.Root)
	}
	if d.Len() != 3 {
		t.Error("Select changed the dataset it was called on")
	}

	none := d.Select(func(File) bool { return false })
	if none.Len() != 0 {
		t.Errorf("kept %d files, want none", none.Len())
	}

	// A predicate on a column the dataset has not got keeps nothing, since
	// every file reads as missing.
	if got := d.Select(func(f File) bool { return d.Value(f, "day").Text == "17" }); got.Len() != 0 {
		t.Errorf("kept %d files on a column that is not there, want none", got.Len())
	}
}
