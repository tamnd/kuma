package parquet_test

import (
	"bytes"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/parquet"
)

// openFileReader opens one of the files in testdata and makes a reader for it.
func openFileReader(tb testing.TB, name string) *parquet.FileReader {
	tb.Helper()

	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		tb.Fatalf("read: %v", err)
	}
	r, err := parquet.NewFileReader(bytes.NewReader(b), int64(len(b)))
	if err != nil {
		tb.Fatalf("NewFileReader(%s): %v", name, err)
	}
	return r
}

// footerOf is how many bytes reading the footer of a file takes, which is the
// magic at the front, the footer itself, and the length and the magic behind
// it. It is what a reader has read before it has read any data.
func footerOf(tb testing.TB, name string) int64 {
	tb.Helper()

	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		tb.Fatalf("read: %v", err)
	}
	return 4 + 8 + int64(binary.LittleEndian.Uint32(b[len(b)-8:]))
}

// rowGroup reads one row group and fails the test if it cannot.
func rowGroup(tb testing.TB, r *parquet.FileReader, i int) parquet.Batch {
	tb.Helper()

	b, err := r.RowGroup(i)
	if err != nil {
		tb.Fatalf("RowGroup(%d): %v", i, err)
	}
	return b
}

// textColumn is a column of strings as strings, whether it arrived as values or
// as indices into a dictionary.
func textColumn(a *array.Array) []string {
	out := make([]string, a.Len())
	d := a.Dictionary()
	for i := range out {
		if d != nil {
			out[i] = string(d.Bytes(a.Index(i)))
			continue
		}
		out[i] = string(a.Bytes(i))
	}
	return out
}

// numberColumn is a column of numbers as numbers, in the same two shapes.
func numberColumn[T array.Numeric](a *array.Array) []T {
	out := make([]T, a.Len())
	d := a.Dictionary()
	for i := range out {
		if d != nil {
			out[i] = d.Value[T](a.Index(i))
			continue
		}
		out[i] = a.Value[T](i)
	}
	return out
}

// fieldNames is the names of the fields of a schema, which is what a projection
// is checked against.
func fieldNames(s dtype.Schema) []string {
	names := make([]string, len(s.Fields))
	for i, f := range s.Fields {
		names[i] = f.Name
	}
	return names
}

// TestNewFileReader opens a file and checks what the reader says about it
// before anything has been read out of it.
//
// Everything here comes off the footer, and the footer is all that has been
// read: a reader that touched the data to answer any of these would have given
// up the one thing the format is for.
func TestNewFileReader(t *testing.T) {
	r := openFileReader(t, "chunks.parquet")

	if got, want := r.NumRows(), int64(6); got != want {
		t.Errorf("the file holds %d rows, want %d", got, want)
	}
	if got, want := r.NumRowGroups(), 2; got != want {
		t.Errorf("the file holds %d row groups, want %d", got, want)
	}
	if got, want := r.BytesRead(), footerOf(t, "chunks.parquet"); got != want {
		t.Errorf("reading the footer read %d bytes, want %d", got, want)
	}

	columns := r.Columns()
	if len(columns) != 2 || columns[0].Name() != "code" || columns[1].Name() != "n" {
		t.Fatalf("the file holds %d columns, want code and n", len(columns))
	}

	want := dtype.Schema{Fields: []dtype.Field{
		{Name: "code", Type: dtype.String, Nullable: true},
		{Name: "n", Type: dtype.Int64, Nullable: true},
	}, Metadata: dtype.Metadata{}}
	if got := r.Schema(); !got.Equal(want) {
		t.Errorf("the schema is %v, want %v", got, want)
	}
}

// TestNewFileReaderNames checks the names a projection uses on a file whose
// columns are inside groups.
//
// A leaf is named by its whole path, since two groups may hold a field of the
// same name and the file's own schema is where the shape of it lives. The
// nullability is the levels rather than the field: point.y is a field of an
// optional group and is missing whenever the group is.
func TestNewFileReaderNames(t *testing.T) {
	r := openFileReader(t, "nested.parquet")

	want := []string{
		"id", "tags.list.element", "counts.list.element",
		"props.key_value.key", "props.key_value.value",
		"point.x", "point.y",
		"matrix.list.element.list.element",
		"people.list.element.name", "people.list.element.age",
	}
	if got := fieldNames(r.Schema()); !slices.Equal(got, want) {
		t.Errorf("the columns are %q, want %q", got, want)
	}

	for _, f := range r.Schema().Fields {
		nullable := f.Name != "id"
		if f.Nullable != nullable {
			t.Errorf("%s is nullable %v, want %v", f.Name, f.Nullable, nullable)
		}
	}
}

// TestNewFileReaderRefused is the files that are not read at all.
//
// A reader is made from the footer, so a file whose footer cannot be read or
// whose schema is not a schema is refused where it is opened rather than where
// it is read.
func TestNewFileReaderRefused(t *testing.T) {
	// A footer holding an empty struct, which is a file that says nothing about
	// itself: no schema, no row groups, no rows.
	empty := append([]byte("PAR1"), 0)
	empty = binary.LittleEndian.AppendUint32(empty, 1)
	empty = append(empty, "PAR1"...)

	tests := []struct {
		name string
		file []byte
		want string
	}{
		{"a file too short to be one", []byte("hello"), "too small"},
		{"a file that is not one", []byte("hello, world"), "rather than"},
		{"a file with no schema in its footer", empty, "no schema"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := parquet.NewFileReader(bytes.NewReader(tt.file), int64(len(tt.file)))
			if !errors.Is(err, parquet.ErrFormat) {
				t.Fatalf("got %v, want a format error", err)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("got %q, want it to mention %q", err, tt.want)
			}
		})
	}
}

// TestFileReaderRowGroup reads a file written in two row groups.
//
// The row groups are read one at a time and each of them comes back holding its
// own rows, in order, which is what a scan adds up. Reading one twice gives the
// same values both times: the reader keeps its buffers between row groups and a
// buffer that was not cleared would show up as the first group's values in the
// second group's array.
func TestFileReaderRowGroup(t *testing.T) {
	r := openFileReader(t, "chunks.parquet")

	codes := [][]string{{"GB", "JP", "US"}, {"FR", "DE", "GB"}}
	numbers := [][]int64{{0, 1, 2}, {3, 4, 5}}

	for _, i := range []int{0, 1, 0} {
		b := rowGroup(t, r, i)
		if b.Length != 3 {
			t.Fatalf("row group %d holds %d rows, want 3", i, b.Length)
		}
		if len(b.Columns) != 2 {
			t.Fatalf("row group %d holds %d columns, want 2", i, len(b.Columns))
		}
		if got := textColumn(b.Columns[0]); !slices.Equal(got, codes[i]) {
			t.Errorf("row group %d holds %q, want %q", i, got, codes[i])
		}
		if got := numberColumn[int64](b.Columns[1]); !slices.Equal(got, numbers[i]) {
			t.Errorf("row group %d holds %v, want %v", i, got, numbers[i])
		}
	}
}

// TestFileReaderDictionary checks the one place a batch is not the shape the
// schema says it is.
//
// A chunk written as indices into a dictionary comes back that way rather than
// expanded, which is the shape it was written in and the shape the kernels
// would rather have. So the type of a column in a batch is either the type of
// the field or a dictionary of it, and that is worth pinning down rather than
// leaving for somebody to find.
func TestFileReaderDictionary(t *testing.T) {
	r := openFileReader(t, "chunks.parquet")
	b := rowGroup(t, r, 0)

	want := dtype.Dictionary{Index: dtype.Int32, Value: dtype.String}
	if got := b.Columns[0].DType(); !dtype.Equal(got, want) {
		t.Errorf("code came back as %v, want %v", got, want)
	}
	if got := r.Schema().Fields[0].Type; !dtype.Equal(got, dtype.String) {
		t.Errorf("the schema says code is %v, want %v", got, dtype.String)
	}
}

// TestFileReaderProject reads a file a column at a time.
func TestFileReaderProject(t *testing.T) {
	t.Run("one column of two", func(t *testing.T) {
		r := openFileReader(t, "chunks.parquet")
		if err := r.Project("n"); err != nil {
			t.Fatalf("Project: %v", err)
		}
		if got, want := fieldNames(r.Schema()), []string{"n"}; !slices.Equal(got, want) {
			t.Fatalf("the schema holds %q, want %q", got, want)
		}

		b := rowGroup(t, r, 1)
		if b.Length != 3 || len(b.Columns) != 1 {
			t.Fatalf("%d rows of %d columns, want 3 of 1", b.Length, len(b.Columns))
		}
		if got, want := numberColumn[int64](b.Columns[0]), []int64{3, 4, 5}; !slices.Equal(got, want) {
			t.Errorf("got %v, want %v", got, want)
		}
	})

	t.Run("in the order they are named", func(t *testing.T) {
		r := openFileReader(t, "chunks.parquet")
		if err := r.Project("n", "code"); err != nil {
			t.Fatalf("Project: %v", err)
		}
		if got, want := fieldNames(r.Schema()), []string{"n", "code"}; !slices.Equal(got, want) {
			t.Fatalf("the schema holds %q, want %q", got, want)
		}

		b := rowGroup(t, r, 0)
		if got, want := numberColumn[int64](b.Columns[0]), []int64{0, 1, 2}; !slices.Equal(got, want) {
			t.Errorf("the first column is %v, want %v", got, want)
		}
		if got, want := textColumn(b.Columns[1]), []string{"GB", "JP", "US"}; !slices.Equal(got, want) {
			t.Errorf("the second column is %q, want %q", got, want)
		}
	})

	t.Run("the same column twice", func(t *testing.T) {
		r := openFileReader(t, "chunks.parquet")
		if err := r.Project("code", "code"); err != nil {
			t.Fatalf("Project: %v", err)
		}

		b := rowGroup(t, r, 0)
		if len(b.Columns) != 2 {
			t.Fatalf("%d columns, want 2", len(b.Columns))
		}
		if got, want := textColumn(b.Columns[0]), textColumn(b.Columns[1]); !slices.Equal(got, want) {
			t.Errorf("the two copies are %q and %q", got, want)
		}
	})

	t.Run("no columns at all", func(t *testing.T) {
		r := openFileReader(t, "chunks.parquet")
		if err := r.Project(); err != nil {
			t.Fatalf("Project: %v", err)
		}

		read := r.BytesRead()
		b := rowGroup(t, r, 0)
		if b.Length != 3 || len(b.Columns) != 0 {
			t.Fatalf("%d rows of %d columns, want 3 of none", b.Length, len(b.Columns))
		}
		if got := r.BytesRead(); got != read {
			t.Errorf("counting rows read %d bytes, want none", got-read)
		}
	})

	t.Run("and back to all of them", func(t *testing.T) {
		r := openFileReader(t, "chunks.parquet")
		if err := r.Project("n"); err != nil {
			t.Fatalf("Project: %v", err)
		}
		if err := r.Project("code", "n"); err != nil {
			t.Fatalf("Project: %v", err)
		}
		if got, want := fieldNames(r.Schema()), []string{"code", "n"}; !slices.Equal(got, want) {
			t.Fatalf("the schema holds %q, want %q", got, want)
		}
		if b := rowGroup(t, r, 0); len(b.Columns) != 2 {
			t.Errorf("%d columns, want 2", len(b.Columns))
		}
	})
}

// TestFileReaderProjectRefused names a column the file does not have.
//
// Nothing is narrowed when that happens. A projection that was half applied
// would hand back batches in a shape nobody asked for, and the error would be
// somewhere else by then.
func TestFileReaderProjectRefused(t *testing.T) {
	r := openFileReader(t, "chunks.parquet")

	err := r.Project("code", "missing")
	if err == nil {
		t.Fatal("projecting a column that is not there was allowed")
	}
	if !strings.Contains(err.Error(), `"missing"`) {
		t.Errorf("got %q, want it to name the column", err)
	}

	if got, want := fieldNames(r.Schema()), []string{"code", "n"}; !slices.Equal(got, want) {
		t.Errorf("the projection is %q, want the file's own %q", got, want)
	}
}

// TestFileReaderProjectionCosts is the point of the whole thing.
//
// A file keeps each column in a run of pages of its own, so reading one of them
// has to cost what that column takes and nothing more. The count is what says
// it did: a reader that quietly read the whole file would give the same values
// at ten times the price, and a benchmark would not notice on a file this size.
func TestFileReaderProjectionCosts(t *testing.T) {
	r := openFileReader(t, "pages.parquet")
	footer := footerOf(t, "pages.parquet")
	if got := r.BytesRead(); got != footer {
		t.Fatalf("opening the file read %d bytes, want the footer's %d", got, footer)
	}

	if err := r.Project("word"); err != nil {
		t.Fatalf("Project: %v", err)
	}
	b := rowGroup(t, r, 0)
	if b.Length != 500 {
		t.Fatalf("the row group holds %d rows, want 500", b.Length)
	}

	// What one column costs is the chunk that holds it, which is what the
	// footer said it would be.
	chunk := r.Metadata().RowGroups[0].Columns[1].Meta
	if got, want := r.BytesRead()-footer, chunk.TotalCompressedSize; got != want {
		t.Errorf("reading one column read %d bytes, want the %d its chunk takes", got, want)
	}

	// And the rest of the file is untouched. The three columns of this one are
	// nothing like the two hundred a real file has, so the ratio here is the
	// mildest version of what a projection saves.
	if err := r.Project("n", "word", "maybe"); err != nil {
		t.Fatalf("Project: %v", err)
	}
	whole := r.BytesRead()
	if _, err := r.RowGroup(0); err != nil {
		t.Fatalf("RowGroup: %v", err)
	}
	all := r.BytesRead() - whole
	if one := chunk.TotalCompressedSize; all <= one*2 {
		t.Errorf("reading every column read %d bytes and reading one read %d", all, one)
	}
}

// TestFileReaderProjectionSkipsWhatItCannotRead reads files this package cannot
// read all of.
//
// A column is read because it was projected, so a file holding a list next to
// an int64, or a brotli column next to a snappy one, is readable up to the part
// that is still to come. That is not a nicety: real files have a column nobody
// asked for in a codec nobody has, and a reader that refuses a file for a column
// the query never named is a reader that cannot open it at all.
func TestFileReaderProjectionSkipsWhatItCannotRead(t *testing.T) {
	t.Run("a file of lists", func(t *testing.T) {
		r := openFileReader(t, "nested.parquet")
		if err := r.Project("id", "point.x"); err != nil {
			t.Fatalf("Project: %v", err)
		}

		b := rowGroup(t, r, 0)
		if got, want := numberColumn[int32](b.Columns[0]), []int32{1, 2}; !slices.Equal(got, want) {
			t.Errorf("id is %v, want %v", got, want)
		}
		if got, want := numberColumn[float64](b.Columns[1]), []float64{1.5, 3.5}; !slices.Equal(got, want) {
			t.Errorf("point.x is %v, want %v", got, want)
		}
	})

	t.Run("a file with a codec this package does not have", func(t *testing.T) {
		r := openFileReader(t, "codecs.parquet")
		if err := r.Project("word", "zip"); err != nil {
			t.Fatalf("Project: %v", err)
		}

		b := rowGroup(t, r, 0)
		words := textColumn(b.Columns[0])
		zip := numberColumn[int64](b.Columns[1])
		if len(words) != 1000 || len(zip) != 1000 {
			t.Fatalf("%d words and %d numbers, want 1000 of each", len(words), len(zip))
		}
		for i := range words {
			want := []string{"alpha", "beta", "gamma", "delta"}[i%4]
			if words[i] != want || zip[i] != int64(i)*3 {
				t.Fatalf("row %d is %q and %d, want %q and %d", i, words[i], zip[i], want, i*3)
			}
		}
	})
}

// TestFileReaderEmpty reads a file with no rows in it.
//
// It has a row group all the same, holding a chunk per column with nothing in
// it, which is what pyarrow writes for a table that has a schema and no data. A
// batch of no rows is what comes back, rather than an error or no batch at all.
func TestFileReaderEmpty(t *testing.T) {
	r := openFileReader(t, "empty.parquet")

	if got := r.NumRows(); got != 0 {
		t.Errorf("the file holds %d rows, want none", got)
	}
	if got := r.NumRowGroups(); got != 1 {
		t.Fatalf("the file holds %d row groups, want one", got)
	}

	b := rowGroup(t, r, 0)
	if b.Length != 0 || len(b.Columns) != 2 {
		t.Fatalf("%d rows of %d columns, want none of 2", b.Length, len(b.Columns))
	}
	for i, a := range b.Columns {
		if a.Len() != 0 {
			t.Errorf("column %d holds %d values, want none", i, a.Len())
		}
	}
}

// TestFileReaderSchemaMetadata checks that what the file says about itself
// survives a projection.
//
// The key and value metadata describes the file rather than the columns, and
// the Arrow schema in it names every column whether or not the projection did.
// Dropping it would lose the one thing a round trip through pyarrow needs.
func TestFileReaderSchemaMetadata(t *testing.T) {
	r := openFileReader(t, "alltypes.parquet")
	if err := r.Project("flag"); err != nil {
		t.Fatalf("Project: %v", err)
	}

	s := r.Schema()
	if len(s.Fields) != 1 {
		t.Fatalf("the schema holds %d fields, want one", len(s.Fields))
	}
	if v, ok := s.Metadata.Get("ARROW:schema"); !ok || v == "" {
		t.Errorf("the arrow schema is %q and there is %v, want it kept", v, ok)
	}
}

// TestFileReaderRowGroupRefused is the row groups that are not read.
//
// The first two are a caller asking for a row group the file does not have. The
// rest are a footer that contradicts itself, which is what a file damaged in
// the one place a reader trusts looks like: the counts and the offsets in a
// footer are claims, and every one of them is checked against something before
// a column is read from it.
func TestFileReaderRowGroupRefused(t *testing.T) {
	tests := []struct {
		name    string
		file    string
		columns []string
		damage  func(*parquet.Metadata)
		group   int
		want    error
		says    string
	}{
		{
			name:  "a row group before the first",
			file:  "chunks.parquet",
			group: -1,
			says:  "row group -1 of a file that has 2",
		},
		{
			name:  "a row group past the last",
			file:  "chunks.parquet",
			group: 2,
			says:  "row group 2 of a file that has 2",
		},
		{
			name:   "a row group holding fewer rows than none",
			file:   "chunks.parquet",
			damage: func(m *parquet.Metadata) { m.RowGroups[0].NumRows = -1 },
			want:   parquet.ErrFormat,
			says:   "row group 0 says it holds -1 rows",
		},
		{
			name:   "a row group whose chunks hold more rows than it does",
			file:   "chunks.parquet",
			damage: func(m *parquet.Metadata) { m.RowGroups[0].NumRows = 2 },
			want:   parquet.ErrFormat,
			says:   "the chunk for code holds 3 values in a row group of 2 rows",
		},
		{
			name:   "a row group with no chunks in it",
			file:   "chunks.parquet",
			damage: func(m *parquet.Metadata) { m.RowGroups[0].Columns = nil },
			want:   parquet.ErrFormat,
			says:   "row group 0 holds 0 chunks and none of them is code",
		},
		{
			name: "a row group whose chunks are for other columns",
			file: "chunks.parquet",
			damage: func(m *parquet.Metadata) {
				m.RowGroups[0].Columns[0].Meta.Path = []string{"elsewhere"}
			},
			want: parquet.ErrFormat,
			says: "row group 0 holds 2 chunks and none of them is code",
		},
		{
			name:    "a column of lists",
			file:    "nested.parquet",
			columns: []string{"tags.list.element"},
			want:    parquet.ErrUnsupported,
			says:    "lists are not read yet",
		},
		{
			name:    "a column in a codec this package does not have",
			file:    "codecs.parquet",
			columns: []string{"br"},
			want:    parquet.ErrUnsupported,
			says:    "brotli pages",
		},
		{
			name:    "a chunk that is not in the file",
			file:    "chunks.parquet",
			columns: []string{"n"},
			damage: func(m *parquet.Metadata) {
				c := &m.RowGroups[0].Columns[1].Meta
				c.DictionaryPageOffset, c.DataPageOffset = 1<<40, 1<<40
			},
			want: parquet.ErrFormat,
			says: "the chunk for n is",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := openFileReader(t, tt.file)
			if tt.columns != nil {
				if err := r.Project(tt.columns...); err != nil {
					t.Fatalf("Project: %v", err)
				}
			}
			if tt.damage != nil {
				tt.damage(r.Metadata())
			}

			_, err := r.RowGroup(tt.group)
			if err == nil {
				t.Fatal("the row group was read")
			}
			if tt.want != nil && !errors.Is(err, tt.want) {
				t.Errorf("got %v, want %v", err, tt.want)
			}
			if !strings.Contains(err.Error(), tt.says) {
				t.Errorf("got %q, want it to say %q", err, tt.says)
			}
		})
	}
}

// TestFileReaderReadsAfterAnError reads a file that was refused once.
//
// A reader keeps the buffers a column is assembled in from one row group to the
// next, and one that stopped somewhere inside a chunk is holding half a column,
// so the next row group has to get a new one. Half a column left in a buffer
// would come back as the wrong values rather than as an error.
func TestFileReaderReadsAfterAnError(t *testing.T) {
	r := openFileReader(t, "chunks.parquet")
	m := r.Metadata()

	m.RowGroups[0].NumRows = 2
	if _, err := r.RowGroup(0); err == nil {
		t.Fatal("a row group whose count is wrong was read")
	}

	m.RowGroups[0].NumRows = 3
	b := rowGroup(t, r, 0)
	if got, want := textColumn(b.Columns[0]), []string{"GB", "JP", "US"}; !slices.Equal(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
	if got, want := numberColumn[int64](b.Columns[1]), []int64{0, 1, 2}; !slices.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

// BenchmarkRowGroup reads a row group of five hundred rows, once with every
// column projected and once with one of them.
//
// The rate is over the bytes the file holds for the columns being read, so the
// two are comparable: a projection that cost what it saves would show up as the
// one column reading at a fraction of the rate of the three.
func BenchmarkRowGroup(b *testing.B) {
	for _, bb := range []struct {
		name    string
		columns []string
	}{
		{"every column", []string{"n", "word", "maybe"}},
		{"one column", []string{"maybe"}},
	} {
		b.Run(bb.name, func(b *testing.B) {
			r := openFileReader(b, "pages.parquet")
			if err := r.Project(bb.columns...); err != nil {
				b.Fatalf("Project: %v", err)
			}

			var bytes int64
			for _, c := range r.Metadata().RowGroups[0].Columns {
				if slices.Contains(bb.columns, strings.Join(c.Meta.Path, ".")) {
					bytes += c.Meta.TotalUncompressedSize
				}
			}
			b.SetBytes(bytes)

			b.ReportAllocs()
			for b.Loop() {
				if _, err := r.RowGroup(0); err != nil {
					b.Fatalf("RowGroup: %v", err)
				}
			}
		})
	}
}
