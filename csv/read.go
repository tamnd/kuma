package csv

import (
	stdcsv "encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// ReadFile reads the file at path. It is [Read] over an open file, with the
// name of the file in any error the read returns.
func ReadFile(path string, opts *Options) (*Table, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	t, err := Read(f, opts)

	// A close after a read has nothing left to finish and next to nothing to
	// report, but the read error is the interesting one when there is one.
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return t, nil
}

// Read reads a whole file into columns.
//
// The reader looks at the first rows to work out what each column holds, then
// reads the rest into that. What it decides and how to override it is
// described on [Options] and in the package documentation.
//
// Everything is read. A file too large to hold in memory wants the lazy
// kuma.ScanCSV, which reads a chunk at a time and never has more than one of
// them alive. What this returns is a column in chunks either way, so nothing
// asks for one allocation the size of the file.
func Read(r io.Reader, opts *Options) (*Table, error) {
	o := opts.withDefaults()
	if err := checkDelim(o.Delimiter, o.Comment); err != nil {
		return nil, err
	}
	rd := &reader{opts: o, src: stdcsv.NewReader(r)}

	rd.src.Comma = o.Delimiter
	rd.src.Comment = o.Comment
	rd.src.LazyQuotes = o.LazyQuotes
	rd.src.TrimLeadingSpace = o.TrimLeadingSpace

	// The record slice is reused, so every record that outlives the next Read
	// is copied. The strings in it are not reused and do not need copying,
	// since encoding/csv builds them fresh for each record.
	rd.src.ReuseRecord = true

	return rd.read()
}

// reader is one run of Read. Everything it works out about the file, from the
// names to the parse for each column, is worked out once and then used a row
// at a time.
type reader struct {
	src  *stdcsv.Reader
	opts Options

	names []string
	types []dtype.DataType
	add   []appender

	build  []*array.Builder
	chunks [][]*array.Array
	rows   int // rows in the chunk being built
}

// read is the whole job: the header, then a sample to decide the types, then
// the rest of the file into the columns those types imply.
func (r *reader) read() (*Table, error) {
	first, err := r.header()
	if err != nil {
		return nil, err
	}

	sample, lines, err := r.sample(first)
	if err != nil {
		return nil, err
	}
	if err := r.plan(sample); err != nil {
		return nil, err
	}

	for i, row := range sample {
		if err := r.row(row, lines[i]); err != nil {
			return nil, err
		}
	}
	if err := r.rest(); err != nil {
		return nil, err
	}
	return r.table()
}

// header reads the lines to skip and the header, and settles on the column
// names. It returns the first record of data, which is the record it has
// already read when there is no header line.
func (r *reader) header() ([]string, error) {
	// A banner line has whatever number of fields it likes, so the record
	// length is not enforced until the skipping is over.
	r.src.FieldsPerRecord = -1
	for range r.opts.Skip {
		if _, err := r.src.Read(); err != nil {
			return nil, skipError(err)
		}
	}
	r.src.FieldsPerRecord = 0

	rec, err := r.src.Read()
	if err != nil {
		return nil, skipError(err)
	}

	var first []string
	if r.opts.NoHeader {
		first = slices.Clone(rec)
		rec = nil
	}

	names, err := r.columnNames(rec, len(first)+len(rec))
	if err != nil {
		return nil, err
	}
	r.names = names
	return first, nil
}

// columnNames returns the names of the columns, from the header record when
// there is one and from Options.Names when it is set, which wins over both.
func (r *reader) columnNames(header []string, fields int) ([]string, error) {
	names := make([]string, fields)
	switch {
	case len(r.opts.Names) > 0:
		if len(r.opts.Names) != fields {
			return nil, fmt.Errorf("csv: %d names for a file with %d fields: %w",
				len(r.opts.Names), fields, ErrNames)
		}
		copy(names, r.opts.Names)
	default:
		for i := range names {
			// A header cell can be empty, and a column still needs a name to
			// be asked for by.
			if i < len(header) && header[i] != "" {
				names[i] = header[i]
				continue
			}
			names[i] = "column_" + strconv.Itoa(i+1)
		}
	}

	seen := make(map[string]bool, len(names))
	for _, name := range names {
		if seen[name] {
			return nil, fmt.Errorf("csv: two columns named %q: %w", name, ErrNames)
		}
		seen[name] = true
	}
	return names, nil
}

// sample reads the rows the types are decided from, along with the line each
// one starts on, since a value that will not parse has to say where it is and
// these rows are not the ones the csv reader is holding by the time they are
// built.
func (r *reader) sample(first []string) ([][]string, []int, error) {
	want := r.opts.InferRows
	if r.typesKnown() {
		// Nothing is being decided, so nothing has to be held. The rows go
		// straight into the columns as they arrive.
		want = 0
	}

	var rows [][]string
	var lines []int
	if first != nil {
		rows, lines = append(rows, first), append(lines, r.line())
	}

	for want < 0 || len(rows) < want {
		rec, err := r.src.Read()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		rows = append(rows, slices.Clone(rec))
		lines = append(lines, r.line())
	}
	return rows, lines, nil
}

// typesKnown reports whether every column has been given a type, which is the
// case where there is nothing to infer and no sample to hold.
func (r *reader) typesKnown() bool {
	if len(r.opts.Types) < len(r.names) {
		return false
	}
	for _, name := range r.names {
		if _, ok := r.opts.Types[name]; !ok {
			return false
		}
	}
	return true
}

// plan settles the type of every column and builds what it takes to read one:
// a builder to put the values in and the parse that turns text into one.
func (r *reader) plan(sample [][]string) error {
	for name := range r.opts.Types {
		if !slices.Contains(r.names, name) {
			return fmt.Errorf("csv: no column named %q in the file: %w", name, ErrNoColumn)
		}
	}

	guess := infer(sample, len(r.names), &r.opts)
	r.types = make([]dtype.DataType, len(r.names))
	r.add = make([]appender, len(r.names))
	r.build = make([]*array.Builder, len(r.names))
	r.chunks = make([][]*array.Array, len(r.names))

	for i, name := range r.names {
		dt, ok := r.opts.Types[name]
		if !ok {
			dt = guess[i].dtype()
		}

		add, err := appenderFor(dt)
		if err != nil {
			return fmt.Errorf("csv: column %q: %w", name, err)
		}
		b, err := array.NewBuilder(dt)
		if err != nil {
			return fmt.Errorf("csv: column %q: %w", name, err)
		}

		r.types[i], r.add[i], r.build[i] = dt, add, b
	}

	r.grow(r.firstChunk(len(sample)))
	return nil
}

// firstChunk returns how many rows to make room for before reading any.
//
// A file that filled the sample has more rows coming and gets the whole chunk
// at once, which is one allocation per column rather than the seven that
// growing into it a double at a time costs. A file that ended inside the
// sample is as long as it will ever be, so reading ten rows does not allocate
// for sixty five thousand.
func (r *reader) firstChunk(sample int) int {
	switch {
	case r.opts.InferRows < 0:
		// The whole file was the sample, so the length is not a guess.
		return min(sample, r.opts.ChunkSize)
	case sample < r.opts.InferRows && !r.typesKnown():
		// The file ran out before the sample did.
		return min(sample+1, r.opts.ChunkSize)
	default:
		return r.opts.ChunkSize
	}
}

// rest reads what is left of the file, which is everything after the rows the
// sample took.
func (r *reader) rest() error {
	for {
		rec, err := r.src.Read()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if err := r.row(rec, r.line()); err != nil {
			return err
		}
	}
}

// row puts one record into the columns.
//
// A field that is one of the null values is missing and is never parsed, which
// is what makes an empty field cost nothing in a column of numbers.
func (r *reader) row(rec []string, line int) error {
	for i, s := range rec {
		b := r.build[i]
		if r.opts.isNull(s) {
			b.AppendNull()
			continue
		}
		if err := r.add[i](b, s); err != nil {
			if r.opts.IgnoreParseErrors {
				b.AppendNull()
				continue
			}
			return &ValueError{
				Line:   line,
				Column: r.names[i],
				Type:   r.types[i].String(),
				Value:  s,
				Err:    err,
			}
		}
	}

	r.rows++
	if r.rows == r.opts.ChunkSize {
		r.flush()
		r.grow(r.opts.ChunkSize)
	}
	return nil
}

// line returns the line the record just read starts on.
func (r *reader) line() int {
	line, _ := r.src.FieldPos(0)
	return line
}

// grow makes room for n more rows in every column.
//
// The first chunk asks for what the sample showed rather than for the whole
// chunk size, so reading a file of ten rows does not allocate for sixty five
// thousand. A file that has already filled one chunk gets the full size, since
// a file that long is going to fill the next one too.
func (r *reader) grow(n int) {
	for _, b := range r.build {
		b.Grow(n)
	}
}

// flush turns what has been built so far into a chunk of each column.
func (r *reader) flush() {
	if r.rows == 0 {
		return
	}
	for i, b := range r.build {
		r.chunks[i] = append(r.chunks[i], b.Finish())
	}
	r.rows = 0
}

// table assembles the columns and the schema they describe.
func (r *reader) table() (*Table, error) {
	r.flush()

	fields := make([]dtype.Field, len(r.names))
	cols := make([]*array.Chunked, len(r.names))
	for i, name := range r.names {
		c, err := array.NewChunked(r.types[i], r.chunks[i]...)
		if err != nil {
			return nil, fmt.Errorf("csv: %w", err)
		}
		cols[i] = c
		fields[i] = dtype.Field{Name: name, Type: r.types[i], Nullable: c.NullCount() > 0}
	}
	return &Table{Schema: dtype.Schema{Fields: fields}, Columns: cols}, nil
}

// skipError turns the end of the input into ErrNoData, which is what running
// out of file before the header means.
func skipError(err error) error {
	if errors.Is(err, io.EOF) {
		return ErrNoData
	}
	return err
}
