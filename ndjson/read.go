package ndjson

import (
	"bufio"
	"bytes"
	"encoding/json/jsontext"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// lineBuffer is how much of the input the line reader holds at once. It is the
// longest line that costs nothing, and a longer one is gathered into a buffer of
// its own rather than refused.
const lineBuffer = 1 << 16

// decoderOptions is how every decoder in this package is set up, which is once
// here rather than at each of the two places that has to say it, since resetting
// a decoder onto the next line takes the options again.
//
// The one option turns off the decoder's own check for a member name that is on
// the line twice, and the reader does that itself instead. Doing it in the
// decoder means remembering every name in the object, which on a line of a
// hundred members is a map built and thrown away for every line in the file. The
// reader already keeps a note of which columns a line has filled in, so the same
// question is a flag it has to look at anyway.
var decoderOptions = jsontext.AllowDuplicateNames(true)

// ReadFile reads the file at path. It is [Read] over an open file, with the name
// of the file in any error the read returns.
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
// The reader looks at the first lines to work out what the columns are and what
// each one holds, then reads the rest into that. What it decides and how to
// override it is described on [Options] and in the package documentation.
//
// Everything is read. A file too large to hold in memory wants the lazy
// kuma.ScanNDJSON, which reads a chunk at a time and never has more than one of
// them alive. What this returns is a column in chunks either way, so nothing
// asks for one allocation the size of the file.
func Read(r io.Reader, opts *Options) (*Table, error) {
	rd := &reader{
		opts: opts.withDefaults(),
		src:  &lines{src: bufio.NewReaderSize(r, lineBuffer)},
	}

	rd.dec = jsontext.NewDecoder(&rd.line, decoderOptions)
	return rd.read()
}

// reader is one run of Read. Everything it works out about the file, from the
// members it has to the parse for each column, is worked out once and then used
// a line at a time.
type reader struct {
	src  *lines
	opts Options

	// dec reads one line, which line holds. The decoder is pointed at a new line
	// rather than built again for it, and the same goes for the reader
	// underneath, so a file of ten million lines builds one of each.
	dec  *jsontext.Decoder
	line bytes.Reader
	name []byte // a member name with an escape in it, unquoted into here

	// all is every member the sample had, in the order it first saw them, and
	// found says where each one is in that list. names is the columns being
	// read, in the order they come out, and at says which member each one is.
	// With no Options.Columns the two lists are the same and at is 0, 1, 2 and
	// so on.
	all   []string
	found map[string]int
	names []string
	at    []int

	// where is what to do with a member while reading: the column it goes into,
	// or -1 for one the sample had and Options.Columns left out. A member that is
	// not in here at all was not in the sample.
	where map[string]int

	guess []inferred
	types []dtype.DataType
	add   []appender

	build  []*array.Builder
	chunks [][]*array.Array
	filled []bool // which columns this line has had a value for
	rows   int    // rows in the chunk being built
}

// read is the whole job: a sample to decide what the columns are, then the rest
// of the file into the columns that decision implies.
func (r *reader) read() (*Table, error) {
	sample, at, err := r.sample()
	if err != nil {
		return nil, err
	}
	if err := r.plan(len(sample)); err != nil {
		return nil, err
	}

	for i, p := range sample {
		if err := r.row(p, at[i]); err != nil {
			return nil, err
		}
	}
	if err := r.rest(); err != nil {
		return nil, err
	}
	return r.table()
}

// sample reads the lines the columns are decided from, along with the line each
// one was on, since a value that will not parse has to say where it is and these
// lines are not where the line reader is by the time they are read again.
func (r *reader) sample() ([][]byte, []int, error) {
	r.found = make(map[string]int)

	var rows [][]byte
	var at []int
	for want := r.opts.InferRows; want < 0 || len(rows) < want; {
		p, err := r.src.next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, nil, err
		}

		// The clone comes first so that the line held is the line as it was
		// read, whatever the walk over it does.
		p, n := slices.Clone(p), r.src.n
		if err := r.learn(p, n); err != nil {
			return nil, nil, err
		}
		rows, at = append(rows, p), append(at, n)
	}

	if len(r.all) == 0 {
		// Either there was nothing to read or every object was empty. Both leave
		// nothing to say what the columns are, and a table of no columns is not
		// something a caller asked for.
		return nil, nil, ErrNoData
	}
	return rows, at, nil
}

// learn folds one line of the sample into what is known about the columns, which
// is the members it has and the narrowest type each of them holds.
func (r *reader) learn(p []byte, n int) error {
	if err := r.open(p, n); err != nil {
		return err
	}

	for r.dec.PeekKind() != '}' {
		name, err := r.member(n)
		if err != nil {
			return err
		}

		at, ok := r.found[string(name)]
		if !ok {
			at = len(r.all)
			r.all = append(r.all, string(name))
			r.found[r.all[at]] = at
			r.guess = append(r.guess, inferNothing)
		}

		got, err := inferValue(r.dec)
		if err != nil {
			return r.fail(n, err)
		}
		r.guess[at] = r.guess[at].merge(got)
	}
	return r.close(n)
}

// plan settles which columns are being read and what each one holds, and builds
// what it takes to read one: a builder to put the values in and the parse that
// turns a JSON value into one.
func (r *reader) plan(sample int) error {
	if err := r.selection(); err != nil {
		return err
	}

	// The check is against every member the sample had rather than against the
	// ones being read, since a type for a column that Options.Columns leaves out
	// is a caller reading part of a file with the options they use for all of
	// it, which is not a mistake.
	for name := range r.opts.Types {
		if _, ok := r.found[name]; !ok {
			return fmt.Errorf("ndjson: no member named %q in the sample: %w", name, ErrNoColumn)
		}
	}

	r.types = make([]dtype.DataType, len(r.names))
	r.add = make([]appender, len(r.names))
	r.build = make([]*array.Builder, len(r.names))
	r.chunks = make([][]*array.Array, len(r.names))
	r.filled = make([]bool, len(r.names))

	for i, name := range r.names {
		dt, ok := r.opts.Types[name]
		if !ok {
			dt = r.guess[r.at[i]].dtype()
		}

		add, err := appenderFor(dt)
		if err != nil {
			return fmt.Errorf("ndjson: column %q: %w", name, err)
		}
		b, err := array.NewBuilder(dt)
		if err != nil {
			return fmt.Errorf("ndjson: column %q: %w", name, err)
		}

		r.types[i], r.add[i], r.build[i] = dt, add, b
	}

	r.grow(r.firstChunk(sample))
	return nil
}

// selection works out which columns are being read and what happens to each
// member of a line, from Options.Columns and the members the sample had.
func (r *reader) selection() error {
	r.where = make(map[string]int, len(r.all))
	if len(r.opts.Columns) == 0 {
		r.names = r.all
		r.at = make([]int, len(r.all))
		for i, name := range r.all {
			r.where[name], r.at[i] = i, i
		}
		return nil
	}

	for _, name := range r.all {
		r.where[name] = -1
	}
	r.names = make([]string, 0, len(r.opts.Columns))
	r.at = make([]int, 0, len(r.opts.Columns))
	for _, name := range r.opts.Columns {
		at, ok := r.found[name]
		if !ok {
			return fmt.Errorf("ndjson: no member named %q in the sample: %w", name, ErrNoColumn)
		}
		if r.where[name] >= 0 {
			return fmt.Errorf("ndjson: column %q asked for twice: %w", name, ErrNames)
		}
		r.where[name] = len(r.names)
		r.names, r.at = append(r.names, name), append(r.at, at)
	}
	return nil
}

// firstChunk returns how many rows to make room for before reading any.
//
// A file that filled the sample has more lines coming and gets the whole chunk
// at once, which is one allocation per column rather than the seven that growing
// into it a double at a time costs. A file that ended inside the sample is as
// long as it will ever be, so reading ten lines does not allocate for sixty five
// thousand.
func (r *reader) firstChunk(sample int) int {
	if r.opts.InferRows < 0 || sample < r.opts.InferRows {
		// The file ended inside the sample, so its length is not a guess.
		return min(sample, r.opts.ChunkSize)
	}
	return r.opts.ChunkSize
}

// rest reads what is left of the file, which is everything after the lines the
// sample took.
func (r *reader) rest() error {
	for {
		p, err := r.src.next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
		if err := r.row(p, r.src.n); err != nil {
			return err
		}
	}
}

// row puts one line into the columns.
//
// The walk is over the line rather than over the columns, because a line is a
// list of members and there is no telling which of them it has until it has been
// read. What it turns out not to have is filled in at the end, which is the
// other half of what makes a member that is not there a missing value.
func (r *reader) row(p []byte, n int) error {
	if err := r.open(p, n); err != nil {
		return err
	}
	clear(r.filled)

	for r.dec.PeekKind() != '}' {
		name, err := r.member(n)
		if err != nil {
			return err
		}

		at, ok := r.where[string(name)]
		if !ok {
			if !r.opts.IgnoreUnknownFields {
				return &LineError{Line: n, Err: fmt.Errorf(
					"the member %q is not a column: %w", name, ErrUnknownField)}
			}
			at = -1
		}
		if at < 0 {
			// A member the file has and nothing is reading. Walking over it is
			// the whole cost, since a line cannot be read past a member without
			// reading it.
			if err := r.dec.SkipValue(); err != nil {
				return r.fail(n, err)
			}
			continue
		}
		if err := r.value(at, n); err != nil {
			return err
		}
	}
	if err := r.close(n); err != nil {
		return err
	}

	for i, ok := range r.filled {
		if !ok {
			r.build[i].AppendNull()
		}
	}
	r.rows++
	if r.rows == r.opts.ChunkSize {
		r.flush()
		r.grow(r.opts.ChunkSize)
	}
	return nil
}

// value reads the value of one member into the column it belongs to.
//
// A null is a missing value and never reaches the column's own parse, which is
// what makes a null cost nothing in a column of numbers. Everything else is the
// appender's, and what comes back says which of the two things went wrong: an
// appender that hands back the text of the value has read it and found it will
// not go in, and one that hands back nothing has run into a line that stops
// making sense, which is a broken line rather than a bad value.
func (r *reader) value(at, n int) error {
	if r.filled[at] {
		return &LineError{Line: n, Err: fmt.Errorf(
			"the member %q is on the line twice: %w", r.names[at], ErrSyntax)}
	}

	b := r.build[at]
	r.filled[at] = true

	if r.dec.PeekKind() == 'n' {
		if _, err := r.dec.ReadToken(); err != nil {
			return r.fail(n, err)
		}
		b.AppendNull()
		return nil
	}

	text, err := r.add[at](b, r.dec)
	switch {
	case err == nil:
		return nil
	case text == "":
		return r.fail(n, err)
	case r.opts.IgnoreParseErrors:
		b.AppendNull()
		return nil
	}
	return &ValueError{
		Line:   n,
		Column: r.names[at],
		Type:   r.types[at].String(),
		Value:  text,
		Err:    err,
	}
}

// open points the decoder at one line and reads past the brace at the start of
// it.
func (r *reader) open(p []byte, n int) error {
	r.line.Reset(p)
	r.dec.Reset(&r.line, decoderOptions)

	tok, err := r.dec.ReadToken()
	if err != nil {
		return r.fail(n, err)
	}
	if k := tok.Kind(); k != '{' {
		return &LineError{Line: n, Err: fmt.Errorf(
			"a JSON %s where an object should be: %w", kindName(k), ErrSyntax)}
	}
	return nil
}

// close reads the brace at the end of the object and checks that the line ends
// there.
//
// A line with a second value on it is a file this reader would be reading the
// wrong way, since the format is one object to a line, and reading the first
// object and passing over the rest would lose data without saying so.
func (r *reader) close(n int) error {
	if _, err := r.dec.ReadToken(); err != nil {
		return r.fail(n, err)
	}
	if _, err := r.dec.ReadToken(); !errors.Is(err, io.EOF) {
		if err != nil {
			return r.fail(n, err)
		}
		return &LineError{Line: n, Err: fmt.Errorf(
			"more than one value on the line: %w", ErrSyntax)}
	}
	return nil
}

// member reads the name of the next member of the object being read.
//
// The bytes it returns are only good until the next call on the decoder, which
// is all a map lookup needs, and that is the point: a name is looked up without
// a copy of it being made, so a file of ten million lines does not allocate a
// string per member per line.
func (r *reader) member(n int) ([]byte, error) {
	v, err := r.dec.ReadValue()
	if err != nil {
		return nil, r.fail(n, err)
	}

	// Almost every name is what it says between the quotes. The ones with an
	// escape in them are unquoted into a buffer of their own, which is the only
	// place the two spellings of the same name can be told apart.
	if bytes.IndexByte(v, '\\') < 0 {
		return v[1 : len(v)-1], nil
	}
	if r.name, err = jsontext.AppendUnquote(r.name[:0], v); err != nil {
		return nil, r.fail(n, err)
	}
	return r.name, nil
}

// fail says that a line is not one JSON object, which is what every error the
// decoder itself reports comes down to.
func (r *reader) fail(n int, err error) error {
	if errors.Is(err, io.ErrUnexpectedEOF) {
		err = errShort
	}
	return &LineError{Line: n, Err: fmt.Errorf("%w: %w", err, ErrSyntax)}
}

// errShort is what the end of a line in the middle of an object says, which the
// decoder reports as the end of its input.
var errShort = errors.New("the line stops in the middle of the object")

// kindName is what to call a JSON value in a message. It is the decoder's own
// name for one, except that the decoder calls an array by its bracket, which
// reads as a stray character in the middle of a sentence.
func kindName(k jsontext.Kind) string {
	if k == '[' {
		return "array"
	}
	return k.String()
}

// grow makes room for n more rows in every column.
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
			return nil, fmt.Errorf("ndjson: %w", err)
		}
		cols[i] = c
		fields[i] = dtype.Field{Name: name, Type: r.types[i], Nullable: c.NullCount() > 0}
	}
	return &Table{Schema: dtype.Schema{Fields: fields}, Columns: cols}, nil
}

// lines hands out the lines of the input one at a time.
//
// A line is a slice of the buffer underneath and is only good until the next one
// is asked for. That is what the sample copies and what the rest of the read
// never has to, and it is what keeps a file of ten million lines from being ten
// million allocations. A line too long for the buffer is gathered into one of
// its own rather than refused, since there is no length a JSON object has to be
// under.
//
// A blank line is passed over and still counts, so the line numbers in an error
// are the line numbers in the file.
type lines struct {
	src *bufio.Reader
	n   int    // the line just handed out, counting from one
	buf []byte // a line that did not fit in the buffer underneath
}

// next returns the next line that has something on it.
func (l *lines) next() ([]byte, error) {
	for {
		p, err := l.src.ReadSlice('\n')
		if errors.Is(err, bufio.ErrBufferFull) {
			p, err = l.long(p)
		}
		if err != nil && !errors.Is(err, io.EOF) {
			return nil, err
		}
		if len(p) == 0 {
			return nil, io.EOF
		}

		l.n++
		if p = bytes.Trim(p, " \t\r\n"); len(p) > 0 {
			return p, nil
		}
		if err != nil {
			// The file ended on a line with nothing on it, which is what a file
			// that ends with a newline looks like from here.
			return nil, io.EOF
		}
	}
}

// long gathers a line that did not fit in the buffer underneath.
func (l *lines) long(head []byte) ([]byte, error) {
	l.buf = append(l.buf[:0], head...)
	for {
		p, err := l.src.ReadSlice('\n')
		l.buf = append(l.buf, p...)
		if !errors.Is(err, bufio.ErrBufferFull) {
			return l.buf, err
		}
	}
}
