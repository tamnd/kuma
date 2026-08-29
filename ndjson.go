package kuma

import (
	"io"

	"github.com/tamnd/kuma/ndjson"
)

// ReadNDJSON reads a whole newline delimited JSON file into a frame.
//
// One JSON object to a line and one line to a row, which is the shape a log
// file, an export out of a document store and most of what an API streams come
// in.
//
//	f, err := kuma.ReadNDJSON(r, nil)
//
// A nil options is the useful default: the columns are the members the first
// thousand lines had, and the type of each one is worked out from the values on
// those lines. JSON says what a value is, so that is reading rather than
// guessing, and what it decides and how to say otherwise is on [ndjson.Options].
//
// This reads everything. ScanNDJSON, which arrives with the lazy frame, reads a
// chunk at a time and never holds more than one of them, which is what a file
// larger than memory needs. Either way a column comes back in chunks, so nothing
// here asks for one allocation the size of the file.
//
// The frame is Dynamic for the same reason a CSV frame is. A file is not a Go
// type, so the columns are asked for by name and read with [Frame.Series]. A
// typed frame comes from kumagen.
func ReadNDJSON(r io.Reader, opts *ndjson.Options) (*Frame[Dynamic], error) {
	t, err := ndjson.Read(r, opts)
	if err != nil {
		return nil, err
	}
	return frameOf(t)
}

// ReadNDJSONFile reads the file at path. It is [ReadNDJSON] over an open file.
func ReadNDJSONFile(path string, opts *ndjson.Options) (*Frame[Dynamic], error) {
	t, err := ndjson.ReadFile(path, opts)
	if err != nil {
		return nil, err
	}
	return frameOf(t)
}

// WriteNDJSON writes the frame as newline delimited JSON, one object to a line.
//
// A nil options is the useful default: the members named and ordered the way the
// schema is, null where a value is missing, and floats written with the fewest
// digits that read back as the same value. What else it can do is on
// [ndjson.WriteOptions].
//
//	err := f.WriteNDJSON(w, nil)
//
// Reading the result back gives the same frame. There is no empty field problem
// here the way there is in a CSV file, since JSON writes null for a missing value
// and a pair of quotes for a string of no bytes, and those are different things
// on the page.
func (f *Frame[S]) WriteNDJSON(w io.Writer, opts *ndjson.WriteOptions) error {
	return ndjson.Write(w, f.table(), opts)
}

// WriteNDJSONFile writes the frame to the file at path, creating it if it is not
// there and truncating it if it is. It is [Frame.WriteNDJSON] over that file.
func (f *Frame[S]) WriteNDJSONFile(path string, opts *ndjson.WriteOptions) error {
	return ndjson.WriteFile(path, f.table(), opts)
}
