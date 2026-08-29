package dataset

import (
	"fmt"
	"math"
	"net/url"
	"strconv"
	"strings"

	"github.com/tamnd/kuma/dtype"
)

// parse pulls the partition keys and values out of the directories above a
// file.
//
// The path is slash separated and relative to the root of the dataset, and the
// last segment is the file itself, which is a name rather than a partition. A
// file sitting directly in the root has no partitions at all, which is a
// dataset of one partition rather than an error.
func parse(rel string) (keys []string, vals []Value, err error) {
	i := strings.LastIndexByte(rel, '/')
	if i < 0 {
		return nil, nil, nil
	}

	for _, seg := range strings.Split(rel[:i], "/") {
		k, v, found := strings.Cut(seg, "=")
		if !found || k == "" {
			return nil, nil, fmt.Errorf(
				"%q is not a key=value directory: %w", seg, ErrLayout)
		}
		keys = append(keys, unescape(k))
		vals = append(vals, value(unescape(v)))
	}
	return keys, vals, nil
}

// value turns the text of a directory into a partition value, which is the text
// itself unless it is the name a writer uses for a null.
func value(text string) Value {
	if text == DefaultPartition {
		return Value{Null: true}
	}
	return Value{Text: text}
}

// unescape decodes the percent escapes a writer puts in a directory name.
//
// A key or a value can hold a slash, an equals sign or a space, none of which
// can go in a path as itself, so Hive and the engines after it write them as
// %2F and the rest. Text that is not a valid escaping is text: a directory
// called 100%25 is decoded and one called 100% is left alone, because the
// second is a name somebody chose and refusing to read the file over it would
// help nobody.
func unescape(s string) string {
	if !strings.ContainsRune(s, '%') {
		return s
	}
	out, err := url.PathUnescape(s)
	if err != nil {
		return s
	}
	return out
}

// infer works out the type of a partition column from every value in it.
//
// The rule is on [Dataset]: a value is a number only when formatting the number
// back gives the text that was in the path. A column of numbers where one
// directory is named 01 is a column of strings, because 01 has to stay 01 to
// name the directory it came from.
//
// A column that is null the whole way down is a string column, which is the
// same answer the csv and ndjson readers give a column they never saw a value
// in.
func infer(vals []Value) dtype.DataType {
	kind := dtype.Int64
	seen := false
	for _, v := range vals {
		if v.Null {
			continue
		}
		seen = true

		got := numeric(v.Text)
		if dtype.Equal(got, dtype.String) {
			return dtype.String
		}
		if dtype.Equal(got, dtype.Float64) {
			kind = dtype.Float64
		}
	}
	if !seen {
		return dtype.String
	}
	return kind
}

// numeric returns the narrowest number type whose text is exactly s, or
// [dtype.String] when no number writes itself that way.
//
// Parsing and formatting back is what makes this exact. [strconv.ParseInt]
// reads +1 and 007 as numbers, and neither of those is the text a number prints
// as, so neither is a number here. That matters because the text is the name of
// a directory: a value that does not print back the way it arrived cannot be
// used to find the file it came from.
//
// NaN and the infinities are the exception to the rule rather than a case of
// it. They do print back as themselves, but a partition value is there to be
// compared against, and NaN does not equal itself, so a filter on it would
// match nothing and a directory would go missing. They read as text.
func numeric(s string) dtype.DataType {
	if s == "" {
		return dtype.String
	}

	if v, err := strconv.ParseInt(s, 10, 64); err == nil && strconv.FormatInt(v, 10) == s {
		return dtype.Int64
	}
	v, err := strconv.ParseFloat(s, 64)
	if err == nil && finite(v) && strconv.FormatFloat(v, 'g', -1, 64) == s {
		return dtype.Float64
	}
	return dtype.String
}

// finite says whether v is a real number rather than a NaN or an infinity.
func finite(v float64) bool { return v == v && !math.IsInf(v, 0) }
