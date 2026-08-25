package kuma

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math/big"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// PrintOptions says how much of a frame to show and what a missing value looks
// like.
//
// The zero value is what String uses. That is ten rows, twelve columns and
// cells of at most thirty two characters, which is a size that fits both in a
// terminal and in a test failure, the two places a frame ever gets printed.
type PrintOptions struct {
	// MaxRows is how many rows to show. Zero means ten and a negative number
	// means all of them. A frame with more rows than this shows the first and
	// the last few with a line of dots in between.
	MaxRows int

	// MaxCols is how many columns to show, under the same rules. The columns
	// left out are the ones in the middle.
	MaxCols int

	// MaxWidth is how wide one cell may be before it is cut short. Zero means
	// thirty two and a negative number means no limit. A cell that is cut
	// short ends in three dots, so that one column of long strings does not
	// push everything after it off the side of the screen.
	MaxWidth int

	// Null is the text for a missing value. Empty means "null".
	//
	// It is not blank by default because a blank cell reads as a value that
	// happens to be empty, and the difference between a value that is not
	// there and a value that is there and empty is the difference this library
	// exists to keep.
	Null string
}

// The defaults, and the two pieces of text the table is made of that are not
// values.
const (
	defaultRows  = 10
	defaultCols  = 12
	defaultWidth = 32
	nullText     = "null"
	gapText      = "..."
)

// withDefaults returns the options with the zero fields filled in, so that the
// rest of the printer can read them without asking whether they were set.
func (o *PrintOptions) withDefaults() PrintOptions {
	var out PrintOptions
	if o != nil {
		out = *o
	}
	if out.MaxRows == 0 {
		out.MaxRows = defaultRows
	}
	if out.MaxCols == 0 {
		out.MaxCols = defaultCols
	}
	if out.MaxWidth == 0 {
		out.MaxWidth = defaultWidth
	}
	if out.Null == "" {
		out.Null = nullText
	}
	return out
}

// Render returns the frame as a table, with the options applied. String is this
// with the defaults.
func (f *Frame[S]) Render(o *PrintOptions) string {
	title := fmt.Sprintf("kuma.Frame[%T] %d rows x %d cols", *new(S), f.rows, len(f.cols))
	return render(title, f.cols, f.rows, o)
}

// String returns the frame as a table: the shape, then the column names, then
// their types, then the rows.
//
// It shows the first and last few rows rather than all of them, since a frame
// is usually longer than a screen and the whole point of printing one is to
// look at it. Render takes the options if a different amount is wanted.
//
// Numbers are printed at the shortest text that reads back as the same number
// rather than rounded to a fixed number of digits. A printer that rounds is a
// printer that will one day show two values as the same when they are not,
// during the debugging session where that matters most.
func (f *Frame[S]) String() string { return f.Render(nil) }

// Render returns the series as a table of one column, with the options applied.
func (s Series[T]) Render(o *PrintOptions) string {
	title := fmt.Sprintf("kuma.Series[%s] %d rows", s.DType(), s.Len())
	return render(title, []Column{s.Column()}, s.Len(), o)
}

// String returns the series as a table of one column. It follows the same rules
// as [Frame.String].
func (s Series[T]) String() string { return s.Render(nil) }

// Render returns the column as a table of one column, with the options applied.
func (c Column) Render(o *PrintOptions) string {
	title := fmt.Sprintf("kuma.Column %d rows", c.Len())
	return render(title, []Column{c}, c.Len(), o)
}

// String returns the column as a table of one column. It follows the same rules
// as [Frame.String].
func (c Column) String() string { return c.Render(nil) }

// render draws the table under a line that says what it is.
//
// Widths are counted in runes, which is right for the text a column name and a
// value normally are and is wrong for the scripts where one character takes two
// cells on the screen. Getting that right means a table of character widths and
// a decision about terminals that do not agree with it, and it is not worth
// that for a debugging aid.
func render(title string, cols []Column, rows int, o *PrintOptions) string {
	opts := o.withDefaults()
	if len(cols) == 0 {
		return title
	}

	which := pick(len(cols), opts.MaxCols)
	at := pick(rows, opts.MaxRows)

	// Every column is laid out in full before anything is written, because the
	// width of a column is not known until its last value has been rendered.
	body := make([][]string, len(which))
	width := make([]int, len(which))
	right := make([]bool, len(which))
	for i, c := range which {
		cells := make([]string, len(at)+2)
		if c < 0 {
			for j := range cells {
				cells[j] = gapText
			}
		} else {
			col := cols[c]
			cells[0] = printable(col.Name())
			cells[1] = col.DType().String()
			right[i] = rightAligned(col.DType().Kind())
			for j, r := range at {
				if r < 0 {
					cells[j+2] = gapText
					continue
				}
				cells[j+2] = cellText(col.data, r, &opts)
			}
		}
		for j, s := range cells {
			cells[j] = shorten(s, opts.MaxWidth)
			if w := utf8.RuneCountInString(cells[j]); w > width[i] {
				width[i] = w
			}
		}
		body[i] = cells
	}

	var sb strings.Builder
	sb.WriteString(title)
	sb.WriteString("\n")
	for j := range len(at) + 2 {
		sb.WriteString("\n  ")
		for i := range which {
			if i > 0 {
				sb.WriteString(" | ")
			}
			pad(&sb, body[i][j], width[i], right[i], i == len(which)-1)
		}
		if j == 1 {
			sb.WriteString("\n")
			sb.WriteString(rule(width))
		}
	}
	return sb.String()
}

// pick returns the positions to show out of n, where a negative position is the
// gap where the ones left out were. It returns every position if limit is
// negative or there is nothing to leave out.
//
// The half that is kept is the front when the count is odd, so that asking for
// one row gives the first row rather than the last.
func pick(n, limit int) []int {
	if limit < 0 || n <= limit {
		out := make([]int, n)
		for i := range out {
			out[i] = i
		}
		return out
	}

	head := (limit + 1) / 2
	tail := limit - head
	out := make([]int, 0, limit+1)
	for i := range head {
		out = append(out, i)
	}
	out = append(out, -1)
	for i := n - tail; i < n; i++ {
		out = append(out, i)
	}
	return out
}

// rule returns the line of dashes under the header, with a plus where the bars
// above it are.
func rule(width []int) string {
	var sb strings.Builder
	sb.WriteString("--")
	for i, w := range width {
		if i > 0 {
			sb.WriteString("-+-")
		}
		sb.WriteString(strings.Repeat("-", w))
	}
	return sb.String()
}

// pad writes a cell out at the width of its column.
//
// The last cell on a line is not padded on the right, since that would be a
// line with trailing whitespace on it, which a diff will one day complain
// about and a person will one day have to explain.
func pad(sb *strings.Builder, s string, w int, right, last bool) {
	n := w - utf8.RuneCountInString(s)
	if right {
		spaces(sb, n)
		sb.WriteString(s)
		return
	}
	sb.WriteString(s)
	if !last {
		spaces(sb, n)
	}
}

// blanks is written a piece at a time rather than a byte at a time. It is wide
// enough that one of them covers almost every gap there is to fill.
const blanks = "                                "

// spaces writes n of them, and nothing at all if n is not positive.
func spaces(sb *strings.Builder, n int) {
	for n > len(blanks) {
		sb.WriteString(blanks)
		n -= len(blanks)
	}
	if n > 0 {
		sb.WriteString(blanks[:n])
	}
}

// shorten cuts a cell down to limit runes, ending it in dots to say that it was
// cut. A max with no room for the dots just cuts.
func shorten(s string, limit int) string {
	if limit < 0 || utf8.RuneCountInString(s) <= limit {
		return s
	}
	keep := limit - len(gapText)
	if keep < 1 {
		return string([]rune(s)[:limit])
	}
	return string([]rune(s)[:keep]) + gapText
}

// rightAligned reports whether a column of this kind reads better against the
// right hand edge, which is the kinds whose values are numbers. Lining up the
// last digit is what makes a column of numbers comparable at a glance.
func rightAligned(k dtype.Kind) bool {
	switch k {
	case dtype.Int8Kind, dtype.Int16Kind, dtype.Int32Kind, dtype.Int64Kind,
		dtype.Uint8Kind, dtype.Uint16Kind, dtype.Uint32Kind, dtype.Uint64Kind,
		dtype.Float32Kind, dtype.Float64Kind,
		dtype.Decimal128Kind, dtype.Decimal256Kind:
		return true
	default:
		return false
	}
}

// cellText renders one value as it appears in the table.
//
// This is not the CSV writer's job done twice. That one writes what a file has
// to hold, so a timestamp is the integer it is stored as until the calendar
// casts land, and a byte is whatever byte it is. This one writes what a person
// wants to read, so a timestamp is a date and a run of bytes is hex.
func cellText(data *array.Chunked, i int, o *PrintOptions) string {
	if data.IsNull(i) {
		return o.Null
	}

	// The types that carry a parameter are picked out by their Go type, since
	// the parameter is the thing being read and a kind has had it taken off.
	switch t := data.DType().(type) {
	case dtype.Time32:
		return clockText(int64(data.Value[int32](i)), t.Unit)
	case dtype.Time64:
		return clockText(data.Value[int64](i), t.Unit)
	case dtype.Timestamp:
		return stampText(data.Value[int64](i), t)
	case dtype.Duration:
		return spanText(data.Value[int64](i), t.Unit)
	case dtype.Interval:
		return intervalText(data.Bytes(i), t.Unit)
	case dtype.Decimal128:
		return decimalText(data.Bytes(i), t.Scale)
	case dtype.Decimal256:
		return decimalText(data.Bytes(i), t.Scale)
	}

	switch data.DType().Kind() {
	case dtype.NullKind:
		return o.Null
	case dtype.BoolKind:
		return strconv.FormatBool(data.Bool(i))
	case dtype.Int8Kind:
		return strconv.FormatInt(int64(data.Value[int8](i)), 10)
	case dtype.Int16Kind:
		return strconv.FormatInt(int64(data.Value[int16](i)), 10)
	case dtype.Int32Kind:
		return strconv.FormatInt(int64(data.Value[int32](i)), 10)
	case dtype.Int64Kind:
		return strconv.FormatInt(data.Value[int64](i), 10)
	case dtype.Uint8Kind:
		return strconv.FormatUint(uint64(data.Value[uint8](i)), 10)
	case dtype.Uint16Kind:
		return strconv.FormatUint(uint64(data.Value[uint16](i)), 10)
	case dtype.Uint32Kind:
		return strconv.FormatUint(uint64(data.Value[uint32](i)), 10)
	case dtype.Uint64Kind:
		return strconv.FormatUint(data.Value[uint64](i), 10)
	case dtype.Float32Kind:
		return strconv.FormatFloat(float64(data.Value[float32](i)), 'g', -1, 32)
	case dtype.Float64Kind:
		return strconv.FormatFloat(data.Value[float64](i), 'g', -1, 64)
	case dtype.StringKind, dtype.LargeStringKind:
		return printable(string(data.Bytes(i)))
	case dtype.BinaryKind, dtype.LargeBinaryKind, dtype.FixedSizeBinaryKind:
		return "0x" + hex.EncodeToString(data.Bytes(i))
	case dtype.Date32Kind:
		return dateText(int64(data.Value[int32](i)) * secondsPerDay)
	case dtype.Date64Kind:
		return dateText(data.Value[int64](i) / 1000)
	default:
		// The nested types land here, and so does anything else that turns up
		// with a type this package has never heard of. Neither is something an
		// array can be built of today, and when one can it gets a case of its
		// own rather than this line changing.
		return "?"
	}
}

// printable returns text that can go in a cell without wrecking the table,
// which means quoting it if it holds a newline, a tab or anything else that
// moves the cursor rather than drawing.
//
// Text that begins or ends in a space is quoted as well. That space is a real
// difference between two values and it is invisible in a table where every cell
// is padded with spaces anyway, and finding out why two values that look the
// same are not equal is most of what printing a frame is for.
func printable(s string) string {
	for _, r := range s {
		if !unicode.IsPrint(r) {
			return strconv.Quote(s)
		}
	}
	if s == "" {
		return s
	}
	first, _ := utf8.DecodeRuneInString(s)
	last, _ := utf8.DecodeLastRuneInString(s)
	if unicode.IsSpace(first) || unicode.IsSpace(last) {
		return strconv.Quote(s)
	}
	return s
}

// The bounds of a date the calendar has a name for, as seconds from the Unix
// epoch: the first instant of year one and the last of year 9999.
//
// A count outside them is printed as the number it is. Time formatting past
// those bounds gives a date that looks real and is not, and a value that is
// obviously a raw number is easier to make sense of than a year in five digits.
const (
	secondsPerDay  = 86400
	minCalendarSec = -62135596800
	maxCalendarSec = 253402300799
)

// dateText renders a count of seconds from the epoch as a date.
func dateText(sec int64) string {
	if sec < minCalendarSec || sec > maxCalendarSec {
		return strconv.FormatInt(sec, 10) + "s"
	}
	return time.Unix(sec, 0).UTC().Format(time.DateOnly)
}

// stampText renders an instant. A timestamp carrying a zone is shown in that
// zone, since that is what the zone is for, and one without is shown as the
// clock reading it is.
//
// A zone the machine cannot find falls back to UTC rather than failing. The
// database of zone names is a property of where the program is running and not
// of the data, and a frame that will not print on a machine with no tzdata
// installed is worse than one that prints in UTC.
func stampText(n int64, t dtype.Timestamp) string {
	sec, frac := split(n, t.Unit)
	if sec < minCalendarSec || sec > maxCalendarSec {
		return strconv.FormatInt(n, 10) + t.Unit.String()
	}

	loc := time.UTC
	if t.Zone != "" {
		if z, err := time.LoadLocation(t.Zone); err == nil {
			loc = z
		}
	}
	return time.Unix(sec, frac).In(loc).Format("2006-01-02 15:04:05.999999999")
}

// clockText renders a count from midnight as a time of day. The fraction is
// left off when there is none, so a column of whole seconds does not carry nine
// zeros down the page.
func clockText(n int64, u dtype.TimeUnit) string {
	sec, frac := split(n, u)
	if sec < 0 || sec >= secondsPerDay {
		return strconv.FormatInt(n, 10) + u.String()
	}

	out := fmt.Sprintf("%02d:%02d:%02d", sec/3600, sec/60%60, sec%60)
	if frac == 0 {
		return out
	}
	return out + strings.TrimRight(fmt.Sprintf(".%09d", frac), "0")
}

// spanText renders a length of time. A count that fits in a time.Duration is
// shown the way Go shows one, and a count too big for that is shown as itself
// with its unit after it.
func spanText(n int64, u dtype.TimeUnit) string {
	if ns, ok := nanos(n, u); ok {
		return time.Duration(ns).String()
	}
	return strconv.FormatInt(n, 10) + u.String()
}

// intervalText renders calendar arithmetic, which is months, days and a length
// of time in whatever mixture the unit holds. A part that is zero is left out
// unless every part is zero.
func intervalText(b []byte, u dtype.IntervalUnit) string {
	var months, days int32
	var ns int64
	switch u {
	case dtype.DayTime:
		days = int32(binary.LittleEndian.Uint32(b))
		ns = int64(int32(binary.LittleEndian.Uint32(b[4:]))) * 1e6
	case dtype.MonthDayNano:
		months = int32(binary.LittleEndian.Uint32(b))
		days = int32(binary.LittleEndian.Uint32(b[4:]))
		ns = int64(binary.LittleEndian.Uint64(b[8:]))
	default:
		// YearMonth, and nothing else. A unit outside the three has no width,
		// so there is no array of one for this to be called on.
		months = int32(binary.LittleEndian.Uint32(b))
	}

	var parts []string
	if months != 0 {
		parts = append(parts, strconv.FormatInt(int64(months), 10)+"mo")
	}
	if days != 0 {
		parts = append(parts, strconv.FormatInt(int64(days), 10)+"d")
	}
	if ns != 0 || len(parts) == 0 {
		parts = append(parts, time.Duration(ns).String())
	}
	return strings.Join(parts, " ")
}

// decimalText renders a decimal, which is a whole number of bytes holding a
// two's complement integer, least significant byte first, with the point scale
// digits from the right.
//
// The arithmetic is math/big rather than a pair of uint64 words, because a
// decimal256 is four words and printing is not a hot path. When the cast kernel
// learns decimals it will want the fast version, and it can have it there.
func decimalText(raw []byte, scale int32) string {
	be := make([]byte, len(raw))
	for i, c := range raw {
		be[len(raw)-1-i] = c
	}

	n := new(big.Int).SetBytes(be)
	if len(be) > 0 && be[0]&0x80 != 0 {
		n.Sub(n, new(big.Int).Lsh(big.NewInt(1), uint(len(be)*8)))
	}

	digits := n.String()
	if scale == 0 || n.Sign() == 0 {
		return digits
	}
	if scale < 0 {
		// A negative scale multiplies rather than divides, which Arrow allows
		// and readers in the wild produce.
		return digits + strings.Repeat("0", int(-scale))
	}

	sign := ""
	if digits[0] == '-' {
		sign, digits = "-", digits[1:]
	}
	if n := int(scale) + 1 - len(digits); n > 0 {
		digits = strings.Repeat("0", n) + digits
	}
	at := len(digits) - int(scale)
	return sign + digits[:at] + "." + digits[at:]
}

// split turns a count of units into whole seconds and the nanoseconds left
// over, which is the pair time.Unix takes.
func split(n int64, u dtype.TimeUnit) (sec, frac int64) {
	scale := nanosPerUnit(u)
	per := int64(time.Second) / scale

	sec, rem := n/per, n%per
	if rem < 0 {
		// Go rounds a negative quotient towards zero and time.Unix wants the
		// nanoseconds to be the part after the second, so the borrow is here.
		sec, rem = sec-1, rem+per
	}
	return sec, rem * scale
}

// nanos turns a count of units into nanoseconds, and reports whether it fits.
func nanos(n int64, u dtype.TimeUnit) (int64, bool) {
	scale := nanosPerUnit(u)
	ns := n * scale
	if scale != 1 && ns/scale != n {
		return 0, false
	}
	return ns, true
}
