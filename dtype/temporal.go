package dtype

import "strconv"

// TimeUnit is the resolution of a temporal value.
//
// The zero value is Second, which is the coarsest unit and the one that cannot
// silently lose data by being wrong: reading a value that is really in
// nanoseconds as if it were seconds gives an obviously absurd date rather than
// a plausible wrong one.
type TimeUnit uint8

// The time units.
const (
	Second TimeUnit = iota
	Millisecond
	Microsecond
	Nanosecond
)

var timeUnitNames = [...]string{
	Second:      "s",
	Millisecond: "ms",
	Microsecond: "us",
	Nanosecond:  "ns",
}

// String returns the unit's short name, which is the one Arrow uses in a type
// name: s, ms, us or ns.
func (u TimeUnit) String() string {
	if int(u) >= len(timeUnitNames) {
		return "TimeUnit(" + strconv.Itoa(int(u)) + ")"
	}
	return timeUnitNames[u]
}

// Valid reports whether u is one of the four defined units.
func (u TimeUnit) Valid() bool { return u <= Nanosecond }

// IntervalUnit is what an interval counts.
//
// An interval is calendar arithmetic rather than a fixed span of time, which is
// why it is a separate type from Duration. One month is not a number of
// seconds, and adding one month to the 31st of January has to land somewhere
// the caller agrees with.
type IntervalUnit uint8

// The interval units.
const (
	// YearMonth counts whole months, stored as one int32.
	YearMonth IntervalUnit = iota

	// DayTime counts days and milliseconds, stored as two int32 values.
	DayTime

	// MonthDayNano counts months, days and nanoseconds, stored as an int32, an
	// int32 and an int64. It is the one that can express everything the other
	// two can.
	MonthDayNano
)

var intervalUnitNames = [...]string{
	YearMonth:    "year_month",
	DayTime:      "day_time",
	MonthDayNano: "month_day_nano",
}

// String returns the unit's name.
func (u IntervalUnit) String() string {
	if int(u) >= len(intervalUnitNames) {
		return "IntervalUnit(" + strconv.Itoa(int(u)) + ")"
	}
	return intervalUnitNames[u]
}

// Valid reports whether u is one of the three defined units.
func (u IntervalUnit) Valid() bool { return u <= MonthDayNano }

// Time32 is a time of day with no date, stored in 32 bits as a count of Unit
// since midnight. Only Second and Millisecond fit, which Validate checks.
type Time32 struct {
	Unit TimeUnit
}

// Kind returns Time32Kind.
func (t Time32) Kind() Kind { return Time32Kind }

// String returns the canonical name, such as "time32[s]".
func (t Time32) String() string { return "time32[" + t.Unit.String() + "]" }

// Bits returns 32.
func (t Time32) Bits() int { return 32 }

// Time64 is a time of day with no date, stored in 64 bits as a count of Unit
// since midnight. Only Microsecond and Nanosecond are allowed, so that there is
// exactly one representation of every resolution across the two time types.
type Time64 struct {
	Unit TimeUnit
}

// Kind returns Time64Kind.
func (t Time64) Kind() Kind { return Time64Kind }

// String returns the canonical name, such as "time64[ns]".
func (t Time64) String() string { return "time64[" + t.Unit.String() + "]" }

// Bits returns 64.
func (t Time64) Bits() int { return 64 }

// Timestamp is an instant, stored as an int64 count of Unit since the Unix
// epoch.
//
// Zone is an IANA name such as "Europe/London", or empty. The distinction is
// not a display detail. An empty Zone means the value is naive local time with
// no instant attached, so two of them cannot be compared across zones and
// truncating to a day is unambiguous. A non-empty Zone means the int64 is a
// real instant in UTC and the zone says how to render it and how calendar
// arithmetic on it behaves around a daylight saving transition.
//
// Nothing in this package resolves Zone against the tzdata database, because a
// binary built without tzdata would then reject a schema that is perfectly
// valid on the machine that wrote it. Resolution happens where the arithmetic
// happens.
type Timestamp struct {
	Unit TimeUnit
	Zone string
}

// Kind returns TimestampKind.
func (t Timestamp) Kind() Kind { return TimestampKind }

// String returns the canonical name, such as "timestamp[us]" or
// "timestamp[us, tz=UTC]".
func (t Timestamp) String() string {
	if t.Zone == "" {
		return "timestamp[" + t.Unit.String() + "]"
	}
	return "timestamp[" + t.Unit.String() + ", tz=" + t.Zone + "]"
}

// Bits returns 64.
func (t Timestamp) Bits() int { return 64 }

// Duration is an elapsed span of time, stored as an int64 count of Unit.
//
// It is the difference between two timestamps and it is exact, unlike Interval,
// which is calendar arithmetic.
type Duration struct {
	Unit TimeUnit
}

// Kind returns DurationKind.
func (t Duration) Kind() Kind { return DurationKind }

// String returns the canonical name, such as "duration[ns]".
func (t Duration) String() string { return "duration[" + t.Unit.String() + "]" }

// Bits returns 64.
func (t Duration) Bits() int { return 64 }

// Interval is a span expressed in calendar units, which do not have a fixed
// length. See IntervalUnit for what each one counts and how wide it is.
type Interval struct {
	Unit IntervalUnit
}

// Kind returns IntervalKind.
func (t Interval) Kind() Kind { return IntervalKind }

// String returns the canonical name, such as "interval[month_day_nano]".
func (t Interval) String() string { return "interval[" + t.Unit.String() + "]" }

// Bits returns the storage width, which depends on the unit: 32 for YearMonth,
// 64 for DayTime and 128 for MonthDayNano.
func (t Interval) Bits() int {
	switch t.Unit {
	case YearMonth:
		return 32
	case DayTime:
		return 64
	case MonthDayNano:
		return 128
	}
	return 0
}
