package dtype

// CanCast reports whether an explicit cast from one type to another is a thing
// this library will attempt.
//
// It is much looser than Coerce, and it has to be, because a cast is what the
// caller writes when Coerce has refused. Coerce answers "may I do this without
// being asked", and the answer is almost always no. CanCast answers "is this a
// meaningful thing to ask for", and the answer is usually yes.
//
// It is a question about types, not about values. A permitted cast can still
// fail on a particular row: casting int64 to int8 overflows, casting string to
// int64 meets a row that is not a number, casting a nanosecond timestamp to
// seconds throws away the fraction. Whether such a row becomes an error or a
// null is the kernel's decision and the caller's option, and neither of those
// is a question this package can answer. What CanCast rules out is the cast
// that has no meaning at all, such as a list to a struct, so that the plan
// fails while it is being built rather than partway through the second file.
func CanCast(from, to DataType) bool {
	return canCast(from, to, 0)
}

func canCast(from, to DataType, depth int) bool {
	if from == nil || to == nil || depth > MaxNestingDepth {
		return false
	}
	if Equal(from, to) {
		return true
	}

	// A column of nothing casts to anything, since the result is a column of
	// nothing in the new type. Casting to null throws every value away, which
	// is allowed because the caller asked for it in as many words.
	if from.Kind() == NullKind || to.Kind() == NullKind {
		return true
	}

	// Dictionary encoding is storage, so a cast to or from one is a cast of the
	// values with an encode or a decode around it.
	if d, ok := from.(Dictionary); ok {
		return canCast(d.Value, to, depth+1)
	}
	if d, ok := to.(Dictionary); ok {
		return canCast(from, d.Value, depth+1)
	}

	if IsNested(from) || IsNested(to) {
		return canCastNested(from, to, depth)
	}

	// Everything below here is a flat value: a number, a bool, some bytes or a
	// point in time.
	switch {
	case IsNumeric(from):
		return IsNumeric(to) || to.Kind() == BoolKind || IsString(to) || isEpochLike(to)
	case from.Kind() == BoolKind:
		return IsNumeric(to) || IsString(to)
	case IsString(from):
		// Parsing. A row that does not parse is the kernel's problem.
		return IsNumeric(to) || to.Kind() == BoolKind || IsTemporal(to) ||
			IsString(to) || IsBinary(to)
	case IsBinary(from):
		// Bytes to text is a validity check rather than a conversion, and bytes
		// to bytes is a change of layout or of width.
		return IsString(to) || IsBinary(to)
	case IsTemporal(from):
		return canCastTemporal(from, to)
	}
	return false
}

// isEpochLike reports whether a type is stored as a plain count that a number
// can be reinterpreted as, which is every temporal type except interval.
//
// An interval is three separate counts and there is no single number that means
// one of them, so turning a number into an interval is a constructor and not a
// cast.
func isEpochLike(t DataType) bool {
	return IsTemporal(t) && t.Kind() != IntervalKind
}

func canCastTemporal(from, to DataType) bool {
	if IsString(to) {
		return true
	}
	if IsNumeric(to) {
		return isEpochLike(from)
	}
	if !IsTemporal(to) {
		return false
	}

	// An interval only casts to another interval. The other temporal types are
	// all a count since some origin, so any of them can be recomputed into any
	// other, losing precision or gaining zeroes on the way.
	if from.Kind() == IntervalKind || to.Kind() == IntervalKind {
		return from.Kind() == IntervalKind && to.Kind() == IntervalKind
	}

	// A duration is a span and the rest are points. Adding a span to an origin
	// that was never stated is a guess, so that pair is a function and not a
	// cast, except that a duration and a time of day are both a count from a
	// midnight and the caller may well mean it.
	fromSpan := from.Kind() == DurationKind
	toSpan := to.Kind() == DurationKind
	if fromSpan != toSpan {
		return isTimeOfDay(from) || isTimeOfDay(to)
	}
	return true
}

func isTimeOfDay(t DataType) bool {
	k := t.Kind()
	return k == Time32Kind || k == Time64Kind
}

// canCastNested handles the cases where at least one side has children. The
// shapes have to line up, since a cast changes types and not structure.
func canCastNested(from, to DataType, depth int) bool {
	// Text is the one flat type a nested value can become, because printing one
	// is always possible.
	if IsNested(from) && IsString(to) {
		return true
	}

	fromElem, fromList := listElem(from)
	toElem, toList := listElem(to)
	if fromList && toList {
		return canCast(fromElem, toElem, depth+1)
	}

	if fs, ok := from.(Struct); ok {
		ts, ok := to.(Struct)
		if !ok || len(fs.Fields) != len(ts.Fields) {
			return false
		}
		for i := range fs.Fields {
			// Fields are matched by position and the names have to agree, so
			// that a cast never quietly moves a column's values into a
			// different field.
			if fs.Fields[i].Name != ts.Fields[i].Name {
				return false
			}
			if !canCast(fs.Fields[i].Type, ts.Fields[i].Type, depth+1) {
				return false
			}
		}
		return true
	}

	if fm, ok := from.(Map); ok {
		tm, ok := to.(Map)
		if !ok {
			return false
		}
		return canCast(fm.Key, tm.Key, depth+1) && canCast(fm.Value, tm.Value, depth+1)
	}

	return false
}

// listElem returns the element type of the three list shaped types. The three
// differ in how the rows are delimited and not in what they hold, so a cast
// between any two of them is a cast of the elements.
func listElem(t DataType) (DataType, bool) {
	switch x := t.(type) {
	case List:
		return x.Elem, true
	case LargeList:
		return x.Elem, true
	case FixedSizeList:
		return x.Elem, true
	}
	return nil, false
}
