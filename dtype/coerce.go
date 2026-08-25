package dtype

import "fmt"

// Coerce returns the type that two columns have in common, or an error saying
// they have none.
//
// It is strict on purpose. Two int64 columns give an int64. An int64 column and
// a float64 column give an error, not a float64, even though every int64 has a
// float64 near it. This is the rule from document 02 and it is the single
// biggest correctness difference from pandas, where the upcast happens quietly
// and an id column that fit exactly in an int64 comes out the other side as a
// float64 with the low bits rounded off. Polars is strict here and is right to
// be. The error arrives when the plan is built, before any data is read, and it
// names the cast the caller has to write.
//
// The exceptions are the cases where nothing can be lost:
//
// A null column combines with anything and takes the other type. Every value in
// it is missing, so there is nothing to convert.
//
// A dictionary combines with its own value type and with another dictionary
// over that value type. Dictionary encoding is how the values are stored, not
// what they are, so a dictionary of strings and a column of strings are the
// same column written two ways. Two dictionaries keep the encoding and take
// whichever index type holds both.
//
// The nested types combine element by element, so a list of null and a list of
// int64 give a list of int64, which is the case that turns up whenever an empty
// list is read from JSON.
//
// Coerce is for two columns. A literal follows different rules, because the
// caller wrote it and it has no storage to preserve. See CoerceLiteral.
func Coerce(a, b DataType) (DataType, error) {
	t, ok := coerce(a, b, 0)
	if !ok {
		return nil, fmt.Errorf("dtype: cannot combine %s and %s, cast one side explicitly",
			typeName(a), typeName(b))
	}
	return t, nil
}

func coerce(a, b DataType, depth int) (DataType, bool) {
	if a == nil || b == nil || depth > MaxNestingDepth {
		return nil, false
	}
	if Equal(a, b) {
		return a, true
	}

	// A column of nothing has no values to convert, so it takes the other side.
	if a.Kind() == NullKind {
		return b, true
	}
	if b.Kind() == NullKind {
		return a, true
	}

	// Dictionary encoding is storage rather than meaning, so it is unwrapped
	// and reapplied rather than compared.
	if da, ok := a.(Dictionary); ok {
		return coerceDictionary(da, b, depth)
	}
	if db, ok := b.(Dictionary); ok {
		return coerceDictionary(db, a, depth)
	}

	if a.Kind() != b.Kind() {
		return nil, false
	}

	switch x := a.(type) {
	case List:
		y, ok := b.(List)
		if !ok {
			return nil, false
		}
		return coerceChild(x.Elem, y.Elem, depth, func(e DataType) DataType {
			return List{Elem: e}
		})
	case LargeList:
		y, ok := b.(LargeList)
		if !ok {
			return nil, false
		}
		return coerceChild(x.Elem, y.Elem, depth, func(e DataType) DataType {
			return LargeList{Elem: e}
		})
	case FixedSizeList:
		y, ok := b.(FixedSizeList)
		if !ok || x.Len != y.Len {
			return nil, false
		}
		return coerceChild(x.Elem, y.Elem, depth, func(e DataType) DataType {
			return FixedSizeList{Elem: e, Len: x.Len}
		})
	case Map:
		y, ok := b.(Map)
		if !ok {
			return nil, false
		}
		key, ok := coerce(x.Key, y.Key, depth+1)
		if !ok {
			return nil, false
		}
		value, ok := coerce(x.Value, y.Value, depth+1)
		if !ok {
			return nil, false
		}
		return Map{Key: key, Value: value}, true
	case Struct:
		y, ok := b.(Struct)
		if !ok {
			return nil, false
		}
		return coerceStruct(x, y, depth)
	}

	// Same kind, different parameters, and not a nested type. That is a
	// timestamp in two different units, or two decimals of different scales,
	// and picking one of them for the caller is exactly the guess this package
	// refuses to make.
	return nil, false
}

// coerceChild is the single child case, which three of the list types share.
func coerceChild(a, b DataType, depth int, wrap func(DataType) DataType) (DataType, bool) {
	elem, ok := coerce(a, b, depth+1)
	if !ok {
		return nil, false
	}
	return wrap(elem), true
}

// coerceDictionary combines a dictionary with something that is not null and
// may or may not be another dictionary.
func coerceDictionary(a Dictionary, b DataType, depth int) (DataType, bool) {
	if db, ok := b.(Dictionary); ok {
		value, ok := coerce(a.Value, db.Value, depth+1)
		if !ok {
			return nil, false
		}
		index, ok := widenInteger(a.Index, db.Index)
		if !ok {
			// The values agree and the indexes do not fit in one integer type.
			// Decoding is still correct, it just costs the encoding.
			return value, true
		}
		return Dictionary{Index: index, Value: value}, true
	}

	// A dictionary against a plain column decodes, since the plain side has no
	// dictionary to merge into.
	return coerce(a.Value, b, depth+1)
}

func coerceStruct(a, b Struct, depth int) (DataType, bool) {
	if len(a.Fields) != len(b.Fields) {
		return nil, false
	}

	out := Struct{Fields: make([]Field, len(a.Fields))}
	for i := range a.Fields {
		x, y := a.Fields[i], b.Fields[i]
		if x.Name != y.Name {
			return nil, false
		}
		t, ok := coerce(x.Type, y.Type, depth+1)
		if !ok {
			return nil, false
		}
		// A field is nullable in the result if it was nullable on either side,
		// since the values from that side are still values.
		out.Fields[i] = Field{
			Name:     x.Name,
			Type:     t,
			Nullable: x.Nullable || y.Nullable,
			Metadata: x.Metadata,
		}
	}
	return out, true
}

// CoerceLiteral returns the type a comparison or an arithmetic operation
// between a column and a literal is carried out in.
//
// A literal is looser than a column because there is nothing to preserve. The
// caller wrote 1 in their own source and meant the number, not an int64, so the
// literal takes the column's type wherever that is exact and the column keeps
// its own storage. Writing df.Col("count").Gt(0) should not be a type error and
// should not quietly turn a uint32 column into an int64 one.
//
// It works on types, so it says whether a literal of that type can take the
// column's type at all. Whether the particular value fits, meaning whether 300
// fits in the int8 column it is being compared against, is a question about the
// value and is answered by the layer that has it.
//
// A float literal against an integer column is an error rather than a
// truncation, because 1.5 has no int64 spelling and rounding it silently is the
// same class of mistake as upcasting the column silently. The same goes for a
// float literal against a decimal column, where being exact is the entire
// reason the column is a decimal.
func CoerceLiteral(column, literal DataType) (DataType, error) {
	t, ok := coerceLiteral(column, literal, 0)
	if !ok {
		return nil, fmt.Errorf("dtype: cannot use a %s literal with a %s column, cast the column or write a %s literal",
			typeName(literal), typeName(column), typeName(column))
	}
	return t, nil
}

func coerceLiteral(column, literal DataType, depth int) (DataType, bool) {
	if column == nil || literal == nil || depth > MaxNestingDepth {
		return nil, false
	}
	if Equal(column, literal) {
		return column, true
	}

	// A null literal is the missing value, which every column already has a
	// place for. A null column takes the literal's type, since it has no type
	// of its own worth keeping.
	if literal.Kind() == NullKind {
		return column, true
	}
	if column.Kind() == NullKind {
		return literal, true
	}

	// The column keeps its encoding. A string literal against a dictionary of
	// strings compares against the dictionary, which is one comparison per
	// distinct value rather than one per row.
	if dc, ok := column.(Dictionary); ok {
		if _, ok := coerceLiteral(dc.Value, literal, depth+1); !ok {
			return nil, false
		}
		return column, true
	}

	switch {
	case IsInteger(column):
		// An integer literal takes the column's type. The range check is on
		// the value, not on the type, and belongs to the caller that has it.
		if IsInteger(literal) {
			return column, true
		}
	case IsFloat(column):
		if IsInteger(literal) || IsFloat(literal) {
			return column, true
		}
	case IsDecimal(column):
		if IsInteger(literal) {
			return column, true
		}
	case IsString(column):
		if IsString(literal) {
			return column, true
		}
	case IsBinary(column):
		if IsBinary(literal) {
			return column, true
		}
	case column.Kind() == BoolKind:
		// Nothing but a bool, which Equal already handled. An integer literal
		// against a bool column is the pandas behavior this package does not
		// want.
	case IsTemporal(column):
		// A temporal literal of the same kind takes the column's unit and zone.
		// Converting a literal between units is exact in one direction and the
		// caller has the value to check the other, which is the same split as
		// the integer range check above.
		if literal.Kind() == column.Kind() {
			return column, true
		}
	case IsNested(column):
		// A nested literal has to match structurally, and the element rules are
		// the column rules, since the elements are stored the same way.
		if t, ok := coerce(column, literal, depth+1); ok {
			return t, true
		}
	}
	return nil, false
}

// widenInteger returns the narrowest integer type that holds every value of
// both a and b, and whether there is one.
//
// It exists for dictionary indexes, where the index type is bookkeeping rather
// than data and merging two dictionaries has to put the indexes somewhere. It
// is deliberately not reachable from Coerce for two ordinary integer columns.
// The two cases look alike and only one of them is safe: nothing reads a
// dictionary index, so widening it changes nothing a caller can observe, while
// widening a column changes what comes out of the other end.
func widenInteger(a, b DataType) (DataType, bool) {
	if !IsInteger(a) || !IsInteger(b) {
		return nil, false
	}
	if Equal(a, b) {
		return a, true
	}

	abits, _ := Bits(a)
	bbits, _ := Bits(b)
	if IsSigned(a) == IsSigned(b) {
		if abits >= bbits {
			return a, true
		}
		return b, true
	}

	// One of each. The result has to be signed, and wide enough to hold the
	// unsigned side as well, which costs a doubling.
	signed, unsigned := a, b
	if IsUnsigned(a) {
		signed, unsigned = b, a
	}
	sbits, _ := Bits(signed)
	ubits, _ := Bits(unsigned)

	need := max(sbits, 2*ubits)
	switch need {
	case 16:
		return Int16, true
	case 32:
		return Int32, true
	case 64:
		return Int64, true
	}
	// A uint64 alongside anything signed does not fit in any integer type here.
	return nil, false
}
