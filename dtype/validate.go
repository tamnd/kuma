package dtype

import (
	"errors"
	"fmt"
)

// MaxNestingDepth is how deep a type may be nested before Validate gives up.
//
// The limit exists because DataType is an interface anyone can implement, so a
// type that contains itself is constructible, and a recursive walk over one
// never returns. Real schemas are a few levels deep. A list of structs of lists
// is three.
const MaxNestingDepth = 64

// ErrTooDeep is what Validate returns, wrapped, when a type nests past
// MaxNestingDepth.
var ErrTooDeep = errors.New("type nests too deeply")

// Validate reports whether t describes a real type.
//
// The parameterized types are composite literals, so a caller can write one
// that does not mean anything: a time32 in nanoseconds does not fit in 32 bits,
// a decimal with a scale larger than its precision has no digits left, a
// dictionary indexed by a string cannot be indexed. Validate is where those are
// caught, in one place, rather than in every kernel that receives a type.
//
// It walks the children of the nested types, so validating the outermost type
// validates the whole tree, and the error says which part is wrong.
//
// Timestamp.Zone is not checked. It is deliberately not resolved against the
// timezone database, because a binary built without tzdata, which is most
// containers, would then reject a schema that is entirely valid on the machine
// that wrote it. A zone is resolved where the arithmetic that needs it happens.
func Validate(t DataType) error {
	if err := validate(t, 0); err != nil {
		return fmt.Errorf("dtype: %w", err)
	}
	return nil
}

// Validate reports whether s is a schema a frame can be built from, meaning
// every field has a name, no two fields share one, and every type is valid.
//
// Duplicate names are rejected here rather than by the Schema type itself,
// because a CSV with two columns called "id" is a real file, and a reader has
// to be able to describe it before it can rename anything.
func (s Schema) Validate() error {
	if err := validateFields(s.Fields, 0); err != nil {
		return fmt.Errorf("dtype: schema: %w", err)
	}
	return nil
}

// Validate reports whether f has a name and a valid type.
func (f Field) Validate() error {
	if err := validateField(f, 0); err != nil {
		return fmt.Errorf("dtype: %w", err)
	}
	return nil
}

// validate returns errors with no package prefix. The exported entry points
// above add it once, so that a nested failure reads as one sentence rather than
// as "dtype: list: dtype: struct: dtype: ...".
func validate(t DataType, depth int) error {
	if depth > MaxNestingDepth {
		return ErrTooDeep
	}
	if t == nil {
		return errors.New("nil type")
	}

	switch x := t.(type) {
	case Time32:
		// Seconds and milliseconds since midnight fit in an int32 and the finer
		// units do not, which is why there are two time types rather than one.
		if x.Unit != Second && x.Unit != Millisecond {
			return fmt.Errorf("time32 unit must be s or ms, have %s", x.Unit)
		}
	case Time64:
		if x.Unit != Microsecond && x.Unit != Nanosecond {
			return fmt.Errorf("time64 unit must be us or ns, have %s", x.Unit)
		}
	case Timestamp:
		if !x.Unit.Valid() {
			return fmt.Errorf("timestamp has unknown unit %s", x.Unit)
		}
	case Duration:
		if !x.Unit.Valid() {
			return fmt.Errorf("duration has unknown unit %s", x.Unit)
		}
	case Interval:
		if !x.Unit.Valid() {
			return fmt.Errorf("interval has unknown unit %s", x.Unit)
		}
	case Decimal128:
		return validateDecimal("decimal128", x.Precision, x.Scale, MaxDecimal128Precision)
	case Decimal256:
		return validateDecimal("decimal256", x.Precision, x.Scale, MaxDecimal256Precision)
	case FixedSizeBinary:
		if x.ByteWidth < 0 {
			return fmt.Errorf("fixed_size_binary has negative width %d", x.ByteWidth)
		}
	case List:
		return wrap("list", validate(x.Elem, depth+1))
	case LargeList:
		return wrap("large_list", validate(x.Elem, depth+1))
	case FixedSizeList:
		if x.Len < 0 {
			return fmt.Errorf("fixed_size_list has negative length %d", x.Len)
		}
		return wrap("fixed_size_list", validate(x.Elem, depth+1))
	case Struct:
		return wrap("struct", validateFields(x.Fields, depth))
	case Map:
		if err := wrap("map key", validate(x.Key, depth+1)); err != nil {
			return err
		}
		return wrap("map value", validate(x.Value, depth+1))
	case Dictionary:
		if !IsInteger(x.Index) {
			return fmt.Errorf("dictionary index must be an integer type, have %s",
				typeName(x.Index))
		}
		// A dictionary whose values are themselves dictionary encoded has two
		// levels of indirection and no reader agrees on what it means.
		if x.Value != nil && x.Value.Kind() == DictionaryKind {
			return errors.New("dictionary of dictionary")
		}
		return wrap("dictionary value", validate(x.Value, depth+1))
	default:
		// Everything left is parameterless, so the only thing that can be wrong
		// is that it is not one of ours.
		if k := t.Kind(); k == InvalidKind || int(k) >= len(kindNames) {
			return fmt.Errorf("unknown type %s", typeName(t))
		}
	}
	return nil
}

func validateDecimal(name string, precision, scale, maxPrecision int32) error {
	if precision < 1 || precision > maxPrecision {
		return fmt.Errorf("%s precision %d out of range 1 to %d", name, precision, maxPrecision)
	}
	// A scale larger than the precision leaves no significant digits, and a
	// scale below the negative of it is the same problem the other way around.
	// Arrow draws the line in the same place.
	if scale < -precision || scale > precision {
		return fmt.Errorf("%s scale %d out of range %d to %d",
			name, scale, -precision, precision)
	}
	return nil
}

// validateFields is the shared body of validating a struct type and validating
// a schema, since a schema is a struct with metadata as far as these rules go.
func validateFields(fields []Field, depth int) error {
	seen := make(map[string]struct{}, len(fields))
	for i, f := range fields {
		if f.Name == "" {
			return fmt.Errorf("field %d has no name", i)
		}
		if _, dup := seen[f.Name]; dup {
			return fmt.Errorf("two fields named %q", f.Name)
		}
		seen[f.Name] = struct{}{}

		if err := validateField(f, depth); err != nil {
			return err
		}
	}
	return nil
}

func validateField(f Field, depth int) error {
	if f.Name == "" {
		return errors.New("field has no name")
	}
	return wrap(fmt.Sprintf("field %q", f.Name), validate(f.Type, depth+1))
}

// wrap adds one level of context, except to ErrTooDeep, which would otherwise
// arrive with sixty four levels of it.
func wrap(what string, err error) error {
	if err == nil || errors.Is(err, ErrTooDeep) {
		return err
	}
	return fmt.Errorf("%s: %w", what, err)
}
