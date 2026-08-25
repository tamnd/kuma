package dtype_test

import (
	"testing"

	"github.com/tamnd/kuma/dtype"
)

// FuzzNames checks the promise the rest of the package leans on, that two types
// print the same name if and only if Equal reports them equal.
//
// It is worth fuzzing rather than tabulating because the two sides are written
// separately: String is a method on each type and Equal is one switch, so a
// parameter added to a type reaches one of them and not the other. The failure
// is an error message that says the two columns have the same type and a Concat
// that refuses them anyway.
func FuzzNames(f *testing.F) {
	f.Add([]byte{0}, []byte{0})
	f.Add([]byte{20, 1}, []byte{20, 1})
	f.Add([]byte{20, 1}, []byte{20, 2})
	f.Add([]byte{28, 3, 0, 5}, []byte{28, 3, 0, 6})

	f.Fuzz(func(t *testing.T, sa, sb []byte) {
		a := buildType(&seed{b: sa}, 0)
		b := buildType(&seed{b: sb}, 0)

		equal := dtype.Equal(a, b)
		sameName := a.String() == b.String()
		if equal != sameName {
			t.Errorf("Equal = %v but names %q and %q compare %v",
				equal, a, b, sameName)
		}

		// Equal has to agree with itself both ways round, which the recursive
		// cases are where an implementation stops doing.
		if dtype.Equal(b, a) != equal {
			t.Errorf("Equal(%s, %s) = %v, reversed = %v", a, b, equal, !equal)
		}
		if !dtype.Equal(a, a) {
			t.Errorf("Equal(%s, %s) = false", a, a)
		}
	})
}

// FuzzValidate checks that Validate returns rather than panicking or running
// forever on any type the generator can produce, including the deliberately
// broken ones, and that a type it accepts has a name it can also produce.
func FuzzValidate(f *testing.F) {
	f.Add([]byte{0})
	f.Add([]byte{20, 200, 3})
	f.Add([]byte{28, 28, 28, 28, 28, 28, 1})

	f.Fuzz(func(t *testing.T, s []byte) {
		typ := buildType(&seed{b: s}, 0)

		err := dtype.Validate(typ)
		if err != nil && err.Error() == "" {
			t.Errorf("Validate(%s) returned an error with no message", typ)
		}

		// Validating twice has to give the same answer, since a schema is
		// validated once when it is built and relied on afterwards.
		if second := dtype.Validate(typ); (second == nil) != (err == nil) {
			t.Errorf("Validate(%s) = %v then %v", typ, err, second)
		}

		if name := typ.String(); name == "" {
			t.Errorf("a type with kind %v has no name", typ.Kind())
		}
	})
}

// FuzzCoerce checks the three promises the planner is built on: that Coerce
// gives the same answer whichever column is named first, that a type it hands
// back is one both sides can actually be cast to, and that any type at all can
// be printed.
//
// The first one matters because a query does not say which side of a union or
// a join was written first, and a rule that leans on argument order turns into
// a plan that succeeds one way and fails the other. The second matters because
// Coerce runs at plan time and the cast runs later, so a pair where they
// disagree is a plan that type checks and then cannot be executed.
func FuzzCoerce(f *testing.F) {
	f.Add([]byte{0}, []byte{3})
	f.Add([]byte{26, 3}, []byte{26, 0})
	f.Add([]byte{31, 6, 12}, []byte{31, 8, 12})
	f.Add([]byte{29, 0, 3, 0, 29, 0, 3, 1}, []byte{29, 0, 0, 0, 29, 0, 3, 1})

	f.Fuzz(func(t *testing.T, sa, sb []byte) {
		a := buildType(&seed{b: sa}, 0)
		b := buildType(&seed{b: sb}, 0)

		got, err := dtype.Coerce(a, b)
		back, backErr := dtype.Coerce(b, a)
		if (err == nil) != (backErr == nil) {
			t.Fatalf("Coerce(%s, %s) = %v but Coerce(%s, %s) = %v", a, b, err, b, a, backErr)
		}
		if err != nil {
			return
		}
		if !dtype.Equal(got, back) {
			t.Errorf("Coerce(%s, %s) = %s but reversed = %s", a, b, got, back)
		}

		if !dtype.CanCast(a, got) {
			t.Errorf("Coerce(%s, %s) = %s, which %s cannot be cast to", a, b, got, a)
		}
		if !dtype.CanCast(b, got) {
			t.Errorf("Coerce(%s, %s) = %s, which %s cannot be cast to", a, b, got, b)
		}

		// Printing is the one cast with no conditions on it, and a good deal
		// of debugging stops working the day that stops being true.
		if !dtype.CanCast(got, dtype.String) {
			t.Errorf("CanCast(%s, string) = false", got)
		}
	})
}

// seed hands out bytes from a fuzz input, repeating the input rather than
// running out, so that a short input still builds a whole type.
type seed struct {
	b []byte
	i int
}

func (s *seed) next() byte {
	if len(s.b) == 0 {
		return 0
	}
	v := s.b[s.i%len(s.b)]
	s.i++
	return v
}

// buildType turns fuzz bytes into a type. The nested cases recurse, so a
// pathological input builds a deep tree, which is the case the depth limit in
// Validate exists for. The limit here is lower so that both sides of it get
// exercised.
func buildType(s *seed, depth int) dtype.DataType {
	flat := []dtype.DataType{
		dtype.Null, dtype.Bool,
		dtype.Int8, dtype.Int16, dtype.Int32, dtype.Int64,
		dtype.Uint8, dtype.Uint16, dtype.Uint32, dtype.Uint64,
		dtype.Float32, dtype.Float64,
		dtype.String, dtype.Binary, dtype.LargeString, dtype.LargeBinary,
		dtype.Date32, dtype.Date64,
	}

	choice := int(s.next())
	if depth > 8 {
		return flat[choice%len(flat)]
	}

	switch choice % 32 {
	case 18:
		return dtype.FixedSizeBinary{ByteWidth: int32(s.next()) - 8}
	case 19:
		return dtype.Time32{Unit: dtype.TimeUnit(s.next() % 6)}
	case 20:
		return dtype.Time64{Unit: dtype.TimeUnit(s.next() % 6)}
	case 21:
		return dtype.Timestamp{Unit: dtype.TimeUnit(s.next() % 6), Zone: zone(s.next())}
	case 22:
		return dtype.Duration{Unit: dtype.TimeUnit(s.next() % 6)}
	case 23:
		return dtype.Interval{Unit: dtype.IntervalUnit(s.next() % 5)}
	case 24:
		return dtype.Decimal128{Precision: int32(s.next()) - 8, Scale: int32(s.next()) - 8}
	case 25:
		return dtype.Decimal256{Precision: int32(s.next()) - 8, Scale: int32(s.next()) - 8}
	case 26:
		return dtype.List{Elem: buildType(s, depth+1)}
	case 27:
		return dtype.LargeList{Elem: buildType(s, depth+1)}
	case 28:
		return dtype.FixedSizeList{Elem: buildType(s, depth+1), Len: int32(s.next()) - 8}
	case 29:
		n := int(s.next() % 4)
		fields := make([]dtype.Field, n)
		for i := range fields {
			fields[i] = dtype.Field{
				Name:     fieldName(s.next()),
				Type:     buildType(s, depth+1),
				Nullable: s.next()%2 == 0,
			}
		}
		return dtype.Struct{Fields: fields}
	case 30:
		return dtype.Map{Key: buildType(s, depth+1), Value: buildType(s, depth+1)}
	case 31:
		return dtype.Dictionary{Index: buildType(s, depth+1), Value: buildType(s, depth+1)}
	default:
		return flat[choice%len(flat)]
	}
}

// zone returns an empty zone half the time, since a naive timestamp and a
// zoned one are different types and both are common.
func zone(b byte) string {
	names := []string{"", "", "UTC", "Europe/London", "America/New_York", "Not/AZone"}
	return names[int(b)%len(names)]
}

// fieldName draws from a small pool so that duplicate names come up often,
// which is what the schema validation rule is about.
func fieldName(b byte) string {
	names := []string{"a", "b", "c", ""}
	return names[int(b)%len(names)]
}
