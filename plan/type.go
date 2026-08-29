package plan

import (
	"fmt"
	"time"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

// TypeOf returns the type an expression has over a schema, and an error saying
// what is wrong with it when it has none.
//
// This is where a query is turned away before it runs. A name that is not in
// the schema, two columns with no type in common, a literal that does not fit
// the column it was written against and a filter on something that is not a
// condition are all caught here, against nothing but the column types, so the
// caller hears about them while the plan is being built rather than partway
// through the second file.
//
// The answer is the type the column would come out as, which is what the
// operator above this expression needs to know and what an explain prints.
// Every rule it uses is the kernel's own, by way of [kernel.ArithType] and
// [kernel.CompareType], so an expression this accepts is one the kernels will
// run and the type it promises is the type that arrives.
func TypeOf(e *Expr, s dtype.Schema) (dtype.DataType, error) {
	return typeOf(e, s, nil)
}

// typeOf is TypeOf with the hint the evaluator carries: the type the other side
// of a two sided step turned out to be. It is used by a literal and ignored by
// everything else, for the reason the evaluator gives.
func typeOf(e *Expr, s dtype.Schema, hint dtype.DataType) (dtype.DataType, error) {
	switch e.Kind() {
	case KindColumn:
		f, ok := s.Field(e.Name())
		if !ok {
			return nil, noColumn("", e.Name(), s.Names())
		}
		return f.Type, nil
	case KindLiteral:
		return LiteralTypeAgainst(e.Value(), hint)
	case KindCompare:
		a, b, err := operandTypes(e, s)
		if err != nil {
			return nil, err
		}
		return kernel.CompareType(a, b)
	case KindArith:
		a, b, err := operandTypes(e, s)
		if err != nil {
			return nil, err
		}
		return kernel.ArithType(a, b, e.ArithOp())
	case KindAnd, KindOr:
		if err := conditionType(e.Left(), s); err != nil {
			return nil, err
		}
		if err := conditionType(e.Right(), s); err != nil {
			return nil, err
		}
		return dtype.Bool, nil
	case KindNot:
		if err := conditionType(e.Left(), s); err != nil {
			return nil, err
		}
		return dtype.Bool, nil
	case KindIsNull, KindIsNotNull:
		// Whether a value is there is a question any column answers, so the
		// only thing to check is that the column is.
		if _, err := typeOf(e.Left(), s, nil); err != nil {
			return nil, err
		}
		return dtype.Bool, nil
	default:
		from, err := typeOf(e.Left(), s, nil)
		if err != nil {
			return nil, err
		}
		if !dtype.CanCast(from, e.DType()) {
			return nil, fmt.Errorf("kuma: %s cannot be cast to %s: %w", e.Left(), e.DType(), ErrWrongType)
		}
		return e.DType(), nil
	}
}

// operandTypes works out the types of both sides of a two sided step, doing the
// side that has a type of its own first so that the literal on the other side
// knows what to become.
//
// It is the evaluator's own order. What a literal becomes depends on what it is
// used with, so working the two sides out the other way round would give a plan
// that says int64 where the engine produces uint32.
func operandTypes(e *Expr, s dtype.Schema) (a, b dtype.DataType, err error) {
	l, r := e.Left(), e.Right()
	if l.Kind() == KindLiteral && r.Kind() != KindLiteral {
		b, err = typeOf(r, s, nil)
		if err != nil {
			return nil, nil, err
		}
		a, err = typeOf(l, s, b)
		if err != nil {
			return nil, nil, err
		}
		return a, b, nil
	}

	a, err = typeOf(l, s, nil)
	if err != nil {
		return nil, nil, err
	}
	b, err = typeOf(r, s, a)
	if err != nil {
		return nil, nil, err
	}
	return a, b, nil
}

// conditionType checks that one side of a logical step is something there is a
// three valued logic for, which is a boolean or a column of nothing.
func conditionType(e *Expr, s dtype.Schema) error {
	dt, err := typeOf(e, s, dtype.Bool)
	if err != nil {
		return err
	}
	if !kernel.IsCondition(dt) {
		return fmt.Errorf("kuma: %s is a %s and not a condition: %w", e, dt, ErrWrongType)
	}
	return nil
}

// LiteralType returns the type a value written in a query has on its own,
// before it meets a column, and false for a value no column can hold.
//
// The list is the one [github.com/tamnd/kuma.Lit] accepts, and it is short on
// purpose: the numbers, the two kinds of text, a bool, a time and nothing else.
// A value of any other type is a mistake at the point it was written, and
// saying so there is more use than a column of something nobody can read.
func LiteralType(v any) (dtype.DataType, bool) {
	switch v.(type) {
	case nil:
		return dtype.Null, true
	case bool:
		return dtype.Bool, true
	case int:
		return dtype.Int64, true
	case int8:
		return dtype.Int8, true
	case int16:
		return dtype.Int16, true
	case int32:
		return dtype.Int32, true
	case int64:
		return dtype.Int64, true
	case uint:
		return dtype.Uint64, true
	case uint8:
		return dtype.Uint8, true
	case uint16:
		return dtype.Uint16, true
	case uint32:
		return dtype.Uint32, true
	case uint64:
		return dtype.Uint64, true
	case float32:
		return dtype.Float32, true
	case float64:
		return dtype.Float64, true
	case string:
		return dtype.String, true
	case []byte:
		return dtype.Binary, true
	case time.Time:
		return dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "UTC"}, true
	default:
		return nil, false
	}
}

// LiteralTypeAgainst returns the type a literal takes when it is used with a
// column of type dt, which is nil for a literal that is not being used with one
// and gets [LiteralType]'s answer.
//
// This is the rule that keeps a comparison against a uint32 column a uint32
// comparison rather than widening the column to an int64, and it is the rule
// that refuses 1.5 against an integer column instead of quietly comparing
// against 1. What is allowed is [dtype.CoerceLiteral]'s decision. A time is the
// one value with no type of its own to coerce, so it takes the column's unit
// and zone, or nanoseconds in UTC when there is no column to take them from.
func LiteralTypeAgainst(v any, dt dtype.DataType) (dtype.DataType, error) {
	lit, ok := LiteralType(v)
	if !ok {
		return nil, fmt.Errorf("kuma: %T is not a value a column can hold: %w", v, ErrWrongType)
	}
	if dt == nil {
		return lit, nil
	}
	if _, isTime := v.(time.Time); isTime {
		if ts, ok := dt.(dtype.Timestamp); ok {
			return ts, nil
		}
		// The column is not a timestamp, so there is no unit to take from it.
		// What the other side makes of nanoseconds is its own business, and it
		// is usually an error a line further on.
		return lit, nil
	}

	want, err := dtype.CoerceLiteral(dt, lit)
	if err != nil {
		return nil, fmt.Errorf("kuma: %w", err)
	}
	return want, nil
}
