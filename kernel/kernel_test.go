package kernel_test

import (
	"testing"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// col builds a column of type dt out of one chunk per list of values, where a
// nil value is a null.
//
// Tests here are about positions and nulls rather than about arithmetic, so
// what matters is being able to write the shape of a column down in one line
// and see it. The cost is a type switch per value, which is fine in a test.
func col(t *testing.T, dt dtype.DataType, chunks ...[]any) *array.Chunked {
	t.Helper()

	b, err := array.NewBuilder(dt)
	if err != nil {
		t.Fatalf("NewBuilder(%s): %v", dt, err)
	}

	arrays := make([]*array.Array, len(chunks))
	for i, values := range chunks {
		for _, v := range values {
			appendValue(t, b, v)
		}
		arrays[i] = b.Finish()
	}

	c, err := array.NewChunked(dt, arrays...)
	if err != nil {
		t.Fatalf("NewChunked(%s): %v", dt, err)
	}
	return c
}

// appendValue adds one value of whatever type it turns out to be.
func appendValue(t *testing.T, b *array.Builder, v any) {
	t.Helper()

	switch v := v.(type) {
	case nil:
		b.AppendNull()
	case bool:
		b.AppendBool(v)
	case int8:
		b.Append(v)
	case int16:
		b.Append(v)
	case int32:
		b.Append(v)
	case int64:
		b.Append(v)
	case uint8:
		b.Append(v)
	case uint16:
		b.Append(v)
	case uint32:
		b.Append(v)
	case uint64:
		b.Append(v)
	case float32:
		b.Append(v)
	case float64:
		b.Append(v)
	case string:
		b.AppendString(v)
	case []byte:
		b.AppendBytes(v)
	default:
		t.Fatalf("no way to append a %T", v)
	}
}

// valueAt returns value i of a column as something comparable, or nil if it is
// missing. It is what the tests compare, since the point of a gather is that
// the value at the new position is the value that was at the old one.
func valueAt(t *testing.T, c *array.Chunked, i int) any {
	t.Helper()

	if c.IsNull(i) {
		return nil
	}

	// A dictionary encoded column is the value behind the index rather than the
	// index, since the point of a gather over one is that the value at the new
	// position is still the value that was at the old one. The dictionary is an
	// ordinary array of the value type, so reading it is this function again.
	if _, ok := c.DType().(dtype.Dictionary); ok {
		a, k := c.At(i)
		d := a.Dictionary()
		one, err := array.NewChunked(d.DType(), d)
		if err != nil {
			t.Fatalf("NewChunked(%s): %v", d.DType(), err)
		}
		return valueAt(t, one, a.Index(k))
	}

	switch c.DType().Kind() {
	case dtype.BoolKind:
		return c.Bool(i)
	case dtype.Int8Kind:
		return c.Value[int8](i)
	case dtype.Int16Kind:
		return c.Value[int16](i)
	case dtype.Int32Kind, dtype.Date32Kind, dtype.Time32Kind:
		return c.Value[int32](i)
	case dtype.Int64Kind, dtype.Date64Kind, dtype.Time64Kind,
		dtype.TimestampKind, dtype.DurationKind:
		return c.Value[int64](i)
	case dtype.Uint8Kind:
		return c.Value[uint8](i)
	case dtype.Uint16Kind:
		return c.Value[uint16](i)
	case dtype.Uint32Kind:
		return c.Value[uint32](i)
	case dtype.Uint64Kind:
		return c.Value[uint64](i)
	case dtype.Float32Kind:
		return c.Value[float32](i)
	case dtype.Float64Kind:
		return c.Value[float64](i)
	default:
		return string(c.Bytes(i))
	}
}

// checkTake is the property every gather has to hold: the result is as long as
// the list of positions, and value k of the result is the value that was at
// position idx[k], or a null when that position is below zero.
func checkTake(t *testing.T, got, src *array.Chunked, idx []int) {
	t.Helper()

	if !dtype.Equal(got.DType(), src.DType()) {
		t.Errorf("the result is a %s column, want %s", got.DType(), src.DType())
	}
	if got.Len() != len(idx) {
		t.Fatalf("the result has %d values, want %d", got.Len(), len(idx))
	}

	nulls := 0
	for k, i := range idx {
		var want any
		if i >= 0 {
			want = valueAt(t, src, i)
		}
		if want == nil {
			nulls++
		}
		if have := valueAt(t, got, k); have != want {
			t.Errorf("value %d, taken from %d, is %v, want %v", k, i, have, want)
		}
	}
	if got.NullCount() != nulls {
		t.Errorf("the result counts %d nulls, want %d", got.NullCount(), nulls)
	}
}
