package kuma_test

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

func TestNewColumn(t *testing.T) {
	c, err := array.NewChunked(dtype.Int64, array.Of[int64](1, 2, 3))
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}

	col, err := kuma.NewColumn("qty", c)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	if col.Name() != "qty" || col.Len() != 3 || col.NullCount() != 0 {
		t.Errorf("%s, want a column of 3 values called qty", col)
	}
	if !dtype.Equal(col.DType(), dtype.Int64) {
		t.Errorf("DType() = %s, want int64", col.DType())
	}
	if col.Data() != c {
		t.Error("NewColumn copied the values")
	}
	if !col.IsValid(0) || col.IsNull(0) {
		t.Error("value 0 is missing, want it present")
	}

	if _, err := kuma.NewColumn("qty", nil); !errors.Is(err, kuma.ErrNoValues) {
		t.Errorf("NewColumn with no values gave %v, want ErrNoValues", err)
	}
}

func TestColumnAs(t *testing.T) {
	col := kuma.NewSeries("price", 1.5, 2.5).Column()

	s, err := col.As[float64]()
	if err != nil {
		t.Fatalf("As: %v", err)
	}
	if s.Name() != "price" || s.Value(1) != 2.5 {
		t.Errorf("As gave %s", s)
	}

	if _, err := col.As[int64](); !errors.Is(err, kuma.ErrWrongType) {
		t.Errorf("As[int64] on a float64 column gave %v, want ErrWrongType", err)
	}

	if got := col.MustAs[float64]().Value(0); got != 1.5 {
		t.Errorf("MustAs gave %v", got)
	}

	t.Run("MustAs panics", func(t *testing.T) {
		defer func() {
			r := recover()
			if r == nil {
				t.Fatal("did not panic")
			}
			msg, ok := r.(string)
			if !ok {
				t.Fatalf("panicked with %T, want a string", r)
			}
			if !strings.Contains(msg, "does not read as a int64") {
				t.Errorf("panicked with %q", msg)
			}
		}()
		_ = col.MustAs[int64]()
	})
}

// TestColumnField is the rule that a frame's schema describes the data rather
// than a declaration about it, so a column is nullable when it has nulls.
func TestColumnField(t *testing.T) {
	plain := kuma.NewSeries("qty", int64(1), 2, 3).Column()
	if f := plain.Field(); f.Nullable {
		t.Errorf("%s is nullable, want it not null", f)
	}

	nulls := nullableInts(t, 10).Column()
	f := nulls.Field()
	if !f.Nullable {
		t.Errorf("%s is not nullable, want it nullable", f)
	}
	if f.Name != "qty" || !dtype.Equal(f.Type, dtype.Int64) {
		t.Errorf("Field() = %s", f)
	}

	// Cutting the nulls out of the range makes the column not nullable, which
	// is what lets a writer pick the narrower encoding.
	if f := nulls.Slice(1, 3).Field(); f.Nullable {
		t.Errorf("%s is nullable, want it not null", f)
	}
}

func TestColumnSliceAndRename(t *testing.T) {
	col := kuma.NewSeries("qty", int64(0), 1, 2, 3, 4).Column()

	cut := col.Slice(1, 4)
	if cut.Len() != 3 {
		t.Fatalf("Slice(1, 4).Len() = %d, want 3", cut.Len())
	}
	if got := cut.MustAs[int64]().Values(); got[0] != 1 || got[2] != 3 {
		t.Errorf("Slice(1, 4) holds %v", got)
	}
	if col.Len() != 5 {
		t.Errorf("Slice changed the column it was called on, which now has %d values", col.Len())
	}

	if got := col.Rename("quantity").Name(); got != "quantity" {
		t.Errorf("Rename gave %q", got)
	}
	if col.Name() != "qty" {
		t.Errorf("Rename changed the column it was called on, which is now %q", col.Name())
	}
}

func TestColumnString(t *testing.T) {
	col := nullableInts(t, 10).Column()

	want := `kuma.Column{"qty", int64, len 10, nulls 4}`
	if got := col.String(); got != want {
		t.Errorf("String() = %q, want %q", got, want)
	}
}

func TestColumnTake(t *testing.T) {
	c := kuma.NewSeries("qty", int64(10), 20, 30).Column()

	got := c.Take([]int{2, -1, 0})
	if got.Len() != 3 || got.NullCount() != 1 {
		t.Fatalf("%s, want 3 values and 1 null", got)
	}
	if got.Name() != "qty" {
		t.Errorf("Take gave a column called %q", got.Name())
	}
	if want := []int64{30, 0, 10}; !slices.Equal(got.MustAs[int64]().Values(), want) {
		t.Errorf("Take gave %v, want %v", got.MustAs[int64]().Values(), want)
	}
	if !got.Field().Nullable {
		t.Error("the field is not nullable, and the column has a null in it")
	}
}

func TestColumnCast(t *testing.T) {
	c := kuma.NewSeries("qty", "1", "2", "x").Column()

	if _, err := c.Cast(dtype.Int64); err == nil {
		t.Fatal("Cast of a column with an x in it succeeded")
	}

	got, err := c.TryCast(dtype.Int64)
	if err != nil {
		t.Fatalf("TryCast: %v", err)
	}
	if got.Name() != "qty" {
		t.Errorf("TryCast gave a column called %q, want %q", got.Name(), "qty")
	}
	if got.NullCount() != 1 {
		t.Errorf("TryCast gave %d nulls, want 1", got.NullCount())
	}
	if c.NullCount() != 0 {
		t.Error("TryCast changed the column it was called on")
	}

	if _, err := got.Cast(dtype.Binary); err == nil {
		t.Fatal("a cast from int64 to binary succeeded")
	}
}
