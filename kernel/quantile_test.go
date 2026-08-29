package kernel_test

import (
	"math"
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
	"github.com/tamnd/kuma/kernel"
)

func TestMedian(t *testing.T) {
	g := groupsOf(t, "odd", "even", "odd", "even", "odd")
	c := col(t, dtype.Float64, []any{3.0, 10.0, 1.0, 20.0, 2.0})

	got, err := kernel.Median(c, g)
	if err != nil {
		t.Fatalf("Median: %v", err)
	}
	if got.DType() != dtype.Float64 {
		t.Errorf("the result is a %s column, want float64", got.DType())
	}
	// Three values give the middle one and two give the average of them.
	checkAgg(t, got, []any{2.0, 15.0})
}

func TestMedianSkipsTheMissing(t *testing.T) {
	g := groupsOf(t, "a", "a", "a", "b", "b")
	c := col(t, dtype.Int64, []any{int64(1), nil, int64(3), nil, nil})

	got, err := kernel.Median(c, g)
	if err != nil {
		t.Fatalf("Median: %v", err)
	}
	checkAgg(t, got, []any{2.0, nil})
}

// TestQuantileInterpolation is the five ways a quantile can land between two
// values, on the same four numbers so the answers can be read next to each
// other.
func TestQuantileInterpolation(t *testing.T) {
	g := kernel.OneGroup(4)
	c := col(t, dtype.Float64, []any{1.0, 2.0, 3.0, 4.0})

	tests := []struct {
		how  kernel.Interpolation
		name string
		q    float64
		want float64
	}{
		// A q of a quarter falls on position 0.75, which is three quarters of
		// the way from 1 to 2.
		{kernel.Linear, "linear", 0.25, 1.75},
		{kernel.Lower, "lower", 0.25, 1.0},
		{kernel.Higher, "higher", 0.25, 2.0},
		{kernel.Nearest, "nearest", 0.25, 2.0},
		{kernel.Midpoint, "midpoint", 0.25, 1.5},

		// A q of a half falls exactly between the two middle values, which is
		// where nearest has to break a tie and does it by taking the even
		// index, index 2 here, the same rule numpy rounds a position with.
		{kernel.Linear, "linear", 0.5, 2.5},
		{kernel.Nearest, "nearest", 0.5, 3.0},
		{kernel.Midpoint, "midpoint", 0.5, 2.5},

		// The ends land on a value, so every interpolation agrees.
		{kernel.Linear, "linear", 0, 1.0},
		{kernel.Higher, "higher", 0, 1.0},
		{kernel.Linear, "linear", 1, 4.0},
		{kernel.Lower, "lower", 1, 4.0},
	}

	for _, tt := range tests {
		t.Run(tt.name+" at "+dtype.Float64.String(), func(t *testing.T) {
			if tt.how.String() != tt.name {
				t.Fatalf("the name of %d is %q, want %q", int(tt.how), tt.how, tt.name)
			}
			got, err := kernel.Quantile(c, g, tt.q, tt.how)
			if err != nil {
				t.Fatalf("Quantile: %v", err)
			}
			if have := got.Value[float64](0); have != tt.want {
				t.Errorf("%s at %v is %v, want %v", tt.how, tt.q, have, tt.want)
			}
		})
	}
}

// TestQuantileNearestTakesTheEvenIndex is the tie rule on its own, checked at a
// position where the even index is the higher of the two.
func TestQuantileNearestTakesTheEvenIndex(t *testing.T) {
	g := kernel.OneGroup(5)
	c := col(t, dtype.Float64, []any{0.0, 1.0, 2.0, 3.0, 4.0})

	// Position 2.5, between index 2 and index 3, so the even one is 2.
	got, err := kernel.Quantile(c, g, 0.625, kernel.Nearest)
	if err != nil {
		t.Fatalf("Quantile: %v", err)
	}
	if have := got.Value[float64](0); have != 2 {
		t.Errorf("the answer is %v, want 2 from the even index", have)
	}
}

func TestQuantileNaN(t *testing.T) {
	g := kernel.OneGroup(3)
	c := col(t, dtype.Float64, []any{1.0, math.NaN(), 3.0})

	got, err := kernel.Quantile(c, g, 1, kernel.Lower)
	if err != nil {
		t.Fatalf("Quantile: %v", err)
	}
	if !math.IsNaN(got.Value[float64](0)) {
		t.Errorf("the largest is %v, want NaN, which sorts after every number",
			got.Value[float64](0))
	}
}

func TestQuantileErrors(t *testing.T) {
	g := kernel.OneGroup(2)
	c := col(t, dtype.Float64, []any{1.0, 2.0})

	tests := []struct {
		name string
		q    float64
		how  kernel.Interpolation
		want string
	}{
		{"below zero", -0.1, kernel.Linear, "between 0 and 1"},
		{"above one", 1.5, kernel.Linear, "between 0 and 1"},
		{"not a number", math.NaN(), kernel.Linear, "between 0 and 1"},
		{"not an interpolation", 0.5, kernel.Interpolation(9), "Interpolation(9)"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := kernel.Quantile(c, g, tt.q, tt.how)
			if err == nil {
				t.Fatal("it succeeded")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("the message is %q, want it to mention %q", err.Error(), tt.want)
			}
		})
	}

	if _, err := kernel.Median(col(t, dtype.String, []any{"a", "b"}), g); err == nil {
		t.Fatal("the median of a string column succeeded")
	} else if !strings.Contains(err.Error(), "median") {
		t.Errorf("the message is %q, want it to name the operation the caller wrote", err.Error())
	}
}

func TestQuantileEveryType(t *testing.T) {
	g := kernel.OneGroup(2)

	tests := []struct {
		dt   dtype.DataType
		vals []any
	}{
		{dtype.Bool, []any{false, true}},
		{dtype.Int8, []any{int8(1), int8(3)}},
		{dtype.Int16, []any{int16(1), int16(3)}},
		{dtype.Int32, []any{int32(1), int32(3)}},
		{dtype.Int64, []any{int64(1), int64(3)}},
		{dtype.Uint8, []any{uint8(1), uint8(3)}},
		{dtype.Uint16, []any{uint16(1), uint16(3)}},
		{dtype.Uint32, []any{uint32(1), uint32(3)}},
		{dtype.Uint64, []any{uint64(1), uint64(3)}},
		{dtype.Float32, []any{float32(1), float32(3)}},
		{dtype.Float64, []any{1.0, 3.0}},
	}

	for _, tt := range tests {
		t.Run(tt.dt.String(), func(t *testing.T) {
			want := 2.0
			if tt.dt == dtype.Bool {
				want = 0.5
			}
			got, err := kernel.Median(col(t, tt.dt, tt.vals), g)
			if err != nil {
				t.Fatalf("Median: %v", err)
			}
			if have := got.Value[float64](0); have != want {
				t.Errorf("the median is %v, want %v", have, want)
			}
		})
	}
}
