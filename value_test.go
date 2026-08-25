package kuma_test

import (
	"testing"
	"time"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/dtype"
)

func TestDTypeOf(t *testing.T) {
	tests := []struct {
		name string
		got  dtype.DataType
		want dtype.DataType
	}{
		{"bool", kuma.DTypeOf[bool](), dtype.Bool},
		{"int8", kuma.DTypeOf[int8](), dtype.Int8},
		{"int16", kuma.DTypeOf[int16](), dtype.Int16},
		{"int32", kuma.DTypeOf[int32](), dtype.Int32},
		{"int64", kuma.DTypeOf[int64](), dtype.Int64},
		{"uint8", kuma.DTypeOf[uint8](), dtype.Uint8},
		{"uint16", kuma.DTypeOf[uint16](), dtype.Uint16},
		{"uint32", kuma.DTypeOf[uint32](), dtype.Uint32},
		{"uint64", kuma.DTypeOf[uint64](), dtype.Uint64},
		{"float32", kuma.DTypeOf[float32](), dtype.Float32},
		{"float64", kuma.DTypeOf[float64](), dtype.Float64},
		{"string", kuma.DTypeOf[string](), dtype.String},
		{"time.Time", kuma.DTypeOf[time.Time](), dtype.Timestamp{Unit: dtype.Nanosecond, Zone: "UTC"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !dtype.Equal(tt.got, tt.want) {
				t.Errorf("DTypeOf = %s, want %s", tt.got, tt.want)
			}
		})
	}
}

// TestCanRead is the table that says which column types a Go type is allowed to
// read. The interesting rows are the ones where the answer is yes for a type
// that is not the one DTypeOf hands out, since those are the reinterpretations
// that keep a date column out of a copy on its way into an integer kernel.
func TestCanRead(t *testing.T) {
	tests := []struct {
		name string
		can  func(dtype.DataType) bool
		dt   dtype.DataType
		want bool
	}{
		{"int64 reads int64", kuma.CanRead[int64], dtype.Int64, true},
		{"int64 reads a timestamp", kuma.CanRead[int64], dtype.Timestamp{Unit: dtype.Microsecond}, true},
		{"int64 reads a duration", kuma.CanRead[int64], dtype.Duration{Unit: dtype.Second}, true},
		{"int64 reads date64", kuma.CanRead[int64], dtype.Date64, true},
		{"int64 reads time64", kuma.CanRead[int64], dtype.Time64{Unit: dtype.Nanosecond}, true},
		{"int64 does not read a float64", kuma.CanRead[int64], dtype.Float64, false},
		{"int64 does not read a uint64", kuma.CanRead[int64], dtype.Uint64, false},
		{"int64 does not read an int32", kuma.CanRead[int64], dtype.Int32, false},
		{"int32 reads date32", kuma.CanRead[int32], dtype.Date32, true},
		{"int32 reads time32", kuma.CanRead[int32], dtype.Time32{Unit: dtype.Second}, true},
		{"int32 does not read date64", kuma.CanRead[int32], dtype.Date64, false},
		{"string reads a string", kuma.CanRead[string], dtype.String, true},
		{"string reads binary", kuma.CanRead[string], dtype.Binary, true},
		{"string does not read an int64", kuma.CanRead[string], dtype.Int64, false},
		{"bool reads bool", kuma.CanRead[bool], dtype.Bool, true},
		{"bool does not read a uint8", kuma.CanRead[bool], dtype.Uint8, false},
		{"float64 reads a float64", kuma.CanRead[float64], dtype.Float64, true},
		{"float64 does not read a float32", kuma.CanRead[float64], dtype.Float32, false},
		{"a time reads any timestamp", kuma.CanRead[time.Time], dtype.Timestamp{Unit: dtype.Second, Zone: "Asia/Tokyo"}, true},
		{"a time does not read an int64", kuma.CanRead[time.Time], dtype.Int64, false},
		{"a time does not read date32", kuma.CanRead[time.Time], dtype.Date32, false},
		{"nothing reads a nil type", kuma.CanRead[int64], nil, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.can(tt.dt); got != tt.want {
				t.Errorf("CanRead(%v) = %v, want %v", tt.dt, got, tt.want)
			}
		})
	}
}
