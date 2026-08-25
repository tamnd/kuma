package kuma_test

import (
	"errors"
	"testing"
	"time"

	"github.com/tamnd/kuma"
	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

func TestBind(t *testing.T) {
	f := trades(t)

	typed, err := kuma.Bind[Trade](f)
	if err != nil {
		t.Fatalf("Bind: %v", err)
	}
	if typed.NumRows() != f.NumRows() || typed.NumCols() != f.NumCols() {
		t.Fatalf("%s, want the same shape as %s", typed, f)
	}

	// Nothing is copied, so the columns are the ones that went in.
	for i, c := range typed.Columns() {
		if c.Data() != f.ColumnAt(i).Data() {
			t.Errorf("column %d was rebuilt, want the one that went in", i)
		}
	}
}

// TestBindExtraColumns is the rule that makes Bind usable on a file that holds
// more than the program cares about.
func TestBindExtraColumns(t *testing.T) {
	type Sym struct {
		Symbol string `kuma:"symbol"`
	}

	typed, err := kuma.Bind[Sym](trades(t))
	if err != nil {
		t.Fatalf("Bind: %v", err)
	}
	if typed.NumCols() != 3 {
		t.Errorf("%s, want the price and quantity columns to still be there", typed)
	}
	if _, err := kuma.NewF64Col[Sym]("price").Series(typed); err != nil {
		t.Errorf("the price column is not readable after Bind: %v", err)
	}
}

// TestBindSkipsFields is the rule that a field can say it is not a column, by
// the tag or by being unexported, and that Bind then does not look for one.
func TestBindSkipsFields(t *testing.T) {
	type Row struct {
		Symbol string `kuma:"symbol"`
		Note   string `kuma:"-"`
		cache  []byte
	}

	typed, err := kuma.Bind[Row](trades(t))
	if err != nil {
		t.Fatalf("Bind: %v", err)
	}
	if typed.NumCols() != 3 {
		t.Errorf("%s, want the three columns that went in", typed)
	}

	// The unexported field is read here so that the compiler and the linters
	// agree it is part of the struct rather than a leftover.
	var r Row
	if len(r.cache) != 0 {
		t.Error("a zero Row has a cache in it")
	}
}

// TestBindTagWins is the tag deciding the column, where the field name would
// have named a different one.
func TestBindTagWins(t *testing.T) {
	type Row struct {
		Ticker string `kuma:"symbol"`
	}

	if _, err := kuma.Bind[Row](trades(t)); err != nil {
		t.Fatalf("Bind: %v", err)
	}

	type Untagged struct {
		Ticker string
	}
	if _, err := kuma.Bind[Untagged](trades(t)); !errors.Is(err, kuma.ErrNoColumn) {
		t.Fatalf("Bind on the untagged field gave %v, want an ErrNoColumn about ticker", err)
	}
}

// TestBindSnakeNames is the name a field with no tag gets, which is the rule the
// rest of the Go world uses for JSON.
func TestBindSnakeNames(t *testing.T) {
	f, err := kuma.NewFrame(
		kuma.NewSeries("order_id", int64(1), 2).Column(),
		kuma.NewSeries("http_code", int64(200), 404).Column(),
		kuma.NewSeries("ts", int64(0), 1).Column(),
		kuma.NewSeries("id2", int64(7), 8).Column(),
	)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}

	type Row struct {
		OrderID  int64
		HTTPCode int64
		TS       int64
		ID2      int64
	}

	if _, err := kuma.Bind[Row](f); err != nil {
		t.Fatalf("Bind: %v", err)
	}
}

func TestBindErrors(t *testing.T) {
	f := trades(t)

	t.Run("a column that is not there", func(t *testing.T) {
		type Row struct {
			Volume int64 `kuma:"volume"`
		}
		_, err := kuma.Bind[Row](f)
		if !errors.Is(err, kuma.ErrNoColumn) {
			t.Fatalf("Bind gave %v, want an ErrNoColumn", err)
		}
	})

	t.Run("a field of the wrong type", func(t *testing.T) {
		type Row struct {
			Price int64 `kuma:"price"`
		}
		_, err := kuma.Bind[Row](f)
		if !errors.Is(err, kuma.ErrWrongType) {
			t.Fatalf("Bind gave %v, want an ErrWrongType", err)
		}
	})

	t.Run("a field of a type no column holds", func(t *testing.T) {
		type Row struct {
			Price []float64 `kuma:"price"`
		}
		_, err := kuma.Bind[Row](f)
		if !errors.Is(err, kuma.ErrWrongType) {
			t.Fatalf("Bind gave %v, want an ErrWrongType", err)
		}
	})

	t.Run("a schema that is not a struct", func(t *testing.T) {
		_, err := kuma.Bind[int](f)
		if !errors.Is(err, kuma.ErrWrongType) {
			t.Fatalf("Bind gave %v, want an ErrWrongType", err)
		}
	})
}

// TestBindTypes walks the field types a column can be read into, which is the
// same table [kuma.CanRead] answers and has to stay in step with it.
func TestBindTypes(t *testing.T) {
	tests := []struct {
		name string
		dt   dtype.DataType
		bind func(*kuma.Frame[kuma.Dynamic]) error
	}{
		{"bool", dtype.Bool, bindAs[struct {
			V bool `kuma:"v"`
		}]},
		{"int8", dtype.Int8, bindAs[struct {
			V int8 `kuma:"v"`
		}]},
		{"int16", dtype.Int16, bindAs[struct {
			V int16 `kuma:"v"`
		}]},
		{"int32", dtype.Int32, bindAs[struct {
			V int32 `kuma:"v"`
		}]},
		{"int64", dtype.Int64, bindAs[struct {
			V int64 `kuma:"v"`
		}]},
		{"uint8", dtype.Uint8, bindAs[struct {
			V uint8 `kuma:"v"`
		}]},
		{"uint16", dtype.Uint16, bindAs[struct {
			V uint16 `kuma:"v"`
		}]},
		{"uint32", dtype.Uint32, bindAs[struct {
			V uint32 `kuma:"v"`
		}]},
		{"uint64", dtype.Uint64, bindAs[struct {
			V uint64 `kuma:"v"`
		}]},
		{"float32", dtype.Float32, bindAs[struct {
			V float32 `kuma:"v"`
		}]},
		{"float64", dtype.Float64, bindAs[struct {
			V float64 `kuma:"v"`
		}]},
		{"string", dtype.String, bindAs[struct {
			V string `kuma:"v"`
		}]},
		{"binary as a string", dtype.Binary, bindAs[struct {
			V string `kuma:"v"`
		}]},
		{"date32 as an int32", dtype.Date32, bindAs[struct {
			V int32 `kuma:"v"`
		}]},
		{"date64 as an int64", dtype.Date64, bindAs[struct {
			V int64 `kuma:"v"`
		}]},
		{"time32 as an int32", dtype.Time32{Unit: dtype.Second}, bindAs[struct {
			V int32 `kuma:"v"`
		}]},
		{"time64 as an int64", dtype.Time64{Unit: dtype.Nanosecond}, bindAs[struct {
			V int64 `kuma:"v"`
		}]},
		{"a timestamp as a time", dtype.Timestamp{Unit: dtype.Second, Zone: "UTC"}, bindAs[struct {
			V time.Time `kuma:"v"`
		}]},
		{"a timestamp as an int64", dtype.Timestamp{Unit: dtype.Second, Zone: "UTC"}, bindAs[struct {
			V int64 `kuma:"v"`
		}]},
		{"a duration as an int64", dtype.Duration{Unit: dtype.Second}, bindAs[struct {
			V int64 `kuma:"v"`
		}]},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.bind(emptyOf(t, tt.dt)); err != nil {
				t.Errorf("a %s column: %v", tt.dt, err)
			}
		})
	}
}

// TestBindWrongTypes is the other half of the table, where the field and the
// column are both types kuma knows and are not each other's.
func TestBindWrongTypes(t *testing.T) {
	tests := []struct {
		name string
		dt   dtype.DataType
		bind func(*kuma.Frame[kuma.Dynamic]) error
	}{
		{"an int64 field on a bool column", dtype.Bool, bindAs[struct {
			V int64 `kuma:"v"`
		}]},
		{"an int8 field on an int16 column", dtype.Int16, bindAs[struct {
			V int8 `kuma:"v"`
		}]},
		{"an int16 field on an int8 column", dtype.Int8, bindAs[struct {
			V int16 `kuma:"v"`
		}]},
		{"an int32 field on an int64 column", dtype.Int64, bindAs[struct {
			V int32 `kuma:"v"`
		}]},
		{"an int64 field on a string column", dtype.String, bindAs[struct {
			V int64 `kuma:"v"`
		}]},
		{"a uint8 field on an int8 column", dtype.Int8, bindAs[struct {
			V uint8 `kuma:"v"`
		}]},
		{"a uint16 field on a uint8 column", dtype.Uint8, bindAs[struct {
			V uint16 `kuma:"v"`
		}]},
		{"a uint32 field on a uint64 column", dtype.Uint64, bindAs[struct {
			V uint32 `kuma:"v"`
		}]},
		{"a uint64 field on a uint32 column", dtype.Uint32, bindAs[struct {
			V uint64 `kuma:"v"`
		}]},
		{"a float32 field on a float64 column", dtype.Float64, bindAs[struct {
			V float32 `kuma:"v"`
		}]},
		{"a float64 field on a float32 column", dtype.Float32, bindAs[struct {
			V float64 `kuma:"v"`
		}]},
		{"a string field on a float64 column", dtype.Float64, bindAs[struct {
			V string `kuma:"v"`
		}]},
		{"a bool field on an int8 column", dtype.Int8, bindAs[struct {
			V bool `kuma:"v"`
		}]},
		{"a time field on an int64 column", dtype.Int64, bindAs[struct {
			V time.Time `kuma:"v"`
		}]},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.bind(emptyOf(t, tt.dt)); !errors.Is(err, kuma.ErrWrongType) {
				t.Errorf("a %s column gave %v, want an ErrWrongType", tt.dt, err)
			}
		})
	}
}

// bindAs is one row of the tables above, which needs a function because the
// schema is a type parameter and a table holds values.
func bindAs[S any](f *kuma.Frame[kuma.Dynamic]) error {
	_, err := kuma.Bind[S](f)
	return err
}

// emptyOf returns a frame whose one column is called v and holds nothing, since
// what Bind reads is the type rather than the values.
func emptyOf(t *testing.T, dt dtype.DataType) *kuma.Frame[kuma.Dynamic] {
	t.Helper()

	b, err := array.NewBuilder(dt)
	if err != nil {
		t.Fatalf("NewBuilder: %v", err)
	}
	data, err := array.NewChunked(dt, b.Finish())
	if err != nil {
		t.Fatalf("NewChunked: %v", err)
	}
	c, err := kuma.NewColumn("v", data)
	if err != nil {
		t.Fatalf("NewColumn: %v", err)
	}
	f, err := kuma.NewFrame(c)
	if err != nil {
		t.Fatalf("NewFrame: %v", err)
	}
	return f
}
