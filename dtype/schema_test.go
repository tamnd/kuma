package dtype_test

import (
	"strings"
	"testing"

	"github.com/tamnd/kuma/dtype"
)

func sample() dtype.Schema {
	return dtype.Schema{
		Fields: []dtype.Field{
			{Name: "id", Type: dtype.Int64},
			{Name: "name", Type: dtype.String, Nullable: true},
			{Name: "price", Type: dtype.Decimal128{Precision: 18, Scale: 2}, Nullable: true},
		},
		Metadata: dtype.Metadata{{Key: "source", Value: "orders.parquet"}},
	}
}

func TestSchemaLookup(t *testing.T) {
	s := sample()

	if got := s.Len(); got != 3 {
		t.Errorf("Len() = %d, want 3", got)
	}
	if got := s.Index("name"); got != 1 {
		t.Errorf(`Index("name") = %d, want 1`, got)
	}
	if got := s.Index("missing"); got != -1 {
		t.Errorf(`Index("missing") = %d, want -1`, got)
	}

	f, ok := s.Field("price")
	if !ok {
		t.Fatal(`Field("price") not found`)
	}
	if !dtype.Equal(f.Type, dtype.Decimal128{Precision: 18, Scale: 2}) {
		t.Errorf(`Field("price").Type = %s`, f.Type)
	}
	if _, ok := s.Field("missing"); ok {
		t.Error(`Field("missing") found something`)
	}

	want := []string{"id", "name", "price"}
	got := s.Names()
	if len(got) != len(want) {
		t.Fatalf("Names() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Names() = %v, want %v", got, want)
			break
		}
	}
}

// TestSchemaIndexDuplicate pins the documented behavior for a schema with two
// fields of the same name, which is a thing a CSV reader has to be able to
// describe before it renames anything.
func TestSchemaIndexDuplicate(t *testing.T) {
	s := dtype.Schema{Fields: []dtype.Field{
		{Name: "id", Type: dtype.Int64},
		{Name: "id", Type: dtype.String},
	}}

	if got := s.Index("id"); got != 0 {
		t.Errorf(`Index("id") = %d, want 0`, got)
	}
	f, _ := s.Field("id")
	if !dtype.Equal(f.Type, dtype.Int64) {
		t.Errorf(`Field("id").Type = %s, want int64`, f.Type)
	}
}

func TestSchemaSelect(t *testing.T) {
	s := sample()

	got, err := s.Select("price", "id")
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	want := dtype.Schema{
		Fields: []dtype.Field{
			{Name: "price", Type: dtype.Decimal128{Precision: 18, Scale: 2}, Nullable: true},
			{Name: "id", Type: dtype.Int64},
		},
		Metadata: s.Metadata,
	}
	if !got.Equal(want) {
		t.Errorf("Select = %s, want %s", got, want)
	}

	if got, err := s.Select(); err != nil || got.Len() != 0 {
		t.Errorf("Select() = %s, %v, want an empty schema and no error", got, err)
	}
}

func TestSchemaSelectMissing(t *testing.T) {
	s := sample()

	_, err := s.Select("id", "Name")
	if err == nil {
		t.Fatal("Select of a missing field returned no error")
	}
	// The message has to name the column that is wrong and the ones that are
	// there, because the cause is almost always a typo or the wrong case.
	for _, want := range []string{`"Name"`, "id", "name", "price"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %s", err, want)
		}
	}
}

func TestSchemaEqual(t *testing.T) {
	tests := []struct {
		name string
		a, b dtype.Schema
		want bool
	}{
		{"identical", sample(), sample(), true},
		{"both empty", dtype.Schema{}, dtype.Schema{}, true},
		{
			"different lengths",
			dtype.Schema{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Schema{},
			false,
		},
		{
			"same fields in a different order",
			dtype.Schema{Fields: []dtype.Field{
				{Name: "a", Type: dtype.Int64},
				{Name: "b", Type: dtype.String},
			}},
			dtype.Schema{Fields: []dtype.Field{
				{Name: "b", Type: dtype.String},
				{Name: "a", Type: dtype.Int64},
			}},
			false,
		},
		{
			"different field types",
			dtype.Schema{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Schema{Fields: []dtype.Field{{Name: "a", Type: dtype.Int32}}},
			false,
		},
		{
			"different nullability",
			dtype.Schema{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			dtype.Schema{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64, Nullable: true}}},
			false,
		},
		{
			"different metadata",
			dtype.Schema{Metadata: dtype.Metadata{{Key: "k", Value: "1"}}},
			dtype.Schema{Metadata: dtype.Metadata{{Key: "k", Value: "2"}}},
			false,
		},
		{
			"nil metadata against empty metadata",
			dtype.Schema{},
			dtype.Schema{Metadata: dtype.Metadata{}},
			true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.a.Equal(tt.b); got != tt.want {
				t.Errorf("Equal = %v, want %v", got, tt.want)
			}
			if got := tt.b.Equal(tt.a); got != tt.want {
				t.Errorf("reversed Equal = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSchemaString(t *testing.T) {
	tests := []struct {
		schema dtype.Schema
		want   string
	}{
		{dtype.Schema{}, "schema<>"},
		{
			dtype.Schema{Fields: []dtype.Field{
				{Name: "id", Type: dtype.Int64},
				{Name: "name", Type: dtype.String, Nullable: true},
			}},
			"schema<id: int64 not null, name: string>",
		},
	}

	for _, tt := range tests {
		if got := tt.schema.String(); got != tt.want {
			t.Errorf("String() = %q, want %q", got, tt.want)
		}
	}
}

// TestSchemaClone is the test that catches a clone that shares its slice with
// the original, which shows up much later as one frame's schema changing when
// another is renamed.
func TestSchemaClone(t *testing.T) {
	orig := sample()
	clone := orig.Clone()

	if !clone.Equal(orig) {
		t.Fatalf("Clone() = %s, want %s", clone, orig)
	}

	clone.Fields[0].Name = "changed"
	clone.Fields[1].Metadata = append(clone.Fields[1].Metadata, dtype.KeyValue{Key: "k", Value: "v"})
	clone.Metadata[0].Value = "elsewhere.parquet"

	if orig.Fields[0].Name != "id" {
		t.Errorf("the original field name changed to %q", orig.Fields[0].Name)
	}
	if len(orig.Fields[1].Metadata) != 0 {
		t.Errorf("the original field metadata changed to %v", orig.Fields[1].Metadata)
	}
	if v, _ := orig.Metadata.Get("source"); v != "orders.parquet" {
		t.Errorf("the original schema metadata changed to %q", v)
	}
}

func TestSchemaCloneEmpty(t *testing.T) {
	clone := dtype.Schema{}.Clone()
	if clone.Fields != nil || clone.Metadata != nil {
		t.Errorf("Clone() of an empty schema = %#v, want the zero value", clone)
	}
}

func TestMetadata(t *testing.T) {
	m := dtype.Metadata{
		{Key: "unit", Value: "pence"},
		{Key: "unit", Value: "pounds"},
	}

	// Duplicate keys are legal in both Arrow and Parquet, and Get takes the
	// first, so the pair order is worth keeping.
	if v, ok := m.Get("unit"); !ok || v != "pence" {
		t.Errorf(`Get("unit") = %q, %v, want "pence", true`, v, ok)
	}
	if v, ok := m.Get("missing"); ok || v != "" {
		t.Errorf(`Get("missing") = %q, %v, want "", false`, v, ok)
	}

	if !m.Equal(m.Clone()) {
		t.Error("a clone does not equal the original")
	}
	if m.Equal(m[:1]) {
		t.Error("metadata of different lengths compared equal")
	}
	if m.Equal(dtype.Metadata{{Key: "unit", Value: "pounds"}, {Key: "unit", Value: "pence"}}) {
		t.Error("metadata compared equal to itself reordered")
	}
	if got := dtype.Metadata(nil).Clone(); got != nil {
		t.Errorf("Clone() of nil metadata = %v, want nil", got)
	}
}

func TestFieldString(t *testing.T) {
	tests := []struct {
		field dtype.Field
		want  string
	}{
		{dtype.Field{Name: "id", Type: dtype.Int64}, "id: int64 not null"},
		{dtype.Field{Name: "name", Type: dtype.String, Nullable: true}, "name: string"},
		{dtype.Field{Name: "broken"}, "broken: <nil> not null"},
	}

	for _, tt := range tests {
		if got := tt.field.String(); got != tt.want {
			t.Errorf("String() = %q, want %q", got, tt.want)
		}
	}
}

func TestFieldEqual(t *testing.T) {
	base := dtype.Field{
		Name:     "price",
		Type:     dtype.Float64,
		Nullable: true,
		Metadata: dtype.Metadata{{Key: "unit", Value: "pence"}},
	}

	same := base
	same.Metadata = base.Metadata.Clone()
	if !base.Equal(same) {
		t.Error("a field does not equal its copy")
	}

	renamed := base
	renamed.Name = "cost"
	if base.Equal(renamed) {
		t.Error("fields with different names compared equal")
	}

	retyped := base
	retyped.Type = dtype.Float32
	if base.Equal(retyped) {
		t.Error("fields with different types compared equal")
	}

	required := base
	required.Nullable = false
	if base.Equal(required) {
		t.Error("fields with different nullability compared equal")
	}

	tagged := base
	tagged.Metadata = dtype.Metadata{{Key: "unit", Value: "pounds"}}
	if base.Equal(tagged) {
		t.Error("fields with different metadata compared equal")
	}
}
