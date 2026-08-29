package arrowgo

import (
	"strings"
	"testing"

	"github.com/apache/arrow-go/v18/arrow"
	arrowarray "github.com/apache/arrow-go/v18/arrow/array"

	"github.com/tamnd/kuma/array"
	"github.com/tamnd/kuma/dtype"
)

// tradeSchema is the schema the table tests are written around: one string
// column, one number and one timestamp, which is enough to have a shared
// buffer, a shared block and a parameterised type in the same table.
var tradeSchema = arrow.NewSchema([]arrow.Field{
	{Name: "symbol", Type: arrow.BinaryTypes.StringView, Nullable: true},
	{Name: "price", Type: arrow.PrimitiveTypes.Float64, Nullable: true},
	{Name: "ts", Type: &arrow.TimestampType{Unit: arrow.Microsecond, TimeZone: "UTC"}, Nullable: true},
}, &arrow.Metadata{})

func TestRecordBatchRoundTrip(t *testing.T) {
	in := tradeBatch(t, 0)
	defer in.Release()

	mid, err := ImportRecordBatch(in)
	if err != nil {
		t.Fatalf("ImportRecordBatch: %v", err)
	}
	if mid.NumCols() != 3 || mid.NumRows() != 3 {
		t.Fatalf("came in as %d columns of %d rows, want 3 of 3", mid.NumCols(), mid.NumRows())
	}
	for i, c := range mid.Columns {
		if c.NumChunks() != 1 {
			t.Errorf("column %d came in as %d chunks, and a record batch is one per column", i, c.NumChunks())
		}
	}
	if got := mid.Schema.Fields[0].Name; got != "symbol" {
		t.Errorf("the first column is %q, want symbol", got)
	}

	out, err := ExportRecordBatch(mid)
	if err != nil {
		t.Fatalf("ExportRecordBatch: %v", err)
	}
	defer out.Release()

	if !out.Schema().Equal(in.Schema()) {
		t.Errorf("the schema came back as\n%s\nand went out as\n%s", out.Schema(), in.Schema())
	}
	if out.NumRows() != in.NumRows() {
		t.Fatalf("came back as %d rows, want %d", out.NumRows(), in.NumRows())
	}
	for i := range int(out.NumCols()) {
		if got, want := out.Column(i).String(), in.Column(i).String(); got != want {
			t.Errorf("column %d came back as %s and went out as %s", i, got, want)
		}
	}
}

func TestTableRoundTrip(t *testing.T) {
	first, second := tradeBatch(t, 0), tradeBatch(t, 3)
	defer first.Release()
	defer second.Release()

	in := arrowarray.NewTableFromRecords(tradeSchema, []arrow.RecordBatch{first, second})
	defer in.Release()

	mid, err := ImportTable(in)
	if err != nil {
		t.Fatalf("ImportTable: %v", err)
	}
	if mid.NumRows() != 6 {
		t.Fatalf("came in as %d rows, want 6", mid.NumRows())
	}
	for i, c := range mid.Columns {
		if c.NumChunks() != 2 {
			t.Errorf("column %d came in as %d chunks, want the 2 it was written in", i, c.NumChunks())
		}
	}
	if got := string(mid.Columns[0].Bytes(3)); got != "MSFT" {
		t.Errorf("row 3 of the first column is %q, want MSFT, so the second chunk "+
			"is not where it should be", got)
	}

	out, err := ExportTable(mid)
	if err != nil {
		t.Fatalf("ExportTable: %v", err)
	}
	defer out.Release()

	if out.NumRows() != 6 || out.NumCols() != 3 {
		t.Fatalf("came back as %d columns of %d rows, want 3 of 6", out.NumCols(), out.NumRows())
	}
	if n := len(out.Column(0).Data().Chunks()); n != 2 {
		t.Errorf("the first column came back as %d chunks, want 2", n)
	}
	if !out.Schema().Equal(in.Schema()) {
		t.Errorf("the schema came back as\n%s\nand went out as\n%s", out.Schema(), in.Schema())
	}
}

func TestChunkedRoundTrip(t *testing.T) {
	first := fromJSON(t, arrow.PrimitiveTypes.Int64, `[1, 2, null]`)
	defer first.Release()
	second := fromJSON(t, arrow.PrimitiveTypes.Int64, `[4, 5]`)
	defer second.Release()

	in := arrow.NewChunked(arrow.PrimitiveTypes.Int64, []arrow.Array{first, second})
	defer in.Release()

	mid, err := ImportChunked(in)
	if err != nil {
		t.Fatalf("ImportChunked: %v", err)
	}
	if mid.Len() != 5 || mid.NullCount() != 1 {
		t.Fatalf("came in as %d values with %d nulls, want 5 and 1", mid.Len(), mid.NullCount())
	}
	if got := mid.Value[int64](4); got != 5 {
		t.Errorf("value 4 is %d, want 5", got)
	}

	out, err := ExportChunked(mid)
	if err != nil {
		t.Fatalf("ExportChunked: %v", err)
	}
	defer out.Release()

	if out.Len() != 5 || len(out.Chunks()) != 2 {
		t.Fatalf("came back as %d values in %d chunks, want 5 in 2", out.Len(), len(out.Chunks()))
	}
	if got, want := out.Chunk(0).String(), first.String(); got != want {
		t.Errorf("the first chunk came back as %s and went out as %s", got, want)
	}
}

// TestExportRecordBatchWantsOneChunk is the difference between the two export
// functions, and the error says which one to reach for instead.
func TestExportRecordBatchWantsOneChunk(t *testing.T) {
	first, second := tradeBatch(t, 0), tradeBatch(t, 3)
	defer first.Release()
	defer second.Release()

	in := arrowarray.NewTableFromRecords(tradeSchema, []arrow.RecordBatch{first, second})
	defer in.Release()

	mid, err := ImportTable(in)
	if err != nil {
		t.Fatalf("ImportTable: %v", err)
	}

	_, err = ExportRecordBatch(mid)
	if err == nil {
		t.Fatal("ExportRecordBatch said nothing about a column of two chunks")
	}
	if !strings.Contains(err.Error(), "ExportTable") {
		t.Errorf("ExportRecordBatch says %q, and it should say what to use instead", err)
	}
	if !strings.Contains(err.Error(), "symbol") {
		t.Errorf("ExportRecordBatch says %q, and it should name the column", err)
	}
}

// TestExportColumnWithNoChunks is the table of no rows, where there is no chunk
// to hand over and arrow-go still wants an array.
func TestExportColumnWithNoChunks(t *testing.T) {
	symbol, err := array.NewChunked(dtype.String)
	if err != nil {
		t.Fatal(err)
	}
	nothing, err := array.NewChunked(dtype.Null)
	if err != nil {
		t.Fatal(err)
	}

	in := &array.Table{
		Schema: dtype.Schema{Fields: []dtype.Field{
			{Name: "symbol", Type: dtype.String, Nullable: true},
			{Name: "nothing", Type: dtype.Null, Nullable: true},
		}},
		Columns: []*array.Chunked{symbol, nothing},
	}

	out, err := ExportRecordBatch(in)
	if err != nil {
		t.Fatalf("ExportRecordBatch: %v", err)
	}
	defer out.Release()

	if out.NumRows() != 0 || out.NumCols() != 2 {
		t.Fatalf("came back as %d columns of %d rows, want 2 of 0", out.NumCols(), out.NumRows())
	}
	for i := range int(out.NumCols()) {
		if n := out.Column(i).Len(); n != 0 {
			t.Errorf("column %d came back %d values long, want 0", i, n)
		}
	}

	table, err := ExportTable(in)
	if err != nil {
		t.Fatalf("ExportTable: %v", err)
	}
	defer table.Release()

	if table.NumRows() != 0 || table.NumCols() != 2 {
		t.Errorf("came back as %d columns of %d rows, want 2 of 0", table.NumCols(), table.NumRows())
	}
}

// TestImportRecordBatchSharesBuffers is the whole table version of the claim
// the package rests on, since a table that copies one column has copied the
// file it was read out of.
func TestImportRecordBatchSharesBuffers(t *testing.T) {
	in := tradeBatch(t, 0)
	defer in.Release()

	got, err := ImportRecordBatch(in)
	if err != nil {
		t.Fatalf("ImportRecordBatch: %v", err)
	}

	price := got.Columns[1].Chunk(0)
	if a, b := &price.Buffer().Bytes()[0], &in.Column(1).Data().Buffers()[1].Bytes()[0]; a != b {
		t.Errorf("the prices are at %p on the kuma side and %p on the arrow side, "+
			"which means the column was copied", a, b)
	}
}

func TestImportTableErrors(t *testing.T) {
	if _, err := ImportRecordBatch(nil); err == nil {
		t.Error("ImportRecordBatch(nil) said nothing")
	}
	if _, err := ImportTable(nil); err == nil {
		t.Error("ImportTable(nil) said nothing")
	}
	if _, err := ImportChunked(nil); err == nil {
		t.Error("ImportChunked(nil) said nothing")
	}

	xs := fromJSON(t, arrow.ListOf(arrow.PrimitiveTypes.Int64), `[[1, 2], null]`)
	defer xs.Release()

	schema := arrow.NewSchema([]arrow.Field{{Name: "xs", Type: xs.DataType(), Nullable: true}}, nil)
	rec := arrowarray.NewRecordBatch(schema, []arrow.Array{xs}, 2)
	defer rec.Release()

	if _, err := ImportRecordBatch(rec); err == nil {
		t.Error("ImportRecordBatch said nothing about a list column")
	}
}

func TestExportTableErrors(t *testing.T) {
	if _, err := ExportRecordBatch(nil); err == nil {
		t.Error("ExportRecordBatch(nil) said nothing")
	}
	if _, err := ExportTable(nil); err == nil {
		t.Error("ExportTable(nil) said nothing")
	}
	if _, err := ExportChunked(nil); err == nil {
		t.Error("ExportChunked(nil) said nothing")
	}

	cases := []struct {
		name string
		in   *array.Table
		want string
	}{
		{
			name: "more fields than columns",
			in: &array.Table{
				Schema: dtype.Schema{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
			},
			want: "a schema of 1 fields",
		},
		{
			name: "a column that is not there",
			in: &array.Table{
				Schema:  dtype.Schema{Fields: []dtype.Field{{Name: "a", Type: dtype.Int64}}},
				Columns: []*array.Chunked{nil},
			},
			want: `the column "a" is nil`,
		},
		{
			name: "a column of a type that does not cross",
			in: &array.Table{
				Schema:  dtype.Schema{Fields: []dtype.Field{{Name: "xs", Type: dtype.List{Elem: dtype.Int64}}}},
				Columns: []*array.Chunked{nil},
			},
			want: "does not cross",
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := ExportTable(c.in)
			if err == nil {
				t.Fatal("ExportTable said nothing")
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("ExportTable says %q, and it should mention %q", err, c.want)
			}
		})
	}
}

// tradeBatch builds three rows against [tradeSchema], starting at the row given
// so that two calls make a table of six rows that can be told apart.
func tradeBatch(t *testing.T, from int) arrow.RecordBatch {
	t.Helper()

	rows := []string{
		`["AAPL", null, "GOOG"]`,
		`[1.5, 2.5, null]`,
		`["2026-08-29T08:00:00Z", "2026-08-29T08:00:01Z", "2026-08-29T08:00:02Z"]`,
		`["MSFT", "NVDA", null]`,
		`[null, 4.5, 5.5]`,
		`["2026-08-29T09:00:00Z", "2026-08-29T09:00:01Z", "2026-08-29T09:00:02Z"]`,
	}

	cols := make([]arrow.Array, tradeSchema.NumFields())
	for i := range cols {
		cols[i] = fromJSON(t, tradeSchema.Field(i).Type, rows[from+i])
		defer cols[i].Release()
	}
	return arrowarray.NewRecordBatch(tradeSchema, cols, 3)
}
