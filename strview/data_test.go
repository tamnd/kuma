package strview_test

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"

	"github.com/tamnd/kuma/buffer"
	"github.com/tamnd/kuma/strview"
)

// build returns a column of the given values, which is what most of these tests
// start from.
func build(values ...string) *strview.Data {
	var b strview.Builder
	for _, v := range values {
		b.AppendString(v)
	}
	return b.Build()
}

// ref hand assembles a long view, so that the tests can describe views no
// builder in this package would ever produce. Views arriving over Arrow IPC
// were written by something else and this is how that is imitated.
func ref(length, block, offset int, prefix string) strview.View {
	var v strview.View
	binary.LittleEndian.PutUint32(v[0:4], uint32(length))
	copy(v[4:8], prefix)
	binary.LittleEndian.PutUint32(v[8:12], uint32(block))
	binary.LittleEndian.PutUint32(v[12:16], uint32(offset))
	return v
}

const long = "a value that does not fit inline"

func TestNewData(t *testing.T) {
	block := buffer.Wrap([]byte(long))
	views := []strview.View{
		strview.MakeInline([]byte("kuma")),
		ref(len(long), 0, 0, long[:4]),
	}

	d, err := strview.NewData(views, []*buffer.Buffer{block})
	if err != nil {
		t.Fatalf("NewData: %v", err)
	}
	if d.Len() != 2 {
		t.Fatalf("Len() = %d, want 2", d.Len())
	}
	if got := string(d.At(0)); got != "kuma" {
		t.Errorf("At(0) = %q", got)
	}
	if got := string(d.At(1)); got != long {
		t.Errorf("At(1) = %q", got)
	}
}

func TestNewDataEmpty(t *testing.T) {
	d, err := strview.NewData(nil, nil)
	if err != nil {
		t.Fatalf("NewData(nil, nil): %v", err)
	}
	if d.Len() != 0 {
		t.Errorf("Len() = %d, want 0", d.Len())
	}
}

// TestNewDataRejects is the reason validation exists. Every case here is a view
// that would otherwise read memory that is not the value it claims, and the
// only difference between a rejected column and a silently wrong one is this
// function.
func TestNewDataRejects(t *testing.T) {
	block := buffer.Wrap([]byte(long))
	blocks := []*buffer.Buffer{block}

	negative := strview.View{}
	negative[3] = 0x80

	pad := strview.MakeInline([]byte("kuma"))
	pad[15] = 1

	for _, c := range []struct {
		name string
		view strview.View
		want string
	}{
		{"negative length", negative, "negative length"},
		{"nonzero pad", pad, "nonzero pad"},
		{"block past the end", ref(len(long), 1, 0, long[:4]), "names block 1"},
		{"negative block", ref(len(long), -1, 0, long[:4]), "names block -1"},
		{"value past the end", ref(len(long), 0, 8, long[:4]), "block that is"},
		{"length past the end", ref(len(long)+1, 0, 0, long[:4]), "block that is"},
		{"offset past the end", ref(len(long), 0, 1<<20, long[:4]), "block that is"},
		{"negative offset", ref(len(long), 0, -1, long[:4]), "block that is"},
		{"prefix does not match", ref(len(long), 0, 0, "nope"), "prefix does not match"},
	} {
		t.Run(c.name, func(t *testing.T) {
			d, err := strview.NewData([]strview.View{c.view}, blocks)
			if err == nil {
				t.Fatalf("NewData accepted a view with a %s: At(0) = %q", c.name, d.At(0))
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error is %q, want it to mention %q", err, c.want)
			}
			if !strings.HasPrefix(err.Error(), "strview: view 0:") {
				t.Errorf("error is %q, want it to say which view", err)
			}
		})
	}
}

func TestNewDataAliases(t *testing.T) {
	// NewData does not copy, which is what makes a column arriving over IPC free
	// to open. The views and blocks it was handed are the ones it uses.
	views := []strview.View{strview.MakeInline([]byte("kuma"))}
	blocks := []*buffer.Buffer{buffer.Wrap([]byte(long))}

	d, err := strview.NewData(views, blocks)
	if err != nil {
		t.Fatalf("NewData: %v", err)
	}
	if &d.Views()[0] != &views[0] {
		t.Error("Views() is a copy")
	}
	if d.Blocks()[0] != blocks[0] {
		t.Error("Blocks() is a copy")
	}
}

func TestView(t *testing.T) {
	d := build("kuma", long)
	if got := d.View(0); got != strview.MakeInline([]byte("kuma")) {
		t.Errorf("View(0) = %s", got)
	}
	if got := d.View(1); got.IsInline() {
		t.Errorf("View(1) = %s, want a reference", got)
	}
}

func TestAt(t *testing.T) {
	values := []string{"", "a", "exactly12345", "thirteen char", long, strings.Repeat("x", 1000)}
	d := build(values...)
	for i, want := range values {
		if got := string(d.At(i)); got != want {
			t.Errorf("At(%d) = %q, want %q", i, got, want)
		}
	}
}

func TestEqual(t *testing.T) {
	values := []string{
		"",
		"a",
		"ab",
		"abc",
		"kuma",
		"exactly12345",
		"exactly12346",
		"thirteen char",
		"thirteen chars",
		long,
		long + " and more",
		strings.Repeat("x", 1000),
		strings.Repeat("x", 1000),
	}
	d := build(values...)

	for i := range values {
		for j := range values {
			want := values[i] == values[j]
			if got := d.Equal(i, j); got != want {
				t.Errorf("Equal(%d, %d) = %v for %q and %q", i, j, got, values[i], values[j])
			}
			if got := d.EqualValue(i, []byte(values[j])); got != want {
				t.Errorf("EqualValue(%d, %q) = %v", i, values[j], got)
			}

			want2 := bytes.Compare([]byte(values[i]), []byte(values[j]))
			if got := d.Compare(i, j); sign(got) != sign(want2) {
				t.Errorf("Compare(%d, %d) = %d, want the sign of %d for %q and %q",
					i, j, got, want2, values[i], values[j])
			}
		}
	}
}

// TestEqualLongValuesInDifferentBlocks covers the case the fast paths do not
// reach. Two copies of the same long value written far enough apart land in
// different blocks, so their views differ in every field but the length and the
// prefix, and the only way to get the answer right is to read both values.
func TestEqualAcrossBlocks(t *testing.T) {
	var b strview.Builder
	value := strings.Repeat("kuma", 500)
	b.AppendString(value)
	for range 20 {
		b.AppendString(strings.Repeat("filler", 1000))
	}
	b.AppendString(value)
	d := b.Build()

	last := d.Len() - 1
	if d.View(0).Block() == d.View(last).Block() {
		t.Skip("both copies landed in the same block, so this test proves nothing")
	}
	if !d.Equal(0, last) {
		t.Error("two copies of the same value in different blocks are not equal")
	}
	if d.Compare(0, last) != 0 {
		t.Errorf("Compare of two copies of the same value is %d", d.Compare(0, last))
	}
}

// TestComparePrefixDecides is the property the layout is for. When two values
// differ inside their first four bytes, the answer is in the views and no data
// block is touched. There is no way to observe that from outside the package,
// so this checks the answer is right for the cases that take the path.
func TestComparePrefixDecides(t *testing.T) {
	d := build(long, "zebra"+long)
	if d.Compare(0, 1) >= 0 {
		t.Errorf("Compare = %d, want a value that sorts %q before %q", d.Compare(0, 1), long, "zebra")
	}
	if d.Equal(0, 1) {
		t.Error("two values with different prefixes came back equal")
	}
}

func TestEqualValueAgainstADifferentLength(t *testing.T) {
	d := build("kuma")
	if d.EqualValue(0, []byte("kum")) {
		t.Error("EqualValue said a shorter value matched")
	}
	if d.EqualValue(0, []byte("kumas")) {
		t.Error("EqualValue said a longer value matched")
	}
	if !d.EqualValue(0, []byte("kuma")) {
		t.Error("EqualValue said a value did not match itself")
	}
}

func BenchmarkNewData(b *testing.B) {
	d := buildBench(1 << 14)
	views, blocks := d.Views(), d.Blocks()
	b.ReportAllocs()
	for b.Loop() {
		if _, err := strview.NewData(views, blocks); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAtShort(b *testing.B) {
	d := build("kuma")
	b.ReportAllocs()
	for b.Loop() {
		byteSink = d.At(0)
	}
}

func BenchmarkAtLong(b *testing.B) {
	d := build(long)
	b.ReportAllocs()
	for b.Loop() {
		byteSink = d.At(0)
	}
}

// BenchmarkComparePrefix and BenchmarkCompareFull are the same call on values
// that take different paths through it. The first pair differ in their first
// four bytes and are settled by the views alone. The second pair share a
// prefix, so both values have to be read out of their blocks.
func BenchmarkComparePrefix(b *testing.B) {
	d := build(long, "zebra"+long)
	b.ReportAllocs()
	for b.Loop() {
		intSink = d.Compare(0, 1)
	}
}

func BenchmarkCompareFull(b *testing.B) {
	d := build(long+"a", long+"b")
	b.ReportAllocs()
	for b.Loop() {
		intSink = d.Compare(0, 1)
	}
}

func BenchmarkEqualShort(b *testing.B) {
	d := build("kuma", "kuma")
	b.ReportAllocs()
	for b.Loop() {
		boolSink = d.Equal(0, 1)
	}
}

func BenchmarkEqualLong(b *testing.B) {
	d := build(long, long)
	b.ReportAllocs()
	for b.Loop() {
		boolSink = d.Equal(0, 1)
	}
}

var (
	byteSink []byte
	intSink  int
	boolSink bool
)
