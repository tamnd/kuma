package main

import "testing"

// sink is where the generated source goes, so that nothing is optimized away.
var sink []byte

// BenchmarkLoadAndGenerate is the whole tool over one small package, which is
// what a go generate line costs every time somebody runs it. The property this
// is here to keep is the one in document 03: fast enough that running it on
// every build is not worth thinking about.
func BenchmarkLoadAndGenerate(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		p, err := loadPackage("testdata/trades")
		if err != nil {
			b.Fatalf("loadPackage: %v", err)
		}
		src, err := generate(p, "Trade")
		if err != nil {
			b.Fatalf("generate: %v", err)
		}
		sink = src
	}
}

// BenchmarkGenerate is the same without the parsing, which is the part that
// grows with the package rather than with the struct.
func BenchmarkGenerate(b *testing.B) {
	p, err := loadPackage("testdata/trades")
	if err != nil {
		b.Fatalf("loadPackage: %v", err)
	}

	b.ReportAllocs()
	for b.Loop() {
		src, err := generate(p, "Trade")
		if err != nil {
			b.Fatalf("generate: %v", err)
		}
		sink = src
	}
}
