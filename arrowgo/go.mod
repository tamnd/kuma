module github.com/tamnd/kuma/arrowgo

go 1.27.0

require (
	github.com/apache/arrow-go/v18 v18.7.0
	github.com/tamnd/kuma v0.0.23
)

require (
	github.com/goccy/go-json v0.10.6 // indirect
	github.com/google/flatbuffers v25.12.19+incompatible // indirect
	github.com/klauspost/cpuid/v2 v2.4.0 // indirect
	github.com/zeebo/xxh3 v1.1.0 // indirect
	golang.org/x/exp v0.0.0-20260112195511-716be5621a96 // indirect
	golang.org/x/sys v0.47.0 // indirect
)

// The bridge and the library it bridges are in the same repository, so a change
// to both is one commit and the tests here run against the working tree rather
// than against the last tag. A replace in a module that is not the main one is
// ignored, so this line does nothing to anybody who depends on this package and
// the require above is what they get.
replace github.com/tamnd/kuma => ../
