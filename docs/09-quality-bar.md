# Quality bar

The stated goal is that this reads like a Go standard library package. That is mostly a constraint on process rather than on cleverness, so this document is about what it means in practice.

## API design

Keep the surface small and the names obvious. Every exported identifier gets a doc comment starting with the identifier name, enforced in CI with no exceptions. Every package gets a package comment explaining what it is for rather than listing what is in it. Every package gets at least one runnable example, and every non trivial exported function gets one too, because `go test` compiles and runs them and that makes them the only documentation that cannot rot.

Prefer a second function over a boolean parameter. `SortBy` and `SortDesc`, not `Sort(desc bool)`. Where an operation genuinely has more than three knobs, use an options struct rather than variadic option functions; Go 1.27's nested struct literal keys make those pleasant to write now.

Make zero values useful where you can. `var b bitmap.Bitmap` should be an empty bitmap, not a panic waiting to happen.

Accept interfaces and return concrete types, with the caveat from document 01 that our core types have to be concrete anyway for generic methods to work at all.

`context.Context` is the first parameter of anything that can block. Always.

For errors: they are values, sentinel errors are comparable with `errors.Is`, and the structured ones carry the column name, the plan position and the available alternatives. Nothing panics across an API boundary. Panics are for programmer error in unexported code, and even there returning an error is usually better. Error strings are lowercase with no trailing punctuation, and wrapping uses `%w`.

## Testing

| Layer | What is required |
|---|---|
| Unit | Table driven. 85 percent statement coverage overall, 95 percent on `bitmap`, `dtype` and `kernel`. |
| Fuzz | Every kernel against its scalar twin. Every parser against malformed input. Corpus committed to the repo. |
| Differential | Eager against lazy, SIMD against scalar, kuma against pandas 3.0 and DuckDB for parity functions. |
| Property | Round trip laws, so sort of sort equals sort, filter by true is identity, deserialize of serialize is identity. |
| Race | `go test -race` on the whole suite on every commit. Non negotiable for a parallel executor. |
| Leak | The Go 1.27 `goroutineleak` profile asserted after every cancelled query test. |
| Golden | Kernel outputs and `Explain` plan shapes committed as golden files, so a rewrite cannot silently change semantics. |
| Compile failure | A `go/types` driven suite asserting that the misuse listed in document 03 does not compile. |
| Benchmark | `testing.B` on every kernel and operator, with a `benchstat` gate in CI. |
| Conformance | Every checkbox in document 06 has a test, run from `kuma-py` against pandas 3.0 in the same process. |
| Boundary | Every exported C function forced to panic internally, asserting an error comes back rather than a crash. Handle leak and memory leak tests over a hundred thousand query cycles. |

Two of those matter more than the rest, and both are easy to defer.

The kernel differential fuzz, including every tail length from zero to twice the vector width. The failure mode of hand vectorized code is almost always the tail or the mask, and it is silent. Write this before writing the kernel.

The eager against lazy differential suite. A wrong query plan still returns a result that looks fine. Without a second independent path to compare against, optimizer bugs are invisible until a user finds them.

The compile failure suite is new with the typed column design and deserves a note. The main promise of that design is that certain mistakes do not compile, and a promise like that needs a test, otherwise a refactor can quietly loosen a type and nobody notices until the guarantee is gone.

## Floating point

Vectorized summation changes the association order, so `Sum` will not produce bit identical results to a naive scalar loop. That is expected and the vectorized version is actually more accurate, since summing in a tree accumulates less error than summing in a line. But it has consequences.

Document it on every affected function. Test with a tolerance derived from the magnitude and length of the input rather than an epsilon someone picked out of the air. Keep it deterministic for a fixed chunk size and thread count, and where parallel reduction order makes that impossible, say so and offer a deterministic mode. Offer Neumaier compensated summation for people who want accuracy over speed.

## Documentation

godoc is the documentation. If there is ever a separate site it is generated from godoc rather than written alongside it.

Every pandas name in document 06 needs a page under the name people will search for, including the five genuine omissions and the ones whose shape changed. Someone looking for `inplace` should land on an explanation of why frames are immutable, not on nothing.

The behavioural divergences from pandas need to be prominent rather than buried: RE2 regex has no backreferences or lookaround, floating point summation order differs, the timezone ambiguity policy is explicit and must be chosen, and an index is optional and never aligns automatically.

The migration guide is organized by pandas function name, so that it is searchable by what the reader already knows rather than by what we decided to call things.

The `Explain` output format is documented and stable, because people will parse it whether we want them to or not.

## Compatibility

Stay on v0.x until 1.0 and use it. Break things deliberately and early rather than shipping an API you will regret. After 1.0 the Go 1 compatibility promise applies to the tier 1 packages listed in document 11, and `apidiff` gates CI against the previous release.

The minimum Go version lives in `go.mod` and only ever goes up in a minor release, never a patch.

The library builds and passes without `GOEXPERIMENT=simd`. This is a compatibility property as much as a performance one: it is what makes it safe to build on an experimental package at all.

No cgo in the core. The C Data Interface in `kuma/ipc` is build tagged separately so the core stays pure Go and cross compiles cleanly.

## Dependencies

The core, meaning `kuma`, `kuma/dtype`, `kuma/bitmap`, `kuma/kernel`, `kuma/plan` and `kuma/exec`, depends on the standard library and nothing else.

Packages around the edge may take narrow, well maintained dependencies: compression codecs in `kuma/parquet`, `apache/arrow-go` at the boundary in `kuma/ipc`, `database/sql` in `kuma/sqlio`. `kuma-plot` is a separate module and does not appear in the main `go.mod` at all.

Every new dependency needs a written justification in the pull request. The core staying standard library only is what makes `kuma/bitmap` and `kuma/kernel` credible as packages someone might import on their own, which is part of the argument in document 11 for not having an `internal/` directory.

The bindings repositories are exempt from this in the obvious way. `kuma-py` depends on cffi and, for testing, on pandas, Polars, DuckDB and pyarrow. `kuma-bench` depends on all of those plus Docker. That is precisely why they are separate repositories: none of it can reach the Go module.

## CI

On every commit: `go vet`, `staticcheck`, `gofumpt -l`, `go test -race`, the coverage gate, `apidiff` once past 1.0, a benchmark run compared with `benchstat`, and the doc comment completeness check.

Nightly: the full fuzz corpus, db-benchmark at 0.5 GB and 5 GB in `kuma-bench`, TPC-H SF1, and the complete SIMD matrix from document 05 including the SDE backed AVX-512 run.

Per release: db-benchmark at 50 GB, TPC-H SF10, the cross language differential against pandas 3.0 and DuckDB, and the full pandas conformance suite with a per section pass rate.

On every binding commit: the boundary tests, a build of all six Python wheels and all five shared library platforms, and the example from document 07 run unchanged on Node, Bun and Deno.

Also triggered by a new pandas or Polars release, since a comparison against a version nobody runs is not a comparison.

## Performance discipline

The full treatment is document 10. Four things belong here because they are quality rules rather than benchmark design.

No performance claim without a committed benchmark behind it, in `tamnd/kuma-bench`, with the result JSON in the repository.

Publish the results we lose. A benchmark suite that only reports wins is an advertisement, and readers correctly discount everything in it including the true parts.

Track allocations with `-benchmem` alongside time. For a columnar engine the allocation count is usually the first sign of a design problem, well before the wall clock shows it. Track the shared library size per platform the same way, since it only grows and nobody notices until a package is unshippable.

Only cite the official `duckdblabs/db-benchmark` report. Secondary sources turned out to be unreliable during the research for this spec: several blog aggregations contained internally contradictory version numbers and dates, and the old `h2oai.github.io` page is frozen at 2021 results and should never be used for comparison.
