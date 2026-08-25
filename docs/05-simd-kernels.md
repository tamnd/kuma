# SIMD kernels

## Why this needs doing by hand

The Go compiler does not auto vectorize. It never has, and there is no sign that it is about to start. Every bit of vectorization in a Go program is something a person wrote deliberately.

Until Go 1.26 that meant assembly or cgo. As of Go 1.27 there are two packages that make it tractable in ordinary Go, both still behind `GOEXPERIMENT=simd`:

`simd` is new in 1.27. It gives you portable, vector size agnostic types, named after the primitive they hold with an `s` on the end, so `Float64s`, `Int32s` and so on. It is backed by real hardware instructions on amd64 and on arm64 NEON, and emulated in plain Go everywhere else. The vector length is at least 128 bits and stays constant for the life of the process. It supports a useful subset of the architecture specific operations.

`simd/archsimd` is the architecture specific layer. On amd64 it covers 128, 256 and 512 bit vectors. Go 1.27 added arm64 NEON at 128 bits and WebAssembly at 128 bits, and revised the amd64 API in a way that broke existing code. Feature detection is available at runtime through package level variables, so `archsimd.X86.AVX512()` tells you whether the wide path is safe to take.

Both packages are explicitly unstable. That fact drives most of the decisions below.

## Four rules

**SIMD is an optimization and never a dependency.** `go build ./...` and `go test ./...` on stock Go 1.27 with no experiment flag must produce a complete, correct, fully tested library. Someone on a normal Go toolchain gets right answers at scalar speed. This is not a nice to have, it is the thing that makes it safe to depend on an experimental package at all.

**No `simd` or `archsimd` type appears in any exported signature.** Kernels take and return plain slices. Pinning a public API to a package whose amd64 surface already broke once during the 1.27 cycle would be a mistake we could not walk back after 1.0.

**Every vectorized kernel has a scalar twin, and the twin is the specification.** Not a fallback, a specification. When the two disagree the scalar one is right by definition. This is what makes the correctness story below work.

**Dispatch resolves once.** At package init, into a table of function values. Not per call, and certainly not per element.

## Layout

```
kuma/kernel/
  kernel.go          public surface: the dispatch table, registration, the Scalar constraint
  add_scalar.go      build tag !goexperiment.simd, the reference implementation
  add_portable.go    build tag goexperiment.simd, using the portable simd package
  add_amd64.go       build tag goexperiment.simd && amd64, using archsimd, AVX-512 path
  add_arm64.go       build tag goexperiment.simd && arm64, using archsimd, NEON path
  dispatch_init.go   runtime feature detection, fills in the table
```

The public part is just this:

```go
package kernel

type BinaryF64 func(dst, a, b []float64)

var AddF64 BinaryF64 = addF64Scalar   // replaced at init if something better is available
```

And a portable implementation looks like this:

```go
//go:build goexperiment.simd

package kernel

import "simd"

func addF64Portable(dst, a, b []float64) {
    w := simd.Float64s{}.Len()      // lanes, constant for the process
    i := 0
    for ; i+w <= len(a); i += w {
        simd.LoadFloat64sSlice(a[i:]).
            Add(simd.LoadFloat64sSlice(b[i:])).
            StoreSlice(dst[i:])
    }
    for ; i < len(a); i++ {          // scalar tail
        dst[i] = a[i] + b[i]
    }
}
```

Check the exact identifier names against the Go 1.27 documentation before writing any of these. The API is unstable and it moved during the 1.27 cycle. Keeping all of it inside one package is precisely so that the next round of churn costs an afternoon instead of a refactor.

## What to build, in what order

The instinct is to start with arithmetic because it is the easiest thing to vectorize. That is the wrong order. Arithmetic on large arrays is limited by memory bandwidth, not by instruction throughput, so the ceiling is low. Build in this order instead.

| Rank | Family | Approach | Rough speedup | Why here |
|---|---|---|---|---|
| 1 | CSV and NDJSON parsing | SIMD delimiter scan, vectorized integer and float parsing, UTF-8 validation | 5x to 15x | Ingestion dominates every first impression. This is what "fast CSV" actually means. |
| 2 | Comparison to bitmap | AVX-512 mask registers line up exactly with the validity bitmap | 8x to 16x | Feeds every filter, and the mask register is already the output format we want. |
| 3 | Filter and compaction | AVX-512 VPCOMPRESSQ, shuffle table on AVX2 and NEON | 4x to 10x | The other half of every predicate. |
| 4 | Aggregations | Four to eight independent accumulators to break the floating point dependency chain, Neumaier compensated summation | 4x to 8x | Latency bound rather than bandwidth bound, so there is real headroom. |
| 5 | Bitmap operations | popcount, AND, OR, NOT over validity | 10x and up | Trivial to write and used absolutely everywhere. |
| 6 | String comparison and search | StringView inline prefix, SIMD memchr, contains, starts with | 3x to 10x | Strings are the hottest dtype in real data. |
| 7 | Sort | Vectorized quicksort, radix sort for integers, dates and decimals | 3x to 8x | Large win, large effort. |
| 8 | Group by hashing | Vectorized hashing across eight lanes, dictionary encoded keys skip hashing entirely | 2x to 5x | The dictionary shortcut beats the SIMD hash, so do that first. |
| 9 | Elementwise arithmetic | The obvious one | 2x to 4x | Last, because memory bandwidth caps it. |

## The two decisions that matter more than any kernel

### Branch on nulls once per chunk

```go
func addChunk(dst, a, b *Array) {
    if a.nullCount == 0 && b.nullCount == 0 {
        kernel.AddF64(dst.f64, a.f64, b.f64)   // full width, no masking
        dst.validity = nil
        return
    }
    kernel.AddF64Masked(dst.f64, a.f64, b.f64, a.validity, b.validity)
}
```

Once per chunk, never per element. Most real columns have no nulls at all, so the unmasked path is the common case and the masked kernel is the exception. Getting this backwards costs more than any individual kernel gains, and it is easy to get backwards because the masked version is the more general one and therefore feels like the natural default.

### Pick the chunk size by measuring

The chunk is simultaneously the unit of vectorization, the unit of parallelism, and the unit of the null check above. Too small and dispatch overhead swamps the work. Too large and it falls out of L2 cache and you are back to waiting on memory. The right answer is somewhere between 1024 and 8192 rows depending on how wide the rows are, and it should be determined by benchmark during the parallel execution milestone rather than guessed now.

## Proving it works

The scalar twin is the specification, and there are three layers of checking built on that.

Differential fuzzing is the important one. Every vectorized kernel runs against its scalar twin over random inputs, and this catches almost everything, because the failure mode of hand vectorized code is nearly always the tail or the mask rather than the main loop. The fuzz corpus has to include every tail length from zero to twice the vector width, all null and no null inputs, and the awkward floating point values, meaning NaN, both infinities and denormals. If you write only one test in this package, write this one, and write it before the kernel.

Golden vectors come second. Fixed inputs with expected outputs checked into the repo, so that rewriting a kernel cannot quietly change its semantics.

Cross implementation checking comes third. For the aggregations where summation order is observable, compare against pyarrow or Polars through the C Data Interface.

One thing to be upfront about: vectorized summation changes the order of associations, so `Sum` will not produce bit identical results to a naive scalar loop. This is expected, and the vectorized version is actually more accurate, because summing in a tree has lower error growth than summing in a line. But it has to be documented on every affected function, and the tests have to compare with a tolerance derived from the magnitude and length of the input rather than some arbitrary epsilon someone picked. Offer compensated summation as an option for people who want accuracy over speed.

## CI matrix

| Target | Covers |
|---|---|
| darwin/arm64, no experiment flag | The scalar reference, which is what most users will run |
| darwin/arm64 with GOEXPERIMENT=simd | NEON |
| linux/amd64, no experiment flag | The scalar reference again |
| linux/amd64 with GOEXPERIMENT=simd | AVX2 |
| linux/amd64 on Intel SDE, or a Zen 4 or Ice Lake or newer runner | AVX-512 |
| js/wasm with GOEXPERIMENT=simd | WebAssembly 128 bit, nice to have |

The AVX-512 row is not optional. Intel fused AVX-512 off on consumer 12th through 14th generation parts, so a laptop will not exercise that path. Untested vector code is worse than no vector code, because it looks like it works.

## When assembly is still the answer

Go 1.27's intrinsic coverage is incomplete. arrow-go still hand writes ARM64 NEON assembly for its min and max kernels, and the Go team acknowledges that some AVX-512 intrinsics are not covered yet and that hand written assembly still wins in places.

The dispatch table takes function values, so an assembly implementation generated with avo drops in beside an intrinsic one with no change to any API. Keep that door open, but do not walk through it first. Write the portable version, benchmark it, and descend to assembly only against a gap you have actually measured.
