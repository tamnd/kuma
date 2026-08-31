# Package layout

## The constraint

No `internal/` directory anywhere. This was a requirement rather than a preference, and it changes how the code has to be organized, so it is worth being explicit about what it costs and how the cost gets paid.

`internal/` is how Go projects normally get a private implementation. Without it, every package in the module is importable by anyone, which means every exported identifier is a promise. A large library with no `internal/` and no discipline turns into a library where every refactor is a breaking change.

Three things pay for it.

**Unexport aggressively.** The default answer to "should this be exported" is no. A package can be public while almost nothing in it is, and that is the normal case here. `kuma/exec` is importable and contains perhaps four exported identifiers.

**Document stability in tiers**, described below, so that being importable and being stable are separate statements.

**Keep the package count low.** Fewer packages means fewer boundaries that have to be exported across. Several things that would naturally be their own package are files in a larger one for exactly this reason.

There is an upside, and it is the reason the constraint is a good one rather than merely a constraint. `kuma/bitmap` and `kuma/kernel` are genuinely useful on their own. Someone writing an unrelated columnar system in Go should be able to import a well tested aligned bitmap implementation without taking the whole engine. Hiding those in `internal/` would waste them.

## Stability tiers

Every package declares its tier in the first line of its package comment, and the tier is what the compatibility promise attaches to.

**Tier 1, stable.** The Go 1 compatibility promise applies after 1.0. Breaking changes require a major version. `apidiff` gates every commit against the last release.

**Tier 2, evolving.** Stable within a minor version. Breaking changes are allowed in a minor release and are listed in the changelog with a migration note. Most of the engine internals live here.

**Tier 3, experimental.** No promise at all. May change or disappear in a patch release. Anything touching `GOEXPERIMENT=simd` is here by necessity, since the packages underneath it are themselves unstable.

The declaration is mechanical, not decorative:

```go
// Package exec executes physical query plans.
//
// Stability: tier 2, evolving. The API may change in any minor release.
// Programs should use the kuma package rather than importing this directly.
package exec
```

A CI check asserts that every package has a tier line and that no tier 1 package imports a tier 3 package, because a stable API built on an unstable one is not stable.

## The tree

```
github.com/tamnd/kuma

  kuma.go                       Frame, LazyFrame, Series, the top level functions   T1
  lazy.go                       the query that is written down and not run yet      T1
  run.go                        the engine that works a plan out over the frame     T1
  expr.go                       the typed expression interface, evaluation          T1
  col.go                        F64Col, I64Col, StrCol, BoolCol, TimeCol and friends T1
  agg.go                        aggregation constructors                            T1
  io.go                         ScanCSV, ScanParquet, ScanIPC, ScanNDJSON           T1
  typed.go                      Typed[T], CollectInto, Reduce, generic methods      T1
  dynamic.go                    Dynamic, Dyn, Bind, the escape hatch                T1

  dtype/                        DataType, Schema, Field, coercion rules             T1
  bitmap/                       validity bitmaps, popcount, boolean ops             T1
  buffer/                       aligned allocation, size class pool                 T2
  strview/                      the string and binary view layout                   T1
  array/                        Array, ChunkedArray, builders                       T2

  kernel/                       the dispatch table and every compute kernel         T2
    kernel.go                   dispatch table, registration, Scalar constraint
    *_scalar.go                 the reference implementations, always built
    *_portable.go               //go:build goexperiment.simd
    *_amd64.go                  //go:build goexperiment.simd && amd64
    *_arm64.go                  //go:build goexperiment.simd && arm64
    dispatch_init.go            runtime feature detection

  plan/                         logical plan, optimizer passes, Explain             T2
    expr.go                     the untyped expression the typed handles carry
    intern.go                   the table that makes two equal expressions one
    plan.go                     node types
    agg.go                      the aggregations, and what each one reads
    type.go                     what an expression is checked to be, before it runs
    schema.go                   what a plan produces, and Validate
    error.go                    what a plan says when the check turns it away
    optimize.go                 the passes from document 02
    explain.go                  the documented output format
    serialize.go                the protobuf plan format for document 07

  exec/                         physical operators, morsel scheduler                T2

  compress/                     the codecs the file formats need and Go has not     T2
    snappy/                     the snappy block format

  csv/                          reader and writer, schema inference                 T1
  parquet/                      reader with pushdown, writer with statistics        T1
  ipc/                          Arrow IPC and the C Data Interface                  T1
  ndjson/                       on encoding/json/v2                                 T1
  orc/  fwf/  xlsx/  htmlio/  xmlio/                                                T2

  strs/                         the string namespace from document 06               T1
  temporal/                     the datetime namespace                              T1
  calendar/                     business days, holidays, the pandas date offsets    T1
  window/                       rolling, expanding, EWM, Over                       T1
  reshape/                      pivot, unpivot, stack, unstack, explode             T1
  stats/                        corr, cov, rank, quantile, describe                 T1

  sql/                          SQL to logical plan                                 T2
  sqlio/                        database/sql and ADBC                               T2

  capi/                         the C shared library from document 07               T2
    capi.go                     //go:build cgo, the exported functions
    handle.go                   the handle table
    kuma.h                      the hand written public header

  kumatest/                     frame equality, readable diffs, random data         T1
  kumagen/                      the code generator, main package                    T1

  arrowgo/                      the arrow-go bridge, a module of its own            T2
```

The tier column is the point of the table. Nothing in tier 1 may name a `simd` type, and nothing outside `kernel` may either.

`kernel` was tier 3 in the first draft of this document, on the grounds that it will one day be full of build tagged files calling an unstable package. It is tier 2 instead. Its exported surface is arrays and slices and never names a `simd` type, which is the second of the four rules in document 05, so the churn stays inside the package. A tier says what a caller can rely on rather than what the implementation is made of, and the root package has to be able to call a kernel.

## Notes on specific choices

**The root package is large.** `Frame`, `LazyFrame`, `Series`, `Expr` and all the column handle types live together because they are mutually recursive and splitting them would mean exporting the recursion. This is the same reason `net/http` is one big package.

**The column handle types are generated, once.** There are five of them now and there will be a dozen eventually, and their method sets overlap heavily but not completely. They are generated into `col.go` by a small program in `kumagen`, checked in, and reviewed like any other code. This is generation to avoid copy paste, not generation as architecture.

**`kernel` is one package, not one per family.** It has to be, because dispatch is a single table and because the build tag combinations multiply. One package with a hundred files is easier to reason about here than fifteen packages with the same build tag matrix repeated in each.

**`capi` is build tagged on cgo** so that `go build ./...` on a machine without a C toolchain still works. Nothing else in the module imports it.

**`kuma-plot` is a separate module**, not a package here, so that a plotting dependency can never end up in a server binary. Same reasoning for the bindings repositories in document 07.

**`arrowgo` is a nested module in the same repository.** Document 01 put the arrow-go bridge inside `ipc/`, and that would have put `apache/arrow-go` and its six transitive dependencies into the `go.mod` of everyone who imports kuma at all, including the people who only want to read a CSV. It is a directory here with a `go.mod` of its own, so `go get github.com/tamnd/kuma/arrowgo` brings the dependency and `go get github.com/tamnd/kuma` still brings nothing. It is in this repository rather than its own because a change to a kuma buffer layout and the change to the bridge that reads it are one commit, and a `replace` line in a module that is not the main one is ignored by consumers, so the tests here run against the working tree while everybody else gets the tagged version named in the require.

## Naming

Package names are lowercase, single word, no underscores. `strs` rather than `stringops` because `strings` is taken and the alternative was worse. `temporal` rather than `time` for the same reason.

Do not stutter. `bitmap.New`, not `bitmap.NewBitmap`. `dtype.Int64`, not `dtype.DTypeInt64`.

Files are named for what they contain, and the build tagged variants share a stem so that the family is visible in a directory listing.

## The modules, all five

| Module | Path | Contains |
|---|---|---|
| the library | `github.com/tamnd/kuma` | everything above except the bridge |
| the arrow-go bridge | `github.com/tamnd/kuma/arrowgo` | nested here, the one module with a dependency |
| plotting | `github.com/tamnd/kuma-plot` | Vega-Lite output, post 1.0 |
| benchmarks | `github.com/tamnd/kuma-bench` | document 10 |
| bindings | `tamnd/kuma-py`, `tamnd/kuma-js` | document 07, not Go modules |

Separate repositories rather than one, because the release cadences differ, because the dependency sets are incompatible, and because a Python user should never have to clone Go source to install a wheel. The bridge is the exception that stays in this repository, for the reason given above.
