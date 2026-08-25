# Landscape research, August 2026

Everything here was checked against primary sources on 25 August 2026. Links are at the bottom.

The point of this document is not to survey the field for its own sake. It is that eight or nine specific facts about the state of Go, pandas, Polars and Arrow in 2026 determine most of the architecture in the rest of these documents, and it is worth writing down what those facts are so that when one of them changes we know which decisions to revisit.

## 1. Go 1.27, released August 2026

This is the release that makes the design possible.

### Generic methods

Methods on concrete types may now declare their own type parameters. This is proposal 77273, from Robert Griesemer, and it reverses a position the Go FAQ had held since generics landed in 1.18.

Two restrictions come with it, and neither is a footnote.

**Interfaces cannot declare generic methods, and a generic method cannot satisfy an interface method.** The reason is dispatch: monomorphization and runtime interface dispatch do not fit together.

**Generic methods are invisible to reflection.** `reflect.TypeOf(x)` will not list them.

The standard library example is `(*rand.Rand).N[Int intType](n Int) Int`, where the receiver is the generator and the type parameter belongs to the method. There is also a smaller rule worth knowing: using a generic receiver type in a method expression requires instantiation, so `List[string].format` is legal and `List.format` is not.

What this means for kuma:

`Expr`, `Typed[T]`, `Frame`, `LazyFrame` and `GroupBy` have to be concrete structs. If any of them were an interface we would lose the entire typed API from document 03, because `Agg[R]` and `Select[R]` could not exist.

Polymorphism over dtype cannot be written as `interface { Apply[T](...) }`. Every extension point, meaning user kernels, custom aggregations and custom optimizer rules, has to be a registry of concrete function values keyed by dtype. This is a real API shape decision and it has to be made before M0, because retrofitting it means rewriting every extension surface in the library.

Anything driven by reflection, meaning the struct to frame mapping and `CollectInto[R]`, must reflect over the type argument and never over kuma's own methods.

### SIMD

Go 1.27 continues the `GOEXPERIMENT=simd` work that began in 1.26.

`simd/archsimd` is the architecture specific layer. The amd64 API was revised in a breaking way during the 1.27 cycle, and 1.27 added arm64 NEON at 128 bits and WebAssembly at 128 bits. Runtime feature detection is exposed through package level variables, so on amd64 you can ask `archsimd.X86.AVX512()`.

`simd` is new in 1.27. It provides portable, vector size agnostic types named after the primitive they hold with an `s` appended, so `Int8s`, `Uint16s` and `Float64s`. It is implemented with real instructions where the hardware has them and emulated in plain Go otherwise. Vector length is at least 128 bits and stays constant for the life of the process. It supports a scalable subset of the archsimd operations.

Both packages are explicitly unstable and neither builds without the experiment flag.

arrow-go's experience corroborates the state of things. Their v18.6.0 optimized ARM64 NEON min and max in hand written assembly, and the Go team notes that some AVX-512 intrinsics are still uncovered and that assembly still wins in places.

What this means for kuma: the kernel layer is a dispatch table over function values, with three implementations per kernel, meaning portable `simd`, architecture specific `archsimd`, and a plain Go scalar reference, selected by build tag plus a runtime feature check. The library must build, pass its whole test suite and be genuinely usable with no experiment flag set. SIMD is an optimization and never a dependency. And no `simd` type may appear in an exported signature, because pinning a public API to a package whose amd64 surface already broke once is not a decision we could walk back after 1.0.

### Other 1.27 items that matter

Size specialized allocation makes small allocations under 80 bytes up to 30 percent cheaper, worth around 1 percent overall on allocation heavy programs. Mostly marginal for us since we allocate few large buffers by design, but it helps the expression building path, which allocates many small nodes.

The `goroutineleak` profile graduates from experimental to a regular `runtime/pprof` profile with a `net/http/pprof` endpoint. This is directly useful, because a morsel driven executor with worker pools and channel fan in is exactly the shape that leaks goroutines on cancellation. It goes into the executor test suite.

pprof goroutine labels now appear in tracebacks for modules declaring `go 1.27`, so executor goroutines get labelled by operator and morsel and crash dumps become readable.

Struct literal keys may now address nested and embedded fields, which is a modest ergonomic win for the options structs.

Function type inference is extended to all assignment contexts, which improves inference on the generic expression constructors.

`encoding/json/v2` is available and is what the NDJSON reader should use rather than v1.

Unicode 17 is in, which affects the case conversion and normalization behaviour of the string kernels.

macOS 13 or newer is now required.

## 2. pandas 3.0, released 21 January 2026

This is the most important finding for scoping, because it changes what 1:1 parity even means.

pandas 3.0 adopted three things kuma was going to do anyway.

**Copy-on-Write is the default and the only mode.** The decade old ambiguity about whether an operation returns a view or a copy is gone, and `df[cond]['col'] = value` now raises `ChainedAssignmentError`.

**The PyArrow backed string dtype is the default.** String columns infer as a native `str` dtype backed by Arrow rather than as `object`. Reported gains are 5x to 10x on text heavy work, plus zero copy interchange with Polars, DuckDB and Spark.

**`pd.col()` expressions exist.** pandas now has an expression builder, so column operations no longer require lambdas.

What this means for kuma:

Parity is easier and more coherent than it would have been a year ago, because the API we are matching has moved toward the immutable, columnar, expression oriented model we were designing for anyway. `pd.col()` maps almost directly onto `kuma.Col()`.

The checklist in document 06 targets pandas 3.0 rather than 2.x. Several 2.x behaviours, meaning `inplace=`, chained assignment, `object` strings and NaN as null, are deprecated or removed upstream, which retroactively justifies not reproducing them.

pandas closed part of the performance gap. The honest positioning is no longer that pandas is slow. It is that Go programmers have no credible dataframe at all, which is section 5.

## 3. Polars, the performance bar

Polars is the reference implementation of the design kuma is pursuing, and it is what we measure against.

The new streaming engine shipped from 1.31.1 and is progressively becoming the default. It is opt in today through `pl.Config.set_engine_affinity("streaming")` or `collect(engine="streaming")`, and operators without a streaming implementation fall back transparently to the in memory engine.

The architecture is out of core group by, equi join and sort operators over a lock free memory manager with spill to disk, plus a fully out of core multiplexer.

Version 1.37 added the new sink pipeline for NDJSON, CSV and IPC. Version 1.38 added a streaming merge join. Version 1.39 added a streaming as of join. All major formats now have streaming scans, including CSV.

The July 2026 release, py-1.43.0 on 21 July, had Hive partition awareness as its headline: group bys and joins organize work around partition keys, and inner joins can be rewritten as a union of partition filtered joins. It also added `POLARS_OOC_DISK_BUDGET_MB` for out of core disk budgeting.

Polars 2.0 has not shipped. The roadmap issue notes that July 2026 was two years since 1.0, with 45 issues blocking the 2.0 milestone and the question of whether streaming becomes the default in 2.0 still open.

What this means for kuma:

The three year arc of Polars' engineering says the ordering is a correct in memory vectorized engine first and streaming second. Do not attempt streaming before M9.

Partition aware planning on Hive keys is a large real world win and it is cheap to design for early. The scan node needs to carry partition metadata from day one even if the optimizer ignores it until later.

Transparent per operator fallback from streaming to in memory is the right pattern, because it is what allows a streaming engine to ship while incomplete.

## 4. The Arrow ecosystem

Arrow 25.0.0 for C++ and Python was released on 10 July 2026.

arrow-go v18.7.0 was released on 21 July 2026, on its own v18.x track with roughly quarterly minor releases. v18.6.0 added compute sorting functions, an `array/arreflect` package for round tripping between Arrow and Go structs, a 20 to 30 percent improvement to the take kernel, and the optimized ARM64 NEON min and max assembly mentioned earlier.

StringView and BinaryView have a three tier adoption story. C++, Python and Rust have the layout and the compute and cast support, and arrow-rs is now optimizing view decoders and filling in kernels like `substring`. arrow-go has the type definitions and IPC support but historically thin compute coverage, to the point that casting utf8 to string_view panicked with "not implemented".

This is the decisive finding in this document, and three things follow from it.

**Adopt the Arrow layout and do not build on arrow-go's compute.** The layout gives us zero copy interop with the entire ecosystem for free. The compute kernels are the exact thing kuma exists to provide, and arrow-go's are both incomplete and not optimized in the ways we need.

**Ship an explicit `kuma/ipc` bridge** covering the Arrow C Data Interface and IPC, converting to and from arrow-go types at the boundary only. This is an immediate escape hatch, because anything kuma has not implemented yet can be handed to DuckDB or pyarrow with no copy. Build it early, at M2, not late. It de-risks the entire parity roadmap and it is also the foundation the bindings in document 07 sit on.

**StringView is a genuine differentiator.** Since arrow-go's view support is thin, a Go engine with first class StringView, meaning a 16 byte inline prefix and comparisons that never touch the heap, is ahead of the Go ecosystem on the single hottest real world dtype. Make it the default string representation rather than an option.

## 5. The Go dataframe ecosystem, and the actual gap

gota is the best known and describes its own API as "in flux, use at your own risk". It covers concat, joins and CSV and JSON. No Arrow, no Parquet, no lazy evaluation, no SIMD.

qframe has a cleaner API and beat gota in every benchmark of the well known comparison, but it is likewise not Arrow based.

The GitHub Go `dataframe` topic holds 18 public repositories. The Arrow based ones are dormant: one last updated in 2020, another in 2023, and Gandalff last touched in December 2023.

The consequence is that there is no credible Go dataframe. This is not a crowded field where kuma needs a differentiator, it is an empty one. The competition for a Go programmer doing data work today is shelling out to Python or writing a for loop.

That sets the bar for v0.1 much lower than beating Polars, and it means ergonomics and correctness matter more than the last 30 percent of performance. The design should reflect that, and specifically it should not sacrifice API quality for benchmark numbers.

## 6. The benchmark landscape

The H2O.ai db-benchmark is no longer run by H2O.ai. It went dormant after 2 July 2021 and DuckDB Labs took it over. The live suite is `duckdblabs/db-benchmark`, published at duckdblabs.github.io/db-benchmark. The old h2oai.github.io page is frozen at 2021 numbers and must not be used for comparison.

Its scope is group by, join and advanced queries at one million, ten million, one hundred million and one billion rows, covering data.table, Polars, dplyr, ClickHouse, DuckDB and others. The hardware is a `c6id.metal` instance with 250 GB of RAM and 128 cores on Ubuntu 22.04, with a reproduction script at `_utils/repro.sh`.

One methodological note matters more than the rest. ClickHouse and DuckDB use `CREATE TABLE ans AS SELECT` so that lazy engines are forced to materialize. kuma's harness has to do the same, because benchmarking a lazy engine without materializing the result measures nothing at all.

At the one billion row tier the results are sparse, with several solutions running out of memory or lacking implementations entirely.

The consequence is to adopt db-benchmark for group by and join, plus TPC-H at scale factors 1 and 10, as the two gates, wired into CI from M1 so that regressions are caught continuously rather than discovered at M5. Secondary sources on benchmark numbers proved unreliable during this research, with several blog aggregations containing internally contradictory versions and dates, so only the official report gets cited. The full treatment is document 10.

## 7. The binding landscape

Relevant because document 07 commits to Python and JavaScript bindings, and the right technique for each changed recently.

**Free threaded Python is now real.** Python 3.13 introduced the free threaded build and 3.14 made it officially supported in October 2025. By 2026 it is what performance minded users are running. The practical consequence is that a binding going through the stable C ABI works there unmodified, while one compiled against CPython internals needs porting and needs a separate wheel per interpreter version. That is the argument for cffi in ABI mode over a compiled extension module, and it cuts the wheel build matrix from roughly forty artifacts to six.

**The Arrow PyCapsule interface is the interop standard.** `__arrow_c_array__` and `__arrow_c_stream__` are what pandas 3.0, Polars, DuckDB and pyarrow all agreed on. Implementing that protocol once gives zero copy interchange with all of them and requires no library specific code paths.

**N-API covers all three JavaScript runtimes.** Node has it, Bun has supported it since 1.0 and Deno since 1.38. So one addon covers all three, and the per runtime FFI mechanisms, meaning koffi, `bun:ffi` and `Deno.dlopen`, are a secondary path rather than three separate ports.

**cgo's pointer rules force a handle table.** Go pointers cannot be held by C across a call, so every long lived object crosses the boundary as an opaque integer. This is well trodden ground and the pattern is settled, but it has to be designed in rather than discovered.

## 8. Consolidated design consequences

| Finding | Consequence |
|---|---|
| Generic methods only on concrete types | `Expr`, `Frame` and `Typed[T]` are structs. Extension points are dtype keyed function registries, never interfaces. |
| Generic methods invisible to reflection | Struct mapping reflects over the type argument only. |
| The `simd` API is unstable and experiment gated | Kernels sit behind a dispatch table with three implementations. No `simd` type in any exported signature. Full functionality without the flag. |
| arrow-go compute is thin | Adopt the Arrow layout, write our own kernels, bridge at `kuma/ipc`. |
| arrow-go StringView compute is thin | Make StringView the default string representation. It is a real edge. |
| pandas 3.0 went Copy-on-Write, Arrow and expressions | The checklist targets 3.0, and not reproducing `inplace=` now matches upstream rather than diverging from it. |
| Polars ordering was in memory then streaming | Streaming is M9 and not earlier. |
| Polars added Hive partition awareness | Scan nodes carry partition metadata from M2 even though nothing uses it until M9. |
| The Go dataframe ecosystem is empty | Ergonomics and correctness outrank the last 30 percent of throughput. |
| db-benchmark moved to DuckDB Labs | Use the current suite, force materialization, gate in CI from M1. |
| Go 1.27 `goroutineleak` profile | Wire it into the executor tests, because worker pools leak on cancel. |
| Go 1.27 pprof labels in tracebacks | Label executor goroutines by operator and morsel. |
| Free threaded Python is supported | Bind through the stable C ABI with cffi, not a compiled extension module. |
| Arrow PyCapsule is the interop standard | Implement it once and get pandas, Polars, DuckDB and pyarrow for free. |
| N-API works on Node, Bun and Deno | One addon covers all three runtimes. |

## Sources

**Go 1.27**

- Go 1.27 release notes, https://go.dev/doc/go1.27
- Go 1.27 is released, https://go.dev/blog/go1.27
- golang/go issue 77273, the generic methods proposal, https://github.com/golang/go/issues/77273
- Generic Methods Arrive in Go 1.27, Gopher Guides, https://www.gopherguides.com/articles/golang-generic-methods
- Go 1.27 Adds Generic Methods to Concrete Types, Leaves Interfaces and Reflection Untouched, https://xenospectrum.com/en/go127-generic-methods/
- Ready for Go 1.27 on Day One, JetBrains, https://blog.jetbrains.com/go/2026/08/20/ready-for-go-1-27-on-day-one/
- Go 1.27 interactive tour, VictoriaMetrics, https://victoriametrics.com/blog/go-1-27/

**Go SIMD**

- golang/go issue 73787, simd and archsimd under a GOEXPERIMENT, https://github.com/golang/go/issues/73787
- https://pkg.go.dev/simd
- https://pkg.go.dev/simd/archsimd

**pandas 3.0**

- pandas 3.0 released, https://pandas.pydata.org/community/blog/pandas-3.0.html
- What's new in 3.0.0, 21 January 2026, https://pandas.pydata.org/docs/whatsnew/v3.0.0.html

**Polars**

- Polars in Aggregate, April 2026, https://pola.rs/posts/polars-in-aggregate-apr26/
- Polars in Aggregate, December 2025, https://pola.rs/posts/polars-in-aggregate-dec25/
- pola-rs/polars issue 20947, the new streaming engine tracking issue, https://github.com/pola-rs/polars/issues/20947
- pola-rs/polars issue 26148, the Polars 2.0 release roadmap, https://github.com/pola-rs/polars/issues/26148

**Arrow**

- Apache Arrow 25.0.0 release, https://arrow.apache.org/blog/2026/07/10/25.0.0-release/
- Apache Arrow Go 18.6.0 release, https://arrow.apache.org/blog/2026/04/28/arrow-go-18.6.0/
- apache/arrow-go releases, https://github.com/apache/arrow-go/releases
- apache/arrow-go issue 184, unsupported cast to string_view from utf8, https://github.com/apache/arrow-go/issues/184
- apache/arrow-rs issue 5374, the StringViewArray and BinaryViewArray epic, https://github.com/apache/arrow-rs/issues/5374
- The Arrow PyCapsule interface, https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html

**Go dataframe ecosystem**

- go-gota/gota, https://github.com/go-gota/gota
- DataFrames in Go with gota, qframe, and dataframe-go, https://www.mungingdata.com/go/dataframes-gota-qframe/

**Benchmarks**

- duckdblabs/db-benchmark, https://github.com/duckdblabs/db-benchmark
- The Return of the H2O.ai Database-like Ops Benchmark, https://duckdb.org/2023/04/14/h2oai
- Updates to the H2O.ai db-benchmark, https://duckdb.org/2023/11/03/db-benchmark-update

**Bindings**

- PEP 703 and the free threaded build status, https://docs.python.org/3/howto/free-threading-python.html
- cgo pointer passing rules, https://pkg.go.dev/cmd/cgo
- Node-API, https://nodejs.org/api/n-api.html
- Bun N-API support, https://bun.sh/docs/api/node-api
- Deno FFI, https://docs.deno.com/runtime/reference/deno_namespace_apis/
