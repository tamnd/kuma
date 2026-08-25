# Milestones

Sizing is in focused person weeks for one experienced engineer. These are for sequencing, not commitments.

There are three tracks. The **engine track**, M0 through M10, is roughly 62 weeks and is what produces a defensible 1.0. The **bindings track**, B1 through B4, is roughly 15 weeks and runs in parallel once M3 exists. The **benchmark track** is not a milestone at all, it is a thing that runs continuously from M1 onward, and the reasoning for that is in document 10.

M0 through M4, about 19 weeks, produces something independently useful, and that is the point at which the project either proves itself or should be stopped.

Every milestone has an exit criterion you can check mechanically. "Feels done" is not an exit criterion.

## One thing that changed

Making typed columns the default rather than an optional layer moves Go 1.27's generic methods from late and optional to early and load bearing. The original plan deferred all generic method usage to M6 specifically so that the newest language feature in the toolchain was not underneath the risky milestones.

That hedge is gone. The typed expression API needs generic methods from M3 onward, because `Agg[R]` and `Select[R]` are methods that introduce a type parameter the receiver does not have, and that is the whole reason the API is shaped this way.

The mitigation is to prove the pattern in M1, which is small and low risk, before committing the query engine to it in M3. If generic methods turn out to have sharp edges we have not found yet, M1 is where we find out, and the cost of discovering it there is a week rather than a quarter.

# The engine track

## M0, Foundations, 3 weeks

The layer everything else sits on. Get it wrong and every later milestone pays for it.

Build the type lattice from document 02, `Schema`, `Field` and the coercion rules. Build `kuma/bitmap` with get, set, popcount, the boolean operations, append, slice and a builder. Build `Buffer` with 64 byte aligned allocation and a size class pool. Build `Array` and `ChunkedArray` with construction, slicing, appending and null counting. Implement the StringView layout with its 16 byte inline prefix representation.

Done when every dtype round trips through construction, slice, append and read; when the bitmap operations survive ten million fuzz cases against a `[]bool` reference model; and when `go vet` and `staticcheck` are clean with 90 percent statement coverage on `bitmap` and `dtype`.

The StringView representation is the risk here. Getting the short versus long discriminant wrong is a silent data corruption bug rather than a crash. Write the fuzz test before the implementation.

## M1, Eager frame, scalar kernels, CSV, 7 weeks

Correctness first, with no SIMD anywhere. The scalar kernels written here become the specification that M5's vectorized kernels are checked against, so they need to be obviously correct rather than fast. Resist optimizing.

Build `Frame` and `Series`, then select, cast, filter, sort with multiple keys and null placement, and `Head`, `Tail`, `Slice` and `Take`. Add group by with the basic aggregations, meaning sum, mean, min, max, count, first, last, std, var, median, quantile and distinct count. Add the joins: inner, left, right, outer, semi, anti and cross. Add `Concat` and `HStack`, and the null handling functions. Write the CSV reader and writer with schema inference, and the pretty printer from document 04.

Also in this milestone, because it is small and because M3 depends on the pattern working: the typed column handle types, the typed expression types, and the first version of `kumagen` generating a `Cols` variable from a tagged Go struct.

Build `kuma/kumatest` with frame equality and a readable diff.

**Stand up `tamnd/kuma-bench` in this milestone.** It can only measure CSV reading and a handful of aggregations at this point and that is fine. The reason to build it now rather than at M5 is that by M5 there will be a year of history to check the SIMD claims against, and a benchmark harness created at the end of a project only ever proves what its author already believed.

Done when the db-benchmark group by and join queries produce results identical to DuckDB at one and ten million rows, when the harness forces materialization the way the official suite does for DuckDB and ClickHouse, and when CI publishes timings on every commit from here on. Done when a query written with typed handles and one written with strings produce the same plan, checked by a test. Done when CSV reading beats pandas `read_csv` with the PyArrow engine, and when the basic aggregations beat pandas at ten million rows.

## M2, Arrow interop and Parquet, 4 weeks

Deliberately early, because this is the milestone that makes every later gap survivable.

Implement the Arrow C Data Interface in both directions with no copying, the IPC file and stream formats, and a bridge to arrow-go that converts at the boundary only and never builds on its compute kernels. Write the Parquet reader with projection and predicate pushdown, row group skipping and bloom filter skipping, and the writer with dictionary encoding and statistics. Add Hive partitioned dataset scanning: parse the paths, expose partition columns as virtual columns, and carry the partition metadata on the scan node even though nothing will use it until M9. Add NDJSON on top of `encoding/json/v2`.

Extend `kumagen` to read a schema out of a Parquet file, an IPC file or a CSV sample and emit the Go struct, so the struct is derived from real data rather than typed by hand.

Done when every dtype round trips through pyarrow via the C Data Interface with zero copies, verified by checking buffer pointer identity rather than by eye. Done when TPC-H SF1 loads from Parquet with column projection verified by a decoded bytes counter. Done when a kuma aggregation over TPC-H matches DuckDB's answer exactly. Done when Parquet read is within 2x of Polars.

The reason this comes before the lazy engine is that from here on, anything we have not implemented can be handed to DuckDB or pyarrow without copying. That converts incompleteness from a blocker into an inconvenience, and it is the single highest leverage risk reduction in the plan. It is also the mechanism the bindings in document 07 are built on, so this milestone is a prerequisite for the entire bindings track.

## M3, Lazy engine, typed expressions and optimizer, 7 weeks

The milestone that defines the product, and the most complex one.

Build the expression node types, immutable and structurally shared so that common subexpression elimination is a pointer comparison. Build `LazyFrame[S]` with `Filter`, `With`, `Select[R]`, `GroupBy`, `Agg[R]`, `Join[R, Out]`, `SortBy`, `Head`, `Distinct` and `Explode`. Build the logical plan, plan time type checking and `Validate`.

Write the optimizer passes: projection pushdown, predicate pushdown, slice pushdown, common subexpression elimination, constant folding, type coercion insertion and expression fusion.

Ship `Explain` and `Profile` in this milestone, not later. Build the error model: poisoned nodes, did you mean suggestions, and the plan position in the message.

Add the plan serialization format, meaning the protobuf definition and the parser, at the end of this milestone. It is a week of work here and it unblocks the entire bindings track, which otherwise cannot start.

Done when every optimizer pass has a test that asserts on the `Explain` output, because the plan shape is the assertion. Done when pushdown is verified by instrumented counters for columns decoded and row groups read, not by timing. Done when every eager operation from M1 is reachable through the lazy API with identical results, checked by a shared test suite parameterized over both paths. Done when the typed API is within five percent of the dynamic API on the benchmark suite. Done when a plan round trips through serialization unchanged, checked by comparing `Explain` output on both sides.

That dual path test suite is the main defence against the risk in this milestone, which is that a wrong query plan still returns a plausible looking result. It has to be written alongside the optimizer rather than after it.

## M4, Parallel execution, 4 weeks

Build the morsel driven scheduler with a worker pool sized to `GOMAXPROCS` and work stealing. Build partitioned hash aggregation where each worker owns a private table over a radix partition of the key space, so the merge is a concatenation rather than a contended reduction. Parallelize the joins and the sort. Add late materialization through selection vectors. Thread `context.Context` cancellation through to morsel boundaries. Put pprof labels on the executor goroutines, tagged by operator and morsel, so that Go 1.27 tracebacks are readable.

Determine the chunk size by benchmark in this milestone rather than guessing earlier.

Done when group by scales at least 12x on 16 cores at 100 million rows, and at better than 60 percent efficiency on 8 cores. Done when `go test -race` is clean across the suite. Done when the `goroutineleak` profile, which graduated to a regular profile in Go 1.27, shows zero blocked goroutines after every cancelled query test. A morsel executor with worker pools and channel fan in is exactly the shape that leaks on cancellation, so this needs to be an assertion rather than an assumption. Done when cancellation latency is under 50 milliseconds at 100 million rows.

## M5, SIMD kernels, 5 weeks

Build the kernels in the order given in document 05, which is deliberately not easiest first: CSV and NDJSON parsing, comparison to bitmap, filter and compaction, aggregations, bitmap operations, string comparison and search, sort, group by hashing, and elementwise arithmetic last. Build the dispatch table, the build tagged portable, amd64, arm64 and scalar variants, and runtime feature detection.

Done when differential fuzzing against every scalar twin passes a hundred million cases with zero failures, covering all tail lengths from zero to twice the vector width, all null and no null inputs, and NaN, infinity and denormal floats. Done when the full CI matrix from document 05 is green including a genuinely exercised AVX-512 path, which needs Intel SDE or a Zen 4 or Ice Lake or newer runner, because untested vector code is worse than no vector code. Done when kernels one through six are at least 4x faster than the M1 scalar baseline, measured against the recorded M1 numbers in `kuma-bench` rather than against a fresh run. Done when everything still builds and passes with no `GOEXPERIMENT` set.

The risk is that the `simd` API is unstable and its amd64 surface already broke once during the 1.27 cycle. Keeping it inside `kuma/kernel` is what limits the cost of the next break to an afternoon. Do not let a `simd` type reach an exported signature.

## M6, Struct mapping and extension points, 3 weeks

Close the loop between Go types and frames, and open the library up to users.

Build `Rows(ctx)` returning `[]S`, `FromStructs[T]`, `CollectInto[R]`, `Reduce[T]` and `ValueAt[T]`. Finish `kumagen` with SQL table introspection, nested struct flattening, TypeScript output for document 07, and deterministic output. Build `AggFold[Acc, Out]` for user defined aggregations with a real accumulator type. Build the dtype keyed function registry that user kernels plug into.

That registry is worth calling out. Because Go 1.27 does not allow generic methods on interfaces and does not allow a generic method to satisfy an interface method, an extension point cannot be an interface with a generic method on it. It has to be a registry of concrete function values keyed by dtype. This shape has to be decided before M0 in practice, since retrofitting it means rewriting every extension surface, but this is the milestone where it gets built out properly.

Done when a struct covering every supported dtype, including nested structs and slices, round trips from `[]T` to a frame and back. Done when there is a compile failure test suite, driven by `go/types`, asserting that the mistakes listed in document 03 actually fail to compile.

## M7, pandas parity, first tier, 10 weeks

The largest milestone by volume and the smallest by risk. It parallelizes well across contributors.

Everything marked M7 in document 06: the complete string namespace at around fifty methods, the datetime components and predicates, categoricals, the nested data accessors, rolling and expanding and exponentially weighted windows, `Over` window functions, the reshaping operations, the optional explicit index and everything that depends on it, `Rank`, `Cut`, `QCut`, `Factorize`, `Interpolate`, `Replace`, `TopK`, `Describe`, `Corr`, `Cov`, `Distinct` and `IsDuplicated`.

Done when every M7 checkbox in document 06 is ticked, with a runnable example test behind each one. Done when the differential test against pandas 3.0 over generated frames passes for every one of them, with the documented divergences being the only permitted differences. Done when the full db-benchmark suite runs green at 0.5 GB and 5 GB and the results are published whatever they say.

The string namespace is where a partial implementation reads as a toy, so ship it whole. RE2's lack of backreferences and lookaround is a real user visible divergence from pandas and needs documenting prominently rather than being discovered in a bug report.

## M8, Time series, 6 weeks

Build `Resample`, `AsFreq` and `Upsample`. Build `JoinAsof` with backward, forward and nearest directions, `by` grouping and tolerance, plus `JoinOrdered`. Build `RollingBy` over durations. Implement full IANA timezone support with DST aware arithmetic, `DateRange` and `BDateRange`, `kuma/calendar` covering the forty odd pandas date offsets plus business days and holidays, `AtTime`, `BetweenTime` and `InferFreq`, the period and interval types, and the sorted key fast paths.

Done when every M8 checkbox in document 06 is ticked. Done when `JoinAsof` matches `pandas.merge_asof` across a generated suite covering every combination of direction, tolerance and `by`. Done when the DST correctness suite passes across at least ten zones, with the explicit earliest, latest or raise policy asserted for ambiguous and nonexistent local times.

This is the milestone that earns adoption from finance and observability users specifically, because as of joins and DST correct resampling are simultaneously where pandas is most used and most error prone.

## M9, Streaming and out of core, 8 weeks

Polars took roughly three years to get here and shipped it incrementally behind per operator fallback. Copy that approach.

Add a streamable declaration to the operator interface, with transparent fallback to the in memory engine for operators that do not stream. Build spillable sinks for group by, join and sort over a memory manager with a disk budget. Build `SinkParquet`, `SinkCSV`, `SinkIPC` and `SinkNDJSON` that never materialize the whole result. Add partition aware planning: prune Hive partitions, pre partition group bys and joins on partition keys, and rewrite inner joins as unions of partition filtered joins.

Done when db-benchmark at 50 GB completes within a memory budget smaller than the dataset, and when every operator either streams or falls back with the choice visible in `Explain`.

Highest effort, least parity value. Ship 1.0 without it if the calendar demands, since nothing earlier depends on it.

## M10, SQL, ADBC and 1.0, 7 weeks

Build `kuma/sql`, parsing into the same logical plan so the optimizer is shared, plus `kuma.SQLExpr` for `query` and `eval` parity, `kuma/sqlio` over `database/sql`, and an ADBC driver. Add the remaining IO formats from document 06 that are marked M10, meaning ORC, fixed width, Excel, HTML and XML. Then do the API review and freeze, complete the godoc with runnable examples throughout, and write the migration guide for pandas users organized by pandas function name.

Done when all 22 TPC-H queries run correctly at SF10 with published timings against DuckDB and Polars. Done when the public API is frozen, every exported symbol is documented, every package has a runnable example, and `apidiff` is gating CI against the 1.0 baseline. Done when every checkbox in document 06 outside the explicitly post 1.0 sections is ticked.

# The bindings track

Runs in parallel with M4 onward, once M3 has produced a plan serializer. It touches almost none of the engine code, so it is the natural place to put a second contributor. Full design is in document 07.

## B1, the C ABI, 4 weeks

Build `kuma/capi`: the handle table, the exported functions listed in document 07, panic recovery on every boundary crossing, thread local error storage, context creation and cancellation, and the Arrow C Data Interface as the only data path. Set up `-buildmode=c-shared` builds for linux amd64 and arm64, darwin amd64 and arm64, and windows amd64.

Done when a C test program builds a plan, executes it, and iterates the resulting `ArrowArrayStream` with no copies. Done when every exported function has a test that forces a panic inside it and asserts that an error comes back instead. Done when a leak test running a hundred thousand query cycles shows flat memory. Done when the shared library size is recorded in CI and has a threshold.

Depends on M3.

## B2, Python, 4 weeks

Build `tamnd/kuma-py`: the expression and lazy frame classes in Python, plan serialization, the cffi ABI mode binding, the Arrow PyCapsule protocol in both directions, GIL release around execution, SIGINT to context cancellation, type stubs, and cibuildwheel across six platform wheels.

Then move the pandas conformance suite from document 06 into this repository, because this is the only place where kuma and pandas can be loaded into one process and compared directly.

Done when the example in document 07 runs. Done when `to_pandas`, `to_polars` and passing the object straight to DuckDB all work with no copy, verified by pointer identity. Done when Ctrl-C interrupts a running query. Done when the conformance suite runs in CI against pandas 3.0 and reports a pass rate per checklist section.

Depends on B1. Wants M7 to be underway for the conformance suite to be interesting.

## B3, TypeScript, Node, Bun and Deno, 4 weeks

Build `tamnd/kuma-js`: the N-API addon, the three FFI adapters for koffi, `bun:ffi` and `Deno.dlopen`, runtime detection at import, the expression and lazy frame classes in TypeScript, worker thread offload so the event loop never blocks, arrow-js integration, `kumagen` TypeScript output, and prebuilt binaries published as optional dependencies.

Done when the same example runs unchanged on Node, Bun and Deno. Done when a four second query does not block the event loop, asserted by a concurrent timer test. Done when the result is an arrow-js table sharing memory with Go rather than a copy. Done when `bigint` is used for every 64 bit integer column.

Depends on B1. Best done after B2, since B2 will have found the bugs in the C layer.

## B4, WebAssembly, 3 weeks, post 1.0

A `wasip1` and `js/wasm` build, size optimization with Brotli, the Arrow C Data Interface emulated over linear memory, and the single threaded execution path. Explicitly after 1.0.

# Dependencies

```
M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M9
                   |     |     |
                   |     +-----+---> M10
                   +---> M6 -> M7 -> M8
                   |
                   +---> B1 -> B2 -> B3 -> B4

kuma-bench: continuous from M1
```

M6, M7 and M8 are surface area. M5 and M9 are performance. B1 through B4 are ecosystem. All three groups are independent of each other after M3, so with more than one contributor they run in parallel, and they need different skills, which makes them a natural split.

# Four points to stop and reassess

After M2: does the Arrow bridge work well enough that being incomplete is survivable? If not, stop, because the whole plan is built on that assumption.

After M4: is the engine within 4x of Polars on db-benchmark before any SIMD work? If not, the problem is architectural and M5 will not rescue it. The numbers to answer this with are already in `kuma-bench` because it has been running since M1.

After B2: does anyone install the Python package? It is the cheapest possible test of whether there is demand, because it reaches an audience thousands of times larger than the Go one.

After M7: is anyone actually using it? Parity is expensive, and without users M8 through M10 is speculative.

# Ordering principles

Correctness before speed. M1's scalar kernels are the specification that M5 is checked against, and reversing that order leaves nothing to check against.

Interop early. M2 comes before the lazy engine because it makes every subsequent gap survivable, and because it is the foundation the bindings sit on.

Benchmarks from M1, continuously, in their own repository, so that a regression is attributable to a commit instead of being discovered at M5.

Prove risky language features on small things first. Generic methods are one release old and now underpin the API from M3 onward, so M1 exists partly to find out whether that is a problem while it is still cheap to change course.

Surface area parallelizes and engine work does not. Schedule contributors accordingly.
