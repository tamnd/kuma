# Bindings for other languages

## Why bother

A dataframe engine that only Go programs can call has a ceiling on how useful it can be. The people with the data problems are mostly in Python, and the people building dashboards and internal tools around that data are mostly in TypeScript. If kuma is only reachable from Go, then the Go program has to be the whole application, and that is a much narrower bet than it needs to be.

There is also a selfish reason. Bindings are the best correctness test we will ever write. Once a Python binding exists we can run the same query through kuma, pandas and Polars in one process and compare the results, and that turns "we think our quantile is right" into a test that fails when it is not. The bindings and the benchmark repo in document 10 are the same piece of infrastructure viewed from two angles.

## The one design decision that matters

Every foreign function call has a cost, and every design failure in this area comes from making too many of them. The naive binding exposes `filter`, `group_by`, `agg` and so on as individual C functions, each one crossing the boundary, each one allocating and returning a handle. That works and it is slow, and worse, it is slow in a way that gets blamed on the engine rather than on the binding.

So the boundary is deliberately narrow, and it carries two things only.

**Data crosses as Arrow.** Specifically the Arrow C Data Interface, which is a pair of plain C structs, `ArrowSchema` and `ArrowArray`, with a release callback. No copying, no serialization, no format negotiation. Every language we care about already understands it: pyarrow, polars, duckdb and pandas 3.0 on the Python side, apache-arrow on the JavaScript side.

**Queries cross as a serialized plan.** The host language builds a query as a data structure in its own idiom, serializes the whole thing once, and hands it over. One call executes it. The host is doing the ergonomic work of building the query and kuma is doing the execution, which is the correct split because the ergonomics have to be native to feel right and the execution is the part that has to be fast.

The consequence is that a query which touches a hundred million rows crosses the FFI boundary about four times. That is the whole point.

```
Python or TypeScript                    Go
--------------------                    --

build query in native idiom
serialize to plan bytes  ------------>  kuma_plan_parse
                                        optimize
kuma_query_execute       ------------>  execute, streaming
                         <------------  ArrowArrayStream
iterate batches, zero copy
```

Our own protobuf plan format first, because we control both ends and it is a week of work. Substrait as a later interop goal, since being able to accept a plan produced by another engine is a real thing to want, but it is not worth blocking on.

## The C ABI

Go builds a shared library with `-buildmode=c-shared`, which produces `libkuma.so`, `libkuma.dylib` or `kuma.dll` plus a generated header. Everything else in this document sits on top of that one artifact.

### Handles, not pointers

cgo forbids passing a Go pointer to C in a way that outlives the call, because the garbage collector may move or free it. So nothing crosses as a pointer. Every object that lives across calls is an opaque integer handle, and the Go side keeps a table.

```go
var (
    handleMu  sync.RWMutex
    handles   = map[uint64]any{}
    handleSeq atomic.Uint64
)
```

The table is the only mutable global in the binding, which makes it the only place a concurrency bug can live, which is exactly where you want that risk concentrated. Handles are never reused, so a double free is a clean error rather than a use after free of somebody else's object.

### The surface

Small on purpose. This is close to all of it.

```c
// lifecycle
kuma_error*  kuma_init(void);
void         kuma_shutdown(void);
const char*  kuma_version(void);

// context and cancellation
uint64_t     kuma_context_new(int64_t timeout_ms);
void         kuma_context_cancel(uint64_t ctx);
void         kuma_context_free(uint64_t ctx);

// plans
uint64_t     kuma_plan_parse(const uint8_t* buf, size_t len, kuma_error** err);
char*        kuma_plan_explain(uint64_t plan);
void         kuma_plan_free(uint64_t plan);

// frames
uint64_t     kuma_frame_from_arrow(struct ArrowArrayStream* in, kuma_error** err);
kuma_error*  kuma_frame_to_arrow(uint64_t frame, struct ArrowArrayStream* out);
int64_t      kuma_frame_num_rows(uint64_t frame);
void         kuma_frame_free(uint64_t frame);

// execution
kuma_error*  kuma_execute(uint64_t ctx, uint64_t plan, struct ArrowArrayStream* out);
kuma_error*  kuma_execute_collect(uint64_t ctx, uint64_t plan, uint64_t* out_frame);

// io shortcuts, so a simple read does not need a plan
uint64_t     kuma_scan_parquet(const char* path, kuma_error** err);
uint64_t     kuma_scan_csv(const char* path, const uint8_t* opts, size_t len, kuma_error** err);

// errors
const char*  kuma_error_message(kuma_error*);
int32_t      kuma_error_code(kuma_error*);
void         kuma_error_free(kuma_error*);

// memory
void         kuma_string_free(char*);
```

Rules that apply to all of it. Every function that can fail takes a `kuma_error**` out parameter and returns null or zero on failure, so there is no sentinel value ambiguity. Nothing panics across the boundary, which means every exported function has a deferred recover that converts a panic into an error, because a Go panic unwinding into a Python interpreter is not a debuggable event. Every allocation has a named free function and the header documents which one. Handles are safe to use from multiple host threads.

The result of a query is an `ArrowArrayStream`, not a materialized array. That means the host can iterate batches while Go is still producing them, and a query whose result does not fit in memory still works.

### Known hazards

These are the things that go wrong when embedding a Go shared library, all of them documented up front because each one costs a day to rediscover.

**Signal handlers.** The Go runtime installs handlers for SIGSEGV and others and will fight with a host runtime that does the same. `os/signal.Ignore` at init, plus documentation for the case where a host really does need them.

**Fork.** A forked child inherits none of the Go runtime's threads and will hang the moment it touches the library. Python's `multiprocessing` must use the spawn start method. This gets a loud note in the Python README because the failure looks like a mysterious deadlock rather than an error.

**The GIL.** The Python binding releases the GIL for the duration of `kuma_execute` and reacquires it after. Without this, a kuma query blocks every other Python thread and the whole point of the exercise is lost.

**Interrupts.** `kuma_context_cancel` is what makes Ctrl-C work. The Python binding installs a SIGINT handler that cancels the active context, so a runaway query is interruptible instead of requiring a kill.

**Binary size.** A Go shared library carrying the runtime and the tzdata is on the order of thirty to fifty megabytes before compression. That is acceptable for a wheel and irritating for an npm package. `-ldflags="-s -w"` and dropping `time/tzdata` in favour of the system database on platforms that have one bring it down. It should be measured and tracked in CI, because it only ever grows.

**Thread local errors.** The last error is stored per host thread, not globally, so two threads failing at once do not overwrite each other.

## Python

The package is `kuma` on PyPI, from the `tamnd/kuma-py` repository.

### What it looks like

The API is intentionally close to Polars, because that is the shape Python users already have in their fingers for a lazy, expression based dataframe library, and gratuitous novelty here buys nothing.

```python
import kuma as km

df = (
    km.scan_parquet("trades/*.parquet")
      .filter((km.col("price") > 100) & (km.col("side") == "BUY"))
      .group_by("symbol", km.col("ts").dt.truncate("1m"))
      .agg(
          (km.col("price") * km.col("qty")).sum().alias("volume"),
          km.col("price").quantile(0.99).alias("p99"),
          km.count().alias("n"),
      )
      .sort("volume", descending=True)
      .head(20)
      .collect()
)
```

`km.col("price") > 100` builds a Python expression object. Nothing crosses the boundary until `.collect()`, at which point the whole plan is serialized once and executed.

### Interop is the actual selling point

The returned object implements the Arrow PyCapsule interface, `__arrow_c_array__` and `__arrow_c_stream__`. That protocol is what pandas 3.0, Polars, DuckDB and pyarrow all agreed on, and implementing it means everything below is zero copy and none of it required us to write a pandas specific code path.

```python
df.to_pandas()        # pandas 3.0, Arrow backed, no copy
df.to_polars()        # no copy
pl.DataFrame(df)      # works directly, via the protocol
duckdb.sql("select * from df")   # works directly

km.from_pandas(pdf)   # the other direction, also no copy
km.from_arrow(tbl)
```

This is what turns a partial implementation into a usable one. Anything kuma does not do yet, hand to DuckDB and hand back.

### Binding mechanics

cffi in ABI mode, or ctypes. Not cgo generated CPython extension modules and not pybind11.

The reason is the free threaded build. Python 3.13 introduced it, 3.14 made it officially supported in October 2025, and by 2026 it is what performance minded users are running. A binding that goes through the stable C ABI works there without modification. One that compiles against CPython internals needs porting and needs a separate wheel per Python version. Going through the plain C ABI means one wheel per platform covers every interpreter version, which cuts the build matrix from around forty wheels to six.

Wheels for manylinux x86-64 and aarch64, macOS arm64 and x86-64, and Windows x86-64. Build with cibuildwheel. Type stubs shipped in the package so that mypy and editors see real signatures, which matters more than usual here because the objects are opaque handles with no introspectable structure.

### Testing

`kuma-py` is where the conformance suite from document 06 actually runs, because it is the only place all three libraries can be loaded into one process. Every checklist item gets a test that runs the same operation through kuma and through pandas on the same input and compares, with a documented tolerance for the float cases. When a behaviour deliberately differs, the test asserts the difference rather than skipping.

## TypeScript, Node, Bun and Deno

The package is `@tamnd/kuma` on npm, from the `tamnd/kuma-js` repository. One package covers all three runtimes.

### How one package covers three runtimes

Node, Bun and Deno all implement N-API. Bun has supported it since 1.0 and Deno since 1.38. So the primary path is a single N-API addon that loads on all three, and there is no per runtime native build.

There is a secondary path using each runtime's own FFI, meaning `koffi` on Node, `bun:ffi` on Bun and `Deno.dlopen` on Deno. That path calls `libkuma` directly with no addon at all, which is useful for Deno users who do not want a native module in their dependency tree and for anyone doing `deno run` without a build step. It is genuinely less code, since the C ABI is already narrow, and it costs a small amount of per call overhead which does not matter given how few calls there are.

The loader picks at import time:

```ts
// runtime detection, in order of specificity
const impl =
  typeof Bun !== "undefined"  ? await import("./ffi/bun.js")  :
  typeof Deno !== "undefined" ? await import("./ffi/deno.js") :
                                await import("./napi/node.js");
```

Prebuilt binaries per platform published as optional dependencies, the pattern esbuild uses, so installing does not require a toolchain.

### What it looks like

```ts
import * as km from "@tamnd/kuma";

const df = await km
  .scanParquet("trades/*.parquet")
  .filter(km.col("price").gt(100).and(km.col("side").eq("BUY")))
  .groupBy("symbol", km.col("ts").dt.truncate("1m"))
  .agg(
    km.col("price").mul(km.col("qty")).sum().as("volume"),
    km.col("price").quantile(0.99).as("p99"),
    km.count().as("n"),
  )
  .sortDesc("volume")
  .head(20)
  .collect();

for await (const batch of df.batches()) {
  // arrow-js RecordBatch, backed by the same memory Go wrote
}
```

Everything that touches the engine is async and runs on a worker thread, so a query never blocks the event loop. This is not optional. A Node process that stops serving requests for four seconds because someone ran an aggregation is a bug report, and it is the failure mode every naive native binding has.

### TypeScript types from the Go struct

`kumagen` already reads a schema from Parquet, CSV or SQL to generate the Go struct in document 03. Emitting a TypeScript interface from the same schema is nearly free, and it gives the JavaScript side something close to the typed column experience.

```
kumagen -parquet trades/x.parquet -type Trade -ts trade.ts
```

```ts
export interface Trade {
  symbol: string;
  price: number;
  qty: bigint;
  side: string;
  ts: Date;
}

export const TradeCols = {
  symbol: km.strCol<Trade>("symbol"),
  price:  km.f64Col<Trade>("price"),
  qty:    km.i64Col<Trade>("qty"),
  side:   km.strCol<Trade>("side"),
  ts:     km.timeCol<Trade>("ts"),
} as const;
```

TypeScript's structural typing gets us most of the way. `TradeCols.price.contains("x")` is a type error because `f64Col` has no `contains`. It is weaker than the Go version since it disappears at runtime, but it catches the same class of mistake at the same time, which is when you are typing.

Note `bigint` for int64. JavaScript numbers lose precision above 2^53 and silently returning a wrong integer is worse than an awkward type. Arrow JS already made this decision.

### Interop

The result is an arrow-js `Table` or `RecordBatchReader` with no copy. That connects to Observable Plot, Perspective, DuckDB-Wasm and anything else that speaks Arrow, which is most of the JavaScript data ecosystem.

## WebAssembly

Optional and worth doing, but not before 1.0.

Go compiles to `wasip1` and to `js/wasm`. A kuma build that runs in a browser means notebooks and dashboards that query Parquet client side with no server. DuckDB-Wasm proved there is real demand for this.

The catches are real. Binary size, since a Go wasm binary starts around ten megabytes and needs Brotli to be tolerable. Threads, since `GOMAXPROCS` is one under wasm and the parallel executor is a no-op. SIMD, since Go 1.27 added wasm 128 bit support to `archsimd` and it needs the same kernel treatment as the native targets. And the Arrow C Data Interface has to be emulated over the wasm linear memory rather than passed as real pointers.

It shares all the plan serialization work with the other bindings, so the marginal cost after those exist is small. That is the argument for sequencing it last rather than dropping it.

## Repository layout

Separate repositories, not a monorepo, because the release cadences genuinely differ and because a Python user should not need to clone Go source.

| Repository | Contents | Published to |
|---|---|---|
| `tamnd/kuma` | the Go library, plus `kuma/capi` which builds the shared library | pkg.go.dev |
| `tamnd/kuma-py` | the Python package and the pandas conformance suite | PyPI as `kuma` |
| `tamnd/kuma-js` | the N-API addon, the FFI adapters and the TypeScript types | npm as `@tamnd/kuma` |
| `tamnd/kuma-bench` | benchmarks against pandas and Polars, see document 10 | nothing, it publishes results |

The plan protobuf definition lives in `tamnd/kuma` and is vendored into the others at release time, since it is the one thing all four must agree on exactly.

Version numbers are independent, but every binding release records the `libkuma` version it was built against, and `kuma_init` rejects a plan produced by an incompatible generator rather than misinterpreting it.

## Order of work

The C ABI is the only hard dependency, and it cannot be built before there is a plan serializer and a working executor, which means M3. Everything after that is parallel work that does not touch the Go engine.

1. Plan serialization and `kuma/capi`, once the lazy engine exists.
2. Python, next, because that is where the conformance suite runs and it starts paying for itself immediately.
3. TypeScript, once the Python binding has shaken the bugs out of the C layer.
4. WebAssembly, after 1.0.

Precise milestone placement is in document 08.
