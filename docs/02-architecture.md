# Architecture

## Layers

Four of them, with dependencies pointing strictly downward.

```
L3  Typed         Typed[T], CollectInto[R], AggFold        Go 1.27 generic methods
L2  Lazy          LazyFrame, Expr, plan, optimizer         the pandas replacement
L1  Eager         Frame, Series, ChunkedArray              notebooks, tests, escape hatch
L0  Storage       Buffer, Bitmap, Array, DataType          Arrow layout
    Kernels       dispatch table over simd and scalar
```

L3 adds no capability that L2 does not have. It is a facade, and that is deliberate, because it means the typed surface can be reshaped without touching the engine underneath it. The one thing to be careful about is that the typed columns from document 03 are not part of L3. They are part of L2, they exist from M3, and the engine is committed to them. L3 is only the reflection based bridge between frames and Go structs.

## Memory

Arrow columnar layout, with our own buffer types rather than arrow-go's.

```
Array
  DataType          the dtype descriptor
  len, nullCount    nullCount == 0 is the hot path
  validity Bitmap   one bit per value, 1 means valid, nil when nullCount is 0
  data     Buffer   primitive values, or offsets, or views
  children []Array  for list, struct and map
```

A `Buffer` is a `[]byte` with a 64 byte aligned backing allocation and an optional pool reference. Alignment is not free. Go's allocator guarantees 8 or 16 byte alignment, so an aligned buffer is over allocated by 63 bytes and resliced. It is worth paying for because it enables aligned loads on AVX-512 and because it stops every kernel from straddling a cache line at the boundary.

There is no reference counting. arrow-go's `Retain` and `Release` discipline is a tax that Go programmers will forget to pay, and the garbage collector already exists. Buffers are ordinary Go memory. A `sync.Pool` keyed by size class is available to the executor for scratch buffers whose lifetimes are provably scoped to a morsel, which is the only place manual lifetime management is safe. Off heap and memory mapped buffers are reserved for the out of core work in M9.

### Strings

The default representation for `String` and `Binary` is the Arrow variable size binary view layout. Each element is a 16 byte view struct holding the length, a 4 byte inline prefix, and then either the remaining 8 bytes inline for strings of 12 bytes or fewer, or a buffer index and offset pair for longer ones.

This is the default rather than an option, for three reasons.

Equality and prefix comparison resolve from the inline prefix alone for the overwhelming majority of real data, which means no pointer chase and no cache miss on the common path.

Filtering and sorting become dense scans over fixed width 16 byte records, and fixed width scans vectorize.

arrow-go's compute support for views is thin, which makes this a place where a Go engine can lead rather than follow. Being the fastest string implementation in the Go ecosystem is a more defensible position than being a second implementation of something that already exists.

The classic `Utf8` layout of offsets plus data is kept for IPC compatibility and converted at the boundary. Low cardinality columns additionally get dictionary encoding, which turns string group by keys into integer keys and skips hashing entirely, and which is usually a bigger win than any amount of SIMD applied to the string path.

### Chunking

A column is a `ChunkedArray`, an ordered slice of `Array` chunks sharing a dtype.

The chunk is four things at once. It is the unit of vectorization, since kernels operate on one chunk. It is the unit of parallelism, since a morsel is one chunk or a slice of one. It is the unit of the null fast path, since `nullCount` is per chunk. And it is the unit of append, so growth never reallocates a whole column.

The target size is whatever keeps the working set in L2 cache, which is somewhere between 1024 and 8192 rows depending on how wide the rows are. The concrete default is determined by benchmark at M4 rather than guessed now, because guessing it now would just mean carrying an arbitrary number around until somebody measured it anyway.

## Types

```
Primitive   Bool Int8 Int16 Int32 Int64 Uint8 Uint16 Uint32 Uint64 Float32 Float64
Decimal     Decimal128(precision, scale)  Decimal256
Temporal    Date32 Date64 Time32(u) Time64(u) Timestamp(u, tz) Duration(u) Interval
Binary      String Binary in view layout, LargeString LargeBinary, FixedSizeBinary(n)
Nested      List(T) LargeList(T) FixedSizeList(T, n) Struct(fields) Map(K, V)
Encoded     Dictionary(indexType, valueType)
Null
```

`DataType` is an interface. It needs runtime polymorphism and it is never the receiver of a generic method, so the restriction from document 01 does not bite here. `Schema` is an ordered slice of `Field`, where a field is a name, a dtype, a nullable flag and metadata.

**Nulls are a validity bitmap and are distinct from NaN.** `NaN != NaN` and NaN is a valid value; null propagates. Aggregations skip nulls by default, matching pandas `skipna`, with an option to change that. This is the model pandas only reached in 3.0 by way of Arrow, and starting there avoids the whole family of bugs where an integer column silently becomes a float because something was missing.

**There is no implicit upcasting.** Adding an int64 column to a float64 column is an error unless one side is a literal or the user writes a cast. pandas' silent upcasting is a correctness hazard and Polars' strictness is the right call. The error arrives at plan time, before any data is read.

## Expressions

An expression is a wrapper around an immutable node pointer.

```go
type Expr struct{ n *node }
```

It is a concrete struct rather than an interface, and it has to be, because generic methods are only allowed on concrete types. That single constraint from Go 1.27 shapes this entire layer.

The node kinds are `Column`, `Literal`, `Unary`, `Binary`, `Function`, `Cast`, `Ternary`, `Agg`, `Window`, `Sort`, `Alias`, `Wildcard`, `Exclude` and `DTypeSelector`.

Nodes are immutable and shared further than one level. A step is looked up in a table of every node that anything still holds, so two expressions that say the same thing are the same node however they were written and whoever wrote them, and common subexpression elimination is a pointer comparison rather than a tree walk. The table holds each node weakly, so an expression built for one query and then dropped is collected like any other garbage instead of being kept alive by the table that made it. Two things are deliberately left out of it: a NaN literal, because NaN is not equal to itself and an entry under such a key can never be found or removed again, and a literal of a type no column can hold, because hashing one would panic where the caller deserves an error.

Errors are carried inside the node. A malformed expression poisons everything built on top of it and surfaces exactly once, at `Collect`. This is what pays for the single error return in the chaining API from document 04. Expression building is the one genuinely allocation heavy path in the library, and Go 1.27's cheaper small object allocation helps more here than anywhere else.

## Plan and optimizer

```
LogicalPlan := Scan | Filter | Project | Aggregate | Join | Sort | Limit | Distinct
             | Union | Explode | Pivot | Unpivot | Window | Sink
```

The passes below run to fixpoint.

| Pass | Effect |
|---|---|
| Projection pushdown | The scan reads only the columns actually referenced. The single biggest Parquet win, and usually the largest speedup in the whole optimizer. |
| Predicate pushdown | Filters sink to the scan, enabling Parquet row group and bloom filter skipping. |
| Slice pushdown | `Head(n)` bounds how much the scan decodes. |
| Common subexpression elimination | A pointer identity check over shared nodes. |
| Expression simplification | Constant folding, `x AND true`, and De Morgan normalization so that more predicates become pushable. |
| Type coercion | Explicit cast insertion, failing at plan time rather than partway through the data. |
| Join reordering | Driven by cardinality estimates, so the smaller side builds the hash table. |
| Partition pruning | On Hive partitioned scans, prune by path and pre partition group bys and joins on the partition keys. The metadata is carried from M2, the pass lands at M9. |
| Aggregate pushdown | A count over a Parquet scan is answered from file metadata without reading anything. |
| Fusion | Adjacent elementwise expressions compile into one pass over the chunk. |

`Explain` prints the optimized plan and `Profile` adds per operator wall time, rows in and out, and bytes read.

These are shipping features rather than debugging tools, and they exist from M3. There are two reasons and both matter. Users need to see why a query is slow, and pandas gives them nothing at all here, so this is one of the larger developer experience advantages available. And internally, every optimization we claim becomes a test that asserts on the plan shape, rather than a claim in a commit message that nobody can check.

## Execution

Vectorized and morsel driven. Operators process one chunk at a time. A scheduler hands morsels to a fixed worker pool sized to `GOMAXPROCS`, with work stealing.

```
Scan --> [morsel queue] --> worker 0 --+
                            worker 1 --+--> partitioned aggregate --> merge --> Sink
                            worker 2 --+
```

Six rules govern this layer.

**Late materialization.** A filter produces a selection vector, and compaction happens only when a downstream operator actually needs dense data. This avoids copying columns that are about to be dropped.

**The null fast path branches once per chunk**, on `nullCount == 0`, never per element. Most real columns are dense, so the unmasked kernel is the common case and the masked one is the exception. Getting this backwards is easy, because the masked version is the more general one and therefore feels like the natural default, and it costs more than any individual kernel gains.

**Aggregation is partitioned.** Each worker builds a private hash table over a radix partition of the key space, so there is no contention on a shared table and the merge is a concatenation rather than a reduction.

**`context.Context` runs through everything**, with cancellation checked at morsel boundaries. pandas has never had this, and it is a genuine reason to reach for Go: a query serving an HTTP request should die when the request does.

**Deterministic output is the default**, with the faster nondeterministic group ordering available as an option. Nondeterministic output makes tests unwritable, and a library whose tests are annoying to write ends up with bad tests.

**Goroutine hygiene is tested rather than assumed.** Go 1.27's `goroutineleak` profile runs in the executor test suite and every cancelled query must leave zero blocked goroutines. A morsel executor with worker pools and channel fan in is exactly the shape that leaks on cancellation. Executor goroutines also carry pprof labels naming the operator and morsel, so that a traceback is readable.

## Streaming and out of core

This is M9, but the operator interface has to accommodate it from M4, so it is designed now.

Polars' history is instructive here. They built a correct in memory engine first, added streaming second, and made operators without a streaming implementation fall back transparently to the in memory path. That fallback is the design decision that let the streaming engine ship while it was still incomplete, which is the only way a project of this size ever ships anything.

Three things to hold open in the operator interface from M4. Operators declare whether they can stream, and the planner chooses per operator. Group by, join and sort need spillable sinks over a memory manager with a disk budget, along the lines of Polars' `POLARS_OOC_DISK_BUDGET_MB`. And there needs to be a sink API, meaning `SinkParquet`, `SinkCSV` and `SinkIPC`, that never materializes the full result.

## What is deliberately not built

| Not built | Why |
|---|---|
| An implicit index with automatic alignment | pandas' single largest source of surprise. An explicit optional index is provided instead, as described in document 06, and joins always take explicit keys. |
| `inplace=True` mutation | Frames are immutable. pandas 3.0 went the same way with Copy-on-Write. |
| `dtype=object` | There is no column of boxed anything, ever. Use a struct column. |
| NaN as the missing value sentinel | A validity bitmap. NaN is a valid float. |
| Silent dtype upcasting | Explicit casts, with type errors at plan time. |
| MultiIndex as a default | Group keys are ordinary columns. A compound index is available when explicitly asked for. |
| Reference counted buffers | The garbage collector exists, and `Retain` and `Release` is a tax users will get wrong. |
| Building on arrow-go's compute kernels | Those kernels are the thing we exist to provide, and their coverage is thin enough that we would be blocked constantly. |
