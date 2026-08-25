# Benchmarking, and the kuma-bench repository

## Why this is a separate repository

`tamnd/kuma-bench` exists because comparing against pandas and Polars means installing pandas and Polars, and that must never become a condition of building kuma. The Go module stays dependency free and the benchmark harness carries the Python, the Docker images, the datasets and the result history.

There is a second reason, which is that a benchmark repository has a different rhythm. The library changes when someone writes code; the benchmarks need to re-run when pandas releases, when Polars releases, when a new machine type appears, and on a schedule regardless. Tying those two cadences together makes both worse.

## The rule that makes this worth doing

Benchmarks that run once, at the end, before an announcement, are marketing. Benchmarks that run continuously, from the first milestone, are engineering. The difference is that the second kind tells you which commit made things slow while you still remember what you were doing.

So the harness exists from M1, when there is barely anything to measure, and it grows with the library. At M1 it can only compare CSV reading and a few aggregations, and that is fine, because by M5 when the SIMD work lands there is a year of history to check the claims against.

The second rule is that we publish results we lose. A benchmark suite that only shows wins is not information, and anyone experienced reads it as an advertisement and discounts everything in it. Where pandas or Polars is faster, the number goes in the table with a note about why. This is not modesty, it is the only way the numbers we do win stay credible.

## What gets measured

### The db-benchmark suite

The h2oai database benchmark is the standard comparison for this class of library. h2oai stopped maintaining it in 2021 and DuckDB Labs revived it as `duckdblabs/db-benchmark`, which is the version everyone now cites.

It is ten group by queries and five join queries at 0.5 GB, 5 GB and 50 GB. We run the first two sizes in CI and the 50 GB size on demand, since it needs a machine most CI runners are not.

This is the suite people will compare us against whether we run it or not, so we run it ourselves, on our own hardware, with the harness in our own repository, and we do it before anyone else does it for us.

### TPC-H

Twenty two analytical queries at scale factor 1 and scale factor 10. Where db-benchmark stresses group by and join in isolation, TPC-H stresses the optimizer, because the wins there come from projection pushdown, predicate pushdown, join ordering and partition pruning rather than from kernel speed.

This is the suite that catches optimizer regressions, and it is the one that will hurt early, since Polars and DuckDB have had years of work on exactly these queries.

### Microbenchmarks

Go `testing.B` benchmarks inside `tamnd/kuma` itself, not in the bench repository, because they need to run on every pull request and must not require Python. These cover individual kernels, the null fast path against the masked path, chunk size sweeps, and the typed API against the dynamic API, which is where the five percent overhead budget from document 03 is enforced.

`benchstat` on every pull request, with a regression threshold that fails the build.

### Ingestion

Read time for CSV, Parquet and NDJSON at several sizes and column mixes, since this is what a first impression is made of and it is where the SIMD parsing work should show up most clearly. Reported separately from query time, because a suite that bundles them lets a slow parser hide behind a fast engine.

### Memory and binary size

Peak resident set for each query, tracked alongside wall time. An engine that is twice as fast and uses five times the memory has not necessarily won, and the only way anyone finds that out is if it is in the table.

Shared library size for each binding platform, tracked per commit, since it only ever grows and nobody notices until an npm package is eighty megabytes.

## Making the comparison honest

This is most of the work. It is easy to produce a benchmark that is wrong in your own favour without meaning to.

**Force materialization.** A lazy engine that returns a plan has done nothing. Every timed query ends in something that requires the full result to exist. This is the single most common way dataframe benchmarks lie, and it is usually an accident.

**Time the whole thing.** Include reading the input. Excluding IO measures a scenario nobody is in.

**Same data on disk.** One generator produces the datasets once, and all three libraries read the identical files. Not one file per library, not a regenerated file per run.

**Warm and cold both.** Report the first run and the median of subsequent runs separately. First run includes page cache misses and any lazy initialization, and it is the one users actually feel.

**Configure the competition properly.** Polars gets its lazy API and its streaming engine. pandas gets the Arrow dtype backend and PyArrow engine readers, since pandas 3.0 defaults to Arrow backed strings and benchmarking it against the NumPy object path would be a straw man. If a competitor has a tuning knob that helps, we turn it on and record that we did.

**Pin everything.** Library versions, dataset generator seed, CPU model, core count, memory, kernel version, Go version and whether `GOEXPERIMENT=simd` was on. Results without this context are not reproducible and therefore not evidence.

**Same machine, same run.** All three libraries in one job on one machine. Numbers from different runs on different runners are not comparable no matter how similar the specs look.

## How it runs

```
kuma-bench/
  data/         generators, deterministic, seeded
  suites/
    dbbench/    the ten group by and five join queries, three implementations each
    tpch/       twenty two queries, three implementations each
    io/         ingestion timings
  runner/       orchestration, isolation, timing, result collection
  results/      committed JSON, one file per run
  site/         static site generated from results
  docker/       pinned images per library
```

Each library runs in its own container with pinned versions, in a fresh process per query so that one query cannot warm a cache for the next, with the machine otherwise idle. The runner writes one JSON record per query per library per run, and those records are committed to the repository. The history is the point: a chart of query time by commit over eighteen months is worth more than any single table.

The site is a static page generated from those JSON files. No database, no service to keep running.

Schedule is nightly against the main branch, on every tagged release, on demand for a branch by label, and automatically when a new pandas or Polars version appears.

CI runs on GitHub Actions for the small sizes and on a dedicated bare metal machine for the large ones, because shared cloud runners have noisy neighbours and 50 GB benchmarks on a noisy machine produce numbers that are worse than none.

## What we expect to find

Worth writing down in advance so we cannot revise our expectations after seeing the results.

We should beat pandas on nearly everything by a wide margin, because pandas is single threaded and eager. If we do not beat pandas on a query, something is wrong and it is a bug, not a benchmark result.

We should be in range of Polars, meaning within roughly 2x, on group by and join. Polars has years of tuning and a mature streaming engine. Being close is a good outcome; being faster on specific queries is plausible where StringView helps.

We will probably lose to Polars on TPC-H early, because those queries reward optimizer maturity and ours will be new.

We should be able to win on string heavy work, because StringView by default with inline prefix comparison is a genuine structural advantage and arrow-go's own compute layer does not have it.

We will lose on ecosystem, forever, and no benchmark measures that. The reasons to use kuma are compile time safety, a single binary, real cancellation and no Python runtime, and none of those show up in a wall clock number. The benchmarks exist to prove that choosing those things does not cost speed, not to claim that speed is the reason.

## Milestone gates

Numbers that have to be met before a milestone is considered done. These belong in document 08 as exit criteria and are collected here so the whole picture is in one place.

| Milestone | Gate |
|---|---|
| M1 | CSV read faster than pandas `read_csv` with the PyArrow engine. Aggregations faster than pandas on 10M rows. |
| M2 | Parquet read within 2x of Polars. |
| M3 | Projection and predicate pushdown demonstrably reduce bytes read, asserted in a test. Typed API within 5 percent of the dynamic API. |
| M4 | Group by scales to at least 8 cores at better than 60 percent efficiency. |
| M5 | Every SIMD kernel beats its scalar twin by the factor claimed in document 05, on both amd64 and arm64. |
| M7 | The full db-benchmark suite runs green at 0.5 GB and 5 GB, results published whatever they say. |
| M9 | The 50 GB size completes within a memory budget smaller than the dataset. |
| 1.0 | TPC-H SF10 complete, all 22 queries, results published. |
