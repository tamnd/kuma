# The high level API

This document covers what using kuma feels like once you have the typed columns from document 03. The test it has to pass is that someone who knows pandas is productive within an hour, and someone who has never used pandas does not have to learn a foreign mental model to get started.

## The normal path

Most work is lazy. You describe a query, nothing runs, and then you collect it.

```go
t := TradeCols

bars, err := kuma.ScanParquet[Trade]("trades/*.parquet").
    Filter(t.Price.Gt(100).And(t.Side.Eq("BUY"))).
    GroupBy(t.Symbol, t.TS.Trunc(time.Minute)).
    Agg[Bar](
        t.Price.Mul(t.Qty).Sum().To(BarCols.Volume),
        t.Price.Quantile(0.99).To(BarCols.P99),
        kuma.Count().To(BarCols.N),
    ).
    SortDesc(BarCols.Volume).
    Head(20).
    Rows(ctx)
```

Four things about that are deliberate answers to specific pandas problems.

There is one error return, at the end. Errors that happen while you are building the query get attached to the expression node that caused them and carried along, so a bad expression poisons everything downstream of it and surfaces once, when you collect. You do not write `if err != nil` between the steps of a query, because there is nothing sensible to do with an error at that point anyway.

There is a `context.Context`. A long running group by can be cancelled. pandas has never had this and it is one of the honest reasons to reach for Go for this kind of work: a query that is part of an HTTP request should die when the request does.

Nothing executes until `Rows` or `Collect`. That is what allows the optimizer to push the filter and the column list down into the Parquet reader, so the glob above never decodes the columns it does not need or the row groups the filter excludes.

There is no index. Joins take explicit keys and nothing aligns itself behind your back.

## Eager mode

Lazy only is miserable in a notebook and worse in a test, so everything is available eagerly too. `Collect` gives you a `Frame[S]` with the same verbs on it.

```go
f, err := lf.Collect(ctx)

f.Shape()          // rows, cols
f.Schema()
fmt.Println(f)     // prints a table with dtypes and null counts
```

The same operation names work on both, so moving a prototype to a lazy pipeline means deleting a `Collect` from the middle of the chain rather than rewriting it. That symmetry costs some duplication inside the library and it is worth paying for.

If you want to get at the raw memory, you can:

```go
v := f.Col(t.Price)      // Series[float64], no copy
raw := v.Values()        // []float64, contiguous, 64 byte aligned
mask := v.Validity()     // *bitmap.Bitmap, nil when there are no nulls
```

That is the door out to hand written kernels, and it is a supported door rather than an accident of the implementation. Someone will want to do something we did not think of, and the answer should be "here is the slice" rather than "file an issue".

## Getting data in and out

```go
kuma.ScanParquet[Trade](path)      // lazy
kuma.ScanCSV[Trade](path)
kuma.ScanIPC[Trade](path)
kuma.ScanNDJSON[Trade](path)

kuma.FromStructs(rows)             // []Trade to Frame[Trade]
kuma.FromArrow[Trade](rec)         // zero copy from arrow-go
kuma.FromSQL[Trade](ctx, db, query)

f.Rows(ctx)                        // back to []Trade
f.ToArrow()
f.WriteParquet(w)
f.WriteCSV(w)
```

The Arrow C Data Interface sits underneath `FromArrow` and `ToArrow` and is more important than it looks. It means anything kuma has not implemented yet can be handed to DuckDB or pyarrow without copying, which turns "kuma is incomplete" from a blocker into an inconvenience. That is why it lands early in the milestone plan rather than late.

## Errors

There are three kinds of failure and they surface at three different times.

A malformed expression is caught when you build it and reported when you collect. A schema or type problem is caught at plan time, before any IO happens, so a query against a file with the wrong columns fails immediately rather than after reading half a terabyte. Runtime problems, meaning IO errors, memory pressure and cancellation, surface during execution wrapped with the operator that was running.

A value written in a query is checked as well as its type. An int8 column can be compared against an integer and 300 is an integer, so the pair of types is fine and the pair of a type and a value is not, and the second is caught at plan time along with the first.

```
kuma: kernel: 300 does not fit in int8, which holds -128 to 127, cast the column or write a value it holds: wrong type
```

The quality of the message matters more than almost anything else in the library, because it is the thing users see most often. The bar is this:

```
kuma: column "sym" not found in Filter
  available: symbol, price, qty, side, ts
  did you mean: symbol?
  in plan:
    Filter [col("sym") > 100]      <- here
    Scan parquet trades/*.parquet
```

Note that with typed columns this particular error mostly stops happening, because the compiler catches it first. It still exists for the dynamic path and for the case where the file schema does not match the struct, which is the more common failure once the typed layer is in place.

Sentinel errors are comparable with `errors.Is`. Nothing panics across an API boundary.

## Explain and profile

```go
text, err := lf.Explain()
```

```
the query as written
  Limit 20
    Project symbol, price
      Filter (price > 100)
        Sort by qty desc
          Scan trades/*.parquet

the query that runs
  Project symbol, price
    Limit 20
      Sort by qty desc
        Filter (price > (100 as float64))
          Scan trades/*.parquet

changed by predicate pushdown, slice pushdown and type coercion
```

The query as written is what the caller built, the query that runs is what the passes turned it into, and the last line names the passes that made the difference. A query no pass changes is printed once with a line saying so, because printing the same plan twice tells nobody anything. The format is documented and kept, since people will parse it whether we want them to or not.

The `as float64` on the filter is worth a word. A value written in a query has no type of its own, and 100 against a float64 column is a float64 rather than dragging the column up to an int64. That rule used to live only in the engine, so the plan said `price > 100` and what happened was something you had to know. The type coercion pass works it out once and writes it into the plan, which means the comparison you can read is the comparison that runs. A value already at the type it is used with is left plain, so a float written against a float column reads exactly as it was written and only the conversions show up.

The other thing an explain will show you is that the steps you wrote are not the steps that run. Adding a column and then adding another one out of it is two calls to write and there is no reason for it to be two passes over the data, so a query written as

```go
lf.With("notional", t.Price.Mul(2)).With("doubled", kuma.F64("notional").Mul(2)).Select("doubled")
```

runs as one projection over a scan of one column:

```
Project ((price * 2) * 2) as doubled
  Scan frame [price]
```

The rule the fusion pass follows is that a value is only brought up into the step that reads it if that step reads it once. A value read twice stays where it is and stays worked out once, because working it out twice would cost more than the operator it saves. That is the same rule the common subexpression pass follows from the other direction, which is why a query can be written either way and end up in the same place.

Not in it yet: an estimated row count per operator, which needs statistics on the source, and the row groups a scan skips, which needs the page index. Both go in as the scan learns to report them, under the operator they belong to.

`lf.Profile(ctx)` runs the query and returns the answer along with the same output with the numbers on.

```go
out, text, err := lf.Profile(ctx)
```

```
the query as written
  Limit 20
    Project symbol, price
      Filter (price > 100)
        Scan trades/*.parquet

the query that ran
  Project symbol, price                          10.0us   2 rows
    Limit 20                                     10.0us   2 rows
      Filter (price > (100 as float64))          50.0us   3 rows
        Scan trades/*.parquet [symbol, price]    30.0us   4 rows

changed by slice pushdown, projection pushdown and type coercion
ran in 100us
```

The time on a line is what that operator spent and not what its inputs spent, so the lines add up to the total on the last one and the largest of them is the operator to go and look at. The rows on a line are the rows it produced, which are also the rows the operator above it read, so a line and the line under it are a count in and a count out and neither needs printing twice. A join has two lines under it and reads both.

The answer comes back too rather than being worked out and thrown away, because a query worth timing is usually a query somebody wanted the answer to, and running it twice to have both would be a profile of a run that is not the one the answer came from. The clock is read once per operator, not once per row or once per batch, so a profiled query is the same query and the numbers are a fact about the run rather than an estimate of one.

Bytes read is not in it yet. It needs the sources to report what they read, and it goes on the scan line when they do.

These are shipping features, not debugging tools, and they exist from the milestone where the optimizer lands. Two reasons. Users need to see why a query is slow, and pandas gives them nothing at all here. And internally, every optimization we claim to have made becomes something we can demonstrate in a test by asserting on the plan, rather than something we assert in a commit message.

## Display

```
kuma.Frame[market.Bar] 4 rows x 3 cols

  symbol | minute                |   volume
  string | timestamp[us, tz=UTC] |  float64
---------+-----------------------+---------
  AAPL   | 2026-08-25 09:30:00   | 1.24e+07
  MSFT   | 2026-08-25 09:30:00   | 8.91e+06
  null   | 2026-08-25 09:31:00   | 4.02e+06
  GOOG   | 2026-08-25 09:31:00   |  5.5e+06
```

Dtypes in the header, shape always shown, and `null` rendered differently from `NaN` because they are different things. The names in the type row are the ones the schema prints, so there is no second vocabulary of abbreviations to learn. `Frame` implements `fmt.Stringer`, and `Series` and `Column` print the same way as a table of one column.

Ten rows and twelve columns are shown by default, with the rest replaced by a line of dots in the middle, and `Frame.Render` takes a `PrintOptions` when a different amount is wanted. `MaxRows` of -1 means all of them. Numbers print at the shortest text that reads back as the same value rather than rounded to a fixed number of digits, because a printer that rounds will one day show two different numbers as the same during the debugging session where that difference is the whole point. A string that begins or ends in a space is quoted for the same reason.

For tests, `kumatest.EqualFrames` compares two frames and prints the first few rows that differ rather than dumping both of them. Anyone who has debugged a failing pandas test by reading four thousand lines of output knows why this is worth building on day one rather than later.

## The dynamic escape hatch

When you genuinely do not know the schema at compile time, which mostly means reading arbitrary user supplied files, the string based API is still there:

```go
lf := kuma.ScanCSV[kuma.Dynamic]("whatever.csv")
lf = lf.FilterDyn(kuma.Dyn("price").Gt(100))

typed, err := kuma.Bind[Trade](lf)   // check once, then you are back in typed land
```

This is covered properly in document 03. The short version is that it exists, it works, and the naming is a bit awkward on purpose so that using it is a decision.

## Things that are easy to defer and should not be

A few pieces of the developer experience get pushed to "later" by every project that builds one of these, and later never comes. Listing them here so that they end up in the milestone plan with dates attached.

Explain and profile, from the milestone where the optimizer exists. If they arrive later, every optimization in between goes unverified.

The test helpers, from the first milestone. Assertion helpers written after the fact never get adopted, because by then everyone has their own comparison function.

Deterministic group ordering by default, with the faster nondeterministic version available as an option. Nondeterministic output makes tests unwritable, and a library whose tests are annoying to write will have bad tests.

Runnable examples on every exported function. This is the Go standard library convention and it is also the cheapest documentation that cannot rot, because `go test` compiles and runs it.

A migration guide organized by pandas function name. People will search for `set_index` and they should land on a page explaining why there is no index and what to do instead, rather than on nothing.
