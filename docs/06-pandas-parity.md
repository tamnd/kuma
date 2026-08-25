# pandas conformance checklist

The target is pandas 3.0, released 21 January 2026, not the 2.x series most people are still running. That distinction matters: 3.0 made Copy-on-Write the only mode, made the Arrow backed string dtype the default, and shipped `pd.col()` expressions. The API we are matching has already moved toward the design in this spec.

The rule for this document is that everything pandas can do, kuma can do. Where the pandas shape is not idiomatic Go we change the shape, not the capability. The only genuine omissions are things that are meaningless outside Python, and each one is listed at the bottom with what to use instead.

Tags used below:

- `(M1)` through `(M10)` is the milestone from document 08
- `same` means the name carries over directly
- `adapted` means same capability, different shape, with a note explaining why

## A note on the index

The earlier draft of this spec rejected the pandas index outright. That was too blunt given the conformance goal, because it silently dropped `loc`, `reindex`, `align`, `sort_index`, `asof` and a dozen others.

The position now is narrower and it is worth stating clearly, because it decides about thirty rows below. kuma frames have **no implicit index and never align automatically**. A frame may carry an **optional, explicitly declared key index**, and when it does, the label based operations work. What never happens is pandas' behaviour of silently aligning two frames on their indexes during an arithmetic operation, which is the actual footgun. You get the capability without the surprise.

```go
f = f.WithIndex(t.TS)          // explicit, opt in
f.Loc(someTimestamp)           // now available
a.Add(b)                       // still positional, never aligned behind your back
```

---

## 1. Top level functions

### Reshaping and combining

- [ ] `melt` -> `Unpivot` (M7)
- [ ] `pivot` -> `Pivot`, returns a dynamic frame since output columns come from data (M7)
- [ ] `pivot_table` -> `PivotTable` with aggfunc (M7)
- [ ] `crosstab` -> `Crosstab` (M7)
- [ ] `cut` -> `Cut` (M7)
- [ ] `qcut` -> `QCut` (M7)
- [x] `merge` -> `Frame.Join` with an explicit join type, plus `InnerJoin`, `LeftJoin` and `CrossJoin` for the common ones (M1). Semi and anti joins are here too, which pandas has no direct spelling for. The typed form over a generated schema is still M3.
- [ ] `merge_ordered` -> `JoinOrdered` (M8)
- [ ] `merge_asof` -> `JoinAsof` (M8)
- [x] `concat` -> `Concat` for rows, `HStack` for columns, adapted since there is no axis parameter (M1). `ConcatUnion` is the row version for frames that do not hold the same columns, which is what the pandas default does.
- [ ] `get_dummies` -> `ToDummies` (M7)
- [ ] `from_dummies` -> `FromDummies` (M7)
- [ ] `factorize` -> `Factorize` (M7)
- [ ] `unique` -> `Unique` (M7)
- [ ] `lreshape` -> `Unpivot` with grouped value columns, adapted (M7)
- [ ] `wide_to_long` -> `WideToLong` (M7)
- [ ] `json_normalize` -> `Flatten` on nested struct columns, adapted (M7)

### Missing data and conversion

- [x] `isna` `isnull` -> `IsNull` (M1). A frame gives a frame of boolean columns, a column gives `NullMask`, since `IsNull` on a column already answers about one row.
- [x] `notna` `notnull` -> `IsNotNull` (M1). `ValidMask` on a column and a series, which is the mask a filter usually wants.
- [x] `to_numeric` -> `Cast` with an error policy (M1)
- [ ] `to_datetime` -> `ParseTimestamp` with format and inference (M8)
- [ ] `to_timedelta` -> `ParseDuration` (M8)
- [ ] `array` -> `NewSeries[T]` (M1)

### Date ranges and offsets

- [ ] `date_range` -> `DateRange` (M8)
- [ ] `bdate_range` -> `BDateRange` (M8)
- [ ] `timedelta_range` -> `DurationRange` (M8)
- [ ] `period_range` -> `PeriodRange` (M8)
- [ ] `interval_range` -> `IntervalRange` (M8)
- [ ] `infer_freq` -> `InferFreq` (M8)

### Evaluation and options

- [ ] `eval` -> `kuma.SQLExpr` (M10)
- [ ] `set_option` `get_option` `reset_option` `describe_option` -> `kuma.Options` struct plus `SetDefaults` (M1)
- [ ] `option_context` -> options are passed per call, so the context form is unnecessary (adapted, M1)
- [ ] `show_versions` -> `kuma.Version()` (M1)

---

## 2. DataFrame and Series

### Attributes

- [ ] `index` -> `Index()`, present only when explicitly set (M7)
- [ ] `columns` -> `Names()` (M1)
- [ ] `dtypes` -> `Schema()` (M1)
- [ ] `values` -> `Values()` per column, adapted since there is no single homogeneous 2D value (M1)
- [ ] `axes` -> `Names()` and `Len()`, adapted (M1)
- [ ] `ndim` -> always 2 for a frame, 1 for a series (M1)
- [x] `size` -> `Size()` on a frame, and `Size()` as a group by aggregation (M1)
- [ ] `shape` -> `Shape()` (M1)
- [ ] `empty` -> `IsEmpty()` (M1)
- [ ] `memory_usage` -> `MemoryUsage()` (M1)
- [ ] `info` -> `Info()` (M1)
- [ ] `select_dtypes` -> `SelectByType` (M7)
- [ ] `attrs` `flags` `set_flags` -> `Metadata()`, carried through operations (M7)
- [x] `hasnans` -> `HasNulls()` (M1). On a frame, a column and a series, and `NullCounts()` on a frame for how many are missing in each column.
- [ ] `is_unique` -> `IsUnique()` (M7)
- [ ] `is_monotonic_increasing` `is_monotonic_decreasing` -> `IsSorted(asc)` (M7)
- [ ] `nbytes` -> `NBytes()` (M1)
- [ ] `dtype` -> `DataType()` (M1)
- [ ] `name` -> `Name()` (M1)

### Conversion

- [x] `astype` -> `Cast` (M1)
- [ ] `convert_dtypes` `infer_objects` -> `ShrinkDtypes()`, adapted since there is no object dtype to convert away from (M7)
- [ ] `copy` -> not needed, frames are immutable (adapted, M1)
- [ ] `to_numpy` -> `Values()` (M1)
- [ ] `to_records` -> `Rows(ctx)` returning `[]T` (M6)
- [ ] `to_dict` -> `ToMap` with orientation options (M6)
- [ ] `to_list` -> `Values()` (M1)
- [ ] `to_period` `to_timestamp` -> `ToPeriod` `ToTimestamp` (M8)
- [ ] `item` -> `Scalar[T]()` (M6)
- [ ] `bool` -> `Scalar[bool]()` (M6)

### Indexing and iteration

- [ ] `head` `tail` -> same (M1)
- [ ] `at` `iat` -> `ValueAt[T](row, col)` (M6)
- [ ] `loc` -> `Loc`, requires an explicit index (M7)
- [ ] `iloc` -> `Row`, `Slice`, `Take` (M1)
- [ ] `insert` -> `InsertAt` (M7)
- [ ] `items` -> `Columns()` iterator (M1)
- [ ] `keys` -> `Names()` (M1)
- [ ] `iterrows` -> `Iter(ctx)` yielding typed rows, using Go 1.23 range over func (M6)
- [ ] `itertuples` -> `Iter(ctx)` (M6)
- [ ] `pop` -> `Drop` returning both parts (M7)
- [ ] `xs` -> `Loc` with a level selector (M7)
- [ ] `get` -> `TryCol` (M1)
- [ ] `isin` -> `IsIn` (M3)
- [ ] `where` `mask` -> `kuma.When(c).Then(a).Otherwise(b)`, adapted (M3)
- [ ] `query` -> `kuma.SQLExpr` (M10)
- [ ] `between` -> `Between` (M3)
- [ ] `searchsorted` -> `SearchSorted` (M7)

### Binary operators

- [x] `add` `sub` `mul` `div` `mod` -> `Add` `Sub` `Mul` `Div` `Mod` on a column handle or an expression, with an `AddExpr` and friends for the two column form (M1). Integer division truncates the way Go does, so it is the `floordiv` of pandas rather than `div`. `truediv`, a `pow` and the frame wide forms that take another frame are still to do.
- [x] the `r` prefixed reflected variants -> not needed, argument order is explicit in Go, and `Lit` puts a value on the left of an operator when the query reads better that way (adapted, M1)
- [x] `lt` `gt` `le` `ge` `ne` `eq` -> `Lt` `Gt` `Le` `Ge` `Ne` `Eq`, and `LtExpr` and friends against another column (M1). A timestamp column spells them `Before`, `AtOrBefore`, `After` and `AtOrAfter`, since that is how people say it about time.
- [ ] `dot` -> `Dot` (M7)
- [ ] `combine` -> `Combine` (M7)
- [ ] `combine_first` -> `CombineFirst` (M7)

### Function application

- [ ] `apply` -> `Apply` on a typed expression, or `ApplyGroups` (M6)
- [ ] `map` -> `Typed[T].Map[U]`, the generic method from document 03 (M6)
- [ ] `applymap` -> removed in pandas 3.0, use `map` (adapted, M6)
- [ ] `pipe` -> ordinary Go function composition (adapted, M1)
- [ ] `agg` `aggregate` -> `Agg[R]` (M3)
- [ ] `transform` -> `Over(keys)` (M7)
- [x] `groupby` -> `GroupBy`, eagerly, returning a `GroupedFrame` (M1). The lazy form over expressions is still M3.
- [ ] `rolling` -> `Rolling` (M7)
- [ ] `expanding` -> `Expanding` (M7)
- [ ] `ewm` -> `EWM` (M7)

### Computations and descriptive statistics

- [ ] `abs` -> `Abs` (M1)
- [ ] `all` `any` -> same (M1)
- [ ] `clip` -> `Clip` (M1)
- [ ] `corr` -> `Corr`, pearson, spearman, kendall (M7)
- [ ] `corrwith` -> `CorrWith` (M7)
- [ ] `cov` -> `Cov` (M7)
- [x] `count` -> `Count`, counting the values that are there rather than the rows (M1)
- [ ] `cummax` `cummin` `cumprod` `cumsum` -> `CumMax` `CumMin` `CumProd` `CumSum` (M7)
- [ ] `describe` -> `Describe` (M7)
- [ ] `diff` -> `Diff` (M7)
- [ ] `kurt` `kurtosis` -> `Kurtosis` (M7)
- [x] `max` `min` `mean` `median` `sum` -> same names (M1). `prod` is still to do.
- [ ] `mode` -> `Mode` (M7)
- [ ] `pct_change` -> `PctChange` (M7)
- [x] `quantile` -> `Quantile` with all five interpolations (M1)
- [ ] `rank` -> `Rank`, all six methods (M7)
- [ ] `round` -> `Round` (M1)
- [ ] `sem` -> `SEM` (M1)
- [ ] `skew` -> `Skew` (M7)
- [x] `std` `var` -> same, with ddof (M1)
- [x] `nunique` -> `NUnique` (M1, earlier than planned since the group by key encoder already had the answer)
- [ ] `value_counts` -> `ValueCounts` (M7)
- [ ] `idxmax` `idxmin` -> `IdxMax` `IdxMin`, returning a label with an index or a position without (M7)
- [ ] `argmax` `argmin` -> `ArgMax` `ArgMin` (M7)
- [x] `argsort` -> `SortIndex`, on a series, a column and a frame (adapted, M1)
- [ ] `autocorr` -> `AutoCorr` (M7)
- [ ] `nlargest` `nsmallest` -> `TopK` `BottomK` (M7)
- [ ] `mad` -> removed from pandas, but `MAD` is provided anyway since it is cheap and people ask (M7)

### Reindexing, selection and labels

- [ ] `add_prefix` `add_suffix` -> `AddPrefix` `AddSuffix` (M7)
- [ ] `align` -> `Align`, requires an explicit index (M7)
- [ ] `at_time` `between_time` -> `AtTime` `BetweenTime` (M8)
- [ ] `drop` -> `Drop` (M1)
- [ ] `drop_duplicates` -> `Distinct` with keep semantics (M7)
- [ ] `duplicated` -> `IsDuplicated` (M7)
- [ ] `equals` -> `Equals` (M1)
- [ ] `filter` with `like` and `regex` -> `SelectMatching` `SelectRegex` (M7)
- [ ] `first` `last` -> `FirstBy` `LastBy` over a time offset (M8)
- [ ] `first_valid_index` `last_valid_index` -> `FirstValid` `LastValid` (M7)
- [ ] `reindex` `reindex_like` -> `Reindex`, requires an explicit index (M7)
- [ ] `rename` -> `Rename` (M1)
- [ ] `rename_axis` -> `RenameIndex` (M7)
- [ ] `reset_index` -> `DropIndex` (M7)
- [ ] `set_index` -> `WithIndex` (M7)
- [ ] `set_axis` -> `WithNames` (M7)
- [ ] `sample` -> `Sample`, with weights, replacement and a seed (M1)
- [x] `take` -> `Take` (M1)
- [ ] `truncate` -> `Truncate` (M8)

### Missing data

- [ ] `bfill` `ffill` -> `BackwardFill` `ForwardFill` (M7)
- [x] `dropna` with `how`, `thresh` and `subset` -> `DropNulls` (M1). The subset is the names passed to it, `thresh` is `KeepAtLeast`, and the two settings of `how` are the two ends of `KeepAtLeast` rather than a string.
- [x] `fillna` -> `FillNull` with a value (M1). The strategies, forward and backward fill, are M7 and are listed above.
- [ ] `interpolate` -> `Interpolate`, linear, nearest, pad, polynomial and spline (M7)
- [ ] `replace` -> `Replace` and `ReplaceStrict` (M7)
- [ ] `pad` `backfill` -> deprecated aliases in pandas, not carried over (adapted)

### Reshaping and sorting

- [x] `sort_values` -> `Sort`, `SortBy` and `SortDesc`, with null placement per key (adapted, M1). The sort is always stable, so there is no equivalent of the `kind` argument.
- [ ] `sort_index` -> `SortByIndex`, requires an explicit index (M7)
- [ ] `stack` `unstack` -> `Stack` `Unstack` (M7)
- [ ] `swaplevel` `reorder_levels` `droplevel` -> `SwapKeys` `ReorderKeys` `DropKey` over compound index keys (M7)
- [ ] `explode` -> `Explode` (M7)
- [ ] `squeeze` -> `Squeeze` (M7)
- [ ] `transpose` and `T` -> `Transpose` (M7)
- [ ] `swapaxes` -> deprecated in pandas, use `Transpose` (adapted)
- [ ] `to_xarray` -> not applicable outside Python, use the Arrow bridge (see omissions)

### Combining and comparing

- [x] `assign` -> `WithExpr` for a column worked out from an expression and `WithColumn` for one you already have (M1, earlier than planned since the expression engine landed with the filter)
- [ ] `compare` -> `Compare` (M7)
- [ ] `join` -> `Join` on an explicit index (M7)
- [ ] `update` -> `Update` (M7)

### Time series

- [ ] `asfreq` -> `AsFreq` (M8)
- [ ] `asof` -> `AsOf` (M8)
- [ ] `shift` including `freq` -> `Shift` (M7, duration form M8)
- [ ] `resample` -> `Resample` with closed, label and origin (M8)
- [ ] `tz_convert` `tz_localize` -> `TZConvert` `TZLocalize` with an explicit ambiguity policy (M8)
- [ ] `repeat` -> `Repeat` (M7)

---

## 3. GroupBy

- [ ] `agg` `aggregate` in all four calling forms -> `Agg[R]` with expressions, which subsumes them (M3)
- [ ] `apply` -> `ApplyGroups`, and `AggFold[Acc, Out]` for the accumulator case (M6)
- [ ] `transform` -> `Over` (M7)
- [ ] `filter` -> `FilterGroups` (M7)
- [ ] `pipe` -> ordinary composition (adapted)
- [x] `count` `size` `sum` `mean` `median` `min` `max` `std` `var` -> same names (M1). `all`, `any`, `prod` and `sem` are still to do.
- [x] `first` `last` -> same names (M1). `nth`, `head` and `tail` on a group are still to do.
- [ ] `idxmax` `idxmin` -> same (M7)
- [x] `quantile` -> same, with all five interpolations (M1). `describe` and `rank` are still to do (M7).
- [ ] `shift` `diff` `pct_change` -> same (M7)
- [ ] `cumsum` `cumprod` `cummax` `cummin` `cumcount` -> same (M7)
- [x] `ngroups` -> `NumGroups` (M1). `ngroup`, the group number of each row, is `Groups().IDs()` and wants a name of its own (M7).
- [ ] `fillna` `ffill` `bfill` -> same (M7)
- [x] `nunique` -> `NUnique` (M1). `value_counts` and `unique` are still to do (M7).
- [ ] `rolling` `expanding` `resample` on groups -> same (M8)
- [ ] `get_group` `groups` `indices` -> `Group`, `Partitions` (M7). The raw form is `GroupedFrame.Groups()` today.
- [ ] `dropna` `observed` `sort` `as_index` options -> options on `GroupBy` (M3). Today the behaviour is fixed: nulls are a group, the order is first appearance, and the keys come back as columns.
- [ ] `corr` `cov` `skew` `kurt` -> same (M7)
- [ ] `sample` -> same (M7)

## 4. Window operations

Rolling, Expanding and EWM all support the same aggregation set.

- [ ] `rolling(window, min_periods, center, closed, step)` -> `Rolling` (M7)
- [ ] `rolling(win_type=...)` weighted windows -> `RollingWeighted`, all scipy window types (M7)
- [ ] `rolling("30min")` time based -> `RollingBy` (M8)
- [ ] `expanding` -> `Expanding` (M7)
- [ ] `ewm(com, span, halflife, alpha, adjust, ignore_na, times)` -> `EWM` (M7)
- [ ] window `count` `sum` `mean` `median` `var` `std` `min` `max` -> same (M7)
- [ ] window `corr` `cov` `skew` `kurt` `sem` `quantile` `rank` -> same (M7)
- [ ] window `apply` `aggregate` -> same (M7)
- [ ] beyond pandas: `Over(partition, order)` SQL style window functions with `lag`, `lead`, `row_number`, `dense_rank`, `ntile`, `first_value`, `last_value`, `nth_value` (M7)

## 5. The string namespace

Everything here lands together in M7. Half a string namespace reads as a toy.

- [ ] `cat` `center` `contains` `count` `endswith` `startswith` (M7)
- [ ] `extract` `extractall` `find` `findall` `fullmatch` `match` `index` `rindex` `rfind` (M7)
- [ ] `get` `get_dummies` `join` `len` `repeat` `slice` `slice_replace` (M7)
- [ ] `ljust` `rjust` `pad` `zfill` `wrap` (M7)
- [ ] `lower` `upper` `title` `capitalize` `casefold` `swapcase` (M7)
- [ ] `strip` `lstrip` `rstrip` `removeprefix` `removesuffix` (M7)
- [ ] `partition` `rpartition` `split` `rsplit` (M7)
- [ ] `replace` `translate` `normalize` (M7)
- [ ] `isalnum` `isalpha` `isdigit` `isspace` `islower` `isupper` `istitle` `isnumeric` `isdecimal` `isascii` (M7)
- [ ] `encode` `decode` -> explicit encoding conversion functions, since Go strings are already bytes (adapted, M7)
- [ ] `len` on bytes versus runes -> both provided, as `Len` and `ByteLen`, because Go users will expect the distinction (M7)
- [ ] beyond pandas: `JSONExtract`, `JSONPath`, `Base64Encode`, `Base64Decode`, `ContainsAny` for multi pattern search in a single SIMD pass (M7)

One real divergence to document loudly rather than bury. Go's `regexp` is RE2, which has no backreferences and no lookaround, in exchange for guaranteed linear time matching. Some pandas regex patterns will not work. We are not going to vendor a backtracking engine to fix this; we are going to document it and provide `ContainsAny` and the literal string functions so that most people never need a regex in the first place.

## 6. The datetime namespace

- [ ] `year` `month` `day` `hour` `minute` `second` `microsecond` `nanosecond` (M7)
- [ ] `dayofweek` `day_of_week` `weekday` `dayofyear` `day_of_year` `days_in_month` `daysinmonth` (M7)
- [ ] `quarter` `week` `weekofyear` `isocalendar` (M7)
- [ ] `date` `time` `timetz` (M7)
- [ ] `is_month_start` `is_month_end` `is_quarter_start` `is_quarter_end` `is_year_start` `is_year_end` `is_leap_year` (M7)
- [ ] `normalize` `round` `floor` `ceil` (M8)
- [ ] `strftime` and parsing with format (M8)
- [ ] `tz` `tz_localize` `tz_convert` (M8)
- [ ] `to_period` `to_pydatetime` -> `ToPeriod`, `ToTime` returning `time.Time` (M8)
- [ ] timedelta `days` `seconds` `microseconds` `nanoseconds` `components` `total_seconds` (M7)
- [ ] `freq` and offset arithmetic -> `kuma/calendar` (M8)

Timezones use the full IANA database through `time/tzdata`, with DST aware arithmetic. Ambiguous and nonexistent local times take an explicit policy of earliest, latest or raise. pandas gets this wrong often enough that being explicit counts as a feature.

## 7. Categorical namespace

- [ ] `categories` `ordered` `codes` (M7)
- [ ] `rename_categories` `reorder_categories` `add_categories` `remove_categories` `remove_unused_categories` `set_categories` (M7)
- [ ] `as_ordered` `as_unordered` (M7)

Backed by the Arrow dictionary type. This is not only a parity feature: dictionary encoded group by keys are integers, which lets us skip hashing altogether.

## 8. Nested data namespaces

pandas 3.0 has `.list` and `.struct` accessors for Arrow backed nested columns. These map directly since our layout is the same.

- [ ] `.list` `len` `__getitem__` `flatten` (M7)
- [ ] `.struct` `dtypes` `field` `explode` (M7)
- [ ] beyond pandas: map type accessors, `keys`, `values`, `get` (M7)

## 9. Index types

Only relevant when an index has been explicitly set.

- [ ] `Index` general purpose (M7)
- [ ] `RangeIndex` -> implicit positions, always available (M1)
- [ ] `DatetimeIndex` -> a timestamp column set as the index (M8)
- [ ] `TimedeltaIndex` -> a duration column set as the index (M8)
- [ ] `CategoricalIndex` -> a dictionary column set as the index (M7)
- [ ] `IntervalIndex` -> an interval column set as the index (M8)
- [ ] `MultiIndex` -> a compound index of several columns (M7)
- [ ] `PeriodIndex` -> a period column set as the index (M8)
- [ ] index `get_loc` `get_indexer` `slice_locs` `union` `intersection` `difference` `symmetric_difference` (M7)

A compound index replaces MultiIndex. It is a list of key columns, not a separate nested object with its own levels and codes, which removes most of what makes MultiIndex hard to use while keeping what it is for.

## 10. Date offsets

pandas ships around forty `DateOffset` classes. These map to `kuma/calendar` (M8).

- [ ] `Day` `Week` `MonthBegin` `MonthEnd` `QuarterBegin` `QuarterEnd` `YearBegin` `YearEnd`
- [ ] `BusinessDay` `BusinessHour` `CustomBusinessDay` `CustomBusinessHour`
- [ ] `BMonthBegin` `BMonthEnd` `BQuarterBegin` `BQuarterEnd` `BYearBegin` `BYearEnd`
- [ ] `SemiMonthBegin` `SemiMonthEnd` `LastWeekOfMonth` `WeekOfMonth`
- [ ] `Easter` `FY5253` `FY5253Quarter`
- [ ] `Hour` `Minute` `Second` `Milli` `Micro` `Nano`
- [ ] holiday calendars, `AbstractHolidayCalendar` equivalent, and US federal holidays as a built in

## 11. Input and output

- [ ] `read_csv` `to_csv` -> `ScanCSV` `WriteCSV` with a SIMD parser (M1)
- [ ] `read_parquet` `to_parquet` -> `ScanParquet` `WriteParquet` with pushdown (M2)
- [ ] `read_feather` `to_feather` and Arrow IPC -> `ScanIPC` `WriteIPC` (M2)
- [ ] `read_json` `to_json` and NDJSON -> `ScanNDJSON` `WriteNDJSON` (M2)
- [ ] `read_orc` `to_orc` -> `ScanORC` `WriteORC` (M10)
- [ ] `read_sql` `read_sql_query` `read_sql_table` `to_sql` -> `kuma/sqlio` (M10)
- [ ] `read_fwf` -> `ScanFWF` (M10)
- [ ] `read_excel` `to_excel` `ExcelWriter` -> `kuma/xlsx` (M10)
- [ ] `read_html` `to_html` -> `kuma/htmlio` (M10)
- [ ] `read_xml` `to_xml` -> `kuma/xmlio` (M10)
- [ ] `read_stata` `to_stata` -> `kuma/statio` (post 1.0)
- [ ] `read_sas` `read_spss` -> `kuma/statio` (post 1.0)
- [ ] `read_hdf` `to_hdf` -> post 1.0, needs an HDF5 dependency, likely cgo
- [ ] `read_pickle` `to_pickle` -> `WriteIPC`, since Arrow IPC is the equivalent role and is portable (adapted, M2)
- [ ] `read_clipboard` `to_clipboard` -> `kuma/clipboard` (post 1.0)
- [ ] `read_gbq` `to_gbq` -> use ADBC (adapted, M10)
- [ ] `to_markdown` -> `ToMarkdown` (M1)
- [ ] `to_latex` -> `ToLatex` (M10)
- [x] `to_string` -> `String` (M1)
- [ ] beyond pandas: Hive partitioned dataset scan (M2 metadata, M9 pruning), Arrow C Data Interface (M2), ADBC driver (M10), Delta and Iceberg readers (post 1.0)

## 12. Plotting

pandas `.plot` covers line, bar, barh, hist, box, kde, density, area, pie, scatter and hexbin, plus `plotting.scatter_matrix`, `andrews_curves`, `parallel_coordinates`, `lag_plot`, `autocorrelation_plot` and `bootstrap_plot`.

- [ ] all of the above -> `kuma-plot`, a separate module emitting Vega-Lite JSON (post 1.0)

Kept out of the core module so that a plotting dependency never lands in a server binary. Vega-Lite rather than a Go image library because it renders in notebooks, in browsers and in the terminal via a converter, and because writing a chart library is not this project.

## 13. Styler

pandas `Styler` has roughly forty methods for conditional formatting and HTML export.

- [ ] `background_gradient` `bar` `highlight_max` `highlight_min` `highlight_null` `highlight_between` `highlight_quantile` (post 1.0)
- [ ] `format` `format_index` `relabel_index` `hide` `set_caption` `set_properties` `set_table_styles` (post 1.0)
- [ ] `to_html` `to_latex` `to_excel` `to_string` (post 1.0)

Grouped into `kuma/style`, post 1.0. It is genuinely useful for reports and genuinely not urgent.

## 14. Testing helpers

- [x] `assert_frame_equal` -> `kumatest.EqualFrames` (M1)
- [x] `assert_series_equal` -> `kumatest.EqualSeries` (M1)
- [ ] `assert_index_equal` -> `kumatest.EqualIndex` (M7)
- [x] `makeDataFrame` and the random data helpers -> `kumatest.Random` (M1)

## 15. Extension points

- [ ] `ExtensionDtype` `ExtensionArray` -> a registration API for user defined dtypes (M6)
- [ ] custom accessors, meaning `register_dataframe_accessor` -> ordinary Go functions on your own types (adapted)
- [ ] user defined aggregations -> `AggFold[Acc, Out]` (M6)
- [ ] user defined kernels -> the dtype keyed function registry (M6)

Note the constraint from document 01: Go 1.27 does not allow generic methods on interfaces, so an extension point cannot be an interface with a generic method. Every one of these is a registry of concrete function values keyed by dtype.

---

## Genuine omissions

Five things, each meaningless outside Python rather than merely inconvenient.

| pandas | Why not, and what to use |
|---|---|
| `to_xarray` | xarray is a Python library. Export through the Arrow bridge and call xarray from Python. |
| `read_pickle` of arbitrary objects | Pickle is a Python bytecode format and reading it safely from Go is not tractable. Arrow IPC fills the same role. |
| `register_*_accessor` | Monkey patching a class at runtime. In Go you write a function. |
| `option_context` | A Python context manager. Options are per call arguments here. |
| the `r` prefixed reflected operators | They exist because Python dispatches `a + b` on the left operand's type. Go has explicit argument order. |

## Behaviour we match in capability but not in shape

These are the places where 1:1 in capability means deliberately not 1:1 in behaviour, and each needs a documentation page under the name people will search for.

**No automatic alignment.** An explicit index gives you `loc`, `reindex` and `align`. What never happens is two frames silently aligning during arithmetic. That behaviour is the source of most pandas surprises and reproducing it would be reproducing a bug.

**No `inplace=True`.** Frames are immutable and every operation returns a new one. pandas 3.0 went the same direction with Copy-on-Write, and chained assignment now raises there too.

**No `object` dtype.** There is no column of boxed anything. If you need heterogeneous data, use a struct column.

**NaN is not null.** Missing values live in a validity bitmap and NaN is a perfectly valid float. pandas 3.0 reached the same model through Arrow.

**No silent upcasting.** `int64 + float64` needs an explicit cast, and the error arrives at plan time rather than partway through the data.

**Float summation order differs.** Vectorized summation associates differently from a scalar loop. The result is more accurate, not less, but it is not bit identical.

## What we have that pandas does not

The checklist above is the floor. The reasons to use this instead of calling into Python are compile time checking of column names and types, lazy evaluation with a real optimizer and `Explain` to inspect it, multi threaded execution by default, `context.Context` cancellation, streaming execution with spill for data larger than memory, SQL over the same plan, zero copy Arrow interop in both directions, and a single static binary with no runtime to install.
