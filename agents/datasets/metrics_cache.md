# metrics_cache dataset

## File

`metrics_cache_{asset}.parquet` in `CWD/data/`.

## Purpose

Companion to an asset's cached candle files. Caches pre-calculated metric/indicator columns aligned 1:1 with the candle timestamps so they do not need to be recomputed on every run.

## Alignment

- Same number of rows as the asset's full cached candle range.
- `ts` column is a verbatim copy of the candle `ts` (same values, same dtype, ascending order).

## Mutation model

Parquet is immutable. Adding columns means: read the file → add the column(s) → write a new file at the same path (append-by-rewrite).

## Maintenance rule

Whenever a column is added or changed, append a short description of that column to the **Columns** section below (name, meaning, source column, rolling window, units). The schema here must always be complete.

## Columns

| Column | Type | Source | Window | Description |
|---|---|---|---|---|
| `ts` | int64 (ms epoch) | candles `ts` | — | Candle open timestamp, UTC milliseconds. Verbatim copy from candle file. |
| `v_median` | float64 | candles `v` | 10080 (1 week) | Rolling median of candle volume over the preceding 10080 1-minute candles (partial look-back for early rows). |
| `v_iqr` | float64 | candles `v` | 10080 (1 week) | Rolling IQR (Q3 − Q1) of candle volume over the same window. Q1 = sorted[m//4], Q3 = sorted[3m//4]. |
| `v_mean` | float64 | candles `v` | 10080 (1 week) | Rolling mean of candle volume over the preceding 10080 1-minute candles (partial look-back for early rows). |
| `v_stddev` | float64 | candles `v` | 10080 (1 week) | Rolling population standard deviation (÷ m) of candle volume over the same window. |
