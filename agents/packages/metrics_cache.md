# metrics_cache

Builds and maintains a per-asset metrics cache parquet file (`CWD/data/metrics_cache_{asset}.parquet`) that stores pre-calculated indicator columns aligned 1:1 with candle timestamps. Local cache only — never downloads candles.

See `agents/datasets/metrics_cache.md` for the file-format specification.

## Location

`packages/tools/metrics_cache/`

## Import

```python
from packages.tools.metrics_cache import (
    create_metrics_cache_base_file,
    metrics_cache_volume_median_iqr,
    metrics_cache_volume_mean_stddev,
)
```

## Prerequisites

Candle files (`{asset}_1m_*.parquet`) must already be cached in `CWD/` or `CWD/data/`. Preload them with `packages.tools.candle_preloader` if missing.

## Functions

### `create_metrics_cache_base_file(assetname) → Path`

Builds (or resets) the metrics cache for `assetname` to a single `ts` column.

- Locates all `{asset}_1m_*.parquet` files in `CWD/` then `CWD/data/` (excludes `metrics_cache_*`).
- Concatenates, deduplicates, sorts ascending by `ts`.
- Validates that every consecutive `ts` difference equals `60000` ms; raises `ValueError` on any gap.
- Writes `CWD/data/metrics_cache_{asset}.parquet` (creates `data/` if absent). Overwrites any existing file.
- Raises `FileNotFoundError` if no candle files are found.
- Returns the written path.

### `metrics_cache_volume_median_iqr(assetname) → Path`

Appends `v_median` and `v_iqr` (float64) to the metrics cache using `rolling_median_iqr` with `window=10080`.

- Requires the base file to exist; raises `FileNotFoundError` otherwise.
- Verifies candle `ts` matches cache `ts`; raises `ValueError` on mismatch.
- Prints labeled timing: load / compute / write / total.
- Returns the cache path.

### `metrics_cache_volume_mean_stddev(assetname) → Path`

Appends `v_mean` and `v_stddev` (float64) to the metrics cache using `rolling_mean_stddev` with `window=10080`.

- Same preconditions, error handling, and timing output as `metrics_cache_volume_median_iqr`.
- Returns the cache path.

## Rolling window semantics

All windows are causal look-back: index `i` uses `array[max(0, i-window+1) : i+1]`. Early rows use a shrinking partial window down to `m=1` — no `0.0` padding.

## Usage example

```python
from packages.tools.metrics_cache import (
    create_metrics_cache_base_file,
    metrics_cache_volume_median_iqr,
    metrics_cache_volume_mean_stddev,
)

path = create_metrics_cache_base_file("btcusdt")
metrics_cache_volume_median_iqr("btcusdt")
metrics_cache_volume_mean_stddev("btcusdt")
```
