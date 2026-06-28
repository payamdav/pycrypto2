# candle_cache

In-memory candle cache for fast repeated slicing within a session. Sits on top of
`candle_preloader` (file cache) and provides O(1) anchor resolution from a timestamp.

## Location

`packages/tools/candle_cache/`

## Import

```python
from packages.tools.candle_cache import (
    preload_asset_candles,
    get_cached_candles,
    is_cached,
    clear_cache,
)
```

## "Raise if not pre-loaded" contract

`get_cached_candles` raises `RuntimeError` immediately if the asset is not in
memory. The caller must call `preload_asset_candles` once before any pipeline
call. This mirrors the local-cache-only rule: the cache never hits the network.

## Functions

### `preload_asset_candles(asset) → dict`

Locate all `{asset}_1m_*.parquet` files in `CWD/` then `CWD/data/` (excludes
`metrics_cache_*` files). Concatenates, deduplicates, sorts ascending by `ts`,
loads into numpy arrays. Idempotent (cache hit returns immediately). Prints
load timing. Raises `FileNotFoundError` if no files are found — run
`candle_preloader.preload_candles` first.

### `get_cached_candles(asset) → dict`

Return the in-memory entry for `asset`. Raises `RuntimeError` if not pre-loaded.
Used by pipeline functions that need fast repeated candle access.

### `is_cached(asset) → bool`

Return `True` if the asset is currently held in memory.

### `clear_cache(asset=None)`

Drop one entry (by lowercase asset name) or all entries from memory.

## Cached entry structure

A dict of contiguous `np.ndarray`s:

| Key | dtype | Description |
|-----|-------|-------------|
| `ts` | `int64` | Candle open timestamps, ms epoch, ascending |
| `o`, `h`, `l`, `c` | `float64` | OHLC prices |
| `v`, `q`, `n` | `float64` | Volume, quote volume, trade count |
| `vwap` | `float64` | Volume-weighted average price |
| `vb`, `vs` | `float64` | Buy volume, sell volume |
| `_ts_start` | `int` | `ts[0]` — base for O(1) index lookup |
| `_ts_step` | `int` | `60_000` ms (constant step assumption) |
| `_len` | `int` | Number of candles |

**O(1) ts → index:** `idx = (target_ts - _ts_start) // _ts_step`. Callers
should verify `ts[idx] == target_ts` after the lookup.

## Local-cache-only rule

The package never downloads data. All reads are from local parquet files
already fetched by `candle_preloader`.
