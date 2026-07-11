# candle_loader

The single, authoritative way to load 1-minute candles in this project. Downloads each asset's full candle history from the HuggingFace dataset (`payamdavaee/candles`) into one local parquet file per asset, then serves all loads from that local file.

## Import

```python
from packages.candle_loader import local_cache, load_candles
```

## `local_cache(assets)`

Ensures `CWD/data/{asset}_1m_all.parquet` exists for every asset in the list (a single asset string is also accepted). Assets whose file is already present are skipped — no re-download ever happens. Each asset prints one report line (file path + elapsed time).

## `load_candles(asset, date_from=None, date_to=None)`

Loads candles for one asset from the local file. Calls `local_cache(asset)` internally, so no manual pre-download is needed and repeated calls never re-download.

| Parameter | Type | Description |
|-----------|------|-------------|
| `asset` | `str` | Asset symbol, e.g. `"btcusdt"` (case-insensitive) |
| `date_from` | `None` \| `str` \| `int` | Inclusive start. `None` = first available candle. String like `"2026-05-01 13:55:44"` (UTC, seconds truncated to zero) or 13-digit unix ms timestamp. |
| `date_to` | `None` \| `str` \| `int` | Inclusive end, same formats. `None` = last available candle. |

Returns an `(n, 11)` `float64` ndarray holding all candle columns, in this order:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-------|---|---|---|---|---|---|---|---|---|---|----|
| Column | `ts` | `o` | `h` | `l` | `c` | `v` | `q` | `n` | `vwap` | `vb` | `vs` |

`ts` is ms epoch. Prints one report line with shape, first/last timestamp, and load time (download time excluded).

## Example

```python
data = load_candles("btcusdt", "2026-06-01", "2026-06-07 23:59:00")
ts, close, vwap = data[:, 0], data[:, 4], data[:, 8]
```
