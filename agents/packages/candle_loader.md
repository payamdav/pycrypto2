# candle_loader

Loads OHLCV candlestick data for crypto assets from a Hugging Face parquet dataset into a NumPy array. Always includes `ts` (epoch ms, float64) as column 0; requested columns follow.

## Import

```python
from packages.candle_loader import load_candles
```

## `load_candles(asset, date_from, date_to, columns)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `asset` | `str` | Asset symbol, e.g. `"btcusdt"` (case-insensitive) |
| `date_from` | `str` | Start timestamp, e.g. `"2024-01-01"` (inclusive) |
| `date_to` | `str` | End timestamp, e.g. `"2024-12-31"` (inclusive) |
| `columns` | `list[str]` | One or more column names to load (see below) |

**Valid column names:** `o`, `h`, `l`, `c`, `v`, `q`, `n`, `vwap`, `vb`, `vs`

- Do **not** include `ts` in `columns` — it is always prepended automatically as column 0.
- Returns a `np.ndarray` of shape `(n_rows, 1 + len(columns))`, dtype `float64`.
- Prints a summary line: shape, first/last timestamp, elapsed time, and column index map.

## Example

```python
data = load_candles("btcusdt", "2024-01-01", "2024-06-01", ["c", "v"])
# data[:, 0]  → ts (epoch ms)
# data[:, 1]  → close price
# data[:, 2]  → volume
```
