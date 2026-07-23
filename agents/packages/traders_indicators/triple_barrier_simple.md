# triple_barrier_simple

Triple-barrier labeling trader indicator (long trade per item). Lives in `packages/traders_indicators/triple_barrier_simple.py`; general trader-indicator semantics per `agents/packages/traders_indicators.md`.

## Import

```python
from packages.traders_indicators import triple_barrier_simple
```

## `triple_barrier_simple(prices, upper_barrier_bps=20.0, lower_barrier_bps=20.0, look_ahead=240, next_entry=True)`

| Parameter | Type | Description |
|---|---|---|
| `prices` | 1D `float64` ndarray | Price per candle (e.g. `vwap`), oldest → newest |
| `upper_barrier_bps` | float | Take-profit barrier, bps of entry price |
| `lower_barrier_bps` | float | Stop-loss barrier, bps of entry price |
| `look_ahead` | int | Max items checked for exit after the entry item |
| `next_entry` | bool | `True` → entry = next item's price; `False` → current item's price |

## Output

1D `float64` ndarray, same length as `prices`. Per index `i` (exit-cause codes):

- `1.0` — take profit: a scanned price first reached `entry * (1 + upper_barrier_bps/1e4)` (inclusive).
- `-1.0` — stop loss: a scanned price first reached `entry * (1 - lower_barrier_bps/1e4)` (inclusive).
- `0.0` — time limit: no barrier hit within the window, end of array reached, or no entry item exists (`next_entry=True` at the last index).

## Behavior

- Entry item: `i+1` when `next_entry=True`, else `i`. The entry item itself is never checked for exit; the scan covers `entry_idx + 1 … entry_idx + look_ahead` (clipped at array end), first touch wins.
- Single back-test price per item; upper barrier checked before lower at the same item (moot for positive barriers).
- `@njit` compiled; output has no nan and only values `{1.0, -1.0, 0.0}`.
