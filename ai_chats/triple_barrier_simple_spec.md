# Spec: Trader Indicator — triple_barrier_simple

## 1. Task Summary

Create the first trader indicator, `triple_barrier_simple`, under a new `packages/traders_indicators/` package. Given a 1D price array, it labels each item with the outcome of a simulated long trade: `1.0` if price first reaches the upper barrier (take profit), `-1.0` if it first reaches the lower barrier (stop loss), `0.0` if neither is hit within the look-ahead window (time limit). Output is a 1D `float64` ndarray, same length as the input. Numba `@njit`. Also create a CLI test script that runs the indicator on real candle `vwap` prices and prints label counts and timing.

## 2. Background and Context

Trader indicators simulate a single trade per candle for back-test labeling (ML labels / studies) — see `agents/packages/traders_indicators.md`. `packages/traders_indicators/` does not exist yet; this task creates it. This indicator is the classic triple-barrier labeling method with fixed symmetric-style bps barriers and a vertical (time) barrier, operating on a plain 1D price series (typically candle `vwap`), single back-test price per item.

## 3. Relevant Conventions from `/agents/`

- `agents/packages/traders_indicators.md` — core semantics: output per item, same length as input; entry candle never checked for exit; look-ahead window = max candles checked for exit; exit codes `1` = take profit, `-1` = stop loss, `0` = time limit; barriers as bps of entry price; numba `@njit` for the simulation loop; commission not involved here (default `0`, and this indicator returns only the exit cause).
- Per-indicator description file `agents/packages/traders_indicators/triple_barrier_simple.md` — already created by spec_writer alongside this spec; update it only if implementation deviates (it must not without approval).
- `agents/general/rules.md` — `requirements.txt` next to the code; shortest useful writing style.
- `agents/packages/indicators.md` style — 1D `float64` in, newly allocated `float64` out, explicit loops inside the jitted body.
- No nan anywhere in the output.

## 4. Functional Requirements

### Package and files

- New package root: `packages/traders_indicators/`
  - `__init__.py` — exports: `from packages.traders_indicators.triple_barrier_simple import triple_barrier_simple`
  - `triple_barrier_simple.py` — the indicator
  - `requirements.txt` — `numpy`, `numba`

### API

```python
@njit
def triple_barrier_simple(prices, upper_barrier_bps=20.0, lower_barrier_bps=20.0,
                          look_ahead=240, next_entry=True):
    ...  # returns 1D float64, len == len(prices)
```

- `prices`: 1D `np.float64`, oldest → newest (one price per candle, e.g. `vwap`).
- `upper_barrier_bps` / `lower_barrier_bps`: positive numbers, bps of entry price.
- `look_ahead`: max items checked for exit after entry.
- `next_entry`: `True` → entry price is the **next** item's price; `False` → the current item's price.

### Computation (per index `i`, n = len(prices))

1. `entry_idx = i + 1` if `next_entry` else `i`. If `entry_idx >= n` → `out[i] = 0.0` (no entry possible).
2. `entry = prices[entry_idx]`
   - `upper = entry * (1.0 + upper_barrier_bps / 10_000.0)`
   - `lower = entry * (1.0 - lower_barrier_bps / 10_000.0)`
3. Scan `j = entry_idx + 1 … min(entry_idx + look_ahead, n - 1)` in order (entry item itself is never checked):
   - `prices[j] >= upper` → `out[i] = 1.0`, stop.
   - else `prices[j] <= lower` → `out[i] = -1.0`, stop.
4. No hit by the end of the scan (window exhausted or end of array) → `out[i] = 0.0`.

Barrier touch is inclusive (`>=` / `<=`). At one item the upper check runs first (moot for positive barriers — a single price cannot hit both).

### Output

- 1D `np.float64`, length `n`, values only `{1.0, -1.0, 0.0}` — no nan, no padding; tail items with short or empty windows simply resolve by the scan rules above.
- Empty input → empty `float64` array.

### Test script

- Location: `scripts/tests/triple_barrier_simple/test_triple_barrier_simple.py`, with its own `requirements.txt` (`numpy`, `numba`, plus `packages/candle_loader` deps).
- CLI arguments: `asset` (e.g. `btcusdt`), `date_from`, `date_to` (`"YYYY-MM-DD HH:MM:SS"` UTC, formats accepted by `load_candles`) — required; `upper_barrier_bps` (default `20`), `lower_barrier_bps` (default `20`), `look_ahead` (default `240`), `next_entry` (default `True`) — optional.
- Flow:
  1. Load candles via `packages.candle_loader.load_candles(asset, date_from, date_to)`; price = `vwap` column (index 8), as contiguous `float64`.
  2. Run `triple_barrier_simple(prices, upper_barrier_bps, lower_barrier_bps, look_ahead, next_entry)`.
  3. Print: number of candles, count of `1.0`, count of `-1.0`, count of `0.0`, and the indicator's time consumption in seconds (time the indicator call; measure the second of two calls or note JIT compile is included — either way state what is measured in the print).
- Prints: single short lines per `agents/general/rules.md` writing style.

## 5. Non-Goals / Out of Scope

- No aggregate back-test metrics or helper reporting functions.
- No commission handling (returns exit cause only; no returns/pnl output).
- No candle-array / ecandles input variant — 1D prices only, single back-test price.
- No short-position variant.
- No test notebooks and no unit-test suite — only the CLI test script above.

## 6. Assumptions

- The trade is long-only: upper barrier = take profit, lower = stop loss.
- Per package default, the entry item is never checked for exit; the look-ahead window is the `look_ahead` items **after the entry item** (`entry_idx + 1 … entry_idx + look_ahead`).
- Bps parameters may be passed as int or float; treated as float.
- Prices are strictly positive; no validation required inside the jitted function.

## 7. Acceptance Criteria

1. `triple_barrier_simple` compiles under `@njit`; explicit loops; returns newly allocated 1D `float64` of input length.
2. `next_entry=True` shifts entry to `prices[i+1]`; last index (and any index with `entry_idx >= n`) yields `0.0`.
3. First-touch ordering respected: earliest scanned item that meets a barrier decides the label.
4. Inclusive barrier comparison (`>=` upper → 1, `<=` lower → -1).
5. Window exhaustion or end-of-array without a touch → `0.0`; output contains only `{1.0, -1.0, 0.0}`.
6. `__init__.py` export and `requirements.txt` in place; behavior matches `agents/packages/traders_indicators/triple_barrier_simple.md`.
7. Test script runs end-to-end from the CLI (e.g. `python3 scripts/tests/triple_barrier_simple/test_triple_barrier_simple.py btcusdt "2026-06-01 00:00:00" "2026-06-07 23:59:00"`), uses candle `vwap` as prices, and prints candle count, counts of `1.0` / `-1.0` / `0.0`, and indicator time consumption. Label counts sum to the candle count.

## 8. Open Questions

None blocking. If the look-ahead window should instead be counted from the current item `i` (not the entry item), or the entry item itself should be scanned, say so before implementation — the defaults above follow `agents/packages/traders_indicators.md`.

## 9. Notes for the Downstream Coding Agent

- Keep the scan loop `O(look_ahead)` per item with early exit — no vectorized full-matrix construction.
- `packages/__init__.py` already exists; only the new sub-package files are needed.
- Do not modify `packages/indicators/` or any other package.
- Verify the work by running the test script (default barriers, a real asset/date range); first run downloads the asset's candle cache via `candle_loader`.
