# Spec: mixture_of_trends_following (motf)

Implementation-ready spec for `code_writer`. Source request: multi-window trend-following alignment study. Fits `agents/general/strategy_study_guidelines.md`; everything unstated there applies.

## 1. Task Summary

Build study `mixture_of_trends_following`: compute 4 look-back regression slopes and 4 look-back volume imbalances on 1m candles, form a binary `trigger` when all 8 are positive, label each point with look-ahead slopes, then report how well `trigger=1` predicts `l_slopes=1` (stats + phi correlation + confusion matrix) and draw trigger points on a price chart.

## 2. Background & Context

- Trend-following idea: alignment of trend (slope) and order-flow (imbalance) across 4 time scales (7d / 1d / 4h / 1h) as an entry trigger; predictive target is a positive regression slope in any of 4 forward horizons (1h–4h).
- Candles come from `packages/candle_loader` (11 columns; `vwap`, `vb`, `vs` are the ones used here).
- No rolling linear-regression slope or volume-imbalance indicator exists yet in `packages/indicators` — both are standard and reusable, so they are built there (not in the study folder).

## 3. Relevant Conventions from `/agents/`

- `agents/general/strategy_study_guidelines.md` — locations, tags, config resolution, defaults, cache layout, normal items, evaluation, charts, timing.
- `agents/packages/indicators.md` — indicator contract: float64 in/out, full candle length, backfill (no nan/0 padding), `@njit` with explicit loops.
- `agents/ideas/idea_look_back_look_ahead.md` — look-back window includes current candle; look-ahead starts at next candle; anchor range.
- `agents/datasets/huggingface_candles.md` — column semantics (`vb` = buy volume, `vs` = sell volume, `vwap`).
- `agents/general/rules.md` — `requirements.txt` per script folder, package doc duty, terse writing style.

## 4. Functional Requirements

### 4.1 Locations & files

- Code: `scripts/studies/mixture_of_trends_following/`
- Per-execution files: `data/mixture_of_trends_following/{tag}/`, file prefix `motf_`.
- Every runtime script takes the tag as first CLI argument; missing → `default`. Create the tag folder if absent.

### 4.2 Config

`config.json` in the study folder holds the defaults below; scripts resolve `data/mixture_of_trends_following/{tag}/config.json` first, else the study-folder copy. Each pipeline step is a function taking the full params dict.

```json
{
  "assets": ["btcusdt"],
  "date_from": "2024-01-01 00:00:00",
  "date_to": "2026-06-30 23:59:00",
  "slope_windows": [10080, 1440, 240, 60],
  "imbalance_windows": [10080, 1440, 240, 60],
  "label_windows": [60, 120, 180, 240],
  "l_slopes_threshold": 0.00001,
  "draw_from": "2026-04-15 00:00:00",
  "draw_to": "2026-04-16 00:00:00"
}
```

All datetimes `"YYYY-MM-DD HH:MM:SS"` UTC.

### 4.3 New package indicators (in `packages/indicators`)

- `linreg_slope(array, window=60)` — rolling OLS slope of `array` against `x = 0..window-1` over the look-back window (inclusive of current item). Raw units: array-units per candle. Must be O(n) via incremental running sums (not O(n·window)); `@njit`, explicit loops; first `window-1` values backfilled with the first computed value.
- `volume_imbalance(vb, vs, window=60)` — `(Σvb − Σvs) / (Σvb + Σvs)` over the look-back window; `0.0` where the denominator is `0`; same conventions.
- Document both in `agents/packages/indicators.md` (package-doc rule).

### 4.4 Cache build — `build_cache.py {tag}`

Per asset: load the **full** candle range with `load_candles(asset)`, sanitize `vwap` (any non-finite value or `v == 0` row → previous valid `vwap`; leading rows → first valid), compute the cache, save `motf_cache_{asset}.npy`.

**Relative slope** (user-confirmed): `rel_slope_W = linreg_slope(vwap, W) / vwap` — per-candle fractional change, divisor is the window's own last `vwap`.

Cache: 2D float64, shape `(n_candles, 17)`:

| Col | Name | Definition |
|---|---|---|
| 0 | `ts` | candle open time (ms) |
| 1–4 | `slope_1..4` | rel. slope, windows 10080 / 1440 / 240 / 60 |
| 5 | `slopes` | 1 if all `slope_1..4 > 0` else 0 |
| 6–9 | `imbalance_1..4` | volume imbalance, windows 10080 / 1440 / 240 / 60 |
| 10 | `imbalances` | 1 if all `imbalance_1..4 > 0` else 0 |
| 11 | `trigger` | 1 if `slopes == 1` and `imbalances == 1` else 0 |
| 12–15 | `l_slope_1..4` — **LABEL** | rel. slope over look-ahead windows 60 / 120 / 180 / 240 |
| 16 | `l_slopes` — **LABEL** | 1 if any `l_slope_k > l_slopes_threshold` else 0 |

Label construction: the look-ahead window `[i+1, i+W]` equals the look-back window ending at `i+W`, so `l_slope_W[i] = rel_slope_W[i+W]` (pure left-shift by `W`). Tail rows where `i+W >= n` are edge-filled with the last computed value — harmless because evaluation is restricted to normal items. Binary columns store `1.0/0.0`. All comparisons strict (`> 0`, `> threshold`).

### 4.5 Normal items

`LB = 10080` (longest look-back), `LA = 240` (longest look-ahead). Normal anchor range: `LB-1 <= i < n-LA`. Evaluation additionally restricts `ts` to `[date_from, date_to]`.

### 4.6 Report — `report.py {tag}`

Per asset, on normal items within the date range, print (one line each) and save to `motf_report_{asset}.json`:

- counts: total candles, evaluated points, and count + % with `slopes==1`, `imbalances==1`, `trigger==1`, `l_slopes==1`.
- trigger→l_slopes prediction: confusion matrix (rows = trigger predicted, cols = l_slopes actual: TP FP / FN TN), phi coefficient (= Pearson on the two binaries), precision `P(l=1|t=1)`, recall `P(t=1|l=1)`, baseline `P(l=1)`, lift = precision / baseline.
- per-horizon: precision of `trigger` vs each `l_slope_k > threshold` (compact 4-row table).

### 4.7 Drawing — `draw.py {tag}`

Per asset: slice cache to `[draw_from, draw_to]`; interactive Plotly figure, 3 stacked panes with shared x (UTC datetime), unified hover/crosshair, native zoom/pan:

1. `vwap` line with small bullet markers on `trigger == 1` candles
2. `slope_1..4` lines + zero line
3. `imbalance_1..4` lines + zero line

Save `motf_chart_{asset}.html` to the tag folder and print the path.

### 4.8 Pipeline & docs

- `run_all.py {tag}` runs build → report → draw. Assets are independent: run the build step via multiprocessing when `len(assets) > 1`.
- Study-folder `README.md`: runbook (steps, tag usage, config override) + documentation of every tag-folder file (columns and properties). `requirements.txt` in the study folder.
- Every sub-task (each indicator, cache save, report, chart) prints a single elapsed-time line.

## 5. Non-Goals / Out of Scope

- No ML model training or optimization.
- No parameter-range sweeps / run ids — one run per tag.
- No cross-asset cumulative report (not requested).
- No Jupyter notebook (not requested).
- No indicators beyond the two named package additions.

## 6. Assumptions

- Slope is **relative** (per-candle fractional change) — confirmed by user 2026-07-14; `0.00001` ≈ 1.44%/day drift. Threshold and windows are config defaults, user-tunable.
- Look-back includes the current candle; look-ahead excludes it (starts next candle) — per guidelines.
- Imbalance denominator `Σvb + Σvs = Σv`; zero-volume windows → imbalance `0.0`.
- Cache format `.npy`; config format json; draw default range = 1 day containing the guideline default datetime.

## 7. Acceptance Criteria

1. `python3 build_cache.py` produces `data/mixture_of_trends_following/default/motf_cache_btcusdt.npy`, shape `(n, 17)`, all values finite, col 0 identical to candle `ts`.
2. Consistency: `trigger==1` ⟺ `slopes==1 ∧ imbalances==1`; `slopes==1` ⟺ all 4 slope cols `> 0` (same for imbalances).
3. No future leak: recomputing any feature column (1–11) at index `i` from `candles[0:i+1]` alone reproduces the cached value, for sampled `i` in the normal range. Labels (12–16) are exempt by design.
4. Label shift correct: `l_slope_W[i] == rel_slope_W[i+W]` for sampled normal `i`.
5. `report.py` prints all §4.6 lines and writes valid json; confusion-matrix cells sum to the evaluated-point count.
6. `draw.py` writes an HTML chart whose bullets appear only at `trigger == 1` candles.
7. A `config.json` placed in the tag folder (e.g. different `draw_from`) overrides the study-folder one.
8. Both new indicators follow the package contract (length, backfill, float64, `@njit`) and are documented in `agents/packages/indicators.md`.
9. Every step prints elapsed-time lines; all output terse.

## 8. Open Questions

None — slope-units ambiguity resolved (relative).

## 9. Notes for the Coding Agent

- O(n) rolling OLS: keep running `S1 = Σy` and `Sx = Σ k·y[·+k]`; on slide, `Sx ← Sx − S1 + y_out + (window−1)·y_in`, then `S1 ← S1 − y_out + y_in`; slope = `(W·Sxy − ΣxΣy)/(W·Σx² − (Σx)²)` form with constant `x` sums precomputed. float64 running sums are fine at n≈1.3M; optionally re-anchor sums every ~100k steps.
- Compute `rel_slope` once per **distinct** window `{10080, 1440, 240, 120, 60, 180}` and reuse: windows 60 and 240 serve both a feature column and (shifted) a label column.
- `load_candles` returns `(n, 11)` float64; column indices per `agents/packages/candle_loader.md` (`ts`=0, `v`=5, `vwap`=8, `vb`=9, `vs`=10).
- Datetime→ms: parse as UTC; slice arrays with `np.searchsorted` on `ts`.
- Numba warm-up compiles once per process — harmless for the default single-asset run; keep njit functions importable at module level so multiprocessing workers reuse them.
