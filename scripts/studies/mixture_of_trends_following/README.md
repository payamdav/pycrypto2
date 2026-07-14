# mixture_of_trends_following (motf)

Multi-window trend-following alignment study: 4 look-back regression slopes + 4 look-back
volume imbalances on 1m candles form a binary `trigger` (all 8 positive); evaluates how well
`trigger` predicts a positive look-ahead slope over 4 forward horizons. Spec:
`ai_chats/mixture_of_trends_following_spec.md`.

## Run

```bash
python3 scripts/studies/mixture_of_trends_following/build_cache.py [tag]
python3 scripts/studies/mixture_of_trends_following/report.py [tag]
python3 scripts/studies/mixture_of_trends_following/draw.py [tag]
# or the full pipeline in one step:
python3 scripts/studies/mixture_of_trends_following/run_all.py [tag]
```

`tag` defaults to `default`. All outputs go to `data/mixture_of_trends_following/{tag}/`
(created on first run). `run_all.py` runs the build step across assets with multiprocessing
when the config lists more than one asset.

## Config

Resolution order: `data/mixture_of_trends_following/{tag}/config.json` if present, else
`scripts/studies/mixture_of_trends_following/config.json`. To override one tag only, copy the
study config into the tag folder and edit it there.

| Key | Default | Meaning |
|---|---|---|
| `assets` | `["btcusdt"]` | asset symbol(s) |
| `date_from` / `date_to` | `2024-01-01 00:00:00` / `2026-06-30 23:59:00` | evaluation date range (UTC) |
| `slope_windows` | `[10080, 1440, 240, 60]` | look-back windows (minutes) for the 4 slope features: 7d/1d/4h/1h |
| `imbalance_windows` | `[10080, 1440, 240, 60]` | look-back windows (minutes) for the 4 imbalance features |
| `label_windows` | `[60, 120, 180, 240]` | look-ahead windows (minutes) for the 4 label slopes |
| `l_slopes_threshold` | `0.00001` | strict threshold on a label slope to count as positive |
| `draw_from` / `draw_to` | `2026-04-15 00:00:00` / `2026-04-16 00:00:00` | chart time slice (UTC) |

## Pipeline

1. **`build_cache.py`** — per asset: load the full candle range (`packages/candle_loader`),
   sanitize `vwap`, compute the 17-column cache, save `motf_cache_{asset}.npy`.
2. **`report.py`** — per asset: restrict to normal items in `[date_from, date_to]`, print
   counts/confusion-matrix/phi/precision/recall/baseline/lift + per-horizon precision table,
   save `motf_report_{asset}.json`.
3. **`draw.py`** — per asset: slice the cache to `[draw_from, draw_to]`, save an interactive
   3-pane `motf_chart_{asset}.html`.

## Indicators used

`linreg_slope` and `volume_imbalance` (`packages/indicators`, documented in
`agents/packages/indicators.md`). Relative slope = `linreg_slope(vwap, W) / vwap`
(fractional change per candle).

## Normal items

Longest look-back `LB = max(slope_windows + imbalance_windows)`, longest look-ahead
`LA = max(label_windows)` (10080 / 240 under defaults). Normal anchor range:
`LB-1 <= i < n-LA`. `report.py` additionally restricts to `ts` in `[date_from, date_to]`.

## Tag-folder files

### `config.json` (optional)

Tag-specific override of the study config — same schema as the study-folder `config.json`.

### `motf_cache_{asset}.npy`

`float64` ndarray, shape `(n_candles, 17)`, one row per candle (full asset range, not date-filtered).
Column 0 == candle `ts`; feature/flag columns (1–11) are look-back only (no future leak); label
columns (12–16) use look-ahead data (excluded from criterion 3 by design). Binary columns store
`1.0`/`0.0`; all comparisons strict.

| Col | Name | Definition |
|---|---|---|
| 0 | `ts` | candle open time, ms epoch |
| 1–4 | `slope_1..4` | relative OLS slope of vwap, windows 10080/1440/240/60 |
| 5 | `slopes` | 1 if `slope_1..4` all `> 0` |
| 6–9 | `imbalance_1..4` | volume imbalance, windows 10080/1440/240/60 |
| 10 | `imbalances` | 1 if `imbalance_1..4` all `> 0` |
| 11 | `trigger` | 1 if `slopes==1 and imbalances==1` |
| 12–15 | `l_slope_1..4` (label) | look-ahead relative slope, windows 60/120/180/240; `l_slope_W[i] = slope_W[i+W]`, tail edge-filled |
| 16 | `l_slopes` (label) | 1 if any `l_slope_1..4 > l_slopes_threshold` |

### `motf_report_{asset}.json`

```json
{
  "n_total": 0, "n_eval": 0,
  "slopes": {"count": 0, "pct": 0.0}, "imbalances": {...}, "trigger": {...}, "l_slopes": {...},
  "confusion": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
  "phi": 0.0, "precision": 0.0, "recall": 0.0, "baseline": 0.0, "lift": 0.0,
  "per_horizon": [{"window": 60, "precision": 0.0, "baseline": 0.0, "lift": 0.0}, "..."]
}
```

`n_eval` = normal items inside `[date_from, date_to]`. Confusion matrix: rows = `trigger`
predicted, cols = `l_slopes` actual (`tp`/`fp` top row, `fn`/`tn` bottom row); cells sum to
`n_eval`. `phi` = Pearson correlation of the two binaries. `precision = P(l_slopes=1|trigger=1)`,
`recall = P(trigger=1|l_slopes=1)`, `baseline = P(l_slopes=1)`, `lift = precision / baseline`.
`per_horizon[k]` repeats precision/baseline/lift for `trigger` vs `l_slope_k > threshold` alone.
Any ratio with a zero denominator is `null`.

### `motf_chart_{asset}.html`

Interactive Plotly chart, `[draw_from, draw_to]` slice, 3 stacked panes sharing an x-axis
(UTC datetime) with unified hover/crosshair and native zoom/pan:

1. `vwap` line, with a bullet marker on each candle where `trigger == 1`.
2. `slope_1..4` lines + zero line.
3. `imbalance_1..4` lines + zero line.

Each window gets one fixed color, reused across the slope and imbalance panes.
