# Spec: `draw_chart_vp_continued_width` — VP chart variant

## 1. Task summary

Add a new function `draw_chart_vp_continued_width(data)` to
`strategies/lbla_n_vp/lbla_n_vp_chart.py`. It is a variant of the existing
`draw_chart_vp(data)` (same `data` dict, same Plotly two-panel layout, peak lines,
reference lines, and two tables) with four changes:

1. **Do not draw the `width_h1` rectangles** at all.
2. **Continue each `width_h05` band into the right (price-vs-time) panel**, extending it
   horizontally to the end of `la_x`.
3. **Show the left-panel volume amounts as a robust z-score** `(v - v_median) / v_iqr`.
4. **Add two columns to the peaks table**: `height` and `prominence` expressed as robust
   z-scores.

Then append one cell to `strategies/lbla_n_vp/lbla_n_vp.ipynb` that draws a chart with
this new function.

## 2. Background and context

`draw_chart_vp` (already in the module) renders, from the `data` dict produced by
`lookback_lookahead_normalized_vp`:
- Right panel: normalized price path (`lb_p` vs `lb_x`, `la_p` vs `la_x`), vertical
  separator at `x=1.0`, horizontal current-price line at `y=0`, `y∈[-1,1]`.
- Left panel (shared y = price): low-opacity histogram of `vp_hist`, `vp_kde` overlay,
  up to 7 HVN peak lines, and per-peak `width_h1` + `width_h05` rectangles.
- Below: a peaks DataFrame and a metrics DataFrame.

The new function reuses all of this except the four changes below. Prefer factoring
shared logic so the two functions stay consistent (a private helper, or sensible reuse);
do not duplicate the whole body needlessly.

### Relevant `data` keys

| Key | Meaning |
|---|---|
| `lb_x`, `la_x`, `lb_p`, `la_p` | Right-panel time/price axes. `la_x[-1]` = end of look-ahead. |
| `bin_centers`, `vp_hist`, `vp_kde`, `bin_width` | Left-panel volume profile (price on y, density on x). |
| `hvn` | `{"poc": peak|None, "above":[…≤3], "below":[…≤3]}`; peak = `{price, prominence, width_h1, width_h05}` (widths in bins). |
| `metrics` | Cached metrics incl. `v_median`, `v_iqr`, `v_mean`, `v_stddev` (`agents/datasets/metrics_cache.md`). |
| `current_price`, `datetime`, `asset` | Title/labels. |

**Robust z-score** (per request): `z(x) = (x - v_median) / v_iqr`, using
`data["metrics"]["v_median"]` and `data["metrics"]["v_iqr"]`. Guard against
`v_iqr == 0` (e.g. fall back to `0.0` / leave the raw value) to avoid `inf`/`nan`.

## 3. Relevant conventions from `/agents/`

- **Metric keys** (`agents/datasets/metrics_cache.md`): `v_median`, `v_iqr` are the
  rolling (10080-min) median/IQR of candle volume — the basis for the robust z-score.
- **Normalized price space / time axis** (`idea_normalize_based_on_last_price_clip.md`):
  price y `0.0` = current price; `la_x[-1]` is the last look-ahead time.
- **Notebook deps** (`general/rules.md`): `plotly` already in the notebook's inline
  `%pip install` cell — keep it.
- **Script deps** (`general/rules.md`): `plotly` already in
  `strategies/lbla_n_vp/requirements.txt` — no change.
- **Placement** (`general/paths_and_files.md`): code stays under `strategies/lbla_n_vp/`.
- **Writing style** (`general/rules.md`): minimal docstrings/comments.
- **Scope** (`general/access.md`): no tests, no debugging beyond making the chart render.

## 4. Functional requirements

`draw_chart_vp_continued_width(data: dict) -> go.Figure` mirrors `draw_chart_vp`
(two-column shared-y figure, peak lines, `x=1.0` separator, `y=0` line, `y∈[-1,1]`,
legend-toggleable layers, `fig.show()`, both tables rendered, returns the figure),
with these differences:

### 4.1 No `width_h1`
- Omit all `width_h1` rectangles and their legend layer entirely.

### 4.2 `width_h05` continued into the right panel
- Keep the left-panel `width_h05` rectangle as in `draw_chart_vp` (x: `0 → peak height`,
  y: `price ± (width_h05 * bin_width)/2`), using the left panel's robust-z density axis
  (§4.3) for the `peak height` x-extent.
- **Additionally**, for each peak draw the same `width_h05` band as a horizontal
  rectangle in the **right (main) panel**, spanning the full right-panel x range
  `lb_x[0] → la_x[-1]` at `y: price ± (width_h05 * bin_width)/2`. This visualizes the
  half-width zone over the price path, continued to the end of `la_x`.
- Draw these as `go.Scatter(fill="toself", line=dict(width=0))` so they are
  legend-toggleable; group all `width_h05` traces (both panels) under one
  `legendgroup="width_h05"` (one legend click toggles the whole layer).
- POC keeps its distinct color family vs. the other peaks; skip a peak whose
  `width_h05 == 0`.

### 4.3 Robust z-score volume axis (left panel)
- Display the left-panel **volume amounts as robust z-scores**: transform the density
  x-values by `z(x) = (x - v_median) / v_iqr`.
- Apply the same transform to **both** `vp_hist` (the histogram bars — the "volume
  amounts") **and** `vp_kde` (the overlay), so the two stay on one consistent x-axis and
  the KDE still overlays the histogram. Peak height (used for the left-panel peak-line
  and `width_h05` x-extent) is then the **z-scored** KDE value at the peak
  (`z(vp_kde[idx])`, `idx = argmin(|bin_centers - price|)`).
- Keep the histogram low-opacity and the KDE sharper, as in `draw_chart_vp`. Label the
  left panel's x-axis to indicate it is a robust z-score of volume.

### 4.4 Peaks table — two added columns
- Keep the existing peaks-table columns (`label`, `price`, `height`, `prominence`,
  `width_h1`, `width_h05`, in their existing units).
- Add **`height_z`** = `z(height)` and **`prominence_z`** = `z(prominence)`, where
  `height` is the raw KDE value at the peak and `prominence` is the peak's raw
  prominence, both transformed by `z(x) = (x - v_median) / v_iqr`.
- The metrics table is unchanged.

### 4.5 Function contract
- Read-only w.r.t. `data`; no mutation, no pipeline recompute.
- Call `fig.show()`, render the peaks + metrics DataFrames, return the `go.Figure`.
- Handle empty/`None` peak lists gracefully and `v_iqr == 0` safely.

### 4.6 Notebook change (`lbla_n_vp.ipynb`)
- Append one final cell:
  `from strategies.lbla_n_vp.lbla_n_vp_chart import draw_chart_vp_continued_width`
  then `draw_chart_vp_continued_width(data)`. Reuse the `data` already built by the
  existing chart-input cell (no new input cell needed).

## 5. Non-goals / out of scope
- No changes to `draw_chart_vp` behavior (other than optional shared-helper extraction
  that preserves its output), to `lbla_n_vp.py`, or to any `packages/` code.
- No new metrics/peak/KDE computation — chart consumes existing `data`.
- No tests, no CLI, no image export.

## 6. Assumptions
- `data["metrics"]` contains `v_median` and `v_iqr` (it does, per pipeline + cache).
- `vp_hist`, `vp_kde`, `bin_centers` share length `bins_count`.
- Robust z-scored values may be negative; that is acceptable on the left x-axis.
- Width values are non-negative; `0 ⇒` skipped/degenerate rectangle.

## 7. Acceptance criteria
- `draw_chart_vp_continued_width(data)` returns a `go.Figure`, shows it, and renders both
  DataFrames.
- The figure matches `draw_chart_vp` except: **no `width_h1`**; each `width_h05` band is
  also drawn across the right panel from `lb_x[0]` to `la_x[-1]`; the left-panel
  histogram **and** KDE x-values are robust z-scores `(x - v_median)/v_iqr`.
- The peaks table has the original columns **plus** `height_z` and `prominence_z`
  (robust z-scores of the raw height and prominence).
- Legend clicks toggle each layer, including the combined `width_h05` layer.
- No exception when a peak list is empty, POC is the only peak, or `v_iqr == 0`.
- The new notebook cell draws the chart via `draw_chart_vp_continued_width(data)`.

## 8. Open questions / decisions
- **Z-scoring the KDE too (not just the histogram):** chosen so the overlay and peak
  heights remain comparable on one axis. If the user later wants only the histogram bars
  z-scored, that is a small follow-up.
- **`width_h05` kept in the left panel and continued into the right** (rather than moved
  out of the left): chosen because the request says "continue", implying it retains its
  left-panel presence and extends rightward.

## 9. Notes for the downstream coding agent
- Read all of `/agents/` first; read the existing `draw_chart_vp` and factor shared
  helpers (peak gathering, `_peak_height`, table building) so both functions stay
  consistent.
- Compute `z = lambda x: (x - v_median) / v_iqr` once, with a `v_iqr == 0` guard.
- Right-panel `width_h05` bands: `go.Scatter(fill="toself", line=dict(width=0),
  x=[lb_x[0], la_x[-1], la_x[-1], lb_x[0], lb_x[0]], y=[y0,y0,y1,y1,y0])` on `row=1,
  col=2`, `legendgroup="width_h05"`.
- Keep the function read-only w.r.t. `data` and return the figure.
- Append exactly one notebook cell; keep the `.ipynb` JSON valid.
