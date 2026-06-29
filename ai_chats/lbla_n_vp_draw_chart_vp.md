# Spec: `draw_chart_vp` — LBLA Normalized VP Chart

## 1. Task summary

Create `strategies/lbla_n_vp/lbla_n_vp_chart.py` exposing a function
`draw_chart_vp(data)` that renders one interactive, Colab-friendly figure from the
`data` dict produced by `lookback_lookahead_normalized_vp` in
`strategies/lbla_n_vp/lbla_n_vp.py`. Then append two cells to
`strategies/lbla_n_vp/lbla_n_vp.ipynb`: one defining the input variables + datetime
string, and one calling `draw_chart_vp(data)`.

## 2. Background and context

`lookback_lookahead_normalized_vp` runs the pipeline
`lb_la_n_base → append_cached_metrics → vp_analysis → vp_hvn` and returns a single
`data` dict. The chart visualizes one anchor minute: the normalized price path
(look-back + look-ahead) on the right, the volume profile (histogram + KDE + HVN
peaks) on the left sharing the price axis, and two tables below.

### `data` keys consumed by the chart

| Key | Type | Meaning |
|---|---|---|
| `lb_x`, `la_x` | `float64[]` | Time axes. `lb_x∈[0,1)`, `la_x∈[1, 1+(look_ahead-1)/look_back]`. Boundary (current time) = `1.0`. |
| `lb_p`, `la_p` | `float64[]` | Normalized prices clipped to `[-1,1]` (current price → `0.0`). |
| `bin_centers` | `float64[]` | KDE bin-center prices in normalized space `[-1,1]`. |
| `vp_hist` | `float64[]` | Raw volume-weighted histogram counts per bin. |
| `vp_kde` | `float64[]` | Smoothed (convolved) density per bin. |
| `bin_width` | `float` | Width of one bin in normalized-price units. |
| `hvn` | dict | `{"poc": peak|None, "above": [peak…≤3], "below": [peak…≤3]}`. |
| `metrics` | dict | `{col: float}` cached metrics at the anchor (e.g. `v_median`, `v_iqr`, `v_mean`, `v_stddev`). |
| `current_price`, `current_ts`, `last_candle_ts`, `datetime` | — | Labels / annotations. |

A **peak record** is `{"price": float, "prominence": float, "width_h1": float,
"width_h05": float}`. `width_h1`/`width_h05` are in **bins**; multiply by
`bin_width` for normalized-price units (`kde_tools.md`).

## 3. Relevant conventions from `/agents/`

- **Normalized space** (`idea_normalize_based_on_last_price_clip.md`,
  `kde_tools.md`): on the price axis `0.0` = current price, `>0` above, `<0` below,
  clipped extremes at `±1.0`.
- **Time axis** (`idea_normalize_based_on_last_price_clip.md`): `t=1.0` is
  `current_time` (the moment the last look-back candle closes); it splits past
  (look-back) from future (look-ahead).
- **Notebook deps** (`general/rules.md`): every package used by a notebook must be
  installed inline via `%pip install` before imports. Add `plotly` to the
  notebook's existing `%pip install` cell.
- **Script deps** (`general/rules.md`): add `plotly` to
  `strategies/lbla_n_vp/requirements.txt`.
- **Placement** (`general/paths_and_files.md`): strategy-specific code lives under
  `strategies/lbla_n_vp/`. Correct for both the new module and the notebook cells.
- **Writing style** (`general/rules.md`): keep docstrings/comments minimal.
- **Scope** (`general/access.md`): do not write/run tests (not requested); do not
  debug beyond making the chart render.

## 4. Functional requirements

### 4.1 Library & interactivity
- Use **Plotly** (`plotly.graph_objects` + `plotly.subplots.make_subplots`). It is
  native in Colab and supports toggling traces on/off by clicking legend entries —
  satisfying "turn on or off drawings by clicking on its chart map."
- Group related drawings with `legendgroup` + a single legend entry per group so
  one click toggles a whole layer. Suggested toggle groups: `histogram`, `kde`,
  `look-back`, `look-ahead`, `POC`, `peaks (lines)`, `width_h1`, `width_h05`.

### 4.2 Layout
- The **chart** is one Plotly figure with a shared **price (y) axis** between the VP
  panel and the main panel:
  - **Left panel** = volume profile (price on y, volume/density on x).
  - **Right panel** = price-vs-time (wider; e.g. width ratio ~1:3).
  - Build with `make_subplots(rows=1, cols=2, shared_yaxes=True)`.
- The **tables are NOT part of the figure.** Render them separately as pandas
  DataFrames displayed below the chart in the notebook (see 4.5).
- Y axis fixed to `[-1, 1]` on both price panels.

### 4.3 Main panel (price vs time)
- Plot `lb_p` vs `lb_x` (look-back) and `la_p` vs `la_x` (look-ahead) as two lines.
- Use the raw `lb_x`/`la_x` values **unchanged** (no shift).
- **Vertical separator at `x = 1.0`** = current time, marking past↔future.
- **Horizontal line at `y = 0`** = current price.
- Y range `[-1, 1]`.

### 4.4 Left panel (volume profile, shared y = price)
- **Horizontal histogram** of `vp_hist` across `bin_centers` (bars oriented along
  price/y), drawn in a **light/low-opacity color**.
- **`vp_kde` line** drawn over the histogram (sharper color).
- **7 HVN peaks** = `hvn["poc"]` + up to 3 `hvn["above"]` + up to 3 `hvn["below"]`
  (skip `None`/missing): one **horizontal line** at each peak's `price`.
- **Peak height** = the KDE value at the peak price = `vp_kde[idx]`, where `idx` is
  the bin whose center equals the peak `price`
  (`idx = argmin(abs(bin_centers - price))`).
- For each peak, two rectangles. In this panel **x = density** and **y = price**, so:
  - **x-extent (the rectangle's "height")** = `0 → peak height` (the KDE value),
    identical for both rectangles of a peak.
  - **y-extent (the rectangle's "width")** = centered on the peak `price`:
    - `width_h1` rectangle: y from `price - (width_h1*bin_width)/2` to
      `price + (width_h1*bin_width)/2`; lighter color.
    - `width_h05` rectangle: thinner band `width_h05*bin_width` centered on `price`;
      **drawn over** the `width_h1` rectangle in a **sharper color**.
  - Distinguish POC visually from the other six peaks (e.g. distinct color/label).

### 4.5 Tables (pandas DataFrames, printed below the chart)
- **Peaks table**: one row per displayed peak with columns
  `label` (POC / above / below), `price`, `height` (KDE value at peak),
  `prominence`, `width_h1`, `width_h05`. State width units in the header (default:
  normalized-price units = `width × bin_width`).
- **Metrics table**: every key/value in `data["metrics"]` (all cached metrics
  appended by `append_cached_metrics` for this candle), two columns name/value.
- Display both via the notebook (e.g. `display(df)` / last-expression render); do
  not embed them in the Plotly figure.

### 4.6 Function contract
- Signature: `def draw_chart_vp(data: dict) -> go.Figure:`.
- Show the chart (`fig.show()`) and render the two DataFrames so all three outputs
  appear when called from a Colab cell; return the Figure for further customization.
- Read only from `data`; do not mutate it; do not recompute pipeline values (peak
  height lookup from `vp_kde`/`bin_centers` is allowed — it is a read, not a
  recompute).
- Handle empty/`None` peak lists gracefully (POC may exist with no above/below).

### 4.7 Notebook changes (`lbla_n_vp.ipynb`)
- Add `plotly` to the first cell's `%pip install` line.
- Append a cell defining all chart inputs and the datetime string, mirroring the
  existing input-parameter cell (`asset`, `look_back`, `look_ahead`, `k`,
  `bins_count`, `bandwidth`, `kernel_type`, `kde_ignore_borders`, and a
  `dt_str` / `datetime` string), then producing `data` via
  `lookback_lookahead_normalized_vp(...)` if a `data` dict is not already in scope.
- Append a final cell:
  `from strategies.lbla_n_vp.lbla_n_vp_chart import draw_chart_vp` then
  `draw_chart_vp(data)`.

## 5. Non-goals / out of scope
- No changes to `lbla_n_vp.py` pipeline logic or to any `packages/` code.
- No new metrics, peak-finding, or KDE computation — chart consumes existing `data`.
- No tests, no CLI, no saving images to disk.
- No multi-anchor / animation views — one anchor per figure.

## 6. Assumptions
- Plotly renders inline in Colab without extra config (default renderer is fine).
- `data` already contains a successful pipeline run (all keys present, no exception).
- `vp_hist`, `vp_kde`, `bin_centers` share length `bins_count`.
- Cached metrics are scalar floats (per `append_cached_metrics`).
- Width values are non-negative; a zero width yields a degenerate (skipped or
  zero-height) rectangle.

## 7. Acceptance criteria
- `strategies/lbla_n_vp/lbla_n_vp_chart.py` defines `draw_chart_vp(data)` returning
  a `go.Figure`, showing it, and rendering the two DataFrames.
- Running the two new notebook cells top-to-bottom produces, in Colab: one
  interactive Plotly figure with a price-vs-time main panel (y∈[-1,1]); a
  shared-axis VP panel with low-opacity histogram + KDE overlay; vertical
  separator at `x=1.0`; horizontal current-price line at `y=0`; up to 7 peak lines,
  each with `width_h1`/`width_h05` rectangles (x: 0→peak height, y: width band
  centered on price; h05 over h1, sharper) — followed by the peaks DataFrame and
  metrics DataFrame printed below the chart.
- Peaks table includes a `height` column (KDE value at the peak).
- Legend clicks toggle drawing layers on/off.
- `plotly` added to both `requirements.txt` and the notebook `%pip install` cell.
- No exception when a peak list is empty or POC is the only peak.

## 8. Resolved decisions (from user)
1. **X-axis:** use raw `lb_x`/`la_x` (no shift); vertical past/future separator at
   `x = 1.0` (current time). Current price shown by the horizontal line at `y = 0`.
2. **Charting library:** Plotly.
3. **Tables:** rendered as pandas DataFrames printed below the chart — NOT embedded
   in the Plotly figure.
4. **Peak rectangles:** centered on the peak `price` along the price (y) axis with
   thickness = `width × bin_width`; length along the density (x) axis = `0 → peak
   height`, where peak height = KDE value at the peak price (`vp_kde[idx]`).
5. **Peaks table:** include the peak `height` column.

## 9. Notes for the downstream coding agent
- Read all of `/agents/` before coding (`CLAUDE.md`, `AGENTS.md`).
- Build the chart with `make_subplots(rows=1, cols=2, shared_yaxes=True,
  column_widths=[~0.25, ~0.75])`. Tables are separate DataFrames, not subplots.
- Reverse the VP panel's x-axis (or not) so the histogram reads naturally toward the
  shared price axis; keep both panels on the same y so peak lines align with the
  price path.
- For peak rectangles use `fig.add_shape` (rect) with `x0=0, x1=peak_height,
  y0=price-w/2, y1=price+w/2`; add the `width_h05` rect **after** `width_h1` so it
  renders on top. (Shapes are not legend-toggleable; if the width layers must be
  toggleable, draw them as `go.Scatter` filled rectangles with `legendgroup`.)
- Compute peak height via `idx = argmin(abs(bin_centers - price))` →
  `vp_kde[idx]`; reuse the same `idx` for consistency.
- Use `bin_centers` (not bin edges) for histogram/KDE y-positions to match
  `kde_tools` geometry.
- Keep the function read-only w.r.t. `data` and return the figure.
- Mirror the existing notebook's input-parameter cell style when adding cells.
