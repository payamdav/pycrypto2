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
- One figure, shared **price (y) axis** between the VP panel and the main panel:
  - **Left panel** = volume profile (price on y, volume/density on x).
  - **Right panel** = price-vs-time (wider; e.g. width ratio ~1:3).
- Below the panels, **two tables** (see 4.5). Implement as Plotly `Table` traces in
  the same `make_subplots` grid so the output is one self-contained figure.
- Y axis fixed to `[-1, 1]` on the price panels.

### 4.3 Main panel (price vs time)
- Plot `lb_p` vs `lb_x` (look-back) and `la_p` vs `la_x` (look-ahead) as two lines.
- **Vertical separator at current time** marking past↔future (see Open Question 1
  for x-axis origin; default: re-center so current time = `0`).
- **Horizontal line at `y=0`** = current price.
- Y range `[-1, 1]`.

### 4.4 Left panel (volume profile, shared y = price)
- **Horizontal histogram** of `vp_hist` across `bin_centers` (bars oriented along
  price/y), drawn in a **light/low-opacity color**.
- **`vp_kde` line** drawn over the histogram (sharper color).
- **7 HVN peaks** = `hvn["poc"]` + up to 3 `hvn["above"]` + up to 3 `hvn["below"]`
  (skip `None`/missing): one **horizontal line** at each peak's `price`.
- For each peak, two rectangles giving the peak's price band:
  - `width_h1` rectangle: vertical extent = `width_h1 * bin_width`, centered on the
    peak price; lighter color.
  - `width_h05` rectangle: vertical extent = `width_h05 * bin_width`, centered on the
    peak price; **drawn over** the `width_h1` rectangle in a **sharper color**.
  - Default horizontal extent = full VP panel width (see Open Question 3).
  - Distinguish POC visually from the other six peaks (e.g. distinct color/label).

### 4.5 Tables (bottom of figure)
- **Peaks table**: one row per displayed peak with columns
  `price`, `prominence`, `width_h1`, `width_h05`. Include a label column marking POC
  vs above/below. Use price values directly; widths may be shown in bins and/or
  normalized-price units (state which in the header).
- **Metrics table**: every key/value in `data["metrics"]` (all cached metrics
  appended by `append_cached_metrics` for this candle), two columns name/value.

### 4.6 Function contract
- Signature: `def draw_chart_vp(data: dict) -> go.Figure:`.
- Build and return the Figure; also call `fig.show()` so it renders when invoked
  from a Colab cell. Returning the figure lets callers further customize.
- Read only from `data`; do not mutate it; do not recompute pipeline values.
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
  a `go.Figure` and showing it.
- Running the two new notebook cells top-to-bottom renders one interactive figure in
  Colab with: price-vs-time main panel (y∈[-1,1]); shared-axis VP panel with
  low-opacity histogram + KDE overlay; vertical current-time separator; horizontal
  current-price line; 7 (or fewer) peak lines with `width_h1`/`width_h05` rectangles
  (h05 over h1, sharper); peaks table; metrics table.
- Legend clicks toggle drawing layers on/off.
- `plotly` added to both `requirements.txt` and the notebook `%pip install` cell.
- No exception when a peak list is empty or POC is the only peak.

## 8. Open questions (defaults chosen; confirm if wrong)
1. **X-axis origin.** Request says "vertical line at **0** of x axis," but `lb_x`/`la_x`
   put the past/future boundary at `1.0`. **Default: re-center the time axis so
   current time = `0`** (look-back negative, look-ahead positive) and draw the
   vertical line at `x=0`. Alternative: keep raw `lb_x`/`la_x` and place the line at
   `1.0`.
2. **"0 on x axis shows the current price."** Read as a wording mix-up: the
   **horizontal** line at `y=0` shows current price; the **vertical** line at current
   time splits past/future. Confirm.
3. **Width-rectangle horizontal extent.** Default: span the full VP panel width.
   Alternative: extend only from `0` to each peak's KDE value.
4. **Width units in the peaks table.** Default: show normalized-price units
   (`width × bin_width`); optionally also raw bins. Confirm preference.

## 9. Notes for the downstream coding agent
- Read all of `/agents/` before coding (`CLAUDE.md`, `AGENTS.md`).
- Build the grid with `make_subplots(rows=2, cols=2, shared_yaxes=True,
  column_widths=[~0.25, ~0.75], specs=[[{}, {}], [{"type":"table"}, {"type":"table"}]])`
  (or a layout that places the two tables beneath the two panels).
- Reverse the VP panel's x-axis (or not) so the histogram reads naturally toward the
  shared price axis; keep both panels on the same y so peak lines align with the
  price path.
- For width rectangles use `fig.add_shape` (rect) or bar traces with
  `legendgroup`; ensure `width_h05` is added **after** `width_h1` so it renders on
  top.
- Use `bin_centers` (not bin edges) for histogram/KDE y-positions to match
  `kde_tools` geometry.
- Keep the function pure w.r.t. `data` (read-only) and return the figure.
- Mirror the existing notebook's input-parameter cell style when adding cells.
