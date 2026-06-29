# Spec: `draw_chart_vp` — LBLA Normalized VP Chart (width-rectangle continuation)

## 1. Task summary

Implement/refine `draw_chart_vp(data)` in
`strategies/lbla_n_vp/lbla_n_vp_chart.py`: one interactive, Colab-friendly Plotly
figure built from the `data` dict produced by `lookback_lookahead_normalized_vp`
(`strategies/lbla_n_vp/lbla_n_vp.py`). Then ensure `strategies/lbla_n_vp/lbla_n_vp.ipynb`
ends with two cells: one defining all chart inputs + a datetime string and building
`data`, one calling `draw_chart_vp(data)`.

A working first pass already exists. This continuation re-states the full request as
the authoritative source and puts special emphasis on the **per-peak width_h1 /
width_h05 rectangles** (see §4.4). Reconcile the existing code against this spec and
correct anything that diverges; do not regress the parts that already comply.

## 2. Background and context

`lookback_lookahead_normalized_vp` runs `lb_la_n_base → append_cached_metrics →
vp_analysis → vp_hvn` and returns one `data` dict describing a single anchor minute.
The chart visualizes that minute: the normalized price path (look-back + look-ahead)
on the right, the volume profile (histogram + KDE + HVN peaks) on the left sharing the
price axis, and two tables below.

### `data` keys consumed by the chart

| Key | Type | Meaning |
|---|---|---|
| `lb_x`, `la_x` | `float64[]` | Time axes. `lb_x∈[0,1)`, `la_x∈[1, 1+(look_ahead-1)/look_back]`. Current-time boundary = `1.0`. |
| `lb_p`, `la_p` | `float64[]` | Normalized prices clipped to `[-1,1]` (current price → `0.0`). |
| `bin_centers` | `float64[]` | KDE bin-center prices in normalized space `[-1,1]`. |
| `vp_hist` | `float64[]` | Raw volume-weighted histogram counts per bin. |
| `vp_kde` | `float64[]` | Smoothed (convolved) density per bin. |
| `bin_width` | `float` | Width of one bin in normalized-price units. |
| `hvn` | dict | `{"poc": peak|None, "above": [peak…≤3], "below": [peak…≤3]}`. |
| `metrics` | dict | `{col: float}` — all cached metrics at the anchor (`append_cached_metrics`). |
| `current_price`, `current_ts`, `last_candle_ts`, `datetime`, `asset` | — | Labels / title. |

A **peak record** is `{"price": float, "prominence": float, "width_h1": float,
"width_h05": float}`. `width_h1`/`width_h05` are in **bins**; multiply by `bin_width`
for normalized-price units (`agents/packages/kde_tools.md`). For a normal peak
`width_h05 ≤ width_h1` (the band at half height is narrower than at the base).

## 3. Relevant conventions from `/agents/`

- **Normalized price space** (`idea_normalize_based_on_last_price_clip.md`,
  `kde_tools.md`): on the price/y axis `0.0` = current price, `>0` above, `<0` below,
  clipped extremes at `±1.0`.
- **Time axis** (`idea_normalize_based_on_last_price_clip.md`): `t=1.0` is
  `current_time` — the moment the last look-back candle closes; it separates past
  (look-back) from future (look-ahead).
- **Notebook deps** (`general/rules.md`): every package a notebook uses must be in its
  inline `%pip install` cell before imports — `plotly` included.
- **Script deps** (`general/rules.md`): `plotly` must be in
  `strategies/lbla_n_vp/requirements.txt`.
- **Placement** (`general/paths_and_files.md`): strategy code lives under
  `strategies/lbla_n_vp/`.
- **Writing style** (`general/rules.md`): minimal docstrings/comments, no filler.
- **Scope** (`general/access.md`): no tests, no debugging beyond making the chart
  render correctly.

## 4. Functional requirements

### 4.1 Library & interactivity
- Use **Plotly** (`plotly.graph_objects` + `plotly.subplots.make_subplots`) — native in
  Colab, supports toggling traces by clicking the legend ("turn drawings on/off by
  clicking on its chart map").
- Group each drawing layer with a shared `legendgroup` and a single legend entry so one
  click toggles the whole layer. Toggle groups: `histogram`, `kde`, `look-back`,
  `look-ahead`, `POC`, `peaks (lines)`, `width_h1`, `width_h05`.
- Because width rectangles must be legend-toggleable, draw them as **`go.Scatter`
  filled shapes** (`fill="toself"`), not `fig.add_shape` (shapes are not toggleable).

### 4.2 Layout
- One figure, two columns sharing the **price (y) axis**:
  `make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[~0.25, ~0.75])`.
  - **Left panel (col 1)** = volume profile: price on y, volume/density on x.
  - **Right panel (col 2)** = price-vs-time (wider).
- Y axis fixed to `[-1, 1]` on both panels.
- Tables are **NOT** subplots — render them as pandas DataFrames below the chart (§4.5).

### 4.3 Main panel (price vs time)
- Plot `lb_p` vs `lb_x` (look-back) and `la_p` vs `la_x` (look-ahead) as two lines,
  using the raw `lb_x`/`la_x` values unchanged (no shift).
- **Vertical separator at `x = 1.0`** = current time, splitting past↔future.
- **Horizontal line at `y = 0`** = current price.
- Y range `[-1, 1]`.

### 4.4 Left panel (volume profile, shared y = price) — width rectangles emphasized
- **Horizontal histogram** of `vp_hist` across `bin_centers` (bars along price/y) in a
  **light, low-opacity color**.
- **`vp_kde` line** over the histogram in a sharper color.
- **Up to 7 HVN peaks** = `hvn["poc"]` + up to 3 `hvn["above"]` + up to 3 `hvn["below"]`
  (skip `None`/missing): one **horizontal line** at each peak's `price`, spanning the
  density (x) range `0 → peak height`. Distinguish **POC** from the other six
  (distinct color/label/legend group).
- **Peak height** = KDE value at the peak price = `vp_kde[idx]`, where
  `idx = argmin(abs(bin_centers - price))`. Reuse the same `idx` everywhere for that
  peak.
- **Two nested rectangles per peak** (this is the continuation's focus). In this panel
  x = density and y = price, so a "width" is a price band centered on the peak:
  - **x-extent** (the rectangle's depth) = `0 → peak height`, identical for both
    rectangles of a peak.
  - **y-extent** = centered on the peak `price`:
    - `width_h1` rectangle: `price ± (width_h1 * bin_width) / 2` — **lighter** fill,
      drawn **first**.
    - `width_h05` rectangle: `price ± (width_h05 * bin_width) / 2` — narrower band,
      **sharper/more opaque** fill, drawn **after** `width_h1` so it renders on top.
  - Use POC-vs-peak coloring consistent with the peak line (e.g. POC red family, other
    peaks orange family); the h05 fill of each family is the more saturated/opaque
    variant of its h1 fill.
  - Skip a rectangle whose width is `0` (degenerate).
  - All `width_h1` rectangles share one `legendgroup`/legend entry; all `width_h05`
    rectangles share another — one legend click toggles every rectangle in the layer.

### 4.5 Tables (pandas DataFrames, below the chart — not in the figure)
- **Peaks table**: one row per displayed peak with columns `label` (POC/above/below),
  `price`, `height` (KDE value at the peak), `prominence`, `width_h1`, `width_h05`.
  State width units in the headers (normalized-price units = `width × bin_width`).
- **Metrics table**: every key/value in `data["metrics"]`, two columns name/value.
- Display both via the notebook (`IPython.display.display`, falling back to
  `print(df.to_string())` when IPython is absent).

### 4.6 Function contract
- Signature: `def draw_chart_vp(data: dict) -> go.Figure:`.
- Call `fig.show()` and render the two DataFrames so all three outputs appear from a
  Colab cell; return the Figure.
- Read-only w.r.t. `data` (the `argmin` peak-height lookup is a read, not a recompute);
  do not mutate `data`, do not recompute pipeline values.
- Handle empty/`None` peak lists gracefully (POC may exist with no above/below; any list
  may be empty).

### 4.7 Notebook changes (`lbla_n_vp.ipynb`)
- Ensure `plotly` is in the first cell's `%pip install` line.
- Penultimate cell: define all chart inputs (`asset`, `look_back`, `look_ahead`, `k`,
  `bins_count`, `bandwidth`, `kernel_type`, `kde_ignore_borders`) plus a datetime
  string (`dt_str`), then build `data` via `lookback_lookahead_normalized_vp(...)`
  (reuse an in-scope `data` if present), mirroring the existing input-cell style.
- Final cell: `from strategies.lbla_n_vp.lbla_n_vp_chart import draw_chart_vp` then
  `draw_chart_vp(data)`.

## 5. Non-goals / out of scope
- No changes to `lbla_n_vp.py` pipeline logic or to any `packages/` code.
- No new metrics, peak-finding, or KDE computation — the chart only consumes `data`.
- No tests, no CLI, no saving images to disk, no multi-anchor/animation views.

## 6. Assumptions
- Plotly renders inline in Colab with the default renderer.
- `data` holds a successful pipeline run (all keys present, no exception).
- `vp_hist`, `vp_kde`, `bin_centers` share length `bins_count`.
- Cached metrics are scalar floats.
- Width values are non-negative; `0` ⇒ a skipped/degenerate rectangle.

## 7. Acceptance criteria
- `draw_chart_vp(data)` returns a `go.Figure`, shows it, and renders both DataFrames.
- Running the two notebook cells top-to-bottom in Colab produces one interactive figure:
  price-vs-time main panel (y∈[-1,1]) with vertical separator at `x=1.0` and horizontal
  current-price line at `y=0`; shared-axis VP panel with low-opacity histogram + KDE
  overlay; up to 7 peak lines; and for every peak the **nested width_h1 (lighter) and
  width_h05 (narrower, sharper, on top) rectangles**, each spanning x `0→peak height`
  and y `price ± width·bin_width/2`.
- Peaks DataFrame (incl. a `height` column) and metrics DataFrame print below the chart.
- Legend clicks toggle each layer — including `width_h1` and `width_h05` — on/off.
- `plotly` present in both `requirements.txt` and the notebook `%pip install` cell.
- No exception when a peak list is empty or POC is the only peak.

## 8. Resolved decisions (carried forward from prior iteration)
1. **X-axis:** raw `lb_x`/`la_x` (no shift); vertical past/future separator at `x=1.0`
   (current time). The attached request's "vertical line at 0 / current price" is
   honored as the **horizontal** `y=0` current-price line plus the `x=1.0` time
   boundary — keep this mapping.
2. **Library:** Plotly.
3. **Tables:** pandas DataFrames printed below the chart, not embedded in the figure.
4. **Peak rectangles:** centered on the peak `price` along y with thickness
   `width × bin_width`; depth along x = `0 → peak height` (`vp_kde[idx]`); `width_h05`
   over `width_h1`, sharper.
5. **Peaks table:** include the `height` column.

## 9. Open questions
- None blocking. If the user later wants the width bands drawn at their true scipy
  evaluation heights (`rel_height` levels) rather than as full-depth nested rectangles,
  that is a separate refinement; this spec keeps the nested full-depth rectangles the
  request literally describes ("h05 smaller, drawn over h1, sharper color").

## 10. Notes for the downstream coding agent
- Read all of `/agents/` first (`CLAUDE.md`, `AGENTS.md`) and the existing
  `lbla_n_vp_chart.py`; refine in place rather than rewrite from scratch where it
  already complies.
- Width rectangles as `go.Scatter(fill="toself", line=dict(width=0))` with the polygon
  `x=[0,h,h,0,0]`, `y=[y0,y0,y1,y1,y0]`; add `width_h1` before `width_h05` so the
  sharper band sits on top.
- Compute peak height via `idx = argmin(abs(bin_centers - price))` → `vp_kde[idx]`.
- Use `bin_centers` (not edges) for histogram/KDE y-positions to match `kde_tools`.
- Keep the function read-only w.r.t. `data` and return the figure.
- Mirror the existing notebook input-cell style when touching the cells.
