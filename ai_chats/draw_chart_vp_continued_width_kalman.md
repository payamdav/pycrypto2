# Spec: `draw_chart_vp_continued_width_kalman_{1d,2d,3d}` — Kalman-smoothed VP chart variants

## 1. Task summary

Add three functions to `strategies/lbla_n_vp/lbla_n_vp_chart.py`:

1. `draw_chart_vp_continued_width_kalman_1d`
2. `draw_chart_vp_continued_width_kalman_2d`
3. `draw_chart_vp_continued_width_kalman_3d`

Each is a variant of the existing `draw_chart_vp_continued_width(data)` (same `data` dict,
same two-panel layout, peak lines, continued `width_h05` bands, robust-z volume axis, and
the two tables) plus a Kalman-filtered overlay:

- Run the matching `packages/kalman_filter` **batch** filter over `data["lb_pnc"]` (the
  unclipped normalized look-back curve), take the smoothed **value** series, **clip it to
  `[-1, 1]`**, and draw it on the right (price-vs-time) panel over `lb_x` as an extra line.
- **1d**: overlay only.
- **2d**: overlay **plus one subchart** under the price panel, sharing its x-axis (`lb_x`),
  showing estimated **speed**.
- **3d**: overlay **plus two subcharts** under the price panel, sharing its x-axis (`lb_x`),
  showing **speed** and **acceleration**.

Then edit `strategies/lbla_n_vp/lbla_n_vp.ipynb`: remove the two existing chart-drawing
cells and add three cells, one per new function, each exposing `measurement_variance` and
`process_noise` knobs with reasonable defaults.

## 2. Background and context

`draw_chart_vp_continued_width(data)` (already in the module) renders, from the `data`
dict produced by `lookback_lookahead_normalized_vp`:
- **Right panel** (`row=1, col=2`): normalized price path (`lb_p` vs `lb_x`, `la_p` vs
  `la_x`), vertical separator at `x=1.0`, horizontal current-price line at `y=0`,
  `y ∈ [-1, 1]`, each `width_h05` band continued across `lb_x[0] → la_x[-1]`.
- **Left panel** (`row=1, col=1`, shared y = price): robust-z histogram + KDE, HVN peak
  lines, `width_h05` rectangles (no `width_h1`).
- Below: a peaks DataFrame and a metrics DataFrame.

The three new functions reuse all of this unchanged and add the Kalman overlay (and, for
2d/3d, the speed/acceleration subcharts). Factor shared logic so the variants stay
consistent with `draw_chart_vp_continued_width`; do not needlessly duplicate the whole
body.

### Relevant `data` keys

| Key | Meaning |
|---|---|
| `lb_pnc` | **Unclipped** normalized look-back curve, `float64`, length `look_back`. Kalman input. |
| `lb_x`, `la_x`, `lb_p`, `la_p` | Right-panel time/price axes. `lb_x` has length `look_back`; aligns 1:1 with `lb_pnc`. |
| `bin_centers`, `vp_hist`, `vp_kde`, `bin_width` | Left-panel volume profile. |
| `hvn`, `metrics`, `current_price`, `datetime`, `asset` | As in `draw_chart_vp_continued_width`. |

`lb_pnc` and `lb_x` have the same length, so the smoothed series plots directly against
`lb_x`. The look-ahead (`la_*`) is **not** filtered.

### Kalman package (`packages/kalman_filter`, see `agents/packages/kalman_filter.md`)

```python
from packages.kalman_filter import kalman_1d_batch, kalman_2d_batch, kalman_3d_batch
```

| Func | Signature | Returns | Smoothed value / derivatives |
|---|---|---|---|
| `kalman_1d_batch` | `(measurements, initial_estimate, initial_error_cov, process_variance, measurement_variance)` | `(estimates (N,), covs (N,))` | value = `estimates` |
| `kalman_2d_batch` | `(measurements, initial_state (2,1), initial_covariance (2,2), process_noise (2,2), measurement_variance, dt)` | `(states (N,2,1), covs (N,2,2))` | value=`states[:,0,0]`, speed=`states[:,1,0]` |
| `kalman_3d_batch` | `(measurements, initial_state (3,1), initial_covariance (3,3), process_noise (3,3), measurement_variance, dt)` | `(states (N,3,1), covs (N,3,3))` | value=`states[:,0,0]`, speed=`states[:,1,0]`, accel=`states[:,2,0]` |

All are Numba `@njit`; `measurements` must be a contiguous `float64` 1-D array — pass
`np.ascontiguousarray(data["lb_pnc"], dtype=np.float64)`.

## 3. Relevant conventions from `/agents/`

- **kalman_filter** (`agents/packages/kalman_filter.md`): batch APIs, state layouts, and
  the `Q↑/R↓ ⇒ tracks measurements; Q↓/R↑ ⇒ smoother` tuning relationship above.
- **Normalized price space / time axis** (`idea_normalize_based_on_last_price_clip.md`):
  `lb_x = arange(look_back)/look_back`, price y `0.0` = current price, `y ∈ [-1, 1]`.
- **Notebook deps** (`general/rules.md`): the notebook's first cell already `%pip install`s
  `numba`/`numpy`/`plotly`; `kalman_filter` is in-repo (imported via the existing repo-clone
  cell). No new install needed; confirm `numba` is present (it is).
- **Script deps** (`general/rules.md`): `kalman_filter` is a local package, not a PyPI dep —
  no change to `strategies/lbla_n_vp/requirements.txt`.
- **Placement** (`general/paths_and_files.md`): all code stays under `strategies/lbla_n_vp/`.
- **Writing style** (`general/rules.md`): minimal docstrings/comments.
- **Scope** (`general/access.md`): no tests, no debugging beyond making the charts render.

## 4. Functional requirements

### 4.0 Common to all three functions

Signature:

```python
def draw_chart_vp_continued_width_kalman_1d(data, measurement_variance=1.0, process_noise=None) -> go.Figure
def draw_chart_vp_continued_width_kalman_2d(data, measurement_variance=1.0, process_noise=None) -> go.Figure
def draw_chart_vp_continued_width_kalman_3d(data, measurement_variance=1.0, process_noise=None) -> go.Figure
```

- **`measurement_variance`**: passed straight through as the filter's `R`. Default `1.0`,
  exposed as an input so it can be tuned.
- **`process_noise`**: the filter's `Q`, exposed as an input. Use the mutable-default-safe
  `None` sentinel; build the default inside the function (§4.4):
  - 1d → scalar `0.03`.
  - 2d → `np.eye(2, dtype=np.float64) * 0.03`.
  - 3d → `np.eye(3, dtype=np.float64) * 0.03`.
  If a caller passes a value, use it as-is. A 2d/3d caller may pass either a full
  `(n,n)` matrix or a scalar; if a scalar is passed, coerce it to `scalar * np.eye(n)` so
  the notebook can tune one number. (1d expects a scalar.)
- **`dt = 1.0`** for 2d/3d.
- Initialization:
  - 1d: `initial_estimate = float(lb_pnc[0])`, `initial_error_cov = 1.0`.
  - 2d: `x0 = [[lb_pnc[0]], [0.0]]` `(2,1)`, `P0 = np.eye(2)`.
  - 3d: `x0 = [[lb_pnc[0]], [0.0], [0.0]]` `(3,1)`, `P0 = np.eye(3)`.
- Compute the smoothed **value** series, then `kalman_clipped = np.clip(value, -1.0, 1.0)`.
- Draw `kalman_clipped` vs `lb_x` on the right price panel as a distinct line
  (e.g. color `purple`/`magenta`, width ~2), `name="kalman"`, `legendgroup="kalman"`,
  legend-toggleable. It must sit on the same axes as `lb_p`/`la_p` (`y ∈ [-1, 1]`).
- Everything else from `draw_chart_vp_continued_width` is preserved on the price panel and
  left panel: the `lb_p`/`la_p` paths, `x=1.0` separator, `y=0` line, continued `width_h05`
  bands, robust-z histogram+KDE, peak lines, and **both tables** (unchanged).
- Read-only w.r.t. `data`; call `fig.show()`; render the peaks + metrics DataFrames; return
  the `go.Figure`.

### 4.1 `draw_chart_vp_continued_width_kalman_1d`
- Layout identical to `draw_chart_vp_continued_width` (`make_subplots(rows=1, cols=2,
  shared_yaxes=True, column_widths=[0.25, 0.75])`).
- Add only the clipped Kalman value overlay (§4.0). No subchart.

### 4.2 `draw_chart_vp_continued_width_kalman_2d`
- Everything in §4.0/§4.1 **plus** one subchart beneath the price panel sharing its x-axis
  (`lb_x`), showing the estimated **speed** (`states[:,1,0]`) vs `lb_x`.
- Suggested layout: `make_subplots(rows=2, cols=2, shared_yaxes=True, shared_xaxes=True,
  column_widths=[0.25, 0.75], row_heights=[0.75, 0.25], vertical_spacing=…)`, with the
  **left VP panel at `(1,1)`** and `(2,1)` left empty (use `specs`/leave blank). Price path
  + overlays go in `(1,2)`; speed trace in `(2,2)`.
- The right column (`(1,2)` and `(2,2)`) must share the x-axis so the speed aligns under the
  price/Kalman curve. The left↔right y-share applies to row 1 only (VP ↔ price).
- Label the speed subchart's y-axis (e.g. `"speed"`). The speed axis is **not** clipped to
  `[-1, 1]` (autoscale). Optionally extend the `x=1.0` separator into the speed subchart for
  visual alignment.
- Name/group the speed trace `"speed"` / `legendgroup="speed"`.

### 4.3 `draw_chart_vp_continued_width_kalman_3d`
- Everything in §4.0/§4.1 **plus two** subcharts beneath the price panel, both sharing its
  x-axis (`lb_x`): **speed** (`states[:,1,0]`) and **acceleration** (`states[:,2,0]`).
- Suggested layout: `make_subplots(rows=3, cols=2, shared_yaxes=True, shared_xaxes=True,
  column_widths=[0.25, 0.75], row_heights=[~0.6, ~0.2, ~0.2])`, left VP panel at `(1,1)`,
  `(2,1)`/`(3,1)` empty. Price+overlays in `(1,2)`, speed in `(2,2)`, acceleration in
  `(3,2)`; the whole right column shares x.
- Label the subchart y-axes (`"speed"`, `"acceleration"`); both autoscale (not clipped).
  Group traces `"speed"` and `"acceleration"`.

### 4.4 Process-noise defaults — chosen value & rationale
- **1d default `0.03`** is given by the request (matches the existing 1D random-walk
  intensity).
- **2d/3d default `0.03 · I`** (`np.eye(n) * 0.03`): keep the **same per-state diagonal
  intensity** as the 1D value, so the eye-scaled `Q` stays directly comparable to `0.03`.
  This is the simplest defensible relation to 0.03 and honors the request that `Q` be an
  identity (eye) matrix. It is intentionally a simplification of the physically-derived
  constant-velocity / constant-acceleration `Q` (which is `σ² · [[dt⁴/4, dt³/2, …], …]`,
  not diagonal); the request explicitly asks for an eye matrix, so we scale identity by
  the 1D intensity. The value is a notebook input, so it is meant to be tuned. See sources
  in §8.

### 4.5 Notebook changes (`lbla_n_vp.ipynb`)
- **Remove** the two existing chart-drawing cells: the one importing/calling
  `draw_chart_vp(data)` and the one importing/calling `draw_chart_vp_continued_width(data)`
  (current last two cells).
- **Add three cells**, one per new function, each:
  - Reusing the `data` already built by the existing chart-input cell (no new `data` build).
  - Defining its own `measurement_variance` (default `1.0`) and `process_noise` knob:
    - 1d cell: `process_noise = 0.03`.
    - 2d cell: `process_noise = 0.03` (cell builds `np.eye(2) * process_noise`, or passes the
      scalar relying on §4.0 coercion — pick one and be consistent).
    - 3d cell: `process_noise = 0.03` (likewise for `np.eye(3)`).
  - Importing the function from `strategies.lbla_n_vp.lbla_n_vp_chart` and calling it with
    `data`, `measurement_variance`, and `process_noise`.
- After the edit the notebook ends with three comparable charts (1d, 2d, 3d) the user can
  retune by editing each cell's two knobs.

## 5. Non-goals / out of scope
- No changes to `draw_chart_vp`, `draw_chart_vp_continued_width`, `lbla_n_vp.py`, or any
  `packages/` code (other than optional shared-helper extraction in the chart module that
  preserves existing output).
- Do **not** Kalman-filter the look-ahead (`la_*`) series.
- No new metrics/peak/KDE computation; charts consume existing `data`.
- No tests, no CLI, no image export.

## 6. Assumptions
- `data["lb_pnc"]` exists, is `float64`, length `look_back`, and aligns with `lb_x`
  (it does, per `lbla_n_vp.py`).
- The Kalman overlay is plotted in normalized price space and clipping to `[-1, 1]` matches
  the panel's y-range.
- Speed/acceleration are in normalized-price units per step (`dt = 1`); their subcharts
  autoscale.
- `numba` is available at notebook runtime (first batch call triggers JIT compile).

## 7. Acceptance criteria
- Three functions exist with the names and signatures in §4.0 and each returns a
  `go.Figure`, shows it, and renders both DataFrames.
- Each figure reproduces `draw_chart_vp_continued_width` (no `width_h1`; `width_h05`
  continued; robust-z histogram **and** KDE; peak lines; `x=1.0` separator; `y=0` line;
  `y ∈ [-1, 1]`) **plus** a legend-toggleable `"kalman"` line = clipped smoothed value of
  `lb_pnc` plotted vs `lb_x`.
- **1d**: overlay only. **2d**: one speed subchart sharing x (`lb_x`) with the price panel.
  **3d**: speed **and** acceleration subcharts sharing x with the price panel.
- `measurement_variance` (default `1.0`) and `process_noise` (defaults: `0.03` scalar /
  `0.03·eye(2)` / `0.03·eye(3)`) are inputs; passing other values changes the result.
- `dt = 1.0` used in 2d/3d.
- The notebook no longer contains the `draw_chart_vp` / `draw_chart_vp_continued_width`
  cells and instead has three cells (1d/2d/3d), each with its own `measurement_variance`
  and `process_noise` knobs, reusing `data`. The `.ipynb` JSON stays valid.

## 8. Open questions / decisions
- **2d/3d default = `0.03 · I` (decided):** chosen for a direct, simple relation to the 1D
  `0.03` and to satisfy the eye-matrix requirement; it is a knob the user will tune. If the
  user later wants the physically-derived constant-velocity/acceleration `Q`
  (`σ² · [[dt⁴/4, dt³/2],[dt³/2, dt²]]` etc.), that is a follow-up — but it would not be an
  identity matrix, which the request asked for.
- **Subchart layout (`specs` / empty left cells):** the spec fixes intent (left VP spans
  row 1; right column shares x across all rows); the coding agent finalizes the exact
  `make_subplots(specs=…)` so the left panel renders once and rows align.
- **Scalar-vs-matrix `process_noise` for 2d/3d:** spec allows passing a scalar (coerced to
  `scalar·eye(n)`) or a full matrix; pick the scalar-knob form in the notebook for easy
  tuning.

## 9. Notes for the downstream coding agent
- Read all of `/agents/` first. Reuse the helpers in `lbla_n_vp_chart.py` (`_gather_peaks`,
  `_make_show_fn`, `_display_dfs`, the robust-z and peak-height logic) so the variants stay
  consistent with `draw_chart_vp_continued_width`; factor a shared builder if it avoids
  copying the whole body across four functions.
- Kalman input: `m = np.ascontiguousarray(data["lb_pnc"], dtype=np.float64)`.
  - 1d: `est, _ = kalman_1d_batch(m, float(m[0]), 1.0, process_noise, measurement_variance)`;
    `value = est`.
  - 2d: `x0 = np.array([[m[0]],[0.0]]); P0 = np.eye(2); states,_ = kalman_2d_batch(m, x0, P0,
    Q2, measurement_variance, 1.0)`; `value = states[:,0,0]`, `speed = states[:,1,0]`.
  - 3d: `x0 = np.array([[m[0]],[0.0],[0.0]]); P0 = np.eye(3); states,_ = kalman_3d_batch(m,
    x0, P0, Q3, measurement_variance, 1.0)`; `value = states[:,0,0]`, `speed =
    states[:,1,0]`, `accel = states[:,2,0]`.
  - Overlay: `go.Scatter(x=lb_x, y=np.clip(value, -1, 1), mode="lines",
    line=dict(color="purple", width=2), name="kalman", legendgroup="kalman",
    showlegend=…)` on the price panel (`row=1, col=2`).
- Build `process_noise` defaults via a `None` check; coerce scalar→`eye` for 2d/3d.
- For 2d/3d use `make_subplots(..., shared_xaxes=True)` and add speed/accel traces in the
  lower-right rows; keep the left VP panel on `(1,1)` only.
- Keep functions read-only w.r.t. `data` and return the figure.
- Notebook: delete exactly the two chart cells; append three cells; keep the JSON valid.

### Sources consulted (2d/3d process-noise default)
- [Kalman Filter Explained Through Examples — kalmanfilter.net](https://kalmanfilter.net/multiExamples.html)
- [Tuning Q matrix for CV and CA models in Kalman Filter — Dr Barak Or (Medium)](https://medium.com/data-science/tuning-q-matrix-for-cv-and-ca-models-in-kalman-filter-67084185d08c)
- [Tuning Kalman Filter to Improve State Estimation — MathWorks](https://www.mathworks.com/help/fusion/ug/tuning-kalman-filter-to-improve-state-estimation.html)
