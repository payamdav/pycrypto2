# Spec: KDE peak parameters + LBLA VP notebook/chart enhancements

## 1. Task summary

Extend the KDE peak-finding API and the `lbla_n_vp` strategy with new
peak-selection controls, then clean up and fix the strategy notebook and its
3D-Kalman chart. Concretely:

1. `packages/kde_tools/peaks.py`
   - Add `top_identifier` (`"prominence"` | `"height"`) to `top_kde_peaks` and
     `kde_peaks_above_below`.
   - Change `kde_peak_widths` to take `rel_height` (default `0.5`) and return a
     **single** width instead of the two fixed-height widths.
2. `strategies/lbla_n_vp/lbla_n_vp.py`
   - Thread `rel_height`, `top_identifier`, `peak_numbers`, and
     `minimum_peak_score` through `lookback_lookahead_normalized_vp` into the VP
     HVN step.
3. `strategies/lbla_n_vp/lbla_n_vp_chart.py`
   - Update every caller to the single-width API.
   - Fix the duplicate 3D-Kalman chart render (inside the chart function).
   - Add vertical + horizontal crosshairs visible across all subcharts.
4. `strategies/lbla_n_vp/lbla_n_vp.ipynb`
   - Remove timing cells, merge the two parameter cells, expose the new
     parameters, and keep only the 3D-Kalman chart.
5. Update `agents/packages/kde_tools.md` to match the new signatures.

## 2. Background and context

- `kde_tools` is a pure KDE construction + peak-finding package (no data
  loading or plotting). `find_peaks`/`peak_prominences`/`peak_widths` stay in
  scipy; only the histogram/convolution core is numba-jitted.
- The `lbla_n_vp` strategy pipeline is
  `lb_la_n_base → append_cached_metrics → vp_analysis → vp_hvn`. By the time
  `vp_hvn` runs, `data["metrics"]` (including `v_median`, `v_iqr`) is populated.
- `vp_hvn` currently has its own peak logic (`_top_peaks`, `_peak_record`) that
  duplicates the package functions: it hardcodes `n=3`, sorts by prominence,
  filters `prominence > 0`, and stores both `width_h1` and `width_h05` per peak.
- Charts: `_add_continued_width_content` (used by all `continued_width` and
  Kalman variants) consumes only `width_h05`. The legacy `draw_chart_vp` draws
  both `width_h1` (base band) and `width_h05`.
- The "current price" is normalized to `0.0`; peaks are split above/below at
  `bin_centers >= 0.0` / `< 0.0`.

## 3. Relevant conventions from `/agents/`

- `general/rules.md`: notebooks must `%pip install` all deps inline (already
  present in cell 0); package changes must be reflected in
  `agents/packages/<pkg>.md`; writing must be terse and non-redundant.
- `general/access.md`: full file access, no permission needed; **do not write
  or run tests** (testing not requested here); **do not debug** beyond the one
  explicitly-requested duplicate-chart fix.
- `packages/kde_tools.md`: mirror notebook semantics precisely; keep
  `np.argsort(...)[::-1][:n]` descending tie order; keep scipy peak functions.
- `paths_and_files.md`: strategy code stays under `strategies/lbla_n_vp/`;
  reusable peak logic stays in `packages/kde_tools/`.

## 4. Functional requirements

### 4.1 `kde_tools/peaks.py` — `top_identifier`

Add `top_identifier: str = "prominence"` to both `top_kde_peaks` and
`kde_peaks_above_below` (the latter forwards it to `top_kde_peaks`).

In `top_kde_peaks`, after `find_peaks` and `peak_prominences`:
- `top_identifier == "prominence"` → rank by `proms` (current behavior).
- `top_identifier == "height"` → rank by peak height `kde_series[peaks]`.
- Any other value → `ValueError`.

Ranking stays `np.argsort(score)[::-1][:n]` (descending, numpy tie order). The
return contract is unchanged: `(peak_prices, peak_proms)` — i.e. **always return
the prominences** in the selected order regardless of which key was used for
ranking, so callers still have prominence available. `kde_peaks_above_below`'s
returned dict keys are unchanged.

### 4.2 `kde_tools/peaks.py` — `kde_peak_widths` single width

New signature:

```python
def kde_peak_widths(
    kde_series: np.ndarray,
    peak_indices: np.ndarray,
    rel_height: float = 0.5,
) -> dict:
```

Returns `{"proms", "widths"}` where `widths = peak_widths(kde_series,
peak_indices, rel_height=rel_height)[0]` (in bins). Drop `widths_h1` /
`widths_h05`. Empty arrays when `peak_indices` is empty (keep the same dtype
handling). Keep `proms` via `peak_prominences`.

### 4.3 `lbla_n_vp.py` — new pipeline parameters

Add to `lookback_lookahead_normalized_vp` (store each verbatim in `data`, as the
existing params are):

| Param | Default | Meaning |
|-------|---------|---------|
| `rel_height` | `0.5` | Relative height for peak-width measurement |
| `top_identifier` | `"prominence"` | `"prominence"` or `"height"` ranking key |
| `peak_numbers` | `3` | Number of peaks selected above and below |
| `minimum_peak_score` | `0.0` | Minimum robust z-score a peak must reach to survive |

Thread these into `vp_hvn` (read from `data`). Behavioral changes inside the
HVN step:

- **Selection count**: replace hardcoded `n=3` with `peak_numbers` for both the
  above and below halves.
- **Ranking key**: rank by prominence or peak height per `top_identifier`
  (mirror §4.1). The POC remains `argmax(vp_kde)`.
- **Width**: compute a single width at `rel_height` via the new
  `kde_peak_widths`; store it on each peak record as `width` (replace the
  `width_h1` / `width_h05` fields).
- **Minimum-score filter**: compute each candidate peak's robust z-score and
  drop peaks scoring `< minimum_peak_score`.

#### Robust z-score (minimum_peak_score filter)

`score = (raw - v_median) / v_iqr`, with `v_median`/`v_iqr` from
`data["metrics"]`, where `raw` is:
- the peak **prominence** when `top_identifier == "prominence"`, or
- the peak **height** (`vp_kde` at the peak bin) when `top_identifier ==
  "height"`.

Keep a peak only if `score >= minimum_peak_score`. Guard `v_iqr == 0` (treat
score as `0.0`, consistent with the chart's existing `z_score` guard). With the
defaults (`minimum_peak_score = 0.0`) this still removes any peak scoring below
the volume median; that is the intended behavior. Apply this filter in addition
to the existing `prominence > 0` sanity filter, then take the top `peak_numbers`
by the ranking key.

> Prefer routing selection through the package functions
> (`kde_peaks_above_below` / `top_kde_peaks` with `top_identifier`) rather than
> growing the duplicated `_top_peaks` logic, as long as the per-peak record
> (price, prominence, height, single `width`) and the score filter are still
> produced. If reuse is impractical, keep the logic in `_top_peaks`/`vp_hvn` but
> apply all four parameters there. Either way, do not change the structure of
> `data["hvn"]` beyond the width-field rename.

### 4.4 `lbla_n_vp_chart.py` — single-width refactor

Update **all** callers to the single `width` field (decision: cascade fully):

- `_add_continued_width_content`: use `peak["width"]` where it currently uses
  `peak["width_h05"]`. Update the peaks DataFrame columns to a single
  `width (norm-price)` plus existing `height_z` / `prominence_z`.
- `draw_chart_vp`: it currently draws a `width_h1` base band **and** a
  `width_h05` band. Collapse to the single `width` band (one rectangle per peak)
  and drop the `width_h1` traces and the `width_h1`/`width_h05` table columns.
- `draw_chart_vp_continued_width`, `_kalman_1d/2d/3d`: inherit the change via
  `_add_continued_width_content`; verify no remaining `width_h1`/`width_h05`
  references.

### 4.5 `lbla_n_vp_chart.py` — duplicate-render fix

Root cause: each `draw_*` function calls `fig.show()` (renders the chart above
the tables) **and** returns `fig`; in Jupyter the returned figure is
auto-displayed (duplicate below the tables). Fix **inside the chart function**
(decision): keep `fig.show()` + `_display_dfs(...)` for the
"chart-then-tables" order, but stop the trailing auto-display — e.g. return
`None` (or otherwise ensure the function's return value is not an auto-rendered
figure). Apply the same fix to the other `draw_*` functions for consistency so
none of them double-render. If any caller relies on the returned `fig`, document
that it now returns `None`.

### 4.6 `lbla_n_vp_chart.py` — crosshairs

Add vertical **and** horizontal crosshairs that are visible across all
subcharts, at minimum on the 3D-Kalman chart (apply to the shared helper so
every variant benefits). Use Plotly axis spikes:

- Enable `showspikes=True` on x and y axes with `spikemode="across"` so the
  vertical spike crosses every stacked subchart (they already
  `shared_xaxes=True`) and the horizontal spike crosses the panel row.
- Set `spikesnap="cursor"`, thin spike line, and `spikedistance=-1`; set
  `hovermode` so spikes track the cursor (e.g. `"closest"`).
- Keep the existing `x=1.0` separator and `y=0` reference lines.

Exact colors/dash are at the implementer's discretion; keep them subtle and
distinct from the existing dashed reference lines.

### 4.7 Notebook `lbla_n_vp.ipynb`

- **Remove** the "Single-call timing" cell (cell 7) and the "100-call averages"
  cell (cell 8). Removing single-call timing also removes the cell that defines
  `data`; preserve a `data = lookback_lookahead_normalized_vp(...)` call so the
  chart cell still has `data` (the merged parameter cell + a compute cell, or
  fold the compute into the chart cell's existing `try/except NameError`).
- **Merge** the two parameter cells (cell 6 "Input parameters" and cell 9
  "Chart inputs") into a **single** parameter cell. Add `rel_height`,
  `top_identifier`, `peak_numbers`, and `minimum_peak_score` with their defaults
  (`0.5`, `"prominence"`, `3`, `0.0`) and pass them into
  `lookback_lookahead_normalized_vp`. Keep `dt_str` here.
- **Keep only** the 3D-Kalman chart cell (cell 12). Remove the 1D (cell 10) and
  2D (cell 11) chart cells.
- Leave the pre-load / metrics-cache cells (0–5, minus the removed timing cell)
  intact; keep the existing inline `%pip install` cell.

### 4.8 Docs

Update `agents/packages/kde_tools.md`:
- Add `top_identifier` to the `top_kde_peaks` / `kde_peaks_above_below` sections.
- Replace the `kde_peak_widths` section (new `rel_height` param; returns
  `{"proms", "widths"}`; drop the two-height table). Keep the wording terse per
  `general/rules.md`.

## 5. Non-goals / out of scope

- No changes to KDE construction (`kde.py`, `histogram.py`, `kernels.py`),
  Kalman filters, candle/metrics caching, or data loading.
- No new tests and no test runs (not requested).
- No debugging beyond the single duplicate-render fix in §4.5.
- No new chart types or strategy logic beyond the listed parameters.

## 6. Assumptions

- `data["metrics"]` always contains `v_median` and `v_iqr` at the anchor row
  (guaranteed by `append_cached_metrics` running before `vp_hvn`).
- "Height" means the KDE value at the peak bin (same quantity the chart calls
  `height`), consistent with the volume-z-score units used for the filter.
- The `minimum_peak_score` filter applies to both above and below halves and to
  the POC's siblings, but the POC itself (the global `argmax`) is always kept.
- Returning `None` from the `draw_*` functions is acceptable (the notebook does
  not use their return value).

## 7. Acceptance criteria

- `top_kde_peaks` / `kde_peaks_above_below` accept `top_identifier`; `"height"`
  ranks by KDE height, `"prominence"` reproduces current output; bad values
  raise `ValueError`; returned arrays/keys unchanged.
- `kde_peak_widths(kde, idx, rel_height=r)` returns `{"proms", "widths"}` with
  `widths` at `rel_height=r`; empty-input contract preserved.
- `lookback_lookahead_normalized_vp` accepts and stores `rel_height`,
  `top_identifier`, `peak_numbers`, `minimum_peak_score`; `data["hvn"]` peaks
  carry a single `width` and reflect the count, ranking key, and score filter.
- With defaults, pipeline output matches the previous behavior except: peaks
  carry one `width` (at 0.5) instead of two, and peaks below the volume-median
  z-score (`< 0.0`) are filtered out.
- No `width_h1` / `width_h05` references remain anywhere in the strategy.
- The 3D-Kalman chart renders **once** (above the peaks + metrics tables), with
  vertical and horizontal crosshairs visible across all subcharts.
- Notebook contains a single parameter cell (with the four new params), no
  timing cells, and only the 3D-Kalman chart cell.
- `agents/packages/kde_tools.md` matches the new signatures.

## 8. Open questions

None — width-cascade scope (update all callers) and duplicate-fix location
(inside the chart function) were confirmed with the requester.

## 9. Notes for the downstream coding agent

- Read all of `/agents/` first (mandatory).
- Keep scipy peak functions and the descending `argsort[::-1][:n]` tie order.
- Prefer reusing the package peak functions over the duplicated `_top_peaks`,
  but preserve the per-peak record fields and the score filter either way.
- Guard `v_iqr == 0` in the score filter (mirror the chart's `z_score`).
- For crosshairs, axis spikes with `spikemode="across"` on shared-x subplots is
  the simplest path; verify the horizontal spike shows on the price panel.
- After edits, restart-run the notebook mentally/with a smoke check that imports
  resolve and the 3D chart cell references only defined names — but do not add
  or run test suites.
- Branch/commit/push per the task's branch instructions
  (`claude/skill-spec-writer-task-2winpc`).
