# Spec: bowl_and_dome combined test — one chart, symmetric config, POC VA lines, look-ahead

New test script at `scripts/tests/bowl_and_dome/` combining `scripts/tests/rising_from_bowl/` and `scripts/tests/falling_from_dome/` on a single chart, plus POC tooltip/VA enhancements, a chart-only look-ahead window, and dual vwap lines. Implementation-ready for `code_writer`; self-contained.

## 1. Task Summary

1. New standalone script `scripts/tests/bowl_and_dome/bowl_and_dome_test.py` (+ `config.json`, `requirements.txt`): runs both `rising_from_bowl_scan` and `falling_from_dome_scan`, computes the volume profile + `recursive_poc`, and writes **one** HTML chart showing candles, vwap line(s), all bowls, all domes, and POC levels.
2. Config uses **symmetric pattern properties** — one value drives both detectors (same concept as `notebooks/tests/volume_profile/volume_profile_bowl_dome_colab.ipynb` cell 3); no separate bowl/dome keys.
3. All bowls drawn in one color, all domes in another; legend has exactly one toggle item `Bowls` and one `Domes` (not one item per pattern).
4. POC lines get a rich hover tooltip (the current table info) plus a new value `va_bps = (vah − val) / current_price × 1e4`.
5. Each POC also gets VAL and VAH horizontal lines — same color as the POC, thinner. Preferred: visible only while the POC line's tooltip is shown; always-visible is the accepted fallback.
6. New config property `look_ahead` (minutes, default `240`) — chart-only extension after `date_to`: candles and vwap drawn there, POC/VAL/VAH lines continued there, a vertical line marks the boundary; **no** detection or volume-profile calculation uses look-ahead candles.
7. When `vwap_period > 1` draw **both** vwap lines: plain candle `VWAP` (chart-only) and `VWAP(N)` (rolling; still the series feeding detection and the volume profile).

The two existing test scripts, their configs, the notebook, and all `packages/` code stay **untouched** (user decision: combined test only).

## 2. Background & Context

- The two existing scripts are near-identical 365-line siblings: `analyze(cfg)` (load → vwap → scan → dedupe → stats → volume profile) and `build_figure(res, cfg)` (candles + vwap + per-pattern parabola/markers + POC dashed lines + POC table), `run(cfg)` writes `<name>_{asset}.html` next to the script, `main()` takes an optional config-path argv. The new script mirrors this structure and style; duplicating their helpers is acceptable (the pair already duplicates each other — test scripts are not packages).
- Detector fan-out (exact mapping, from the colab notebook):

| symmetric key (config) | bowl kwarg | dome kwarg |
|---|---|---|
| `min_pattern_width` | `min_bowl_width` | `min_dome_width` |
| `max_pattern_width` | `max_bowl_width` | `max_dome_width` |
| `min_pattern_extent_bps` | `min_bowl_depth_bps` | `min_dome_height_bps` |
| `extremum_position_limit` | `bottom_position_limit` | `top_position_limit` |
| `wall_retrace_limit_bps` | `peak_drawdown_limit_bps` | `trough_rally_limit_bps` |
| `max_wall_search_width` | `max_peak_search_width` | `max_trough_search_width` |

- Scan contracts: `rising_from_bowl_scan` / `falling_from_dome_scan` return `(m, 14)` float64, columns per `SCAN_COLUMNS` / `DOME_SCAN_COLUMNS`; dedupe by `bottom_idx` (bowls) / `top_idx` (domes), first row per group + counts (see `agents/packages/pattern_detection.md`).
- Detectors only look **backward** from each anchor, so loading extra candles after `date_to` cannot affect scan results as long as anchors stay in `[start_idx, end_idx)`.
- `vp["current_price"]` = last vwap of the profile window (= vwap at `end_idx − 1`) — the reference price for `va_bps`.
- Candle cols: `ts`=0, `o..c`=1..4, `v`=5, `q`=6, `vwap`=8.

## 3. Relevant Conventions from `/agents/`

- `agents/general/paths_and_files.md` — test scripts under `scripts/tests/<name>/`; spec under `ai_chats/`.
- `agents/general/rules.md` — `requirements.txt` in the script folder; terse text everywhere.
- `agents/general/access.md` — no unit tests required; `code_writer` verifies by running the script.
- `agents/general/strategy_study_guidelines.md` — vwap is the candle price; no future leak in any calculation (look-ahead is display-only here); per-step elapsed prints.
- `agents/packages/pattern_detection.md`, `agents/packages/volume_profile.md`, `agents/packages/indicators.md` (`rolling_vwap`), `agents/packages/candle_loader.md` — authoritative API contracts; none of these packages change.

## 4. Functional Requirements

### 4.1 Files

- `scripts/tests/bowl_and_dome/bowl_and_dome_test.py` — same skeleton as the existing pair (repo-root sys.path/chdir preamble, `to_ms`, `sanitize_vwap`, `compute_vwap`, dedupe helpers, `analyze`, `build_figure`, `run`, `main`).
- `scripts/tests/bowl_and_dome/config.json` — defaults in 4.2.
- `scripts/tests/bowl_and_dome/requirements.txt` — `numpy numba pandas plotly scipy` (same as siblings).
- Output: `scripts/tests/bowl_and_dome/bowl_and_dome_{asset}.html` via `fig.write_html(..., config={"scrollZoom": True})`.

### 4.2 Config (single flat dict, `config.json`)

```json
{
  "asset": "btcusdt",
  "date_from": "2026-04-15 00:00:00",
  "date_to": "2026-04-21 23:59:00",
  "vwap_period": 1,
  "look_ahead": 240,
  "min_pattern_width": 10,
  "max_pattern_width": 120,
  "min_pattern_extent_bps": 20.0,
  "extremum_position_limit": 0.8,
  "wall_retrace_limit_bps": 15.0,
  "max_wall_search_width": 240,
  "vp_lookback": 1440,
  "vp_bins": 200,
  "vp_bps_range": 100.0,
  "vp_kernel_type": "Triangular",
  "vp_bandwidth": 5,
  "vp_va_pct": 70.0,
  "vp_min_poc_volume_ratio": 0.1
}
```

- Only symmetric pattern keys exist — the script fans them out to each detector's native kwargs per the 2. table.
- `look_ahead`: int minutes, `cfg.get("look_ahead", 240)`; `< 0` → `ValueError`; `0` = no look-ahead area.
- `vwap_period`: as in siblings (`>= 1`, else `ValueError`).
- vp keys optional with the same `VP_DEFAULTS` fallback as siblings.

### 4.3 Data loading & index layout (`analyze`)

- `pad_minutes = max(max_pattern_width, max_wall_search_width, vwap_period)`; `load_from = date_from − pad_minutes`.
- `load_to = date_to + look_ahead minutes` (string via the sibling `DT_FMT` arithmetic). `data = load_candles(asset, load_from, load_to)` — dataset may end earlier; whatever comes back is the truth.
- `start_idx = searchsorted(ts, to_ms(date_from), "left")`; `end_idx = searchsorted(ts, to_ms(date_to), "right")` — identical to siblings, so scans and volume profile are byte-identical to running the two originals over the same range. Rows `>= end_idx` are the look-ahead area (possibly empty).
- vwap computed over the **full** loaded array (`compute_vwap` as siblings — period 1 = col 8, N>1 = `rolling_vwap(q, v, N)`, both through `sanitize_vwap`). Rolling vwap is look-back-only, so look-ahead rows never influence in-range values. When `vwap_period > 1` also produce `vwap_plain = sanitize_vwap(data[:, 8])` for drawing.
- Scans: jit warm-up as siblings, then `rising_from_bowl_scan(vwap, start_idx, end_idx, **bowl_kwargs)` and `falling_from_dome_scan(vwap, start_idx, end_idx, **dome_kwargs)` — anchors never enter the look-ahead area.
- Dedupe both (bowl key `bottom_idx`, dome key `top_idx`); print sibling-style stats **per pattern type** (detections, distinct count, detections/pattern, width/extent/recovery-or-decline/r² summaries).
- Volume profile: identical to siblings — window `[max(start_idx, end_idx − vp_lookback), end_idx)` over `vwap` (the detection series) and `data[:, 5]`; `compute_kde` + `recursive_poc`; console POC lines additionally print `va {va_bps:.1f} bps`.
- Every step keeps the one-line elapsed-time print style.

### 4.4 Chart (`build_figure`) — single figure, 2 rows (plot + POC table)

Layout, colors, axes, spikes, vertical sidebar legend, title style: copy the siblings. Title: `{ASSET} 1m — bowl_and_dome` with a symmetric-params subtitle (`min_w`, `max_w`, `extent`, `pos`, `retrace`, `wall_search`, `vwap_p`, `look_ahead`).

**Candles & vwap** — over the entire loaded array (pad + range + look-ahead), exactly as siblings. vwap lines:
- `vwap_period == 1`: one line `VWAP` (current style, `#2e2e2e`).
- `vwap_period > 1`: `VWAP(N)` (detection series, `#2e2e2e`) **and** `VWAP` (plain, thinner, distinct muted color, e.g. `#9a8c98`), each its own legend item.

**Bowls** — one fixed color for all (`#2e6f95`). Per distinct bowl keep the sibling traces (dotted parabola + 4 markers with the full per-bowl hover info incl. `×detections`), but: every bowl trace has `legendgroup="bowls"`, only the first sets `showlegend=True` with `name="Bowls"`; legend `groupclick` must toggle the whole group (`"togglegroup"`), so one click hides/shows all bowls.

**Domes** — same scheme, color `#c1666b`, group `"domes"`, single legend item `Domes`.

**POC lines** — dashed, per-rank palette color as siblings, but x-span extends from `times[start_idx]` to the **last loaded candle** (through look-ahead). Hover text per POC (the table info + the new value):

```
POC {rank}
price {poc_price:,.2f}
kde volume {poc_volume:,.3f} ({pct_of_poc1:.1f}% of POC 1)
VA [{val:,.2f}, {vah:,.2f}]
va {va_bps:.1f} bps
```

`va_bps = (vah − val) / vp["current_price"] * 1e4`.

**VAL/VAH lines** — per POC, two horizontal lines at `val` and `vah`, same x-span and color as their POC, thinner (~0.8 vs 1.6, e.g. dotted), `legendgroup` = their POC's group, `showlegend=False`, hover naming the POC (`POC {rank} VAL {val:,.2f}` / `... VAH ...`).
- **Preferred behavior:** hidden by default (`visible=False`), shown only while the cursor hovers the POC line. Implement with `fig.write_html(..., post_script=JS)`: build a rank→[val_trace_idx, vah_trace_idx] map in Python, embed it in the JS; on `plotly_hover` over a POC trace `Plotly.restyle(gd, {visible: true}, pair)`, on `plotly_unhover` restyle back to `false`. Identify POC traces via `meta` (e.g. `meta={"poc_rank": r}`).
- **Fallback (accepted by user):** if the hover wiring proves unreliable, make them always visible; the shared `legendgroup` then lets the POC legend item toggle them with their POC.

**Look-ahead boundary** — when the look-ahead area is non-empty, a full-height vertical line at `times[end_idx]` (first look-ahead candle) on the price subplot, subtle solid/dashed gray (e.g. `#9a9990`), with a small `look-ahead` annotation. `look_ahead == 0` or no rows after `end_idx` → no line.

**POC table** — sibling table plus one new column `VA bps` (`{va_bps:.1f}`), i.e. `Rank | Price | KDE Volume | % of POC 1 | VAL | VAH | VA bps`; rank cells colored per rank as now; empty-poc handling as siblings.

## 5. Non-Goals / Out of Scope

- No changes to `scripts/tests/rising_from_bowl/`, `scripts/tests/falling_from_dome/`, their configs/HTML, or `notebooks/tests/volume_profile/volume_profile_bowl_dome_colab.ipynb`.
- No changes to any `packages/` code or `agents/packages/` doc (no new/changed package API).
- No shared/refactored common module for the three test scripts; no unit tests; no notebook.
- No committed sample HTML required (siblings have one, but generating/committing it is optional — the script writing it locally on a verification run is enough).

## 6. Assumptions

- Bowls and domes may overlap on the chart; both are drawn, no interaction/merging between the two detectors.
- POC legend behavior stays per-rank (one legend item per POC), as in the siblings — the single-item requirement applies to bowls/domes only.
- The volume profile is computed once at `end_idx` (as siblings) — not recomputed inside the look-ahead area.
- `vwap_period > 1` detection/VP source stays `VWAP(N)` (user-confirmed); plain `VWAP` is a display-only extra.
- Vertical-line x position `times[end_idx]` (first look-ahead candle) is an acceptable rendering of "end of date_to / beginning of look-ahead".

## 7. Acceptance Criteria

1. `python3 scripts/tests/bowl_and_dome/bowl_and_dome_test.py` (default config) runs end-to-end and writes `bowl_and_dome_btcusdt.html`; an explicit config path as argv[1] also works.
2. With the default config, bowl detections/dedupe counts equal `rising_from_bowl_test.py`'s output for the same range, and dome counts equal `falling_from_dome_test.py`'s (same candles, same series, same anchors).
3. The chart shows: candles + vwap continuing ~240 min past `date_to`, a vertical boundary line at the start of that area, no bowl/dome parabola or marker at/after the boundary, POC + VAL/VAH lines reaching the chart's right edge.
4. Legend contains exactly one `Bowls` item and one `Domes` item; clicking each toggles every bowl/dome trace at once. All bowls share one color; all domes share another.
5. POC hover shows price, kde volume, % of POC 1, VA range, and `va bps`; the table has the matching `VA bps` column.
6. VAL/VAH lines: thinner than POC lines, same color as their POC; either appearing on POC hover (preferred) or always visible (fallback) — one of the two, working.
7. With `"vwap_period": 5` two vwap lines appear (`VWAP`, `VWAP(5)`); detection/VP numbers match a `VWAP(5)`-driven run.
8. With `"look_ahead": 0` the chart ends at `date_to` with no boundary line, and all calculations are unchanged.
9. Config contains only symmetric pattern keys; renaming e.g. `min_pattern_extent_bps` changes both detectors' thresholds.
10. Console output keeps the sibling style: per-step elapsed lines, per-pattern-type stats, VP summary + POC lines (now with `va ... bps`).

## 8. Open Questions

None — scope (combined test only) and vwap source (`VWAP(N)` drives detection/VP) were confirmed by the user.

## 9. Notes for the Downstream Coding Agent

- Start from `rising_from_bowl_test.py`; the dome side differs only in imported scan/columns, dedupe key, marker labels/symbols, and stats field names — diff the two siblings first to see the exact delta.
- Keep two `COL` maps (`BOWL_COL` from `SCAN_COLUMNS`, `DOME_COL` from `DOME_SCAN_COLUMNS`); never hardcode column numbers.
- Warm both scan jits (sibling warm-up pattern, one call each) before the timed scans.
- `times[end_idx:]` may be empty (dataset ends at/before `date_to`) — guard the boundary line and the POC x-span right edge (`times[-1]`).
- For the hover-linked VAL/VAH, `post_script` receives the plot div id as `{plot_id}` in `write_html` — use it to grab the graph div. Test the interaction in a browser once; if hover events misbehave with `hovermode="closest"` + spikes, take the always-visible fallback without further ceremony.
- Do not modify the sibling scripts even for trivially shareable helpers — copy locally.
- Commit and push per the session's branch instructions.
