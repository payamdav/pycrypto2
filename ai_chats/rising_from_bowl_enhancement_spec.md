# Spec: rising_from_bowl test — dedupe by bottom_idx + vwap_period

Enhancement of the existing `scripts/tests/rising_from_bowl/` script (built per `ai_chats/rising_from_bowl_spec.md`). Implementation-ready for `code_writer`; self-contained.

## 1. Task Summary

1. `dedupe_bowls` groups detections by `bottom_idx` (scan col 2) instead of `left_wall_peak_idx` (col 6); align every dependent text/doc.
2. New config key `vwap_period`, default `1` — candles per vwap value. `1` = candle `vwap` column; `N > 1` = `rolling_vwap` over the last `N` candles (current + `N−1` previous), from `packages/indicators/rolling_vwap.py`.
3. `sanitize_vwap` corrected: input is the vwap array itself (not the candle matrix); flags `nan`/`inf`/`<= 0`/any invalid number, prints the bad count, then fills each bad item with the previous valid value. Runs after every vwap source (candle column and rolling).
4. Refactor whatever else these changes touch (timing prints, chart labels, pad width, package doc).

## 2. Background & Context

- `rising_from_bowl_scan` (packages/pattern_detection) re-triggers one physical bowl at many consecutive anchors; the test script keeps one row per bowl. Identity key switches from `left_wall_peak_idx` to `bottom_idx` — per the package doc both stay fixed across re-detections; `bottom_idx` counts a shared crest with two dips as two bowls.
- Candle array cols: `ts`=0, `v`=5, `q`=6, `vwap`=8 (`vwap = q/v`). Rolling vwap over N candles = Σq/Σv → `rolling_vwap(quotes=q, volumes=v, window=N)` (`@njit`; indices `< N−1` backfilled with the first full-window value; `n < window` → all zeros).
- Bad vwap sources: zero-volume candles (`q/v` → nan or stored 0), zero-volume rolling windows (÷0 → inf/nan).

## 3. Relevant Conventions from `/agents/`

- `agents/packages/indicators.md` — `rolling_vwap` contract: 1D float64 in/out, window inclusive of the current item, backfilled head, no nan.
- `agents/general/strategy_study_guidelines.md` — nan is a bad value anywhere in arrays; look-back only (no future leak); per-step elapsed prints.
- `agents/general/rules.md` — terse text; `requirements.txt` per folder (unchanged: numpy, numba, pandas, plotly); package docs stay authoritative.
- `agents/general/paths_and_files.md` — spec lives in `ai_chats/`; older spec files are history, never edited.

## 4. Functional Requirements

### 4.1 `dedupe_bowls`

- Group key: `COL["bottom_idx"]` (col 2). Mechanics unchanged: `np.unique(key, return_index=True, return_counts=True)`, keep the first (earliest, smallest anchor) row per group, chronological order via argsort of first positions, counts aligned. Empty-input path unchanged. Docstring updated.
- `print_stats` and `build_chart` are key-agnostic — no logic change.

### 4.2 `vwap_period`

- `config.json` gains `"vwap_period": 1`. Code reads `int(cfg.get("vwap_period", 1))` (absent → 1); `< 1` → `ValueError`.
- Period 1 → vwap = candle col 8. Period N>1 → `rolling_vwap(data[:, 6], data[:, 5], window=N)` (pass contiguous float64 copies of the strided column views).
- Both paths go through `sanitize_vwap` before any further use (scan, chart).
- Small helper `compute_vwap(data, period)` in the script; `run()` prints one elapsed line `vwap period {N}  [..s]`. When N>1, warm the `rolling_vwap` jit on a tiny array before timing so the line reflects real work.
- Load pad: `pad_minutes = max(max_bowl_width, max_peak_search_width, vwap_period)` so in-range anchors never sit on backfilled rolling-head values.

### 4.3 `sanitize_vwap`

```python
def sanitize_vwap(vwap: np.ndarray) -> np.ndarray
```

- Input: 1D vwap array. Output: new float64 array; input never mutated.
- `bad = ~np.isfinite(vwap) | (vwap <= 0.0)` — covers nan, ±inf, zero, negative.
- Print one line with the bad count (always, 0 included), then correct: each bad item ← nearest valid value to its left (previous vwap, forward-fill); leading bad items (no left neighbor) ← first valid value. All items bad → `ValueError`.
- The former `v == 0` check is dropped — zero-volume candles yield non-finite/0 vwap, caught by the new predicate.

### 4.4 Chart & doc alignment

- VWAP trace legend/hover name: `VWAP` for period 1, `VWAP({N})` otherwise; title param subtitle gains `vwap_p {N}`.
- `agents/packages/pattern_detection.md` dedup section: both `left_wall_peak_idx` (col 6) and `bottom_idx` (col 2) are valid stable group keys; `bottom_idx` separates two dips sharing one crest and is what the repo test script uses; update the example snippet to resolve the column via `SCAN_COLUMNS.index`. Package code unchanged.

## 5. Non-Goals / Out of Scope

- No changes to `packages/pattern_detection/` or `packages/indicators/` code.
- `scripts/studies/mixture_of_trends_following/` has its own independent `sanitize_vwap(candles)` — untouched.
- No regeneration of the committed sample HTML; no tests/notebooks; `ai_chats/rising_from_bowl_spec.md` stays as history.

## 6. Assumptions

1. "fill them by the previous vwap from right to the left of array" = each bad item takes the nearest valid value on its left (the previous vwap in time) — matches the script's existing ffill behavior and the look-back-only rule; leading bad items keep the existing first-valid backfill since the output must be fully valid.
2. `vwap_period` maps 1:1 to `rolling_vwap`'s `window` (both count the current candle).
3. The bad-count line prints even when 0 — cheap confirmation the check ran.
4. Sanitize still runs for period 1 (unchanged from current behavior).

## 7. Acceptance Criteria

1. `dedupe_bowls` groups by col 2 (`bottom_idx`); docstring says so; first-row/counts/order semantics preserved; empty input still returns empty rows + counts.
2. Default config (period 1) produces the same detections as before; dedup now keyed by bottom.
3. `"vwap_period": 3` → `vwap[i] = Σq[i−2..i] / Σv[i−2..i]` for `i ≥ 2`, head backfilled, sanitized, scan runs on it.
4. `sanitize_vwap([nan, 2, inf, 0, -1, 3])` → prints bad count 4 → returns `[2, 2, 2, 2, 2, 3]`.
5. All-bad input raises `ValueError`; missing `vwap_period` behaves as 1; `0` → `ValueError`.
6. Chart: vwap trace named per period; subtitle shows `vwap_p`.
7. `agents/packages/pattern_detection.md` names both grouping keys.
8. All new prints/docstrings terse per `agents/general/rules.md`.

## 8. Open Questions

None blocking — assumption 1 is the judgment call to review.

## 9. Notes for `code_writer`

- `data[:, 6]` / `data[:, 5]` are strided views; wrap in `np.ascontiguousarray` before `rolling_vwap` for the fast jit path.
- Keep pandas `ffill().bfill()` for the fill (already a dependency); set bad → nan first.
- `DETECTOR_KEYS` must not gain `vwap_period` — it is not a detector parameter.
- `rolling_vwap` with `n < window` returns all zeros → sanitize raises all-invalid — acceptable (pathological config only).
- Warm-up call: `rolling_vwap(np.ones(N), np.ones(N), N)` before starting the vwap timer.
