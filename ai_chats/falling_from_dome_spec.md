# Spec: falling_from_dome — mirror detector of rising_from_bowl + test script

Implementation-ready spec for `code_writer`. Self-contained; source: user request ("falling from dome is the opposite of rising from bowl — same package, similar pattern detection, vice versa") + the existing `rising_from_bowl` implementation as the authoritative template.

## 1. Task Summary

1. New module `packages/pattern_detection/falling_from_dome.py`: numba-jitted detector deciding whether the **last** point of a 1D price array is **falling out of a dome/∩-shape top** — the exact vertical mirror of `rising_from_bowl`. Public API: `falling_from_dome` (single anchor) and `falling_from_dome_scan` (anchor range), plus `DOME_SCAN_COLUMNS`.
2. Test script `scripts/tests/falling_from_dome/`: a mirror of `scripts/tests/rising_from_bowl/` — json config → scan on (rolling) vwap → dedupe → stats prints → one interactive Plotly HTML chart.
3. Update `agents/packages/pattern_detection.md` to document the new detector.

## 2. Background & Context

- Template code: `packages/pattern_detection/rising_from_bowl.py` — copy its structure exactly (thin Python wrappers around `@nb.njit(cache=True)` cores `_detect_at` / `_scan_core`, same validation helpers, same docstring style). The dome detector is that algorithm with the vertical axis flipped.
- Mirror mapping (bowl → dome): rim condition `>=` → `<=` · bottom/min → top/max · depth below current price → height above current price · left-wall **peak** climb (running max, drawdown stop) → left-wall **trough** climb (running min, rally stop) · concave-up `a > 0` → concave-down `a < 0` · recovery_ratio → decline_ratio · theoretical_bottom_idx → theoretical_top_idx.
- Do **not** implement the dome by negating prices and calling the bowl core: bps ratios divide by the local reference price (`p_t`, `peak`) and are not sign-invariant, so negation changes thresholds. Write a standalone mirrored core.
- One physical dome re-triggers at many consecutive anchors while price keeps falling; the package never dedups (same convention as the bowl). Stable identity keys across re-detections: `top_idx` and `left_wall_trough_idx`; the test script groups by `top_idx`.
- Candles via `packages/candle_loader` — `(n, 11)` float64, cols `ts`=0, `o`=1, `h`=2, `l`=3, `c`=4, `v`=5, `q`=6, `vwap`=8. Detector input = sanitized (rolling) vwap, as in the bowl test script.
- Template test script: `scripts/tests/rising_from_bowl/rising_from_bowl_test.py` (already includes `vwap_period` / `sanitize_vwap` / `dedupe_bowls` per `ai_chats/rising_from_bowl_enhancement_spec.md`). Mirror it 1:1.

## 3. Relevant Conventions from `/agents/`

- `agents/general/rules.md` — `requirements.txt` per folder; package doc in `agents/packages/` maintained on significant change; terse text everywhere.
- `agents/packages/indicators.md` — numba contract: float64 in/out, `@njit`, explicit loops.
- `agents/general/strategy_study_guidelines.md` — price = vwap, asset `btcusdt`, UTC `"YYYY-MM-DD HH:MM:SS"`, per-step elapsed prints, look-back only (no future leak).
- `agents/general/paths_and_files.md` — reusable code → `packages/`, test script → `scripts/tests/`.

## 4. Functional Requirements

### 4.1 Module `packages/pattern_detection/falling_from_dome.py`

`packages/pattern_detection/__init__.py` additionally re-exports `falling_from_dome`, `falling_from_dome_scan`, `DOME_SCAN_COLUMNS`. Existing exports (`rising_from_bowl`, `rising_from_bowl_scan`, `SCAN_COLUMNS`) unchanged. `requirements.txt` unchanged (numpy, numba).

#### Public API

```python
def falling_from_dome(
    prices: np.ndarray,                # 1D float64, oldest → newest
    min_dome_width: int = 10,          # min dome width, steps (minutes on 1m candles)
    max_dome_width: int = 120,         # ray-scan limit, steps
    min_dome_height_bps: float = 20.0, # min height vs current price, bps
    top_position_limit: float = 0.8,   # allowed top band centered at 0.5
    trough_rally_limit_bps: float = 15.0,  # trailing stop of the trough climb, bps
    max_trough_search_width: int = 240,    # trough climb goes back at most this far from t
) -> dict | None

def falling_from_dome_scan(
    prices: np.ndarray, start_idx: int = 0, end_idx: int | None = None,
    ... same detector params ...
) -> np.ndarray                        # shape (m, 14) float64, see column table
```

Semantics identical to the bowl pair: `falling_from_dome` evaluates only `t = len(prices) - 1`; the scan evaluates every anchor in `range(start_idx, end_idx)` (`end_idx=None` → `len(prices)`), one row per detection, ascending by anchor, no dedup. Wrappers do validation + dict/array packing; scan loop is jit-to-jit.

Validation (`ValueError`): `min_dome_width >= 2`, `max_dome_width >= min_dome_width`, `0.0 <= top_position_limit <= 1.0`, `max_trough_search_width >= 1`; prices cast/checked to 1D float64 contiguous. Scan additionally checks `start_idx`/`end_idx` bounds as the bowl scan does.

#### Detection algorithm (per anchor `t`, `P_t = prices[t]`, early exit at every step)

1. **Context**: `t < max_dome_width` → None.
2. **Horizontal ray scan**: `j = t-1` down to `t - max_dome_width` (inclusive); left rim `i` = first `j` with `prices[j] <= P_t`; none → None. `k = t - i`; `k < min_dome_width` → None.
3. **Top & position**: over `W = prices[i..t]` find maximum `P_max` at first-maximum index `t_max` (ties → first). `r = (t_max - i) / k`; None if `r < 0.5 - L/2` or `r > 0.5 + L/2`, `L = top_position_limit` (boundaries pass).
4. **Height**: `height_bps = (P_max - P_t) / P_t * 1e4`; `< min_dome_height_bps` → None.
5. **Left-wall trough climb** (never rejects): `trough = prices[i]`, `trough_idx = i`; for `j = i-1` down to `max(0, t - max_trough_search_width)`: if `prices[j] < trough` → update trough/trough_idx; else if `(prices[j] - trough) / trough * 1e4 > trough_rally_limit_bps` → stop.
6. **Quadratic fit** over `W`, `x = 0..k`: closed-form normal equations + Cramer (copy the bowl's fit code verbatim — power-sum accumulation, det guard `det == 0 → None`). `a >= 0` (not concave-down) → None. `r_squared = 1 - SSres/SStot` (`SStot <= 0` → `0.0`). `theoretical_top_idx = i - b / (2a)` (float, absolute, may fall outside `[i, t]`).
7. **Decline**: `decline_ratio = (P_max - P_t) / (P_max - trough)`; denominator `<= 0` → `0.0`.

#### Return

`falling_from_dome` dict (None when rejected); `*_idx`/`dome_width` as Python int:

```python
{"detected": True,
 "left_rim_idx": i, "right_rim_idx": t, "top_idx": t_max,
 "dome_width": k, "dome_height_bps": height_bps, "top_position_ratio": r,
 "left_wall_trough_idx": trough_idx, "left_wall_trough_price": trough,
 "decline_ratio": ..., "fit_coef_a": a, "fit_coef_b": b, "fit_coef_c": c,
 "r_squared": ..., "theoretical_top_idx": ...}
```

`falling_from_dome_scan` row columns (float64; same fields minus `detected`) = `DOME_SCAN_COLUMNS`:

| col | field | col | field |
|---|---|---|---|
| 0 | left_rim_idx | 7 | left_wall_trough_price |
| 1 | right_rim_idx (anchor t) | 8 | decline_ratio |
| 2 | top_idx | 9 | fit_coef_a |
| 3 | dome_width | 10 | fit_coef_b |
| 4 | dome_height_bps | 11 | fit_coef_c |
| 5 | top_position_ratio | 12 | r_squared |
| 6 | left_wall_trough_idx | 13 | theoretical_top_idx |

Indices are absolute positions in the `prices` array as passed.

### 4.2 Test script `scripts/tests/falling_from_dome/`

Files: `falling_from_dome_test.py`, `config.json`, `requirements.txt` (numpy, numba, pandas, plotly). Output `falling_from_dome_{asset}.html` next to the script. Mirror `rising_from_bowl_test.py` function-for-function (`to_ms`, `sanitize_vwap`, `compute_vwap`, `dedupe_domes`, `print_stats`, `build_chart`, `run`, `main`), renamed to dome vocabulary.

#### Config (`config.json`; optional CLI arg = alternate config path)

```json
{
  "asset": "btcusdt",
  "date_from": "2026-04-15 00:00:00",
  "date_to": "2026-04-21 23:59:00",
  "vwap_period": 1,
  "min_dome_width": 10,
  "max_dome_width": 120,
  "min_dome_height_bps": 20.0,
  "top_position_limit": 0.8,
  "trough_rally_limit_bps": 15.0,
  "max_trough_search_width": 240
}
```

#### Flow (identical to the bowl test, dome-renamed)

1. Repo-root pattern: `REPO_ROOT = parents[3]`, `sys.path.insert`, `os.chdir(REPO_ROOT)`.
2. `vwap_period` ≥ 1 (`< 1` → `ValueError`, absent → 1); `pad_minutes = max(max_dome_width, max_trough_search_width, vwap_period)`; load `[date_from - pad, date_to]`.
3. `compute_vwap` (period 1 = col 8; N>1 = `rolling_vwap(q, v, N)` on contiguous copies) → `sanitize_vwap` (bad = non-finite or `<= 0`; print bad count; ffill then bfill; all-bad → `ValueError`).
4. Anchors via `np.searchsorted` on `ts`; jit warm-up on a short slice off the clock; `falling_from_dome_scan(vwap, start_idx, end_idx, **params)`.
5. `dedupe_domes`: group by `DOME_SCAN_COLUMNS.index("top_idx")` (col 2), keep first (earliest) row per dome, chronological order, counts aligned; empty input → empty rows + counts.
6. Stats prints, chart, HTML path print; one elapsed line per step; `n_anchors > 50_000` → size warning, still draw. Zero detections runs cleanly.

#### Statistics (one line each, terse)

- `{asset} {date_from} -> {date_to} | candles {n_anchors}`
- `detections {m} | distinct domes {d}`
- `detections/dome min/mean/max`
- Over distinct domes (first detections): `width(m)`, `height(bps)`, `decline`, `r2` — mean/median each.

#### Chart (Plotly, single pane, style of the bowl chart)

- Candlestick "Candles" + vwap line (`VWAP` / `VWAP({N})`), same colors/styling.
- Per distinct dome `n` (chronological, 1-based): legend entry `Dome {n} ×{detections}`, `legendgroup="dome{n}"`, `groupclick="togglegroup"`, colors cycling the same qualitative palette:
  - fitted parabola `ŷ = a·x² + b·x + c`, `x = 0..k`, at candle times `[i..t]`, dotted;
  - one marker trace: left rim (circle-open), top (circle), left-wall trough (star), anchor (triangle-**down** — mirrors the bowl's triangle-up);
  - multi-line tooltip on every dome trace point: dome #, ×count, width (m), height (bps), position ratio, decline, fit a, R², theoretical-top time (`times[i] + (theoretical_top_idx - i)` minutes), and per-marker time+price.
- Vertical sidebar legend, crosshair spikes, `hovermode="closest"`, `config={"scrollZoom": True}`, paper `#f9f9f7` / plot `#fcfcfb` / grid `#e1e0d9`, system-ui font, height 800, rangeslider off. Title `{ASSET} 1m — falling_from_dome` + param subtitle (all detector params + `vwap_p`).

### 4.3 Doc `agents/packages/pattern_detection.md`

Extend the existing doc (keep the bowl sections intact): update the intro/import line, add `falling_from_dome` / `falling_from_dome_scan` sections mirroring the bowl ones (algorithm, param table meaning, return dict, `DOME_SCAN_COLUMNS` table, validation), and extend the re-detection/dedup section — dome group keys are `top_idx` (col 2, used by the test script) or `left_wall_trough_idx` (col 6); resolve columns via `DOME_SCAN_COLUMNS.index`.

## 5. Non-Goals / Out of Scope

- No changes to `rising_from_bowl` code, its test script, or its behavior; no shared/refactored common core (duplication is deliberate — two independent jitted modules).
- No study framework, trading logic, labels, ML, multi-asset loop, or notebook.
- No regeneration of the committed bowl sample HTML.

## 6. Assumptions

1. Exact vertical mirror of the bowl algorithm, parameter-for-parameter, with the bowl's defaults (10 / 120 / 20.0 bps / 0.8 / 15.0 bps / 240) — user gave no dome-specific values.
2. Mirrored naming (`dome`, `top`, `height`, `trough`, `rally`, `decline`) rather than reusing bowl vocabulary — keeps every field self-describing.
3. Module-level column tuple exported as `DOME_SCAN_COLUMNS` to avoid colliding with the bowl's `SCAN_COLUMNS` in the package namespace; `SCAN_COLUMNS` keeps meaning the bowl (backward compatible).
4. bps ratios use the mirrored local references: height vs `P_t`, rally vs `trough` — the natural mirror of depth vs `P_t`, drawdown vs `peak`. Consequence: dome results are not numerically identical to `rising_from_bowl(-prices)`; that is expected and correct.
5. Ties: rim = first backward `j` with `prices[j] <= P_t`; top = first maximum.
6. "Implement the test" = a `scripts/tests/` script mirroring the existing bowl test (the repo's established test form for this package), not a unit-test suite.
7. Test dedupe key = `top_idx`, mirroring the bowl test's `bottom_idx` choice (counts two tops sharing one trough as two domes).

## 7. Acceptance Criteria

1. `from packages.pattern_detection import falling_from_dome, falling_from_dome_scan, DOME_SCAN_COLUMNS` works from repo root; existing bowl imports unaffected.
2. None cases: monotone fall (no rim ≤ current) → None; monotone rise (rim at `t-1`, `k=1`) → None; flat array → None; `len(prices) < max_dome_width + 1` → None.
3. Synthetic noise-free parabolic dome (e.g. 100 → +0.5 % over 40 steps → back near 100, preceded by a trough 0.3 % below 100) → detected: `fit_coef_a < 0`, `r_squared > 0.99`, `|top_position_ratio - 0.5| <= 0.1`, `left_wall_trough_idx` = constructed trough, `decline_ratio` ∈ (0, 1].
4. Mirror sanity: on that synthetic dome, `rising_from_bowl` on the vertically flipped series (`2*base - prices`) finds a bowl with the same `left_rim_idx`, `right_rim_idx`, and bottom index = the dome's `top_idx` (bps values may differ slightly per assumption 4).
5. Fit coefficients match `np.polyfit(x, y, 2)` within 1e-6 relative on sampled real windows.
6. Scan ≡ single-point: scan rows equal `falling_from_dome(prices[:t+1], **params)` at every anchor of a multi-day real slice — proves look-back-only behavior.
7. Param validation raises `ValueError` per §4.1.
8. `python3 falling_from_dome_test.py` with default config prints candle/detection/distinct counts + summaries + per-step elapsed lines and writes `falling_from_dome_btcusdt.html`; `"vwap_period": N>1` path works; zero-detection run stays clean.
9. Dedup correct: distinct = unique `top_idx`, first detection kept, ×counts match.
10. Chart contains candles, vwap, and per-dome groups; legend toggles every drawing; crosshair spikes, tooltips with all §4.2 fields, zoom/pan/scrollZoom active.
11. `agents/packages/pattern_detection.md` documents both detectors; `requirements.txt` in the new script folder; all prints/docs terse.

## 8. Open Questions

None blocking — assumptions 1, 3, and 7 are the judgment calls to review.

## 9. Notes for `code_writer`

- Start from `rising_from_bowl.py` and flip: comparison directions (steps 2, 3, 5), the height/rally numerators, the `a` sign gate, and the field names. The fit/Cramer/R² block and both wrapper skeletons copy over unchanged.
- Keep `N_COLS = 14`, preallocate `(n_anchors, 14)` in the scan core, return a trimmed copy, `@nb.njit(cache=True)` on module-level cores.
- Test script: copy `rising_from_bowl_test.py`, rename vocabulary, swap detector import/params/columns, anchor marker symbol `triangle-down`, y-axis and layout untouched.
- Theoretical-top time on the chart: `times[i] + pd.to_timedelta(theoretical_top_idx - i, unit="m")`.
- Worst case per anchor O(`max_dome_width` + `max_trough_search_width` + k), early exits dominate — full-history scans must stay in the seconds range, matching the bowl.
