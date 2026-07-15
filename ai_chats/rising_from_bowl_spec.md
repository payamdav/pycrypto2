# Spec: rising_from_bowl — pattern-detection package + test script

Implementation-ready spec for `code_writer`. Source: user request + external algorithm description; both are fully restated here — this file is self-contained.

## 1. Task Summary

1. New package `packages/pattern_detection/`, module `rising_from_bowl.py`: numba-jitted detector deciding whether the last point of a 1D price array is rising out of a "bowl" (U/V dip). Backward horizontal-ray scan finds the bowl, a backward peak climb finds the left wall's true crest, a quadratic fit extracts shape features. Also a jitted batch scanner over a range of anchors.
2. Test script `scripts/tests/rising_from_bowl/`: json config (asset, date range, detector params) → run detector on `vwap` at every candle in range → print statistics → write one interactive Plotly HTML chart (candles + vwap + distinct bowls).

## 2. Background & Context

- A bowl detected at anchor `t` is normally re-detected at later anchors while price keeps rising: the left part stays fixed, the rising part extends. All re-detections of one bowl share `left_wall_peak_idx`. **Distinct bowl** = unique `left_wall_peak_idx`; keep the **first** detection (smallest anchor) for counting and drawing.
- Candles via `packages/candle_loader` — `(n, 11)` float64, cols `ts`=0, `o`=1, `h`=2, `l`=3, `c`=4, `v`=5, `vwap`=8. Detector input = `vwap`.
- Precedent for detection test scripts: `scripts/tests/turn_point_detect_1/` (repo-root chdir, njit kernels, styled Plotly HTML next to the script).

## 3. Relevant Conventions from `/agents/`

- `agents/general/rules.md` — `requirements.txt` per folder; every package gets a doc in `agents/packages/`; terse writing everywhere.
- `agents/packages/indicators.md` — numba contract: float64 in/out, `@njit`, explicit loops, no high-level numpy inside jit.
- `agents/general/strategy_study_guidelines.md` — defaults (price = `vwap`, asset `btcusdt`, UTC `"YYYY-MM-DD HH:MM:SS"`), elapsed-time print per sub-task, look-back-only discipline (no future leak).
- `agents/general/paths_and_files.md` — reusable code → `packages/`, test script → `scripts/tests/`.

## 4. Functional Requirements

### 4.1 Package `packages/pattern_detection/`

Files: `__init__.py` (re-export both public functions), `rising_from_bowl.py`, `requirements.txt` (numpy, numba). New doc `agents/packages/pattern_detection.md`: purpose, API, scan column table, re-detection/dedup note, minimum-length note.

#### Public API

```python
def rising_from_bowl(
    prices: np.ndarray,               # 1D float64, oldest → newest
    min_bowl_width: int = 10,         # min bowl width, minutes
    max_bowl_width: int = 120,        # ray-scan limit, minutes
    min_bowl_depth_bps: float = 20.0, # min depth vs current price, bps (1 bps = 0.01%)
    bottom_position_limit: float = 0.8,   # allowed bottom band centered at 0.5
    peak_drawdown_limit_bps: float = 15.0,# trailing stop of the peak climb, bps
    max_peak_search_width: int = 240,     # peak climb goes back at most this far from t
) -> dict | None

def rising_from_bowl_scan(
    prices: np.ndarray, start_idx: int = 0, end_idx: int | None = None,
    ... same detector params ...
) -> np.ndarray                       # shape (m, 14) float64, see column table
```

`rising_from_bowl` evaluates only the last index (`t = len(prices) - 1`). `rising_from_bowl_scan` evaluates every anchor `t` in `range(start_idx, end_idx)` (`end_idx=None` → `len(prices)`), appends one row per detection, rows in ascending `t`. The package never dedups — every detection is reported, including re-detections of one bowl; distinct-bowl grouping is test-script logic only (§4.2). Both are thin Python wrappers (validation + dict/array packing) around `@njit(cache=True)` cores; the scan loop and per-anchor core are jit-to-jit — no Python object per anchor.

Validation (wrappers, `ValueError`): `min_bowl_width >= 2` (quadratic fit needs ≥ 3 points), `max_bowl_width >= min_bowl_width`, `0.0 <= bottom_position_limit <= 1.0`, `max_peak_search_width >= 1`; prices cast/checked to 1D float64 contiguous.

#### Detection algorithm (per anchor `t`, `P_t = prices[t]`, early exit at every step)

1. **Context**: `t < max_bowl_width` → None (array length insufficient for the full ray span).
2. **Horizontal ray scan**: scan `j = t-1` down to `t - max_bowl_width` (inclusive); left rim `i` = first `j` with `prices[j] >= P_t`; none found → None. `k = t - i`; `k < min_bowl_width` → None.
3. **Bottom & position**: over `W = prices[i..t]` find minimum `P_min` at first-minimum index `t_min` (ties → first). `r = (t_min - i) / k`; reject (→ None) if `r < 0.5 - L/2` or `r > 0.5 + L/2` where `L = bottom_position_limit` (boundary values pass).
4. **Depth**: `depth_bps = (P_t - P_min) / P_t * 1e4`; `< min_bowl_depth_bps` → None.
5. **Left-wall peak climb**: `peak = prices[i]`, `peak_idx = i`; for `j = i-1` down to `max(0, t - max_peak_search_width)`: if `prices[j] > peak` → update peak/peak_idx; else if `(peak - prices[j]) / peak * 1e4 > peak_drawdown_limit_bps` → stop (crest passed).
6. **Quadratic fit** over `W` with `x = 0..k`, `y = W`: least-squares `y = a·x² + b·x + c`. `a <= 0` → None. `r_squared = 1 - SSres/SStot` (`SStot <= 0` → `0.0`). `theoretical_bottom_idx = i - b / (2a)` (float, absolute index, may fall outside `[i, t]`).
7. **Recovery**: `recovery_ratio = (P_t - P_min) / (peak - P_min)`; denominator `<= 0` → `0.0`.

Ordering note: the source description narrates the peak climb before steps 3–4; it has no rejection power (only feeds `recovery_ratio` and identity fields), so it is deliberately run after the cheap rejects — results identical, less wasted work.

#### Return

`rising_from_bowl` dict (None when rejected) — exactly these keys; `*_idx`/`bowl_width` as Python int:

```python
{"detected": True,
 "left_rim_idx": i, "right_rim_idx": t, "bottom_idx": t_min,
 "bowl_width": k, "bowl_depth_bps": depth_bps, "bottom_position_ratio": r,
 "left_wall_peak_idx": peak_idx, "left_wall_peak_price": peak, "recovery_ratio": ...,
 "fit_coef_a": a, "fit_coef_b": b, "fit_coef_c": c,
 "r_squared": ..., "theoretical_bottom_idx": ...}
```

`rising_from_bowl_scan` row columns (float64; same fields minus `detected`):

| col | field | col | field |
|---|---|---|---|
| 0 | left_rim_idx | 7 | left_wall_peak_price |
| 1 | right_rim_idx (anchor t) | 8 | recovery_ratio |
| 2 | bottom_idx | 9 | fit_coef_a |
| 3 | bowl_width | 10 | fit_coef_b |
| 4 | bowl_depth_bps | 11 | fit_coef_c |
| 5 | bottom_position_ratio | 12 | r_squared |
| 6 | left_wall_peak_idx | 13 | theoretical_bottom_idx |

Indices are into the `prices` array as passed.

### 4.2 Test script `scripts/tests/rising_from_bowl/`

Files: `rising_from_bowl_test.py`, `config.json`, `requirements.txt` (numpy, numba, pandas, plotly). Output `rising_from_bowl_{asset}.html` next to the script.

#### Config (`config.json`; optional CLI arg = alternate config path)

```json
{
  "asset": "btcusdt",
  "date_from": "2026-04-15 00:00:00",
  "date_to": "2026-04-21 23:59:00",
  "min_bowl_width": 10,
  "max_bowl_width": 120,
  "min_bowl_depth_bps": 20.0,
  "bottom_position_limit": 0.8,
  "peak_drawdown_limit_bps": 15.0,
  "max_peak_search_width": 240
}
```

Datetimes UTC. Single asset string. The whole run is one function receiving the config dict.

#### Flow

1. Repo-root pattern (as `turn_point_detect_1`): `REPO_ROOT = Path(__file__).resolve().parents[3]`, `sys.path.insert`, `os.chdir(REPO_ROOT)` — candle cache stays at `<repo>/data`.
2. Load candles `[date_from - pad, date_to]` with `pad = max(max_bowl_width, max_peak_search_width)` minutes, so every in-range anchor has full look-back context. Sanitize `vwap`: non-finite or `v == 0` → previous valid value; leading bad rows → first valid.
3. Anchors = indices with `ts` in `[date_from, date_to]` (`np.searchsorted`). Run `rising_from_bowl_scan(vwap, start_idx, end_idx, **params)`.
4. Dedupe: unique col 6 (`left_wall_peak_idx`), keep first occurrence (rows ascend by anchor → first = earliest detection), count per bowl.
5. Print statistics, build chart, write HTML, print its path. Every step prints one elapsed-time line (warm the JIT on a tiny array first, or state that the first run includes compile).

#### Statistics (one line each, terse)

- `{asset} {date_from} → {date_to} | candles {n_anchors}`
- `detections {m} | distinct bowls {d}` ← the three mandatory counts
- `detections/bowl min {..} mean {..} max {..}`
- Over distinct bowls (first detections): `width mean/median`, `depth_bps mean/median`, `recovery mean/median`, `r2 mean/median`
- Per-step elapsed lines. Zero detections must run cleanly (zeros, chart without bowls).

#### Chart (Plotly, single pane, "beautiful & well done")

- **Candles**: candlestick from o/h/l/c, rangeslider off, muted green/red, legend entry "Candles".
- **VWAP**: line, legend entry "VWAP".
- **Per distinct bowl** `n` (chronological, 1-based), one legend entry `Bowl {n} ×{detections}`, `legendgroup="bowl{n}"`, `legend.groupclick="togglegroup"`; colors cycle a small qualitative palette:
  - fitted parabola `ŷ = a·x² + b·x + c`, `x = 0..k`, plotted at candle times `[i..t]`;
  - one marker trace, symbol per point: left_wall_peak (star), left rim (open circle), bottom (circle), anchor/right rim (triangle-up);
  - tooltip on every point of the bowl's traces, multi-line: bowl #, ×detection count, peak/rim/bottom/anchor time+price, width (m), depth (bps), position ratio, recovery, fit a, R², theoretical-bottom time (`times[i] + (theoretical_bottom_idx - i)` minutes).
- **Interactions**: legend click removes/restores any drawing (each bowl, candles, vwap); crosshair = x+y spikes (`showspikes`, `spikemode="across"`, `spikesnap="cursor"`), `hovermode="closest"`; native zoom/pan + `config={"scrollZoom": True}` in `write_html`.
- Styling in the spirit of `turn_point_detect_1` (paper `#f9f9f7`, plot `#fcfcfb`, grid `#e1e0d9`, system-ui font); title `{ASSET} 1m — rising_from_bowl` with a param subtitle; height ≈ 800.
- `n_anchors > 50_000` → print a chart-size warning, still draw.

## 5. Non-Goals / Out of Scope

- No study framework (no tags, no `data/{study}/` tree, no indicator cache) — this is a `scripts/tests/` script per the request wording and precedent.
- No trading logic, backtest, labels, or ML; no multi-asset loop; no notebook; no changes to existing packages.

## 6. Assumptions

1. Location `scripts/tests/rising_from_bowl/` (user said "test script"; `turn_point_detect_1` precedent), outputs next to the script.
2. Default test range 7 days (2026-04-15 → 2026-04-21, ≈10k candles): the chart embeds all candles, so the guideline's full 2024→2026 range would produce an unusable ~100 MB HTML. User widens via config when needed.
3. Signature adds `max_peak_search_width: int = 240` — makes the description's "optionally stop if `j < t - 2·max_bowl_width`" explicit and configurable (240 = 2 × default `max_bowl_width`). Without a bound the climb is O(n) per anchor during long declines, killing the full-range scan.
4. Peak climb runs after position/depth rejects (see ordering note) — result-identical.
5. "Insufficient array length" = `t < max_bowl_width` → None (full ray span unavailable).
6. "Chart map" = Plotly legend; per-bowl legend entries + Candles + VWAP satisfy "all drawings removable by click".
7. Ties: rim = first backward `j` with `prices[j] >= P_t`; bottom = first minimum.
8. Detector runs on sanitized `vwap`.

## 7. Acceptance Criteria

1. `from packages.pattern_detection import rising_from_bowl, rising_from_bowl_scan` works from repo root.
2. None cases: monotone rise (no rim) → None; monotone fall (rim at `t-1`, `k=1`) → None; flat array → None; array shorter than `max_bowl_width + 1` → None.
3. Synthetic noise-free parabolic bowl (e.g. 100 → −0.5 % over 40 m → back near 100, preceded by a crest +0.3 % above 100) → detected: `r_squared > 0.99`, `|bottom_position_ratio - 0.5| <= 0.1`, `left_wall_peak_idx` = constructed crest, `fit_coef_a > 0`, `recovery_ratio` ∈ (0, 1].
4. Fit coefficients match `np.polyfit(x, y, 2)` within 1e-6 relative on sampled real windows (polyfit is the reference only — never called inside njit).
5. Scan ≡ single-point: on a ~3-day real slice, scan rows equal `rising_from_bowl(prices[:t+1], **params)` at every anchor — also proves look-back-only behavior.
6. Per-anchor path fully jitted; default 7-day scan ≲ 1 s after JIT; elapsed lines printed per step.
7. `python3 rising_from_bowl_test.py` with default config prints the three mandatory counts + summaries and writes `rising_from_bowl_btcusdt.html`.
8. Dedup correct: distinct = unique `left_wall_peak_idx`, first detection kept, ×counts match.
9. Chart contains candles, vwap, and per-bowl groups; legend toggles each drawing; crosshair spikes work; bowl tooltips show all §4.2 fields; zoom/pan/scrollZoom active.
10. Param validation raises `ValueError` per §4.1.
11. `agents/packages/pattern_detection.md` written; `requirements.txt` in both new folders; all prints/docs terse.

## 8. Open Questions

None blocking — assumptions 1–3 are the judgment calls to review.

## 9. Notes for the Coding Agent

- Quadratic fit inside njit via normal equations: closed-form power sums for `x = 0..k` — `Σx = k(k+1)/2`, `Σx² = k(k+1)(2k+1)/6`, `Σx³ = (k(k+1)/2)²`, `Σx⁴ = k(k+1)(2k+1)(3k²+3k−1)/30` — one O(k) pass for `Σy, Σxy, Σx²y`, then Cramer's rule on the 3×3 system. float64 is ample for k ≤ a few thousand (default 120).
- `SStot` uses the window mean; guard `SStot <= 0 → r_squared = 0.0`. Guard recovery denominator likewise (§4.1.7).
- Scan core: preallocate `(n_anchors, 14)`, fill sequentially, return a trimmed copy.
- `@njit(cache=True)` on cores; keep them module-level.
- Dedupe one-liner: `np.unique(det[:, 6], return_index=True, return_counts=True)` — `return_index` picks the first detection because rows ascend by anchor.
- Times: `pd.to_datetime(ts, unit="ms")`; 1-minute candles → theoretical-bottom time = `times[i] + (theoretical_bottom_idx - i)` minutes.
- ~2 traces per bowl keeps the figure light even with hundreds of bowls.
- Worst-case per anchor is O(`max_bowl_width` + `max_peak_search_width` + k); early exits dominate in practice — a full-history (~1.3 M candles) scan must stay in the seconds range.
