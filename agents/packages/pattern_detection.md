# pattern_detection

Numba-JIT chart-pattern detectors for 1D price arrays (not indicators — each detector answers "does this shape exist right now?", returning `None`/no-row when it doesn't). Two mirror detectors: `rising_from_bowl` (price rising out of a U-shape dip) and `falling_from_dome` (price falling out of a ∩-shape top).

## Import

```python
from packages.pattern_detection import (
    rising_from_bowl, rising_from_bowl_scan, SCAN_COLUMNS,
    falling_from_dome, falling_from_dome_scan, DOME_SCAN_COLUMNS,
)
```

## `rising_from_bowl(prices, min_bowl_width=10, max_bowl_width=120, min_bowl_depth_bps=20.0, bottom_position_limit=0.8, peak_drawdown_limit_bps=15.0, max_peak_search_width=240)`

Detects whether the **last** point of `prices` (1D `float64`, oldest → newest) is rising out of a bowl/U-shape dip. Returns a dict on detection, `None` otherwise. Only `prices[-1]` is ever evaluated as the anchor — call this once per new price if scanning live, or use `rising_from_bowl_scan` for a historical range.

Algorithm (early-exit at every step, `t = len(prices) - 1`):

1. **Horizontal ray scan** — backward from `t-1` (at most `max_bowl_width` steps) for the nearest index `i` where `prices[i] >= prices[t]` (the left rim). Not found, or width `k = t - i < min_bowl_width` → `None`.
2. **Bottom & position** — minimum of `prices[i..t]` at `t_min` (leftmost on ties). Relative position `r = (t_min - i) / k` must land in `[0.5 - L/2, 0.5 + L/2]` (`L = bottom_position_limit`), else `None`.
3. **Depth** — `(prices[t] - min) / prices[t]` in bps must be `>= min_bowl_depth_bps`, else `None`.
4. **Left-wall peak climb** — from `i-1` backward (at most `max_peak_search_width` steps from `t`), track the running peak; stop once a pullback from that peak exceeds `peak_drawdown_limit_bps`. Feeds `recovery_ratio` and the peak fields only — never rejects.
5. **Quadratic fit** — least-squares `y = a·x² + b·x + c` over `prices[i..t]` (`x = 0..k`), via closed-form normal equations (power sums of `x`) — not `np.polyfit`, which isn't `@njit`-compatible. `a <= 0` (not concave-up) → `None`.

### Return dict

| Key | Type | Meaning |
|---|---|---|
| `detected` | bool | always `True` when a dict is returned |
| `left_rim_idx` | int | left rim `i` |
| `right_rim_idx` | int | anchor `t` (`len(prices)-1`) |
| `bottom_idx` | int | index of the minimum |
| `bowl_width` | int | `k = t - i` |
| `bowl_depth_bps` | float | depth vs. `prices[t]`, bps |
| `bottom_position_ratio` | float | `r`, in `[0, 1]` |
| `left_wall_peak_idx` | int | true crest of the left wall |
| `left_wall_peak_price` | float | price at the crest |
| `recovery_ratio` | float | `(prices[t] - min) / (peak - min)`; `0.0` if `peak <= min` |
| `fit_coef_a` / `fit_coef_b` / `fit_coef_c` | float | quadratic fit coefficients |
| `r_squared` | float | fit goodness; `0.0` if the window is flat (`SStot <= 0`) |
| `theoretical_bottom_idx` | float | `i - b/(2a)`, absolute index (may fall outside `[i, t]`) |

Raises `ValueError` if `prices` is not 1D, `min_bowl_width < 2` (quadratic fit needs ≥ 3 points), `max_bowl_width < min_bowl_width`, `bottom_position_limit` outside `[0, 1]`, or `max_peak_search_width < 1`.

## `rising_from_bowl_scan(prices, start_idx=0, end_idx=None, **same detector params)`

Runs the same detector at every anchor `t` in `range(start_idx, end_idx)` (`end_idx=None` → `len(prices)`). Returns a `float64` array, shape `(m, 14)` — one row per **detection** (rejected anchors produce no row), ascending by anchor index. Column order matches `SCAN_COLUMNS` (identical fields to the `rising_from_bowl` dict, minus `detected`):

| Col | Field | Col | Field |
|---|---|---|---|
| 0 | `left_rim_idx` | 7 | `left_wall_peak_price` |
| 1 | `right_rim_idx` | 8 | `recovery_ratio` |
| 2 | `bottom_idx` | 9 | `fit_coef_a` |
| 3 | `bowl_width` | 10 | `fit_coef_b` |
| 4 | `bowl_depth_bps` | 11 | `fit_coef_c` |
| 5 | `bottom_position_ratio` | 12 | `r_squared` |
| 6 | `left_wall_peak_idx` | 13 | `theoretical_bottom_idx` |

All indices are absolute positions in the `prices` array passed in. `SCAN_COLUMNS` is the name tuple in this order — use `SCAN_COLUMNS.index("field_name")` instead of a hardcoded column number.

## `falling_from_dome(prices, min_dome_width=10, max_dome_width=120, min_dome_height_bps=20.0, top_position_limit=0.8, trough_rally_limit_bps=15.0, max_trough_search_width=240)`

Vertical mirror of `rising_from_bowl`: detects whether the **last** point of `prices` is falling out of a dome/∩-shape top. Same contract — dict on detection, `None` otherwise, only `prices[-1]` evaluated.

Algorithm (mirror of the bowl, early-exit at every step):

1. **Horizontal ray scan** — backward from `t-1` (at most `max_dome_width` steps) for the nearest index `i` where `prices[i] <= prices[t]` (the left rim). Not found, or width `k < min_dome_width` → `None`.
2. **Top & position** — maximum of `prices[i..t]` at `t_max` (leftmost on ties); `r = (t_max - i) / k` must land in `[0.5 - L/2, 0.5 + L/2]` (`L = top_position_limit`), else `None`.
3. **Height** — `(max - prices[t]) / prices[t]` in bps must be `>= min_dome_height_bps`, else `None`.
4. **Left-wall trough climb** — from `i-1` backward (at most `max_trough_search_width` steps from `t`), track the running trough; stop once a rally from that trough exceeds `trough_rally_limit_bps`. Feeds `decline_ratio` and the trough fields only — never rejects.
5. **Quadratic fit** — same closed-form fit; `a >= 0` (not concave-down) → `None`.

Not implemented by negating prices into the bowl core: bps ratios divide by the local reference price and are not sign-invariant, so the dome is a standalone mirrored module.

### Return dict

| Key | Type | Meaning |
|---|---|---|
| `detected` | bool | always `True` when a dict is returned |
| `left_rim_idx` | int | left rim `i` |
| `right_rim_idx` | int | anchor `t` (`len(prices)-1`) |
| `top_idx` | int | index of the maximum |
| `dome_width` | int | `k = t - i` |
| `dome_height_bps` | float | height vs. `prices[t]`, bps |
| `top_position_ratio` | float | `r`, in `[0, 1]` |
| `left_wall_trough_idx` | int | true low of the left wall |
| `left_wall_trough_price` | float | price at the trough |
| `decline_ratio` | float | `(max - prices[t]) / (max - trough)`; `0.0` if `trough >= max` |
| `fit_coef_a` / `fit_coef_b` / `fit_coef_c` | float | quadratic fit coefficients (`a < 0`) |
| `r_squared` | float | fit goodness; `0.0` if the window is flat (`SStot <= 0`) |
| `theoretical_top_idx` | float | `i - b/(2a)`, absolute index (may fall outside `[i, t]`) |

Raises `ValueError` under the mirrored conditions: `prices` not 1D, `min_dome_width < 2`, `max_dome_width < min_dome_width`, `top_position_limit` outside `[0, 1]`, `max_trough_search_width < 1`.

## `falling_from_dome_scan(prices, start_idx=0, end_idx=None, **same detector params)`

Same scan contract as `rising_from_bowl_scan`: `(m, 14)` `float64`, one row per detection, ascending by anchor. Column order = `DOME_SCAN_COLUMNS`:

| Col | Field | Col | Field |
|---|---|---|---|
| 0 | `left_rim_idx` | 7 | `left_wall_trough_price` |
| 1 | `right_rim_idx` | 8 | `decline_ratio` |
| 2 | `top_idx` | 9 | `fit_coef_a` |
| 3 | `dome_width` | 10 | `fit_coef_b` |
| 4 | `dome_height_bps` | 11 | `fit_coef_c` |
| 5 | `top_position_ratio` | 12 | `r_squared` |
| 6 | `left_wall_trough_idx` | 13 | `theoretical_top_idx` |

Use `DOME_SCAN_COLUMNS.index("field_name")` instead of hardcoded column numbers.

### Re-detection and deduplication (read before counting or plotting bowls/domes)

**The package never deduplicates.** A single bowl typically re-triggers at many consecutive anchors while price keeps rising out of it — the left part of the bowl (`left_rim_idx`, `left_wall_peak_idx`, `bottom_idx`) stays fixed across those rows, while `right_rim_idx` (the anchor) advances. The same holds for a dome while price keeps falling (`left_rim_idx`, `left_wall_trough_idx`, `top_idx` fixed). Callers that want **distinct patterns** (for counting or drawing) must group scan rows by a stable identity column — for bowls `bottom_idx` or `left_wall_peak_idx`, for domes `top_idx` or `left_wall_trough_idx`; the extremum key (`bottom_idx`/`top_idx`) counts two dips/tops sharing one wall as two patterns and is what `scripts/tests/rising_from_bowl` and `scripts/tests/falling_from_dome` use — keeping one row per group, conventionally the first (smallest `right_rim_idx`, i.e. earliest row since rows are ascending by anchor):

```python
bottom_idx = detections[:, SCAN_COLUMNS.index("bottom_idx")]   # domes: DOME_SCAN_COLUMNS.index("top_idx")
uniq, first_pos, counts = np.unique(bottom_idx, return_index=True, return_counts=True)
distinct_bowls = detections[first_pos]   # first detection of each bowl
detections_per_bowl = counts             # how many times each bowl re-triggered
```

## Notes

- All four functions are thin Python wrappers (validation + packing) around `@nb.njit(cache=True)` cores (`_detect_at`, `_scan_core` per module) — the scan loops are fully jitted, no per-anchor Python overhead.
- `prices` must have at least `max_bowl_width` / `max_dome_width` points behind an anchor to detect anything; shorter context → `None` / no row for that anchor, not an error.
- Worst case per anchor is O(max width + max search width + pattern width); early exits dominate in practice.
