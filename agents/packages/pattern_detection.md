# pattern_detection

Numba-JIT chart-pattern detectors for 1D price arrays (not indicators — each detector answers "does this shape exist right now?", returning `None`/no-row when it doesn't).

## Import

```python
from packages.pattern_detection import rising_from_bowl, rising_from_bowl_scan, SCAN_COLUMNS
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

### Re-detection and deduplication (read before counting or plotting bowls)

**The package never deduplicates.** A single bowl typically re-triggers at many consecutive anchors while price keeps rising out of it — the left part of the bowl (`left_rim_idx`, `left_wall_peak_idx`, `bottom_idx`) stays fixed across those rows, while `right_rim_idx` (the anchor) advances. All re-detections of one physical bowl share the same `left_wall_peak_idx`. Callers that want **distinct bowls** (for counting or drawing) must group scan rows by column 6 and keep one row per group — the first (smallest `right_rim_idx`, i.e. earliest row since rows are ascending by anchor) is the conventional choice:

```python
peak_idx = detections[:, 6]
uniq, first_pos, counts = np.unique(peak_idx, return_index=True, return_counts=True)
distinct_bowls = detections[first_pos]   # first detection of each bowl
detections_per_bowl = counts             # how many times each bowl re-triggered
```

## Notes

- Both functions are thin Python wrappers (validation + packing) around `@nb.njit(cache=True)` cores (`_detect_at`, `_scan_core`) — the scan loop itself is fully jitted, no per-anchor Python overhead.
- `prices` must have at least `max_bowl_width` points behind an anchor to detect anything; shorter context → `None` / no row for that anchor, not an error.
- Worst case per anchor is O(`max_bowl_width` + `max_peak_search_width` + `bowl_width`); early exits dominate in practice.
