# Volume Profile Package

## Identity

| Key          | Value                       |
|--------------|-----------------------------|
| Package path | `packages/volume_profile/`  |
| Purpose      | Raw-price-space volume-weighted KDE + POC (Point of Control) / Value-Area peak analysis. Generalizes `packages/kde_tools/` from normalized `[-1, 1]` space to raw prices, and adds `point_of_control` / `recursive_poc`. |

| Exported function       | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `make_kernel`            | Re-export of `kde_tools.kernels.make_kernel` (unchanged)                     |
| `weighted_histogram`     | Volume-weighted histogram over `current_price ± bps_range`                  |
| `compute_kde`            | Orchestrator: `weighted_histogram` + kernel convolution                     |
| `point_of_control`       | POC (highest-volume bin) + Value Area of a KDE profile                      |
| `top_kde_peaks`          | Re-export of `kde_tools.peaks.top_kde_peaks` (unchanged)                    |
| `kde_peaks_above_below`  | Top-`n` KDE peaks above/below the current price                            |
| `recursive_poc`          | Iterative POC extraction by Value-Area removal — ranked support/resistance  |

`packages/kde_tools/` is **not modified** — only imported.

---

## Raw-Price-Space Convention

`kde_tools` operates on look-back prices already normalized to `[-1, 1]`, where the
current price maps to `0.0` (see `agents/packages/kde_tools.md`). `volume_profile`
instead works directly on **raw prices**:

- `current_price = prices[-1]` (last element of the input array — no separate
  normalization step).
- Histogram/KDE range: `[current_price * (1 - bps_range / 1e4), current_price * (1 +
  bps_range / 1e4)]` — `bps_range` is the **half-range in basis points**
  (`100` → ±1%).
- Prices strictly outside this range are ignored (`n_excluded` counts them); the
  range borders themselves are **included** (unlike `kde_tools`' default
  `ignore_borders=True`, which excludes exact-border values).
- There is no `range_min`/`range_max`/`ignore_borders` parameter — the range is
  always derived from `current_price` and `bps_range`.

---

## Setup

```python
from packages.volume_profile import (
    make_kernel, weighted_histogram, compute_kde,
    point_of_control, top_kde_peaks, kde_peaks_above_below, recursive_poc,
)
```

Dependencies: `packages/volume_profile/requirements.txt` (`numpy`, `numba`, `scipy`
— the last transitively via reused `kde_tools` code). In a notebook: `%pip install
numpy numba scipy` before import (plus a repo clone — see
`agents/general/rules.md`).

---

## `weighted_histogram`

```python
def weighted_histogram(prices, volumes, bins=200, bps_range=100.0) -> dict
```

`current_price = prices[-1]`; range = `current_price * (1 ± bps_range / 1e4)`.
Delegates binning to the jitted `kde_tools.histogram.weighted_histogram(..., 
ignore_borders=False)` — inclusive borders, `v == range_max` clamps to the last bin.
`bin_width = (range_max - range_min) / bins`; `bin_centers = range_min + (arange(bins)
+ 0.5) * bin_width`. `n_excluded` = count of prices outside `[range_min, range_max]`.

Returns `{"counts", "bin_centers", "bin_width", "current_price", "range_min",
"range_max", "n_excluded"}` (raw price units).

Raises `ValueError`: `prices` not 1D/empty; `volumes` shape mismatch; `bins < 1`;
`bps_range <= 0`; `current_price` (`prices[-1]`) not finite or `<= 0`.

---

## `compute_kde`

```python
def compute_kde(prices, volumes, bins=200, bps_range=100.0,
                 kernel_type="Triangular", bandwidth=5) -> dict
```

Calls `weighted_histogram`, builds a normalized kernel via `make_kernel`, convolves
with `convolve_same` (both reused from `kde_tools` unchanged). Returns the histogram
dict plus `{"kde", "kernel"}`:
`{"kde", "counts", "bin_centers", "bin_width", "kernel", "current_price",
"range_min", "range_max", "n_excluded"}`.

---

## Glossary

- **POC** (Point of Control) — the highest-volume price bin of a profile.
- **Value Area (VA)** — the contiguous price band around the POC holding `va_pct`%
  of the profile's volume.
- **VAL / VAH** — Value Area Low / High, the VA's lower/upper bin-center price.

## `point_of_control`

```python
def point_of_control(kde, bin_centers, va_pct=70.0) -> dict | None
```

`poc_idx = argmax(kde)` (first on ties); Value Area grows by **greedy single-bin
expansion**: at each step absorb whichever unblocked neighbor (`lo-1` or `hi+1`) has
the larger value (tie → **above**), until accumulated volume `>= va_pct% of
kde.sum()` or both sides are exhausted (an under-target VA capped by array edges is
valid). `None` when `kde` is empty or `kde.sum() <= 0`.

Returns `{"poc_idx", "poc_price", "poc_volume", "val_idx", "vah_idx", "val", "vah",
"va_volume", "total_volume"}` (`*_idx` are `int`, rest `float`).

Raises `ValueError`: `kde`/`bin_centers` shape mismatch; `va_pct` outside `(0, 100]`.

The expansion is implemented once in a private helper `_value_area(kde, removed,
poc_idx, target)` shared with `recursive_poc` — `removed` is a bool mask that blocks
expansion exactly like an array edge (all-`False` for `point_of_control`).

## `top_kde_peaks` / `kde_peaks_above_below`

`top_kde_peaks` is `kde_tools.peaks.top_kde_peaks` unchanged (scipy
`find_peaks`/`peak_prominences`, already price-space-agnostic).

```python
def kde_peaks_above_below(kde, bin_centers, current_price, distance=5, n=3,
                            top_identifier="prominence") -> dict
```

Thin wrapper over `kde_tools.peaks.kde_peaks_above_below` with
`split_at=current_price`. "Above" = `bin_centers >= current_price`, "below" =
`bin_centers < current_price`. Returns `{"above_prices", "above_proms",
"below_prices", "below_proms"}` (raw prices).

---

## `recursive_poc`

```python
def recursive_poc(kde, bin_centers, current_price, va_pct=70.0,
                    min_poc_volume_ratio=0.1, max_pocs=None) -> list[dict]
```

Iterative POC extraction: take the strongest remaining level, remove its Value
Area, repeat — producing ranked support/resistance levels. Each rank:

1. `poc_idx` = argmax of `kde` over not-yet-removed bins (first on ties). No
   candidates left → stop.
2. `poc_volume = kde[poc_idx]`. `<= 0` → stop. `rank > 1` and `poc_volume <
   min_poc_volume_ratio * (rank 1's poc_volume)` → stop ("no comparable volume").
3. Value Area via the shared `_value_area` (expansion blocked by already-removed
   bins too), target = `va_pct% of kde` summed over **currently unremoved** bins
   (percentage of volume remaining at this iteration, not the original total).
4. Append `{"rank", "poc_idx", "poc_price", "poc_volume", "val_idx", "vah_idx",
   "val", "vah", "va_volume"}`.
5. **Removal**, relative to `current_price` (fixed across all ranks):
   - VA fully above (`val > current_price`) → `removed[lo:] = True`.
   - VA fully below (`vah < current_price`) → `removed[:hi + 1] = True`.
   - VA straddles `current_price` (otherwise) → `removed[lo:hi + 1] = True`.

   The POC bin always lies inside the removed span, so every iteration removes
   >= 1 bin — guaranteed termination in <= `bins` iterations.
6. `max_pocs` reached → stop.

**Rank order = strength order**: POC volumes are non-increasing with rank. **Entry 1
equals `point_of_control(kde, bin_centers, va_pct)`** on their shared fields (no
bins removed yet). Empty profile → `[]`.

Raises `ValueError`: `kde`/`bin_centers` shape mismatch; `va_pct` outside `(0,
100]`; `min_poc_volume_ratio` outside `[0, 1]`; `max_pocs` neither `None` nor `>=
1`.

---

## End-to-End Usage Example

```python
import numpy as np
from packages.candle_loader import load_candles
from packages.volume_profile import compute_kde, recursive_poc

data = load_candles("btcusdt", "2026-04-15", "2026-04-21 23:59:00")
vwap = data[:, 8]         # price
volumes = data[:, 5]      # candle base volume ("v")

vp = compute_kde(vwap, volumes, bins=200, bps_range=100.0)   # +/-1% of last price
pocs = recursive_poc(vp["kde"], vp["bin_centers"], vp["current_price"],
                      va_pct=70.0, min_poc_volume_ratio=0.1)

for p in pocs:
    print(f"POC {p['rank']}: price {p['poc_price']:.2f} vol {p['poc_volume']:.3f} "
          f"VA [{p['val']:.2f}, {p['vah']:.2f}]")
```

---

## Notes for Agents

- Reuses `make_kernel`, `convolve_same`, the jitted `weighted_histogram` core, and
  `top_kde_peaks` from `kde_tools` — do not duplicate them; `kde_tools` itself stays
  byte-for-byte unchanged.
- `point_of_control(kde, bin_centers, va_pct)` is exactly `recursive_poc(...)[0]`
  restricted to their shared fields (before any Value-Area removal) — use whichever
  is more convenient; both are backed by the same `_value_area` helper.
- Removal in `recursive_poc` uses index slices (`removed[lo:]`, `removed[:hi+1]`,
  `removed[lo:hi+1]`), equivalent to the price-based rule and float-safe.
- All public functions cast `prices`/`volumes`/`kde`/`bin_centers` inputs with
  `np.ascontiguousarray(np.asarray(x, dtype=np.float64))` before use.
- Used by `scripts/tests/rising_from_bowl/` and `scripts/tests/falling_from_dome/`
  for the POC overlay + table on their charts (see those scripts' `analyze()`).
