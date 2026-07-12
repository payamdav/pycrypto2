# indicators

A collection of Numba-JIT compiled technical indicators that operate on 1D NumPy arrays. All functions are `@njit(inline='always')` — they compile on first call and can be inlined into larger Numba kernels.

## Import

```python
from packages.indicators import ma, wma, vwma, rsi_1_1, stddev, rolling_robust_z_score, rolling_median_iqr, rolling_mean_stddev, rolling_vwap, motion
```

Individual imports:

```python
from packages.indicators.ma import ma
from packages.indicators.wma import wma
from packages.indicators.vwma import vwma
from packages.indicators.rsi import rsi_1_1
from packages.indicators.stddev import stddev
from packages.indicators.rolling_robust_z_score import rolling_robust_z_score
from packages.indicators.rolling_robust_z_score import rolling_median_iqr
from packages.indicators.rolling_mean_stddev import rolling_mean_stddev
from packages.indicators.rolling_vwap import rolling_vwap
from packages.indicators.motion import motion
```

## Conventions

- Input: 1D `np.ndarray`, **must be `dtype=np.float64`**.
- Output: newly allocated 1D `np.ndarray`, same shape, `dtype=np.float64`.
- `window` defaults to `60`.
- Indices where a full window is not yet available are backfilled with the first computed (fully-windowed) value, so the output has no leading `0.0`/zero-warmup segment.
- All internal loops use explicit `for` loops (no NumPy high-level calls inside `@njit`).

---

## Functions

### `ma(array, window=60)`
Simple moving average. `output[i] = mean(array[i-window+1 : i+1])` for `i >= window-1`; indices `< window-1` are backfilled with `output[window-1]`.

```python
out = ma(prices, window=20)
```

---

### `wma(array, weights, window=60)`
Weighted moving average. `weights` is a 1D `float64` array of length `window`; normalized internally.
`output[i] = sum(array[i-window+1:i+1] * weights) / sum(weights)` for `i >= window-1`; indices `< window-1` are backfilled with `output[window-1]`.

```python
w = np.arange(1, 21, dtype=np.float64)
out = wma(prices, w, window=20)
```

---

### `vwma(array, volume, window=60)`
Volume-weighted moving average. `volume` must be the same shape as `array`.
`output[i] = sum(array * volume) / sum(volume)` over the window for `i >= window-1`; indices `< window-1` are backfilled with `output[window-1]`.

```python
out = vwma(prices, volumes, window=20)
```

---

### `rsi_1_1(array, window=60)`
RSI using Wilder's smoothing (`alpha = 1/window`), scaled to `[-1, 1]` via `(RSI - 50) / 50`.
Seeded from the first `window` price changes; valid from index `window` onward.
Indices `< window` are backfilled with `output[window]`. When `avg_loss == 0`, output is `1.0`.

```python
out = rsi_1_1(prices, window=14)
# -1 = oversold, 0 = neutral, +1 = overbought
```

---

### `stddev(array, window=60)`
Rolling **population** standard deviation (divides by N, not N-1).
`output[i] = std(array[i-window+1:i+1])` for `i >= window-1`; indices `< window-1` are backfilled with `output[window-1]`.

```python
out = stddev(prices, window=20)
```

---

### `rolling_robust_z_score(array, window=60)`
Rolling robust z-score: `(x - median) / IQR` where `Q1 = sorted[W//4]`, `Q3 = sorted[3*W//4]`.
Returns `0.0` when `IQR == 0` (including at `i == window-1`). Indices `< window-1` are backfilled with `output[window-1]`.
Uses an incremental sort: the first window is fully sorted once; subsequent windows replace the outgoing value with the incoming value and bubble it into position in O(W).

```python
out = rolling_robust_z_score(prices, window=60)
```

---

### `rolling_median_iqr(array, window=60)`
Returns the rolling median and IQR for each index as a 2-column array.

- **Output:** shape `(n, 2)`, `dtype=float64`. `out[i, 0]` = median, `out[i, 1]` = IQR.
- **Window:** left look-back window `array[max(0, i-window+1) : i+1]`, effective length `m = min(i+1, window)`.
- **Partial early windows:** every index gets a real computed value — no `0.0` padding (deviation from the package-wide convention). `out[0] == [array[0], 0.0]`.
- **Quartile convention:** `Q1 = sorted[m//4]`, `Q3 = sorted[3*m//4]` — identical to `rolling_robust_z_score`.
- Empty input returns shape `(0, 2)`.

```python
out = rolling_median_iqr(prices, window=60)
medians = out[:, 0]
iqrs    = out[:, 1]
```

---

---

### `rolling_mean_stddev(array, window=60)`
Rolling mean and population standard deviation over a left look-back window. Returns `(n, 2)` float64 array: `out[i, 0]` = mean, `out[i, 1]` = stddev.
Window for index `i`: `array[max(0, i-window+1) : i+1]`, effective length `m = min(i+1, window)`.
Partial early windows are NOT padded — every index gets a real value (deviates from package-wide `0.0` convention). Stddev is population (divide by `m`).

```python
out = rolling_mean_stddev(prices, window=60)
means   = out[:, 0]
stddevs = out[:, 1]
```

---

### `rolling_vwap(quotes, volumes, window=60)`
Rolling VWAP. `quotes` and `volumes` must be the same shape.
`output[i] = sum(quotes[i-window+1:i+1]) / sum(volumes[i-window+1:i+1])` for `i >= window-1` (window inclusive of the current item); indices `< window-1` are backfilled with `output[window-1]`.

```python
out = rolling_vwap(quotes, volumes, window=60)
```

---

### `motion(position, window=60)`
Velocity, acceleration, jerk of a 1D position series as a 3-column array.

- **Output:** shape `(n, 3)`, `dtype=float64`. `out[:, 0]` = velocity, `out[:, 1]` = acceleration, `out[:, 2]` = jerk.
- **Formula:** with `step = max(window-1, 1)`, each stage `k = 1..3` is the per-step endpoint difference of the previous stage: `x[i] = (prev[i] - prev[i-step]) / step`, valid from `i = k*step`. Look-back only — no future leak.
- **Backfill:** per column, indices before its first valid index get the first valid value; a column with no valid index stays all `0.0`.
- `window == 1` means exact one-step differences. Empty input returns shape `(0, 3)`.

```python
out = motion(prices, window=60)
vel, acc, jerk = out[:, 0], out[:, 1], out[:, 2]
```

---

## Usage Example

```python
import numpy as np
from packages.indicators import ma, wma, vwma, rsi_1_1, stddev, rolling_robust_z_score, rolling_median_iqr, rolling_mean_stddev, rolling_vwap, motion

prices = np.random.randn(200).astype(np.float64).cumsum() + 100.0
volume = np.random.rand(200).astype(np.float64) * 1000.0
quotes = prices * volume
weights = np.arange(1, 21, dtype=np.float64)  # length must equal window

ma_out    = ma(prices, window=20)
wma_out   = wma(prices, weights, window=20)
vwma_out  = vwma(prices, volume, window=20)
rsi_out   = rsi_1_1(prices, window=14)
std_out   = stddev(prices, window=20)
rzs_out   = rolling_robust_z_score(prices, window=60)
rmi_out   = rolling_median_iqr(prices, window=60)   # shape (200, 2)
rms_out   = rolling_mean_stddev(prices, window=60)  # shape (200, 2)
rv_out    = rolling_vwap(quotes, volume, window=20)
mot_out   = motion(prices, window=20)               # shape (200, 3)
```
