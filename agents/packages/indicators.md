# indicators

A collection of Numba-JIT compiled technical indicators that operate on 1D NumPy arrays. All functions are `@njit(inline='always')` — they compile on first call and can be inlined into larger Numba kernels.

## Import

```python
from packages.indicators import ma, wma, vwma, rsi_1_1, stddev, rolling_robust_z_score
```

Individual imports:

```python
from packages.indicators.ma import ma
from packages.indicators.wma import wma
from packages.indicators.vwma import vwma
from packages.indicators.rsi import rsi_1_1
from packages.indicators.stddev import stddev
from packages.indicators.rolling_robust_z_score import rolling_robust_z_score
```

## Conventions

- Input: 1D `np.ndarray`, **must be `dtype=np.float64`**.
- Output: newly allocated 1D `np.ndarray`, same shape, `dtype=np.float64`.
- `window` defaults to `60`.
- Indices where a full window is not yet available are padded with `0.0`.
- All internal loops use explicit `for` loops (no NumPy high-level calls inside `@njit`).

---

## Functions

### `ma(array, window=60)`
Simple moving average. `output[i] = mean(array[i-window+1 : i+1])` for `i >= window-1`, else `0.0`.

```python
out = ma(prices, window=20)
```

---

### `wma(array, weights, window=60)`
Weighted moving average. `weights` is a 1D `float64` array of length `window`; normalized internally.
`output[i] = sum(array[i-window+1:i+1] * weights) / sum(weights)` for `i >= window-1`, else `0.0`.

```python
w = np.arange(1, 21, dtype=np.float64)
out = wma(prices, w, window=20)
```

---

### `vwma(array, volume, window=60)`
Volume-weighted moving average. `volume` must be the same shape as `array`.
`output[i] = sum(array * volume) / sum(volume)` over the window for `i >= window-1`, else `0.0`.

```python
out = vwma(prices, volumes, window=20)
```

---

### `rsi_1_1(array, window=60)`
RSI using Wilder's smoothing (`alpha = 1/window`), scaled to `[-1, 1]` via `(RSI - 50) / 50`.
Seeded from the first `window` price changes; valid from index `window` onward.
`output[i] = 0.0` for `i < window`. When `avg_loss == 0`, output is `1.0`.

```python
out = rsi_1_1(prices, window=14)
# -1 = oversold, 0 = neutral, +1 = overbought
```

---

### `stddev(array, window=60)`
Rolling **population** standard deviation (divides by N, not N-1).
`output[i] = std(array[i-window+1:i+1])` for `i >= window-1`, else `0.0`.

```python
out = stddev(prices, window=20)
```

---

### `rolling_robust_z_score(array, window=60)`
Rolling robust z-score: `(x - median) / IQR` where `Q1 = sorted[W//4]`, `Q3 = sorted[3*W//4]`.
Returns `0.0` when `IQR == 0` or `i < window-1`.
Uses an incremental sort: the first window is fully sorted once; subsequent windows replace the outgoing value with the incoming value and bubble it into position in O(W).

```python
out = rolling_robust_z_score(prices, window=60)
```

---

## Usage Example

```python
import numpy as np
from packages.indicators import ma, wma, vwma, rsi_1_1, stddev, rolling_robust_z_score

prices = np.random.randn(200).astype(np.float64).cumsum() + 100.0
volume = np.random.rand(200).astype(np.float64) * 1000.0
weights = np.arange(1, 21, dtype=np.float64)  # length must equal window

ma_out    = ma(prices, window=20)
wma_out   = wma(prices, weights, window=20)
vwma_out  = vwma(prices, volume, window=20)
rsi_out   = rsi_1_1(prices, window=14)
std_out   = stddev(prices, window=20)
rzs_out   = rolling_robust_z_score(prices, window=60)
```
