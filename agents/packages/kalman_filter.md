# kalman_filter

High-performance Kalman filter suite (Numba JIT) covering 1D, 2D, and 3D state dimensionalities. Each model provides a stateless step function and a pre-allocated batch function.

## Import

```python
from packages.kalman_filter import (
    kalman_1d_step, kalman_1d_batch,
    kalman_2d_step, kalman_2d_batch,
    kalman_3d_step, kalman_3d_batch,
)
```

All six functions are `@njit` compiled — the first call triggers JIT; subsequent calls run at native speed.

---

## 1D Model — `kalman_fast.py`

Scalar filter: state = `value` (float). `F = 1`, `H = 1`.

### `kalman_1d_step(measurement, prev_estimate, prev_error_cov, process_variance, measurement_variance)`

One predict-update-correct cycle. Returns `(current_estimate, current_error_cov)` as `(float, float)`.

### `kalman_1d_batch(measurements, initial_estimate, initial_error_cov, process_variance, measurement_variance)`

Processes a 1D `float64` array of length N. Returns `(estimates, error_covariances)`, both shape `(N,)`.

```python
estimates, covs = kalman_1d_batch(prices, prices[0], 1.0, 1e-4, 1e-2)
```

### `kalman_1d_batch_adaptive(measurements, process_variance, window)`

Variance-adaptive variant of `kalman_1d_batch`: `measurement_variance` is derived per index
instead of fixed. Computes `rolling_mean_stddev(measurements, window)` (from
`packages.indicators.rolling_mean_stddev`), squares its stddev column to a per-index
`variance` array, and uses `variance[k]` as the measurement variance at step `k`.
`process_variance` stays fixed across all indices. `initial_estimate = measurements[0]`,
`initial_error_cov = variance[0]`. Returns `(estimates, error_covariances)`, both shape
`(N,)`, same contract as `kalman_1d_batch` — no padding, one value per input index.

```python
estimates, covs = kalman_1d_batch_adaptive(prices, 1e-4, window=60)
```

---

## 2D Model — `kalman_2d.py`

Constant-velocity filter: state `x̂ = [value, speed]ᵀ`, shape `(2,1)`.  
`F = [[1, dt],[0, 1]]`, `H = [[1, 0]]`. `Q` is `(2,2)`, `R` is a scalar float.

### `kalman_2d_step(measurement, prev_state, prev_covariance, process_noise, measurement_variance, dt)`

Returns `(state (2,1), covariance (2,2))`.

### `kalman_2d_batch(measurements, initial_state, initial_covariance, process_noise, measurement_variance, dt)`

Returns `(states (N,2,1), covariances (N,2,2))` as `float64` arrays.

```python
import numpy as np
Q = np.eye(2, dtype=np.float64) * 1e-4
x0 = np.zeros((2, 1), dtype=np.float64)
P0 = np.eye(2, dtype=np.float64)
states, covs = kalman_2d_batch(prices, x0, P0, Q, 1e-2, dt=1.0)
# states[:, 0, 0] → filtered values; states[:, 1, 0] → estimated speed
```

---

## 3D Model — `kalman_3d.py`

Constant-acceleration filter: state `x̂ = [value, speed, acceleration]ᵀ`, shape `(3,1)`.  
`F = [[1, dt, 0.5·dt²],[0, 1, dt],[0, 0, 1]]`, `H = [[1, 0, 0]]`. `Q` is `(3,3)`, `R` is a scalar float.

### `kalman_3d_step(measurement, prev_state, prev_covariance, process_noise, measurement_variance, dt)`

Returns `(state (3,1), covariance (3,3))`.

### `kalman_3d_batch(measurements, initial_state, initial_covariance, process_noise, measurement_variance, dt)`

Returns `(states (N,3,1), covariances (N,3,3))` as `float64` arrays.

```python
Q = np.eye(3, dtype=np.float64) * 1e-4
x0 = np.zeros((3, 1), dtype=np.float64)
P0 = np.eye(3, dtype=np.float64)
states, covs = kalman_3d_batch(prices, x0, P0, Q, 1e-2, dt=1.0)
# states[:, 0, 0] → value; states[:, 1, 0] → speed; states[:, 2, 0] → acceleration
```

---

## Tuning

- **Higher `Q` / lower `R`** → trusts measurements more; faster tracking, noisier output.
- **Lower `Q` / higher `R`** → trusts model more; smoother output, slower to react.
