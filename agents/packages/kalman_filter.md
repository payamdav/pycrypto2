# kalman_filter

A high-performance 1D scalar Kalman filter implemented with Numba JIT compilation. Provides both a single-step function and a batch function for processing full sequences.

## Import

```python
from packages.kalman_filter.kalman_fast import kalman_filter_step, kalman_filter_batch
```

Both functions are `@njit` compiled — the first call triggers JIT compilation; subsequent calls run at native speed.

---

## Functions

### `kalman_filter_step(measurement, prev_estimate, prev_error_cov, process_variance, measurement_variance)`

Executes one predict-update-correct cycle. Stateless — the caller maintains state between steps.

| Parameter | Description |
|-----------|-------------|
| `measurement` | Current raw observation `z_k` |
| `prev_estimate` | Posteriori state estimate from previous step `x̂_{k-1}` |
| `prev_error_cov` | Posteriori error covariance from previous step `P_{k-1}` |
| `process_variance` | Process noise `Q` |
| `measurement_variance` | Measurement noise `R` |

Returns `(current_estimate, current_error_cov)` as `(float, float)`.

```python
x, P = kalman_filter_step(z[0], x0=0.0, P0=1.0, Q=1e-4, R=1e-2)
```

---

### `kalman_filter_batch(measurements, initial_estimate, initial_error_cov, process_variance, measurement_variance)`

Processes a full 1D array of measurements in a loop, returning pre-allocated output arrays.

| Parameter | Description |
|-----------|-------------|
| `measurements` | 1D `np.ndarray` of observations |
| `initial_estimate` | Initial state guess `x̂_0` |
| `initial_error_cov` | Initial error covariance `P_0` |
| `process_variance` | Constant `Q` applied across the batch |
| `measurement_variance` | Constant `R` applied across the batch |

Returns `(estimates, error_covariances)` — two `np.ndarray` of shape `(n,)`, dtype `float64`.

```python
estimates, covs = kalman_filter_batch(
    measurements=prices,
    initial_estimate=prices[0],
    initial_error_cov=1.0,
    process_variance=1e-4,
    measurement_variance=1e-2,
)
```

## Tuning

- **Higher `Q` / lower `R`** → filter trusts measurements more, tracks faster but noisier.
- **Lower `Q` / higher `R`** → filter trusts the model more, smoother but slower to react.
