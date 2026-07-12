import numba as nb
import numpy as np


@nb.njit(inline='always')
def motion(position, window=60):
    """Velocity, acceleration, jerk of a 1D position series.

    Output shape (n, 3), dtype float64: out[:, 0]=velocity, out[:, 1]=acceleration,
    out[:, 2]=jerk. With step = max(window-1, 1), stage k (1..3) is the per-step
    endpoint difference of the previous stage: x[i] = (prev[i] - prev[i-step]) / step,
    valid from i = k*step. Look-back only — no future leak. Early indices of each
    column are backfilled with its first valid value; columns with no valid index
    stay 0.0. Empty input returns shape (0, 3).
    """
    n = len(position)
    out = np.zeros((n, 3), dtype=np.float64)
    step = window - 1
    if step < 1:
        step = 1
    inv = 1.0 / step
    for i in range(step, n):
        out[i, 0] = (position[i] - position[i - step]) * inv
    for i in range(2 * step, n):
        out[i, 1] = (out[i, 0] - out[i - step, 0]) * inv
    for i in range(3 * step, n):
        out[i, 2] = (out[i, 1] - out[i - step, 1]) * inv
    for c in range(3):
        first = (c + 1) * step
        if first < n:
            v = out[first, c]
            for i in range(first):
                out[i, c] = v
    return out


def _precompute_hat_matrix(window_size):
    t = np.arange(window_size, dtype=np.float64)
    X = np.column_stack([np.ones_like(t), t, t**2, t**3])
    return np.linalg.inv(X.T @ X) @ X.T


@nb.njit()
def _rolling_kinematics_core(prices, H, window_size):
    n = len(prices)
    
    # Kinematic outputs
    velocities = np.full(n, np.nan, dtype=np.float64)
    accelerations = np.full(n, np.nan, dtype=np.float64)
    jerks = np.full(n, np.nan, dtype=np.float64)
    
    # Raw coefficient outputs for plotting/debugging
    c0_arr = np.full(n, np.nan, dtype=np.float64)
    c1_arr = np.full(n, np.nan, dtype=np.float64)
    c2_arr = np.full(n, np.nan, dtype=np.float64)
    c3_arr = np.full(n, np.nan, dtype=np.float64)
    
    t_edge = float(window_size - 1)
    
    for i in range(window_size - 1, n):
        y = prices[i - window_size + 1 : i + 1]
        
        c0 = 0.0
        c1 = 0.0
        c2 = 0.0
        c3 = 0.0
        
        for j in range(window_size):
            p = y[j]
            c0 += H[0, j] * p
            c1 += H[1, j] * p
            c2 += H[2, j] * p
            c3 += H[3, j] * p
            
        # Store raw coefficients
        c0_arr[i] = c0
        c1_arr[i] = c1
        c2_arr[i] = c2
        c3_arr[i] = c3
        
        # Calculate kinematics at the leading edge
        velocities[i] = c1 + (2.0 * c2 * t_edge) + (3.0 * c3 * (t_edge ** 2))
        accelerations[i] = (2.0 * c2) + (6.0 * c3 * t_edge)
        jerks[i] = 6.0 * c3
        
    return velocities, accelerations, jerks, c0_arr, c1_arr, c2_arr, c3_arr


def calculate_market_kinematics(prices, window_size):
    prices_arr = np.asarray(prices, dtype=np.float64)
    H = _precompute_hat_matrix(window_size)
    return _rolling_kinematics_core(prices_arr, H, window_size)

