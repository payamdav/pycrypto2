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
