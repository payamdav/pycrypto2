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
    return np.ascontiguousarray(np.linalg.inv(X.T @ X) @ X.T)


@nb.njit()
def _rolling_kinematics_core(prices, H, window_size):
    n = len(prices)
    out = np.zeros((n, 7), dtype=np.float64)
    t_edge = float(window_size - 1)
    start = window_size - 1
    for i in range(start, n):
        c0 = 0.0
        c1 = 0.0
        c2 = 0.0
        c3 = 0.0
        for j in range(window_size):
            p = prices[i - window_size + 1 + j]
            c0 += H[0, j] * p
            c1 += H[1, j] * p
            c2 += H[2, j] * p
            c3 += H[3, j] * p
        out[i, 0] = c1 + (2.0 * c2 * t_edge) + (3.0 * c3 * (t_edge ** 2))
        out[i, 1] = (2.0 * c2) + (6.0 * c3 * t_edge)
        out[i, 2] = 6.0 * c3
        out[i, 3] = c0
        out[i, 4] = c1
        out[i, 5] = c2
        out[i, 6] = c3
    if start < n:
        for c in range(7):
            v = out[start, c]
            for i in range(start):
                out[i, c] = v
    return out


def calculate_market_kinematics(prices, window_size=60):
    """Rolling cubic-polynomial (OLS) kinematics of a 1D series.

    Fits y = c0 + c1*t + c2*t^2 + c3*t^3 over each look-back window and
    evaluates derivatives at the leading edge (the current index). Output
    shape (n, 7), dtype float64: columns are velocity, acceleration, jerk,
    c0, c1, c2, c3. Valid from index window_size-1; earlier indices are
    backfilled with the first valid row. window_size must be >= 4 (cubic
    fit needs 4 points). Not @njit itself (uses np.linalg for the hat
    matrix), but the core loop is JIT compiled.
    """
    if window_size < 4:
        raise ValueError("window_size must be >= 4 for a cubic fit")
    prices_arr = np.ascontiguousarray(prices, dtype=np.float64)
    H = _precompute_hat_matrix(window_size)
    return _rolling_kinematics_core(prices_arr, H, window_size)


@nb.njit(inline='always')
def total_speed(position, window=60):
    """Average total path speed over a rolling look-back window.

    Output shape (n,), dtype float64. With step = max(window-1, 1),
    out[i] = sum(|position[j] - position[j-1]| for j in i-step+1..i) / step —
    total absolute distance travelled inside the window divided by elapsed
    steps. Valid from i = step; earlier indices are backfilled with the
    first valid value.
    """
    n = len(position)
    out = np.zeros(n, dtype=np.float64)
    step = window - 1
    if step < 1:
        step = 1
    inv = 1.0 / step
    for i in range(step, n):
        s = 0.0
        for j in range(i - step + 1, i + 1):
            d = position[j] - position[j - 1]
            if d < 0.0:
                d = -d
            s += d
        out[i] = s * inv
    if step < n:
        v = out[step]
        for i in range(step):
            out[i] = v
    return out


@nb.njit(inline='always')
def kaufman_er(position, window=60):
    """Kaufman's Efficiency Ratio: abs(velocity) / total_speed, in [0, 1].

    Velocity is motion(position, window)[:, 0] (net displacement per step);
    total_speed is the path distance per step over the same window, so the
    ratio is |net move| / total distance — 1 for a straight move, near 0 for
    pure chop. out[i] = 0.0 where total_speed[i] == 0. Output shape (n,),
    dtype float64, backfilled via its backfilled inputs.
    """
    mot = motion(position, window)
    ts = total_speed(position, window)
    n = len(position)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if ts[i] > 0.0:
            v = mot[i, 0]
            if v < 0.0:
                v = -v
            out[i] = v / ts[i]
    return out

