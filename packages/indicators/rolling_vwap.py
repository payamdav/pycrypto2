import numba as nb
import numpy as np


@nb.njit(inline='always')
def rolling_vwap(quotes, volumes, window=60):
    n = len(quotes)
    out = np.zeros(n, dtype=np.float64)
    for i in range(window - 1, n):
        q_sum = 0.0
        v_sum = 0.0
        for j in range(i - window + 1, i + 1):
            q_sum += quotes[j]
            v_sum += volumes[j]
        out[i] = q_sum / v_sum
    if n >= window:
        first = out[window - 1]
        for i in range(window - 1):
            out[i] = first
    return out
