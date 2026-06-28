import numba as nb
import numpy as np


@nb.njit(inline='always')
def rolling_mean_stddev(array, window=60):
    """Rolling mean and population stddev over a left look-back window.

    Output shape (n, 2), dtype float64. out[i, 0]=mean, out[i, 1]=stddev.
    Window for index i: array[max(0, i-window+1):i+1], length m=min(i+1, window).
    Partial early windows are computed over available items — no 0.0 padding.
    Stddev is population (divide by m). Empty input returns shape (0, 2).
    """
    n = len(array)
    out = np.empty((n, 2), dtype=np.float64)
    if n == 0:
        return out
    for i in range(n):
        m = min(i + 1, window)
        start = i - m + 1
        mean = 0.0
        for k in range(start, i + 1):
            mean += array[k]
        mean /= m
        var = 0.0
        for k in range(start, i + 1):
            diff = array[k] - mean
            var += diff * diff
        var /= m
        out[i, 0] = mean
        out[i, 1] = var ** 0.5
    return out
