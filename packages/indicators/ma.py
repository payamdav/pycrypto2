import numba as nb
import numpy as np


@nb.njit(inline='always')
def ma(array, window=60):
    n = len(array)
    out = np.zeros(n, dtype=np.float64)
    for i in range(window - 1, n):
        s = 0.0
        for j in range(i - window + 1, i + 1):
            s += array[j]
        out[i] = s / window
    if n >= window:
        first = out[window - 1]
        for i in range(window - 1):
            out[i] = first
    return out
