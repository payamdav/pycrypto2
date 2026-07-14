import numba as nb
import numpy as np


@nb.njit(inline='always')
def linreg_slope(array, window=60):
    """Rolling OLS slope of array vs x=0..window-1 (look-back window, current item = last x).
    O(n) via incremental running sums S1=sum(y), Sxy=sum(k*y), k=0..window-1 local index.
    output[i] = slope for i >= window-1; indices < window-1 backfilled with output[window-1].
    n < window returns all zeros.
    """
    n = len(array)
    out = np.zeros(n, dtype=np.float64)
    w = window
    if n < w:
        return out

    sx = w * (w - 1) / 2.0
    sxx = (w - 1) * w * (2 * w - 1) / 6.0
    denom = w * sxx - sx * sx

    s1 = 0.0
    sxy = 0.0
    for k in range(w):
        y = array[k]
        s1 += y
        sxy += k * y

    out[w - 1] = (w * sxy - sx * s1) / denom
    for i in range(w, n):
        y_out = array[i - w]
        y_in = array[i]
        sxy = sxy - s1 + y_out + (w - 1) * y_in
        s1 = s1 - y_out + y_in
        out[i] = (w * sxy - sx * s1) / denom

    first = out[w - 1]
    for i in range(w - 1):
        out[i] = first
    return out
