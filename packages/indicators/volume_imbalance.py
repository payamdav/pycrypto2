import numba as nb
import numpy as np


@nb.njit(inline='always')
def volume_imbalance(vb, vs, window=60):
    """Rolling volume imbalance: (sum(vb)-sum(vs)) / (sum(vb)+sum(vs)) over window.
    0.0 where the denominator is 0. O(n) via incremental running sums.
    output[i] valid for i >= window-1; indices < window-1 backfilled with output[window-1].
    n < window returns all zeros.
    """
    n = len(vb)
    out = np.zeros(n, dtype=np.float64)
    w = window
    if n < w:
        return out

    b_sum = 0.0
    s_sum = 0.0
    for k in range(w):
        b_sum += vb[k]
        s_sum += vs[k]

    denom = b_sum + s_sum
    out[w - 1] = (b_sum - s_sum) / denom if denom != 0.0 else 0.0
    for i in range(w, n):
        b_sum += vb[i] - vb[i - w]
        s_sum += vs[i] - vs[i - w]
        denom = b_sum + s_sum
        out[i] = (b_sum - s_sum) / denom if denom != 0.0 else 0.0

    first = out[w - 1]
    for i in range(w - 1):
        out[i] = first
    return out
