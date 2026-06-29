import numba as nb
import numpy as np


@nb.njit(inline='always')
def rolling_median_iqr(array, window=60):
    """Return (median, IQR) for each index over a left look-back window.

    Output shape (n, 2), dtype float64. out[i, 0]=median, out[i, 1]=IQR.
    Window for index i: array[max(0, i-window+1) : i+1], length m=min(i+1, window).
    Partial early windows are NOT padded with 0.0 — every index gets a real value.
    For m==1: median=array[i], IQR=0.0. Empty input returns shape (0, 2).
    Quartile convention: Q1=sorted[m//4], Q3=sorted[3*m//4] (matches rolling_robust_z_score).
    """
    n = len(array)
    out = np.empty((n, 2), dtype=np.float64)
    if n == 0:
        return out

    buf = np.empty(window, dtype=np.float64)

    # Growing phase: maintain sorted buf[0:m] as m grows from 1 to min(window, n)
    for i in range(min(window, n)):
        m = i + 1
        buf[i] = array[i]
        j = i
        while j > 0 and buf[j] < buf[j - 1]:
            tmp = buf[j]; buf[j] = buf[j - 1]; buf[j - 1] = tmp
            j -= 1
        if m % 2 == 1:
            median = buf[m // 2]
        else:
            median = (buf[m // 2 - 1] + buf[m // 2]) / 2.0
        out[i, 0] = median
        out[i, 1] = buf[3 * m // 4] - buf[m // 4]

    # Sliding phase: buf is full at size window; remove outgoing, insert incoming
    for i in range(window, n):
        old_val = array[i - window]
        new_val = array[i]

        k = 0
        while k < window and buf[k] != old_val:
            k += 1

        buf[k] = new_val
        while k > 0 and buf[k] < buf[k - 1]:
            tmp = buf[k]; buf[k] = buf[k - 1]; buf[k - 1] = tmp
            k -= 1
        while k < window - 1 and buf[k] > buf[k + 1]:
            tmp = buf[k]; buf[k] = buf[k + 1]; buf[k + 1] = tmp
            k += 1

        if window % 2 == 1:
            median = buf[window // 2]
        else:
            median = (buf[window // 2 - 1] + buf[window // 2]) / 2.0
        out[i, 0] = median
        out[i, 1] = buf[3 * window // 4] - buf[window // 4]

    return out


@nb.njit(inline='always')
def rolling_robust_z_score(array, window=60):
    n = len(array)
    out = np.zeros(n, dtype=np.float64)

    if n < window:
        return out

    # Allocate sorted buffer once
    buf = np.empty(window, dtype=np.float64)

    # --- First valid window: copy and insertion-sort ---
    for k in range(window):
        buf[k] = array[k]

    for k in range(1, window):
        key = buf[k]
        j = k - 1
        while j >= 0 and buf[j] > key:
            buf[j + 1] = buf[j]
            j -= 1
        buf[j + 1] = key

    # Compute and store result for i == window - 1
    i = window - 1
    if window % 2 == 1:
        median = buf[window // 2]
    else:
        median = (buf[window // 2 - 1] + buf[window // 2]) / 2.0
    q1 = buf[window // 4]
    q3 = buf[3 * window // 4]
    iqr = q3 - q1
    if iqr == 0.0:
        out[i] = 0.0
    else:
        out[i] = (array[i] - median) / iqr

    # --- Subsequent windows: incremental update ---
    for i in range(window, n):
        old_val = array[i - window]
        new_val = array[i]

        # Find old_val in sorted buffer (linear scan)
        k = 0
        while k < window and buf[k] != old_val:
            k += 1

        # Overwrite with new value and bubble into sorted position
        buf[k] = new_val
        while k > 0 and buf[k] < buf[k - 1]:
            tmp = buf[k]
            buf[k] = buf[k - 1]
            buf[k - 1] = tmp
            k -= 1
        while k < window - 1 and buf[k] > buf[k + 1]:
            tmp = buf[k]
            buf[k] = buf[k + 1]
            buf[k + 1] = tmp
            k += 1

        if window % 2 == 1:
            median = buf[window // 2]
        else:
            median = (buf[window // 2 - 1] + buf[window // 2]) / 2.0
        q1 = buf[window // 4]
        q3 = buf[3 * window // 4]
        iqr = q3 - q1
        if iqr == 0.0:
            out[i] = 0.0
        else:
            out[i] = (array[i] - median) / iqr

    return out
