import numba as nb
import numpy as np


@nb.njit
def triple_barrier_simple(prices, upper_barrier_bps=20.0, lower_barrier_bps=20.0,
                           look_ahead=240, next_entry=True):
    """Triple-barrier labeling: per item, simulate a long trade and label the
    exit cause. See agents/packages/traders_indicators/triple_barrier_simple.md."""
    n = len(prices)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        entry_idx = i + 1 if next_entry else i
        if entry_idx >= n:
            out[i] = 0.0
            continue
        entry = prices[entry_idx]
        upper = entry * (1.0 + upper_barrier_bps / 10_000.0)
        lower = entry * (1.0 - lower_barrier_bps / 10_000.0)
        end = min(entry_idx + look_ahead, n - 1)
        label = 0.0
        for j in range(entry_idx + 1, end + 1):
            if prices[j] >= upper:
                label = 1.0
                break
            elif prices[j] <= lower:
                label = -1.0
                break
        out[i] = label
    return out
