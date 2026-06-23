"""KDE computation: border filter + weighted histogram + kernel convolution.
"""

import numpy as np
import numba as nb

from packages.kde_tools.kernels import make_kernel
from packages.kde_tools.histogram import weighted_histogram


@nb.njit
def convolve_same(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Reproduce ``np.convolve(signal, kernel, mode="same")`` exactly.

    Computes the full convolution (length ``N + M - 1``) with explicit loops
    and returns the centered slice ``full[(M - 1) // 2 : (M - 1) // 2 + N]``,
    where ``N = len(signal)`` and ``M = len(kernel)``.

    Returns
    -------
    np.ndarray
        Newly allocated ``np.float64`` array of length ``N`` (``len(signal)``).
    """
    n = signal.shape[0]
    m = kernel.shape[0]
    full_len = n + m - 1

    full = np.zeros(full_len, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    offset = (m - 1) // 2

    for i in range(n):
        s = signal[i]
        for j in range(m):
            full[i + j] += s * kernel[j]

    for i in range(n):
        out[i] = full[offset + i]

    return out


def compute_kde(
    scaled_prices: np.ndarray,
    weights: np.ndarray,
    bins: int = 200,
    kernel_type: str = "Triangular",
    bandwidth: int = 5,
    range_min: float = -1.0,
    range_max: float = 1.0,
    ignore_borders: bool = True,
):
    """Compute a volume-weighted KDE over normalized look-back prices.

    Parameters
    ----------
    scaled_prices : np.ndarray
        Normalized prices in ``[range_min, range_max]``.
    weights : np.ndarray
        Per-entry weights (e.g. normalized volumes); same length as
        ``scaled_prices``.
    bins : int
        Number of histogram bins over the fixed range.
    kernel_type : {"Triangular", "Epanechnikov", "Uniform"}
        Smoothing kernel shape.
    bandwidth : int
        Kernel half-width; the kernel has length ``2 * bandwidth + 1``.
    range_min, range_max : float
        Fixed histogram range, defaulting to ``(-1.0, 1.0)``.
    ignore_borders : bool
        When True, drop entries sitting exactly at the borders using strict
        inequalities (``range_min < v < range_max``); ``n_excluded`` counts the
        dropped entries. When False, use all entries and ``n_excluded = 0``.

    Returns
    -------
    dict
        ``{"kde", "counts", "bin_centers", "bin_width", "kernel", "n_excluded"}``.
    """
    n_excluded = int(np.sum((scaled_prices <= range_min) | (scaled_prices >= range_max))) if ignore_borders else 0

    counts = weighted_histogram(scaled_prices, weights, bins, range_min, range_max, ignore_borders)

    bin_width = (range_max - range_min) / bins
    bin_centers = range_min + (np.arange(bins) + 0.5) * bin_width

    kernel_arr = make_kernel(kernel_type, bandwidth)
    kde = convolve_same(counts, kernel_arr)

    return {
        "kde": kde,
        "counts": counts,
        "bin_centers": bin_centers,
        "bin_width": bin_width,
        "kernel": kernel_arr,
        "n_excluded": n_excluded,
    }
