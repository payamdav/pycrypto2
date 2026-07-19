"""Volume-weighted KDE over raw prices around the current price.
"""

import numpy as np

from packages.kde_tools.kernels import make_kernel
from packages.kde_tools.kde import convolve_same
from packages.volume_profile.histogram import weighted_histogram


def compute_kde(
    prices: np.ndarray,
    volumes: np.ndarray,
    bins: int = 200,
    bps_range: float = 100.0,
    kernel_type: str = "Triangular",
    bandwidth: int = 5,
) -> dict:
    """Volume-weighted KDE over raw prices, range = current price +/- bps_range bps.

    Calls :func:`weighted_histogram` (range derivation + validation), builds a
    normalized kernel via ``make_kernel``, and convolves with ``convolve_same``
    (both reused from ``kde_tools`` unchanged).

    Parameters
    ----------
    prices, volumes : np.ndarray
        See :func:`weighted_histogram`.
    bins : int
        Number of histogram bins.
    bps_range : float
        Half-range in basis points around the current price.
    kernel_type : {"Triangular", "Epanechnikov", "Uniform"}
        Smoothing kernel shape.
    bandwidth : int
        Kernel half-width; the kernel has length ``2 * bandwidth + 1``.

    Returns
    -------
    dict
        ``{"kde", "counts", "bin_centers", "bin_width", "kernel",
        "current_price", "range_min", "range_max", "n_excluded"}``.

    Raises
    ------
    ValueError
        See :func:`weighted_histogram`; ``kernel_type`` errors propagate from
        ``make_kernel``.
    """
    hist = weighted_histogram(prices, volumes, bins, bps_range)

    kernel_arr = make_kernel(kernel_type, bandwidth)
    kde = convolve_same(hist["counts"], kernel_arr)

    return {
        "kde": kde,
        "counts": hist["counts"],
        "bin_centers": hist["bin_centers"],
        "bin_width": hist["bin_width"],
        "kernel": kernel_arr,
        "current_price": hist["current_price"],
        "range_min": hist["range_min"],
        "range_max": hist["range_max"],
        "n_excluded": hist["n_excluded"],
    }
