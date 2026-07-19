"""Volume-weighted histogram over a raw-price range around the current price.
"""

import numpy as np

from packages.kde_tools.histogram import weighted_histogram as _weighted_histogram_core


def weighted_histogram(
    prices: np.ndarray,
    volumes: np.ndarray,
    bins: int = 200,
    bps_range: float = 100.0,
) -> dict:
    """Volume-weighted histogram of raw prices around the current price.

    ``current_price = prices[-1]``; ``range_min = current_price * (1 - bps_range / 1e4)``,
    ``range_max = current_price * (1 + bps_range / 1e4)`` (``bps_range`` = half-range in
    basis points; 100 -> +/-1%). Binning delegates to the jitted
    ``packages.kde_tools.histogram.weighted_histogram`` core with
    ``ignore_borders=False`` (prices outside the range are skipped; the borders
    themselves are included; ``v == range_max`` clamps to the last bin).

    Parameters
    ----------
    prices : np.ndarray
        1D, oldest -> newest; ``prices[-1]`` is the current price.
    volumes : np.ndarray
        Per-entry weights, same length as ``prices``.
    bins : int
        Number of histogram bins over the derived range.
    bps_range : float
        Half-range in basis points around the current price.

    Returns
    -------
    dict
        ``{"counts", "bin_centers", "bin_width", "current_price", "range_min",
        "range_max", "n_excluded"}`` — ``counts``/``bin_centers`` are
        ``np.float64`` arrays of shape ``(bins,)``, in raw price units.

    Raises
    ------
    ValueError
        If ``prices`` is not 1D or is empty, ``volumes`` doesn't match its
        shape, ``bins < 1``, ``bps_range <= 0``, or ``current_price`` (=
        ``prices[-1]``) is not finite and > 0.
    """
    prices = np.ascontiguousarray(np.asarray(prices, dtype=np.float64))
    volumes = np.ascontiguousarray(np.asarray(volumes, dtype=np.float64))

    if prices.ndim != 1 or prices.size == 0:
        raise ValueError("prices must be a non-empty 1D array")
    if volumes.shape != prices.shape:
        raise ValueError("volumes must have the same shape as prices")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if bps_range <= 0:
        raise ValueError("bps_range must be > 0")

    current_price = float(prices[-1])
    if not np.isfinite(current_price) or current_price <= 0:
        raise ValueError("current_price (prices[-1]) must be finite and > 0")

    range_min = current_price * (1.0 - bps_range / 1e4)
    range_max = current_price * (1.0 + bps_range / 1e4)

    counts = _weighted_histogram_core(prices, volumes, bins, range_min, range_max, False)

    bin_width = (range_max - range_min) / bins
    bin_centers = range_min + (np.arange(bins) + 0.5) * bin_width
    n_excluded = int(np.sum((prices < range_min) | (prices > range_max)))

    return {
        "counts": counts,
        "bin_centers": bin_centers,
        "bin_width": bin_width,
        "current_price": current_price,
        "range_min": range_min,
        "range_max": range_max,
        "n_excluded": n_excluded,
    }
