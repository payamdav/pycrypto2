"""KDE peak finding (scipy-based).
"""

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths


def top_kde_peaks(
    kde_series: np.ndarray,
    prices: np.ndarray,
    distance: float,
    n: int = 3,
    top_identifier: str = "prominence",
):
    """Return the ``n`` top peaks of ``kde_series``.

    Parameters
    ----------
    kde_series : np.ndarray
        KDE values to find peaks in.
    prices : np.ndarray
        Bin-center prices aligned with ``kde_series``.
    distance : float
        Minimum horizontal distance between peaks (``find_peaks`` ``distance``).
    n : int
        Maximum number of peaks to return (top-n by descending score).
    top_identifier : str
        Ranking key: ``"prominence"`` (default) ranks by peak prominence,
        ``"height"`` ranks by KDE value at the peak.

    Returns
    -------
    (np.ndarray, np.ndarray)
        ``(peak_prices, peak_proms)``; both empty when no peaks are found.
        Prominences are always returned (in the selected order) regardless of
        ``top_identifier``.
    """
    peaks, _ = find_peaks(kde_series, distance=distance)
    if len(peaks) == 0:
        return np.array([]), np.array([])
    proms = peak_prominences(kde_series, peaks)[0]
    if top_identifier == "prominence":
        score = proms
    elif top_identifier == "height":
        score = kde_series[peaks]
    else:
        raise ValueError(
            f"top_identifier must be 'prominence' or 'height', got {top_identifier!r}"
        )
    order = np.argsort(score)[::-1][:n]
    return prices[peaks[order]], proms[order]


def kde_peaks_above_below(
    kde: np.ndarray,
    bin_centers: np.ndarray,
    distance: float = 5,
    n: int = 3,
    split_at: float = 0.0,
    top_identifier: str = "prominence",
):
    """Top-``n`` KDE peaks above and below the current price.

    ``split_at`` is the current price in normalized space (defaults to ``0.0``
    because ``price_l`` normalizes to ``0.0``). "Above" = ``bin_centers >=
    split_at``, "below" = ``bin_centers < split_at`` (matching the notebook's
    ``pos_mask`` / ``neg_mask``). ``top_identifier`` selects the ranking key
    (``"prominence"`` or ``"height"``) and is forwarded to ``top_kde_peaks``.

    Returns
    -------
    dict
        ``{"above_prices", "above_proms", "below_prices", "below_proms"}``.
    """
    pos_mask = bin_centers >= split_at
    neg_mask = bin_centers < split_at

    pos_kde = kde[pos_mask]
    pos_prices = bin_centers[pos_mask]
    neg_kde = kde[neg_mask]
    neg_prices = bin_centers[neg_mask]

    above_prices, above_proms = top_kde_peaks(
        pos_kde, pos_prices, distance=distance, n=n, top_identifier=top_identifier
    )
    below_prices, below_proms = top_kde_peaks(
        neg_kde, neg_prices, distance=distance, n=n, top_identifier=top_identifier
    )

    return {
        "above_prices": above_prices,
        "above_proms": above_proms,
        "below_prices": below_prices,
        "below_proms": below_proms,
    }


def kde_peak_widths(
    kde_series: np.ndarray,
    peak_indices: np.ndarray,
    rel_height: float = 0.5,
) -> dict:
    """Return prominences and widths at ``rel_height`` for given peaks.

    Parameters
    ----------
    kde_series : np.ndarray
        The KDE array in which *peak_indices* were found.
    peak_indices : np.ndarray
        Integer indices of peaks within *kde_series* (as returned by
        ``scipy.signal.find_peaks``).
    rel_height : float
        Relative height at which to measure the width (``peak_widths``
        ``rel_height``); defaults to ``0.5``.

    Returns
    -------
    dict
        ``{"proms", "widths"}`` — both ``np.ndarray`` of length
        ``len(peak_indices)``, widths in bins.  Empty arrays when
        *peak_indices* is empty.
    """
    if len(peak_indices) == 0:
        empty = np.array([], dtype=np.float64)
        return {"proms": empty, "widths": empty}

    peak_indices = np.asarray(peak_indices, dtype=np.intp)
    proms = peak_prominences(kde_series, peak_indices)[0]
    widths = peak_widths(kde_series, peak_indices, rel_height=rel_height)[0]

    return {"proms": proms, "widths": widths}
