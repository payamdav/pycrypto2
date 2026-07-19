"""POC / Value-Area / recursive-POC peak analysis over a raw-price volume profile.
"""

import numpy as np

from packages.kde_tools.peaks import top_kde_peaks
from packages.kde_tools.peaks import kde_peaks_above_below as _kde_peaks_above_below_core

__all__ = ["point_of_control", "kde_peaks_above_below", "recursive_poc", "top_kde_peaks"]


def _value_area(kde: np.ndarray, removed: np.ndarray, poc_idx: int, target: float) -> tuple:
    """Greedy single-bin Value-Area expansion from ``poc_idx``.

    Shared by :func:`point_of_control` (``removed`` all-``False``) and
    :func:`recursive_poc` (``removed`` marks bins consumed by earlier ranks).
    A bin is blocked exactly like an array edge when its index is out of range
    **or** ``removed[idx]`` is ``True``. At each step the larger unblocked
    neighbor is absorbed; a tie expands **above** (higher bin index). Stops
    once ``acc >= target`` or both sides are blocked (under-target VA is
    valid in that case).

    Returns
    -------
    (int, int, float)
        ``(lo, hi, acc)`` — inclusive VA bin bounds and the accumulated volume.
    """
    n = kde.shape[0]
    lo = hi = poc_idx
    acc = float(kde[poc_idx])

    while acc < target:
        can_below = lo > 0 and not removed[lo - 1]
        can_above = hi < n - 1 and not removed[hi + 1]
        if not can_below and not can_above:
            break
        below_val = kde[lo - 1] if can_below else -np.inf
        above_val = kde[hi + 1] if can_above else -np.inf
        if above_val >= below_val:  # tie -> above
            hi += 1
            acc += float(kde[hi])
        else:
            lo -= 1
            acc += float(kde[lo])

    return lo, hi, acc


def point_of_control(
    kde: np.ndarray,
    bin_centers: np.ndarray,
    va_pct: float = 70.0,
):
    """Point of Control (POC) + Value Area of a raw-price KDE volume profile.

    ``poc_idx`` is the argmax bin of ``kde`` (first on ties); the Value Area
    is grown around it by greedy single-bin expansion
    (see :func:`_value_area`) until it holds ``va_pct`` % of the total profile
    volume (or both sides are exhausted).

    Parameters
    ----------
    kde : np.ndarray
        Smoothed volume-profile density, shape ``(bins,)`` (from
        :func:`packages.volume_profile.kde.compute_kde`).
    bin_centers : np.ndarray
        Bin-center prices aligned with ``kde``, shape ``(bins,)``.
    va_pct : float
        Value Area target, percent of total profile volume, in ``(0, 100]``.

    Returns
    -------
    dict | None
        ``{"poc_idx", "poc_price", "poc_volume", "val_idx", "vah_idx", "val",
        "vah", "va_volume", "total_volume"}`` (``*_idx`` are Python ``int``,
        the rest ``float``); ``None`` when ``kde`` is empty or its sum is <= 0
        (empty profile).

    Raises
    ------
    ValueError
        If ``kde``/``bin_centers`` shapes mismatch, or ``va_pct`` is outside
        ``(0, 100]``.
    """
    kde = np.ascontiguousarray(np.asarray(kde, dtype=np.float64))
    bin_centers = np.ascontiguousarray(np.asarray(bin_centers, dtype=np.float64))

    if kde.shape != bin_centers.shape:
        raise ValueError("kde and bin_centers must have the same shape")
    if not (0 < va_pct <= 100):
        raise ValueError("va_pct must be in (0, 100]")

    if kde.size == 0 or kde.sum() <= 0:
        return None

    poc_idx = int(np.argmax(kde))
    poc_volume = float(kde[poc_idx])
    total_volume = float(kde.sum())
    target = va_pct / 100.0 * total_volume

    removed = np.zeros(kde.shape[0], dtype=np.bool_)
    lo, hi, acc = _value_area(kde, removed, poc_idx, target)

    return {
        "poc_idx": poc_idx,
        "poc_price": float(bin_centers[poc_idx]),
        "poc_volume": poc_volume,
        "val_idx": int(lo),
        "vah_idx": int(hi),
        "val": float(bin_centers[lo]),
        "vah": float(bin_centers[hi]),
        "va_volume": float(acc),
        "total_volume": total_volume,
    }


def kde_peaks_above_below(
    kde: np.ndarray,
    bin_centers: np.ndarray,
    current_price: float,
    distance: float = 5,
    n: int = 3,
    top_identifier: str = "prominence",
) -> dict:
    """Top-``n`` KDE peaks above and below ``current_price`` (raw price units).

    Thin wrapper around ``packages.kde_tools.peaks.kde_peaks_above_below`` with
    ``split_at=current_price`` — "above" = ``bin_centers >= current_price``,
    "below" = ``bin_centers < current_price``.

    Parameters
    ----------
    kde, bin_centers : np.ndarray
        See :func:`point_of_control`.
    current_price : float
        Split point, in raw price units (typically ``compute_kde``'s
        ``current_price``).
    distance : float
        Minimum horizontal distance between peaks (``find_peaks`` ``distance``).
    n : int
        Maximum number of peaks to return per side.
    top_identifier : {"prominence", "height"}
        Ranking key, forwarded to ``top_kde_peaks``.

    Returns
    -------
    dict
        ``{"above_prices", "above_proms", "below_prices", "below_proms"}``.
    """
    kde = np.ascontiguousarray(np.asarray(kde, dtype=np.float64))
    bin_centers = np.ascontiguousarray(np.asarray(bin_centers, dtype=np.float64))
    return _kde_peaks_above_below_core(
        kde, bin_centers, distance=distance, n=n, split_at=current_price,
        top_identifier=top_identifier,
    )


def recursive_poc(
    kde: np.ndarray,
    bin_centers: np.ndarray,
    current_price: float,
    va_pct: float = 70.0,
    min_poc_volume_ratio: float = 0.1,
    max_pocs=None,
) -> list:
    """Iterative POC extraction: strongest level, remove its Value Area, repeat.

    Ranked support/resistance levels of a raw-price volume profile. Each
    iteration takes the argmax bin among not-yet-removed bins as the next POC,
    grows its Value Area (:func:`_value_area`, blocked by already-removed
    bins), then removes bins relative to ``current_price``:

    - VA fully above current price (``val > current_price``) -> drop bins from
      ``lo`` to the end (``removed[lo:] = True``).
    - VA fully below current price (``vah < current_price``) -> drop bins from
      the start through ``hi`` (``removed[:hi + 1] = True``).
    - VA straddles current price (otherwise) -> drop only the VA span
      (``removed[lo:hi + 1] = True``).

    The POC bin always lies inside the removed span, so every iteration
    removes >= 1 bin (guaranteed termination in <= ``bins`` iterations).
    Entry 1 equals :func:`point_of_control` (``kde``, ``bin_centers``,
    ``va_pct``) on the shared fields (before any removal). POC volumes are
    non-increasing with rank, so rank order is strength order.

    Parameters
    ----------
    kde, bin_centers : np.ndarray
        See :func:`point_of_control`.
    current_price : float
        Reference price for the removal rule (raw price units).
    va_pct : float
        Value Area target, percent of the profile volume **remaining** at
        each iteration, in ``(0, 100]``.
    min_poc_volume_ratio : float
        Stop once a candidate POC's volume falls below
        ``min_poc_volume_ratio * first_poc_volume`` (``rank > 1`` only), in
        ``[0, 1]``. ``0.0`` disables this stop (runs to full removal).
    max_pocs : int | None
        Optional cap on the number of ranks returned; ``None`` = unlimited.

    Returns
    -------
    list[dict]
        One dict per rank, ``{"rank", "poc_idx", "poc_price", "poc_volume",
        "val_idx", "vah_idx", "val", "vah", "va_volume"}``, strongest first.
        ``[]`` for an empty profile.

    Raises
    ------
    ValueError
        If ``kde``/``bin_centers`` shapes mismatch, ``va_pct`` is outside
        ``(0, 100]``, ``min_poc_volume_ratio`` is outside ``[0, 1]``, or
        ``max_pocs`` is neither ``None`` nor >= 1.
    """
    kde = np.ascontiguousarray(np.asarray(kde, dtype=np.float64))
    bin_centers = np.ascontiguousarray(np.asarray(bin_centers, dtype=np.float64))

    if kde.shape != bin_centers.shape:
        raise ValueError("kde and bin_centers must have the same shape")
    if not (0 < va_pct <= 100):
        raise ValueError("va_pct must be in (0, 100]")
    if not (0 <= min_poc_volume_ratio <= 1):
        raise ValueError("min_poc_volume_ratio must be in [0, 1]")
    if max_pocs is not None and max_pocs < 1:
        raise ValueError("max_pocs must be None or >= 1")

    n_bins = kde.shape[0]
    results = []
    if n_bins == 0 or kde.sum() <= 0:
        return results

    removed = np.zeros(n_bins, dtype=np.bool_)
    first_poc_volume = None

    while True:
        unremoved = ~removed
        if not unremoved.any():
            break

        masked = np.where(unremoved, kde, -np.inf)
        poc_idx = int(np.argmax(masked))
        poc_volume = float(kde[poc_idx])
        if poc_volume <= 0:
            break

        rank = len(results) + 1
        if rank == 1:
            first_poc_volume = poc_volume
        elif poc_volume < min_poc_volume_ratio * first_poc_volume:
            break

        remaining_volume = float(kde[unremoved].sum())
        target = va_pct / 100.0 * remaining_volume
        lo, hi, acc = _value_area(kde, removed, poc_idx, target)
        val = float(bin_centers[lo])
        vah = float(bin_centers[hi])

        results.append({
            "rank": rank,
            "poc_idx": poc_idx,
            "poc_price": float(bin_centers[poc_idx]),
            "poc_volume": poc_volume,
            "val_idx": int(lo),
            "vah_idx": int(hi),
            "val": val,
            "vah": vah,
            "va_volume": float(acc),
        })

        if val > current_price:
            removed[lo:] = True
        elif vah < current_price:
            removed[:hi + 1] = True
        else:
            removed[lo:hi + 1] = True

        if max_pocs is not None and rank >= max_pocs:
            break

    return results
