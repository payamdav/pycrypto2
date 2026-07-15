"""rising_from_bowl: detect a price rising out of a bowl/U-shape dip.

Backward horizontal-ray scan locates the symmetrical bowl, a backward peak
climb locates the true crest of the left wall, and a quadratic fit extracts
shape features (curvature, goodness-of-fit, theoretical bottom).
"""

import numpy as np
import numba as nb

N_COLS = 14

# Column order of a rising_from_bowl_scan() row / rising_from_bowl() dict-minus-"detected".
SCAN_COLUMNS = (
    "left_rim_idx", "right_rim_idx", "bottom_idx", "bowl_width",
    "bowl_depth_bps", "bottom_position_ratio", "left_wall_peak_idx",
    "left_wall_peak_price", "recovery_ratio", "fit_coef_a", "fit_coef_b",
    "fit_coef_c", "r_squared", "theoretical_bottom_idx",
)


@nb.njit(cache=True)
def _detect_at(
    prices, t, min_bowl_width, max_bowl_width, min_bowl_depth_bps,
    bottom_position_limit, peak_drawdown_limit_bps, max_peak_search_width,
):
    """Run the bowl detector anchored at index t. Returns (found, row[N_COLS])."""
    row = np.zeros(N_COLS, dtype=np.float64)

    if t < max_bowl_width:
        return False, row
    p_t = prices[t]

    # Horizontal ray scan: nearest j < t (within max_bowl_width) with prices[j] >= p_t.
    i = -1
    lo = t - max_bowl_width
    j = t - 1
    while j >= lo:
        if prices[j] >= p_t:
            i = j
            break
        j -= 1
    if i == -1:
        return False, row

    k = t - i
    if k < min_bowl_width:
        return False, row

    # Bottom of the window and its relative position.
    p_min = prices[i]
    t_min = i
    for m in range(i + 1, t + 1):
        if prices[m] < p_min:
            p_min = prices[m]
            t_min = m

    r = (t_min - i) / k
    half = bottom_position_limit / 2.0
    if r < 0.5 - half or r > 0.5 + half:
        return False, row

    depth_bps = (p_t - p_min) / p_t * 10000.0
    if depth_bps < min_bowl_depth_bps:
        return False, row

    # Left-wall peak climb: running peak with a trailing-drawdown stop.
    peak = prices[i]
    peak_idx = i
    climb_lo = t - max_peak_search_width
    if climb_lo < 0:
        climb_lo = 0
    j = i - 1
    while j >= climb_lo:
        if prices[j] > peak:
            peak = prices[j]
            peak_idx = j
        else:
            drawdown_bps = (peak - prices[j]) / peak * 10000.0
            if drawdown_bps > peak_drawdown_limit_bps:
                break
        j -= 1

    # Quadratic fit y = a*x^2 + b*x + c over x=0..k via closed-form normal equations
    # (np.polyfit is not @njit-compatible; power sums of x are known in closed form).
    n_pts = k + 1
    sum_x = 0.0
    sum_x2 = 0.0
    sum_x3 = 0.0
    sum_x4 = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_x2y = 0.0
    for x in range(n_pts):
        y = prices[i + x]
        xf = float(x)
        x2 = xf * xf
        sum_x += xf
        sum_x2 += x2
        sum_x3 += x2 * xf
        sum_x4 += x2 * x2
        sum_y += y
        sum_xy += xf * y
        sum_x2y += x2 * y

    a11, a12, a13 = sum_x4, sum_x3, sum_x2
    a21, a22, a23 = sum_x3, sum_x2, sum_x
    a31, a32, a33 = sum_x2, sum_x, float(n_pts)
    b1, b2, b3 = sum_x2y, sum_xy, sum_y

    det = (a11 * (a22 * a33 - a23 * a32)
           - a12 * (a21 * a33 - a23 * a31)
           + a13 * (a21 * a32 - a22 * a31))
    if det == 0.0:
        return False, row

    det_a = (b1 * (a22 * a33 - a23 * a32)
             - a12 * (b2 * a33 - a23 * b3)
             + a13 * (b2 * a32 - a22 * b3))
    det_b = (a11 * (b2 * a33 - a23 * b3)
             - b1 * (a21 * a33 - a23 * a31)
             + a13 * (a21 * b3 - b2 * a31))
    det_c = (a11 * (a22 * b3 - b2 * a32)
             - a12 * (a21 * b3 - b2 * a31)
             + b1 * (a21 * a32 - a22 * a31))

    a = det_a / det
    b = det_b / det
    c = det_c / det
    if a <= 0.0:
        return False, row

    mean_y = sum_y / n_pts
    ss_tot = 0.0
    ss_res = 0.0
    for x in range(n_pts):
        y = prices[i + x]
        xf = float(x)
        y_hat = a * xf * xf + b * xf + c
        diff_res = y - y_hat
        diff_tot = y - mean_y
        ss_res += diff_res * diff_res
        ss_tot += diff_tot * diff_tot
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    theoretical_bottom_idx = i - b / (2.0 * a)

    denom = peak - p_min
    recovery_ratio = (p_t - p_min) / denom if denom > 0.0 else 0.0

    row[0] = float(i)
    row[1] = float(t)
    row[2] = float(t_min)
    row[3] = float(k)
    row[4] = depth_bps
    row[5] = r
    row[6] = float(peak_idx)
    row[7] = peak
    row[8] = recovery_ratio
    row[9] = a
    row[10] = b
    row[11] = c
    row[12] = r_squared
    row[13] = theoretical_bottom_idx
    return True, row


@nb.njit(cache=True)
def _scan_core(
    prices, start_idx, end_idx, min_bowl_width, max_bowl_width, min_bowl_depth_bps,
    bottom_position_limit, peak_drawdown_limit_bps, max_peak_search_width,
):
    """Run _detect_at for every anchor in [start_idx, end_idx); return a trimmed copy."""
    out = np.zeros((end_idx - start_idx, N_COLS), dtype=np.float64)
    count = 0
    for t in range(start_idx, end_idx):
        found, row = _detect_at(
            prices, t, min_bowl_width, max_bowl_width, min_bowl_depth_bps,
            bottom_position_limit, peak_drawdown_limit_bps, max_peak_search_width,
        )
        if found:
            out[count] = row
            count += 1
    return out[:count].copy()


def _validate_params(min_bowl_width, max_bowl_width, bottom_position_limit, max_peak_search_width):
    if min_bowl_width < 2:
        raise ValueError(f"min_bowl_width must be >= 2 (quadratic fit needs >= 3 points), got {min_bowl_width}")
    if max_bowl_width < min_bowl_width:
        raise ValueError(f"max_bowl_width ({max_bowl_width}) must be >= min_bowl_width ({min_bowl_width})")
    if not (0.0 <= bottom_position_limit <= 1.0):
        raise ValueError(f"bottom_position_limit must be in [0.0, 1.0], got {bottom_position_limit}")
    if max_peak_search_width < 1:
        raise ValueError(f"max_peak_search_width must be >= 1, got {max_peak_search_width}")


def _as_prices(prices):
    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"prices must be a 1D array, got ndim={arr.ndim}")
    return np.ascontiguousarray(arr)


def _row_to_dict(row):
    return {
        "detected": True,
        "left_rim_idx": int(row[0]),
        "right_rim_idx": int(row[1]),
        "bottom_idx": int(row[2]),
        "bowl_width": int(row[3]),
        "bowl_depth_bps": float(row[4]),
        "bottom_position_ratio": float(row[5]),
        "left_wall_peak_idx": int(row[6]),
        "left_wall_peak_price": float(row[7]),
        "recovery_ratio": float(row[8]),
        "fit_coef_a": float(row[9]),
        "fit_coef_b": float(row[10]),
        "fit_coef_c": float(row[11]),
        "r_squared": float(row[12]),
        "theoretical_bottom_idx": float(row[13]),
    }


def rising_from_bowl(
    prices: np.ndarray,
    min_bowl_width: int = 10,
    max_bowl_width: int = 120,
    min_bowl_depth_bps: float = 20.0,
    bottom_position_limit: float = 0.8,
    peak_drawdown_limit_bps: float = 15.0,
    max_peak_search_width: int = 240,
) -> dict | None:
    """Detect whether the last point of ``prices`` is rising out of a bowl.

    Parameters
    ----------
    prices : np.ndarray
        1D price series (e.g. vwap), oldest -> newest. Only ``prices[-1]`` is
        evaluated as the candidate anchor.
    min_bowl_width, max_bowl_width : int
        Bowl width bounds in array steps (minutes for 1m candles). The
        horizontal ray scan looks back at most ``max_bowl_width`` steps.
    min_bowl_depth_bps : float
        Minimum bowl depth vs. the current price, in basis points (1 bps = 0.01%).
    bottom_position_limit : float
        Allowed band for the bottom's relative position, centered at 0.5;
        ``1.0`` allows anywhere, ``0.0`` allows nowhere.
    peak_drawdown_limit_bps : float
        Trailing-stop drawdown (bps) that ends the left-wall peak climb.
    max_peak_search_width : int
        Maximum backward distance (steps from the anchor) the peak climb may
        travel past the left rim.

    Returns
    -------
    dict | None
        None if no bowl is detected. Otherwise a dict with keys ``detected``,
        ``left_rim_idx``, ``right_rim_idx``, ``bottom_idx``, ``bowl_width``,
        ``bowl_depth_bps``, ``bottom_position_ratio``, ``left_wall_peak_idx``,
        ``left_wall_peak_price``, ``recovery_ratio``, ``fit_coef_a``,
        ``fit_coef_b``, ``fit_coef_c``, ``r_squared``, ``theoretical_bottom_idx``
        (see ``SCAN_COLUMNS`` for the equivalent array-column order).

    Raises
    ------
    ValueError
        If ``prices`` is not 1D, or a parameter is out of range.
    """
    _validate_params(min_bowl_width, max_bowl_width, bottom_position_limit, max_peak_search_width)
    arr = _as_prices(prices)
    t = arr.shape[0] - 1
    if t < 0:
        return None
    found, row = _detect_at(
        arr, t, min_bowl_width, max_bowl_width, min_bowl_depth_bps,
        bottom_position_limit, peak_drawdown_limit_bps, max_peak_search_width,
    )
    return _row_to_dict(row) if found else None


def rising_from_bowl_scan(
    prices: np.ndarray,
    start_idx: int = 0,
    end_idx: int | None = None,
    min_bowl_width: int = 10,
    max_bowl_width: int = 120,
    min_bowl_depth_bps: float = 20.0,
    bottom_position_limit: float = 0.8,
    peak_drawdown_limit_bps: float = 15.0,
    max_peak_search_width: int = 240,
) -> np.ndarray:
    """Run rising_from_bowl at every anchor in ``range(start_idx, end_idx)``.

    Parameters
    ----------
    prices : np.ndarray
        1D price series, oldest -> newest.
    start_idx, end_idx : int
        Anchor range to scan; ``end_idx=None`` means ``len(prices)``.
    (remaining parameters : same as ``rising_from_bowl``)

    Returns
    -------
    np.ndarray
        ``float64`` array, shape ``(m, 14)``, one row per detected anchor
        (rejected anchors produce no row), ascending by anchor index. Column
        order matches ``SCAN_COLUMNS`` (identical to the ``rising_from_bowl``
        dict, minus ``detected``). No deduplication: a bowl commonly
        re-triggers at consecutive anchors while price keeps rising — group
        rows by column 6 (``left_wall_peak_idx``) to collapse re-detections
        into distinct bowls.

    Raises
    ------
    ValueError
        If ``prices`` is not 1D, a parameter is out of range, or
        ``start_idx``/``end_idx`` are out of bounds.
    """
    _validate_params(min_bowl_width, max_bowl_width, bottom_position_limit, max_peak_search_width)
    arr = _as_prices(prices)
    n = arr.shape[0]
    if end_idx is None:
        end_idx = n
    if start_idx < 0 or end_idx > n or start_idx > end_idx:
        raise ValueError(f"start_idx/end_idx out of range: [{start_idx}, {end_idx}) for length {n}")
    return _scan_core(
        arr, start_idx, end_idx, min_bowl_width, max_bowl_width, min_bowl_depth_bps,
        bottom_position_limit, peak_drawdown_limit_bps, max_peak_search_width,
    )
