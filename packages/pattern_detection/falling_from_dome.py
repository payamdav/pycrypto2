"""falling_from_dome: detect a price falling out of a dome/inverted-U top.

Vertical mirror of rising_from_bowl. Backward horizontal-ray scan locates the
symmetrical dome, a backward trough climb locates the true low of the left
wall, and a quadratic fit extracts shape features (curvature, goodness-of-fit,
theoretical top).
"""

import numpy as np
import numba as nb

N_COLS = 14

# Column order of a falling_from_dome_scan() row / falling_from_dome() dict-minus-"detected".
DOME_SCAN_COLUMNS = (
    "left_rim_idx", "right_rim_idx", "top_idx", "dome_width",
    "dome_height_bps", "top_position_ratio", "left_wall_trough_idx",
    "left_wall_trough_price", "decline_ratio", "fit_coef_a", "fit_coef_b",
    "fit_coef_c", "r_squared", "theoretical_top_idx",
)


@nb.njit(cache=True)
def _detect_at(
    prices, t, min_dome_width, max_dome_width, min_dome_height_bps,
    top_position_limit, trough_rally_limit_bps, max_trough_search_width,
):
    """Run the dome detector anchored at index t. Returns (found, row[N_COLS])."""
    row = np.zeros(N_COLS, dtype=np.float64)

    if t < max_dome_width:
        return False, row
    p_t = prices[t]

    # Horizontal ray scan: nearest j < t (within max_dome_width) with prices[j] <= p_t.
    i = -1
    lo = t - max_dome_width
    j = t - 1
    while j >= lo:
        if prices[j] <= p_t:
            i = j
            break
        j -= 1
    if i == -1:
        return False, row

    k = t - i
    if k < min_dome_width:
        return False, row

    # Top of the window and its relative position.
    p_max = prices[i]
    t_max = i
    for m in range(i + 1, t + 1):
        if prices[m] > p_max:
            p_max = prices[m]
            t_max = m

    r = (t_max - i) / k
    half = top_position_limit / 2.0
    if r < 0.5 - half or r > 0.5 + half:
        return False, row

    height_bps = (p_max - p_t) / p_t * 10000.0
    if height_bps < min_dome_height_bps:
        return False, row

    # Left-wall trough climb: running trough with a trailing-rally stop.
    trough = prices[i]
    trough_idx = i
    climb_lo = t - max_trough_search_width
    if climb_lo < 0:
        climb_lo = 0
    j = i - 1
    while j >= climb_lo:
        if prices[j] < trough:
            trough = prices[j]
            trough_idx = j
        else:
            rally_bps = (prices[j] - trough) / trough * 10000.0
            if rally_bps > trough_rally_limit_bps:
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
    if a >= 0.0:
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

    theoretical_top_idx = i - b / (2.0 * a)

    denom = p_max - trough
    decline_ratio = (p_max - p_t) / denom if denom > 0.0 else 0.0

    row[0] = float(i)
    row[1] = float(t)
    row[2] = float(t_max)
    row[3] = float(k)
    row[4] = height_bps
    row[5] = r
    row[6] = float(trough_idx)
    row[7] = trough
    row[8] = decline_ratio
    row[9] = a
    row[10] = b
    row[11] = c
    row[12] = r_squared
    row[13] = theoretical_top_idx
    return True, row


@nb.njit(cache=True)
def _scan_core(
    prices, start_idx, end_idx, min_dome_width, max_dome_width, min_dome_height_bps,
    top_position_limit, trough_rally_limit_bps, max_trough_search_width,
):
    """Run _detect_at for every anchor in [start_idx, end_idx); return a trimmed copy."""
    out = np.zeros((end_idx - start_idx, N_COLS), dtype=np.float64)
    count = 0
    for t in range(start_idx, end_idx):
        found, row = _detect_at(
            prices, t, min_dome_width, max_dome_width, min_dome_height_bps,
            top_position_limit, trough_rally_limit_bps, max_trough_search_width,
        )
        if found:
            out[count] = row
            count += 1
    return out[:count].copy()


def _validate_params(min_dome_width, max_dome_width, top_position_limit, max_trough_search_width):
    if min_dome_width < 2:
        raise ValueError(f"min_dome_width must be >= 2 (quadratic fit needs >= 3 points), got {min_dome_width}")
    if max_dome_width < min_dome_width:
        raise ValueError(f"max_dome_width ({max_dome_width}) must be >= min_dome_width ({min_dome_width})")
    if not (0.0 <= top_position_limit <= 1.0):
        raise ValueError(f"top_position_limit must be in [0.0, 1.0], got {top_position_limit}")
    if max_trough_search_width < 1:
        raise ValueError(f"max_trough_search_width must be >= 1, got {max_trough_search_width}")


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
        "top_idx": int(row[2]),
        "dome_width": int(row[3]),
        "dome_height_bps": float(row[4]),
        "top_position_ratio": float(row[5]),
        "left_wall_trough_idx": int(row[6]),
        "left_wall_trough_price": float(row[7]),
        "decline_ratio": float(row[8]),
        "fit_coef_a": float(row[9]),
        "fit_coef_b": float(row[10]),
        "fit_coef_c": float(row[11]),
        "r_squared": float(row[12]),
        "theoretical_top_idx": float(row[13]),
    }


def falling_from_dome(
    prices: np.ndarray,
    min_dome_width: int = 10,
    max_dome_width: int = 120,
    min_dome_height_bps: float = 20.0,
    top_position_limit: float = 0.8,
    trough_rally_limit_bps: float = 15.0,
    max_trough_search_width: int = 240,
) -> dict | None:
    """Detect whether the last point of ``prices`` is falling out of a dome.

    Parameters
    ----------
    prices : np.ndarray
        1D price series (e.g. vwap), oldest -> newest. Only ``prices[-1]`` is
        evaluated as the candidate anchor.
    min_dome_width, max_dome_width : int
        Dome width bounds in array steps (minutes for 1m candles). The
        horizontal ray scan looks back at most ``max_dome_width`` steps.
    min_dome_height_bps : float
        Minimum dome height vs. the current price, in basis points (1 bps = 0.01%).
    top_position_limit : float
        Allowed band for the top's relative position, centered at 0.5;
        ``1.0`` allows anywhere, ``0.0`` allows nowhere.
    trough_rally_limit_bps : float
        Trailing-stop rally (bps) that ends the left-wall trough climb.
    max_trough_search_width : int
        Maximum backward distance (steps from the anchor) the trough climb may
        travel past the left rim.

    Returns
    -------
    dict | None
        None if no dome is detected. Otherwise a dict with keys ``detected``,
        ``left_rim_idx``, ``right_rim_idx``, ``top_idx``, ``dome_width``,
        ``dome_height_bps``, ``top_position_ratio``, ``left_wall_trough_idx``,
        ``left_wall_trough_price``, ``decline_ratio``, ``fit_coef_a``,
        ``fit_coef_b``, ``fit_coef_c``, ``r_squared``, ``theoretical_top_idx``
        (see ``DOME_SCAN_COLUMNS`` for the equivalent array-column order).

    Raises
    ------
    ValueError
        If ``prices`` is not 1D, or a parameter is out of range.
    """
    _validate_params(min_dome_width, max_dome_width, top_position_limit, max_trough_search_width)
    arr = _as_prices(prices)
    t = arr.shape[0] - 1
    if t < 0:
        return None
    found, row = _detect_at(
        arr, t, min_dome_width, max_dome_width, min_dome_height_bps,
        top_position_limit, trough_rally_limit_bps, max_trough_search_width,
    )
    return _row_to_dict(row) if found else None


def falling_from_dome_scan(
    prices: np.ndarray,
    start_idx: int = 0,
    end_idx: int | None = None,
    min_dome_width: int = 10,
    max_dome_width: int = 120,
    min_dome_height_bps: float = 20.0,
    top_position_limit: float = 0.8,
    trough_rally_limit_bps: float = 15.0,
    max_trough_search_width: int = 240,
) -> np.ndarray:
    """Run falling_from_dome at every anchor in ``range(start_idx, end_idx)``.

    Parameters
    ----------
    prices : np.ndarray
        1D price series, oldest -> newest.
    start_idx, end_idx : int
        Anchor range to scan; ``end_idx=None`` means ``len(prices)``.
    (remaining parameters : same as ``falling_from_dome``)

    Returns
    -------
    np.ndarray
        ``float64`` array, shape ``(m, 14)``, one row per detected anchor
        (rejected anchors produce no row), ascending by anchor index. Column
        order matches ``DOME_SCAN_COLUMNS`` (identical to the
        ``falling_from_dome`` dict, minus ``detected``). No deduplication: a
        dome commonly re-triggers at consecutive anchors while price keeps
        falling — group rows by ``top_idx`` (col 2) or
        ``left_wall_trough_idx`` (col 6) to collapse re-detections into
        distinct domes.

    Raises
    ------
    ValueError
        If ``prices`` is not 1D, a parameter is out of range, or
        ``start_idx``/``end_idx`` are out of bounds.
    """
    _validate_params(min_dome_width, max_dome_width, top_position_limit, max_trough_search_width)
    arr = _as_prices(prices)
    n = arr.shape[0]
    if end_idx is None:
        end_idx = n
    if start_idx < 0 or end_idx > n or start_idx > end_idx:
        raise ValueError(f"start_idx/end_idx out of range: [{start_idx}, {end_idx}) for length {n}")
    return _scan_core(
        arr, start_idx, end_idx, min_dome_width, max_dome_width, min_dome_height_bps,
        top_position_limit, trough_rally_limit_bps, max_trough_search_width,
    )
