"""Look-back / look-ahead normalized volume-profile strategy.

Entry point: lookback_lookahead_normalized_vp(...)
Pipeline:    lb_la_n_base → append_cached_metrics → vp_analysis → vp_hvn
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from scipy.signal import find_peaks, peak_prominences


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def lookback_lookahead_normalized_vp(
    asset: str = "btcusdt",
    look_back: int = 1440,
    look_ahead: int = 240,
    datetime: str = "2025-12-12 20:00:00",
    k: float = 100.0,
    bins_count: int = 200,
    bandwidth: int = 5,
    kernel_type: str = "Triangular",
    kde_ignore_borders: bool = True,
) -> dict:
    """Run the full LBLA normalized VP pipeline for a single anchor minute.

    Parameters match the spec; all inputs are stored verbatim in the returned
    *data* dict along with every intermediate array, KDE/VP results, HVN peaks,
    and per-function + total wall-clock timing.
    """
    data = {
        "asset": asset,
        "look_back": look_back,
        "look_ahead": look_ahead,
        "datetime": datetime,
        "k": k,
        "bins_count": bins_count,
        "bandwidth": bandwidth,
        "kernel_type": kernel_type,
        "kde_ignore_borders": kde_ignore_borders,
    }

    timing: dict[str, float] = {}
    total_t0 = time.perf_counter()

    for fn in (lb_la_n_base, append_cached_metrics, vp_analysis, vp_hvn):
        t0 = time.perf_counter()
        data = fn(data)
        timing[fn.__name__] = time.perf_counter() - t0

    timing["total"] = time.perf_counter() - total_t0
    data["timing"] = timing
    return data


# ---------------------------------------------------------------------------
# Step 1: look-back / look-ahead windows + normalization
# ---------------------------------------------------------------------------

def lb_la_n_base(data: dict) -> dict:
    """Slice look-back and look-ahead candle windows and normalize prices.

    Reads the in-memory candle cache (raises RuntimeError if not pre-loaded).
    Adds all window arrays and time axes to *data*.
    """
    from packages.tools.candle_cache import get_cached_candles

    asset = data["asset"]
    look_back = data["look_back"]
    look_ahead = data["look_ahead"]
    k = data["k"]
    dt_input = data["datetime"]

    # Parse datetime → ms epoch
    if isinstance(dt_input, str):
        dt = datetime.fromisoformat(dt_input).replace(tzinfo=timezone.utc)
    else:
        dt = dt_input if dt_input.tzinfo else dt_input.replace(tzinfo=timezone.utc)
    current_ts = int(dt.timestamp() * 1000)
    last_candle_ts = current_ts - 60_000

    cache = get_cached_candles(asset)
    ts_arr = cache["ts"]
    n = cache["_len"]

    # O(1) anchor resolution
    i = int((last_candle_ts - cache["_ts_start"]) // cache["_ts_step"])
    if i < 0 or i >= n or int(ts_arr[i]) != last_candle_ts:
        raise ValueError(
            f"last_candle_ts={last_candle_ts} not found in cached candles "
            f"for '{asset}'. Ensure the datetime falls on a candle present in "
            "the local data and that it is on an exact minute boundary."
        )

    # Bounds check
    if i - look_back + 1 < 0:
        raise ValueError(
            f"Look-back window of {look_back} candles starting at anchor i={i} "
            "exceeds the start of the cached data."
        )
    if i + look_ahead >= n:
        raise ValueError(
            f"Look-ahead window of {look_ahead} candles from anchor i={i} "
            "exceeds the end of the cached data."
        )

    lb_sl = slice(i - look_back + 1, i + 1)
    la_sl = slice(i + 1, i + 1 + look_ahead)

    lb_ts   = cache["ts"][lb_sl].copy()
    la_ts   = cache["ts"][la_sl].copy()
    lb_vwap = cache["vwap"][lb_sl].copy()
    la_vwap = cache["vwap"][la_sl].copy()
    lb_v    = cache["v"][lb_sl].copy()
    la_v    = cache["v"][la_sl].copy()
    lb_vb   = cache["vb"][lb_sl].copy()
    la_vb   = cache["vb"][la_sl].copy()
    lb_vs   = cache["vs"][lb_sl].copy()
    la_vs   = cache["vs"][la_sl].copy()

    current_price = float(cache["c"][i])

    # Normalize (per idea_normalize_based_on_last_price_clip.md)
    price_l = lb_vwap[-1]
    lb_pnc = k * (lb_vwap - price_l) / price_l
    la_pnc = k * (la_vwap - price_l) / price_l
    lb_p = np.clip(lb_pnc, -1.0, 1.0)
    la_p = np.clip(la_pnc, -1.0, 1.0)

    # Time axes (per idea_normalize_based_on_last_price_clip.md)
    lb_x = np.arange(look_back, dtype=np.float64) / look_back
    la_x = (look_back + np.arange(look_ahead, dtype=np.float64)) / look_back

    data.update(
        current_ts=current_ts,
        last_candle_ts=last_candle_ts,
        current_price=current_price,
        lb_ts=lb_ts, la_ts=la_ts,
        lb_vwap=lb_vwap, la_vwap=la_vwap,
        lb_pnc=lb_pnc, la_pnc=la_pnc,
        lb_p=lb_p, la_p=la_p,
        lb_v=lb_v, la_v=la_v,
        lb_vb=lb_vb, la_vb=la_vb,
        lb_vs=lb_vs, la_vs=la_vs,
        lb_x=lb_x, la_x=la_x,
    )
    return data


# ---------------------------------------------------------------------------
# Step 2: append metrics at the anchor row
# ---------------------------------------------------------------------------

def append_cached_metrics(data: dict) -> dict:
    """Read the metrics cache and add anchor-row values to data["metrics"].

    Raises FileNotFoundError if the cache file is missing.
    Raises ValueError if the anchor timestamp is absent.
    All metric columns (except ts) are picked up automatically.
    """
    asset = data["asset"]
    last_candle_ts = data["last_candle_ts"]

    cache_path = Path.cwd() / "data" / f"metrics_cache_{asset}.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Metrics cache not found: {cache_path}. "
            "Run metrics_cache.create_metrics_cache_base_file, "
            "metrics_cache_volume_median_iqr, and metrics_cache_volume_mean_stddev."
        )

    df = pd.read_parquet(cache_path)
    row = df[df["ts"] == last_candle_ts]
    if row.empty:
        raise ValueError(
            f"last_candle_ts={last_candle_ts} not found in metrics cache "
            f"for '{asset}'. Ensure candles and the metrics cache are in sync."
        )

    row0 = row.iloc[0]
    data["metrics"] = {col: float(row0[col]) for col in df.columns if col != "ts"}
    return data


# ---------------------------------------------------------------------------
# Step 3: volume-profile KDE
# ---------------------------------------------------------------------------

def vp_analysis(data: dict) -> dict:
    """Build volume-weighted KDE over look-back normalized prices.

    Calls kde_tools.compute_kde and maps its outputs to vp_hist / vp_kde /
    kde_kernel. Prints n_excluded on a single line.
    """
    from packages.kde_tools import compute_kde

    result = compute_kde(
        data["lb_p"],
        data["lb_v"],
        bins=data["bins_count"],
        kernel_type=data["kernel_type"],
        bandwidth=data["bandwidth"],
        ignore_borders=data["kde_ignore_borders"],
    )

    print(f"vp_analysis: n_excluded={result['n_excluded']}")

    data["kde_kernel"]  = result["kernel"]
    data["bin_width"]   = result["bin_width"]
    data["bin_centers"] = result["bin_centers"]
    data["vp_hist"]     = result["counts"]
    data["vp_kde"]      = result["kde"]
    return data


# ---------------------------------------------------------------------------
# Step 4: high-volume nodes
# ---------------------------------------------------------------------------

def vp_hvn(data: dict) -> dict:
    """Find POC + 3 peaks above + 3 peaks below from vp_kde.

    Uses kde_tools.kde_peak_widths for prominence + widths at rel_height
    1.0 and 0.5. Widths are stored in bins; multiply by bin_width for
    normalized-price units. Results are stored in data["hvn"].
    """
    from packages.kde_tools import kde_peak_widths

    vp_kde     = data["vp_kde"]
    bin_centers = data["bin_centers"]
    bandwidth  = data["bandwidth"]
    bin_width  = data["bin_width"]

    # --- POC ---
    poc_idx = int(np.argmax(vp_kde))
    poc_details = _peak_record(vp_kde, bin_centers, np.array([poc_idx]), bin_width, 0)
    poc = poc_details[0] if poc_details else None

    # --- Above / below peaks (global indices) ---
    above = _top_peaks(vp_kde, bin_centers, bandwidth, n=3, above=True, bin_width=bin_width)
    below = _top_peaks(vp_kde, bin_centers, bandwidth, n=3, above=False, bin_width=bin_width)

    data["hvn"] = {"poc": poc, "above": above, "below": below}
    return data


def _top_peaks(
    kde: np.ndarray,
    bin_centers: np.ndarray,
    distance: int,
    n: int,
    above: bool,
    bin_width: float,
) -> list[dict]:
    """Return up to n peaks from the above or below half, with full details."""
    from packages.kde_tools import kde_peak_widths

    mask = (bin_centers >= 0.0) if above else (bin_centers < 0.0)
    local_kde = kde[mask]
    global_map = np.where(mask)[0]

    if len(local_kde) == 0:
        return []

    local_peaks, _ = find_peaks(local_kde, distance=distance)
    if len(local_peaks) == 0:
        return []

    local_proms = peak_prominences(local_kde, local_peaks)[0]
    valid = local_proms > 0
    local_peaks = local_peaks[valid]
    local_proms = local_proms[valid]

    if len(local_peaks) == 0:
        return []

    order = np.argsort(local_proms)[::-1][:n]
    selected_local = local_peaks[order]
    selected_global = global_map[selected_local]

    return _peak_record(kde, bin_centers, selected_global, bin_width)


def _peak_record(
    kde: np.ndarray,
    bin_centers: np.ndarray,
    global_indices: np.ndarray,
    bin_width: float,
    _unused: int = 0,
) -> list[dict]:
    """Build a list of peak dicts from global KDE indices."""
    from packages.kde_tools import kde_peak_widths

    if len(global_indices) == 0:
        return []

    details = kde_peak_widths(kde, global_indices)
    records = []
    for j, idx in enumerate(global_indices):
        records.append({
            "price":      float(bin_centers[idx]),
            "prominence": float(details["proms"][j]),
            "width_h1":   float(details["widths_h1"][j]),
            "width_h05":  float(details["widths_h05"][j]),
        })
    return records
