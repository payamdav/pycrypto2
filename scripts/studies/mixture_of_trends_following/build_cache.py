"""Build motf_cache_{asset}.npy: 4 look-back slopes, 4 volume imbalances, trigger flags,
4 look-ahead label slopes. Run: python3 build_cache.py [tag]"""
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np

from packages.candle_loader import load_candles
from packages.indicators import linreg_slope, volume_imbalance
from scripts.studies.mixture_of_trends_following.common import (
    COL_IMBALANCES, COL_LSLOPES, COL_SLOPES, COL_TRIGGER, COL_TS, IMB_COLS,
    LABEL_COLS, N_COLS, SLOPE_COLS, assets_of, cli_tag, data_dir, load_params, timed,
)


def sanitize_vwap(candles: np.ndarray) -> np.ndarray:
    """Non-finite vwap or v==0 rows -> previous valid vwap; leading bad rows -> first valid."""
    vwap = candles[:, 8].astype(np.float64)
    v = candles[:, 5]
    bad = ~np.isfinite(vwap) | (v == 0.0)
    good = np.flatnonzero(~bad)
    if good.size == 0:
        raise ValueError("no valid vwap in candle range")
    idx = np.where(bad, -1, np.arange(len(vwap)))
    idx = np.maximum.accumulate(idx)
    idx[: good[0]] = good[0]
    return vwap[idx]


def compute_cache(candles: np.ndarray, params: dict) -> np.ndarray:
    """(n, 17) float64 cache. See spec 4.4 / README for column layout."""
    n = candles.shape[0]
    vwap = sanitize_vwap(candles)
    vb, vs = candles[:, 9], candles[:, 10]

    slope_windows = params["slope_windows"]
    imbalance_windows = params["imbalance_windows"]
    label_windows = params["label_windows"]
    threshold = params["l_slopes_threshold"]

    cache = np.empty((n, N_COLS), dtype=np.float64)
    cache[:, COL_TS] = candles[:, COL_TS]

    # rel_slope: compute once per distinct window (slope + label windows share some values), reuse
    rel_slope = {}
    for w in sorted(set(slope_windows) | set(label_windows)):
        with timed(f"linreg_slope window={w}"):
            rel_slope[w] = linreg_slope(vwap, w) / vwap

    for col, w in zip(SLOPE_COLS, slope_windows):
        cache[:, col] = rel_slope[w]
    cache[:, COL_SLOPES] = np.all(cache[:, list(SLOPE_COLS)] > 0.0, axis=1).astype(np.float64)

    # imbalance: compute once per distinct window, reuse
    imbalance = {}
    for w in sorted(set(imbalance_windows)):
        with timed(f"volume_imbalance window={w}"):
            imbalance[w] = volume_imbalance(vb, vs, w)

    for col, w in zip(IMB_COLS, imbalance_windows):
        cache[:, col] = imbalance[w]
    cache[:, COL_IMBALANCES] = np.all(cache[:, list(IMB_COLS)] > 0.0, axis=1).astype(np.float64)

    cache[:, COL_TRIGGER] = ((cache[:, COL_SLOPES] == 1.0) & (cache[:, COL_IMBALANCES] == 1.0)).astype(np.float64)

    # labels: l_slope_W[i] = rel_slope_W[i+W] (left-shift), tail edge-filled with the last computed value
    for col, w in zip(LABEL_COLS, label_windows):
        src = rel_slope[w]
        shifted = np.empty(n, dtype=np.float64)
        if n > w:
            shifted[: n - w] = src[w:]
            shifted[n - w:] = src[-1]
        else:
            shifted[:] = src[-1]
        cache[:, col] = shifted
    cache[:, COL_LSLOPES] = np.any(cache[:, list(LABEL_COLS)] > threshold, axis=1).astype(np.float64)

    return cache


def build_asset_cache(asset: str, params: dict) -> Path:
    """Load candles, compute cache, save motf_cache_{asset}.npy. Picklable (module-level) for multiprocessing."""
    t0 = time.perf_counter()
    candles = load_candles(asset)
    cache = compute_cache(candles, params)

    out = data_dir(params["tag"]) / f"motf_cache_{asset}.npy"
    with timed(f"[{asset}] cache save"):
        np.save(out, cache)

    print(f"[{asset}] build_cache done: shape={cache.shape} ({time.perf_counter() - t0:.2f}s)")
    return out


def build_cache(params: dict) -> list:
    return [build_asset_cache(asset, params) for asset in assets_of(params)]


if __name__ == "__main__":
    tag = cli_tag(sys.argv)
    build_cache(load_params(tag))
