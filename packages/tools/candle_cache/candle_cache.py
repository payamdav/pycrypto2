"""In-memory candle cache for fast repeated slicing.

Loaded candles are stored as numpy arrays keyed by column name.
Use preload_asset_candles() once per session, then get_cached_candles()
on every pipeline call.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

_CACHE: dict[str, dict] = {}

_COLUMNS = ["ts", "o", "h", "l", "c", "v", "q", "n", "vwap", "vb", "vs"]


def _find_candle_files(asset: str) -> list[Path]:
    cwd = Path.cwd()
    data_dir = cwd / "data"
    pattern = f"{asset}_1m_*.parquet"
    files = []
    for directory in (cwd, data_dir):
        if directory.exists():
            for p in directory.glob(pattern):
                if not p.name.startswith("metrics_cache_"):
                    files.append(p)
    return files


def preload_asset_candles(asset: str) -> dict:
    """Load all local candle parquet files for *asset* into the in-memory cache.

    Idempotent — returns immediately on a cache hit.
    Raises FileNotFoundError if no local files are found (run
    candle_preloader.preload_candles first).
    Prints load timing.
    """
    asset = asset.lower()
    if asset in _CACHE:
        return _CACHE[asset]

    files = _find_candle_files(asset)
    if not files:
        raise FileNotFoundError(
            f"No candle files for '{asset}' found in CWD or CWD/data "
            f"(pattern '{asset}_1m_*.parquet'). "
            "Run packages.tools.candle_preloader.preload_candles first."
        )

    t0 = time.perf_counter()
    frames = [pd.read_parquet(f) for f in files]
    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )

    available_cols = [c for c in _COLUMNS if c in df.columns]
    entry: dict = {}
    for col in available_cols:
        dtype = np.int64 if col == "ts" else np.float64
        entry[col] = df[col].to_numpy(dtype=dtype)

    # O(1) ts → index: candles are exactly 60_000 ms apart
    ts_arr = entry["ts"]
    entry["_ts_start"] = int(ts_arr[0])
    entry["_ts_step"] = 60_000
    entry["_len"] = len(ts_arr)

    _CACHE[asset] = entry
    elapsed = time.perf_counter() - t0
    print(f"candle_cache [{asset}]: loaded {len(ts_arr):,} candles in {elapsed:.3f}s")
    return entry


def get_cached_candles(asset: str) -> dict:
    """Return the in-memory cache entry for *asset*.

    Raises RuntimeError if the asset has not been pre-loaded.
    """
    asset = asset.lower()
    if asset not in _CACHE:
        raise RuntimeError(
            f"Candle cache for '{asset}' is not loaded. "
            f"Call candle_cache.preload_asset_candles('{asset}') first."
        )
    return _CACHE[asset]


def is_cached(asset: str) -> bool:
    """Return True if *asset* candles are currently in memory."""
    return asset.lower() in _CACHE


def clear_cache(asset: str | None = None) -> None:
    """Drop one (by name) or all entries from the in-memory cache."""
    if asset is None:
        _CACHE.clear()
    else:
        _CACHE.pop(asset.lower(), None)
