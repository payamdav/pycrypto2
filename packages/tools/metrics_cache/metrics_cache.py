import time
import numpy as np
import pandas as pd
from pathlib import Path

from packages.indicators.rolling_robust_z_score import rolling_median_iqr
from packages.indicators.rolling_mean_stddev import rolling_mean_stddev

_WINDOW = 7 * 1440  # 10080 one-minute candles


def _load_candle_df(asset: str) -> pd.DataFrame:
    """Locate, load, dedup, sort, and gap-check candle files for asset."""
    cwd = Path.cwd()
    data_dir = cwd / "data"
    pattern = f"{asset}_1m_*.parquet"

    files = []
    for directory in (cwd, data_dir):
        if directory.exists():
            for p in directory.glob(pattern):
                if not p.name.startswith("metrics_cache_"):
                    files.append(p)

    if not files:
        raise FileNotFoundError(
            f"No candle files found for '{asset}' (pattern '{pattern}' in CWD and CWD/data). "
            "Preload candles first via packages.tools.candle_preloader."
        )

    frames = [pd.read_parquet(f) for f in files]
    df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="ts")
        .sort_values("ts")
        .reset_index(drop=True)
    )

    ts = df["ts"].to_numpy(dtype=np.int64)
    for i in range(1, len(ts)):
        diff = int(ts[i]) - int(ts[i - 1])
        if diff != 60_000:
            raise ValueError(
                f"Candle gap for '{asset}': ts[{i-1}]={ts[i-1]}, ts[{i}]={ts[i]}, "
                f"diff={diff} ms (expected 60000). Preload a complete range."
            )
    return df


def _metrics_cache_path(asset: str) -> Path:
    return Path.cwd() / "data" / f"metrics_cache_{asset}.parquet"


def _verify_ts_match(asset: str, candle_df: pd.DataFrame, cache_df: pd.DataFrame) -> None:
    candle_ts = candle_df["ts"].to_numpy(dtype=np.int64)
    cache_ts = cache_df["ts"].to_numpy(dtype=np.int64)
    if len(candle_ts) != len(cache_ts) or not np.array_equal(candle_ts, cache_ts):
        raise ValueError(
            f"Candle ts does not match metrics cache ts for '{asset}'. "
            "Re-run create_metrics_cache_base_file to resync."
        )


def create_metrics_cache_base_file(assetname: str) -> Path:
    """Build (or reset) the metrics cache base file containing only the ts column.

    Overwrites any existing metrics_cache_{asset}.parquet.
    Raises FileNotFoundError if candle files are missing; raises ValueError on gaps.
    """
    asset = assetname.lower()
    df = _load_candle_df(asset)
    out = _metrics_cache_path(asset)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ts": df["ts"]}).to_parquet(out, index=False)
    return out


def metrics_cache_volume_median_iqr(assetname: str) -> Path:
    """Append v_median and v_iqr columns to the metrics cache.

    Uses rolling_median_iqr with a 10080-candle (1-week) look-back window.
    Prints load / compute / write / total timing.
    Raises FileNotFoundError if the metrics cache base file does not exist.
    """
    asset = assetname.lower()
    cache_path = _metrics_cache_path(asset)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Metrics cache not found for '{asset}': {cache_path}. "
            "Run create_metrics_cache_base_file first."
        )

    total_t0 = time.perf_counter()

    t0 = time.perf_counter()
    candle_df = _load_candle_df(asset)
    cache_df = pd.read_parquet(cache_path)
    t_load = time.perf_counter() - t0

    _verify_ts_match(asset, candle_df, cache_df)

    v = candle_df["v"].to_numpy(dtype=np.float64)

    t0 = time.perf_counter()
    med_iqr = rolling_median_iqr(v, _WINDOW)
    t_compute = time.perf_counter() - t0

    cache_df["v_median"] = med_iqr[:, 0]
    cache_df["v_iqr"] = med_iqr[:, 1]

    t0 = time.perf_counter()
    cache_df.to_parquet(cache_path, index=False)
    t_write = time.perf_counter() - t0

    t_total = time.perf_counter() - total_t0
    print(f"metrics_cache_volume_median_iqr [{asset}]")
    print(f"  load    {t_load:.3f}s")
    print(f"  compute {t_compute:.3f}s")
    print(f"  write   {t_write:.3f}s")
    print(f"  total   {t_total:.3f}s")

    return cache_path


def metrics_cache_volume_mean_stddev(assetname: str) -> Path:
    """Append v_mean and v_stddev columns to the metrics cache.

    Uses rolling_mean_stddev with a 10080-candle (1-week) look-back window.
    Prints load / compute / write / total timing.
    Raises FileNotFoundError if the metrics cache base file does not exist.
    """
    asset = assetname.lower()
    cache_path = _metrics_cache_path(asset)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Metrics cache not found for '{asset}': {cache_path}. "
            "Run create_metrics_cache_base_file first."
        )

    total_t0 = time.perf_counter()

    t0 = time.perf_counter()
    candle_df = _load_candle_df(asset)
    cache_df = pd.read_parquet(cache_path)
    t_load = time.perf_counter() - t0

    _verify_ts_match(asset, candle_df, cache_df)

    v = candle_df["v"].to_numpy(dtype=np.float64)

    t0 = time.perf_counter()
    mean_std = rolling_mean_stddev(v, _WINDOW)
    t_compute = time.perf_counter() - t0

    cache_df["v_mean"] = mean_std[:, 0]
    cache_df["v_stddev"] = mean_std[:, 1]

    t0 = time.perf_counter()
    cache_df.to_parquet(cache_path, index=False)
    t_write = time.perf_counter() - t0

    t_total = time.perf_counter() - total_t0
    print(f"metrics_cache_volume_mean_stddev [{asset}]")
    print(f"  load    {t_load:.3f}s")
    print(f"  compute {t_compute:.3f}s")
    print(f"  write   {t_write:.3f}s")
    print(f"  total   {t_total:.3f}s")

    return cache_path
