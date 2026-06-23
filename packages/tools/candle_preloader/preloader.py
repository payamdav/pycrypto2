import io
import re
import time
import requests
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime, timezone
from pathlib import Path

ALL_ASSETS = [
    "btcusdt", "ethusdt", "trumpusdt", "vineusdt",
    "adausdt", "xrpusdt", "dogeusdt",
]

_RESOLUTION = "1m"
_HF_BASE = "https://huggingface.co/datasets/payamdavaee/candles/resolve/main"


def _encode_dt(dt_str: str) -> str:
    return dt_str.replace(" ", "T").replace(":", "").replace("-", "")[:15]


def _final_cache_name(asset: str, start: str | None, end: str | None) -> str:
    if start and end:
        return f"{asset}_{_RESOLUTION}_{_encode_dt(start)}_{_encode_dt(end)}.parquet"
    return f"{asset}_{_RESOLUTION}_all.parquet"


def _resolve_cache(cwd: Path, data_dir: Path, name: str) -> Path | None:
    for candidate in (cwd / name, data_dir / name):
        if candidate.exists():
            return candidate
    return None


def _available_months(asset: str) -> list[tuple[str, str]]:
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files("payamdavaee/candles", repo_type="dataset")
    pattern = re.compile(rf"^{asset}/{asset}-1m-(\d{{4}})-(\d{{2}})\.parquet$")
    return sorted(
        (m.group(1), m.group(2))
        for f in files
        if (m := pattern.match(f))
    )


def _months_in_range(
    months: list[tuple[str, str]],
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> list[tuple[str, str]]:
    result = []
    for year, month in months:
        iy, im = int(year), int(month)
        month_start = datetime(iy, im, 1, tzinfo=timezone.utc)
        next_month = datetime(iy + 1, 1, 1, tzinfo=timezone.utc) if im == 12 else datetime(iy, im + 1, 1, tzinfo=timezone.utc)
        if start_dt and next_month <= start_dt:
            continue
        if end_dt and month_start > end_dt:
            continue
        result.append((year, month))
    return result


def _fetch_monthly(
    asset: str, year: str, month: str,
    monthly_dir: Path, cwd: Path, data_dir: Path,
) -> Path:
    fname = f"{asset}_{_RESOLUTION}_{year}_{month}.parquet"
    for candidate in (cwd / fname, data_dir / fname, monthly_dir / fname):
        if candidate.exists():
            return candidate
    monthly_dir.mkdir(parents=True, exist_ok=True)
    url = f"{_HF_BASE}/{asset}/{asset}-1m-{year}-{month}.parquet"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    out = monthly_dir / fname
    out.write_bytes(resp.content)
    return out


def preload_candles(
    assets: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    data_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Preload candle data from HuggingFace into CWD/data/, respecting local cache.

    Parameters
    ----------
    assets:
        Symbols to load (lowercase). None loads all seven assets.
    start:
        Inclusive start datetime, "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS" (UTC).
        None = from the earliest available month.
    end:
        Inclusive end datetime, same format as start.
        None = through the latest available month.
    data_dir:
        Override the cache directory. Defaults to ``<CWD>/data/``.

    Returns
    -------
    dict mapping asset symbol → absolute Path of the cached parquet file.
    """
    target_assets = [a.lower() for a in (assets or ALL_ASSETS)]

    cwd = Path.cwd()
    if data_dir is None:
        data_dir = cwd / "data"
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir = data_dir / "monthly_cache"

    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc) if start else None
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc) if end else None

    result: dict[str, Path] = {}
    asset_times: dict[str, float] = {}
    overall_start = time.perf_counter()

    for asset in target_assets:
        t0 = time.perf_counter()
        cache_name = _final_cache_name(asset, start, end)
        cached = _resolve_cache(cwd, data_dir, cache_name)
        if cached:
            result[asset] = cached
            elapsed = time.perf_counter() - t0
            asset_times[asset] = elapsed
            print(f"  [{asset}] cache hit → {cached}  ({elapsed:.2f}s)")
            continue

        available = _available_months(asset)
        needed = _months_in_range(available, start_dt, end_dt)

        if not needed:
            print(f"  [{asset}] no data found for requested range — skipped")
            continue

        frames: list[pd.DataFrame] = []
        for year, month in needed:
            monthly_path = _fetch_monthly(asset, year, month, monthly_dir, cwd, data_dir)
            frames.append(pd.read_parquet(monthly_path))

        df = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)

        if start_dt or end_dt:
            ts_ms = df["ts"].astype("int64")
            mask = pd.Series(True, index=df.index)
            if start_dt:
                mask &= ts_ms >= int(start_dt.timestamp() * 1000)
            if end_dt:
                mask &= ts_ms <= int(end_dt.timestamp() * 1000)
            df = df[mask].reset_index(drop=True)

        out = data_dir / cache_name
        df.to_parquet(out, index=False)
        result[asset] = out

        elapsed = time.perf_counter() - t0
        asset_times[asset] = elapsed
        print(f"  [{asset}] downloaded {len(df):,} rows → {out}  ({elapsed:.2f}s)")

    overall_elapsed = time.perf_counter() - overall_start

    print("\n--- preload_candles timing ---")
    for asset, t in asset_times.items():
        print(f"  {asset:<12} {t:6.2f}s")
    print(f"  {'TOTAL':<12} {overall_elapsed:6.2f}s")

    return result
