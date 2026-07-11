import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np

COLUMNS = ["ts", "o", "h", "l", "c", "v", "q", "n", "vwap", "vb", "vs"]

_HF_GLOB = "hf://datasets/payamdavaee/candles/{asset}/*.parquet"


def _cache_path(asset: str) -> Path:
    return Path.cwd() / "data" / f"{asset}_1m_all.parquet"


def local_cache(assets: list[str] | str) -> None:
    """Ensure a single local parquet file per asset exists in CWD/data/.

    Downloads the full available candle range from the HuggingFace dataset
    only for assets whose file is not already present.
    """
    if isinstance(assets, str):
        assets = [assets]
    for asset in assets:
        asset = asset.lower()
        path = _cache_path(asset)
        t0 = time.perf_counter()
        if path.exists():
            print(f"[{asset}] candle file already cached: {path} ({time.perf_counter() - t0:.2f}s)")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        con = duckdb.connect()
        con.execute(f"""
            COPY (
                SELECT * FROM read_parquet('{_HF_GLOB.format(asset=asset)}')
                ORDER BY ts ASC
            ) TO '{tmp}' (FORMAT PARQUET);
        """)
        con.close()
        tmp.rename(path)
        print(f"[{asset}] candle file stored: {path} ({time.perf_counter() - t0:.2f}s)")


def _to_ms(value) -> int | None:
    """Normalize a boundary to a minute-truncated ms epoch. None passes through."""
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        ts = int(value)
        if len(str(ts)) != 13:
            raise ValueError(f"Integer timestamps must be 13-digit unix ms epochs, got: {ts}")
        return ts - ts % 60_000
    dt = datetime.fromisoformat(str(value)).replace(second=0, microsecond=0, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def load_candles(asset: str, date_from=None, date_to=None) -> np.ndarray:
    """Load minute candles for one asset from the local cache file.

    date_from / date_to: None (open end), datetime string like
    "2026-05-01 13:55:44" (seconds truncated, UTC), or 13-digit unix ms
    timestamp. Both inclusive. Returns an (n, 11) float64 ndarray with
    columns ts, o, h, l, c, v, q, n, vwap, vb, vs.
    """
    asset = asset.lower()
    local_cache(asset)

    conditions = []
    from_ms = _to_ms(date_from)
    to_ms = _to_ms(date_to)
    if from_ms is not None:
        conditions.append(f"ts >= epoch_ms({from_ms})")
    if to_ms is not None:
        conditions.append(f"ts <= epoch_ms({to_ms})")
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    t0 = time.perf_counter()
    con = duckdb.connect()
    result = con.execute(f"""
        SELECT {', '.join(COLUMNS)}
        FROM read_parquet('{_cache_path(asset)}')
        {where_clause}
        ORDER BY ts ASC;
    """).fetchnumpy()
    con.close()

    out = np.empty((len(result["ts"]), len(COLUMNS)), dtype=np.float64)
    for i, col in enumerate(COLUMNS):
        arr = np.asarray(result[col])
        if arr.dtype.kind == "M":
            arr = arr.astype("datetime64[ms]").astype(np.int64)
        out[:, i] = arr.astype(np.float64)

    elapsed = time.perf_counter() - t0
    if out.shape[0] > 0:
        first = datetime.fromtimestamp(out[0, 0] / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M")
        last = datetime.fromtimestamp(out[-1, 0] / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M")
        print(f"[{asset}] loaded {out.shape} candles {first} .. {last} ({elapsed:.2f}s)")
    else:
        print(f"[{asset}] loaded {out.shape} candles ({elapsed:.2f}s)")
    return out
