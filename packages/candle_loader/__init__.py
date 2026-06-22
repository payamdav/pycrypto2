import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np

_VALID_COLUMNS = {"o", "h", "l", "c", "v", "q", "n", "vwap", "vb", "vs"}


def _cache_filename(asset: str, date_from: str, date_to: str) -> str:
    def compact(s: str) -> str:
        return s.replace("-", "").replace(" ", "T").replace(":", "")[:15] if s else "all"
    return f"{asset}_1m_{compact(date_from)}_{compact(date_to)}.parquet"


def _find_cache(filename: str) -> Path | None:
    cwd = Path.cwd()
    for candidate in [cwd / filename, cwd / "data" / filename]:
        if candidate.exists():
            return candidate
    return None


def load_candles(asset: str, date_from: str, date_to: str, columns: list[str]) -> np.ndarray:
    if not columns:
        raise ValueError("columns must not be empty")
    invalid = [c for c in columns if c not in _VALID_COLUMNS]
    if invalid:
        if "ts" in invalid:
            raise ValueError(
                "ts must not be listed in columns — it is always included automatically as column 0"
            )
        raise ValueError(f"Invalid column name(s): {invalid}. Valid names: {sorted(_VALID_COLUMNS)}")

    asset = asset.lower()
    cols_sql = ", ".join(columns)

    conditions = []
    if date_from:
        conditions.append(f"ts >= '{date_from}'::TIMESTAMP")
    if date_to:
        conditions.append(f"ts <= '{date_to}'::TIMESTAMP")
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    cache_file = _cache_filename(asset, date_from, date_to)
    cached = _find_cache(cache_file)

    con = duckdb.connect()
    t0 = time.perf_counter()

    if cached:
        source = str(cached)
        query = f"""
            SELECT ts, {cols_sql}
            FROM read_parquet('{source}')
            {where_clause}
            ORDER BY ts ASC;
        """
        result = con.execute(query).fetchnumpy()
    else:
        hf_path = f"hf://datasets/payamdavaee/candles/{asset}/*.parquet"
        # Fetch all columns so the cache serves any future column combination.
        all_query = f"""
            SELECT *
            FROM read_parquet('{hf_path}')
            {where_clause}
            ORDER BY ts ASC;
        """
        full = con.execute(all_query).fetchdf()

        cache_dir = Path.cwd() / "data"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / cache_file
        full.to_parquet(cache_path, index=False)

        # Derive the requested columns from the already-fetched DataFrame.
        selected = full[["ts"] + list(columns)]
        result = {col: selected[col].to_numpy() for col in selected.columns}

    elapsed = time.perf_counter() - t0

    n_rows = len(result["ts"])
    n_cols = 1 + len(columns)
    out = np.empty((n_rows, n_cols), dtype=np.float64)

    ts_raw = result["ts"]
    if hasattr(ts_raw, "dtype") and ts_raw.dtype.kind == "M":
        out[:, 0] = ts_raw.astype("datetime64[ms]").astype(np.float64)
    else:
        out[:, 0] = np.asarray(ts_raw, dtype=np.float64)

    for i, col in enumerate(columns, start=1):
        out[:, i] = np.asarray(result[col], dtype=np.float64)

    col_labels = ["ts"] + list(columns)
    col_index = ",".join(f"{name}:{i}" for i, name in enumerate(col_labels))
    cache_tag = f" [cached:{cached}]" if cached else f" [saved:{Path.cwd()/'data'/cache_file}]"

    if n_rows > 0:
        first_dt = datetime.fromtimestamp(out[0, 0] / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        last_dt = datetime.fromtimestamp(out[-1, 0] / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        print(f"{out.shape} {first_dt} {last_dt} {elapsed:.2f}s [{col_index}]{cache_tag}")
    else:
        print(f"{out.shape} - - {elapsed:.2f}s [{col_index}]{cache_tag}")

    return out
