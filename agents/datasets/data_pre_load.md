# Data Pre-Load and Local Cache Convention

This document defines the mandatory caching rules that all data loading functions and modules must follow, and the pre-load pattern that AI Agents must implement when asked to load data as part of building a notebook or script.

> **Candle data:** do not implement a custom pre-load for 1-minute candles. Use `packages/candle_loader` (`local_cache` / `load_candles`) — it already implements this convention. See `agents/packages/candle_loader.md`.

---

## Rule 1 — Always Check Local Cache Before Connecting to a Data Store

Every data loading function or module — regardless of whether it lives in `packages/`, a strategy sub-folder, or is specific to a single notebook or script — **must** check for a locally cached copy of the requested data before establishing any connection to an external data store (Hugging Face, Google Cloud Storage, any other remote source).

### Cache lookup order

1. `<CWD>/<expected_filename>` — file directly in the current working directory.
2. `<CWD>/data/<expected_filename>` — file inside a `data/` sub-folder of the current working directory.

If a matching file is found at either location, load from it and **skip the remote connection entirely**.

> This rule applies unconditionally: local machine, cloud VM, VPS, hosted notebook (Colab, Kaggle, SageMaker, etc.). The environment does not change the requirement.

### Expected filename convention

The cached filename must be deterministic and encode the key parameters of the request so the same filename is produced on every call. Recommended pattern:

```
<asset>_<resolution>_<start_date>_<end_date>.<ext>
# e.g. btcusd_1h_20250101_20250201.parquet
```

For full-dataset (all-dates) loads omit the date range:

```
<asset>_<resolution>_all.<ext>
# e.g. ethusd_1h_all.parquet
```

---

## Rule 2 — AI Agent Must Implement a Pre-Load Function When Asked

When an instruction for building a notebook or script contains a **pre-load request**, the AI Agent must:

1. Create a dedicated pre-load function (e.g. `preload_data(...)`) **before** any analysis or processing code in the file.
2. The function must:
   - Accept the asset(s) and date range (or "all dates") as parameters.
   - Derive the expected cache filename(s) deterministically.
   - Check `CWD` then `CWD/data/` for each file.
   - Download and save the file to `CWD/data/<filename>` only if it is **not** already cached.
   - Return the path(s) to the cached file(s) so the rest of the notebook/script reads from local disk.
3. Call the pre-load function at the top of the notebook/script (e.g. in the first executable cell) so subsequent cells never re-download.

This ensures that re-running notebook cells or re-executing scripts multiple times does not trigger redundant downloads.

---

## Pre-Load Request Syntax

Instructions may express pre-load requests in natural language. Examples and how to interpret them:

| Request | Interpretation |
|---|---|
| `preload btcusd from "2025-01-01 18:00:00" to "2025-02-01 20:00:00"` | Load BTCUSD candles for the given datetime range; cache as a single file. |
| `preload all assets for all dates` | Load every asset defined in `agents/datasets/assets.md` with no date filter; cache one file per asset. |
| `preload ethusd and solusd for 2024` | Load both assets for the full calendar year 2024. |

When "all assets" is specified, consult `agents/datasets/assets.md` for the authoritative asset list.

When no resolution is specified, use the default resolution defined in the relevant dataset schema under `agents/datasets/`.

---

## Reference Implementation Skeleton

```python
import os
from pathlib import Path

def _cache_path(asset: str, resolution: str, start: str | None, end: str | None) -> Path:
    cwd = Path.cwd()
    if start and end:
        s = start.replace(" ", "T").replace(":", "").replace("-", "")[:15]
        e = end.replace(" ", "T").replace(":", "").replace("-", "")[:15]
        fname = f"{asset}_{resolution}_{s}_{e}.parquet"
    else:
        fname = f"{asset}_{resolution}_all.parquet"
    # prefer CWD/data/
    return cwd / "data" / fname


def preload_data(asset: str, resolution: str = "1h",
                 start: str | None = None, end: str | None = None) -> Path:
    cache = _cache_path(asset, resolution, start, end)
    # also accept file directly in CWD
    cwd_direct = cache.parent.parent / cache.name
    if cwd_direct.exists():
        return cwd_direct
    if cache.exists():
        return cache
    # --- only reaches here on a cache miss ---
    cache.parent.mkdir(parents=True, exist_ok=True)
    df = _download_from_store(asset, resolution, start, end)  # implement per data store
    df.to_parquet(cache)
    return cache
```

Adapt `_download_from_store` to the appropriate data store (see `agents/datasets/huggingface_candles.md` or `agents/datasets/huggingface_depth_snapshot.md`).
