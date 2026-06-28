# Spec: `metrics_cache` package + datasets doc

## 1. Task summary

Introduce a per-asset **metrics cache** parquet file (`metrics_cache_{asset}.parquet` in
`CWD/data/`) that acts as a companion to an asset's cached candle files and holds
pre-calculated metric/indicator columns aligned 1:1 with the candle timestamps. Deliver:

1. A datasets doc describing the concept: `agents/datasets/metrics_cache.md`.
2. A `packages/tools/metrics_cache/` package with:
   - `create_metrics_cache_base_file(assetname)` — build the base file (just `ts`).
   - `metrics_cache_volume_median_iqr(assetname)` — append rolling median + IQR of volume.
   - `metrics_cache_volume_mean_stddev(assetname)` — append rolling mean + stddev of volume.
3. Doc updates to `agents/datasets/metrics_cache.md` each time columns are added.

## 2. Background and context

- Candle data is cached locally as `{asset}_1m_*.parquet` in `CWD/` or `CWD/data/`
  (see `packages/candle_loader/` and `packages/tools/candle_preloader/`). Resolution is
  always **1-minute** (`60_000` ms between consecutive candles).
- A metrics cache file mirrors a candle file's `ts` column exactly (same length, same
  order, same values) and adds metric columns. "Appending a column" means **read the file,
  build a new file with the extra column(s), and write it back** (parquet is immutable).
- Columns default to `float64` unless stated otherwise. `ts` is the exception — it is a
  verbatim copy of the candle `ts` column.

## 3. Relevant conventions from `/agents/`

- `agents/general/paths_and_files.md`: reusable libraries live under `@/packages/`;
  `packages/tools/` already hosts utility sub-packages. Datasets/format docs live under
  `agents/datasets/`.
- `agents/general/rules.md`:
  - Any `.py` with external deps needs a `requirements.txt` in its folder.
  - **Every package in `packages/` must have a doc in `agents/packages/`** — add
    `agents/packages/metrics_cache.md` (the API reference), distinct from the
    `agents/datasets/metrics_cache.md` file-format doc requested in Task 1.
  - Writing style: short and complete.
- `agents/datasets/data_pre_load.md`: always operate on the **local cache**; check
  `CWD/` then `CWD/data/`. These metrics functions are **local-only** — they must not
  download candles. If candle files are missing, raise a clear error telling the caller to
  preload first (via `packages/tools/candle_preloader`).
- `agents/packages/indicators.md`: indicator functions are `@nb.njit`, take `float64`
  1-D arrays, use explicit loops, default `window=60`.

## 4. Dependencies and prerequisites

- **Hard dependency — `rolling_median_iqr`:** Task 3 requires
  `rolling_median_iqr(array, window=60)` in
  `packages/indicators/rolling_robust_z_score.py`. **It does not exist yet** — it is
  specified in `ai_chats/rolling_median_iqr_indicator.md` and must be implemented before
  Task 3 can work. Implement that spec first (or as part of this work).
- **Partial-window mean/stddev — new indicator needed:** Task 4 says "just like task 3",
  i.e. use an indicator function. The existing `packages/indicators/ma.py` and
  `stddev.py` **cannot** be reused: they require a full window and pad early indices with
  `0.0`, violating the shrinking-partial-window rule in the Notes. A new partial-window
  function is required — `rolling_mean_stddev(array, window=60)` returning an `(n, 2)`
  array `[mean, stddev]` with the same look-back/partial-window semantics as
  `rolling_median_iqr` (population stddev, divide by `m`, to match existing `stddev.py`).
  Place it in the indicators package (see Open Questions for file location) and document it
  in `agents/packages/indicators.md`.

## 5. Functional requirements

### 5.1 Metrics cache file format (Task 1 — `agents/datasets/metrics_cache.md`)

Document, concisely and authoritatively:

- **Name / location:** `metrics_cache_{asset}.parquet` in `CWD/data/`.
- **Purpose:** companion to candle files; caches pre-calculated metrics/indicators.
- **Alignment:** same number of rows as the asset's full cached candle range; `ts` column
  is a verbatim copy of the candle `ts` (same order, ascending).
- **Columns:** at minimum `ts`. Additional columns are metrics, `float64` by default.
- **Mutation model:** append-by-rewrite (read → add column(s) → write new parquet).
- **Maintenance rule:** whenever a column is added/changed, append a short description of
  that column (name, meaning, source column, rolling window, units) to this same doc, so
  the file's schema is always fully described here.
- Include a running **"Columns"** section seeded with the columns added by Tasks 3 & 4.

### 5.2 Package layout (Task 2)

```
packages/tools/metrics_cache/
├── __init__.py            # re-export the three public functions
├── metrics_cache.py       # implementation (name at code_writer's discretion)
└── requirements.txt       # numpy, numba, pandas, pyarrow (+ duckdb only if used)
```

### 5.3 `create_metrics_cache_base_file(assetname)` (Task 2)

1. Lowercase `assetname`.
2. Locate the asset's candle file(s): non-recursive glob `{asset}_1m_*.parquet` in `CWD/`
   then `CWD/data/`, **excluding** any `metrics_cache_*.parquet`. If none found, raise a
   clear error (caller must preload candles first).
3. Load all matched files, concatenate, drop duplicate `ts`, sort ascending by `ts`.
   Derive the available date range from `min(ts)` / `max(ts)`.
4. **Gap check:** every consecutive `ts` difference must equal `60_000` ms (1-minute grid).
   If any gap exists, raise `ValueError` reporting the first offending timestamp(s) and the
   gap size. Do not silently fill.
5. Build a single-column frame with `ts` copied verbatim from the candles (same dtype and
   values), same length.
6. Write `CWD/data/metrics_cache_{asset}.parquet` (create `data/` if absent). If the file
   already exists it is **overwritten** — calling base reset the file to `ts`-only.
7. Return the written file path.

### 5.4 `metrics_cache_volume_median_iqr(assetname)` (Task 3)

1. Lowercase `assetname`. Require `metrics_cache_{asset}.parquet` to exist; otherwise raise
   a clear error directing the caller to run `create_metrics_cache_base_file` first.
2. Load candle `ts` and `v` (volume) for the asset from the local candle file(s) using the
   same discovery + gap logic as the base file. Verify `ts` matches the metrics cache `ts`
   (same length and values); raise on mismatch.
3. `window = 7 * 1440` (`10080`).
4. `med_iqr = rolling_median_iqr(v.astype(np.float64), window)` → shape `(n, 2)`.
5. Append two `float64` columns (proposed names `v_median` and `v_iqr`; see Open Questions)
   to the metrics cache via read → add → write.
6. **Print timing measurements**: candle load, compute, write, and total (use
   `time.perf_counter()`), clearly labeled.
7. Update `agents/datasets/metrics_cache.md` with the two new columns (name, = rolling
   median / IQR of volume `v`, 1-week look-back window `10080`).
8. Return the metrics cache path.

### 5.5 `metrics_cache_volume_mean_stddev(assetname)` (Task 4)

Identical to 5.4 but:

- Uses `rolling_mean_stddev(v.astype(np.float64), window)` (window `10080`).
- Appends `float64` columns (proposed names `v_mean` and `v_stddev`).
- Updates `agents/datasets/metrics_cache.md` accordingly.
- Prints the same timing measurements.

### 5.6 Rolling-window semantics (applies to Tasks 3 & 4 — from Notes)

- All rolling windows are **causal look-back** windows that **include the current item**
  and never use future data: for index `i`, the window is
  `v[max(0, i - window + 1) : i + 1]`.
- **Partial early windows shrink**: when fewer than `window` items precede `i`, compute
  over the available `m = min(i + 1, window)` items (down to `m = 1`). Every output row is
  a real value — no `0.0` padding. (This is exactly the `rolling_median_iqr` contract from
  `ai_chats/rolling_median_iqr_indicator.md`, and the required contract for the new
  `rolling_mean_stddev`.)
- Output metric columns have the **same length as the candles**.

## 6. Non-goals / out of scope

- No remote/HuggingFace downloading inside the metrics_cache functions (local cache only).
- No new candle-loading logic beyond reading existing local parquet files.
- No tests required (per `agents/general/access.md`), beyond the timing prints requested.
- No metrics beyond volume median/IQR and volume mean/stddev in this task.
- No change to existing `ma.py` / `stddev.py` behavior.

## 7. Assumptions

- "Candle files of asset name" = locally cached `{asset}_1m_*.parquet` under `CWD/` or
  `CWD/data/`. When several match, concatenate + dedupe + sort by `ts`.
- A "gap" means a missing 1-minute candle: any consecutive `ts` diff ≠ `60_000` ms.
- Population standard deviation (divide by `m`), matching existing `stddev.py`.
- `ts` is stored verbatim from candles; all metric columns are `float64`.
- "A week" = `7 * 1440 = 10080` one-minute candles.
- Metrics functions require the base file to already exist (they append, not create).

## 8. Acceptance criteria

- `agents/datasets/metrics_cache.md` exists and fully describes the file format, naming,
  alignment, mutation model, the maintenance rule, and a Columns section listing
  `ts`, `v_median`, `v_iqr`, `v_mean`, `v_stddev`.
- `agents/packages/metrics_cache.md` documents the package's public API.
- `packages/tools/metrics_cache/` exists with `__init__.py`, implementation, and
  `requirements.txt`; the three functions are importable.
- `create_metrics_cache_base_file(asset)` produces
  `CWD/data/metrics_cache_{asset}.parquet` with a single `ts` column equal in length/values
  to the asset's candle `ts`, and raises on missing candles or candle gaps.
- `metrics_cache_volume_median_iqr(asset)` appends two correct columns computed by
  `rolling_median_iqr` over a `10080` look-back window with shrinking partial windows;
  prints timing; updates the datasets doc.
- `metrics_cache_volume_mean_stddev(asset)` appends two correct columns computed by
  `rolling_mean_stddev` over a `10080` look-back window with shrinking partial windows;
  prints timing; updates the datasets doc.
- All metric columns have the same length as the candles; no `0.0` padding for early rows.

## 9. Open questions

1. **`rolling_median_iqr` not yet implemented** — confirm it should be implemented as part
   of this work (per `ai_chats/rolling_median_iqr_indicator.md`) before Task 3. Assumed
   **yes**.
2. **`rolling_mean_stddev` location** — new module `packages/indicators/rolling_mean_stddev.py`
   vs. adding it alongside `rolling_median_iqr` in `rolling_robust_z_score.py`. Assumed a
   **new module** + export from `packages/indicators/__init__.py`.
3. **Column names** — proposed `v_median`, `v_iqr`, `v_mean`, `v_stddev`. Should the
   window be encoded in the name (e.g. `v_median_10080`)? Assumed **no** (plain names).
4. **Base-file auto-create** — should the metric functions auto-call
   `create_metrics_cache_base_file` when the base is missing, or hard-error? Assumed
   **hard-error** with a clear message.
5. **stddev convention** — population (÷m, matching `stddev.py`) vs sample (÷m−1). Assumed
   **population**.

## 10. Notes for the downstream coding agent

- Implement `rolling_median_iqr` first (its own spec) so Task 3 has its dependency.
- Add `rolling_mean_stddev` as a `@nb.njit` indicator with the same partial-window,
  look-back contract; explicit loops only; population stddev; return `(n, 2)`.
- Keep candle discovery/gap-check logic shared between the base-file and metric functions
  (one internal helper) to avoid divergence.
- Metric functions must read the local candle parquet directly (pandas/pyarrow) — no
  `load_candles` HF round-trips and no network.
- Use `time.perf_counter()` and print clearly labeled timing (load / compute / write /
  total), mirroring the style in `packages/tools/candle_preloader/preloader.py`.
- After adding columns, append their descriptions to `agents/datasets/metrics_cache.md`
  in the same call — this is a hard format-doc rule, not optional.
- Keep docstrings short per the writing-style rule.
