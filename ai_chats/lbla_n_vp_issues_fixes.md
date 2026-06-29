# Spec: LBLA-N-VP fixes (metrics-cache lookup, docstring, caching)

## 1. Task summary

Address four items reported against `strategies/lbla_n_vp/lbla_n_vp.py`:

1. Fix the metrics-cache anchor lookup in `append_cached_metrics` — `last_candle_ts` (13-digit ms int) must be compared as `datetime64[ms]`.
2. Document in `lookback_lookahead_normalized_vp` that the `datetime` parameter is the **current time**, 60000 ms **ahead** of the last candle time.
3. Load the metrics-cache parquet **once** into a module-level cache instead of re-reading it on every `append_cached_metrics` call.
4. (Question, answered below) Confirm whether `get_cached_candles` re-reads from storage on each call.

## 2. Background and context

`lookback_lookahead_normalized_vp` runs a 4-step pipeline (`lb_la_n_base → append_cached_metrics → vp_analysis → vp_hvn`) for a single anchor minute. It is called repeatedly (many files / many anchors), so per-call file I/O is a real cost.

- `lb_la_n_base` parses `datetime` to a ms epoch (`current_ts`) and derives `last_candle_ts = current_ts - 60_000`.
- `append_cached_metrics` reads `CWD/data/metrics_cache_{asset}.parquet` and selects the row where `ts == last_candle_ts`.

## 3. Relevant conventions from `/agents/`

- **Writing style** (`agents/general/rules.md`): all docstrings/comments as short as possible while complete.
- **Local-cache-only** (`agents/packages/candle_cache.md`, `metrics_cache.md`): never download; read only local parquet.
- **candle_cache contract** (`agents/packages/candle_cache.md`): `preload_asset_candles` loads files into memory **once** and is idempotent (cache hit returns immediately); `get_cached_candles` returns the in-memory dict and raises `RuntimeError` if not pre-loaded.
- **metrics_cache dataset** (`agents/datasets/metrics_cache.md`): documents `ts` as `int64` ms-epoch — see Open Questions; this contradicts the observed runtime dtype.
- **No unrequested debugging/tests** (`agents/general/access.md`): only implement the items below.

## 4. Functional requirements

### FR1 — Fix metrics-cache anchor lookup (`append_cached_metrics`)
Replace the failing integer comparison with a `datetime64[ms]` comparison, per the report:

```python
row = df[df["ts"] == pd.to_datetime(last_candle_ts, unit="ms")]
```

- Keep the existing empty-result `ValueError` (message may stay as-is).
- No other behavior changes in this function beyond FR3.

### FR2 — Document the `datetime` parameter semantics
In `lookback_lookahead_normalized_vp`, next to the `datetime` parameter / its default, add a short note (in the docstring and/or an inline comment) stating:

> `datetime` is the **current time**, not the candle start time. It is 60000 ms **ahead** of the last candle time; `lb_la_n_base` derives `last_candle_ts = current_ts - 60_000`.

Keep it concise (writing-style rule). Do not change any logic.

### FR3 — Load metrics cache once (module-level cache)
`append_cached_metrics` must not re-read the parquet on every call.

- Introduce a module-level cache (e.g. a dict keyed by `asset`, or by resolved `cache_path`) holding the loaded DataFrame.
- First call for a given asset reads the parquet and stores it; subsequent calls reuse the stored DataFrame.
- Preserve current error behavior: still raise `FileNotFoundError` when the file is missing (check before/independent of the cache fill), and `ValueError` when the anchor `ts` is absent.
- Do not mutate the cached DataFrame in place; treat it as read-only.

### FR4 — Answer the `get_cached_candles` question (documentation only)
No code change required. The answer (record in the response, not necessarily the file): `get_cached_candles` does **not** re-read from storage. `preload_asset_candles` loads from disk once and is idempotent; `get_cached_candles` only returns the already-in-memory dict (or raises if not pre-loaded). With the required single preload per asset, storage is read once regardless of how many pipeline calls follow.

## 5. Non-goals / out of scope

- No change to `vp_analysis`, `vp_hvn`, or the KDE/HVN logic.
- No change to the `metrics_cache` builder package (the literal-`datetime64` fix was chosen over changing the builder).
- No new tests, benchmarks, or debugging passes.
- No cache-invalidation / reload mechanism for the metrics cache beyond first-load population.

## 6. Assumptions

- The metrics-cache `ts` column is `datetime64[ms]` at runtime (per the report and the chosen fix), even though the dataset spec documents `int64`.
- A single process/session is the unit of caching; a module-level dict surviving for the process lifetime is acceptable.
- The metrics-cache file does not change on disk during a run (no mid-run reload needed).

## 7. Acceptance criteria

- `append_cached_metrics` selects the anchor row using `pd.to_datetime(last_candle_ts, unit="ms")` and returns the same `data["metrics"]` shape (all columns except `ts`, as floats).
- Repeated calls to `append_cached_metrics` for the same asset read the parquet from disk **only on the first call**; later calls use the module-level copy.
- `FileNotFoundError` (missing file) and `ValueError` (missing anchor `ts`) are still raised under the same conditions.
- `lookback_lookahead_normalized_vp` documents that `datetime` is the current time, 60000 ms ahead of the last candle.
- No functional changes outside the items above.

## 8. Open questions

- **Dtype contradiction:** `agents/datasets/metrics_cache.md` documents `ts` as `int64` ms-epoch, but the report and chosen fix assume `datetime64[ms]`. Per user decision, implement the literal `datetime64` fix. If the cache is later confirmed/normalized to `int64`, this comparison must be revisited (and ideally the dataset spec corrected to match reality).

## 9. Notes for the downstream coding agent

- Touch only `strategies/lbla_n_vp/lbla_n_vp.py`.
- For FR3, define the module-level cache near the top of the module; guard reads so the `FileNotFoundError` path still fires when the file is absent. A dict keyed by `asset` (lowercased to match other helpers) is sufficient.
- Keep all added text minimal per the writing-style rule.
