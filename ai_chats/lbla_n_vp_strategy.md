# Spec: `lbla_n_vp` — look-back / look-ahead normalized volume-profile analysis

## 1. Task summary

Build a new strategy `strategies/lbla_n_vp/` that computes a **volume-profile (VP)
analysis** over a normalized look-back window around a chosen minute, plus the
matching look-ahead window for later labeling. Deliver:

1. An importable Python module in `strategies/lbla_n_vp/` exposing
   `lookback_lookahead_normalized_vp(...)` and its helper functions
   (`lb_la_n_base`, `append_cached_metrics`, `vp_analysis`, `vp_hvn`).
   All helpers take and return a single `data` dictionary.
2. A test notebook `strategies/lbla_n_vp/lbla_n_vp.ipynb` that pre-loads candles
   and metrics, exposes all input parameters, and times single + 100-call runs.
3. A new reusable in-memory candle cache package
   `packages/tools/candle_cache/` (see §4) + its doc
   `agents/packages/candle_cache.md`.
4. `strategies/lbla_n_vp/requirements.txt` for the module's external deps.

The pipeline returns one `data` dict carrying every input, every intermediate
array, the VP/KDE results, the high-volume-node (HVN) peaks, and per-function +
total timing.

## 2. Background and context

This strategy reuses three existing project ideas/packages:

- **Look-back/look-ahead windowing** (`agents/ideas/idea_look_back_look_ahead.md`):
  for an observation anchored at the last look-back candle, the look-back window is
  the `look_back` candles ending at (and including) the anchor; the look-ahead
  window is the next `look_ahead` candles. `current_time = last_candle.ts + 60_000`.
- **Normalize-based-on-last-price-with-clip**
  (`agents/ideas/idea_normalize_based_on_last_price_clip.md`): prices are normalized
  around the last look-back candle's `vwap` (`price_l = lb_vwap[-1]`) via
  `scaled = clip(k * (price - price_l) / price_l, -1, 1)`. The same `price_l`
  anchors both look-back and look-ahead. Current price maps to `0.0`.
- **KDE / VP construction** (`packages/kde_tools/`, doc
  `agents/packages/kde_tools.md`): volume-weighted histogram + kernel convolution
  over normalized prices in `[-1, 1]`, plus peak finding.

The strategy operates on a **single minute** at a time (the `datetime` input),
not a swept date range — the look-back/look-ahead idea is applied to one anchor.

## 3. Relevant conventions from `/agents/`

- `agents/general/paths_and_files.md`: each strategy lives in its own folder under
  `strategies/` (no files at `strategies/` root); cross-strategy reusable code goes
  in `packages/`. The candle cache (reusable) therefore belongs in `packages/`,
  the strategy-specific pipeline in `strategies/lbla_n_vp/`.
- `agents/general/rules.md`:
  - Any `.py` with external deps needs a `requirements.txt` in its folder
    (module folder + new package folder).
  - **Every package in `packages/` must have a doc in `agents/packages/`** → add
    `agents/packages/candle_cache.md`. If `vp_hvn` adds functions to `kde_tools`,
    update `agents/packages/kde_tools.md` accordingly.
  - Notebooks must `%pip install` all deps in the first cell and clone the repo /
    set `sys.path` before any project import.
  - Writing style: short and complete (applies to all docstrings/docs here).
- `agents/datasets/data_pre_load.md`: data access is **local cache only**; check
  `CWD/` then `CWD/data/`. The pipeline must **not** download — if data is missing
  it raises and tells the caller to pre-load.
- `agents/datasets/metrics_cache.md` + `agents/packages/metrics_cache.md`: metrics
  live in `CWD/data/metrics_cache_{asset}.parquet`, aligned 1:1 with candle `ts`.
  Columns currently available: `v_median`, `v_iqr`, `v_mean`, `v_stddev`.
- `agents/datasets/assets.md`: asset symbols/folders are lowercase.
- `agents/packages/candle_loader.md`: candle columns are
  `o,h,l,c,v,q,n,vwap,vb,vs` (+ always `ts`).
- `agents/general/access.md`: testing is only required because the notebook is
  explicitly requested; do not add a unit-test suite. Do not debug unrelated code.

## 4. New package: `packages/tools/candle_cache/`

Rationale: `lb_la_n_base` must "look for pre-loaded and cached candles; if it
didn't find, raise an error." The notebook then calls the pipeline on 100 random
minutes, so candles must be held **in memory** for fast repeated slicing — not
re-read from parquet each call. `packages/tools/candle_preloader/` already handles
file caching (download → `CWD/data/{asset}_1m_*.parquet`); `candle_cache` is the
in-memory layer on top of it.

Proposed API (lowercase asset keys; module-level dict cache):

| Function | Behavior |
|---|---|
| `preload_asset_candles(asset) -> dict` | Resolve the asset's local candle parquet (`CWD/` then `CWD/data/`, per `data_pre_load.md`); if absent, raise `FileNotFoundError` telling caller to run `candle_preloader.preload_candles` first. Load once into memory as numpy arrays (`ts` int64 ms + needed float64 columns), store in the module cache, return the cached entry. Idempotent (cache hit returns immediately). Print load timing. |
| `get_cached_candles(asset) -> dict` | Return the in-memory entry for `asset`; raise a clear error if not pre-loaded. Used by `lb_la_n_base`. |
| `is_cached(asset) -> bool` | Whether the asset is in memory. |
| `clear_cache(asset=None)` | Drop one or all entries. |

Cached entry shape (suggested): a dict of contiguous `np.ndarray`s keyed by column
name (`ts`, `c`, `vwap`, `v`, `vb`, `vs`, …), all the same length, ascending by
`ts`, plus a precomputed `ts -> index` lookup (e.g. dict or the first `ts` +
constant 60_000 step) so anchor resolution from a `datetime` is O(1).

Document the package in `agents/packages/candle_cache.md` (purpose, API, the
"raise if not pre-loaded" contract, local-cache-only rule).

## 5. Functional requirements

### 5.1 `lookback_lookahead_normalized_vp(...)` — entry point

Named inputs with defaults:

| Param | Default | Meaning |
|---|---|---|
| `asset` | `"btcusdt"` | lowercase asset |
| `look_back` | `1440` | past candles, **includes** last candle |
| `look_ahead` | `240` | future candles |
| `datetime` | `"2025-12-12 20:00:00"` | current_time: moment after last candle closes (UTC) |
| `k` | `100` | scale factor; ratio `0.01` → `1.0` (`k = 1/R`) |
| `bins_count` | `200` | KDE histogram bins spanning `[-1, 1]` |
| `bandwidth` | `5` | kernel half-width in bins |
| `kernel_type` | `"Triangular"` | `Triangular` \| `Epanechnikov` \| `Uniform` |
| `kde_ignore_borders` | `True` | exclude clipped prices (`±1.0`) from the KDE histogram |

Behavior:

1. Build `data` dict holding **all** inputs as properties (keys named exactly as
   the params above; `datetime` may also be parsed to a ts but keep the raw input).
2. Call, in order, threading `data` through each:
   `lb_la_n_base` → `append_cached_metrics` → `vp_analysis` → `vp_hvn`.
3. Measure wall-clock time of **each** function and the **total**; store them in a
   nested `timing` dict added to `data` (e.g.
   `data["timing"] = {"lb_la_n_base": ..., "append_cached_metrics": ...,
   "vp_analysis": ..., "vp_hvn": ..., "total": ...}`, seconds as float).
4. Return `data`.

### 5.2 `lb_la_n_base(data) -> data`

Reads the in-memory cached candles for `data["asset"]` via
`candle_cache.get_cached_candles` (raise if not pre-loaded). Resolves the anchor
index from `data["datetime"]`:

- `current_time` = the input datetime (UTC) → `current_ts` (ms epoch).
- `last_candle_ts = current_ts - 60_000` (candle `ts` is open time; the candle that
  closed at `current_time` opened one minute earlier — see
  `idea_look_back_look_ahead.md`).
- Anchor index `i` = position of `last_candle_ts` in the cached `ts` array; raise a
  clear error if `last_candle_ts` is absent, or if the window would run past either
  end of the cached data (`i - look_back + 1 < 0` or `i + look_ahead >= len`).

Then slice and build (per the windowing idea, look-back **includes** the anchor):

- look-back rows: `[i - look_back + 1 : i + 1]` (length `look_back`).
- look-ahead rows: `[i + 1 : i + 1 + look_ahead]` (length `look_ahead`).

Normalize using `price_l = lb_vwap[-1]` (the last look-back candle's `vwap`) for
**both** windows (same anchor, hard clip to `[-1, 1]`), per
`idea_normalize_based_on_last_price_clip.md`:
`pnc = k * (vwap - price_l) / price_l` (normalized, not clipped);
`p = clip(pnc, -1, 1)`.

Properties added to `data`:

| Key | Description |
|---|---|
| `current_ts` | ms epoch of `current_time` |
| `last_candle_ts` | ms epoch of last look-back candle open |
| `current_price` | derived from last candle **close** (`c`) |
| `lb_ts`, `la_ts` | look-back / look-ahead `ts` arrays |
| `lb_vwap`, `la_vwap` | original vwaps |
| `lb_pnc`, `la_pnc` | normalized, not clipped |
| `lb_p`, `la_p` | normalized **and** clipped to `[-1,1]` |
| `lb_v`, `la_v` | volume |
| `lb_vb`, `la_vb` | buy volume |
| `lb_vs`, `la_vs` | sell volume |
| `lb_x`, `la_x` | time axes (see below) |

Time axes: use the normalized-time convention from
`idea_normalize_based_on_last_price_clip.md`:
`lb_x = arange(look_back) / look_back` (so `lb_x[-1] = (look_back-1)/look_back`),
`la_x = (look_back + arange(look_ahead)) / look_back` (so `la_x[0] = 1.0`,
the first look-ahead candle opening at `current_time`). All arrays are numpy,
ordered oldest→newest, look-back and look-ahead lengths exactly `look_back` /
`look_ahead`.

### 5.3 `append_cached_metrics(data) -> data`

Reads `CWD/data/metrics_cache_{asset}.parquet` (local only) and adds the metric
values **at the anchor row** (`last_candle_ts`) to `data`. Add every metric column
present in the cache (currently `v_median`, `v_iqr`, `v_mean`, `v_stddev`); use the
cache's column names verbatim as `data` keys (optionally namespaced, e.g.
`data["metrics"] = {...}` — pick one and document it). Raise a clear error if the
cache file is missing or the `last_candle_ts` row is absent (mirrors
`lb_la_n_base`'s "raise if not found" contract). Keep this generic so new metric
columns are picked up automatically without code changes.

### 5.4 `vp_analysis(data) -> data`

Uses `packages/kde_tools` over the **look-back** normalized+clipped prices
(`lb_p`) weighted by `lb_v`. Add to `data`:

- `kde_kernel` — kernel from `make_kernel(kernel_type, bandwidth)`.
- `bin_width`, `bin_centers` — bin geometry over `[-1, 1]` with `bins_count` bins.
- `vp_hist` — volume-weighted histogram from `lb_p` (values) and `lb_v` (weights)
  via `weighted_histogram(...)`.
- `vp_kde` — `convolve_same(vp_hist, kde_kernel)`.
- Respect `kde_ignore_borders`: when `True`, exclude entries exactly at `±1.0`
  before the histogram (strict `>`/`<`, matching `kde_tools.compute_kde`); count
  the dropped entries as `n_excluded` and **print it on a single line**.

Implementation note: `kde_tools.compute_kde(...)` already returns
`kde, counts, bin_centers, bin_width, kernel, n_excluded` for exactly this
pipeline — prefer calling it and mapping its outputs to the names above
(`counts→vp_hist`, `kde→vp_kde`, `kernel→kde_kernel`) rather than re-deriving, so
behavior matches the documented package bit-for-bit.

### 5.5 `vp_hvn(data) -> data`

Find high-volume nodes over `vp_kde` using `scipy.signal` (`find_peaks`,
`peak_prominences`, `peak_widths`). Identify **7 peaks**:

- **POC** (point of control): the dominant node — `bin_center` at `argmax(vp_kde)`.
- **3 peaks above** `0.0` (`bin_center >= 0`) with highest prominence, prominence
  `> 0`.
- **3 peaks below** `0.0` (`bin_center < 0`) with highest prominence, prominence
  `> 0`.

Use `kde_tools.kde_peaks_above_below(vp_kde, bin_centers, distance=bandwidth, n=3,
split_at=0.0)` for the above/below sets (consistent with the package). For **each**
of the 7 peaks store: normalized price (`bin_center`), prominence, peak width at
`rel_height=1.0`, and peak width at `rel_height=0.5`.

Because `kde_tools` currently returns only price + prominence (no widths), add the
width capability rather than duplicating peak logic in the strategy: extend
`kde_tools` with a function that, given the KDE series + selected peak indices,
returns prominences and widths at both `rel_height` values via
`scipy.signal.peak_widths`, and have `vp_hvn` call it. Update
`agents/packages/kde_tools.md` for any new function (per the package-doc rule).
Widths are returned in **bins**; optionally also expose them in normalized-price
units (`width_bins * bin_width`) — document whichever is stored.

Store results in a clear structure, e.g.
`data["hvn"] = {"poc": {price, prominence, width_h1, width_h05},
"above": [ {…}, {…}, {…} ], "below": [ {…}, {…}, {…} ]}`. If fewer than 3 peaks
exist on a side, store what exists (do not pad).

## 6. Test notebook — `strategies/lbla_n_vp/lbla_n_vp.ipynb`

Per `agents/general/rules.md`, the notebook must be self-contained. Cells:

1. `%pip install` all deps (`numpy`, `pandas`, `numba`, `scipy`, `pyarrow`, and any
   needed for HuggingFace download in `candle_preloader`).
2. Clone the repo and set `sys.path` so `packages.*` and the strategy module import.
   Note: `rules.md` shows `pycrypto`; this repo is `pycrypto2` — use the correct
   repo URL/name (see Open Questions).
3. A cell defining the **base asset name** (single variable, e.g. `asset =
   "btcusdt"`).
4. Pre-load **all candles** of that asset:
   `candle_preloader.preload_candles([asset])` (file cache) then
   `candle_cache.preload_asset_candles(asset)` (in-memory). Show timing.
5. Build the metrics cache per `agents/datasets/metrics_cache.md`:
   `create_metrics_cache_base_file(asset)`,
   `metrics_cache_volume_median_iqr(asset)`,
   `metrics_cache_volume_mean_stddev(asset)`. Show timing.
6. Display total time consumption for all of the above.
7. A cell listing **all input parameters with their defaults** (so they can be
   edited): `look_back, look_ahead, k, bins_count, bandwidth, kernel_type,
   kde_ignore_borders`.
8. Pick a **random minute** within the usable candle range (i.e. an anchor whose
   full look-back and look-ahead fit), call
   `lookback_lookahead_normalized_vp(...)`, and print the per-function + total
   timing detail from `data["timing"]`.
9. Pick **100 random minutes** in the usable range, call the pipeline for each, and
   print per-function + total timing with **averages** (mean per function, mean
   total). Reuse the in-memory cache across all 100 calls.

"Usable range" = anchor indices `[look_back-1 : len-look_ahead]` of the cached
array (inclusive-mode anchor range from `idea_look_back_look_ahead.md`), mapped
back to a datetime string for the call.

## 7. Non-goals / out of scope

- No live/remote data fetching inside the pipeline functions (local cache only;
  download happens only via `candle_preloader` in the notebook's pre-load step).
- No new indicators/metric columns (use what `metrics_cache` already provides).
- No plotting requirement (the spec asks for timing, not charts). A simple VP plot
  may be added only if trivial; not required.
- No backtest, signal generation, or position sizing — this is analysis only.
- No formal unit-test suite (only the requested notebook).
- No swept multi-date observation matrix — single-anchor analysis per call.

## 8. Assumptions

- **In-memory candle cache is a new reusable package** at
  `packages/tools/candle_cache/` (chosen over a strategy-local cache or re-reading
  parquet each call, because the 100-call timing test needs fast repeated access
  and "pre-loaded and cached candles" implies a shared in-memory layer). Override
  in §10 if a different location is preferred.
- **POC and all peaks are derived from `vp_kde`** (the smoothed curve), consistent
  with `kde_tools`. POC = `bin_center` at `argmax(vp_kde)`.
- **`append_cached_metrics` raises** if the metrics cache file or anchor row is
  missing (mirrors `lb_la_n_base`).
- `datetime` input is UTC and falls on an exact minute boundary present in the data.
- `price` for normalization is `vwap`; `current_price` is the last candle's `c`
  (close), per the idea docs.
- `distance` for `find_peaks` = `bandwidth` (the value `kde_peaks_above_below`
  defaults to via the kernel half-width); confirm if a different distance is wanted.
- Timing uses `time.perf_counter()`; numba-jitted paths are warmed once (the first
  pipeline call may be slower due to JIT — the 100-call average should reflect
  steady state; consider one warm-up call before the 100-call timing).

## 9. Acceptance criteria

- `from strategies.lbla_n_vp.<module> import lookback_lookahead_normalized_vp`
  works from the repo root (after clone/path setup).
- Calling the entry point with defaults (after pre-loading btcusdt candles +
  metrics) returns a `data` dict containing: all input params; every `lb_la_n_base`
  property in §5.2; the metric values from §5.3; `kde_kernel`, `bin_width`,
  `bin_centers`, `vp_hist`, `vp_kde` from §5.4; the `hvn` POC + 3-above + 3-below
  peaks each with price, prominence, and widths at `rel_height` 1.0 and 0.5 from
  §5.5; and a `timing` dict with per-function + total seconds.
- `vp_analysis` prints `n_excluded` on a single line.
- Look-back arrays have length `look_back`, look-ahead arrays length `look_ahead`;
  `lb_p[-1]` corresponds to normalized `~0.0` (current price), within clip.
- Functions raise clear errors (not silent/empty results) when candles aren't
  pre-loaded, the metrics cache is missing, or the window doesn't fit.
- `packages/tools/candle_cache/` has `requirements.txt` and a matching
  `agents/packages/candle_cache.md`; any new `kde_tools` function is reflected in
  `agents/packages/kde_tools.md`; `strategies/lbla_n_vp/requirements.txt` exists.
- The notebook runs top-to-bottom: pre-loads candles + metrics with timing, lists
  editable params, runs the single random-minute call with timing detail, and the
  100 random-minute call with averaged timing.

## 10. Open questions

1. **Repo URL/name for the notebook clone cell.** `rules.md` examples use
   `pycrypto`; this repo is `pycrypto2`. Confirm the correct
   `REPO_URL`/`REPO_NAME` (and whether the notebook is expected to run on `main` or
   on the feature branch).
2. **Candle cache location** — package vs. strategy-local (Assumption taken:
   reusable package). Confirm.
3. **POC source** — `vp_kde` (assumed) vs. `vp_hist` (raw volume histogram).
4. **`find_peaks` distance** — use `bandwidth` (assumed) or a separate parameter?
   Should POC also be required to be a `find_peaks` peak, or strictly `argmax`?
5. **Metric storage key style** — flat keys (`data["v_median"]`) vs. nested
   (`data["metrics"]["v_median"]`). Spec assumes nested; confirm preference.
6. **Width units** — store peak widths in bins, in normalized-price units, or both.

## 11. Notes for the downstream coding agent

- Read all `agents/` files first (mandatory) and treat `kde_tools`,
  `candle_preloader`, `metrics_cache`, and the two idea docs as authoritative —
  mirror their semantics, don't reinvent them.
- Prefer `kde_tools.compute_kde` + `kde_peaks_above_below` over hand-rolled
  histogram/convolution/peak code; only **add** to `kde_tools` for the missing
  peak-width capability, and document it.
- Keep every helper pure-ish: take `data`, mutate/return `data`; no globals except
  the `candle_cache` module-level store.
- Honor the local-cache-only rule: pipeline functions must never hit the network.
- Keep docstrings short and complete (writing-style rule).
- Add `requirements.txt` to each new `.py` folder (`candle_cache`, strategy module)
  and the notebook's `%pip install` cell.
