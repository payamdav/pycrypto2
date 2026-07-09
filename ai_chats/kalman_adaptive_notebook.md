# Spec: Kalman adaptive-smoothing test notebook

## 1. Task summary

Create a new Colab-runnable test notebook,
`notebooks/tests/kalman_adaptive_smoothing.ipynb`, that: loads candles for an
asset/date-range (local file cache + in-memory cache), computes a rolling
stddev/variance of `vwap` and prints their statistics, runs both
`kalman_1d_batch` (`smoothed`) and `kalman_1d_batch_adaptive`
(`smoothed_adaptive`) over `vwap`, and draws one chart with candlesticks +
`vwap` + `smoothed` + `smoothed_adaptive`, x-axis bounded to the requested
date range.

## 2. Background and context

This is a fresh, standalone test notebook — not part of the `lbla_n_vp`
strategy. It exercises `kalman_1d_batch_adaptive`
(`packages/kalman_filter/kalman_fast.py`, added in a prior task; see
`agents/packages/kalman_filter.md`) side-by-side with the existing
`kalman_1d_batch`, directly on raw candle `vwap`, on a plain OHLC candlestick
chart (no volume-profile/peak machinery from `lbla_n_vp`).

**Existing conventions confirmed by inspecting the repo:**

- `strategies/lbla_n_vp/lbla_n_vp.ipynb` is the only `.ipynb` in the repo and
  is the concrete Colab-notebook precedent: cell 0 is `%pip install ...`,
  cell 1 is the repo-clone-into-`sys.path` cell
  (`REPO_URL = "https://github.com/payamdav/pycrypto2.git"`,
  `REPO_NAME = "pycrypto2"` — note: the actual repo name used in the real
  notebook is `"pycrypto2"`, not the generic `"pycrypto"` example in
  `agents/general/rules.md`; follow the real notebook, not the rules-doc
  example), then an `asset = "btcusdt"` cell, then a candle-preload cell:

  ```python
  import time
  from packages.tools.candle_preloader import preload_candles
  from packages.tools.candle_cache import preload_asset_candles

  t0 = time.perf_counter()
  preload_candles([asset])          # file cache (download if missing)
  preload_asset_candles(asset)      # in-memory cache
  t_preload = time.perf_counter() - t0
  print(f"\nCandle pre-load total: {t_preload:.3f}s")
  ```
  All 8 cells in that notebook are code cells — no markdown header cells are
  used in the existing style.

- `packages/tools/candle_preloader/preloader.py`:
  `preload_candles(assets: list[str] | None = None, start: str | None = None, end: str | None = None, data_dir=None) -> dict[str, Path]`.
  Downloads only the needed months from HuggingFace, filters to
  `[start, end]` inclusive, writes one merged parquet per asset under
  `CWD/data/`, and is a cache-hit no-op on repeat calls with the same
  `(asset, start, end)`. **This is the "load candles and save that locally"
  step** — pass `start=date_from, end=date_to` (unlike the `lbla_n_vp`
  example above, which calls it with no date bounds).

- `packages/tools/candle_cache` (`agents/packages/candle_cache.md`):
  `preload_asset_candles(asset)` loads **all** `{asset}_1m_*.parquet` files
  found under `CWD/` and `CWD/data/` into an in-memory dict (concatenated,
  deduped, sorted by `ts`) — this is **"cache that in memory"**.
  `get_cached_candles(asset) -> dict` returns
  `{"ts": int64[...], "o":..., "h":..., "l":..., "c":..., "v":..., "q":...,
  "n":..., "vwap":..., "vb":..., "vs":..., "_ts_start":..., "_ts_step":...,
  "_len":...}`, all OHLCV/vwap arrays `float64` except `ts` (`int64`).

  **Important subtlety:** because `preload_asset_candles` unions **every**
  matching local parquet file for the asset (not just the one just
  downloaded), if a prior run cached a different/wider date range for the
  same asset, `get_cached_candles` can return more rows than
  `[date_from, date_to]`. The notebook must explicitly slice the returned
  arrays down to `[date_from, date_to]` by `ts` before doing anything else,
  to satisfy "the chart must be bounded to datefrom and dateto" reliably.

- `packages/indicators/rolling_mean_stddev.py`:
  `rolling_mean_stddev(array, window=60) -> np.ndarray` shape `(n, 2)`,
  `out[:, 0]` = rolling mean, `out[:, 1]` = rolling population stddev, causal
  left look-back window, **no padding** (every index gets a real value).

- `packages/kalman_filter/kalman_fast.py`:
  - `kalman_1d_batch(measurements, initial_estimate, initial_error_cov, process_variance, measurement_variance) -> (estimates, error_covariances)`,
    both shape `(N,)`. `measurement_variance` here is a single **fixed
    scalar** — this is the "process_variance and measurement_variance for
    smoothed" the task refers to.
  - `kalman_1d_batch_adaptive(measurements, process_variance, window) -> (estimates, error_covariances)`,
    both shape `(N,)`. Derives its own per-index measurement variance
    internally from `rolling_mean_stddev` — this is why the task asks for
    only **`process_variance`** (no separate `measurement_variance`) for
    `smoothed_adaptive`.
  - Precedent for `initial_error_cov` when seeding `kalman_1d_batch` from a
    raw series: `ai_chats/draw_chart_vp_continued_width_kalman.md` used
    `initial_error_cov = 1.0` (a fixed constant, not derived from data) when
    seeding 1D from `lb_pnc[0]`. This notebook follows the same precedent for
    `smoothed`'s `initial_estimate = vwap[0]`, `initial_error_cov = 1.0`.

## 3. Relevant conventions from `/agents/`

- **File placement** (`agents/general/paths_and_files.md`): "create a test
  notebook" → `notebooks/tests/<meaningful_name>.ipynb`.
- **Colab repo-clone pattern** (`agents/general/rules.md`) — see §2 above for
  the actual repo name to use (`pycrypto2`).
- **`%pip install`** (`agents/general/rules.md`): first cell installs every
  package the notebook needs, before any import.
- **Package docs are read-only inputs here** — this task does not modify
  `packages/` or `agents/packages/*.md`; it only *consumes* the existing
  `kalman_filter`, `indicators`, `candle_preloader`, and `candle_cache`
  packages.
- **Testing / debugging** (`agents/general/access.md`): no test assertions
  required beyond the notebook running end-to-end and rendering the chart;
  do not add a formal test suite.
- **Writing style** (`agents/general/rules.md`): keep comments/markdown
  terse.

## 4. Functional requirements

Notebook path: `notebooks/tests/kalman_adaptive_smoothing.ipynb`. Cell-by-cell
(all code cells, matching the existing repo's notebook style):

1. **`%pip install`**: `numpy pandas numba pyarrow huggingface-hub requests
   plotly` (everything actually imported later; no `scipy`/`sklearn` — not
   used by any package this notebook touches).
2. **Repo clone into `sys.path`**: identical pattern to
   `strategies/lbla_n_vp/lbla_n_vp.ipynb` cell 1, `REPO_NAME = "pycrypto2"`.
3. **Parameters with defaults**:
   ```python
   asset     = "btcusdt"
   date_from = "2026-06-01"
   date_to   = "2026-06-07"
   ```
4. **Load candles (file cache + in-memory cache)**:
   ```python
   from packages.tools.candle_preloader import preload_candles
   from packages.tools.candle_cache import preload_asset_candles, get_cached_candles

   preload_candles([asset], start=date_from, end=date_to)   # saved locally
   preload_asset_candles(asset)                              # cached in memory
   candles = get_cached_candles(asset)
   ```
   Then **slice to `[date_from, date_to]`** (defensive against a wider
   pre-existing in-memory union — see §2):
   ```python
   from datetime import datetime, timezone
   start_ts = int(datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc).timestamp() * 1000)
   end_ts   = int(datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc).timestamp() * 1000) + 86_400_000

   mask = (candles["ts"] >= start_ts) & (candles["ts"] < end_ts)
   ts   = candles["ts"][mask]
   o, h, l, c, vwap = (candles[k][mask] for k in ("o", "h", "l", "c", "vwap"))
   ```
   (The `+ 86_400_000` / `<` pairing makes `date_to` inclusive of its full
   day, matching `load_range`'s convention in
   `agents/datasets/huggingface_candles.md`.)
5. **Window + rolling stddev/variance + statistics summary**:
   ```python
   import pandas as pd
   from packages.indicators.rolling_mean_stddev import rolling_mean_stddev

   window = 60
   mean_std = rolling_mean_stddev(vwap, window)
   stddev   = mean_std[:, 1]
   variance = stddev ** 2

   print(pd.Series(stddev, name="rolling_stddev").describe())
   print(pd.Series(variance, name="rolling_variance").describe())
   ```
6. **Kalman parameters with defaults**:
   ```python
   process_variance            = 1e-4   # smoothed (kalman_1d_batch) — Q
   measurement_variance        = 1.0    # smoothed (kalman_1d_batch) — R
   process_variance_adaptive   = 1e-4   # smoothed_adaptive (kalman_1d_batch_adaptive) — Q
   ```
7. **Run both filters over `vwap`**:
   ```python
   from packages.kalman_filter import kalman_1d_batch, kalman_1d_batch_adaptive

   smoothed, _ = kalman_1d_batch(
       vwap, vwap[0], 1.0, process_variance, measurement_variance
   )
   smoothed_adaptive, _ = kalman_1d_batch_adaptive(
       vwap, process_variance_adaptive, window
   )
   ```
8. **Chart**: one `plotly` figure with:
   - `go.Candlestick(x=dt, open=o, high=h, low=l, close=c, name=asset)`
   - `go.Scatter(x=dt, y=vwap, name="vwap")`
   - `go.Scatter(x=dt, y=smoothed, name="smoothed")`
   - `go.Scatter(x=dt, y=smoothed_adaptive, name="smoothed_adaptive")`

   where `dt = pd.to_datetime(ts, unit="ms", utc=True)`. X-axis explicitly
   bounded: `fig.update_xaxes(range=[date_from, date_to])`. Styling for a
   clean/beautiful result: a clean template (e.g. `template="plotly_white"`
   or `"plotly_dark"`), title including `asset`/date range, `hovermode="x
   unified"`, legible line widths/colors distinguishing the 3 overlay lines
   from the candlesticks, disabled range-slider
   (`xaxis_rangeslider_visible=False`) since the range is already fixed and
   a slider adds clutter, sensible `height`/`margin`. `fig.show()` at the
   end.

## 5. Non-goals / out of scope

- No changes to any file under `packages/` or `agents/`.
- No volume-profile/peaks/KDE machinery from `lbla_n_vp` — this is a plain
  OHLC + line-overlay chart only.
- No 2D/3D Kalman models — only `kalman_1d_batch` /
  `kalman_1d_batch_adaptive`.
- No automated test assertions/CI wiring — a notebook that runs top-to-bottom
  and renders the chart satisfies the task.
- No new reusable charting function in `packages/` — the chart is built
  inline in the notebook's last cell.

## 6. Assumptions

- **Default date range**: `2026-06-01` → `2026-06-07` (one week), chosen to
  be recent, within the dataset's `2024-01 → present` coverage, and small
  enough to load/render quickly in Colab. Any valid range works since all
  three params are notebook inputs.
- **"pandas info" → `.describe()`**: the task says "prints the statistics
  summary of each, something like pandas info." `.info()` itself only
  reports dtype/non-null counts (not useful for two full float64 arrays with
  no nulls); `pd.Series(...).describe()` gives the actual statistics summary
  (count/mean/std/min/quartiles/max) the task is asking for, so `.describe()`
  is used instead of `.info()` literally.
- **`window` default `60`**: no window was specified in the task; `60` is
  reused as an unremarkable, conventional default already used as the
  default in `rolling_mean_stddev` and every other `indicators` function.
- **`kalman_1d_batch` seeding**: `initial_estimate = vwap[0]`,
  `initial_error_cov = 1.0` — the `1.0` follows the existing repo precedent
  in `ai_chats/draw_chart_vp_continued_width_kalman.md` (§2) rather than
  reusing `measurement_variance` or the rolling variance, since no such
  linkage was requested.
- **No markdown header cells** — matches the one real precedent notebook in
  the repo (all-code-cells style); can be added trivially if the user wants
  section headers, but not assumed here. That precedent instead uses
  `# ── Section name ──` comment banners inside parameter-setting cells
  (e.g. its `# ── Parameters (edit here) ──` cell) — reuse that same
  comment-banner style for this notebook's parameter cells (§4 steps 3 and 6)
  instead of markdown.
- **`vwap` is the sole "measurements" input** for both filters, per the
  task's explicit "for both the measurements is vwap that is coming from
  candles."
- Non-empty candle result for the given `(asset, date_from, date_to)` is
  assumed (no defensive empty-range handling), consistent with
  `agents/general/access.md` (no requested defensive validation) and with
  every other function/notebook in this repo not guarding empty input.

## 7. Acceptance criteria

- `notebooks/tests/kalman_adaptive_smoothing.ipynb` exists, valid `.ipynb`
  JSON, runs top-to-bottom in a fresh Colab-like environment (starting from
  `%pip install` and repo clone) without manual intervention.
- Cell order/content matches §4 exactly: pip install → repo clone → asset/
  date params → candle load (file cache + in-memory cache) + range slice →
  window + rolling stddev/variance + `describe()` prints → Kalman params →
  `smoothed`/`smoothed_adaptive` computation → chart.
- `smoothed` comes from `kalman_1d_batch(vwap, vwap[0], 1.0, process_variance, measurement_variance)`;
  `smoothed_adaptive` comes from
  `kalman_1d_batch_adaptive(vwap, process_variance_adaptive, window)`; both
  read `vwap` as `measurements`.
  `len(smoothed) == len(smoothed_adaptive) == len(vwap) == len(candles sliced to date range)`.
  no padding.
- Rolling `stddev`/`variance` arrays are computed via `rolling_mean_stddev`
  over the same `vwap`/`window`, and `pd.Series(...).describe()` is printed
  for each.
  candle data is sliced to `[date_from, date_to]` before any computation, so
  chart and computed series both cover exactly that range.
- The rendered chart contains exactly 4 series — candlesticks, `vwap`,
  `smoothed`, `smoothed_adaptive` — with a legend distinguishing them, an
  x-axis range bounded to `[date_from, date_to]`, and a clean/legible layout
  (template, title, no cluttering range-slider).

## 8. Open questions

1. **Default date range** — assumed `2026-06-01`→`2026-06-07` (§6). Say so if
   a different default window/asset is wanted.
2. **`.describe()` vs `.info()`** — assumed `.describe()` is the intended
   "statistics summary" (§6). Confirm if literal `pandas.Series.info()`
   output is actually wanted instead (it would be far less informative here).
3. **`kalman_1d_batch`'s `initial_error_cov`** — assumed fixed `1.0` per
   existing repo precedent (§2, §6), not derived from `measurement_variance`
   or the rolling variance. Say so if a different seeding is intended.
4. **Chart template/colors** — "well done and beautiful" is not pinned to a
   specific plotly template; the coding agent picks one clean built-in
   template (e.g. `plotly_white`) and clearly distinguishable colors. No
   further design system is implied.

## 9. Notes for the downstream coding agent

- Read all of `/agents/` first, and read
  `strategies/lbla_n_vp/lbla_n_vp.ipynb` directly (via `json.load`) as the
  concrete style precedent for the pip-install/repo-clone cells before
  writing this notebook's JSON.
- Build the `.ipynb` JSON directly (e.g. via `nbformat` or a hand-built cell
  list) rather than trying to author it as a `.py` file — the deliverable is
  a real notebook file.
- Import order per cell exactly as listed in §4 keeps each cell
  self-contained and re-runnable independently after the earlier cells have
  executed once, matching the existing notebook's incremental style.
- Double-check the `ts` dtype from `get_cached_candles` is `int64` millisecond
  epoch before the boundary-mask arithmetic in step 4.
- `kalman_1d_batch` and `kalman_1d_batch_adaptive` are both `@njit` — first
  call triggers JIT compilation; this is expected and fine in Colab.
- Keep all comments/markdown minimal per the writing-style rule; no
  docstrings needed inside a notebook.
