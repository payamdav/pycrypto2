# Spec: volume_profile — generalized KDE volume profile package + POC overlays for bowl/dome tests + Colab notebook

Implementation-ready spec for `code_writer`. Source: user request "KDE PACKAGE GENERALIZE" + `packages/kde_tools` as the template being generalized. Self-contained, but read the referenced files before coding.

## 1. Task Summary

1. New package `packages/volume_profile/`: volume-profile toolkit in **raw price space**. Histogram/KDE range is derived from the current price (last element of `prices`) ± `bps_range` basis points — no `range_min`/`range_max`/`ignore_borders` parameters. Adds volume-profile peak analysis: `point_of_control` (POC + Value Area), `kde_peaks_above_below` split at current price, and `recursive_poc` (iterative POC extraction by Value-Area removal → ranked list of important price levels).
2. Extend **both** test scripts `scripts/tests/rising_from_bowl/` and `scripts/tests/falling_from_dome/`: add volume-profile keys to their `config.json`, compute `recursive_poc`, draw POCs as horizontal lines (distinct colors) on the existing charts, add a table (rank, price, volume, VA) under the chart. Refactor each script into `analyze(cfg)` / `build_figure(res, cfg)` / `run(cfg)` so a notebook can reuse them; CLI behavior unchanged.
3. New Colab test notebook `notebooks/tests/volume_profile/volume_profile_bowl_dome_colab.ipynb`: one symmetric config cell (bowl/dome parameter pairs entered once + volume-profile params), one cell for the bowl chart + POCs, one cell for the dome chart + POCs.
4. New package doc `agents/packages/volume_profile.md` (mandatory per `agents/general/rules.md`).

## 2. Background & Context

- Template: `packages/kde_tools/` (`kernels.py`, `histogram.py`, `kde.py`, `peaks.py`) — volume-weighted histogram → kernel convolution → scipy peak finding, but in **normalized [-1, 1] space** where current price = 0.0. The new package generalizes this to raw prices: current price = `prices[-1]`, range = `current_price * (1 ± bps_range/1e4)`, prices outside the range ignored.
- Reuse, don't duplicate (per `agents/general/strategy_study_guidelines.md` "Code Reuse"): `make_kernel`, `convolve_same`, the jitted `weighted_histogram` core, and `top_kde_peaks` already exist in `kde_tools` and are space-agnostic. `volume_profile` imports/re-exports them. Cross-package import precedent: `kalman_filter` → `packages.indicators`. `kde_tools` itself is **not modified**.
- Volume-profile domain: POC = highest-volume price bin. Value Area = price band around POC holding `va_pct` % (default 70) of profile volume, built by greedy expansion from the POC. VAL/VAH = Value Area Low/High. `recursive_poc` finds the strongest level, removes its Value Area (relative to the current price, per the user's rules below), and repeats on the remaining profile — producing ranked support/resistance levels.
- Test-script template behavior to preserve: `scripts/tests/rising_from_bowl/rising_from_bowl_test.py` and `scripts/tests/falling_from_dome/falling_from_dome_test.py` (config json → padded candle load → sanitized (rolling) vwap → scan → dedupe → stats prints → Plotly HTML). The two scripts stay mirror twins; apply every change to both symmetrically.
- Notebook conventions: `agents/general/rules.md` (`%pip install` first cell, repo clone cell — copy the exact pattern from `notebooks/tests/kalman_adaptive_smoothing.ipynb`, which clones `https://github.com/payamdav/pycrypto2.git`).

## 3. Relevant Conventions from `/agents/`

- `agents/general/rules.md` — `requirements.txt` per script folder / package; `agents/packages/` doc for every package; notebooks self-contained (`%pip install` + clone cell); terse writing everywhere.
- `agents/packages/indicators.md` — numba contract: float64 in/out, explicit loops inside `@njit`.
- `agents/general/strategy_study_guidelines.md` — reuse `packages/` first; numba only where it pays; per-step elapsed prints; price = vwap; UTC `"YYYY-MM-DD HH:MM:SS"`.
- `agents/general/paths_and_files.md` — package → `packages/`, test scripts → `scripts/tests/`, test notebook → `notebooks/tests/` (sub-folder `volume_profile/`).
- `agents/packages/kde_tools.md`, `agents/packages/pattern_detection.md`, `agents/packages/candle_loader.md` — APIs used here.

## 4. Functional Requirements

### 4.1 Package `packages/volume_profile/`

Files: `__init__.py`, `histogram.py`, `kde.py`, `peaks.py`, `requirements.txt` (`numpy`, `numba`, `scipy`).

`__init__.py` exports: `make_kernel` (re-export of `packages.kde_tools.kernels.make_kernel`), `weighted_histogram`, `compute_kde`, `point_of_control`, `top_kde_peaks` (re-export of `packages.kde_tools.peaks.top_kde_peaks`), `kde_peaks_above_below`, `recursive_poc`.

All public functions: cast `prices`/`volumes`/`kde`/`bin_centers` inputs with `np.ascontiguousarray(np.asarray(x, dtype=np.float64))` before use.

#### 4.1.1 `weighted_histogram` (`histogram.py`)

```python
def weighted_histogram(
    prices: np.ndarray,      # 1D float64, oldest → newest; prices[-1] = current price
    volumes: np.ndarray,     # weights, same length
    bins: int = 200,
    bps_range: float = 100.0,  # half-range in basis points around current price (100 = ±1 %)
) -> dict
```

- `current_price = prices[-1]`; `range_min = current_price * (1 - bps_range / 1e4)`; `range_max = current_price * (1 + bps_range / 1e4)`.
- Binning: reuse the jitted `packages.kde_tools.histogram.weighted_histogram(prices, volumes, bins, range_min, range_max, ignore_borders=False)` — inclusive borders, prices outside `[range_min, range_max]` skipped, `v == range_max` clamped to bin `bins - 1`. This is the only "border" behavior; there is no `ignore_borders` option in this package.
- `bin_width = (range_max - range_min) / bins`; `bin_centers = range_min + (arange(bins) + 0.5) * bin_width` (raw price units).
- `n_excluded = int(np.sum((prices < range_min) | (prices > range_max)))`.
- Validation (`ValueError`): `prices` 1D and non-empty; `volumes` same shape; `bins >= 1`; `bps_range > 0`; `current_price` finite and `> 0`.

Returns `{"counts", "bin_centers", "bin_width", "current_price", "range_min", "range_max", "n_excluded"}` — `counts` float64 `(bins,)`.

#### 4.1.2 `compute_kde` (`kde.py`)

```python
def compute_kde(
    prices: np.ndarray,
    volumes: np.ndarray,
    bins: int = 200,
    bps_range: float = 100.0,
    kernel_type: str = "Triangular",
    bandwidth: int = 5,
) -> dict
```

- Calls `weighted_histogram(prices, volumes, bins, bps_range)`, then `kernel = make_kernel(kernel_type, bandwidth)` and `kde = convolve_same(counts, kernel)` (both reused from `kde_tools`).
- Returns the histogram dict plus `{"kde", "kernel"}`:
  `{"kde", "counts", "bin_centers", "bin_width", "kernel", "current_price", "range_min", "range_max", "n_excluded"}`.

#### 4.1.3 `point_of_control` (`peaks.py`)

```python
def point_of_control(
    kde: np.ndarray,          # (bins,) from compute_kde
    bin_centers: np.ndarray,  # (bins,) aligned prices
    va_pct: float = 70.0,     # Value Area percentage of total profile volume
) -> dict | None
```

- `None` when `kde` is empty or `kde.sum() <= 0` (empty profile).
- `poc_idx = int(np.argmax(kde))` (first max on ties); `poc_price = bin_centers[poc_idx]`; `poc_volume = kde[poc_idx]`.
- Value Area — greedy single-bin expansion: `target = va_pct / 100 * kde.sum()`; `acc = kde[poc_idx]`; `lo = hi = poc_idx`. While `acc < target` and at least one side can expand: `below = kde[lo-1]` if `lo > 0` else blocked; `above = kde[hi+1]` if `hi < bins-1` else blocked; take the larger side (tie → above); add its value to `acc`, move that pointer. Stop when `acc >= target` or both sides blocked.
- `val = bin_centers[lo]` (Value Area Low), `vah = bin_centers[hi]` (Value Area High).
- Validation (`ValueError`): `kde`/`bin_centers` same length; `0 < va_pct <= 100`.

Returns `{"poc_idx", "poc_price", "poc_volume", "val_idx", "vah_idx", "val", "vah", "va_volume", "total_volume"}` (`va_volume` = final `acc`; idx fields Python int, the rest float).

Implement the expansion as a private helper `_value_area(kde, removed, poc_idx, target)` (`removed` = bool mask blocking expansion; expansion stops at removed bins exactly like edges) so `point_of_control` (all-False mask) and `recursive_poc` share one code path. Plain numpy/python — loops are over ≤ `bins` items, numba adds no benefit here.

#### 4.1.4 `top_kde_peaks`

Re-export of `packages.kde_tools.peaks.top_kde_peaks` unchanged (scipy `find_peaks`/`peak_prominences`, top-n by `"prominence"`|`"height"`) — it is already price-space-agnostic.

#### 4.1.5 `kde_peaks_above_below` (`peaks.py`)

```python
def kde_peaks_above_below(
    kde: np.ndarray,
    bin_centers: np.ndarray,
    current_price: float,     # split point — required, raw price (from compute_kde)
    distance: float = 5,
    n: int = 3,
    top_identifier: str = "prominence",
) -> dict
```

Delegates to `packages.kde_tools.peaks.kde_peaks_above_below(kde, bin_centers, distance, n, split_at=current_price, top_identifier=top_identifier)`. Same return dict `{"above_prices", "above_proms", "below_prices", "below_proms"}` — "above"/"below" are relative to the current price; prices in raw units.

#### 4.1.6 `recursive_poc` (`peaks.py`)

```python
def recursive_poc(
    kde: np.ndarray,
    bin_centers: np.ndarray,
    current_price: float,
    va_pct: float = 70.0,
    min_poc_volume_ratio: float = 0.1,  # stop when next POC volume < ratio * first POC volume
    max_pocs: int | None = None,        # optional cap; None = unlimited
) -> list[dict]
```

Works on a **copy** of `kde` plus a bool mask `removed` (initially all False). Loop (rank = 1, 2, …):

1. Candidates = unremoved bins. If none → stop ("all bars removed").
2. `poc_idx` = argmax of `kde` over unremoved bins (first on ties); `poc_volume = kde[poc_idx]`. If `poc_volume <= 0` → stop. If `rank > 1` and `poc_volume < min_poc_volume_ratio * first_poc_volume` → stop ("no comparable volume").
3. Value Area via `_value_area(kde, removed, poc_idx, target)` with `target = va_pct / 100 * kde[~removed].sum()` (percentage of the volume remaining at this iteration). Expansion never enters removed bins; if both sides get blocked before reaching `target`, the VA is the accumulated contiguous segment. → `lo`, `hi`, `val = bin_centers[lo]`, `vah = bin_centers[hi]`, `va_volume`.
4. Append `{"rank", "poc_idx", "poc_price", "poc_volume", "val_idx", "vah_idx", "val", "vah", "va_volume"}`.
5. Removal (index form of the user's price rules; identical because `bin_centers` is strictly increasing):
   - if `val > current_price` → `removed[lo:] = True` (drop every bin priced ≥ VAL);
   - elif `vah < current_price` → `removed[:hi + 1] = True` (drop every bin priced ≤ VAH);
   - else (current price inside `[val, vah]`) → `removed[lo:hi + 1] = True`.
   The POC bin always lies inside the removed span, so every iteration removes ≥ 1 bin → guaranteed termination ≤ `bins` iterations.
6. If `max_pocs` reached → stop.

Validation (`ValueError`): same length checks; `0 < va_pct <= 100`; `0 <= min_poc_volume_ratio <= 1`; `max_pocs` None or `>= 1`. Empty profile → `[]`.

Property (document + use as test): entry 1 of `recursive_poc` equals `point_of_control(kde, bin_centers, va_pct)` (before any removal). POC volumes are non-increasing with rank, so rank order = strength order.

### 4.2 Test scripts — `scripts/tests/rising_from_bowl/` and `scripts/tests/falling_from_dome/`

Apply identically (mirror-twin convention) to both scripts. Detector logic, dedupe, stats, and existing chart content unchanged.

#### Config additions (both `config.json` files; existing keys untouched)

```json
"vp_lookback": 1440,
"vp_bins": 200,
"vp_bps_range": 100.0,
"vp_kernel_type": "Triangular",
"vp_bandwidth": 5,
"vp_va_pct": 70.0,
"vp_min_poc_volume_ratio": 0.1
```

All read with these defaults when absent (`cfg.get`), so old configs keep working. `vp_lookback` = number of candles in the profile window, counted back from the end of the displayed range.

#### Volume-profile computation (new step in each script, after dedupe, before chart; one elapsed print)

- `vp_lo = max(start_idx, end_idx - vp_lookback)`; profile slice = `[vp_lo : end_idx]` (clamped to the displayed range; if shorter than `vp_lookback`, use what exists).
- `vp_prices = vwap[vp_lo:end_idx]` (the already-sanitized vwap series), `vp_volumes = data[vp_lo:end_idx, 5]` (candle base volume `v`), current price = `vp_prices[-1]`.
- `vp = compute_kde(vp_prices, vp_volumes, vp_bins, vp_bps_range, vp_kernel_type, vp_bandwidth)`; `pocs = recursive_poc(vp["kde"], vp["bin_centers"], vp["current_price"], vp_va_pct, vp_min_poc_volume_ratio)`.
- Prints (terse): `volume profile: {n} candles | current {price} | range [{range_min}, {range_max}] | excluded {n_excluded}` then one line per POC: `POC {rank}: price {poc_price} vol {poc_volume} VA [{val}, {vah}]`. Zero POCs → single `no POCs found` line, everything still runs.

#### Chart additions

- POC horizontal lines: one `go.Scatter` per POC — `x = [times[start_idx], times[end_idx - 1]]`, `y = [poc_price, poc_price]`, `mode="lines"`, `dash="dash"`, `width 1.6`, color = `QUALITATIVE_PALETTE[(rank - 1) % len]`, `name=f"POC {rank}"`, `legendgroup=f"poc{rank}"`, hovertemplate showing rank, price, kde volume, volume % of POC 1, VAL, VAH. Dashed style keeps them distinguishable from the dotted bowl/dome parabolas.
- POC table: convert the figure to `plotly.subplots.make_subplots(rows=2, cols=1, row_heights=[0.78, 0.22], vertical_spacing=0.06, specs=[[{"type": "xy"}], [{"type": "table"}]])`. Row 1 = existing chart content (candles, vwap, patterns, POC lines); row 2 = `go.Table` with columns `Rank | Price | KDE Volume | % of POC 1 | VAL | VAH`, one row per POC, Rank cell font colored with the POC's line color. Total `height=1000`. Keep title, sidebar legend, spikes (apply spike/grid axis styling to the xy subplot axes), `scrollZoom`. Zero POCs → skip lines and table row content gracefully (empty table or omit table, keep script green).

#### Refactor for notebook reuse (both scripts, same shape)

- `analyze(cfg) -> dict` — everything up to and including the volume profile step (load, vwap, scan, dedupe, stats prints, vp + pocs). Returns at least `{"data", "vwap", "start_idx", "end_idx", "bowls"|"domes", "vp", "pocs"}`.
- `build_figure(res, cfg) -> go.Figure` — full figure (chart + POC lines + table); no file I/O.
- `run(cfg)` — `analyze` → `build_figure` → `fig.write_html(out_path, config={"scrollZoom": True})` → path print. `main()`/CLI unchanged: `python3 <script>.py [config_path]` behaves as before plus the new POC output.
- Add `scipy` to both scripts' `requirements.txt` (now: numpy, numba, pandas, plotly, scipy).

### 4.3 Notebook `notebooks/tests/volume_profile/volume_profile_bowl_dome_colab.ipynb`

Runnable top-to-bottom on Google Colab. Cells:

1. **Markdown** — title + one-line purpose (bowl/dome detection charts with recursive-POC volume-profile levels).
2. **`%pip install numpy numba scipy duckdb pandas plotly`** (covers candle_loader's duckdb; per `agents/general/rules.md`).
3. **Clone cell** — exact pattern from `notebooks/tests/kalman_adaptive_smoothing.ipynb` (`REPO_URL = "https://github.com/payamdav/pycrypto2.git"`, `REPO_NAME = "pycrypto2"`, clone if missing, append `REPO_PATH` to `sys.path`).
4. **Config cell** — single dict, symmetric parameters entered **once** and fanned out to both detectors:

```python
config = {
    # general
    "asset": "btcusdt",
    "date_from": "2026-04-15 00:00:00",
    "date_to": "2026-04-21 23:59:00",
    "vwap_period": 1,
    # symmetric pattern params (bowl ↔ dome pairs share one value)
    "min_pattern_width": 10,        # min_bowl_width  ↔ min_dome_width
    "max_pattern_width": 120,       # max_bowl_width  ↔ max_dome_width
    "min_pattern_extent_bps": 20.0, # min_bowl_depth_bps ↔ min_dome_height_bps
    "extremum_position_limit": 0.8, # bottom_position_limit ↔ top_position_limit
    "wall_retrace_limit_bps": 15.0, # peak_drawdown_limit_bps ↔ trough_rally_limit_bps
    "max_wall_search_width": 240,   # max_peak_search_width ↔ max_trough_search_width
    # volume profile
    "vp_lookback": 1440, "vp_bins": 200, "vp_bps_range": 100.0,
    "vp_kernel_type": "Triangular", "vp_bandwidth": 5,
    "vp_va_pct": 70.0, "vp_min_poc_volume_ratio": 0.1,
}
```

Then build `bowl_cfg` / `dome_cfg` dicts in the same cell by mapping the shared keys to each detector's native key names (general + `vp_*` keys copied through).

5. **Bowl cell** — load `scripts/tests/rising_from_bowl/rising_from_bowl_test.py` via `importlib.util.spec_from_file_location` (script folders are not packages), then `res = mod.analyze(bowl_cfg)`, `fig = mod.build_figure(res, bowl_cfg)`, `fig.show()` (inline chart + POC lines + table).
6. **Dome cell** — same with `scripts/tests/falling_from_dome/falling_from_dome_test.py` and `dome_cfg`.
7. **Markdown** — what to verify by eye: detected patterns, dashed POC levels near the range end (±`vp_bps_range` bps of the last price), table ranks matching line colors, POC 1 = strongest level.

The scripts' module-level `os.chdir(REPO_ROOT)` keeps the candle cache at `<repo>/data` in Colab — no extra handling needed.

### 4.4 Doc `agents/packages/volume_profile.md`

Follow the structure of `agents/packages/kde_tools.md`, terse: identity table; raw-price-space convention (current price = `prices[-1]`, range = ±`bps_range` bps, out-of-range prices ignored) and the contrast with `kde_tools`' normalized space; re-used vs new functions table; each public function's signature, behavior, return dict; `recursive_poc` algorithm (VA removal rules vs current price, stop conditions, rank = strength); volume-profile glossary line (POC / VAL / VAH / Value Area); end-to-end usage example (candles → vwap+v → `compute_kde` → `recursive_poc`); notes for agents (reuse from kde_tools, kde_tools unchanged, `point_of_control` ≡ first recursive entry).

## 5. Non-Goals / Out of Scope

- No changes to `packages/kde_tools/` (only imported) or `packages/pattern_detection/` code and docs.
- No changes to detector algorithms, dedupe logic, or existing chart traces beyond the POC/table additions and the analyze/build_figure refactor.
- No study framework, tags/pipeline, labels, ML, multi-asset loops, unit-test suite.
- No `kde_peak_widths` port (not requested); `top_kde_peaks`/`kde_peaks_above_below` are provided but not wired into the test scripts (the scripts use `recursive_poc` only, as requested).

## 6. Assumptions

1. Bin/border semantics: prices exactly at `range_min`/`range_max` are **included** (`ignore_borders=False` core); "out of this range is ignored" applies strictly outside. `bps_range` is the half-range: 100 bps → ±1 %.
2. Value Area: greedy **single-bin** expansion from the POC (larger neighbor wins, tie → higher-price side), on the KDE-smoothed profile — the standard VP method adapted to a smoothed histogram; `va_pct` interpreted per iteration as % of the **remaining** profile volume in `recursive_poc`.
3. "No comparable volume to the first poc" is made concrete as `min_poc_volume_ratio` (default 0.1 = 10 % of POC 1's kde volume), configurable; `max_pocs` added as an optional safety cap (None = unlimited).
4. VA expansion cannot cross previously removed bins (a removed span behaves like an array edge); an under-target VA capped by blocked sides is valid.
5. Test scripts: "mix" = extend the two existing scripts in place (not a third script); profile window = last `vp_lookback` candles of the displayed range using the script's sanitized vwap as prices and candle `v` as weights; current price = vwap at the range end.
6. Notebook = new Colab notebook under `notebooks/tests/volume_profile/`, reusing the scripts' `analyze`/`build_figure` via importlib; clone target is `payamdav/pycrypto2` `main` (pattern of the existing test notebook).
7. Returned POC containers are Python lists of dicts (small, human-facing); heavy numeric loops stay in the reused jitted/scipy code.

## 7. Acceptance Criteria

1. `from packages.volume_profile import make_kernel, weighted_histogram, compute_kde, point_of_control, top_kde_peaks, kde_peaks_above_below, recursive_poc` works from repo root; `kde_tools` imports and files untouched (`git status` clean for that folder).
2. `compute_kde` on a synthetic set (e.g. `current_price=100.0`, `bps_range=100.0`) yields `range_min=99.0`, `range_max=101.0`, `bin_centers[0] ≈ 99.005` with 200 bins, `n_excluded` counting exactly the out-of-range prices; kde sums ≈ counts sum (kernel normalized).
3. `point_of_control` on a synthetic unimodal profile returns the peak bin as POC and a VA containing ≥ `va_pct` % of total volume (or the whole array); on all-zero kde → `None`.
4. `recursive_poc` on a synthetic bimodal profile (two separated peaks, current price between them): rank 1 = larger peak, rank 2 = smaller; removal rules verified for all three cases (VA fully above / fully below / straddling current price); volumes non-increasing; entry 1 == `point_of_control` result; terminates with `min_poc_volume_ratio=0.0` (runs to full removal) and respects `max_pocs=1`.
5. Validation `ValueError`s per §4.1 fire (empty prices, length mismatch, `bps_range <= 0`, bad `va_pct`, non-positive current price).
6. Both test scripts run green via CLI with their default configs: existing prints + new volume-profile prints + per-step elapsed lines; HTML written with candles, vwap, patterns, dashed POC lines spanning the displayed range, and the POC table (rank colors matching line colors). Regenerated `rising_from_bowl_btcusdt.html` / `falling_from_dome_btcusdt.html` committed.
7. Old configs without `vp_*` keys still run (defaults applied); zero-detection and zero-POC paths stay clean.
8. `analyze`/`build_figure`/`run` split works: `run(cfg)` output identical to before plus POC additions; importing the module and calling `analyze` + `build_figure` returns a showable figure without writing files.
9. Notebook: valid ipynb JSON; cells in the order of §4.3; config cell defines every symmetric pair exactly once; bowl and dome cells produce inline figures with POC lines + tables. (Colab execution cannot be verified locally — verify the same flow by executing the equivalent code path locally.)
10. `agents/packages/volume_profile.md` created; `requirements.txt` present/updated (package: numpy numba scipy; both script folders: + scipy); all new text terse per `agents/general/rules.md`.

## 8. Open Questions

None blocking — assumptions 2, 3, and 5 are the judgment calls to review if results look off.

## 9. Notes for `code_writer`

- Read `packages/kde_tools/*.py` and both test scripts before writing anything; copy their tone (docstrings, print style, palette, layout constants).
- `histogram.py` is a thin wrapper: range math + validation + delegation to the jitted kde_tools core — do not re-implement the binning loop.
- `_value_area`: single implementation used by both `point_of_control` and `recursive_poc`; blocked = index out of range **or** `removed[idx]`. Keep tie → above deterministic.
- Removal uses index slices (`removed[lo:]`, `removed[:hi+1]`, `removed[lo:hi+1]`) — equivalent to the spec's price comparisons and float-safe.
- In `build_figure`, add the table last so all xy traces keep row 1; candlestick + table require the `specs` argument shown in §4.2.
- Chart POC hover needs `% of POC 1` — compute once from `pocs[0]["poc_volume"]`.
- Run both scripts from their folders (`python3 rising_from_bowl_test.py`, `python3 falling_from_dome_test.py`) — first run downloads the btcusdt candle parquet into `<repo>/data/` (git-ignored, several hundred MB; keep it, do not commit).
- The notebook is JSON — build it programmatically or write carefully; every cell's `source` a list of strings, `metadata`/`nbformat` fields present, no execution outputs required.
