# Strategy Study Guidelines

Framework for building a **study** — a runnable implementation that tests a strategy or market-analysis idea. When a request asks to build or study a strategy, `spec_writer` and `code_writer` must fit it to this framework. Everything below is the default: it fills whatever the request leaves unstated and is a requirement unless the request explicitly overrides it.

---

## Locations

| What | Where |
|---|---|
| Study code | `scripts/studies/{study_name}/` |
| Per-execution files (indicator caches, config, models, metrics, reports, …) | `CWD/data/{study_name}/{tag}/` — git-ignored |

## Tags

A **tag** is one release of a study with a specific set of settings. Default tag: `default`.

- Every runtime script takes the tag as its **first CLI argument** — `python3 build_indicator_cache.py tag1` — and reads/writes all its files under `data/{study_name}/{tag}/`. Create the tag folder if it does not exist.
- Config resolution: look for the config file in the tag folder first; if present use it, otherwise use the one in the study code folder. One study can therefore run with multiple setting sets, each kept in its own tag folder.

## Config & Parameters

- The full pipeline must be runnable as a function that receives **all parameters as one dict**; the config file is just a source for that dict.
- Any value written in study instructions as `param: N` or "default N" is a **config default** the user may change in the config file.
- Config accepts a single asset or a list of assets.

### Defaults

| Item | Default |
|---|---|
| Candles | 1-minute (load via `packages/candle_loader`) |
| Price of a candle | `vwap` |
| Assets | `["btcusdt"]` |
| Indicator cache range | full candle range of the asset |
| Study range | `date_from="2024-01-01 00:00:00"`, `date_to="2026-06-30 23:59:00"` |
| Single-datetime studies (e.g. drawing) | `datetime="2026-04-15 20:00:00"` |
| Datetime strings | `"YYYY-MM-DD HH:MM:SS"`, always UTC |

Candle coverage is `2024-01-01 00:00:00` → `2026-06-30 23:59:00` for most assets (see `agents/datasets/assets.md`).

## Indicator Cache

Building and saving the indicator cache is normally the **first pipeline step**.

- 2D `float64` ndarray with the same length as the candle array. Column 0 = candle `ts`; one column per indicator property. E.g. cache for `[ma_fast, ma_slow, rsi, ind1, ind1_prop2]` → shape `(n_candles, 6)`.
- `float64` holds every needed value kind: float, int, timestamp, logical.
- Save it to disk in the tag folder.

## Indicators, Features, Labels

- Indicator outputs must have exactly the candle-array length — no nan/0 padding; use partial-window calculation or backfill (see `agents/packages/indicators.md`). nan is a bad value anywhere in arrays and must be prevented.
- Indicators/features are look-back only. **Double-check every calculation for future leak.**
- Labels are normally computed with look-ahead windows. They may be stored inside the indicator cache, but the spec must explicitly name which columns are labels.
- Look-ahead windows start at the **next** item — the current item is excluded.
- Label entry prices: trading by vwap → entry = next candle's `vwap`; trading by ohlc → entry = next candle's open.

## Valid Observation Range

Compute the study's longest look-back window and longest look-ahead window across all indicators, features, and labels. **Normal items** are the indices with a complete look-back behind and a complete look-ahead ahead (anchor range per `agents/ideas/idea_look_back_look_ahead.md`). All investigation, testing, and evaluation run on the normal-items set only.

## Multi-Asset & Multi-Run

- Per-asset files in the same tag folder get an `_{asset}` postfix; each asset's files are fully isolated from other assets'. Cumulative cross-asset reports only when the study instructions ask for them.
- Parameters may be given as **ranges with steps** → the pipeline function runs once per parameter combination (**run**), all in the same tag. Per-run files get an additional run-id postfix: `{base}_{asset}_{run_id}.{ext}`.
- A human-readable manifest file (json/csv/yaml) in the tag folder maps each run id to its parameter set.
- Multi-run executions end with a cumulative report comparing all runs.
- Assets and runs are independent of each other — use multiprocessing whenever possible and meaningful.

## Code Reuse & Performance

- Use existing indicators/functions from `packages/` first. If something is missing: standard/reusable → develop it in `packages/` (with its `agents/packages/` doc per `agents/general/rules.md`); study-specific → develop it inside the study folder.
- Develop indicators and calculations as numba `@njit` functions unless numba adds no benefit.
- Every sub-task (each indicator calculation, training, optimizing, load/save, drawing, …) prints its elapsed time. Long-running tasks (e.g. model training) also show progress — a progress bar or live metrics.

## Pipeline, Runbook, File Docs

- `spec_writer` decides whether the study is a single run script or a multi-part pipeline. Multi-part studies also get one script that runs the full pipeline in one step.
- Every study has a markdown runbook in the study folder: how to run it and its pipeline from preparation to evaluation and reports.
- Every file written into a tag folder must be documented — what it is, its columns and their properties — in the runbook or a files doc next to it.
- All docs, console prints, and text everywhere: shortest useful form — quick hints, single-line prints, plain column naming (writing style per `agents/general/rules.md`).

## Evaluation & Reports

Mandatory for every study. If the instructions say nothing about it, `spec_writer` must design useful, standard evaluation and reports.

## Charts

When drawing is requested — even without details — the user wants to inspect a **time slice** of the study: at random points or at a specific time, seeing all candles, indicators, and events around that time with some context before and after, to feel what really happens in the market with this strategy. `spec_writer` designs the chart schema that describes the market and strategy situation best: beautiful, interactive, with zooming and panning, and sub-charts with crosshairs when meaningful.

## Notebook

When requested, also prepare a Jupyter notebook so the study can be executed and investigated remotely; default runtime platform is Google Colab. Follow the notebook rules in `agents/general/rules.md`.
