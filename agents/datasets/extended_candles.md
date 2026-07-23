# Extended Candles (ECandles)

Custom pre-built datasets derived from 1-minute candles. Each **ecandle type** has a unique `ecandle_name` and a fixed column layout defined by its description: some columns copied from candles, others pre-calculated indicators or custom computations. Rows align 1:1 with the asset's candle array (same length, same order).

---

## Identity & Storage

| Key | Value |
|---|---|
| Data file | `CWD/data/extended_candles/{ecandle_name}/{ecandle_name}_{asset_name}.npy` (or `.parquet`) |
| Description file | `CWD/data/extended_candles/{ecandle_name}/description_{ecandle_name}.md` |
| Builder script | `scripts/tools/ecandles_build/{ecandle_name}/build_{ecandle_name}.py` |
| Format | numpy `float64` 2D array by default; parquet only if the ecandle description says so |
| Rows | exactly the asset's full candle range (see `agents/packages/candle_loader.md`) |
| Column 0 | always `ts` — ms epoch, identical to the candle `ts` |
| Files | one file per asset (asset names lowercase, see `assets.md`) |

Example: an ecandle defined as "vwap, volume, close of candles + ma 14 of vwap, rsi 14 of close, average velocity 60 of vwap" → 7 columns: `ts, vwap, v, c, ma14_vwap, rsi14_c, vel60_vwap`.

## Description File

`description_{ecandle_name}.md` lives next to the data files and is the authoritative reference for that ecandle type. It must contain:

- Column list: name + single-line description per column (col 0 = `ts`).
- Storage format (numpy or parquet) and dtype.
- Any note needed to use the files correctly (windows, backfill, units, …).

## Builder Contract

`scripts/tools/ecandles_build/{ecandle_name}/build_{ecandle_name}.py`:

- **CLI:** takes one argument — an asset name or `all`.
- **Importable API (preferred over CLI):**

  ```python
  from scripts.tools.ecandles_build.{ecandle_name}.build_{ecandle_name} import build_ecandle

  build_ecandle()              # all assets that have candles
  build_ecandle("btcusdt")     # one asset
  build_ecandle(["btcusdt", "ethusdt"])  # list of assets
  ```

- **Idempotent:** builds an asset's file only if it does not already exist — always safe to call before loading.
- **Timing:** prints build time per asset and in total (CLI and function alike).
- Any extra python files the builder needs live in the same `scripts/tools/ecandles_build/{ecandle_name}/` folder, with a `requirements.txt` per `agents/general/rules.md`.

## Building Columns

- Candle columns: copy from the array returned by `packages/candle_loader.load_candles`.
- Indicator columns: use `packages/indicators` first (see `agents/packages/indicators.md`).
- Custom/special columns: implement as numba `@njit` functions for performance.
- No nan and no zero-padding anywhere — full candle length via partial-window or backfill, same as the indicators package convention.
- All columns are look-back only unless the ecandle spec explicitly defines a look-ahead (label) column and names it as such.

## Loading

Nothing special — call `build_ecandle` first (no-op when files exist), then load:

```python
import numpy as np
from scripts.tools.ecandles_build.{ecandle_name}.build_{ecandle_name} import build_ecandle

build_ecandle("btcusdt")
data = np.load(f"data/extended_candles/{ecandle_name}/{ecandle_name}_btcusdt.npy")
ts = data[:, 0]  # remaining columns per description_{ecandle_name}.md
```

For parquet ecandles use `pd.read_parquet` instead. Always read the ecandle's description file to map columns.

## Creating a New ECandle Type

When a new ecandle is requested, `spec_writer` produces the implementation spec in `ai_chats/` — column layout, names, formulas, format — **and** creates `description_{ecandle_name}.md`. `code_writer` then implements the builder from that spec.

## Interpreting ECandle Requests

| Request | Action |
|---|---|
| "build ecandle X with columns …" | `spec_writer` spec + description file → `code_writer` implements builder → run it |
| "build ecandles X for all assets" | import and call `build_ecandle()` from X's builder |
| "load ecandles named X …" | call X's `build_ecandle` for the needed assets, load files, use columns per X's description file |
