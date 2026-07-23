# traders_indicators

Trader indicators are packages (with optional sub-packages) under `packages/traders_indicators/`. They simulate a single trade per candle and are used for back-test labeling — creating labels for machine learning or studies. New trader indicators are developed **per request**; each request may override or extend the defaults below, but unless it does, everything in this document applies.

Per-indicator documentation: every implemented trader indicator must have its own description file at `agents/packages/traders_indicators/<indicator_name>.md` describing its parameters, output columns, and any special behavior — so agents know how to use it. Creating that file is the duty of `spec_writer`.

---

## Core Semantics

- They are **indicators**: they return one or more properties **per candle**.
- **Input:** a numpy candle array, candles dataset, or ecandles.
- **Output:** numpy array with one column per returned property, **same length as the input** — always, regardless of input kind.
- For each input item, the indicator assumes a trading position is entered at that item (per the indicator's instruction) and writes that single trade's result at the **same index** in the output.
- They do **not** compute an overall/aggregate back-test result — only per-item, per-timestamp trade simulation, for labeling purposes.

## Common Trading Concepts

Each indicator's instruction defines the details; these are the shared defaults and definitions.

### Entry
- **Entry price** — one of:
  - next candle `vwap`
  - current candle close
  - next candle open
  - any other special instruction
- **Entry time** — the timestamp of the candle the entry price comes from (e.g. "next candle vwap" → next candle's `ts`; "current candle close" → current candle's `ts`).

### Back-test prices
Prices used to check exit conditions after entry:
- **Single price** (`vwap`, close, open, …): one price per candle is checked for stop loss / take profit / other exit conditions.
- **`hl`**, or other custom instructions, as requested.
- **`ohlc`**: per upcoming candle, check in this order:
  1. Open — check exit conditions (stop loss, take profit).
  2. High/Low — order depends on position direction: **long** → low checks stop loss first, then high checks take profit; **short** → high checks stop loss first, then low checks take profit.
  3. Close — checked last (normally moot when neither high nor low triggered).
  - Whenever precedence between high and low must be resolved, use O → H → L → C order.

### Exit
- **Exit price** — the price that hits the exit condition per the back-test prices rules above.
- **Exit time** — `ts` of the candle where the exit condition hits.
- **Entry candle is never checked for exit** by default — exit checks start from the candle after entry, unless otherwise instructed.
- **Look-ahead window** — maximum number of candles checked for exit. If the position is still open at the last candle of the window, exit there (time-limit exit).
- **Exit cause** codes (unless otherwise instructed): `1` = take profit, `-1` = stop loss, `0` = time limit.

### Parameters
- **Commission** — default `0` unless otherwise instructed.
- Take profit, stop loss, commission, and similar parameters may be given as absolute numbers or as **bps of entry price**, or as specifically instructed.
- Everything else follows standard trading terms.

## Implementation Notes

- Use numba `@njit` for the trading-simulation loops — strongly recommended for performance.
- By default no aggregate reporting, but an indicator's instruction may request **helper functions** that take the indicator's output plus candles and return back-test metrics.
- Standard package rules apply: doc file per package (`agents/general/rules.md`), `requirements.txt` alongside code, no nan / no zero-padding conventions where applicable.

## Creating a New Trader Indicator

A simple prompt must suffice, e.g.: *"create a trader indicator named `is_long_good` that returns 1.0 for each timestamp that returns 100 bps in 100 minutes and 0 otherwise; use vwap as trading prices and entry price is next candle vwap"* (example only — not to be implemented). Flow:

1. `spec_writer` turns the request into a spec in `ai_chats/`, applying this document's defaults for anything unstated, and creates the indicator's description file under `agents/packages/traders_indicators/`.
2. `code_writer` implements the indicator under `packages/traders_indicators/` from the spec.
