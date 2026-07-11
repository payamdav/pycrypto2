# Spec: Motion Indicators — velocity, acceleration, jerk

## 1. Task Summary

Add a new file to `packages/indicators/` implementing a Numba-JIT motion indicator. Given a `window: int` and a 1D `position` ndarray, it returns an `(n, 3)` float64 ndarray: column 0 = velocity, column 1 = acceleration, column 2 = jerk. The window is a look-back window inclusive of the current item (no future leak). Early indices without enough look-back are backfilled with the first valid value of each column. The function must be `@njit`.

## 2. Background and Context

`packages/indicators/` holds Numba-JIT indicators operating on 1D float64 arrays (see `agents/packages/indicators.md`). Velocity/acceleration/jerk are the 1st/2nd/3rd time derivatives of position, useful as momentum features on price series. Multi-column output precedent: `rolling_median_iqr` and `rolling_mean_stddev` return `(n, 2)` arrays.

## 3. Relevant Conventions from `/agents/`

- `@nb.njit(inline='always')`; explicit `for` loops only — no NumPy high-level calls inside the jitted body.
- Input 1D `np.float64`; output newly allocated `np.float64`.
- `window` defaults to `60`.
- Indices without a full window are backfilled with the first fully-windowed value (per column here).
- `packages/indicators/requirements.txt` already lists `numpy` and `numba` — no change needed.
- Update `agents/packages/indicators.md` to document the new function (mandatory per `agents/general/rules.md`).
- Writing style: all docs/docstrings as short as possible.

## 4. Functional Requirements

### File and API

- New file: `packages/indicators/motion.py`.
- Function: `motion(position, window=60)` decorated `@nb.njit(inline='always')`.
- Export from `packages/indicators/__init__.py`: `from packages.indicators.motion import motion`.

### Computation

Derivatives are window-spaced endpoint differences, normalized to per-step units, cascaded with the same `window`:

```
vel[i]  = (position[i] - position[i - window + 1]) / (window - 1)   valid for i >= window - 1
acc[i]  = (vel[i]      - vel[i - window + 1])      / (window - 1)   valid for i >= 2 * (window - 1)
jerk[i] = (acc[i]      - acc[i - window + 1])      / (window - 1)   valid for i >= 3 * (window - 1)
```

Each stage reads only indices `<= i` of the previous stage, and each earlier stage's value at those indices depends only on positions `<= i` — no future leak anywhere.

### Special case `window == 1`

The divisor `window - 1` is `0`. Define `window == 1` as exact one-step differences: `vel[i] = position[i] - position[i-1]` (valid from `i = 1`), and the same one-step difference cascaded for `acc` (valid from 2) and `jerk` (valid from 3). Implementation hint: use `step = max(window - 1, 1)` as both the offset and divisor; then all formulas above become `x[i] - x[i - step]) / step` with first valid index `k * step` for stage `k = 1, 2, 3`.

### Backfill

Per column: indices before that column's first valid index get the first valid value (`out[i, c] = out[first_valid_c, c]`). If `n` is too short for a column to have any valid index, that column remains all `0.0`.

### Output

- Shape `(n, 3)`, `dtype=np.float64`, `n = len(position)`. `out[:, 0]` velocity, `out[:, 1]` acceleration, `out[:, 2]` jerk.
- Empty input returns shape `(0, 3)`.

## 5. Non-Goals / Out of Scope

- No tests or test notebooks (testing not requested).
- No changes to existing indicator files.
- No multi-asset / 2D-input support; input is a single 1D position series.
- No NaN handling beyond what the formulas produce.

## 6. Assumptions

- "Position" is any 1D float64 series (typically price); time step is uniform (1 unit per index), so derivatives are per-index units.
- Derivative estimator is the endpoint difference over the window (not a regression slope, not a mean of one-step diffs — note the endpoint difference over `window` equals the mean of the `window-1` one-step diffs inside it).
- Acceleration and jerk cascade with the **same** `window` applied to the previous stage's output.
- Backfill is per column, since each column has a different first valid index.

## 7. Acceptance Criteria

1. `motion(position, window)` compiles under `@nb.njit` and returns `(n, 3)` float64.
2. For `i >= 3*(window-1)` all three columns match the formulas above exactly.
3. No output value at index `i` depends on any `position[j]` with `j > i`.
4. For constant input: all columns ≈ 0. For `position = a*i + b`: velocity ≈ `a`, acceleration and jerk ≈ 0 (in fully valid regions).
5. Early indices of each column equal that column's first valid value.
6. `__init__.py` exports `motion`; `agents/packages/indicators.md` documents it in the existing format.
7. Explicit loops only inside the jitted body; output newly allocated.

## 8. Open Questions

None blocking — defaults chosen above. If a regression-slope derivative or a different cascade window is wanted instead, say so before implementation.

## 9. Notes for the Downstream Coding Agent

- Mirror the style of `packages/indicators/rolling_mean_stddev.py` (docstring, loop structure) and `ma.py` (backfill loop).
- Compute stage-by-stage into the three output columns directly; velocity can be written first, then acceleration reads `out[:, 0]`, jerk reads `out[:, 1]` — only backfill each column **after** the next stage has consumed the raw values, or backfill at the very end to keep the cascade exact. Backfilling before cascading would corrupt acceleration/jerk near the start; do the backfill of all three columns as the final step.
- Keep the docstring to the point per the writing-style rule.
