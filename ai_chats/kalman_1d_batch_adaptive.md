# Spec: `kalman_1d_batch_adaptive` — variance-adaptive 1D Kalman batch filter

## 1. Task summary

Add a new function `kalman_1d_batch_adaptive(measurements, process_variance, window)` to
`packages/kalman_filter/kalman_fast.py`. Unlike `kalman_1d_batch` (fixed scalar
`measurement_variance`), this variant derives a **per-index measurement variance** from a
rolling stddev of `measurements` (via `rolling_mean_stddev`), squares it to variance, and
feeds `variance[k]` into the Kalman step at each index `k`. `process_variance` stays fixed
across all indices. Output must match `kalman_1d_batch`'s return type and length exactly,
with no padding.

## 2. Background and context

`packages/kalman_filter/kalman_fast.py` currently has:
- `kalman_1d_step(measurement, prev_estimate, prev_error_cov, process_variance, measurement_variance)` — one predict/update cycle, `@njit`.
- `kalman_1d_batch(measurements, initial_estimate, initial_error_cov, process_variance, measurement_variance)` — loops `kalman_1d_step` over the array with a **fixed scalar** `measurement_variance`, `@njit`. Returns `(estimates, error_covariances)`, both `float64` arrays shape `(N,)`.

`packages/indicators/rolling_mean_stddev.py` has `rolling_mean_stddev(array, window=60)`,
`@nb.njit(inline='always')`, returning shape `(n, 2)` float64: `out[:, 0]` = rolling mean,
`out[:, 1]` = rolling population stddev, over a **left look-back window**
`array[max(0, i-window+1):i+1]`. Partial early windows are computed over the available
`m = min(i+1, window)` items — **no `0.0` padding**, every index gets a real value
(documented deviation from the indicators package-wide convention; see
`agents/packages/indicators.md`).

Cross-package import precedent already exists:
`packages/tools/metrics_cache/metrics_cache.py` imports directly from
`packages.indicators.rolling_mean_stddev` and `packages.indicators.rolling_robust_z_score`.
This spec follows the same pattern, importing `rolling_mean_stddev` into
`packages/kalman_filter/kalman_fast.py`.

Because `rolling_mean_stddev` never pads, and `kalman_1d_batch_adaptive` runs the Kalman
loop over every index of `measurements`, the "same length as `measurements`, no padding"
requirement is satisfied automatically by iterating the full range — no special-case
handling of a warm-up segment is needed.

## 3. Relevant conventions from `/agents/`

From `agents/packages/kalman_filter.md`, `agents/packages/indicators.md`, and
`agents/general/rules.md`:

- Both `kalman_1d_step` and `kalman_1d_batch` are `@njit` (no `inline='always'` on this
  particular pair, unlike the indicators package). New function should match this file's
  existing convention: `@njit`.
- `rolling_mean_stddev` is `@nb.njit(inline='always')` — calling one `@njit` function from
  another in nopython mode is standard Numba behavior and requires no special handling.
- Arrays are `np.ndarray`, `dtype=np.float64`, newly allocated per call.
- Package docs are authoritative and must be kept current (Rule: "Package
  Documentation") — `agents/packages/kalman_filter.md` must be updated to document the new
  function, mirroring the style of the existing `kalman_1d_batch` entry.
- Writing style: docstrings as short as possible while complete, matching the terse style
  already used in `kalman_fast.py`.
- No `requirements.txt` change needed — `numpy`/`numba` already cover both packages; the
  new cross-package import (`packages.indicators`) is an internal repo import, not an
  external dependency.
- Per `agents/general/access.md`: no tests are required unless explicitly requested (they
  were not); do not add test files.

## 4. Functional requirements

1. **Signature:** `kalman_1d_batch_adaptive(measurements: np.ndarray, process_variance: float, window: int) -> tuple`, decorated `@njit`, added to `packages/kalman_filter/kalman_fast.py`.
2. **Import:** add `from packages.indicators.rolling_mean_stddev import rolling_mean_stddev` at the top of `kalman_fast.py`.
3. **Variance derivation:**
   - `mean_std = rolling_mean_stddev(measurements, window)` → shape `(n, 2)`.
   - `stddev = mean_std[:, 1]`.
   - `variance = stddev ** 2` (element-wise square; population variance implied by
     `rolling_mean_stddev`'s population stddev).
4. **Initial conditions:**
   - `initial_estimate = measurements[0]`.
   - `initial_error_cov = variance[0]`.
5. **Per-index Kalman loop:** for `k` in `range(n)`, call `kalman_1d_step` with
   `measurement = measurements[k]`, the running `(est, cov)` state, the fixed
   `process_variance`, and `measurement_variance = variance[k]`. Seed the loop's running
   state with `initial_estimate` / `initial_error_cov` before the first iteration (mirrors
   `kalman_1d_batch`'s structure, but with `measurement_variance` read per-index from
   `variance` instead of a fixed scalar argument).
6. **Output:** `(estimates, error_covariances)` — two newly allocated `float64` arrays,
   each shape `(n,)` where `n = len(measurements)`, identical in type/shape/semantics to
   `kalman_1d_batch`'s return value. No additional return values (the internal `variance`
   array is not returned).
7. **No padding:** every index `0..n-1` gets a real computed estimate/covariance — this
   falls out naturally from looping the full range and from `rolling_mean_stddev`'s
   no-padding behavior; no extra logic is required to satisfy this.
8. **`window`:** passed through unmodified to `rolling_mean_stddev(measurements, window)`;
   no default value (required positional per the task's parameter list, unlike
   `rolling_mean_stddev`'s own `window=60` default).

## 5. Non-goals / out of scope

- No change to `kalman_1d_step` or `kalman_1d_batch`'s existing signatures or behavior.
- No changes to the 2D/3D Kalman models.
- No changes to `rolling_mean_stddev` or any other indicator.
- No test files (not requested; per `agents/general/access.md`).
- No defensive validation (e.g., empty-array checks, `window <= 0`, dtype coercion) beyond
  what the existing functions in this file already do (none) — see Assumptions.

## 6. Assumptions

- `measurements` is non-empty (`n >= 1`) and `dtype=np.float64`, consistent with every
  other function in this package/`indicators` — no other function in either package
  guards against empty input or wrong dtype, so this one won't either. (Note: unlike
  `kalman_1d_batch`, whose `initial_estimate`/`initial_error_cov` are caller-supplied and
  thus never touch `measurements` directly, this new function's hardcoded
  `measurements[0]` / `variance[0]` would raise an `IndexError` on empty input — flagged
  here, not defended against, per the non-goals above.)
- `@njit` (not `@njit(inline='always')`) matches this file's existing pattern for batch
  functions (`kalman_1d_batch` uses plain `@njit`); only the indicators package uses
  `inline='always'`.
- `variance = stddev ** 2` is the intended "powering" operation the task describes (squaring
  stddev to variance), not some other exponent.
- `process_variance` being "fixed among all items" simply means: pass it straight through
  to every `kalman_1d_step` call unchanged — identical to how `kalman_1d_batch` already
  treats its own `process_variance` parameter today (no behavior change needed there,
  since it was already a fixed scalar).
- The new function should be re-exported from `packages/kalman_filter/__init__.py`
  alongside `kalman_1d_step`/`kalman_1d_batch`, since it's part of the public 1D Kalman
  API (mirrors how `rolling_median_iqr` was added to `packages/indicators/__init__.py` in
  a prior spec).

## 7. Acceptance criteria

- `kalman_1d_batch_adaptive(measurements, process_variance, window)` exists in
  `packages/kalman_filter/kalman_fast.py`, decorated `@njit`, compiles and runs.
- Internally calls `rolling_mean_stddev(measurements, window)` from
  `packages.indicators.rolling_mean_stddev`, uses column `1` (stddev) squared as the
  per-index measurement variance.
- `initial_estimate == measurements[0]`; `initial_error_cov == variance[0]`.
- Returns `(estimates, error_covariances)`, both `np.ndarray`, `dtype=float64`,
  `shape == (len(measurements),)` — same type/shape contract as `kalman_1d_batch`.
- Every index has a real (non-padded) value; length of both outputs equals
  `len(measurements)` exactly.
- Re-running with the same inputs is deterministic (no randomness), consistent with all
  other functions in the package.
- `agents/packages/kalman_filter.md` documents the new function (signature, semantics,
  return shape) alongside the existing `kalman_1d_batch` entry.
- `packages/kalman_filter/__init__.py` exports `kalman_1d_batch_adaptive`.

## 8. Open questions

1. **`@njit` vs `@njit(inline='always')`:** Assumed plain `@njit` to match
   `kalman_1d_batch`'s existing style in this same file (not `inline='always'`, which is
   an indicators-package-only convention). Confirm if `inline='always'` is actually wanted
   here instead.
2. **`__init__.py` export:** Assumed yes (see Assumptions §6). Confirm if this function
   should stay unexported/internal instead.
3. **Empty-input behavior:** Assumed no guard is needed (raises naturally on
   `measurements[0]` if empty), matching the package's no-defensive-checks style. Confirm
   if an explicit empty-input path (e.g., returning empty arrays) is actually desired.

## 9. Notes for the downstream coding agent

- Implement the loop directly (measurements → rolling_mean_stddev → variance → per-index
  `kalman_1d_step` calls) rather than calling `kalman_1d_batch` and trying to retrofit a
  varying `measurement_variance` into it — `kalman_1d_batch`'s signature takes a single
  scalar `measurement_variance` and should not be changed.
- Reuse `kalman_1d_step` as the inner step function (don't duplicate its predict/update
  math inline).
- Keep the docstring short, in the same terse style as `kalman_1d_step`/`kalman_1d_batch`
  in this file; state the per-index adaptive variance behavior and the return contract.
- After implementing, update `agents/packages/kalman_filter.md` and
  `packages/kalman_filter/__init__.py` (per Open Questions 1–2 assumptions) unless told
  otherwise.
