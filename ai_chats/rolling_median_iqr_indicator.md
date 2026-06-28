# Spec: `rolling_median_iqr` look-back indicator

## 1. Task summary

Add a Numba-jitted function `rolling_median_iqr(array, window=60)` to
`packages/indicators/rolling_robust_z_score.py` that returns, for every element, the
median and IQR of the **left look-back window ending at that element**. Output is a 2-D
array of shape `(len(array), 2)`. Early elements use whatever partial window is available
(down to size 1).

## 2. Background and context

`packages/indicators/rolling_robust_z_score.py` already contains
`rolling_robust_z_score(array, window=60)`, which computes `(x - median) / IQR` over a
rolling window using an incremental insertion sort. The new function exposes the
underlying **median** and **IQR** themselves as a per-point 2-column output, reusing the
same quartile/median conventions so the two functions stay consistent.

The window is a **left look-back (causal) window**: for index `i` it covers
`array[max(0, i - window + 1) : i + 1]`. This matches the look-back semantics described in
`agents/ideas/idea_look_back_look_ahead.md`.

## 3. Relevant conventions from `/agents/`

From `agents/packages/indicators.md` and `agents/general/rules.md`:

- Input: 1-D `np.ndarray`, **`dtype=np.float64`**.
- Output: newly allocated array, `dtype=np.float64`.
- `window` defaults to `60`.
- Functions are `@nb.njit(inline='always')`; use **explicit `for` loops** only — no
  NumPy high-level calls (`np.median`, `np.sort`, `np.percentile`, slicing reductions)
  inside the jitted body.
- A `requirements.txt` already covers this package's deps (`numpy`, `numba`); no new dep
  is introduced.
- Package docs are authoritative and must be kept current: update
  `agents/packages/indicators.md` to document the new function (Rule: "Package
  Documentation").
- Writing style: docstrings/comments as short as possible while complete.

> **Deviation note:** The generic indicators convention pads incomplete windows with
> `0.0`. This function **intentionally does not pad** — the request requires real
> median/IQR values for every index, including partial early windows. This deviation is
> deliberate and must be called out in the docstring and in the package doc.

## 4. Functional requirements

1. **Signature:** `rolling_median_iqr(array, window=60)`, decorated
   `@nb.njit(inline='always')`, in
   `packages/indicators/rolling_robust_z_score.py`.
2. **Output:** newly allocated `np.ndarray`, shape `(n, 2)`, `dtype=np.float64`, where
   `n = len(array)`.
   - `out[i, 0]` = median of the look-back window ending at `i`.
   - `out[i, 1]` = IQR of the look-back window ending at `i`.
3. **Look-back window:** for index `i`, the window is
   `array[max(0, i - window + 1) : i + 1]`; its effective length is
   `m = min(i + 1, window)`.
4. **Partial early windows:** every index gets a computed value. For `i < window - 1`,
   compute median and IQR over the available `m` elements. For `m == 1`, median = the
   single value and IQR = `0.0`.
5. **Median convention** (match existing `rolling_robust_z_score`), over the `m` sorted
   window values `s[0..m-1]`:
   - `m` odd: `s[m // 2]`.
   - `m` even: `(s[m // 2 - 1] + s[m // 2]) / 2.0`.
6. **IQR convention** (match existing `rolling_robust_z_score`): `IQR = s[3*m//4] - s[m//4]`
   using the same integer-index quartile selection, with `m` substituted for the window
   length so partial windows are handled by the same rule. (For `m == 1`,
   `3*m//4 == m//4 == 0`, giving `IQR = 0.0`, satisfying requirement 4.)
7. **Empty input:** `n == 0` returns an empty array of shape `(0, 2)`.
8. **Correctness over speed:** a straightforward per-index sort of the current window
   (copy `m` values, insertion-sort, then index the quartiles) is acceptable and
   preferred for clarity. An incremental-sort optimization mirroring the existing function
   is optional (see Open Questions) and must not change results.

## 5. Non-goals / out of scope

- No change to the behavior or signature of the existing `rolling_robust_z_score`.
- No new module, package, notebook, or test (testing is not part of this task per
  `agents/general/access.md`).
- No look-ahead / centered window variants.
- No changes to the package `__init__.py` re-exports unless trivially adding the new name
  (see Open Questions).

## 6. Assumptions

- "Look-back window that ends at each item" = causal window inclusive of the current
  element, per requirement 3.
- Median/IQR conventions must match the existing function in the same file (requirements
  5–6) so the two indicators are mutually consistent; this is the intended meaning of the
  request rather than a textbook linear-interpolation quartile.
- "Improve the file" = add the new function cleanly and consistently; optionally factor
  out a shared `@nb.njit` median/IQR helper if it reduces duplication without changing
  existing results. No broader rewrite is implied.
- Input arrays are `float64`, matching all other indicators.

## 7. Acceptance criteria

- `rolling_median_iqr(array, window=60)` exists in
  `packages/indicators/rolling_robust_z_score.py`, is `@nb.njit(inline='always')`, and
  compiles/runs.
- Returned array has shape `(len(array), 2)` and `dtype=np.float64`.
- For every `i`, `out[i]` equals the median and IQR of
  `array[max(0, i - window + 1) : i + 1]` under the conventions in requirements 5–6
  (verifiable against a NumPy reference using the same quartile indexing).
- Early indices (`i < window - 1`) hold real partial-window values, not `0.0` padding;
  `out[0] == [array[0], 0.0]`.
- `n == 0` returns shape `(0, 2)`; the existing function is unchanged.
- `agents/packages/indicators.md` documents the new function (signature, shape,
  look-back/partial-window behavior, and the no-padding deviation).

## 8. Open questions

1. **Quartile method:** Confirm the integer-index quartile/median rule from the existing
   `rolling_robust_z_score` is the intended definition (assumed **yes**). If a different
   quartile method is wanted (e.g. linear interpolation / `numpy.percentile`), say so.
2. **`__init__.py` export:** Should `rolling_median_iqr` be re-exported from
   `packages/indicators/__init__.py` alongside the other indicators? (Assumed **yes**, as
   a one-line addition, since it is part of the public indicators API.)
3. **Incremental-sort optimization:** Acceptable to keep the simple per-index sort for
   clarity, or is the incremental O(W) update required for performance parity with
   `rolling_robust_z_score`? (Assumed simple per-index sort is fine.)

## 9. Notes for the downstream coding agent

- Implement with explicit loops only inside the jitted body — no `np.sort`, `np.median`,
  `np.percentile`, or array-slice reductions inside `@nb.njit`.
- Reuse the existing file's insertion-sort and quartile logic so the median/IQR match
  `rolling_robust_z_score` exactly; consider extracting a small shared `@nb.njit` helper
  rather than duplicating.
- Keep the docstring short; explicitly state the no-`0.0`-padding behavior for partial
  windows.
- After implementing, update `agents/packages/indicators.md` (and add the export to
  `packages/indicators/__init__.py` if Open Question 2 is confirmed).
