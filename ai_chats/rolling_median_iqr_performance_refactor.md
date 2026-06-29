# Spec: Speed up `rolling_median_iqr` + per-function timing in `lbla_n_vp` notebook

## 1. Task summary

Two independent changes:

1. **Refactor `rolling_median_iqr`** in
   `packages/indicators/rolling_robust_z_score.py` to use the **incremental sorted-buffer
   algorithm** already used by `rolling_robust_z_score` in the same file, instead of
   copying and re-sorting the full window at every index. Results must be **bit-identical**
   to the current implementation.
2. **Add per-call timing** to the metrics-cache cell of
   `strategies/lbla_n_vp/lbla_n_vp.ipynb` so each cached-metrics function prints its own
   elapsed time as it is called.

## 2. Background and context

### Current `rolling_median_iqr` (slow path)

For every index `i` the function:
- copies the whole window `array[start : i+1]` (length `m`) into `buf`, then
- insertion-sorts `buf` from scratch.

This is **O(n · W²)** worst case (full copy + full sort per index) and is the reported
performance problem.

### `rolling_robust_z_score` (fast reference, same file)

Maintains a single sorted buffer:
- The **first full window** is copied once and insertion-sorted once.
- Each **subsequent window** removes the outgoing value and inserts the incoming value by
  overwriting it in place and **bubbling** it left/right into sorted position — **O(W)**
  per step, **O(n · W)** total.

The two functions already share the median and quartile conventions
(`Q1 = sorted[k//4]`, `Q3 = sorted[3k//4]`, even-length median averaged), so reusing the
incremental machinery keeps them consistent.

### Key reconciliation (the one non-trivial design point)

`rolling_robust_z_score` only emits values for `i >= window-1` and **pads earlier indices
with `0.0`**. `rolling_median_iqr` has the **opposite contract**: it must emit a *real*
median/IQR for **every** index, including partial early windows (down to `m == 1`), with
**no `0.0` padding** (documented deviation in `agents/packages/indicators.md` and
`ai_chats/rolling_median_iqr_indicator.md`).

Therefore "use exactly the same algorithm" cannot be a literal copy. The same incremental
**sorted-buffer + bubble** primitive is reused, extended with a **growing phase** for the
partial windows:

- **Growing phase** (`i = 0 .. window-1`, buffer not yet full): insert `array[i]` into its
  sorted position in a buffer of current logical size `m = i+1` (one insertion step — the
  same insertion-sort inner loop, applied incrementally). Compute median/IQR over the `m`
  filled entries and write `out[i]`.
- **Sliding phase** (`i >= window`): buffer is full at size `window`; remove the outgoing
  value `array[i-window]` and insert `array[i]` by overwrite + bubble — **identical** to
  the `rolling_robust_z_score` subsequent-window update. Compute median/IQR over the full
  `window` and write `out[i]`.

This preserves the no-padding contract while making the function **O(n · W)** like the
reference.

## 3. Relevant conventions from `/agents/`

From `agents/packages/indicators.md`, `agents/general/rules.md`,
`agents/general/paths_and_files.md`:

- Indicator input/output: 1-D `np.ndarray`, **`dtype=np.float64`**; output newly
  allocated. `window` defaults to `60`.
- Functions are `@nb.njit(inline='always')`; **explicit `for` loops only** inside the
  jitted body — no `np.sort` / `np.median` / `np.percentile` / slice reductions.
- `rolling_median_iqr` keeps its **documented no-`0.0`-padding deviation** for partial
  windows.
- Package docs are authoritative: if the externally observable behavior/contract is
  unchanged, `agents/packages/indicators.md` needs no edit; if anything observable changes
  it must be updated (Rule: "Package Documentation").
- Notebooks must be self-contained: any package used must be `%pip install`-ed at the top
  and the project repo cloned/on `sys.path` (Rule: "Dependency Management",
  "Repository Access in Jupyter Notebooks"). The existing notebook already follows this;
  do not regress it.
- Writing style: docstrings/comments/printed labels as short as possible while complete.

## 4. Functional requirements

### 4A. `rolling_median_iqr` refactor

1. **Signature unchanged:** `rolling_median_iqr(array, window=60)`,
   `@nb.njit(inline='always')`, same file
   `packages/indicators/rolling_robust_z_score.py`.
2. **Output unchanged:** newly allocated `np.ndarray`, shape `(n, 2)`, `dtype=np.float64`;
   `out[i, 0]` = median, `out[i, 1]` = IQR of `array[max(0, i-window+1) : i+1]`.
3. **Algorithm:** single persistent sorted buffer with a growing phase then a sliding
   phase, as described in §2 — **no full-window copy + full re-sort per index**.
   Target complexity **O(n · W)**.
4. **Bit-identical results:** for all `i`, `out[i]` must equal the current implementation's
   output (same float64 values), under the existing median rule (odd: `s[m//2]`; even:
   `(s[m//2-1]+s[m//2])/2`) and quartile rule (`s[3*m//4] - s[m//4]`, with `m` the current
   effective length). `m == 1 ⇒ IQR = 0.0`; `out[0] == [array[0], 0.0]`.
5. **Edge cases preserved:** `n == 0` returns shape `(0, 2)`; works for `n < window`
   (growing phase only, never reaches sliding phase); no `0.0` padding anywhere.
6. **Duplicate-value safety in the sliding phase:** removing the outgoing value uses a
   linear scan for *a* slot holding that value (same as `rolling_robust_z_score`). This is
   safe because the result only depends on the multiset of window values; matching the
   reference behaviour is sufficient.
7. **`rolling_robust_z_score` must not change** (behavior or signature). Optionally factor
   shared logic into a small `@nb.njit` helper **only if** it leaves both functions'
   outputs identical; duplication is acceptable if a clean shared helper is awkward.

### 4B. Notebook per-call timing (`strategies/lbla_n_vp/lbla_n_vp.ipynb`, metrics-cache cell)

Target = **Cell 4**, currently:

```python
t0 = time.perf_counter()
create_metrics_cache_base_file(asset)
metrics_cache_volume_median_iqr(asset)
metrics_cache_volume_mean_stddev(asset)
t_metrics = time.perf_counter() - t0
print(f"\nMetrics cache total: {t_metrics:.3f}s")
```

1. Measure and **print the elapsed time of each of the three calls individually**,
   printed *between* the calls (i.e. each function's time is printed right after it
   returns, before the next is called):
   - `create_metrics_cache_base_file(asset)`
   - `metrics_cache_volume_median_iqr(asset)`
   - `metrics_cache_volume_mean_stddev(asset)`
2. **Keep** the existing total line and the existing `t_metrics` variable (Cell 5 uses
   `t_preload + t_metrics`) — do not break downstream cells.
3. Use `time.perf_counter()` (already imported in Cell 3). Use a consistent, short label
   format, e.g. `print(f"  create_metrics_cache_base_file   {dt:.3f}s")`.
4. No other cells change; no new package imports needed.

## 5. Non-goals / out of scope

- No change to `rolling_robust_z_score`'s behavior, output, or signature.
- No change to the public contract / output shape / values of `rolling_median_iqr` — this
  is a **pure performance refactor**.
- No new modules, packages, notebooks, tests, or requirements files.
- No changes to other notebook cells, to `metrics_cache` package functions, or to other
  indicators.
- No look-ahead / centered-window variants.

## 6. Assumptions

- "Use exactly the same algorithm as `rolling_robust_z_score`" means reuse its incremental
  sorted-buffer + bubble update for the full-window (sliding) phase, plus a growing-insert
  phase for partial windows — **not** adopting its `0.0`-padding behavior, which would
  break this function's documented contract.
- Existing median/quartile conventions are correct and must be preserved exactly
  (consistency with `rolling_robust_z_score` is intentional).
- Inputs are `float64`, as for all indicators.
- Because the observable contract is unchanged, `agents/packages/indicators.md` likely
  needs **no** edit; update it only if the implementer changes any documented behavior.
- The metrics-cache cell to modify is Cell 4 (the one importing from
  `packages.tools.metrics_cache`); `time` is already imported in Cell 3 and Cells 4/5 run
  after it.

## 7. Acceptance criteria

- `rolling_median_iqr` no longer copies+re-sorts the full window per index; it maintains a
  persistent sorted buffer (growing then sliding) and is O(n · W).
- For randomized and edge-case inputs (`n == 0`, `n == 1`, `n < window`, `n >> window`,
  arrays with duplicate values, `window == 1`), the refactored output is **element-wise
  equal** to the pre-refactor output (exact float64 equality) and to a NumPy reference
  using the same integer-index quartile/median rule.
- Output remains shape `(n, 2)`, `dtype=np.float64`; `out[0] == [array[0], 0.0]`; no `0.0`
  padding of partial windows.
- `rolling_robust_z_score` output is unchanged.
- Notebook Cell 4 prints three individual per-function timings (one after each call) plus
  the existing total; `t_metrics` still holds the summed total and Cell 5 still runs.
- Measured wall-clock of `rolling_median_iqr` on a large array (e.g. `n = 1_000_000`,
  `window = 60`) is substantially lower than before (sanity check, not a hard threshold).

## 8. Open questions

1. **Partial-window handling:** Confirm partial early windows must keep emitting real
   median/IQR values (no `0.0` padding), i.e. the refactor preserves the current contract
   and only changes internal performance. *(Assumed **yes** — the no-padding behavior is a
   documented, intentional deviation.)*
2. **Shared helper:** Acceptable to leave the median/quartile + bubble logic duplicated
   between the two functions if extracting a shared `@nb.njit` helper is awkward?
   *(Assumed **yes**, as long as outputs stay identical.)*
3. **Timing label format:** Any preferred label/format for the per-call prints, or is a
   short `name  x.xxxs` line per function acceptable? *(Assumed acceptable.)*

## 9. Notes for the downstream coding agent

- Implement with explicit loops only inside the jitted body.
- Mirror `rolling_robust_z_score`'s sliding-phase update verbatim (linear scan to find the
  outgoing value's slot, overwrite, bubble left then right); add the growing phase for
  `i < window` using single insertion-sort steps so partial windows stay correct and
  unpadded.
- Before/after equivalence check: run the **old** vs **new** function on several random
  `float64` arrays (incl. duplicates and `window=1`) and assert exact equality —
  this is a refactor, results must not drift.
- Keep the docstring short; it should still state the no-`0.0`-padding partial-window
  behavior. Update `agents/packages/indicators.md` only if any documented behavior changes
  (it should not).
- For the notebook, edit only Cell 4: time each call with `time.perf_counter()` deltas,
  print each immediately after its call, and keep `t_metrics` as the total so Cell 5 is
  unaffected.
