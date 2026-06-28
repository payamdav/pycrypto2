# Spec: 1D / 2D / 3D Kalman Filter Suite (`packages/kalman_filter/`)

## 1. Task summary

Build a high-performance, Numba-jitted Kalman filtering suite covering three
state dimensionalities, each exposing a **stateless recursive step** and a
**pre-allocated batch** function:

| Dim | State | Step | Batch |
|-----|-------|------|-------|
| 1D | value | `kalman_1d_step` | `kalman_1d_batch` |
| 2D | value, speed | `kalman_2d_step` | `kalman_2d_batch` |
| 3D | value, speed, acceleration | `kalman_3d_step` | `kalman_3d_batch` |

The existing 1D implementation (`kalman_fast.py`, currently named
`kalman_filter_step` / `kalman_filter_batch`) is **renamed** to
`kalman_1d_step` / `kalman_1d_batch` and its `__main__` test block removed. 2D
and 3D live in **separate modules**. Package docs are updated afterward.

## 2. Background and context

`packages/kalman_filter/kalman_fast.py` already implements the 1D scalar filter
(`@njit`, step + batch, standard predict/update/correct, no default params). It
is functionally correct but: (a) its function names don't match the new
`kalman_{1d,2d,3d}_*` family, and (b) it carries a `__main__` benchmark/test
block that must be removed.

The 2D model adds a constant-velocity kinematic state; the 3D model adds
acceleration. Both derive a hidden velocity (and acceleration) from a single
scalar measurement stream using matrix-form predict/update/correct.

Source of requirements: the user-provided spec
`kalman_filter_1d_2d_3d.md` (attached to the task).

No code outside the package imports the Kalman functions (verified by repo
search), so renaming the 1D functions is safe.

## 3. Relevant conventions from `/agents/`

- **Package docs** (`agents/general/rules.md`, `agents/packages/`): every
  package must have an up-to-date doc in `agents/packages/`. Update
  `agents/packages/kalman_filter.md` to cover all three models and the renamed
  1D API.
- **Dependencies** (`agents/general/rules.md`): `requirements.txt` must list
  every external dep. Current file lists `numpy`, `numba` — still sufficient; no
  new dep is introduced.
- **Writing style**: docstrings/comments as short as possible while complete.
- **Testing** (`agents/general/access.md`): not required; do **not** add tests
  unless asked. (This also justifies removing the existing `__main__` block.)
- **File placement** (`agents/general/paths_and_files.md`): reusable library
  code lives under `packages/`; this spec lives under `ai_chats/`.

## 4. Functional requirements

### 4.0 Global rules (all six functions)

1. Each function is decorated with `@njit`.
2. **No default parameters** — every argument is strictly required.
3. Standard notation in code comments: `z_k`, `x̂_k`, `P_k`, `Q`, `R`, `K_k`,
   `F`, `H`.
4. Batch functions **pre-allocate** all output arrays via NumPy to the full
   timeline shape before the loop, and write into them in place.
5. Each batch function drives its sequence by calling the matching step function
   in a Python `for` loop (mirrors existing 1D structure).
6. Return-type annotation may be `-> tuple:` (Numba-friendly), as in the current
   file; the subscripted forms in the prototypes are documentation only.

### 4.1 File layout (decided)

- `packages/kalman_filter/kalman_fast.py` — 1D model, functions renamed to
  `kalman_1d_step` / `kalman_1d_batch`; `__main__` block removed.
- `packages/kalman_filter/kalman_2d.py` — 2D model.
- `packages/kalman_filter/kalman_3d.py` — 3D model.

### 4.2 1D model (`kalman_fast.py`)

Rename only; keep the existing (correct) scalar math:

```python
@njit
def kalman_1d_step(measurement, prev_estimate, prev_error_cov,
                   process_variance, measurement_variance):
    ...  # returns (current_estimate, current_error_cov) as (float, float)

@njit
def kalman_1d_batch(measurements, initial_estimate, initial_error_cov,
                    process_variance, measurement_variance):
    ...  # returns (estimates, error_covariances) as float64 arrays, shape (N,)
```

- `F = 1`, `H = 1` (identity/scalar). Predict: `P⁻ = P + Q`. Gain:
  `K = P⁻ / (P⁻ + R)`. Correct: `x = x⁻ + K(z − x⁻)`, `P = (1 − K)P⁻`.
- Remove the `if __name__ == "__main__":` block entirely.

### 4.3 2D model (`kalman_2d.py`) — constant velocity

State `x̂ = [value, speed]ᵀ`, shape `(2, 1)`.

- `F = [[1, dt], [0, 1]]`, built internally from `dt`.
- `H = [[1, 0]]`, shape `(1, 2)`.

```python
@njit
def kalman_2d_step(measurement, prev_state, prev_covariance,
                   process_noise, measurement_variance, dt):
    ...  # returns (state (2,1), covariance (2,2))

@njit
def kalman_2d_batch(measurements, initial_state, initial_covariance,
                    process_noise, measurement_variance, dt):
    ...  # returns (states (N,2,1), covariances (N,2,2))
```

- `prev_state` is `(2,1)`, `prev_covariance` is `(2,2)`, `process_noise` (`Q`)
  is `(2,2)`, `measurement_variance` (`R`) is a scalar float, `dt` is a float.
- Batch pre-allocates `states` shape `(N, 2, 1)` and `covariances` shape
  `(N, 2, 2)`, both `float64`.

### 4.4 3D model (`kalman_3d.py`) — constant acceleration

State `x̂ = [value, speed, acceleration]ᵀ`, shape `(3, 1)`.

- `F = [[1, dt, 0.5·dt²], [0, 1, dt], [0, 0, 1]]`, built internally from `dt`.
- `H = [[1, 0, 0]]`, shape `(1, 3)`.

```python
@njit
def kalman_3d_step(measurement, prev_state, prev_covariance,
                   process_noise, measurement_variance, dt):
    ...  # returns (state (3,1), covariance (3,3))

@njit
def kalman_3d_batch(measurements, initial_state, initial_covariance,
                    process_noise, measurement_variance, dt):
    ...  # returns (states (N,3,1), covariances (N,3,3))
```

- Shapes scale to 3: state `(3,1)`, covariance/`Q` `(3,3)`, `R` scalar float,
  `dt` float. Batch pre-allocates `(N,3,1)` and `(N,3,3)`, `float64`.

### 4.5 Matrix predict/update/correct (2D and 3D)

Use `@` and `.T` matrix operators. `I` is the identity matching the state size.
`z_k − H x̂⁻` is computed with `z_k` as the scalar `measurements[k]`.

1. **Predict:** `x̂⁻ = F x̂`; `P⁻ = F P Fᵀ + Q`.
2. **Gain:** `S = H P⁻ Hᵀ + R` (1×1); `K = P⁻ Hᵀ S⁻¹`.
3. **Correct:** `x̂ = x̂⁻ + K (z_k − H x̂⁻)`; `P = (I − K H) P⁻`.

`S` is a 1×1 matrix; invert via scalar reciprocal of `S[0, 0]` (Numba-friendly,
avoids `np.linalg.inv` on a 1×1) — implementer's choice as long as results match.

### 4.6 Package wiring

- `packages/kalman_filter/__init__.py`: create/update to re-export the public
  API — `kalman_1d_step`, `kalman_1d_batch`, `kalman_2d_step`,
  `kalman_2d_batch`, `kalman_3d_step`, `kalman_3d_batch`. (No `__init__.py`
  currently exists in this package; add one.)
- `requirements.txt`: unchanged (`numpy`, `numba`).

## 5. Non-goals / out of scope

- No tests, no `__main__` demo/benchmark blocks in any module.
- No new external dependencies.
- No control-input (`B u`) term, no time-varying `Q`/`R`, no multi-dimensional
  (vector) measurements — `H` always maps to the scalar `value`.
- No changes to other packages or notebooks.
- No `np.linalg.inv` requirement; 1×1 scalar inversion is acceptable.

## 6. Assumptions

- Measurement is always a scalar per step; `measurements` is a 1D `float64`
  array of length `N`.
- All `float`/array inputs are `float64`; arrays are C-contiguous.
- 1D math is already correct and only needs renaming + `__main__` removal — not
  a behavioral rewrite.
- Column-vector convention `(d, 1)` is preserved end-to-end, including the batch
  output shapes `(N, d, 1)` (not flattened to `(N, d)`).
- `Q` and `initial_covariance` are caller-supplied matrices of the correct
  shape; functions do not validate shapes.

## 7. Acceptance criteria

- Six `@njit` functions exist with the exact names and signatures in §4.2–4.4,
  no default parameters, and compile + run under Numba.
- 1D: `kalman_filter_step` / `kalman_filter_batch` no longer exist; the
  `__main__` block is gone; numeric behavior is unchanged from the current
  implementation.
- 2D/3D step functions return `(state, covariance)` of shapes
  `(d,1)` / `(d,d)`; batch functions return pre-allocated
  `(N,d,1)` / `(N,d,d)` `float64` arrays produced by looping the step function.
- `F` and `H` are constructed internally from `dt` exactly as specified;
  predict/update/correct follow §4.5.
- `packages/kalman_filter/__init__.py` re-exports all six functions; importing
  them succeeds.
- `agents/packages/kalman_filter.md` is updated to document all three models and
  the renamed 1D API; `requirements.txt` still lists `numpy`, `numba`.

## 8. Open questions

None blocking. Resolved decisions: (a) 1D functions **renamed** to
`kalman_1d_*`; (b) 2D/3D in **separate modules** (`kalman_2d.py`,
`kalman_3d.py`), 1D stays in `kalman_fast.py`.

## 9. Notes for the downstream coding agent

- Reuse the existing `kalman_fast.py` body for 1D; just rename the two functions
  and delete the `__main__` block. Update the module docstring if it names the
  old functions.
- For 2D/3D, build `F`/`H`/`I` inside each step with explicit `np.array(...,
  dtype=np.float64)` so Numba infers types cleanly; keep `prev_state`/`P` as the
  passed-in arrays. Compute `S` as `(H @ P⁻ @ H.T)[0,0] + R` and `K = P⁻ @ H.T /
  S`.
- In batch loops, assign `states[k] = x̂` and `covariances[k] = P` into the
  pre-allocated arrays; carry the running `x̂`/`P` between iterations.
- Mirror the existing file's comment style (notation labels on each stage); keep
  docstrings short per the writing-style rule.
- After implementation, update `agents/packages/kalman_filter.md` (import paths,
  all six functions, shapes, tuning notes) — required by the package-docs rule.
- Add `packages/kalman_filter/__init__.py` with the six re-exports.
