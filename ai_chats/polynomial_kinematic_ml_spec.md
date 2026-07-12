# Spec: Multi-Horizon Kinematic Regime Classification — `scripts/studies/polynomial_kinematic_ml/`

## 1. Task Summary

Build a decoupled ML study under `scripts/studies/polynomial_kinematic_ml/` that tests whether multi-scale cubic-polynomial kinematics (velocity, acceleration, jerk) of 1-minute VWAP predict directional regimes over four forward horizons (60, 120, 180, 240 minutes). Pipeline: `build_dataset.py` (features + labels → parquet) → `train.py` (LightGBM vs CatBoost vs ElasticNet logistic, custom purged & embargoed CV, out-of-fold predictions) → `evaluate.py` (precision-focused metric matrix, markdown + JSON). Every parameter lives in `config.py`.

## 2. Background and Context

- Origin: an external advisor blueprint, adapted here to this repo's real APIs and conventions.
- Features come from `calculate_market_kinematics(prices, window_size)` in `packages/indicators/motion.py`: rolling cubic OLS fit, output `(n, 7)` — columns 0–2 are velocity, acceleration, jerk ("speed" in the blueprint = velocity, column 0). Valid from index `window_size - 1`; **earlier rows are backfilled with the first valid row, not NaN** — warm-up rows must be removed positionally, `dropna` will not catch them.
- Candles come from `packages.candle_loader.load_candles(asset, date_from, date_to)` → `(n, 11)` float64, columns `ts, o, h, l, c, v, q, n, vwap, vb, vs`. `vwap` (index 8) is precomputed `q / v`.
- Features look back up to 240 bars and labels look forward up to 240 bars, so random K-fold would leak; a custom purged + embargoed splitter is mandatory.

## 3. Relevant Conventions from `/agents/`

- `packages/candle_loader` is the single authoritative candle source; it already implements the local-cache rule (`agents/datasets/data_pre_load.md`) — no custom loaders, no re-download logic.
- Use the `vwap` column directly; never recompute `q / v` (`agents/datasets/huggingface_candles.md`).
- Asset names lowercase, e.g. `"btcusdt"` (`agents/datasets/assets.md`).
- A `requirements.txt` listing every external package must sit in the study directory (`agents/general/rules.md`).
- Study scripts belong under `scripts/studies/` (`agents/general/paths_and_files.md`).
- Boundary handling follows the exclusive (default) mode of `agents/ideas/idea_look_back_look_ahead.md`: extend the loaded range beyond `[start_date, stop_date]` so every observation inside the stated range is usable.
- Writing style: all docstrings/comments as short as possible. No tests, no debugging beyond the task (`agents/general/access.md`).

## 4. Functional Requirements

### 4.0 Layout and Execution

```text
scripts/studies/polynomial_kinematic_ml/
├── __init__.py            # empty
├── config.py              # StudyConfig dataclass + CONFIG instance
├── build_dataset.py       # data → features + labels → parquet
├── validation.py          # purged & embargoed splits
├── train.py               # 3 models × 4 horizons × CV, saves OOF predictions
├── evaluate.py            # metrics matrix: stdout markdown + metrics.json
└── requirements.txt
```

Run from the repo root, in order:

```bash
python -m scripts.studies.polynomial_kinematic_ml.build_dataset
python -m scripts.studies.polynomial_kinematic_ml.train
python -m scripts.studies.polynomial_kinematic_ml.evaluate
```

### 4.1 `config.py`

`@dataclass StudyConfig` — single source of truth; no magic numbers in any other module. Module-level `CONFIG = StudyConfig()`.

| Field | Default |
|---|---|
| `asset` | `"btcusdt"` |
| `start_date` / `stop_date` | `"2024-01-01"` / `"2025-12-31"` |
| `lookback_windows` | `[10, 20, 30, 60, 120, 240]` |
| `forward_horizons` | `[60, 120, 180, 240]` |
| `bps_multiplier` | `0.0030` (30 bps) |
| `cv_splits` | `5` |
| `dataset_output_path` | `"data/processed/kinematic_study_dataset.parquet"` |
| `oof_output_path` | `"data/processed/kinematic_oof_predictions.parquet"` |
| `metrics_output_path` | `"scripts/studies/polynomial_kinematic_ml/metrics.json"` |
| `classification_threshold` | `0.5` |
| `random_state` | `42` |
| `lgbm_params` | `{"max_depth": 4, "num_leaves": 15, "learning_rate": 0.03, "n_estimators": 400, "verbosity": -1}` |
| `catboost_params` | `{"depth": 4, "learning_rate": 0.03, "iterations": 400, "verbose": 0, "allow_writing_files": False}` |
| `elasticnet_params` | `{"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5, "C": 1.0, "max_iter": 200}` |

List fields via `field(default_factory=...)`. `random_state` is injected into all three models at construction.

### 4.2 `build_dataset.py`

1. **Load** with extended boundaries: `load_start = start_date − max(lookback_windows)` minutes, `load_end = stop_date 23:59 + max(forward_horizons)` minutes; call `load_candles(asset, load_start, load_end)`. The loader clamps to available data — validity below is enforced positionally, never assumed from dates.
2. **Price** = `vwap` column (index 8). Sanitize: replace non-finite entries (zero-volume candles) with that candle's close `c`.
3. **Features (18)**: for each `w` in `lookback_windows`, `k = calculate_market_kinematics(vwap, w)`; take `k[:, 0..2]` as columns `vel_w{w}`, `acc_w{w}`, `jerk_w{w}`.
4. **Labels (4)**: for each horizon `H`, at anchor `t`: `X = arange(H)`, `Y = vwap[t+1 : t+1+H]`, slope `m = Cov(X, Y) / Var(X)`, `P_next = vwap[t+1]`; `label_h{H} = uint8(m >= P_next * bps_multiplier / H)`. Labels use only rows `> t` — no same-bar information.
5. **Vectorization**: no per-row Python loops. Preferred: one Numba pass per horizon with running accumulators `S = Σy`, `W = Σ i·y` (local `i = 0..H-1`); slide with `S += y[a+H] − y[a]`, `W = W − (S_old − y[a]) + (H−1)·y[a+H]`; then `m = (W − S·(H−1)/2) / (H·(H²−1)/12)`. A prefix-sum formulation is acceptable if it passes the label spot-check tolerance.
6. **Validity trim (positional)**: keep anchors `t` with `t >= max(lookback_windows) − 1` (full warm-up; removes backfilled feature rows) **and** `t <= n − 1 − max(forward_horizons)` (all four labels computable), then keep `ts` within `[start_date, stop_date]`. Every stored row has all 18 features genuine and all 4 labels defined.
7. **Output**: single pandas DataFrame → parquet at `dataset_output_path` (create parent dirs). Columns: `ts` (int64 ms), 18 float64 features, 4 uint8 labels. Deterministic across reruns.

### 4.3 `validation.py`

```python
def get_purged_embargoed_splits(n_samples, n_splits, purge, embargo):
    """Yield (train_idx, val_idx) positional int arrays."""
```

- Partition `[0, n_samples)` into `n_splits` contiguous blocks (remainder to the last); fold `i` uses block `i` as validation, so every row is in exactly one validation fold.
- For a validation block `[v0, v1]`, exclude train indices in `[v0 − purge, v1 + embargo]`; train = all remaining indices (both sides allowed).
- Callers pass `purge = max(forward_horizons)`, `embargo = max(lookback_windows)` — the advisor rule: no train index inside `[val_start − max_lookforward, val_end + max_lookback]`.
- Assert every fold has non-empty train and val.
- **Forbidden anywhere in the study directory**: `sklearn.model_selection.KFold` / `train_test_split` / any random or shuffled split.

### 4.4 `train.py`

- Read the dataset parquet; `X` = the 18 feature columns (row order = time order; positional indices are valid for the splitter).
- Models (params from config, `random_state` injected):
  1. `lightgbm.LGBMClassifier(**lgbm_params)`
  2. `catboost.CatBoostClassifier(**catboost_params)`
  3. `sklearn.pipeline.Pipeline([StandardScaler(), LogisticRegression(**elasticnet_params)])` — scaler fit on the train fold only.
- For each horizon `H` (y = `label_h{H}`) × each model × each fold from `get_purged_embargoed_splits`: fit on train, `predict_proba` on validation, keep the True-class probability.
- Save all OOF predictions as one long-format parquet at `oof_output_path`: columns `ts` (int64), `fold` (int8), `horizon` (int16), `model` (str: `"lightgbm" | "catboost" | "elasticnet"`), `y_true` (uint8), `y_prob` (float32). Row count = `n_rows × 4 × 3`.
- Print one progress line per (horizon, model, fold).

### 4.5 `evaluate.py`

- Read the OOF parquet. Per (model, horizon) over all OOF rows: Precision on the True class (at `classification_threshold`), Recall, F1, ROC-AUC, positive prevalence, `n`.
- Print one clean markdown table to stdout (rows = model × horizon), noting that True-class Precision is the headline metric — low accuracy is acceptable if Precision is high when the model flags True.
- Write the same metrics as structured JSON to `metrics_output_path` (this file is committed; the parquet artifacts under `data/` are gitignored).

### 4.6 `requirements.txt`

`numpy`, `pandas`, `pyarrow`, `numba`, `duckdb`, `scikit-learn`, `lightgbm`, `catboost`.

## 5. Non-Goals / Out of Scope

- No hyperparameter search or tuning; fixed conservative params only.
- No trading strategy, backtest, threshold optimization, or deployment.
- No plots, notebooks, or tests (testing not requested).
- No multi-asset batch runs — one config-driven asset per run (must work for any asset in `agents/datasets/assets.md`).
- No changes to `packages/` or any file outside the study directory.
- No mocking or synthetic data — real candles via `candle_loader` only.

## 6. Assumptions

- Rows are treated as consecutive 1-minute bars; rare exchange gaps are ignored (horizons are bar counts, not wall-clock).
- The advisor threshold `m ≥ P_next · bps_multiplier / H` is adopted verbatim (per-bar slope; ≈ "fitted move over the horizon ≥ 30 bps").
- Purge/embargo widths are exactly the advisor interval (purge = max horizon before, embargo = max lookback after the validation block).
- `P_next` uses the sanitized vwap at `t + 1`.
- OOF stores the True-class probability; class metrics use the 0.5 config threshold. No class re-weighting or resampling.

## 7. Acceptance Criteria

1. Directory tree matches §4.0; all tunables come from `config.py` only.
2. `build_dataset.py` writes a parquet with exactly 23 columns, no NaN/inf, no rows lacking full lookback or any of the 4 labels; feature computation itself runs in seconds (excluding first-time candle download and Numba JIT warm-up).
3. Label spot-check: for random `(t, H)`, `np.polyfit(np.arange(H), vwap[t+1:t+1+H], 1)[0]` matches the stored slope decision (`rtol=1e-6`) and reproduces `label_h{H}`.
4. Feature spot-check: dataset row anchored at `t` equals `calculate_market_kinematics(vwap, w)[t, 0..2]` for each `w`.
5. Splits: the `cv_splits` validation blocks tile all rows disjointly; for every fold no train index lies inside `[v0 − purge, v1 + embargo]`; no sklearn splitter imported anywhere in the study directory.
6. `train.py` writes OOF parquet with `n_rows × 4 × 3` rows covering every dataset row for all 12 model–horizon combinations.
7. `evaluate.py` prints the markdown matrix and writes valid JSON containing all 12 combinations with the 5 metrics + `n`.
8. `requirements.txt` present and complete; all three entry points run from the repo root via `python -m`.

## 8. Open Questions (non-blocking — defaults chosen above)

- Threshold semantics: dividing by `H` makes the required total fitted move `m·(H−1) ≈ 30 bps · (H−1)/H`. If exactly-30-bps-over-horizon is intended (`m ≥ P_next·bps/(H−1)`), say so; difference is < 2 %.
- Boosting iteration counts (400) and ElasticNet `l1_ratio/C` were not specified by the advisor — chosen conservatively in config; adjust there if desired.
- A stricter symmetric exclusion (purge and embargo both = max lookback + max horizon) is available on request; the advisor interval is used as specified.

## 9. Notes for the Downstream Coding Agent

- `calculate_market_kinematics` JIT-compiles on first call; pass the vwap array once per window size, contiguous float64 (`load_candles` already returns float64).
- Do not materialize `sliding_window_view` matrices for labels (~2 GB per horizon at ~1 M rows); use the running-accumulator recurrence in §4.2.5.
- CatBoost must have `allow_writing_files=False` or it drops a `catboost_info/` directory into the repo; keep LightGBM `verbosity=-1`.
- `data/` is gitignored — dataset and OOF parquets stay local; `metrics.json` is the committed result artifact.
- Expected runtime: dataset build seconds; training tens of minutes (CatBoost dominates: 4 horizons × 5 folds on ~800 k-row train sets). Print progress lines so runs are observable.
- `LogisticRegression(saga)` needs the StandardScaler in the pipeline to converge — jerk features are orders of magnitude smaller than velocity.
- Keep all docstrings terse per the writing-style rule.
