# Spec v2: polynomial_kinematic_ml — per-study data folder, dynamic columns, column toggles, holdout validation

## 1. Task Summary

Modify the existing study `scripts/studies/polynomial_kinematic_ml/` (implemented from `ai_chats/polynomial_kinematic_ml_spec.md`):

1. All generated artifacts — including `metrics.json` — move to `data/polynomial_kinematic_ml/` (gitignored); `data/processed/` is retired.
2. `build_dataset.py` derives feature/label columns entirely from `lookback_windows` / `forward_horizons` — any list lengths, no hard-coded counts (23-column assumption is obsolete).
3. New label `label_any` = OR of all horizon labels, always the last column.
4. New hand-editable `columns.json` (one line per column, `active` flag) controls which features/labels participate in training and evaluation.
5. Per-model `enabled` flags in `config.py`.
6. Replace 5-fold purged CV with one chronological ~80/20 holdout split, keeping purge/embargo separation.

Changes may edit existing files or add new ones, all inside the study folder.

## 2. Background and Context

- v1 spec: `ai_chats/polynomial_kinematic_ml_spec.md`; implementation merged and runs via the three `python -m` entry points (build → train → evaluate).
- v1 wrote parquets under `data/processed/` and `metrics.json` inside the study folder as a committed artifact. Requirement now: no generated file touches git; each study keeps its outputs in its own `data/<study_name>/` subfolder (go-forward pattern for other studies too).
- 5-fold CV is too slow for the current iteration loop; a single holdout is enough for now. The CV splitter stays in `validation.py`, unused.
- Workflow: build once, then iterate by toggling `columns.json` flags and model `enabled` flags between train/evaluate runs — so feature/label counts differ per run and nothing may assume fixed counts.

## 3. Relevant Conventions from `/agents/`

- `data/` is gitignored; `candle_loader` also caches under `CWD/data/` — the study subfolder keeps artifacts separated (`agents/datasets/data_pre_load.md`, `agents/packages/candle_loader.md`).
- Study scripts stay under `scripts/studies/` (`agents/general/paths_and_files.md`); existing `requirements.txt` still valid — no new dependencies (`agents/general/rules.md`).
- Terse docstrings/comments; no tests, no debugging beyond the task (`agents/general/rules.md`, `agents/general/access.md`).
- Chronological splits only — random/shuffled splitting remains forbidden in the study directory (v1 rule, retained).

## 4. Functional Requirements

### 4.0 Layout and Execution

The six module files remain; new code may go into them or into new files in the same folder. Run order and commands unchanged:

```bash
python -m scripts.studies.polynomial_kinematic_ml.build_dataset
python -m scripts.studies.polynomial_kinematic_ml.train
python -m scripts.studies.polynomial_kinematic_ml.evaluate
```

Scripts keep `os.chdir(REPO_ROOT)`, so `CWD/data` = `<repo>/data`.

### 4.1 `config.py`

Changed / new fields (everything else unchanged):

| Field | Value |
|---|---|
| `dataset_output_path` | `"data/polynomial_kinematic_ml/dataset.parquet"` |
| `predictions_output_path` (renames `oof_output_path`) | `"data/polynomial_kinematic_ml/validation_predictions.parquet"` |
| `metrics_output_path` | `"data/polynomial_kinematic_ml/metrics.json"` |
| `columns_output_path` (new) | `"data/polynomial_kinematic_ml/columns.json"` |
| `train_fraction` (new; replaces `cv_splits`, which is removed) | `0.8` |
| `lgbm_enabled` (new) | `True` |
| `catboost_enabled` (new) | `True` |
| `elasticnet_enabled` (new) | `True` |

### 4.2 `build_dataset.py` — dynamic columns + `label_any`

- With `W = len(lookback_windows)`, `Hn = len(forward_horizons)` (each ≥ 1, any length): features = `vel_w{w}`, `acc_w{w}`, `jerk_w{w}` per window (3·W), labels = `label_h{H}` per horizon (Hn), plus `label_any`.
- Column order: `ts`, features in `lookback_windows` order, `label_h{H}` in `forward_horizons` order, `label_any` last. Total = 3·W + Hn + 2.
- `label_any` (uint8): 1 where any `label_h{H}` is 1 (element-wise OR over all horizon labels), computed at build time on the final trimmed rows.
- Load-boundary extension, vwap sanitization, feature/label math, and positional validity trim are unchanged from v1.
- After the parquet, write `columns.json` (§4.3). Print one line per written file.

### 4.3 `columns.json`

- Location: `columns_output_path`.
- JSON array; each dataset column is one object on **one line** (hand-editable):

```json
[
{"index": 0, "name": "ts", "role": "meta", "active": true},
{"index": 1, "name": "vel_w10", "role": "feature", "active": true},
{"index": 22, "name": "label_h240", "role": "label", "active": true},
{"index": 23, "name": "label_any", "role": "label", "active": true}
]
```

- `role`: `"meta"` (`ts`), `"feature"`, `"label"`.
- Written by `build_dataset.py` with all `active: true`. If the file already exists, carry over `active` by column name (new names default `true`, vanished names dropped, `index` rewritten) — a rebuild must not lose hand-toggled flags. Deleting the file resets to defaults.
- Semantics, re-read on every `train.py` / `evaluate.py` run:
  - inactive feature → excluded from `X` in training;
  - inactive label → neither trained nor evaluated;
  - `ts` is always kept; its flag is ignored.
- The dataset parquet always stores **all** columns; flags filter training/evaluation only. `label_any` content is fixed at build time (OR over all horizon labels regardless of flags).

### 4.4 `validation.py` — holdout split

Add:

```python
def get_holdout_split(n_samples, train_fraction, purge, embargo):
    """Return (train_idx, val_idx): chronological holdout with purge/embargo gap."""
```

- `v0 = int(n_samples * train_fraction)`; validation = `[v0, n_samples - 1]`; train = indices outside `[v0 - purge, v1 + embargo]` — effectively `[0, v0 - purge - 1]` (the embargo side is vacuous since validation ends at the last row; keep the formula for generality).
- Assert both sides non-empty.
- Callers pass `purge = max(forward_horizons)`, `embargo = max(lookback_windows)`.
- Keep `get_purged_embargoed_splits` in the file, unused, for future CV re-enable.

### 4.5 `train.py`

- Read dataset parquet + `columns.json`. `X` = active feature columns; targets = active label columns (may include `label_any`).
- Models = enabled subset of lightgbm/catboost/elasticnet; construction unchanged (config params + `random_state`; elasticnet keeps its StandardScaler pipeline).
- Assert with clear messages: ≥ 1 active feature, ≥ 1 active label, ≥ 1 enabled model.
- One split from `get_holdout_split`, shared by all (label, model) pairs.
- Per active label × enabled model: fit on train rows, `predict_proba` on validation rows, keep the True-class probability.
- Startup print: n rows, active feature count, active label names, enabled models, train/val sizes. One progress line per (label, model).
- Predictions parquet at `predictions_output_path`: `ts` int64, `label` str (column name, e.g. `"label_h60"`, `"label_any"`), `model` str, `y_true` uint8, `y_prob` float32. Rows = `n_val × n_active_labels × n_enabled_models`. No `fold` column.

### 4.6 `evaluate.py`

- Read predictions parquet + `columns.json`; drop rows whose label is inactive at evaluation time.
- Per (model, label): Precision on the True class at `classification_threshold`, Recall, F1, ROC-AUC, prevalence, `n` — metric definitions unchanged from v1.
- Markdown table to stdout (rows = model × label; `label` replaces `horizon`); same records as JSON to `metrics_output_path`. `metrics.json` is now gitignored — there is no committed result artifact.

### 4.7 `requirements.txt`

Unchanged.

## 5. Non-Goals / Out of Scope

- No hyperparameter changes, tuning, or threshold optimization.
- No changes outside `scripts/studies/polynomial_kinematic_ml/`.
- No CLI flags — behavior is driven by `config.py` and `columns.json` only.
- No k-fold CV execution (splitter retained but unused); no tests, plots, or notebooks.
- No migration/cleanup code for old `data/processed/` artifacts.

## 6. Assumptions

- "OR label" = inclusive OR over every `label_h{H}` column of the build; name `label_any`; always last.
- Flag preservation across rebuilds is keyed by column name.
- Train = earliest ~80%, validation = latest ~20%; the required separation is the purge gap (`max(forward_horizons)`) before the validation start; embargo is kept in the API for symmetry but has no effect here.
- With CV gone, the "OOF" naming is dropped: file renamed `validation_predictions.parquet`, `fold` column removed.
- Artifact filenames are simplified (`dataset.parquet`, …) since the folder already names the study.

## 7. Acceptance Criteria

1. The three entry points generate files only under `data/polynomial_kinematic_ml/`: `dataset.parquet`, `columns.json`, `validation_predictions.parquet`, `metrics.json`. Nothing is generated in the scripts tree; no reference to `data/processed/` remains.
2. Dataset has 3·W + Hn + 2 columns in §4.2 order for whatever lists config holds; changing the lists changes the column count with no code edits.
3. `label_any` equals the element-wise OR of all `label_h` columns, uint8, last column.
4. `columns.json` has one line per column with `index`/`name`/`role`/`active`; hand-set `false` flags survive a rebuild for surviving column names.
5. Deactivating a feature reduces the trained feature count (visible in the startup print); deactivating a label removes it from predictions and metrics; disabling a model removes it from both.
6. Split: `max(train_idx) <= v0 - purge - 1`, validation = `[v0, n-1]`, `purge = max(forward_horizons)`; both non-empty; no random splitting anywhere.
7. Predictions parquet rows = `n_val × n_active_labels × n_enabled_models` with the §4.5 schema.
8. `evaluate.py` prints the matrix and writes valid JSON with exactly `n_active_labels × n_enabled_models` records.
9. All three commands run from the repo root via `python -m`.

## 8. Open Questions (non-blocking — defaults chosen above)

- Artifact filenames were simplified inside the new folder; say so if the v1 names should be kept instead.
- `role` was added to `columns.json` beyond the requested index/name/active to make hand-editing safer; drop if unwanted.
- `cv_splits` is removed rather than kept dormant — git history plus the retained splitter make CV easy to restore.
- `label_any` is fixed at build time; if it should instead OR only the currently-active labels at train time, say so.

## 9. Notes for the Downstream Coding Agent

- Only `config.py` references `data/processed/` (verified by grep) — no external consumers of the old paths. Old local artifacts there are gitignored; write no cleanup code.
- One-line-per-column JSON: `"[\n" + ",\n".join(json.dumps(e) for e in entries) + "\n]"` — `json.dump(indent=…)` would spread each object over many lines.
- Flag carry-over: read the existing `columns.json` inside try/except; on any parse failure write fresh defaults.
- True-class probability via `list(model.classes_).index(1)` as in v1 — works for all three models.
- Expected runtime ≈ 1/5 of v1 (one fit per label–model instead of five); CatBoost still dominates.
- Markdown table: the former `horizon` column becomes `label` (~10-char strings such as `label_h120`, `label_any`).
- Keep all docstrings terse per the writing-style rule.
