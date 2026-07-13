"""Train LightGBM / CatBoost / ElasticNet-logistic per active label with a chronological
holdout split (columns.json + config.py enable flags control what runs); save
validation predictions."""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.studies.polynomial_kinematic_ml.config import CONFIG
from scripts.studies.polynomial_kinematic_ml.validation import get_holdout_split


def _build_model(name: str):
    if name == "lightgbm":
        return LGBMClassifier(**CONFIG.lgbm_params, random_state=CONFIG.random_state)
    if name == "catboost":
        return CatBoostClassifier(**CONFIG.catboost_params, random_state=CONFIG.random_state)
    if name == "elasticnet":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(**CONFIG.elasticnet_params, random_state=CONFIG.random_state),
        )
    raise ValueError(f"unknown model: {name}")


def _enabled_models() -> list[str]:
    return [name for name, enabled in [
        ("lightgbm", CONFIG.lgbm_enabled),
        ("catboost", CONFIG.catboost_enabled),
        ("elasticnet", CONFIG.elasticnet_enabled),
    ] if enabled]


def main():
    df = pd.read_parquet(REPO_ROOT / CONFIG.dataset_output_path)
    columns_meta = json.loads((REPO_ROOT / CONFIG.columns_output_path).read_text())

    feature_cols = [c["name"] for c in columns_meta if c["role"] == "feature" and c["active"]]
    label_cols = [c["name"] for c in columns_meta if c["role"] == "label" and c["active"]]
    model_names = _enabled_models()

    assert len(feature_cols) > 0, "no active features in columns.json"
    assert len(label_cols) > 0, "no active labels in columns.json"
    assert len(model_names) > 0, "no enabled models in config.py"

    x_full = df[feature_cols].to_numpy(dtype=np.float64)
    ts_full = df["ts"].to_numpy(dtype=np.int64)
    n = len(df)

    purge = max(CONFIG.forward_horizons)
    embargo = max(CONFIG.lookback_windows)
    train_idx, val_idx = get_holdout_split(n, CONFIG.train_fraction, purge, embargo)

    print(f"rows={n} features={len(feature_cols)} labels={label_cols} "
          f"models={model_names} train={len(train_idx)} val={len(val_idx)}")

    chunks = []
    for label_col in label_cols:
        y_full = df[label_col].to_numpy(dtype=np.uint8)
        for model_name in model_names:
            model = _build_model(model_name)
            model.fit(x_full[train_idx], y_full[train_idx])
            proba = model.predict_proba(x_full[val_idx])
            true_col = list(model.classes_).index(1)
            y_prob = proba[:, true_col].astype(np.float32)

            n_val = len(val_idx)
            chunks.append(pd.DataFrame({
                "ts": ts_full[val_idx],
                "label": np.full(n_val, label_col, dtype=object),
                "model": np.full(n_val, model_name, dtype=object),
                "y_true": y_full[val_idx],
                "y_prob": y_prob,
            }))
            print(f"label={label_col:<12} model={model_name:<10} val={n_val:>7} done")

    predictions = pd.concat(chunks, ignore_index=True)
    out_path = REPO_ROOT / CONFIG.predictions_output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out_path, index=False)
    print(f"validation predictions written: {out_path} shape={predictions.shape}")


if __name__ == "__main__":
    main()
