"""Train LightGBM / CatBoost / ElasticNet-logistic per horizon with purged CV; save OOF predictions."""
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
from scripts.studies.polynomial_kinematic_ml.validation import get_purged_embargoed_splits

MODEL_NAMES = ["lightgbm", "catboost", "elasticnet"]


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


def main():
    df = pd.read_parquet(REPO_ROOT / CONFIG.dataset_output_path)
    feature_cols = [c for c in df.columns if c != "ts" and not c.startswith("label_h")]
    x_full = df[feature_cols].to_numpy(dtype=np.float64)
    ts_full = df["ts"].to_numpy(dtype=np.int64)
    n = len(df)

    purge = max(CONFIG.forward_horizons)
    embargo = max(CONFIG.lookback_windows)

    chunks = []
    for horizon in CONFIG.forward_horizons:
        y_full = df[f"label_h{horizon}"].to_numpy(dtype=np.uint8)
        for model_name in MODEL_NAMES:
            for fold, (train_idx, val_idx) in enumerate(
                get_purged_embargoed_splits(n, CONFIG.cv_splits, purge, embargo)
            ):
                model = _build_model(model_name)
                model.fit(x_full[train_idx], y_full[train_idx])
                proba = model.predict_proba(x_full[val_idx])
                true_col = list(model.classes_).index(1)
                y_prob = proba[:, true_col].astype(np.float32)

                n_val = len(val_idx)
                chunks.append(pd.DataFrame({
                    "ts": ts_full[val_idx],
                    "fold": np.full(n_val, fold, dtype=np.int8),
                    "horizon": np.full(n_val, horizon, dtype=np.int16),
                    "model": np.full(n_val, model_name, dtype=object),
                    "y_true": y_full[val_idx],
                    "y_prob": y_prob,
                }))
                print(f"horizon={horizon:>3} model={model_name:<10} fold={fold} "
                      f"train={len(train_idx):>7} val={n_val:>7} done")

    oof = pd.concat(chunks, ignore_index=True)
    out_path = REPO_ROOT / CONFIG.oof_output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(out_path, index=False)
    print(f"OOF predictions written: {out_path} shape={oof.shape}")


if __name__ == "__main__":
    main()
