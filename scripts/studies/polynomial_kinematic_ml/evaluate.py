"""Precision-focused metrics matrix over OOF predictions: stdout markdown + metrics.json."""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from scripts.studies.polynomial_kinematic_ml.config import CONFIG


def main():
    oof = pd.read_parquet(REPO_ROOT / CONFIG.oof_output_path)
    threshold = CONFIG.classification_threshold

    records = []
    for (model_name, horizon), g in oof.groupby(["model", "horizon"]):
        y_true = g["y_true"].to_numpy()
        y_prob = g["y_prob"].to_numpy()
        y_pred = (y_prob >= threshold).astype(np.uint8)
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
        records.append({
            "model": model_name,
            "horizon": int(horizon),
            "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
            "roc_auc": float(auc),
            "prevalence": float(y_true.mean()),
            "n": int(len(g)),
        })
    records.sort(key=lambda r: (r["model"], r["horizon"]))

    print("True-class Precision is the headline metric: low accuracy is acceptable if "
          "Precision is high when the model flags True.\n")
    print("| model      | horizon | precision | recall | f1     | roc_auc | prevalence |      n |")
    print("|------------|---------|-----------|--------|--------|---------|------------|--------|")
    for r in records:
        print(f"| {r['model']:<10} | {r['horizon']:>7} | {r['precision']:>9.4f} "
              f"| {r['recall']:>6.4f} | {r['f1']:>6.4f} | {r['roc_auc']:>7.4f} "
              f"| {r['prevalence']:>10.4f} | {r['n']:>6} |")

    out_path = REPO_ROOT / CONFIG.metrics_output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nMetrics written: {out_path}")


if __name__ == "__main__":
    main()
