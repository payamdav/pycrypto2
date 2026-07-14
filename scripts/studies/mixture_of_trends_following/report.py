"""Evaluate trigger -> l_slopes prediction on normal items; print + save motf_report_{asset}.json.
Run: python3 report.py [tag]"""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np

from scripts.studies.mixture_of_trends_following.common import (
    COL_IMBALANCES, COL_LSLOPES, COL_SLOPES, COL_TRIGGER, COL_TS, LABEL_COLS,
    assets_of, cli_tag, data_dir, load_params, parse_ts_ms, timed,
)


def _safe_div(a, b):
    """None (-> JSON null) when a/b is undefined (missing input or b == 0); keeps the report json strictly valid."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def normal_mask(cache: np.ndarray, params: dict) -> np.ndarray:
    """LB-1 <= i < n-LA (LB/LA = longest configured look-back/look-ahead), AND ts in [date_from, date_to]."""
    n = cache.shape[0]
    lb = max(max(params["slope_windows"]), max(params["imbalance_windows"]))
    la = max(params["label_windows"])
    idx = np.arange(n)
    normal = (idx >= lb - 1) & (idx < n - la)

    ts = cache[:, COL_TS]
    from_ms = parse_ts_ms(params["date_from"])
    to_ms = parse_ts_ms(params["date_to"])
    date_ok = (ts >= from_ms) & (ts <= to_ms)
    return normal & date_ok


def evaluate_asset(cache: np.ndarray, params: dict) -> dict:
    mask = normal_mask(cache, params)
    n_total = cache.shape[0]
    n_eval = int(mask.sum())

    slopes = cache[mask, COL_SLOPES]
    imbalances = cache[mask, COL_IMBALANCES]
    trigger = cache[mask, COL_TRIGGER]
    l_slopes = cache[mask, COL_LSLOPES]

    def count_pct(x):
        c = int(x.sum())
        return {"count": c, "pct": 100.0 * c / n_eval if n_eval else 0.0}

    tp = int(np.sum((trigger == 1.0) & (l_slopes == 1.0)))
    fp = int(np.sum((trigger == 1.0) & (l_slopes == 0.0)))
    fn = int(np.sum((trigger == 0.0) & (l_slopes == 1.0)))
    tn = int(np.sum((trigger == 0.0) & (l_slopes == 0.0)))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    baseline = _safe_div(tp + fn, n_eval)
    lift = _safe_div(precision, baseline)
    phi_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    phi = _safe_div(tp * tn - fp * fn, phi_den)

    threshold = params["l_slopes_threshold"]
    per_horizon = []
    for col, w in zip(LABEL_COLS, params["label_windows"]):
        lk = cache[mask, col] > threshold
        tp_k = int(np.sum((trigger == 1.0) & lk))
        fp_k = int(np.sum((trigger == 1.0) & ~lk))
        prec_k = _safe_div(tp_k, tp_k + fp_k)
        base_k = _safe_div(int(lk.sum()), n_eval)
        lift_k = _safe_div(prec_k, base_k)
        per_horizon.append({"window": w, "precision": prec_k, "baseline": base_k, "lift": lift_k})

    return {
        "n_total": n_total, "n_eval": n_eval,
        "slopes": count_pct(slopes), "imbalances": count_pct(imbalances),
        "trigger": count_pct(trigger), "l_slopes": count_pct(l_slopes),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "phi": phi, "precision": precision, "recall": recall, "baseline": baseline, "lift": lift,
        "per_horizon": per_horizon,
    }


def _fmt(v) -> str:
    return f"{v:.4f}" if v is not None else "nan"


def print_report(asset: str, r: dict) -> None:
    print(f"[{asset}] candles={r['n_total']} evaluated={r['n_eval']}")
    for key in ("slopes", "imbalances", "trigger", "l_slopes"):
        v = r[key]
        print(f"[{asset}] {key}==1: {v['count']} ({v['pct']:.2f}%)")
    c = r["confusion"]
    print(f"[{asset}] confusion TP={c['tp']} FP={c['fp']} / FN={c['fn']} TN={c['tn']}")
    print(f"[{asset}] phi={_fmt(r['phi'])} precision={_fmt(r['precision'])} recall={_fmt(r['recall'])} "
          f"baseline={_fmt(r['baseline'])} lift={_fmt(r['lift'])}")
    print(f"[{asset}] per-horizon trigger precision:")
    for h in r["per_horizon"]:
        print(f"[{asset}]   window={h['window']:>5} precision={_fmt(h['precision'])} "
              f"baseline={_fmt(h['baseline'])} lift={_fmt(h['lift'])}")


def report_asset(asset: str, params: dict) -> Path:
    cache = np.load(data_dir(params["tag"]) / f"motf_cache_{asset}.npy")
    with timed(f"[{asset}] report"):
        r = evaluate_asset(cache, params)
        print_report(asset, r)
        out = data_dir(params["tag"]) / f"motf_report_{asset}.json"
        out.write_text(json.dumps(r, indent=2))
    print(f"[{asset}] report saved: {out}")
    return out


def report(params: dict) -> list:
    return [report_asset(asset, params) for asset in assets_of(params)]


if __name__ == "__main__":
    tag = cli_tag(sys.argv)
    report(load_params(tag))
