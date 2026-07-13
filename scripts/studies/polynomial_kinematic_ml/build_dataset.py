"""Build the kinematic-features + horizon-labels dataset parquet, plus columns.json."""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numba as nb
import numpy as np
import pandas as pd

from packages.candle_loader import load_candles
from packages.indicators.motion import calculate_market_kinematics
from scripts.studies.polynomial_kinematic_ml.config import CONFIG


@nb.njit(cache=True)
def _labels_for_horizon(vwap: np.ndarray, horizon: int, bps_multiplier: float) -> np.ndarray:
    """label_h[t] = 1 if the OLS slope of vwap[t+1 : t+1+horizon] over 0..horizon-1
    is >= vwap[t+1] * bps_multiplier / horizon, else 0. Undefined positions (t with
    no full look-ahead) stay 0; the caller trims those away positionally.

    Running-accumulator recurrence: S = sum(y), W = sum(i*y) over the local window;
    slope = (W - S*(H-1)/2) / (H*(H^2-1)/12), the closed-form OLS slope for x=0..H-1.
    """
    n = vwap.shape[0]
    labels = np.zeros(n, dtype=np.uint8)
    max_t = n - 1 - horizon
    if max_t < 0:
        return labels

    s = 0.0
    w = 0.0
    for i in range(horizon):
        y = vwap[1 + i]
        s += y
        w += i * y

    half = (horizon - 1) / 2.0
    denom = horizon * (horizon * horizon - 1) / 12.0
    t = 0
    while True:
        slope = (w - s * half) / denom
        p_next = vwap[t + 1]
        if slope >= p_next * bps_multiplier / horizon:
            labels[t] = 1
        if t == max_t:
            break
        y_drop = vwap[t + 1]
        y_add = vwap[t + 1 + horizon]
        w = w - (s - y_drop) + (horizon - 1) * y_add
        s = s - y_drop + y_add
        t += 1
    return labels


def _column_role(name: str) -> str:
    if name == "ts":
        return "meta"
    if name.startswith("label_"):
        return "label"
    return "feature"


def _write_columns_json(column_names: list, out_path: Path) -> None:
    """One-line-per-column columns.json. Carries over 'active' by name from an
    existing file (new names default True, vanished names dropped, index rewritten);
    any read/parse failure falls back to fresh defaults."""
    existing_active = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            existing_active = {e["name"]: bool(e["active"]) for e in existing}
        except Exception:
            existing_active = {}

    entries = [
        {"index": i, "name": name, "role": _column_role(name),
         "active": existing_active.get(name, True)}
        for i, name in enumerate(column_names)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("[\n" + ",\n".join(json.dumps(e) for e in entries) + "\n]\n")


def main():
    lookback_max = max(CONFIG.lookback_windows)
    horizon_max = max(CONFIG.forward_horizons)

    start_dt = datetime.strptime(CONFIG.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    stop_dt = datetime.strptime(CONFIG.stop_date, "%Y-%m-%d").replace(
        hour=23, minute=59, tzinfo=timezone.utc)
    load_start = start_dt - timedelta(minutes=lookback_max)
    load_end = stop_dt + timedelta(minutes=horizon_max)

    data = load_candles(
        CONFIG.asset,
        load_start.strftime("%Y-%m-%d %H:%M:%S"),
        load_end.strftime("%Y-%m-%d %H:%M:%S"),
    )

    vwap = data[:, 8].copy()
    close = data[:, 4]
    bad = ~np.isfinite(vwap)
    vwap[bad] = close[bad]
    vwap = np.ascontiguousarray(vwap, dtype=np.float64)
    n = len(vwap)

    columns = {"ts": data[:, 0].astype(np.int64)}
    for w in CONFIG.lookback_windows:
        kin = calculate_market_kinematics(vwap, w)
        columns[f"vel_w{w}"] = kin[:, 0]
        columns[f"acc_w{w}"] = kin[:, 1]
        columns[f"jerk_w{w}"] = kin[:, 2]

    label_any = np.zeros(n, dtype=np.uint8)
    for h in CONFIG.forward_horizons:
        label_h = _labels_for_horizon(vwap, h, CONFIG.bps_multiplier)
        columns[f"label_h{h}"] = label_h
        label_any |= label_h
    columns["label_any"] = label_any  # OR of all horizon labels; always last column

    t_min = lookback_max - 1
    t_max = n - 1 - horizon_max
    pos_mask = np.zeros(n, dtype=bool)
    if t_max >= t_min:
        pos_mask[t_min:t_max + 1] = True

    ts = columns["ts"]
    start_ms = int(start_dt.timestamp() * 1000)
    stop_ms = int(stop_dt.timestamp() * 1000)
    date_mask = (ts >= start_ms) & (ts <= stop_ms)

    keep = pos_mask & date_mask
    df = pd.DataFrame(columns).loc[keep].reset_index(drop=True)
    for h in CONFIG.forward_horizons:
        df[f"label_h{h}"] = df[f"label_h{h}"].astype(np.uint8)
    df["label_any"] = df["label_any"].astype(np.uint8)

    out_path = REPO_ROOT / CONFIG.dataset_output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"dataset written: {out_path} shape={df.shape}")

    columns_path = REPO_ROOT / CONFIG.columns_output_path
    _write_columns_json(list(df.columns), columns_path)
    print(f"columns written: {columns_path}")


if __name__ == "__main__":
    main()
