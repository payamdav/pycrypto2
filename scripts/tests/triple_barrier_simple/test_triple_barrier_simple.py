import os
import sys
import time
from pathlib import Path

import numpy as np

# Make `packages` importable and keep the candle cache at <repo>/data regardless of invocation dir
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from packages.candle_loader import load_candles
from packages.traders_indicators import triple_barrier_simple


def str2bool(v: str) -> bool:
    return v.strip().lower() not in ("0", "false", "no")


def main() -> None:
    asset = "btcusdt"
    date_from = "2024-01-01 00:00:00"
    date_to = "2026-06-02 00:00:00"

    upper_barrier_bps = 20.0
    lower_barrier_bps = 20.0
    look_ahead = 240
    next_entry = True

    data = load_candles(asset, date_from, date_to)
    prices = np.ascontiguousarray(data[:, 8])  # vwap
    n = prices.shape[0]

    triple_barrier_simple(prices[:1], upper_barrier_bps, lower_barrier_bps, look_ahead, next_entry)  # jit warm-up

    t0 = time.perf_counter()
    labels = triple_barrier_simple(prices, upper_barrier_bps, lower_barrier_bps, look_ahead, next_entry)
    elapsed = time.perf_counter() - t0

    n_tp = int(np.sum(labels == 1.0))
    n_sl = int(np.sum(labels == -1.0))
    n_tl = int(np.sum(labels == 0.0))

    print(f"candles {n:,}")
    print(f"take profit (1.0)  {n_tp:,}")
    print(f"stop loss (-1.0)   {n_sl:,}")
    print(f"time limit (0.0)   {n_tl:,}")
    print(f"triple_barrier_simple: {elapsed:.4f}s (post-JIT call, {n:,} items)")


if __name__ == "__main__":
    main()
