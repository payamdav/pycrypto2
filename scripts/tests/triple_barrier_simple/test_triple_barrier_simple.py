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
    if len(sys.argv) < 4:
        print("usage: test_triple_barrier_simple.py <asset> <date_from> <date_to> "
              "[upper_barrier_bps=20] [lower_barrier_bps=20] [look_ahead=240] [next_entry=True]")
        sys.exit(1)

    asset, date_from, date_to = sys.argv[1], sys.argv[2], sys.argv[3]
    upper_barrier_bps = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0
    lower_barrier_bps = float(sys.argv[5]) if len(sys.argv) > 5 else 20.0
    look_ahead = int(sys.argv[6]) if len(sys.argv) > 6 else 240
    next_entry = str2bool(sys.argv[7]) if len(sys.argv) > 7 else True

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
