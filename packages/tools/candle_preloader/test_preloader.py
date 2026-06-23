#!/usr/bin/env python3
"""Manual test script for preload_candles.

Usage examples
--------------
# All assets, all dates
python test_preloader.py

# Specific assets
python test_preloader.py --assets btcusdt,ethusdt

# Date range
python test_preloader.py --assets btcusdt --from 2025-01-01 --to 2025-02-28

# Full date-time precision
python test_preloader.py --assets btcusdt --from "2025-01-01 00:00:00" --to "2025-01-31 23:59:00"
"""

import argparse
import sys
from pathlib import Path

# Allow running from any directory without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from packages.tools.candle_preloader import preload_candles, ALL_ASSETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test candle_preloader.preload_candles()")
    parser.add_argument(
        "--assets",
        metavar="SYMBOL[,SYMBOL...]",
        help="Comma-separated asset symbols (default: all assets)",
    )
    parser.add_argument(
        "--from",
        dest="start",
        metavar="DATETIME",
        help='Inclusive start, e.g. "2025-01-01" or "2025-01-01 18:00:00"',
    )
    parser.add_argument(
        "--to",
        dest="end",
        metavar="DATETIME",
        help='Inclusive end, e.g. "2025-02-28" or "2025-02-28 23:59:00"',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assets = [a.strip().lower() for a in args.assets.split(",")] if args.assets else None

    print("=" * 60)
    print("candle_preloader — test run")
    print(f"  assets : {assets or ALL_ASSETS}")
    print(f"  start  : {args.start or '(all)'}")
    print(f"  end    : {args.end or '(all)'}")
    print("=" * 60)

    result = preload_candles(assets=assets, start=args.start, end=args.end)

    print("\n--- results ---")
    ok, missing = [], []
    for asset in (assets or ALL_ASSETS):
        if asset in result:
            p = result[asset]
            size_mb = p.stat().st_size / 1_048_576
            print(f"  {asset:<12} OK  {size_mb:7.2f} MB  {p}")
            ok.append(asset)
        else:
            print(f"  {asset:<12} MISSING")
            missing.append(asset)

    print(f"\n  {len(ok)} succeeded, {len(missing)} missing")
    if missing:
        print(f"  missing: {missing}")
        sys.exit(1)


if __name__ == "__main__":
    main()
