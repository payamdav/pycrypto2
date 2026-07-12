import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

# Make `packages` importable and keep the candle cache at <repo>/data regardless of invocation dir
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from packages.candle_loader import load_candles

# =====================================================================
#                            PARAMETERS
#            (edit here and re-run to check behavior)
# =====================================================================
ASSET = "btcusdt"
DATE_FROM = "2026-04-20 00:00:00"  # inclusive start (UTC)
N_MINUTES = 1440                   # number of 1m candles from DATE_FROM

K = 5               # short lookback for volatility-normalized velocity (minutes)
THETA_BASE = 0.5     # |z| below this -> Base
THETA_FAST = 1.5     # |z| above this -> Fast Rally / Fast Drop

# Spec suggests 45, but with k=10 the longest raw directional run on this date
# is ~15 min, so anything above ~20 collapses the whole day into a single Base
# segment. 20 lands on the spec's target of 10-15 transitions/day.
MIN_DURATION = 5    # inertia filter: regime blocks shorter than this are merged (minutes)
SEARCH_WINDOW = 30   # pivot search radius around each smoothed regime boundary (minutes)

OUTPUT_HTML = Path(__file__).with_suffix(".html")
# =====================================================================

REGIME_NAMES = {0: "Fast Drop", 1: "Drop", 2: "Base", 3: "Rally", 4: "Fast Rally"}
# Diverging fills: hue = direction (red down / blue up), depth = speed, gray = base
REGIME_FILL = {
    0: "#ee8a89",  # Fast Drop
    1: "#f6c2c1",  # Drop
    2: "#f0efec",  # Base
    3: "#b7d3f6",  # Rally
    4: "#6da7ec",  # Fast Rally
}


@njit(cache=True)
def _rolling_window_std(data: np.ndarray, window: int) -> np.ndarray:
    """Computes O(N) rolling standard deviation using a sliding window accumulation."""
    n = len(data)
    out = np.zeros(n, dtype=np.float64)
    if window > n:
        return out

    current_sum = 0.0
    current_sum_sq = 0.0

    # Initialize first window
    for i in range(window):
        val = data[i]
        current_sum += val
        current_sum_sq += val * val

    inv_w = 1.0 / window
    var = (current_sum_sq * inv_w) - (current_sum * inv_w) ** 2
    out[window - 1] = np.sqrt(max(var, 0.0))

    # Slide across the remaining elements
    for i in range(window, n):
        old = data[i - window]
        new = data[i]
        current_sum += new - old
        current_sum_sq += new * new - old * old
        var = (current_sum_sq * inv_w) - (current_sum * inv_w) ** 2
        out[i] = np.sqrt(max(var, 0.0))

    return out


@njit(parallel=True, cache=True)
def calculate_regimes(
    prices: np.ndarray,
    lookback: int,
    theta_base: float,
    theta_fast: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stages 1 & 2: volatility-normalized velocity Z-score, classified into raw regimes.
    Regime Codes: 0=Fast Drop, 1=Drop, 2=Base, 3=Rally, 4=Fast Rally
    Warm-up bars (i < lookback) default to Base per spec.
    """
    n = len(prices)
    z_scores = np.zeros(n, dtype=np.float64)
    regimes = np.full(n, 2, dtype=np.int8)

    # Calculate log returns
    log_prices = np.log(prices)
    log_rets = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        log_rets[i] = log_prices[i] - log_prices[i-1]

    # Get local volatility profile
    sigmas = _rolling_window_std(log_rets, lookback)
    sqrt_k = np.sqrt(lookback)

    # Classify windows in parallel loops
    for i in prange(lookback, n):
        sigma = sigmas[i]
        if sigma <= 1e-8:
            sigma = 1e-6 # Avoid division by zero in dead zones

        # Dimensionless Z-score using log variance
        z = (log_prices[i] - log_prices[i - lookback]) / (sigma * sqrt_k)
        z_scores[i] = z

        if z > theta_fast:
            regimes[i] = 4       # Fast Rally
        elif z > theta_base:
            regimes[i] = 3       # Rally
        elif z < -theta_fast:
            regimes[i] = 0       # Fast Drop
        elif z < -theta_base:
            regimes[i] = 1       # Drop
        else:
            regimes[i] = 2       # Base

    return z_scores, regimes


@njit(cache=True)
def inertia_filter(regimes: np.ndarray, min_duration: int) -> np.ndarray:
    """
    Stage 3: run-length inertia filter. Repeatedly merges the shortest
    contiguous block below min_duration into its dominant (longer) neighbor,
    coalescing equal neighbors, until every block persists >= min_duration.
    Shortest-first dominant-neighbor merging lets tiny gaps be absorbed into
    directional runs; the pseudo-code's always-merge-into-preceding variant
    collapses everything into one block when k is short.
    """
    n = len(regimes)
    starts = np.zeros(n, dtype=np.int64)
    ends = np.zeros(n, dtype=np.int64)
    codes = np.zeros(n, dtype=np.int8)

    # Run-length encode
    m = 0
    s = 0
    for i in range(1, n + 1):
        if i == n or regimes[i] != regimes[s]:
            starts[m] = s
            ends[m] = i - 1
            codes[m] = regimes[s]
            m += 1
            s = i

    while m > 1:
        # Shortest run below the threshold
        best = -1
        best_len = min_duration
        for r in range(m):
            length = ends[r] - starts[r] + 1
            if length < best_len:
                best_len = length
                best = r
        if best == -1:
            break

        # Dominant (longer) neighbor; prefer the preceding one on ties
        if best == 0:
            tgt = 1
        elif best == m - 1:
            tgt = m - 2
        else:
            prev_len = ends[best - 1] - starts[best - 1] + 1
            next_len = ends[best + 1] - starts[best + 1] + 1
            tgt = best - 1 if prev_len >= next_len else best + 1

        # Absorb the short run into the target and delete it
        if tgt < best:
            ends[tgt] = ends[best]
        else:
            starts[tgt] = starts[best]
        for r in range(best, m - 1):
            starts[r] = starts[r + 1]
            ends[r] = ends[r + 1]
            codes[r] = codes[r + 1]
        m -= 1
        t = tgt if tgt < best else tgt - 1

        # Coalesce the target with now-adjacent runs of the same regime
        if t > 0 and codes[t - 1] == codes[t]:
            ends[t - 1] = ends[t]
            for r in range(t, m - 1):
                starts[r] = starts[r + 1]
                ends[r] = ends[r + 1]
                codes[r] = codes[r + 1]
            m -= 1
            t -= 1
        if t < m - 1 and codes[t + 1] == codes[t]:
            ends[t] = ends[t + 1]
            for r in range(t + 1, m - 1):
                starts[r] = starts[r + 1]
                ends[r] = ends[r + 1]
                codes[r] = codes[r + 1]
            m -= 1

    out = np.empty(n, dtype=np.int8)
    for r in range(m):
        for i in range(starts[r], ends[r] + 1):
            out[i] = codes[r]
    return out


@njit(cache=True)
def boundary_turning_points(
    prices: np.ndarray,
    smooth_regimes: np.ndarray,
    search_window: int
) -> np.ndarray:
    """
    Stage 4: boundary-aligned pivot search. At each smoothed regime transition,
    scan [T-W, T+W] for the local extreme matching the shift direction.
    Output: 0=No Pivot, 1=Trough (Bottom), 2=Peak (Top)
    """
    n = len(prices)
    pivots = np.zeros(n, dtype=np.int8)

    for i in range(1, n):
        curr = smooth_regimes[i]
        prev = smooth_regimes[i - 1]
        if curr == prev:
            continue

        start_idx = max(0, i - search_window)
        end_idx = min(n - 1, i + search_window)
        is_prev_up = prev >= 3
        is_curr_up = curr >= 3
        is_prev_down = prev <= 1
        is_curr_down = curr <= 1

        # Trough: a downward regime ends, or an upward one begins.
        # Peak: an upward regime ends, or a downward one begins.
        # (Union of the spec's prose and pseudo-code rules, so every
        # directional segment gets both of its boundary extremes marked.)
        if (is_prev_down and not is_curr_down) or (is_curr_up and not is_prev_up):
            min_idx = start_idx
            for w in range(start_idx + 1, end_idx + 1):
                if prices[w] < prices[min_idx]:
                    min_idx = w
            pivots[min_idx] = 1
        elif (is_prev_up and not is_curr_up) or (is_curr_down and not is_prev_down):
            max_idx = start_idx
            for w in range(start_idx + 1, end_idx + 1):
                if prices[w] > prices[max_idx]:
                    max_idx = w
            pivots[max_idx] = 2

    return pivots


def regime_segments(regimes: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous (start_idx, end_idx_inclusive, code) runs."""
    segs = []
    start = 0
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[start]:
            segs.append((start, i - 1, int(regimes[start])))
            start = i
    segs.append((start, len(regimes) - 1, int(regimes[start])))
    return segs


def print_report(times, prices, z_scores, raw_regimes, smooth_regimes, pivots, segs, elapsed):
    raw_shifts = int(np.sum(raw_regimes[1:] != raw_regimes[:-1]))
    print(f"\nProcessed {len(prices):,} bars in {elapsed:.4f} s (incl. JIT compile on first run)")
    print(f"Range: {times[0]} .. {times[-1]}  |  first {K} bars are Z-score warm-up (default Base)")
    print(f"Raw regime shifts: {raw_shifts}  ->  after inertia filter (>= {MIN_DURATION} min): {len(segs) - 1}")

    raw_runs = regime_segments(raw_regimes)
    longest = {c: 0 for c in REGIME_NAMES}
    for a, b, code in raw_runs:
        longest[code] = max(longest[code], b - a + 1)
    print("Longest raw run per regime (MIN_DURATION above the directional ones erases them): "
          + ", ".join(f"{REGIME_NAMES[c]} {longest[c]}m" for c in (4, 3, 2, 1, 0)))

    print("\n--- Smoothed regime segments ---")
    print(f"{'#':>3}  {'start':<16} {'end':<16} {'min':>5}  {'regime':<10} {'p_start':>10} {'p_end':>10} {'chg%':>7}")
    for k, (a, b, code) in enumerate(segs, 1):
        chg = (prices[b] / prices[a] - 1.0) * 100.0
        print(f"{k:>3}  {times[a]:%Y-%m-%d %H:%M} {times[b]:%Y-%m-%d %H:%M} "
              f"{b - a + 1:>5}  {REGIME_NAMES[code]:<10} {prices[a]:>10.2f} {prices[b]:>10.2f} {chg:>+7.3f}")

    print("\n--- Regime summary (smoothed) ---")
    total = len(smooth_regimes)
    for code in (4, 3, 2, 1, 0):
        cnt = int(np.sum(smooth_regimes == code))
        n_segs = sum(1 for s in segs if s[2] == code)
        print(f"{REGIME_NAMES[code]:<10} {cnt:>5} min  ({cnt / total * 100:5.1f}%)  in {n_segs} segments")

    troughs = np.where(pivots == 1)[0]
    peaks = np.where(pivots == 2)[0]
    print(f"\n--- Turning points (boundary-aligned): {len(troughs)} troughs, {len(peaks)} peaks ---")
    for i in sorted(np.concatenate([troughs, peaks])):
        kind = "TROUGH" if pivots[i] == 1 else "PEAK  "
        print(f"{kind}  {times[i]:%Y-%m-%d %H:%M}  price {prices[i]:>10.2f}  z {z_scores[i]:>+6.2f}")


def build_chart(times, prices, z_scores, pivots, segs):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
        subplot_titles=(None, "Trend Z-score"),
    )

    # Legend proxies for the five regimes (NaN y: shows in legend, draws nothing)
    for code in (4, 3, 2, 1, 0):
        fig.add_trace(go.Scatter(
            x=[times[0].isoformat()], y=[np.nan], mode="markers", name=REGIME_NAMES[code],
            marker=dict(symbol="square", size=12, color=REGIME_FILL[code],
                        line=dict(width=1, color="rgba(11,11,11,0.25)")),
        ), row=1, col=1)

    # Price line
    fig.add_trace(go.Scatter(
        x=times, y=prices, mode="lines", name="Close",
        line=dict(color="#0b0b0b", width=1.6),
        hovertemplate="%{y:,.2f}<extra>Close</extra>",
    ), row=1, col=1)

    # Turning points
    troughs = np.where(pivots == 1)[0]
    peaks = np.where(pivots == 2)[0]
    fig.add_trace(go.Scatter(
        x=times[troughs], y=prices[troughs], mode="markers", name="Trough",
        marker=dict(symbol="triangle-up", size=11, color="#104281",
                    line=dict(width=1, color="#fcfcfb")),
        hovertemplate="%{y:,.2f}<extra>Trough</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=times[peaks], y=prices[peaks], mode="markers", name="Peak",
        marker=dict(symbol="triangle-down", size=11, color="#a02b2b",
                    line=dict(width=1, color="#fcfcfb")),
        hovertemplate="%{y:,.2f}<extra>Peak</extra>",
    ), row=1, col=1)

    # Z-score panel with regime thresholds
    fig.add_trace(go.Scatter(
        x=times, y=z_scores, mode="lines", name="Z-score", showlegend=False,
        line=dict(color="#52514e", width=1.2),
        hovertemplate="%{y:+.2f}<extra>Z</extra>",
    ), row=2, col=1)

    # Regime background washes across both panels (after traces: add_vrect skips empty subplots)
    for a, b, code in segs:
        fig.add_vrect(
            x0=times[a].isoformat(), x1=times[min(b + 1, len(times) - 1)].isoformat(),
            fillcolor=REGIME_FILL[code], opacity=1.0, layer="below", line_width=0,
            row="all", col=1,
        )

    for level, label in ((THETA_FAST, "+θ fast"), (THETA_BASE, "+θ base"),
                         (-THETA_BASE, "-θ base"), (-THETA_FAST, "-θ fast")):
        fig.add_hline(y=level, line=dict(color="#898781", width=1, dash="dot"),
                      annotation_text=label, annotation_font=dict(size=10, color="#898781"),
                      annotation_position="right", row=2, col=1)

    fig.update_layout(
        title=dict(
            text=f"{ASSET.upper()} 1m — smoothed regime sectorization & boundary-aligned turning points<br>"
                 f"<sup>{DATE_FROM} UTC + {N_MINUTES} min · k {K} · θ base {THETA_BASE} / fast {THETA_FAST} · "
                 f"inertia {MIN_DURATION} min · pivot search ±{SEARCH_WINDOW} min</sup>",
            font=dict(size=17, color="#0b0b0b"),
        ),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color="#52514e"),
        paper_bgcolor="#f9f9f7", plot_bgcolor="#fcfcfb",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.015, xanchor="right", x=1,
                    font=dict(size=12)),
        margin=dict(l=60, r=70, t=90, b=40),
        height=760,
    )
    fig.update_xaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", showline=True, zeroline=False)
    fig.update_yaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", showline=True, zeroline=False)
    fig.update_yaxes(title_text="Price (USDT)", row=1, col=1, tickformat=",.0f")
    fig.update_yaxes(title_text="Z", row=2, col=1)

    fig.write_html(OUTPUT_HTML)
    print(f"\nChart written to: {OUTPUT_HTML}")


if __name__ == "__main__":
    date_to = (datetime.strptime(DATE_FROM, "%Y-%m-%d %H:%M:%S")
               + timedelta(minutes=N_MINUTES - 1)).strftime("%Y-%m-%d %H:%M:%S")
    data = load_candles(ASSET, DATE_FROM, date_to)
    times = pd.to_datetime(data[:, 0], unit="ms")
    prices = np.ascontiguousarray(data[:, 4])

    start_time = time.perf_counter()
    z_scores, raw_regimes = calculate_regimes(prices, K, THETA_BASE, THETA_FAST)
    smooth_regimes = inertia_filter(raw_regimes, MIN_DURATION)
    pivots = boundary_turning_points(prices, smooth_regimes, SEARCH_WINDOW)
    elapsed = time.perf_counter() - start_time

    segs = regime_segments(smooth_regimes)
    print_report(times, prices, z_scores, raw_regimes, smooth_regimes, pivots, segs, elapsed)
    build_chart(times, prices, z_scores, pivots, segs)
