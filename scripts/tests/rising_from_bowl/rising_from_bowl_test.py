import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Make `packages` importable and keep the candle cache at <repo>/data regardless of invocation dir
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from packages.candle_loader import load_candles
from packages.pattern_detection import rising_from_bowl_scan, SCAN_COLUMNS

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.json"
DT_FMT = "%Y-%m-%d %H:%M:%S"

DETECTOR_KEYS = (
    "min_bowl_width", "max_bowl_width", "min_bowl_depth_bps",
    "bottom_position_limit", "peak_drawdown_limit_bps", "max_peak_search_width",
)
COL = {name: idx for idx, name in enumerate(SCAN_COLUMNS)}

QUALITATIVE_PALETTE = [
    "#2e6f95", "#c1666b", "#4c9a6a", "#a76fb0", "#c98a3b",
    "#5b7fbd", "#8a8a3c", "#c15a9e", "#3f9e8f", "#b5533c",
]


def to_ms(dt_str: str) -> int:
    return int(datetime.strptime(dt_str, DT_FMT).replace(tzinfo=timezone.utc).timestamp() * 1000)


def sanitize_vwap(data: np.ndarray) -> np.ndarray:
    """Replace non-finite / zero-volume vwap rows with the previous valid value;
    leading bad rows get the first valid value."""
    vwap = data[:, 8].astype(np.float64)
    bad = ~np.isfinite(vwap) | (data[:, 5] == 0.0)
    vwap[bad] = np.nan
    return pd.Series(vwap).ffill().bfill().to_numpy()


def dedupe_bowls(detections: np.ndarray) -> dict:
    """Group scan rows by left_wall_peak_idx: first (earliest) detection per
    bowl, chronologically ordered, plus a re-detection count per bowl."""
    if detections.shape[0] == 0:
        return {"rows": detections, "counts": np.zeros(0, dtype=np.int64)}
    peak_idx = detections[:, COL["left_wall_peak_idx"]]
    _, first_pos, counts = np.unique(peak_idx, return_index=True, return_counts=True)
    order = np.argsort(first_pos)
    return {"rows": detections[first_pos[order]], "counts": counts[order]}


def print_stats(asset, date_from, date_to, n_anchors, detections, bowls) -> None:
    m = detections.shape[0]
    d = bowls["rows"].shape[0]
    print(f"{asset} {date_from} -> {date_to} | candles {n_anchors:,}")
    print(f"detections {m:,} | distinct bowls {d:,}")
    if d == 0:
        print("detections/bowl: n/a (no bowls detected)")
        return
    counts = bowls["counts"]
    print(f"detections/bowl  min {counts.min()}  mean {counts.mean():.2f}  max {counts.max()}")
    rows = bowls["rows"]
    for label, col in (("width(m)", "bowl_width"), ("depth(bps)", "bowl_depth_bps"),
                        ("recovery", "recovery_ratio"), ("r2", "r_squared")):
        v = rows[:, COL[col]]
        print(f"{label:<11} mean {v.mean():.3f}  median {np.median(v):.3f}")


def build_chart(data: np.ndarray, vwap: np.ndarray, bowls: dict, cfg: dict, out_path: Path) -> None:
    import plotly.graph_objects as go

    times = pd.to_datetime(data[:, 0], unit="ms")
    o, h, l, c = data[:, 1], data[:, 2], data[:, 3], data[:, 4]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=times, open=o, high=h, low=l, close=c, name="Candles",
        increasing=dict(line=dict(color="#4c9a6a"), fillcolor="#a9d1b6"),
        decreasing=dict(line=dict(color="#c1666b"), fillcolor="#e8b9ba"),
    ))
    fig.add_trace(go.Scatter(
        x=times, y=vwap, mode="lines", name="VWAP",
        line=dict(color="#2e2e2e", width=1.3),
        hovertemplate="%{y:,.2f}<extra>VWAP</extra>",
    ))

    rows, counts = bowls["rows"], bowls["counts"]
    for n in range(rows.shape[0]):
        row = rows[n]
        color = QUALITATIVE_PALETTE[n % len(QUALITATIVE_PALETTE)]
        group = f"bowl{n + 1}"
        i, t = int(row[COL["left_rim_idx"]]), int(row[COL["right_rim_idx"]])
        t_min = int(row[COL["bottom_idx"]])
        k = int(row[COL["bowl_width"]])
        depth_bps = row[COL["bowl_depth_bps"]]
        pos_ratio = row[COL["bottom_position_ratio"]]
        peak_idx = int(row[COL["left_wall_peak_idx"]])
        peak_price = row[COL["left_wall_peak_price"]]
        recovery = row[COL["recovery_ratio"]]
        a, b, cc = row[COL["fit_coef_a"]], row[COL["fit_coef_b"]], row[COL["fit_coef_c"]]
        r2 = row[COL["r_squared"]]
        theo_idx = row[COL["theoretical_bottom_idx"]]
        det_count = int(counts[n])

        theo_time = times[i] + pd.to_timedelta(theo_idx - i, unit="m")
        info = (
            f"Bowl {n + 1}  (×{det_count} detections)<br>"
            f"width {k} m | depth {depth_bps:.1f} bps | position {pos_ratio:.2f}<br>"
            f"recovery {recovery:.2f} | fit a {a:.3g} | R² {r2:.3f}<br>"
            f"theoretical bottom: {theo_time:%Y-%m-%d %H:%M}"
        )

        xs = np.arange(0, k + 1, dtype=np.float64)
        y_fit = a * xs ** 2 + b * xs + cc
        fig.add_trace(go.Scatter(
            x=times[i:t + 1], y=y_fit, mode="lines",
            name=f"Bowl {n + 1} ×{det_count}", legendgroup=group,
            line=dict(color=color, width=1.6, dash="dot"),
            hovertemplate=info + "<extra></extra>",
        ))

        marker_x = [times[i], times[t_min], times[peak_idx], times[t]]
        marker_y = [vwap[i], vwap[t_min], peak_price, vwap[t]]
        marker_label = ["left rim", "bottom", "left-wall peak", "anchor"]
        point_hover = [
            f"Bowl {n + 1} — {label}<br>{tm:%Y-%m-%d %H:%M}  {yv:,.2f}<br>{info}"
            for label, tm, yv in zip(marker_label, marker_x, marker_y)
        ]
        fig.add_trace(go.Scatter(
            x=marker_x, y=marker_y, mode="markers", showlegend=False, legendgroup=group,
            marker=dict(symbol=["circle-open", "circle", "star", "triangle-up"],
                        size=[9, 9, 13, 10], color=color, line=dict(width=1, color="#fcfcfb")),
            hovertext=point_hover, hoverinfo="text",
        ))

    title_params = (
        f"min_w {cfg['min_bowl_width']} max_w {cfg['max_bowl_width']} "
        f"depth {cfg['min_bowl_depth_bps']}bps pos {cfg['bottom_position_limit']} "
        f"peak_dd {cfg['peak_drawdown_limit_bps']}bps peak_search {cfg['max_peak_search_width']}"
    )
    fig.update_layout(
        title=dict(
            text=f"{cfg['asset'].upper()} 1m — rising_from_bowl<br><sup>{title_params}</sup>",
            font=dict(size=17, color="#0b0b0b"),
        ),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color="#52514e"),
        paper_bgcolor="#f9f9f7", plot_bgcolor="#fcfcfb",
        hovermode="closest",
        # Vertical sidebar legend: with dozens-to-hundreds of bowls a top horizontal
        # legend wraps into many rows and buries the title/plot. A vertical legend
        # anchored beside the plot is height-capped by the figure and becomes
        # scrollable in the browser instead of growing the page, while every bowl
        # stays individually clickable.
        legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top",
                    font=dict(size=10), tracegroupgap=1, groupclick="togglegroup"),
        margin=dict(l=60, r=190, t=80, b=40),
        height=800,
        xaxis=dict(rangeslider=dict(visible=False)),
    )
    spike = dict(showspikes=True, spikemode="across", spikesnap="cursor",
                 spikecolor="#9a9990", spikethickness=1)
    fig.update_xaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", showline=True, zeroline=False, **spike)
    fig.update_yaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", showline=True, zeroline=False,
                      title_text="Price (USDT)", tickformat=",.0f", **spike)

    fig.write_html(out_path, config={"scrollZoom": True})


def run(cfg: dict) -> None:
    asset = cfg["asset"]
    date_from, date_to = cfg["date_from"], cfg["date_to"]
    detector_params = {k: cfg[k] for k in DETECTOR_KEYS}
    pad_minutes = max(cfg["max_bowl_width"], cfg["max_peak_search_width"])

    t0 = time.perf_counter()
    load_from = (datetime.strptime(date_from, DT_FMT).replace(tzinfo=timezone.utc)
                 - timedelta(minutes=pad_minutes)).strftime(DT_FMT)
    data = load_candles(asset, load_from, date_to)
    print(f"load candles: {data.shape[0]:,} rows  [{time.perf_counter() - t0:.3f}s]")

    t0 = time.perf_counter()
    vwap = sanitize_vwap(data)
    print(f"sanitize vwap  [{time.perf_counter() - t0:.3f}s]")

    ts = data[:, 0]
    start_idx = int(np.searchsorted(ts, to_ms(date_from), side="left"))
    end_idx = int(np.searchsorted(ts, to_ms(date_to), side="right"))
    n_anchors = end_idx - start_idx
    if n_anchors > 50_000:
        print(f"warning: {n_anchors:,} anchors — chart may be large/slow to open")

    t0 = time.perf_counter()
    warmup_len = min(len(vwap), pad_minutes + 5)
    rising_from_bowl_scan(vwap[:warmup_len], 0, None, **detector_params)
    print(f"jit warm-up  [{time.perf_counter() - t0:.3f}s]")

    t0 = time.perf_counter()
    detections = rising_from_bowl_scan(vwap, start_idx, end_idx, **detector_params)
    print(f"scan: {n_anchors:,} anchors -> {detections.shape[0]:,} detections  [{time.perf_counter() - t0:.3f}s]")

    t0 = time.perf_counter()
    bowls = dedupe_bowls(detections)
    print(f"dedupe: {bowls['rows'].shape[0]:,} distinct bowls  [{time.perf_counter() - t0:.3f}s]")

    print_stats(asset, date_from, date_to, n_anchors, detections, bowls)

    t0 = time.perf_counter()
    out_path = SCRIPT_DIR / f"rising_from_bowl_{asset}.html"
    build_chart(data, vwap, bowls, cfg, out_path)
    print(f"chart written: {out_path}  [{time.perf_counter() - t0:.3f}s]")


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        cfg = json.load(f)
    run(cfg)


if __name__ == "__main__":
    main()
