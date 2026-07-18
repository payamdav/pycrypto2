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
from packages.indicators.rolling_vwap import rolling_vwap
from packages.pattern_detection import falling_from_dome_scan, DOME_SCAN_COLUMNS

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.json"
DT_FMT = "%Y-%m-%d %H:%M:%S"

DETECTOR_KEYS = (
    "min_dome_width", "max_dome_width", "min_dome_height_bps",
    "top_position_limit", "trough_rally_limit_bps", "max_trough_search_width",
)
COL = {name: idx for idx, name in enumerate(DOME_SCAN_COLUMNS)}

QUALITATIVE_PALETTE = [
    "#2e6f95", "#c1666b", "#4c9a6a", "#a76fb0", "#c98a3b",
    "#5b7fbd", "#8a8a3c", "#c15a9e", "#3f9e8f", "#b5533c",
]


def to_ms(dt_str: str) -> int:
    return int(datetime.strptime(dt_str, DT_FMT).replace(tzinfo=timezone.utc).timestamp() * 1000)


def sanitize_vwap(vwap: np.ndarray) -> np.ndarray:
    """Return a copy where invalid values (nan/inf/<= 0) are replaced by the
    previous valid value; leading invalid values get the first valid value.
    Prints how many values were bad before correcting them."""
    vwap = np.asarray(vwap, dtype=np.float64).copy()
    bad = ~np.isfinite(vwap) | (vwap <= 0.0)
    n_bad = int(bad.sum())
    print(f"vwap: {n_bad:,} bad values (nan/inf/<=0) of {vwap.size:,}")
    if n_bad == 0:
        return vwap
    if n_bad == vwap.size:
        raise ValueError("all vwap values are invalid")
    vwap[bad] = np.nan
    return pd.Series(vwap).ffill().bfill().to_numpy()


def compute_vwap(data: np.ndarray, period: int) -> np.ndarray:
    """vwap series per vwap_period: 1 = candle vwap column; N>1 = rolling_vwap
    (sum q / sum v) over the last N candles. Sanitized either way."""
    if period == 1:
        return sanitize_vwap(data[:, 8])
    quotes = np.ascontiguousarray(data[:, 6])
    volumes = np.ascontiguousarray(data[:, 5])
    return sanitize_vwap(rolling_vwap(quotes, volumes, period))


def dedupe_domes(detections: np.ndarray) -> dict:
    """Group scan rows by top_idx (dome identity): first (earliest) detection
    per dome, chronologically ordered, plus a re-detection count per dome."""
    if detections.shape[0] == 0:
        return {"rows": detections, "counts": np.zeros(0, dtype=np.int64)}
    top_idx = detections[:, COL["top_idx"]]
    _, first_pos, counts = np.unique(top_idx, return_index=True, return_counts=True)
    order = np.argsort(first_pos)
    return {"rows": detections[first_pos[order]], "counts": counts[order]}


def print_stats(asset, date_from, date_to, n_anchors, detections, domes) -> None:
    m = detections.shape[0]
    d = domes["rows"].shape[0]
    print(f"{asset} {date_from} -> {date_to} | candles {n_anchors:,}")
    print(f"detections {m:,} | distinct domes {d:,}")
    if d == 0:
        print("detections/dome: n/a (no domes detected)")
        return
    counts = domes["counts"]
    print(f"detections/dome  min {counts.min()}  mean {counts.mean():.2f}  max {counts.max()}")
    rows = domes["rows"]
    for label, col in (("width(m)", "dome_width"), ("height(bps)", "dome_height_bps"),
                        ("decline", "decline_ratio"), ("r2", "r_squared")):
        v = rows[:, COL[col]]
        print(f"{label:<11} mean {v.mean():.3f}  median {np.median(v):.3f}")


def build_chart(data: np.ndarray, vwap: np.ndarray, domes: dict, cfg: dict, out_path: Path) -> None:
    import plotly.graph_objects as go

    times = pd.to_datetime(data[:, 0], unit="ms")
    o, h, l, c = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    vwap_period = int(cfg.get("vwap_period", 1))
    vwap_name = "VWAP" if vwap_period == 1 else f"VWAP({vwap_period})"

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=times, open=o, high=h, low=l, close=c, name="Candles",
        increasing=dict(line=dict(color="#4c9a6a"), fillcolor="#a9d1b6"),
        decreasing=dict(line=dict(color="#c1666b"), fillcolor="#e8b9ba"),
    ))
    fig.add_trace(go.Scatter(
        x=times, y=vwap, mode="lines", name=vwap_name,
        line=dict(color="#2e2e2e", width=1.3),
        hovertemplate="%{y:,.2f}<extra>" + vwap_name + "</extra>",
    ))

    rows, counts = domes["rows"], domes["counts"]
    for n in range(rows.shape[0]):
        row = rows[n]
        color = QUALITATIVE_PALETTE[n % len(QUALITATIVE_PALETTE)]
        group = f"dome{n + 1}"
        i, t = int(row[COL["left_rim_idx"]]), int(row[COL["right_rim_idx"]])
        t_max = int(row[COL["top_idx"]])
        k = int(row[COL["dome_width"]])
        height_bps = row[COL["dome_height_bps"]]
        pos_ratio = row[COL["top_position_ratio"]]
        trough_idx = int(row[COL["left_wall_trough_idx"]])
        trough_price = row[COL["left_wall_trough_price"]]
        decline = row[COL["decline_ratio"]]
        a, b, cc = row[COL["fit_coef_a"]], row[COL["fit_coef_b"]], row[COL["fit_coef_c"]]
        r2 = row[COL["r_squared"]]
        theo_idx = row[COL["theoretical_top_idx"]]
        det_count = int(counts[n])

        theo_time = times[i] + pd.to_timedelta(theo_idx - i, unit="m")
        info = (
            f"Dome {n + 1}  (×{det_count} detections)<br>"
            f"width {k} m | height {height_bps:.1f} bps | position {pos_ratio:.2f}<br>"
            f"decline {decline:.2f} | fit a {a:.3g} | R² {r2:.3f}<br>"
            f"theoretical top: {theo_time:%Y-%m-%d %H:%M}"
        )

        xs = np.arange(0, k + 1, dtype=np.float64)
        y_fit = a * xs ** 2 + b * xs + cc
        fig.add_trace(go.Scatter(
            x=times[i:t + 1], y=y_fit, mode="lines",
            name=f"Dome {n + 1} ×{det_count}", legendgroup=group,
            line=dict(color=color, width=1.6, dash="dot"),
            hovertemplate=info + "<extra></extra>",
        ))

        marker_x = [times[i], times[t_max], times[trough_idx], times[t]]
        marker_y = [vwap[i], vwap[t_max], trough_price, vwap[t]]
        marker_label = ["left rim", "top", "left-wall trough", "anchor"]
        point_hover = [
            f"Dome {n + 1} — {label}<br>{tm:%Y-%m-%d %H:%M}  {yv:,.2f}<br>{info}"
            for label, tm, yv in zip(marker_label, marker_x, marker_y)
        ]
        fig.add_trace(go.Scatter(
            x=marker_x, y=marker_y, mode="markers", showlegend=False, legendgroup=group,
            marker=dict(symbol=["circle-open", "circle", "star", "triangle-down"],
                        size=[9, 9, 13, 10], color=color, line=dict(width=1, color="#fcfcfb")),
            hovertext=point_hover, hoverinfo="text",
        ))

    title_params = (
        f"min_w {cfg['min_dome_width']} max_w {cfg['max_dome_width']} "
        f"height {cfg['min_dome_height_bps']}bps pos {cfg['top_position_limit']} "
        f"trough_rally {cfg['trough_rally_limit_bps']}bps trough_search {cfg['max_trough_search_width']} "
        f"vwap_p {vwap_period}"
    )
    fig.update_layout(
        title=dict(
            text=f"{cfg['asset'].upper()} 1m — falling_from_dome<br><sup>{title_params}</sup>",
            font=dict(size=17, color="#0b0b0b"),
        ),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color="#52514e"),
        paper_bgcolor="#f9f9f7", plot_bgcolor="#fcfcfb",
        hovermode="closest",
        # Vertical sidebar legend: with dozens-to-hundreds of domes a top horizontal
        # legend wraps into many rows and buries the title/plot. A vertical legend
        # anchored beside the plot is height-capped by the figure and becomes
        # scrollable in the browser instead of growing the page, while every dome
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
    vwap_period = int(cfg.get("vwap_period", 1))
    if vwap_period < 1:
        raise ValueError("vwap_period must be >= 1")
    pad_minutes = max(cfg["max_dome_width"], cfg["max_trough_search_width"], vwap_period)

    t0 = time.perf_counter()
    load_from = (datetime.strptime(date_from, DT_FMT).replace(tzinfo=timezone.utc)
                 - timedelta(minutes=pad_minutes)).strftime(DT_FMT)
    data = load_candles(asset, load_from, date_to)
    print(f"load candles: {data.shape[0]:,} rows  [{time.perf_counter() - t0:.3f}s]")

    if vwap_period > 1:  # warm rolling_vwap jit off the clock
        ones = np.ones(vwap_period, dtype=np.float64)
        rolling_vwap(ones, ones, vwap_period)
    t0 = time.perf_counter()
    vwap = compute_vwap(data, vwap_period)
    print(f"vwap period {vwap_period}  [{time.perf_counter() - t0:.3f}s]")

    ts = data[:, 0]
    start_idx = int(np.searchsorted(ts, to_ms(date_from), side="left"))
    end_idx = int(np.searchsorted(ts, to_ms(date_to), side="right"))
    n_anchors = end_idx - start_idx
    if n_anchors > 50_000:
        print(f"warning: {n_anchors:,} anchors — chart may be large/slow to open")

    t0 = time.perf_counter()
    warmup_len = min(len(vwap), pad_minutes + 5)
    falling_from_dome_scan(vwap[:warmup_len], 0, None, **detector_params)
    print(f"jit warm-up  [{time.perf_counter() - t0:.3f}s]")

    t0 = time.perf_counter()
    detections = falling_from_dome_scan(vwap, start_idx, end_idx, **detector_params)
    print(f"scan: {n_anchors:,} anchors -> {detections.shape[0]:,} detections  [{time.perf_counter() - t0:.3f}s]")

    t0 = time.perf_counter()
    domes = dedupe_domes(detections)
    print(f"dedupe: {domes['rows'].shape[0]:,} distinct domes  [{time.perf_counter() - t0:.3f}s]")

    print_stats(asset, date_from, date_to, n_anchors, detections, domes)

    t0 = time.perf_counter()
    out_path = SCRIPT_DIR / f"falling_from_dome_{asset}.html"
    build_chart(data, vwap, domes, cfg, out_path)
    print(f"chart written: {out_path}  [{time.perf_counter() - t0:.3f}s]")


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        cfg = json.load(f)
    run(cfg)


if __name__ == "__main__":
    main()
