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
from packages.pattern_detection import (
    rising_from_bowl_scan, SCAN_COLUMNS,
    falling_from_dome_scan, DOME_SCAN_COLUMNS,
)
from packages.volume_profile import compute_kde, recursive_poc

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.json"
DT_FMT = "%Y-%m-%d %H:%M:%S"

# Symmetric pattern-param config keys -> each detector's native kwarg name
# (fan-out mapping, matches notebooks/tests/volume_profile/volume_profile_bowl_dome_colab.ipynb cell 3).
BOWL_PARAM_MAP = {
    "min_pattern_width": "min_bowl_width",
    "max_pattern_width": "max_bowl_width",
    "min_pattern_extent_bps": "min_bowl_depth_bps",
    "extremum_position_limit": "bottom_position_limit",
    "wall_retrace_limit_bps": "peak_drawdown_limit_bps",
    "max_wall_search_width": "max_peak_search_width",
}
DOME_PARAM_MAP = {
    "min_pattern_width": "min_dome_width",
    "max_pattern_width": "max_dome_width",
    "min_pattern_extent_bps": "min_dome_height_bps",
    "extremum_position_limit": "top_position_limit",
    "wall_retrace_limit_bps": "trough_rally_limit_bps",
    "max_wall_search_width": "max_trough_search_width",
}
VP_DEFAULTS = {
    "vp_lookback": 1440,
    "vp_bins": 200,
    "vp_bps_range": 100.0,
    "vp_kernel_type": "Triangular",
    "vp_bandwidth": 5,
    "vp_va_pct": 70.0,
    "vp_min_poc_volume_ratio": 0.1,
}
BOWL_COL = {name: idx for idx, name in enumerate(SCAN_COLUMNS)}
DOME_COL = {name: idx for idx, name in enumerate(DOME_SCAN_COLUMNS)}

BOWL_COLOR = "#2e6f95"
DOME_COLOR = "#c1666b"
VWAP_PLAIN_COLOR = "#9a8c98"
LOOK_AHEAD_COLOR = "#9a9990"

QUALITATIVE_PALETTE = [
    "#2e6f95", "#c1666b", "#4c9a6a", "#a76fb0", "#c98a3b",
    "#5b7fbd", "#8a8a3c", "#c15a9e", "#3f9e8f", "#b5533c",
]

# VAL/VAH visibility: hover-linked toggling was tried (Plotly.restyle from a
# plotly_hover listener) but proved unreliable in practice — a POC sits where
# price consolidated the most, so almost everywhere along the line a candle
# occupies the same price and wins Plotly's "closest" hover contest, making
# the POC trace itself nearly impossible to hover. Per spec fallback: VAL/VAH
# are always visible instead, toggled together with their POC via legendgroup.


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


def compute_vwap(data: np.ndarray, period: int):
    """vwap series per vwap_period: 1 = candle vwap column (no plain series
    needed, they'd be identical); N>1 = rolling_vwap (sum q / sum v) over the
    last N candles — the detection/volume-profile series — plus the plain
    candle vwap for chart-only display. Both sanitized. Returns
    (vwap_detection, vwap_plain_or_None)."""
    if period == 1:
        return sanitize_vwap(data[:, 8]), None
    quotes = np.ascontiguousarray(data[:, 6])
    volumes = np.ascontiguousarray(data[:, 5])
    vwap_n = sanitize_vwap(rolling_vwap(quotes, volumes, period))
    vwap_plain = sanitize_vwap(data[:, 8])
    return vwap_n, vwap_plain


def dedupe(detections: np.ndarray, col_map: dict, key: str) -> dict:
    """Group scan rows by their identity column (bottom_idx for bowls, top_idx
    for domes): first (earliest) detection per pattern, chronologically
    ordered, plus a re-detection count per pattern."""
    if detections.shape[0] == 0:
        return {"rows": detections, "counts": np.zeros(0, dtype=np.int64)}
    ident = detections[:, col_map[key]]
    _, first_pos, counts = np.unique(ident, return_index=True, return_counts=True)
    order = np.argsort(first_pos)
    return {"rows": detections[first_pos[order]], "counts": counts[order]}


def print_stats(kind: str, detections: np.ndarray, dedup: dict, col_map: dict, fields) -> None:
    """fields: list of (label, column_name) for the per-field mean/median summary."""
    m = detections.shape[0]
    d = dedup["rows"].shape[0]
    print(f"detections {m:,} | distinct {kind}s {d:,}")
    if d == 0:
        print(f"detections/{kind}: n/a (no {kind}s detected)")
        return
    counts = dedup["counts"]
    print(f"detections/{kind}  min {counts.min()}  mean {counts.mean():.2f}  max {counts.max()}")
    rows = dedup["rows"]
    for label, col in fields:
        v = rows[:, col_map[col]]
        print(f"{label:<11} mean {v.mean():.3f}  median {np.median(v):.3f}")


def compute_volume_profile(data: np.ndarray, vwap: np.ndarray, start_idx: int, end_idx: int, cfg: dict) -> dict:
    """Volume profile + recursive POC over the last vp_lookback candles of the
    displayed (non-look-ahead) range. Adds `va_bps` to each POC dict. Prints a
    terse summary, one line per POC (now incl. va bps), or a single
    "no POCs found" line, and the step's elapsed time."""
    t0 = time.perf_counter()
    vp_cfg = {k: cfg.get(k, default) for k, default in VP_DEFAULTS.items()}
    vp_lookback = int(vp_cfg["vp_lookback"])

    vp_lo = max(start_idx, end_idx - vp_lookback)
    vp_prices = np.ascontiguousarray(vwap[vp_lo:end_idx])
    vp_volumes = np.ascontiguousarray(data[vp_lo:end_idx, 5])

    vp = compute_kde(
        vp_prices, vp_volumes,
        int(vp_cfg["vp_bins"]), float(vp_cfg["vp_bps_range"]),
        vp_cfg["vp_kernel_type"], int(vp_cfg["vp_bandwidth"]),
    )
    pocs = recursive_poc(
        vp["kde"], vp["bin_centers"], vp["current_price"],
        float(vp_cfg["vp_va_pct"]), float(vp_cfg["vp_min_poc_volume_ratio"]),
    )
    for p in pocs:
        p["va_bps"] = (p["vah"] - p["val"]) / vp["current_price"] * 1e4

    print(f"volume profile: {vp_prices.shape[0]:,} candles | current {vp['current_price']:,.2f} | "
          f"range [{vp['range_min']:,.2f}, {vp['range_max']:,.2f}] | excluded {vp['n_excluded']:,}")
    if pocs:
        for p in pocs:
            print(f"POC {p['rank']}: price {p['poc_price']:,.2f} vol {p['poc_volume']:,.3f} "
                  f"VA [{p['val']:,.2f}, {p['vah']:,.2f}] va {p['va_bps']:.1f} bps")
    else:
        print("no POCs found")
    print(f"volume profile + recursive_poc  [{time.perf_counter() - t0:.3f}s]")

    return {"vp": vp, "pocs": pocs}


def build_figure(res: dict, cfg: dict):
    """Full chart (candles, vwap line(s), bowls, domes, POC + VAL/VAH lines,
    look-ahead boundary) + POC table. No file I/O."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    data = res["data"]
    vwap, vwap_plain, vwap_period = res["vwap"], res["vwap_plain"], res["vwap_period"]
    start_idx, end_idx, look_ahead = res["start_idx"], res["end_idx"], res["look_ahead"]
    bowls, domes, pocs = res["bowls"], res["domes"], res["pocs"]

    times = pd.to_datetime(data[:, 0], unit="ms")
    o, h, l, c = data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    vwap_name = "VWAP" if vwap_period == 1 else f"VWAP({vwap_period})"

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.78, 0.22], vertical_spacing=0.06,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
    )

    def add_trace(trace, **kw):
        idx = len(fig.data)
        fig.add_trace(trace, **kw)
        return idx

    # --- candles + vwap (entire loaded array: pad + range + look-ahead) ---
    add_trace(go.Candlestick(
        x=times, open=o, high=h, low=l, close=c, name="Candles",
        increasing=dict(line=dict(color="#4c9a6a"), fillcolor="#a9d1b6"),
        decreasing=dict(line=dict(color="#c1666b"), fillcolor="#e8b9ba"),
    ), row=1, col=1)
    add_trace(go.Scatter(
        x=times, y=vwap, mode="lines", name=vwap_name,
        line=dict(color="#2e2e2e", width=1.3),
        hovertemplate="%{y:,.2f}<extra>" + vwap_name + "</extra>",
    ), row=1, col=1)
    if vwap_plain is not None:
        add_trace(go.Scatter(
            x=times, y=vwap_plain, mode="lines", name="VWAP",
            line=dict(color=VWAP_PLAIN_COLOR, width=0.9),
            hovertemplate="%{y:,.2f}<extra>VWAP</extra>",
        ), row=1, col=1)

    # --- bowls: one color, one legend item "Bowls", one legendgroup ---
    rows, counts = bowls["rows"], bowls["counts"]
    for n in range(rows.shape[0]):
        row = rows[n]
        show_legend = n == 0
        i, t = int(row[BOWL_COL["left_rim_idx"]]), int(row[BOWL_COL["right_rim_idx"]])
        t_min = int(row[BOWL_COL["bottom_idx"]])
        k = int(row[BOWL_COL["bowl_width"]])
        depth_bps = row[BOWL_COL["bowl_depth_bps"]]
        pos_ratio = row[BOWL_COL["bottom_position_ratio"]]
        peak_idx = int(row[BOWL_COL["left_wall_peak_idx"]])
        peak_price = row[BOWL_COL["left_wall_peak_price"]]
        recovery = row[BOWL_COL["recovery_ratio"]]
        a, b, cc = row[BOWL_COL["fit_coef_a"]], row[BOWL_COL["fit_coef_b"]], row[BOWL_COL["fit_coef_c"]]
        r2 = row[BOWL_COL["r_squared"]]
        theo_idx = row[BOWL_COL["theoretical_bottom_idx"]]
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
        add_trace(go.Scatter(
            x=times[i:t + 1], y=y_fit, mode="lines",
            name="Bowls" if show_legend else f"Bowl {n + 1}",
            legendgroup="bowls", showlegend=show_legend,
            line=dict(color=BOWL_COLOR, width=1.6, dash="dot"),
            hovertemplate=info + "<extra></extra>",
        ), row=1, col=1)

        marker_x = [times[i], times[t_min], times[peak_idx], times[t]]
        marker_y = [vwap[i], vwap[t_min], peak_price, vwap[t]]
        marker_label = ["left rim", "bottom", "left-wall peak", "anchor"]
        point_hover = [
            f"Bowl {n + 1} — {label}<br>{tm:%Y-%m-%d %H:%M}  {yv:,.2f}<br>{info}"
            for label, tm, yv in zip(marker_label, marker_x, marker_y)
        ]
        add_trace(go.Scatter(
            x=marker_x, y=marker_y, mode="markers", showlegend=False, legendgroup="bowls",
            marker=dict(symbol=["circle-open", "circle", "star", "triangle-up"],
                        size=[9, 9, 13, 10], color=BOWL_COLOR, line=dict(width=1, color="#fcfcfb")),
            hovertext=point_hover, hoverinfo="text",
        ), row=1, col=1)

    # --- domes: one color, one legend item "Domes", one legendgroup ---
    rows, counts = domes["rows"], domes["counts"]
    for n in range(rows.shape[0]):
        row = rows[n]
        show_legend = n == 0
        i, t = int(row[DOME_COL["left_rim_idx"]]), int(row[DOME_COL["right_rim_idx"]])
        t_max = int(row[DOME_COL["top_idx"]])
        k = int(row[DOME_COL["dome_width"]])
        height_bps = row[DOME_COL["dome_height_bps"]]
        pos_ratio = row[DOME_COL["top_position_ratio"]]
        trough_idx = int(row[DOME_COL["left_wall_trough_idx"]])
        trough_price = row[DOME_COL["left_wall_trough_price"]]
        decline = row[DOME_COL["decline_ratio"]]
        a, b, cc = row[DOME_COL["fit_coef_a"]], row[DOME_COL["fit_coef_b"]], row[DOME_COL["fit_coef_c"]]
        r2 = row[DOME_COL["r_squared"]]
        theo_idx = row[DOME_COL["theoretical_top_idx"]]
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
        add_trace(go.Scatter(
            x=times[i:t + 1], y=y_fit, mode="lines",
            name="Domes" if show_legend else f"Dome {n + 1}",
            legendgroup="domes", showlegend=show_legend,
            line=dict(color=DOME_COLOR, width=1.6, dash="dot"),
            hovertemplate=info + "<extra></extra>",
        ), row=1, col=1)

        marker_x = [times[i], times[t_max], times[trough_idx], times[t]]
        marker_y = [vwap[i], vwap[t_max], trough_price, vwap[t]]
        marker_label = ["left rim", "top", "left-wall trough", "anchor"]
        point_hover = [
            f"Dome {n + 1} — {label}<br>{tm:%Y-%m-%d %H:%M}  {yv:,.2f}<br>{info}"
            for label, tm, yv in zip(marker_label, marker_x, marker_y)
        ]
        add_trace(go.Scatter(
            x=marker_x, y=marker_y, mode="markers", showlegend=False, legendgroup="domes",
            marker=dict(symbol=["circle-open", "circle", "star", "triangle-down"],
                        size=[9, 9, 13, 10], color=DOME_COLOR, line=dict(width=1, color="#fcfcfb")),
            hovertext=point_hover, hoverinfo="text",
        ), row=1, col=1)

    # --- POC lines (dashed) + VAL/VAH lines (thinner, hover-linked visibility) ---
    poc1_volume = pocs[0]["poc_volume"] if pocs else 0.0
    x_span = [times[start_idx], times[len(data) - 1]]  # reaches through look-ahead
    for p in pocs:
        color = QUALITATIVE_PALETTE[(p["rank"] - 1) % len(QUALITATIVE_PALETTE)]
        pct_of_poc1 = (p["poc_volume"] / poc1_volume * 100.0) if poc1_volume > 0 else 0.0
        hover = (
            f"POC {p['rank']}<br>price {p['poc_price']:,.2f}<br>"
            f"kde volume {p['poc_volume']:,.3f} ({pct_of_poc1:.1f}% of POC 1)<br>"
            f"VA [{p['val']:,.2f}, {p['vah']:,.2f}]<br>"
            f"va {p['va_bps']:.1f} bps"
        )
        add_trace(go.Scatter(
            x=x_span, y=[p["poc_price"], p["poc_price"]], mode="lines",
            name=f"POC {p['rank']}", legendgroup=f"poc{p['rank']}",
            line=dict(color=color, width=1.6, dash="dash"),
            hovertemplate=hover + "<extra></extra>",
        ), row=1, col=1)

        add_trace(go.Scatter(
            x=x_span, y=[p["val"], p["val"]], mode="lines",
            name=f"POC {p['rank']} VAL", legendgroup=f"poc{p['rank']}", showlegend=False,
            line=dict(color=color, width=0.8, dash="dot"),
            hovertemplate=f"POC {p['rank']} VAL {p['val']:,.2f}<extra></extra>",
        ), row=1, col=1)
        add_trace(go.Scatter(
            x=x_span, y=[p["vah"], p["vah"]], mode="lines",
            name=f"POC {p['rank']} VAH", legendgroup=f"poc{p['rank']}", showlegend=False,
            line=dict(color=color, width=0.8, dash="dot"),
            hovertemplate=f"POC {p['rank']} VAH {p['vah']:,.2f}<extra></extra>",
        ), row=1, col=1)

    # --- look-ahead boundary (chart-only; no detection/VP uses these candles) ---
    if look_ahead > 0 and end_idx < len(data):
        fig.add_vline(
            x=times[end_idx], line_width=1, line_dash="dash", line_color=LOOK_AHEAD_COLOR,
            annotation_text="look-ahead", annotation_position="top",
            annotation_font=dict(size=10, color=LOOK_AHEAD_COLOR),
            row=1, col=1,
        )

    # --- POC table ---
    if pocs:
        ranks = [p["rank"] for p in pocs]
        rank_colors = [QUALITATIVE_PALETTE[(r - 1) % len(QUALITATIVE_PALETTE)] for r in ranks]
        cell_values = [
            ranks,
            [f"{p['poc_price']:,.2f}" for p in pocs],
            [f"{p['poc_volume']:,.3f}" for p in pocs],
            [f"{(p['poc_volume'] / poc1_volume * 100.0 if poc1_volume > 0 else 0.0):.1f}%" for p in pocs],
            [f"{p['val']:,.2f}" for p in pocs],
            [f"{p['vah']:,.2f}" for p in pocs],
            [f"{p['va_bps']:.1f}" for p in pocs],
        ]
        cell_font_colors = [rank_colors, "#52514e", "#52514e", "#52514e", "#52514e", "#52514e", "#52514e"]
    else:
        cell_values = [[], [], [], [], [], [], []]
        cell_font_colors = "#52514e"
    fig.add_trace(go.Table(
        header=dict(values=["Rank", "Price", "KDE Volume", "% of POC 1", "VAL", "VAH", "VA bps"],
                    fill_color="#eceae2", font=dict(color="#52514e", size=12), align="left"),
        cells=dict(values=cell_values, fill_color="#fcfcfb",
                   font=dict(color=cell_font_colors, size=12), align="left"),
    ), row=2, col=1)

    title_params = (
        f"min_w {cfg['min_pattern_width']} max_w {cfg['max_pattern_width']} "
        f"extent {cfg['min_pattern_extent_bps']}bps pos {cfg['extremum_position_limit']} "
        f"retrace {cfg['wall_retrace_limit_bps']}bps wall_search {cfg['max_wall_search_width']} "
        f"vwap_p {vwap_period} look_ahead {look_ahead}m"
    )
    fig.update_layout(
        title=dict(
            text=f"{cfg['asset'].upper()} 1m — bowl_and_dome<br><sup>{title_params}</sup>",
            font=dict(size=17, color="#0b0b0b"),
        ),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color="#52514e"),
        paper_bgcolor="#f9f9f7", plot_bgcolor="#fcfcfb",
        hovermode="closest",
        # Vertical sidebar legend: with dozens-to-hundreds of bowls/domes a top
        # horizontal legend wraps into many rows and buries the title/plot. A
        # vertical legend anchored beside the plot is height-capped by the figure
        # and becomes scrollable in the browser instead of growing the page.
        # groupclick="togglegroup" makes the single "Bowls"/"Domes" legend item
        # toggle every trace in that group at once.
        legend=dict(orientation="v", x=1.01, xanchor="left", y=1, yanchor="top",
                    font=dict(size=10), tracegroupgap=1, groupclick="togglegroup"),
        margin=dict(l=60, r=190, t=80, b=40),
        height=1000,
    )
    spike = dict(showspikes=True, spikemode="across", spikesnap="cursor",
                 spikecolor="#9a9990", spikethickness=1)
    fig.update_xaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", showline=True, zeroline=False,
                      rangeslider=dict(visible=False), row=1, col=1, **spike)
    fig.update_yaxes(gridcolor="#e1e0d9", linecolor="#c3c2b7", showline=True, zeroline=False,
                      title_text="Price (USDT)", tickformat=",.0f", row=1, col=1, **spike)

    return fig


def analyze(cfg: dict) -> dict:
    """Load candles (padded before date_from for detector context, extended
    by look_ahead minutes after date_to for chart-only display), scan for
    bowls and domes, dedupe, compute the volume profile + recursive POC.
    Returns everything build_figure needs; no plotting or I/O beyond
    load_candles' own local cache.

    Detection and the volume profile only ever see anchors/windows inside
    [start_idx, end_idx) — the look-ahead candles (if any) are display-only
    and never influence scan results or VP/POC values."""
    asset = cfg["asset"]
    date_from, date_to = cfg["date_from"], cfg["date_to"]

    bowl_kwargs = {native: cfg[sym] for sym, native in BOWL_PARAM_MAP.items()}
    dome_kwargs = {native: cfg[sym] for sym, native in DOME_PARAM_MAP.items()}

    vwap_period = int(cfg.get("vwap_period", 1))
    if vwap_period < 1:
        raise ValueError("vwap_period must be >= 1")
    look_ahead = int(cfg.get("look_ahead", 240))
    if look_ahead < 0:
        raise ValueError("look_ahead must be >= 0")

    pad_minutes = max(cfg["max_pattern_width"], cfg["max_wall_search_width"], vwap_period)

    t0 = time.perf_counter()
    load_from = (datetime.strptime(date_from, DT_FMT).replace(tzinfo=timezone.utc)
                 - timedelta(minutes=pad_minutes)).strftime(DT_FMT)
    load_to = (datetime.strptime(date_to, DT_FMT).replace(tzinfo=timezone.utc)
               + timedelta(minutes=look_ahead)).strftime(DT_FMT)
    data = load_candles(asset, load_from, load_to)
    print(f"load candles: {data.shape[0]:,} rows  [{time.perf_counter() - t0:.3f}s]")

    if vwap_period > 1:  # warm rolling_vwap jit off the clock
        ones = np.ones(vwap_period, dtype=np.float64)
        rolling_vwap(ones, ones, vwap_period)
    t0 = time.perf_counter()
    vwap, vwap_plain = compute_vwap(data, vwap_period)
    print(f"vwap period {vwap_period}  [{time.perf_counter() - t0:.3f}s]")

    ts = data[:, 0]
    start_idx = int(np.searchsorted(ts, to_ms(date_from), side="left"))
    end_idx = int(np.searchsorted(ts, to_ms(date_to), side="right"))
    n_anchors = end_idx - start_idx
    if n_anchors > 50_000:
        print(f"warning: {n_anchors:,} anchors — chart may be large/slow to open")
    n_look_ahead = data.shape[0] - end_idx
    print(f"look_ahead: {look_ahead} min requested | {n_look_ahead:,} candles available past date_to")

    t0 = time.perf_counter()
    warmup_len = min(len(vwap), pad_minutes + 5)
    rising_from_bowl_scan(vwap[:warmup_len], 0, None, **bowl_kwargs)
    falling_from_dome_scan(vwap[:warmup_len], 0, None, **dome_kwargs)
    print(f"jit warm-up  [{time.perf_counter() - t0:.3f}s]")

    t0 = time.perf_counter()
    bowl_detections = rising_from_bowl_scan(vwap, start_idx, end_idx, **bowl_kwargs)
    print(f"bowl scan: {n_anchors:,} anchors -> {bowl_detections.shape[0]:,} detections  "
          f"[{time.perf_counter() - t0:.3f}s]")

    t0 = time.perf_counter()
    dome_detections = falling_from_dome_scan(vwap, start_idx, end_idx, **dome_kwargs)
    print(f"dome scan: {n_anchors:,} anchors -> {dome_detections.shape[0]:,} detections  "
          f"[{time.perf_counter() - t0:.3f}s]")

    t0 = time.perf_counter()
    bowls = dedupe(bowl_detections, BOWL_COL, "bottom_idx")
    domes = dedupe(dome_detections, DOME_COL, "top_idx")
    print(f"dedupe: {bowls['rows'].shape[0]:,} distinct bowls, "
          f"{domes['rows'].shape[0]:,} distinct domes  [{time.perf_counter() - t0:.3f}s]")

    print(f"{asset} {date_from} -> {date_to} | candles {n_anchors:,}")
    print("-- bowls --")
    print_stats("bowl", bowl_detections, bowls, BOWL_COL,
                [("width(m)", "bowl_width"), ("depth(bps)", "bowl_depth_bps"),
                 ("recovery", "recovery_ratio"), ("r2", "r_squared")])
    print("-- domes --")
    print_stats("dome", dome_detections, domes, DOME_COL,
                [("width(m)", "dome_width"), ("height(bps)", "dome_height_bps"),
                 ("decline", "decline_ratio"), ("r2", "r_squared")])

    vp_res = compute_volume_profile(data, vwap, start_idx, end_idx, cfg)

    return {
        "data": data, "vwap": vwap, "vwap_plain": vwap_plain, "vwap_period": vwap_period,
        "start_idx": start_idx, "end_idx": end_idx, "look_ahead": look_ahead,
        "bowls": bowls, "domes": domes, "vp": vp_res["vp"], "pocs": vp_res["pocs"],
    }


def run(cfg: dict) -> None:
    res = analyze(cfg)

    t0 = time.perf_counter()
    fig = build_figure(res, cfg)
    out_path = SCRIPT_DIR / f"bowl_and_dome_{cfg['asset']}.html"
    fig.write_html(out_path, config={"scrollZoom": True})
    print(f"chart written: {out_path}  [{time.perf_counter() - t0:.3f}s]")


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        cfg = json.load(f)
    run(cfg)


if __name__ == "__main__":
    main()
