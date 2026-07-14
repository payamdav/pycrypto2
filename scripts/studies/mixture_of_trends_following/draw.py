"""3-pane interactive chart: vwap+trigger bullets / slopes / imbalances. Saves motf_chart_{asset}.html.
Run: python3 draw.py [tag]"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from packages.candle_loader import load_candles
from scripts.studies.mixture_of_trends_following.build_cache import sanitize_vwap
from scripts.studies.mixture_of_trends_following.common import (
    COL_TRIGGER, COL_TS, IMB_COLS, SLOPE_COLS,
    assets_of, cli_tag, data_dir, load_params, parse_ts_ms, timed, window_label,
)

# validated categorical palette (agents/general -> dataviz skill), fixed order, reused across panes
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
TRIGGER_COLOR = "#eb6834"


def _color_map(*window_lists) -> dict:
    """Assign each distinct window value the next fixed palette slot, first-seen order; reused everywhere."""
    colors, i = {}, 0
    for windows in window_lists:
        for w in windows:
            if w not in colors:
                colors[w] = PALETTE[i % len(PALETTE)]
                i += 1
    return colors


def load_draw_slice(asset: str, params: dict):
    candles = load_candles(asset, params["draw_from"], params["draw_to"])
    vwap = sanitize_vwap(candles)

    cache = np.load(data_dir(params["tag"]) / f"motf_cache_{asset}.npy")
    from_ms = parse_ts_ms(params["draw_from"])
    to_ms = parse_ts_ms(params["draw_to"])
    lo = np.searchsorted(cache[:, COL_TS], from_ms, side="left")
    hi = np.searchsorted(cache[:, COL_TS], to_ms, side="right")
    cache_slice = cache[lo:hi]
    assert cache_slice.shape[0] == candles.shape[0], "cache/candle slice length mismatch"

    x = np.array(candles[:, COL_TS], dtype="datetime64[ms]")
    return x, vwap, cache_slice


def build_figure(asset: str, params: dict, x, vwap, cache_slice) -> go.Figure:
    colors = _color_map(params["slope_windows"], params["imbalance_windows"])

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.44, 0.28, 0.28], vertical_spacing=0.05,
        subplot_titles=("vwap + trigger", "relative slope", "volume imbalance"),
    )

    fig.add_trace(go.Scatter(
        x=x, y=vwap, mode="lines", name="vwap",
        line=dict(color=INK, width=1.6),
        hovertemplate="%{y:,.2f}<extra>vwap</extra>",
    ), row=1, col=1)

    trig = cache_slice[:, COL_TRIGGER] == 1.0
    fig.add_trace(go.Scatter(
        x=x[trig], y=vwap[trig], mode="markers", name="trigger",
        marker=dict(symbol="circle", size=8, color=TRIGGER_COLOR,
                    line=dict(width=1, color=SURFACE)),
        hovertemplate="%{y:,.2f}<extra>trigger</extra>",
    ), row=1, col=1)

    for col, w in zip(SLOPE_COLS, params["slope_windows"]):
        fig.add_trace(go.Scatter(
            x=x, y=cache_slice[:, col], mode="lines", name=f"slope {window_label(w)}",
            line=dict(color=colors[w], width=2),
            hovertemplate="%{y:.6f}<extra>slope " + window_label(w) + "</extra>",
        ), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"), row=2, col=1)

    for col, w in zip(IMB_COLS, params["imbalance_windows"]):
        fig.add_trace(go.Scatter(
            x=x, y=cache_slice[:, col], mode="lines", name=f"imbalance {window_label(w)}",
            line=dict(color=colors[w], width=2), showlegend=False,
            hovertemplate="%{y:.3f}<extra>imbalance " + window_label(w) + "</extra>",
        ), row=3, col=1)
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"), row=3, col=1)

    fig.update_layout(
        title=dict(text=f"{asset.upper()} mixture_of_trends_following — {params['draw_from']} .. {params['draw_to']} UTC",
                    font=dict(size=16, color=INK)),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color="#52514e"),
        paper_bgcolor=PAGE, plot_bgcolor=SURFACE,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=60, r=30, t=90, b=40),
        height=780,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS_LINE, showline=True, zeroline=False,
                      showspikes=True, spikemode="across", spikesnap="cursor",
                      spikedash="dot", spikecolor=MUTED, spikethickness=1)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS_LINE, showline=True, zeroline=False)
    fig.update_yaxes(title_text="vwap", row=1, col=1, tickformat=",.0f")
    fig.update_yaxes(title_text="slope", row=2, col=1)
    fig.update_yaxes(title_text="imbalance", row=3, col=1)
    return fig


def draw_asset(asset: str, params: dict) -> Path:
    x, vwap, cache_slice = load_draw_slice(asset, params)
    fig = build_figure(asset, params, x, vwap, cache_slice)
    out = data_dir(params["tag"]) / f"motf_chart_{asset}.html"
    fig.write_html(out)
    return out


def draw(params: dict) -> list:
    outs = []
    for asset in assets_of(params):
        with timed(f"[{asset}] draw"):
            out = draw_asset(asset, params)
        print(f"[{asset}] chart saved: {out}")
        outs.append(out)
    return outs


if __name__ == "__main__":
    tag = cli_tag(sys.argv)
    draw(load_params(tag))
