"""LBLA Normalized VP Chart."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _gather_peaks(hvn: dict) -> list:
    poc = hvn.get("poc")
    above = hvn.get("above") or []
    below = hvn.get("below") or []
    return (
        ([("POC", poc)] if poc is not None else [])
        + [("above", p) for p in above]
        + [("below", p) for p in below]
    )


def _make_show_fn() -> tuple:
    shown: set = set()

    def show(group: str) -> bool:
        if group not in shown:
            shown.add(group)
            return True
        return False

    return show


def _display_dfs(*dfs) -> None:
    try:
        from IPython.display import display
        for df in dfs:
            display(df)
    except ImportError:
        for df in dfs:
            print(df.to_string())


def draw_chart_vp(data: dict) -> go.Figure:
    """Render interactive VP chart; display peaks and metrics DataFrames."""
    lb_x = data["lb_x"]
    la_x = data["la_x"]
    lb_p = data["lb_p"]
    la_p = data["la_p"]
    bin_centers = data["bin_centers"]
    vp_hist = data["vp_hist"]
    vp_kde = data["vp_kde"]
    bin_width = data["bin_width"]
    hvn = data["hvn"]
    metrics = data["metrics"]
    current_price = data["current_price"]

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        column_widths=[0.25, 0.75],
    )

    _show = _make_show_fn()

    def _peak_height(price: float) -> float:
        idx = int(np.argmin(np.abs(bin_centers - price)))
        return float(vp_kde[idx])

    # VP panel – histogram
    fig.add_trace(go.Bar(
        x=vp_hist, y=bin_centers,
        orientation="h", width=bin_width,
        opacity=0.3, marker_color="steelblue",
        name="histogram", legendgroup="histogram",
        showlegend=_show("histogram"),
    ), row=1, col=1)

    # VP panel – KDE line
    fig.add_trace(go.Scatter(
        x=vp_kde, y=bin_centers, mode="lines",
        line=dict(color="royalblue", width=2),
        name="kde", legendgroup="kde",
        showlegend=_show("kde"),
    ), row=1, col=1)

    all_peaks = _gather_peaks(hvn)

    # Peak horizontal lines — each spans 0 → its own peak height
    for label, peak in all_peaks:
        price = peak["price"]
        h = _peak_height(price)
        is_poc = label == "POC"
        group = "POC" if is_poc else "peaks (lines)"
        color = "crimson" if is_poc else "darkorange"
        fig.add_trace(go.Scatter(
            x=[0, h], y=[price, price], mode="lines",
            line=dict(color=color, width=1.5, dash="dot"),
            name=group, legendgroup=group,
            showlegend=_show(group),
        ), row=1, col=1)

    # Width rectangles — all width_h1 first (lighter, behind), then all width_h05 on top
    for label, peak in all_peaks:
        price = peak["price"]
        h = _peak_height(price)
        w1 = float(peak["width_h1"]) * bin_width
        is_poc = label == "POC"
        color_h1 = "rgba(220,20,60,0.2)" if is_poc else "rgba(255,140,0,0.2)"
        if w1 > 0:
            y0, y1 = price - w1 / 2, price + w1 / 2
            fig.add_trace(go.Scatter(
                x=[0, h, h, 0, 0], y=[y0, y0, y1, y1, y0],
                fill="toself", mode="lines", line=dict(width=0),
                fillcolor=color_h1,
                name="width_h1", legendgroup="width_h1",
                showlegend=_show("width_h1"),
            ), row=1, col=1)

    for label, peak in all_peaks:
        price = peak["price"]
        h = _peak_height(price)
        w05 = float(peak["width_h05"]) * bin_width
        is_poc = label == "POC"
        color_h05 = "rgba(220,20,60,0.5)" if is_poc else "rgba(255,140,0,0.5)"
        if w05 > 0:
            y0, y1 = price - w05 / 2, price + w05 / 2
            fig.add_trace(go.Scatter(
                x=[0, h, h, 0, 0], y=[y0, y0, y1, y1, y0],
                fill="toself", mode="lines", line=dict(width=0),
                fillcolor=color_h05,
                name="width_h05", legendgroup="width_h05",
                showlegend=_show("width_h05"),
            ), row=1, col=1)

    # Main panel – look-back
    fig.add_trace(go.Scatter(
        x=lb_x, y=lb_p, mode="lines",
        line=dict(color="steelblue", width=1.5),
        name="look-back", legendgroup="look-back",
        showlegend=_show("look-back"),
    ), row=1, col=2)

    # Main panel – look-ahead
    fig.add_trace(go.Scatter(
        x=la_x, y=la_p, mode="lines",
        line=dict(color="seagreen", width=1.5),
        name="look-ahead", legendgroup="look-ahead",
        showlegend=_show("look-ahead"),
    ), row=1, col=2)

    # Vertical separator at x=1.0 (current time)
    fig.add_shape(
        type="line",
        x0=1.0, x1=1.0, y0=-1, y1=1,
        line=dict(color="gray", width=1, dash="dash"),
        row=1, col=2,
    )

    # Horizontal reference at y=0 (current price)
    fig.add_shape(
        type="line",
        x0=float(lb_x[0]), x1=float(la_x[-1]), y0=0, y1=0,
        line=dict(color="black", width=1, dash="dot"),
        row=1, col=2,
    )

    fig.update_yaxes(range=[-1, 1])
    fig.update_layout(
        title=(
            f"{data.get('asset', '').upper()} | {data.get('datetime', '')} "
            f"| price={current_price:.2f}"
        ),
        height=600,
        legend=dict(groupclick="togglegroup"),
    )

    # --- Tables ---
    peak_rows = []
    for label, peak in all_peaks:
        price = peak["price"]
        peak_rows.append({
            "label": label,
            "price": price,
            "height (KDE)": _peak_height(price),
            "prominence": peak["prominence"],
            "width_h1 (norm-price)": peak["width_h1"] * bin_width,
            "width_h05 (norm-price)": peak["width_h05"] * bin_width,
        })
    peaks_df = pd.DataFrame(peak_rows)
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["name", "value"])

    fig.show()
    _display_dfs(peaks_df, metrics_df)
    return fig


def draw_chart_vp_continued_width(data: dict) -> go.Figure:
    """VP chart variant: no width_h1, width_h05 continued into right panel, z-scored volume axis."""
    lb_x = data["lb_x"]
    la_x = data["la_x"]
    lb_p = data["lb_p"]
    la_p = data["la_p"]
    bin_centers = data["bin_centers"]
    vp_hist = data["vp_hist"]
    vp_kde = data["vp_kde"]
    bin_width = data["bin_width"]
    hvn = data["hvn"]
    metrics = data["metrics"]
    current_price = data["current_price"]

    v_median = float(metrics["v_median"])
    v_iqr = float(metrics["v_iqr"])

    def z_score(x):
        if v_iqr == 0.0:
            return 0.0 if np.isscalar(x) else np.zeros_like(x, dtype=np.float64)
        return (x - v_median) / v_iqr

    vp_hist_z = z_score(vp_hist)
    vp_kde_z = z_score(vp_kde)

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        column_widths=[0.25, 0.75],
    )

    _show = _make_show_fn()

    def _peak_height_z(price: float) -> float:
        idx = int(np.argmin(np.abs(bin_centers - price)))
        return float(vp_kde_z[idx])

    # VP panel – histogram (z-scored x)
    fig.add_trace(go.Bar(
        x=vp_hist_z, y=bin_centers,
        orientation="h", width=bin_width,
        opacity=0.3, marker_color="steelblue",
        name="histogram", legendgroup="histogram",
        showlegend=_show("histogram"),
    ), row=1, col=1)

    # VP panel – KDE line (z-scored x)
    fig.add_trace(go.Scatter(
        x=vp_kde_z, y=bin_centers, mode="lines",
        line=dict(color="royalblue", width=2),
        name="kde", legendgroup="kde",
        showlegend=_show("kde"),
    ), row=1, col=1)

    all_peaks = _gather_peaks(hvn)

    # Peak horizontal lines — each spans 0 → z-scored peak height
    for label, peak in all_peaks:
        price = peak["price"]
        h = _peak_height_z(price)
        is_poc = label == "POC"
        group = "POC" if is_poc else "peaks (lines)"
        color = "crimson" if is_poc else "darkorange"
        fig.add_trace(go.Scatter(
            x=[0, h], y=[price, price], mode="lines",
            line=dict(color=color, width=1.5, dash="dot"),
            name=group, legendgroup=group,
            showlegend=_show(group),
        ), row=1, col=1)

    # width_h05 rectangles: left panel + right panel (no width_h1)
    x_right_start = float(lb_x[0])
    x_right_end = float(la_x[-1])

    for label, peak in all_peaks:
        price = peak["price"]
        h = _peak_height_z(price)
        w05 = float(peak["width_h05"]) * bin_width
        if w05 == 0:
            continue
        is_poc = label == "POC"
        color_h05 = "rgba(220,20,60,0.5)" if is_poc else "rgba(255,140,0,0.5)"
        y0, y1 = price - w05 / 2, price + w05 / 2

        # Left panel rectangle
        fig.add_trace(go.Scatter(
            x=[0, h, h, 0, 0], y=[y0, y0, y1, y1, y0],
            fill="toself", mode="lines", line=dict(width=0),
            fillcolor=color_h05,
            name="width_h05", legendgroup="width_h05",
            showlegend=_show("width_h05"),
        ), row=1, col=1)

        # Right panel rectangle (same band, full x range)
        fig.add_trace(go.Scatter(
            x=[x_right_start, x_right_end, x_right_end, x_right_start, x_right_start],
            y=[y0, y0, y1, y1, y0],
            fill="toself", mode="lines", line=dict(width=0),
            fillcolor=color_h05,
            name="width_h05", legendgroup="width_h05",
            showlegend=False,
        ), row=1, col=2)

    # Main panel – look-back
    fig.add_trace(go.Scatter(
        x=lb_x, y=lb_p, mode="lines",
        line=dict(color="steelblue", width=1.5),
        name="look-back", legendgroup="look-back",
        showlegend=_show("look-back"),
    ), row=1, col=2)

    # Main panel – look-ahead
    fig.add_trace(go.Scatter(
        x=la_x, y=la_p, mode="lines",
        line=dict(color="seagreen", width=1.5),
        name="look-ahead", legendgroup="look-ahead",
        showlegend=_show("look-ahead"),
    ), row=1, col=2)

    # Vertical separator at x=1.0 (current time)
    fig.add_shape(
        type="line",
        x0=1.0, x1=1.0, y0=-1, y1=1,
        line=dict(color="gray", width=1, dash="dash"),
        row=1, col=2,
    )

    # Horizontal reference at y=0 (current price)
    fig.add_shape(
        type="line",
        x0=x_right_start, x1=x_right_end, y0=0, y1=0,
        line=dict(color="black", width=1, dash="dot"),
        row=1, col=2,
    )

    fig.update_yaxes(range=[-1, 1])
    fig.update_xaxes(title_text="robust z-score of volume", row=1, col=1)
    fig.update_layout(
        title=(
            f"{data.get('asset', '').upper()} | {data.get('datetime', '')} "
            f"| price={current_price:.2f}"
        ),
        height=600,
        legend=dict(groupclick="togglegroup"),
    )

    # --- Tables ---
    peak_rows = []
    for label, peak in all_peaks:
        price = peak["price"]
        raw_height = float(vp_kde[int(np.argmin(np.abs(bin_centers - price)))])
        raw_prom = peak["prominence"]
        peak_rows.append({
            "label": label,
            "price": price,
            "height": raw_height,
            "prominence": raw_prom,
            "width_h1": peak["width_h1"] * bin_width,
            "width_h05": peak["width_h05"] * bin_width,
            "height_z": float(z_score(raw_height)),
            "prominence_z": float(z_score(raw_prom)),
        })
    peaks_df = pd.DataFrame(peak_rows)
    metrics_df = pd.DataFrame(list(metrics.items()), columns=["name", "value"])

    fig.show()
    _display_dfs(peaks_df, metrics_df)
    return fig
