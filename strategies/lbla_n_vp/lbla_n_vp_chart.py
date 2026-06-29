"""LBLA Normalized VP Chart."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from packages.kalman_filter import (
    kalman_1d_batch,
    kalman_2d_batch,
    kalman_3d_batch,
)


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


def _add_continued_width_content(
    fig: go.Figure, data: dict, _show, price_row: int, price_col: int
) -> tuple:
    """Add the continued-width VP content (left panel + price panel) to `fig`.

    Draws the z-scored histogram/KDE and peak lines on the left panel (1, 1), the
    look-back/look-ahead paths, continued width_h05 bands, x=1.0 separator and y=0
    line on the price panel (price_row, price_col). Returns (peaks_df, metrics_df).
    """
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

    v_median = float(metrics["v_median"])
    v_iqr = float(metrics["v_iqr"])

    def z_score(x):
        if v_iqr == 0.0:
            return 0.0 if np.isscalar(x) else np.zeros_like(x, dtype=np.float64)
        return (x - v_median) / v_iqr

    vp_hist_z = z_score(vp_hist)
    vp_kde_z = z_score(vp_kde)

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

    # width_h05 rectangles: left panel + price panel (no width_h1)
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

        # Price panel rectangle (same band, full x range)
        fig.add_trace(go.Scatter(
            x=[x_right_start, x_right_end, x_right_end, x_right_start, x_right_start],
            y=[y0, y0, y1, y1, y0],
            fill="toself", mode="lines", line=dict(width=0),
            fillcolor=color_h05,
            name="width_h05", legendgroup="width_h05",
            showlegend=False,
        ), row=price_row, col=price_col)

    # Price panel – look-back
    fig.add_trace(go.Scatter(
        x=lb_x, y=lb_p, mode="lines",
        line=dict(color="steelblue", width=1.5),
        name="look-back", legendgroup="look-back",
        showlegend=_show("look-back"),
    ), row=price_row, col=price_col)

    # Price panel – look-ahead
    fig.add_trace(go.Scatter(
        x=la_x, y=la_p, mode="lines",
        line=dict(color="seagreen", width=1.5),
        name="look-ahead", legendgroup="look-ahead",
        showlegend=_show("look-ahead"),
    ), row=price_row, col=price_col)

    # Vertical separator at x=1.0 (current time)
    fig.add_shape(
        type="line",
        x0=1.0, x1=1.0, y0=-1, y1=1,
        line=dict(color="gray", width=1, dash="dash"),
        row=price_row, col=price_col,
    )

    # Horizontal reference at y=0 (current price)
    fig.add_shape(
        type="line",
        x0=x_right_start, x1=x_right_end, y0=0, y1=0,
        line=dict(color="black", width=1, dash="dot"),
        row=price_row, col=price_col,
    )

    fig.update_xaxes(title_text="robust z-score of volume", row=1, col=1)

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
    return peaks_df, metrics_df


def _chart_title(data: dict) -> str:
    return (
        f"{data.get('asset', '').upper()} | {data.get('datetime', '')} "
        f"| price={float(data['current_price']):.2f}"
    )


def draw_chart_vp_continued_width(data: dict) -> go.Figure:
    """VP chart variant: no width_h1, width_h05 continued into right panel, z-scored volume axis."""
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        column_widths=[0.25, 0.75],
    )

    _show = _make_show_fn()
    peaks_df, metrics_df = _add_continued_width_content(fig, data, _show, 1, 2)

    fig.update_yaxes(range=[-1, 1])
    fig.update_layout(
        title=_chart_title(data),
        height=600,
        legend=dict(groupclick="togglegroup"),
    )

    fig.show()
    _display_dfs(peaks_df, metrics_df)
    return fig


# ---------------------------------------------------------------------------
# Kalman-smoothed variants
# ---------------------------------------------------------------------------

def _kalman_measurements(data: dict) -> np.ndarray:
    """Contiguous float64 look-back curve (unclipped) for the Kalman batch filters."""
    return np.ascontiguousarray(data["lb_pnc"], dtype=np.float64)


def _coerce_process_noise(process_noise, dim: int, default_scalar: float = 0.03):
    """Build an (dim, dim) Q matrix from None / scalar / matrix input."""
    if process_noise is None:
        return np.eye(dim, dtype=np.float64) * default_scalar
    if np.isscalar(process_noise):
        return np.eye(dim, dtype=np.float64) * float(process_noise)
    return np.asarray(process_noise, dtype=np.float64)


def _add_kalman_overlay(fig: go.Figure, lb_x, value, _show, price_row: int, price_col: int):
    """Clip the smoothed value to [-1, 1] and draw it on the price panel."""
    smoothed = np.clip(value, -1.0, 1.0)
    fig.add_trace(go.Scatter(
        x=lb_x, y=smoothed, mode="lines",
        line=dict(color="purple", width=2),
        name="kalman", legendgroup="kalman",
        showlegend=_show("kalman"),
    ), row=price_row, col=price_col)


def draw_chart_vp_continued_width_kalman_1d(
    data: dict, measurement_variance: float = 1.0, process_noise=None
) -> go.Figure:
    """continued_width chart + clipped 1D-Kalman smoothing of lb_pnc on the price panel."""
    if process_noise is None:
        process_noise = 0.03

    m = _kalman_measurements(data)
    estimates, _ = kalman_1d_batch(
        m, float(m[0]), 1.0, float(process_noise), float(measurement_variance)
    )

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        column_widths=[0.25, 0.75],
    )
    _show = _make_show_fn()
    peaks_df, metrics_df = _add_continued_width_content(fig, data, _show, 1, 2)
    _add_kalman_overlay(fig, data["lb_x"], estimates, _show, 1, 2)

    fig.update_yaxes(range=[-1, 1])
    fig.update_layout(
        title=_chart_title(data),
        height=600,
        legend=dict(groupclick="togglegroup"),
    )

    fig.show()
    _display_dfs(peaks_df, metrics_df)
    return fig


def draw_chart_vp_continued_width_kalman_2d(
    data: dict, measurement_variance: float = 1.0, process_noise=None
) -> go.Figure:
    """continued_width chart + 2D-Kalman smoothing, with a speed subchart sharing lb_x."""
    Q = _coerce_process_noise(process_noise, 2)

    m = _kalman_measurements(data)
    x0 = np.array([[m[0]], [0.0]], dtype=np.float64)
    P0 = np.eye(2, dtype=np.float64)
    states, _ = kalman_2d_batch(m, x0, P0, Q, float(measurement_variance), 1.0)
    value = states[:, 0, 0]
    speed = states[:, 1, 0]

    fig = make_subplots(
        rows=2, cols=2,
        shared_yaxes=True,
        shared_xaxes=True,
        column_widths=[0.25, 0.75],
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
        specs=[[{}, {}], [None, {}]],
    )
    _show = _make_show_fn()
    peaks_df, metrics_df = _add_continued_width_content(fig, data, _show, 1, 2)
    _add_kalman_overlay(fig, data["lb_x"], value, _show, 1, 2)

    # Speed subchart (shares x with the price panel)
    fig.add_trace(go.Scatter(
        x=data["lb_x"], y=speed, mode="lines",
        line=dict(color="teal", width=1.5),
        name="speed", legendgroup="speed",
        showlegend=_show("speed"),
    ), row=2, col=2)
    fig.add_vline(x=1.0, line=dict(color="gray", width=1, dash="dash"), row=2, col=2)

    fig.update_yaxes(range=[-1, 1], row=1, col=1)
    fig.update_yaxes(range=[-1, 1], row=1, col=2)
    fig.update_yaxes(title_text="speed", row=2, col=2)
    fig.update_layout(
        title=_chart_title(data),
        height=750,
        legend=dict(groupclick="togglegroup"),
    )

    fig.show()
    _display_dfs(peaks_df, metrics_df)
    return fig


def draw_chart_vp_continued_width_kalman_3d(
    data: dict, measurement_variance: float = 1.0, process_noise=None
) -> go.Figure:
    """continued_width chart + 3D-Kalman smoothing, with speed & acceleration subcharts sharing lb_x."""
    Q = _coerce_process_noise(process_noise, 3)

    m = _kalman_measurements(data)
    x0 = np.array([[m[0]], [0.0], [0.0]], dtype=np.float64)
    P0 = np.eye(3, dtype=np.float64)
    states, _ = kalman_3d_batch(m, x0, P0, Q, float(measurement_variance), 1.0)
    value = states[:, 0, 0]
    speed = states[:, 1, 0]
    accel = states[:, 2, 0]

    fig = make_subplots(
        rows=3, cols=2,
        shared_yaxes=True,
        shared_xaxes=True,
        column_widths=[0.25, 0.75],
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.04,
        specs=[[{}, {}], [None, {}], [None, {}]],
    )
    _show = _make_show_fn()
    peaks_df, metrics_df = _add_continued_width_content(fig, data, _show, 1, 2)
    _add_kalman_overlay(fig, data["lb_x"], value, _show, 1, 2)

    # Speed subchart
    fig.add_trace(go.Scatter(
        x=data["lb_x"], y=speed, mode="lines",
        line=dict(color="teal", width=1.5),
        name="speed", legendgroup="speed",
        showlegend=_show("speed"),
    ), row=2, col=2)
    fig.add_vline(x=1.0, line=dict(color="gray", width=1, dash="dash"), row=2, col=2)

    # Acceleration subchart
    fig.add_trace(go.Scatter(
        x=data["lb_x"], y=accel, mode="lines",
        line=dict(color="indianred", width=1.5),
        name="acceleration", legendgroup="acceleration",
        showlegend=_show("acceleration"),
    ), row=3, col=2)
    fig.add_vline(x=1.0, line=dict(color="gray", width=1, dash="dash"), row=3, col=2)

    fig.update_yaxes(range=[-1, 1], row=1, col=1)
    fig.update_yaxes(range=[-1, 1], row=1, col=2)
    fig.update_yaxes(title_text="speed", row=2, col=2)
    fig.update_yaxes(title_text="acceleration", row=3, col=2)
    fig.update_layout(
        title=_chart_title(data),
        height=850,
        legend=dict(groupclick="togglegroup"),
    )

    fig.show()
    _display_dfs(peaks_df, metrics_df)
    return fig
