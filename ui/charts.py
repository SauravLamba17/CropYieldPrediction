"""Altair charts in the 'Cyber Sand' palette. Every chart goes through style()."""

import math

import altair as alt
import pandas as pd

from ui.theme import INDIGO, NAVY, PURPLE, navy

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
LABEL_SIZE = 14  # typography floor for axis and value labels
BAND_PADDING = 0.3  # gap between bars as a share of each row: bars never touch


def style(chart: alt.TopLevelMixin, height: int) -> alt.TopLevelMixin:
    """Apply the palette: transparent background, navy text, light navy gridlines, no border."""
    return (
        chart.properties(
            height=height,
            width="container",
            background="transparent",
            # "fit" + contains="padding" keeps long axis labels inside the card (no clipping).
            autosize=alt.AutoSizeParams(type="fit", contains="padding"),
            padding={"left": 4, "right": 8, "top": 4, "bottom": 4},
        )
        .configure(font=FONT)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=NAVY,
            titleColor=navy(0.65),
            labelFontSize=LABEL_SIZE,
            titleFontSize=LABEL_SIZE,
            titleFontWeight=500,
            gridColor=navy(0.08),
            domainColor=navy(0.15),
            tickColor=navy(0.15),
            labelLimit=260,
        )
        .configure_text(font=FONT, fontSize=LABEL_SIZE, color=NAVY)
        .configure_legend(
            labelFontSize=LABEL_SIZE, titleFontSize=LABEL_SIZE, labelColor=NAVY, labelLimit=260
        )
    )


def _value_labels(base: alt.Chart, value: str, fmt: str) -> alt.LayerChart:
    """Signed value labels just outside each bar (right of positive, left of negative)."""
    pos = base.transform_filter(f"datum['{value}'] >= 0").mark_text(
        align="left", dx=5, fontWeight=600
    )
    neg = base.transform_filter(f"datum['{value}'] < 0").mark_text(
        align="right", dx=-5, fontWeight=600
    )
    return pos.encode(text=alt.Text(f"{value}:Q", format=fmt)) + neg.encode(
        text=alt.Text(f"{value}:Q", format=fmt)
    )


def contribution_chart(data: pd.DataFrame, height: int) -> alt.TopLevelMixin:
    """Signed per-feature contributions (t/ha): positive indigo, negative purple, largest first.

    `data` has columns: feature (readable name), value (t/ha), order (display rank).
    """
    frame = data.assign(
        direction=["Raises yield" if v >= 0 else "Lowers yield" for v in data["value"]]
    )
    limit = max(abs(frame["value"]).max(), 0.01) * 1.35  # room for the labels
    # An explicit list keeps the magnitude order; field-based sorts are dropped in layers.
    y = alt.Y(
        "feature:N",
        sort=list(frame.sort_values("order")["feature"]),
        title=None,
        scale=alt.Scale(paddingInner=BAND_PADDING),
    )
    # No axis title: the card title already states the unit (t/ha).
    x = alt.X(
        "value:Q",
        title=None,
        scale=alt.Scale(domain=[-limit, limit]),
        axis=alt.Axis(tickCount=7, format="+.2f"),  # few, signed ticks: no label crowding
    )
    base = alt.Chart(frame).encode(y=y, x=x)
    bars = base.mark_bar(cornerRadius=4).encode(
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(domain=["Raises yield", "Lowers yield"], range=[INDIGO, PURPLE]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("feature:N", title="Feature"),
            alt.Tooltip("value:Q", title="t/ha", format="+.3f"),
        ],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=navy(0.35)).encode(x="x:Q")
    return style(alt.layer(bars, zero, _value_labels(base, "value", "+.2f")), height)


def ranked_bar_chart(
    data: pd.DataFrame,
    label: str,
    value: str,
    value_title: str,
    fmt: str,
    height: int,
    highlight: set[str] | None = None,
    muted: set[str] | None = None,
    axis_format: str | None = None,
) -> alt.TopLevelMixin:
    """Horizontal bars in the given row order.

    Rows in `highlight` are indigo, rows in `muted` are faint navy, all others mid navy,
    which keeps accents rare.
    """
    highlight, muted = highlight or set(), muted or set()
    tone = ["accent" if r in highlight else "muted" if r in muted else "base" for r in data[label]]
    frame = data.assign(tone=tone, order=range(len(data)))
    base = alt.Chart(frame).encode(
        y=alt.Y(
            f"{label}:N",
            sort=list(frame[label]),
            title=None,
            scale=alt.Scale(paddingInner=BAND_PADDING),
        ),
        x=alt.X(
            f"{value}:Q",
            title=value_title,
            scale=alt.Scale(domainMax=frame[value].max() * 1.18),
            # A handful of ticks keeps axis labels from colliding on narrow cards.
            axis=alt.Axis(format=axis_format, tickCount=6)
            if axis_format
            else alt.Axis(tickCount=6),
        ),
    )
    bars = base.mark_bar(cornerRadius=4).encode(
        color=alt.Color(
            "tone:N",
            scale=alt.Scale(
                domain=["accent", "base", "muted"], range=[INDIGO, navy(0.45), navy(0.18)]
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip(f"{label}:N", title=""),
            alt.Tooltip(f"{value}:Q", title=value_title, format=fmt),
        ],
    )
    labels = base.mark_text(align="left", dx=5, fontWeight=600).encode(
        text=alt.Text(f"{value}:Q", format=fmt)
    )
    return style(alt.layer(bars, labels), height)


def gauge_chart(
    value: float, lo: float, hi: float, bands: list[tuple[str, float, float]], height: int
):
    """A semicircle gauge on the hero gradient: white-tint bands and a navy needle.

    `bands` is [(name, start, end), ...] in t/ha; the needle is clipped to [lo, hi].
    """

    def angle(v: float) -> float:  # lo -> 9 o'clock, hi -> 3 o'clock (Vega: 0 rad = 12 o'clock)
        return -math.pi / 2 + math.pi * (min(max(v, lo), hi) - lo) / (hi - lo)

    tints = [
        "rgba(255,255,255,0.22)",
        "rgba(255,255,255,0.40)",
        "rgba(255,255,255,0.62)",
        "#FFFFFF",
    ]
    frame = pd.DataFrame(
        {
            "band": [b[0] for b in bands],
            "range": [f"{b[1]:.2f}–{b[2]:.2f} t/ha" for b in bands],
            "t0": [angle(b[1]) for b in bands],
            "t1": [angle(b[2]) for b in bands],
        }
    )
    radius = {"expr": "min(width / 2, height) - 4"}
    centre = {"x": {"expr": "width / 2"}, "y": {"expr": "height - 2"}}
    arcs = (
        alt.Chart(frame)
        .mark_arc(
            innerRadius={"expr": "(min(width / 2, height) - 4) * 0.62"},
            outerRadius=radius,
            padAngle=0.015,
            **centre,
        )
        .encode(
            theta=alt.Theta("t0:Q", scale=None),
            theta2="t1:Q",
            color=alt.Color(
                "band:N", scale=alt.Scale(domain=list(frame["band"]), range=tints), legend=None
            ),
            tooltip=[alt.Tooltip("band:N", title="Band"), alt.Tooltip("range:N", title="Yield")],
        )
    )
    a = angle(value)
    needle = (
        alt.Chart(pd.DataFrame({"t0": [a - 0.035], "t1": [a + 0.035]}))
        .mark_arc(
            innerRadius=0,
            outerRadius={"expr": "(min(width / 2, height) - 4) * 0.98"},
            color=NAVY,
            **centre,
        )
        .encode(theta=alt.Theta("t0:Q", scale=None), theta2="t1:Q")
    )
    hub = (
        alt.Chart(pd.DataFrame({"t0": [0.0], "t1": [2 * math.pi]}))
        .mark_arc(innerRadius=0, outerRadius=7, color=NAVY, **centre)
        .encode(theta=alt.Theta("t0:Q", scale=None), theta2="t1:Q")
    )
    return style(alt.layer(arcs, needle, hub), height)


def curve_chart(curve: pd.DataFrame, current: float | None, x_title: str, height: int):
    """Predicted yield (t/ha) as one factor sweeps its range; a purple dot marks `current`.

    `curve` has columns: x, y.
    """
    x = alt.X(
        "x:Q", title=x_title, scale=alt.Scale(zero=False, nice=False), axis=alt.Axis(tickCount=6)
    )
    y = alt.Y(
        "y:Q",
        title="Yield (t/ha)",
        scale=alt.Scale(zero=False),
        axis=alt.Axis(tickCount=4, format=".2f"),
    )
    tooltip = [
        alt.Tooltip("x:Q", title=x_title, format=",.2f"),
        alt.Tooltip("y:Q", title="Predicted t/ha", format=".2f"),
    ]
    line = (
        alt.Chart(curve)
        .mark_line(color=INDIGO, strokeWidth=3, interpolate="monotone")
        .encode(x=x, y=y, tooltip=tooltip)
    )
    hover = alt.Chart(curve).mark_circle(size=90, opacity=0).encode(x=x, y=y, tooltip=tooltip)
    layers = [line, hover]
    if current is not None:
        dot = curve.iloc[(curve["x"] - current).abs().argsort()[:1]].assign(x=current)
        layers.append(
            alt.Chart(dot)
            .mark_circle(size=140, color=PURPLE, opacity=1, stroke="#FFFFFF", strokeWidth=2)
            .encode(x=x, y=y, tooltip=tooltip)
        )
    return style(alt.layer(*layers), height)


def donut_chart(data: pd.DataFrame, height: int):
    """Share of importance per factor group, largest first; `data` has columns: label, share."""
    colours = [INDIGO, PURPLE, navy(0.45), navy(0.18)]
    order = list(data["label"])
    return style(
        alt.Chart(data.assign(rank=range(len(data))))
        .mark_arc(innerRadius={"expr": "min(width, height) * 0.24"}, padAngle=0.02, cornerRadius=3)
        .encode(
            theta=alt.Theta("share:Q", stack=True),
            order=alt.Order("rank:Q"),
            color=alt.Color(
                "label:N",
                scale=alt.Scale(domain=order, range=colours[: len(order)]),
                legend=alt.Legend(
                    title=None,
                    orient="right",
                    symbolType="circle",
                    symbolSize=160,
                    labelFontWeight=600,
                    rowPadding=8,
                ),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Group"),
                alt.Tooltip("share:Q", title="Share", format=".0%"),
            ],
        ),
        height,
    )


def scatter_chart(data: pd.DataFrame, height: int):
    """Predicted vs real yield for held-out fields with a y = x line (perfect predictions).

    `data` has columns: real, predicted, crop, region.
    """
    lo = float(min(data["real"].min(), data["predicted"].min()))
    hi = float(max(data["real"].max(), data["predicted"].max()))
    scale = alt.Scale(domain=[math.floor(lo), math.ceil(hi)], nice=False)
    axis = alt.Axis(values=list(range(math.floor(lo), math.ceil(hi) + 1)), format="d")
    points = (
        alt.Chart(data)
        .mark_circle(size=34, color=INDIGO, opacity=0.45)
        .encode(
            x=alt.X("real:Q", title="Real yield (t/ha)", scale=scale, axis=axis),
            y=alt.Y("predicted:Q", title="Predicted yield (t/ha)", scale=scale, axis=axis),
            tooltip=[
                alt.Tooltip("real:Q", title="Real yield (t/ha)", format=".2f"),
                alt.Tooltip("predicted:Q", title="Predicted (t/ha)", format=".2f"),
                alt.Tooltip("crop:N", title="Crop"),
                alt.Tooltip("region:N", title="Region"),
            ],
        )
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"v": [math.floor(lo), math.ceil(hi)]}))
        .mark_line(color=PURPLE, strokeDash=[6, 4], strokeWidth=2)
        .encode(x="v:Q", y="v:Q")
    )
    return style(alt.layer(points, diagonal), height)


def verdict_bar_chart(data: pd.DataFrame, height: int):
    """Relative error per crop, coloured 'works well' (indigo) / 'struggles' (purple).

    `data` has columns: crop, relative (share of the crop's average yield), verdict.
    """
    base = alt.Chart(data).encode(
        y=alt.Y(
            "crop:N",
            sort=list(data["crop"]),
            title=None,
            scale=alt.Scale(paddingInner=BAND_PADDING),
        ),
        x=alt.X(
            "relative:Q",
            title="Typical error as a share of the crop's average yield",
            axis=alt.Axis(format="%", tickCount=5),
            scale=alt.Scale(domainMax=float(data["relative"].max()) * 1.2),
        ),
    )
    bars = base.mark_bar(cornerRadius=4).encode(
        color=alt.Color(
            "verdict:N",
            scale=alt.Scale(domain=["works well", "struggles"], range=[INDIGO, PURPLE]),
            legend=alt.Legend(title=None, orient="top", symbolType="circle"),
        ),
        tooltip=[
            alt.Tooltip("crop:N", title="Crop"),
            alt.Tooltip("relative:Q", title="Error / average yield", format=".0%"),
            alt.Tooltip("verdict:N", title="Verdict"),
        ],
    )
    labels = base.mark_text(align="left", dx=5, fontWeight=600).encode(
        text=alt.Text("relative:Q", format=".0%")
    )
    return style(alt.layer(bars, labels), height)
