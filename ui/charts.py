"""Altair charts in the 'Field & Harvest' theme.

Colours are never hard-coded per mode: every mark is coloured by a *series name* from
ui.theme.SERIES, and Streamlit maps those names to the light or dark chartCategoricalColors
from config.toml. Axis text, gridlines and backgrounds also come from Streamlit's theme,
so a chart restyles itself when the viewer switches theme. Charts are rendered with
st.altair_chart(chart) (Streamlit theme on); render() below adds the sizing.
"""

import math

import altair as alt
import numpy as np
import pandas as pd

from ui.theme import SERIES

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
LABEL_SIZE = 13  # never smaller on axes and value labels
BAND_PADDING = 0.3  # gap between bars as a share of each row
PALETTE = alt.Scale(domain=SERIES)  # position in SERIES = position in chartCategoricalColors


def colour(field: str, legend=None) -> alt.Color:
    """Colour by a column that holds series names (e.g. 'green', 'shap_neg')."""
    return alt.Color(f"{field}:N", scale=PALETTE, legend=legend)


SOLID_PREFIX = "_series_"


def solid(series: str) -> alt.Color:
    """One fixed series colour (e.g. the theme's text colour for value labels).

    Vega-Lite cannot put a theme scale on a constant colour, so this colours by a field
    named after the series; style() adds that constant field to every layer using it.
    """
    return alt.Color(f"{SOLID_PREFIX}{series}:N", scale=PALETTE, legend=None)


def _add_solid_fields(chart: alt.TopLevelMixin) -> None:
    """Give each layer coloured by solid(name) a constant column holding `name`."""
    units = chart.layer if isinstance(chart, alt.LayerChart) else [chart]
    for unit in units:
        if isinstance(unit, alt.LayerChart):
            _add_solid_fields(unit)
            continue
        color = getattr(unit.encoding, "color", alt.Undefined) if unit.encoding is not alt.Undefined else alt.Undefined
        shorthand = getattr(color, "shorthand", alt.Undefined)
        if isinstance(shorthand, str) and shorthand.startswith(SOLID_PREFIX):
            field = shorthand.split(":")[0]
            series = field[len(SOLID_PREFIX):]
            existing = [] if unit.transform is alt.Undefined else list(unit.transform)
            unit.transform = existing + [alt.CalculateTransform(calculate=f"'{series}'", **{"as": field})]


def style(chart: alt.TopLevelMixin, height: int) -> alt.TopLevelMixin:
    """Size, font and readable tick density; colours are left to the Streamlit theme."""
    _add_solid_fields(chart)
    return (
        chart.properties(
            height=height,
            width="container",
            autosize=alt.AutoSizeParams(type="fit", contains="padding"),
            padding={"left": 4, "right": 10, "top": 6, "bottom": 4},
        )
        .configure(font=FONT, background="transparent")  # cards show through (both modes)
        .configure_view(stroke=None, fill=None)
        .configure_axis(
            labelFontSize=LABEL_SIZE,
            titleFontSize=LABEL_SIZE,
            titleFontWeight=600,
            labelLimit=240,
            labelFlush=True,
            tickCount=5,
        )
        .configure_axisX(labelOverlap="greedy")  # drop colliding numeric ticks on narrow cards
        .configure_axisY(labelOverlap=False)  # never hide a category name
        .configure_text(font=FONT, fontSize=LABEL_SIZE)
        .configure_legend(labelFontSize=LABEL_SIZE, titleFontSize=LABEL_SIZE, labelLimit=260)
    )


def _value_labels(base: alt.Chart, value: str, fmt: str) -> alt.LayerChart:
    """Signed value labels just outside each bar (right of positive, left of negative)."""
    text = alt.Text(f"{value}:Q", format=fmt)
    pos = base.transform_filter(f"datum['{value}'] >= 0").mark_text(
        align="left", dx=5, fontWeight=600
    )
    neg = base.transform_filter(f"datum['{value}'] < 0").mark_text(
        align="right", dx=-5, fontWeight=600
    )
    return pos.encode(text=text, color=solid("text")) + neg.encode(text=text, color=solid("text"))


def contribution_chart(data: pd.DataFrame, height: int, unit: str = "t/ha") -> alt.TopLevelMixin:
    """Signed per-factor contributions, largest first: green raises, orange lowers.

    `data` has columns: feature (display name), value, order.
    """
    frame = data.assign(
        series=np.where(data["value"] >= 0, "shap_pos", "shap_neg"),
        direction=np.where(data["value"] >= 0, "Raises yield", "Lowers yield"),
    )
    limit = max(float(frame["value"].abs().max()), 0.01) * 1.4
    y = alt.Y(
        "feature:N",
        sort=list(frame.sort_values("order")["feature"]),
        title=None,
        scale=alt.Scale(paddingInner=BAND_PADDING),
    )
    x = alt.X(
        "value:Q", title=None, scale=alt.Scale(domain=[-limit, limit]), axis=alt.Axis(format="+.2f")
    )
    base = alt.Chart(frame).encode(y=y, x=x)
    bars = base.mark_bar(cornerRadius=4).encode(
        color=colour("series"),
        tooltip=[
            alt.Tooltip("feature:N", title="Factor"),
            alt.Tooltip("direction:N", title="Effect"),
            alt.Tooltip("value:Q", title=unit, format="+.3f"),
        ],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(opacity=0.5).encode(
        x="x:Q", color=solid("neutral")
    )
    return style(alt.layer(bars, zero, _value_labels(base, "value", "+.2f")), height)


def waterfall_chart(
    base_value: float, steps: pd.DataFrame, prediction: float, height: int
) -> alt.TopLevelMixin:
    """SHAP waterfall: from the average prediction, each factor's push, to this prediction.

    `steps` has columns: feature (display name), value (t/ha), shown (the input as text),
    already ordered largest |value| first. The axis is zoomed to the running total, so even
    small pushes are visible; the first and last bars mark the average and the prediction.
    """
    running = base_value + steps["value"].cumsum()
    ends = [base_value, *running, prediction]
    lo, hi = min(ends), max(ends)
    pad = max((hi - lo) * 0.25, 0.05)
    left = lo - pad
    rows = [{"feature": "Average prediction", "start": left, "end": base_value, "value": base_value,
             "shown": "starting point", "series": "neutral"}]
    total = base_value
    for r in steps.itertuples():
        rows.append({"feature": r.feature, "start": total, "end": total + r.value, "value": r.value,
                     "shown": r.shown, "series": "shap_pos" if r.value >= 0 else "shap_neg"})
        total += r.value
    rows.append({"feature": "This prediction", "start": left, "end": prediction, "value": prediction,
                 "shown": "average + every push", "series": "green"})
    frame = pd.DataFrame(rows)
    frame["label"] = [
        f"{v:.2f}" if i in (0, len(frame) - 1) else f"{v:+.2f}".replace("-", "−")
        for i, v in enumerate(frame["value"])
    ]
    frame["at"] = frame[["start", "end"]].max(axis=1)
    y = alt.Y("feature:N", sort=list(frame["feature"]), title=None, scale=alt.Scale(paddingInner=0.25))
    x = alt.X("start:Q", title="Predicted yield (t/ha)",
              scale=alt.Scale(domain=[left, hi + pad], zero=False, nice=False, clamp=True))
    tooltip = [
        alt.Tooltip("feature:N", title="Step"),
        alt.Tooltip("shown:N", title="Input"),
        alt.Tooltip("value:Q", title="t/ha", format="+.3f"),
        alt.Tooltip("end:Q", title="Running total (t/ha)", format=".2f"),
    ]
    bars = alt.Chart(frame).mark_bar(cornerRadius=3).encode(
        y=y, x=x, x2="end:Q", color=colour("series"), tooltip=tooltip
    )
    text = alt.Chart(frame).mark_text(align="left", dx=5, fontWeight=600).encode(
        y=y, x="at:Q", text="label:N", color=solid("text")
    )
    return style(alt.layer(bars, text), height)


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
    error: str | None = None,
) -> alt.TopLevelMixin:
    """Horizontal bars in the given row order; optional ± error bars from column `error`.

    Rows in `highlight` are green, `muted` rows faint, all others neutral.
    """
    highlight, muted = highlight or set(), muted or set()
    series = ["green" if r in highlight else "faint" if r in muted else "neutral" for r in data[label]]
    frame = data.assign(series=series)
    top = frame[value] + (frame[error] if error else 0)
    y = alt.Y(f"{label}:N", sort=list(frame[label]), title=None, scale=alt.Scale(paddingInner=BAND_PADDING))
    x = alt.X(
        f"{value}:Q",
        title=value_title,
        scale=alt.Scale(domainMax=float(top.max()) * 1.22),
        axis=alt.Axis(format=axis_format) if axis_format else alt.Axis(),
    )
    tooltip = [alt.Tooltip(f"{label}:N", title=""), alt.Tooltip(f"{value}:Q", title=value_title, format=fmt)]
    if error:
        tooltip.append(alt.Tooltip(f"{error}:Q", title="± sd across folds", format=fmt))
    base = alt.Chart(frame).encode(y=y, x=x)
    layers = [base.mark_bar(cornerRadius=4).encode(color=colour("series"), tooltip=tooltip)]
    if error:
        bounds = frame.assign(lo=frame[value] - frame[error], hi=frame[value] + frame[error])
        layers.append(
            alt.Chart(bounds).mark_rule(strokeWidth=2).encode(
                y=y, x="lo:Q", x2="hi:Q", color=solid("text"), tooltip=tooltip
            )
        )
        labels = alt.Chart(bounds).mark_text(align="left", dx=6, fontWeight=600).encode(
            y=y, x="hi:Q", text=alt.Text(f"{value}:Q", format=fmt), color=solid("text")
        )
    else:
        labels = base.mark_text(align="left", dx=5, fontWeight=600).encode(
            text=alt.Text(f"{value}:Q", format=fmt), color=solid("text")
        )
    layers.append(labels)
    return style(alt.layer(*layers), height)


def gauge_chart(
    value: float,
    lo: float,
    hi: float,
    bands: list[tuple[str, float, float]],
    height: int,
    interval: tuple[float, float] | None = None,
):
    """Semicircle gauge on the dark green hero card (fixed white tints read well in both modes).

    `bands` is [(name, start, end), ...] in t/ha. `interval` (90% range) is drawn as a gold
    ring just outside the bands.
    """

    def angle(v: float) -> float:  # lo -> 9 o'clock, hi -> 3 o'clock
        return -math.pi / 2 + math.pi * (min(max(v, lo), hi) - lo) / (hi - lo)

    tints = ["rgba(255,255,255,0.20)", "rgba(255,255,255,0.38)", "rgba(255,255,255,0.60)", "#FFFFFF"]
    frame = pd.DataFrame(
        {
            "band": [b[0] for b in bands],
            "range": [f"{b[1]:.2f}–{b[2]:.2f} t/ha" for b in bands],
            "t0": [angle(b[1]) for b in bands],
            "t1": [angle(b[2]) for b in bands],
        }
    )
    outer = "min(width / 2, height) - 10"
    centre = {"x": {"expr": "width / 2"}, "y": {"expr": "height - 2"}}
    layers = [
        alt.Chart(frame)
        .mark_arc(innerRadius={"expr": f"({outer}) * 0.62"}, outerRadius={"expr": outer},
                  padAngle=0.015, **centre)
        .encode(
            theta=alt.Theta("t0:Q", scale=None),
            theta2="t1:Q",
            color=alt.Color("band:N", scale=alt.Scale(domain=list(frame["band"]), range=tints), legend=None),
            tooltip=[alt.Tooltip("band:N", title="Band"), alt.Tooltip("range:N", title="Yield")],
        )
    ]
    if interval is not None:
        ring = pd.DataFrame(
            {"t0": [angle(interval[0])], "t1": [angle(interval[1])],
             "range": [f"{interval[0]:.2f}–{interval[1]:.2f} t/ha"]}
        )
        layers.append(
            alt.Chart(ring)
            .mark_arc(innerRadius={"expr": f"({outer}) + 3"}, outerRadius={"expr": f"({outer}) + 9"},
                      cornerRadius=3, color="#F2C14E", **centre)
            .encode(theta=alt.Theta("t0:Q", scale=None), theta2="t1:Q",
                    tooltip=[alt.Tooltip("range:N", title="90% range")])
        )
    a = angle(value)
    layers.append(
        alt.Chart(pd.DataFrame({"t0": [a - 0.035], "t1": [a + 0.035]}))
        .mark_arc(innerRadius=0, outerRadius={"expr": f"({outer}) * 0.98"}, color="#0F2410", **centre)
        .encode(theta=alt.Theta("t0:Q", scale=None), theta2="t1:Q")
    )
    layers.append(
        alt.Chart(pd.DataFrame({"t0": [0.0], "t1": [2 * math.pi]}))
        .mark_arc(innerRadius=0, outerRadius=7, color="#0F2410", **centre)
        .encode(theta=alt.Theta("t0:Q", scale=None), theta2="t1:Q")
    )
    return style(alt.layer(*layers), height).configure_view(fill=None, stroke=None)


def curve_chart(
    curve: pd.DataFrame,
    current: float | None,
    x_title: str,
    height: int,
    band: float | None = None,
):
    """Predicted yield as one factor sweeps its range; a gold dot marks `current`.

    `curve` has columns x, y. `band` (t/ha) shades the 90% range around the line.
    """
    x = alt.X("x:Q", title=x_title, scale=alt.Scale(zero=False, nice=False))
    y = alt.Y("y:Q", title="Predicted yield (t/ha)", scale=alt.Scale(zero=False), axis=alt.Axis(format=".1f"))
    tooltip = [
        alt.Tooltip("x:Q", title=x_title, format=",.2f"),
        alt.Tooltip("y:Q", title="Predicted t/ha", format=".2f"),
    ]
    layers = []
    if band is not None:
        shaded = curve.assign(lo=curve["y"] - band, hi=curve["y"] + band)
        layers.append(
            alt.Chart(shaded).mark_area(opacity=0.18, interpolate="monotone").encode(
                x=x, y="lo:Q", y2="hi:Q", color=solid("green"),
                tooltip=[alt.Tooltip("lo:Q", title="90% range from", format=".2f"),
                         alt.Tooltip("hi:Q", title="to", format=".2f")],
            )
        )
    layers.append(
        alt.Chart(curve).mark_line(strokeWidth=3, interpolate="monotone").encode(
            x=x, y=y, color=solid("green"), tooltip=tooltip
        )
    )
    layers.append(alt.Chart(curve).mark_circle(size=90, opacity=0).encode(x=x, y=y, tooltip=tooltip))
    if current is not None:
        dot = curve.iloc[(curve["x"] - current).abs().argsort()[:1]].assign(x=current)
        layers.append(
            alt.Chart(dot).mark_circle(size=160, opacity=1, strokeWidth=2).encode(
                x=x, y=y, color=solid("gold_chart"), tooltip=tooltip
            )
        )
    return style(alt.layer(*layers), height)


def group_donut(data: pd.DataFrame, height: int):
    """Share of importance per factor group. `data` has columns: label, share, series."""
    names = {r.series: f"{r.label}  {r.share:.0%}" for r in data.itertuples()}
    label_expr = "{" + ", ".join(f"'{k}': '{v}'" for k, v in names.items()) + "}[datum.label]"
    return style(
        alt.Chart(data.assign(rank=range(len(data))))
        .mark_arc(innerRadius={"expr": "min(width, height) * 0.24"}, padAngle=0.02, cornerRadius=3)
        .encode(
            theta=alt.Theta("share:Q", stack=True),
            order=alt.Order("rank:Q"),
            color=alt.Color(
                "series:N",
                scale=PALETTE,
                legend=alt.Legend(
                    title=None, orient="right", values=list(names), labelExpr=label_expr,
                    symbolType="circle", symbolSize=160, labelFontWeight=600, rowPadding=8,
                ),
            ),
            tooltip=[alt.Tooltip("label:N", title="Group"), alt.Tooltip("share:Q", title="Share", format=".0%")],
        ),
        height,
    )


def scatter_chart(data: pd.DataFrame, height: int):
    """Predicted vs real yield for held-out fields, with the y = x line (perfect predictions)."""
    lo = math.floor(min(data["real"].min(), data["predicted"].min()))
    hi = math.ceil(max(data["real"].max(), data["predicted"].max()))
    scale = alt.Scale(domain=[lo, hi], nice=False)
    points = alt.Chart(data).mark_circle(size=34, opacity=0.5).encode(
        x=alt.X("real:Q", title="Real yield (t/ha)", scale=scale),
        y=alt.Y("predicted:Q", title="Predicted yield (t/ha)", scale=scale),
        color=solid("green"),
        tooltip=[
            alt.Tooltip("real:Q", title="Real (t/ha)", format=".2f"),
            alt.Tooltip("predicted:Q", title="Predicted (t/ha)", format=".2f"),
            alt.Tooltip("crop:N", title="Crop"),
            alt.Tooltip("region:N", title="Region"),
        ],
    )
    diagonal = alt.Chart(pd.DataFrame({"v": [lo, hi]})).mark_line(strokeDash=[6, 4], strokeWidth=2).encode(
        x="v:Q", y="v:Q", color=solid("gold_chart")
    )
    return style(alt.layer(points, diagonal), height)


def residual_chart(data: pd.DataFrame, height: int, band: float | None = None):
    """Residual (real − predicted) against the prediction; dashed lines at ± the 90% half-width."""
    points = alt.Chart(data).mark_circle(size=30, opacity=0.5).encode(
        x=alt.X("predicted:Q", title="Predicted yield (t/ha)", scale=alt.Scale(zero=False)),
        y=alt.Y("residual:Q", title="Real − predicted (t/ha)"),
        color=solid("sky"),
        tooltip=[
            alt.Tooltip("predicted:Q", title="Predicted (t/ha)", format=".2f"),
            alt.Tooltip("residual:Q", title="Residual (t/ha)", format="+.2f"),
            alt.Tooltip("crop:N", title="Crop"),
        ],
    )
    layers = [points, alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeWidth=2).encode(y="y:Q", color=solid("text"))]
    if band is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"y": [-band, band]})).mark_rule(strokeDash=[6, 4], strokeWidth=2).encode(
                y="y:Q", color=solid("gold_chart"),
                tooltip=[alt.Tooltip("y:Q", title="90% band edge (t/ha)", format="+.2f")],
            )
        )
    return style(alt.layer(*layers), height)


def histogram(values: pd.Series, title: str, series: str, height: int, bins: int = 30):
    """Distribution of one column with a dashed line at its mean."""
    frame = pd.DataFrame({"v": values.to_numpy()})
    bars = alt.Chart(frame).mark_bar(cornerRadius=2, opacity=0.9).encode(
        x=alt.X("v:Q", bin=alt.Bin(maxbins=bins), title=title),
        y=alt.Y("count():Q", title="Fields"),
        color=solid(series),
        tooltip=[alt.Tooltip("v:Q", bin=alt.Bin(maxbins=bins), title=title), alt.Tooltip("count():Q", title="Fields")],
    )
    mean = alt.Chart(pd.DataFrame({"m": [float(values.mean())]})).mark_rule(strokeDash=[6, 4], strokeWidth=2).encode(
        x="m:Q", color=solid("text"), tooltip=[alt.Tooltip("m:Q", title="Mean", format=".2f")]
    )
    return style(alt.layer(bars, mean), height)


def heatmap(corr: pd.DataFrame, height: int):
    """Correlation matrix (−1 to 1) with the value printed in every cell."""
    names = list(corr.columns)
    long = corr.reset_index(names="a").melt(id_vars="a", var_name="b", value_name="r")
    base = alt.Chart(long).encode(
        x=alt.X("b:N", sort=names, title=None, axis=alt.Axis(labelAngle=-40, labelOverlap=False)),
        y=alt.Y("a:N", sort=names, title=None, axis=alt.Axis(labelOverlap=False)),
    )
    cells = base.mark_rect(cornerRadius=2).encode(
        color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 1], scheme="brownbluegreen"),
                        legend=alt.Legend(title="r", orient="right")),
        tooltip=[alt.Tooltip("a:N", title=""), alt.Tooltip("b:N", title=""), alt.Tooltip("r:Q", format="+.3f")],
    )
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text("r:Q", format=".2f"),
        color=alt.condition("abs(datum.r) > 0.55", alt.value("#FFFFFF"), alt.value("#1F2A1F")),
    )
    return style(alt.layer(cells, text), height)


def box_chart(frame: pd.DataFrame, category: str, value: str, title: str, series: str, height: int):
    """Yield distribution per category level (box = middle half, whiskers = 1.5 IQR)."""
    order = sorted(frame[category].unique())
    return style(
        alt.Chart(frame).mark_boxplot(size=26, extent=1.5, median={"color": "#FFFFFF"}).encode(
            x=alt.X(f"{category}:N", sort=order, title=title, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{value}:Q", title="Yield (t/ha)", scale=alt.Scale(zero=False)),
            color=solid(series),
        ),
        height,
    )


def verdict_bar_chart(data: pd.DataFrame, height: int):
    """Relative error per crop: green 'works well', orange 'struggles'.

    `data` has columns: crop, relative (share of the crop's average yield), verdict.
    """
    frame = data.assign(series=np.where(data["verdict"] == "works well", "green", "shap_neg"))
    names = {"green": "works well", "shap_neg": "struggles"}
    label_expr = "{'green': 'works well', 'shap_neg': 'struggles'}[datum.label]"
    base = alt.Chart(frame).encode(
        y=alt.Y("crop:N", sort=list(frame["crop"]), title=None, scale=alt.Scale(paddingInner=BAND_PADDING)),
        x=alt.X(
            "relative:Q",
            title="Typical error ÷ the crop's average yield",
            axis=alt.Axis(format="%"),
            scale=alt.Scale(domainMax=float(frame["relative"].max()) * 1.25),
        ),
    )
    bars = base.mark_bar(cornerRadius=4).encode(
        color=alt.Color(
            "series:N",
            scale=PALETTE,
            legend=alt.Legend(title=None, orient="top", values=list(names), labelExpr=label_expr, symbolType="circle"),
        ),
        tooltip=[
            alt.Tooltip("crop:N", title="Crop"),
            alt.Tooltip("relative:Q", title="Error ÷ average yield", format=".0%"),
            alt.Tooltip("verdict:N", title="Verdict"),
        ],
    )
    labels = base.mark_text(align="left", dx=5, fontWeight=600).encode(
        text=alt.Text("relative:Q", format=".0%"), color=solid("text")
    )
    return style(alt.layer(bars, labels), height)


def beeswarm_chart(frame: pd.DataFrame, order: list[str], height: int):
    """Every held-out field's SHAP value per feature; colour = how high its input value was.

    `frame` has columns: feature (display name), shap, value_rank (0–1, NaN for categories),
    jitter (vertical offset, precomputed with a fixed seed so the plot never moves).
    """
    return style(
        alt.Chart(frame).mark_circle(size=16, opacity=0.75).encode(
            y=alt.Y("feature:N", sort=order, title=None),
            yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[-1, 1])),
            x=alt.X("shap:Q", title="Effect on predicted yield (t/ha)"),
            color=alt.Color(
                "value_rank:Q",
                scale=alt.Scale(scheme="goldgreen"),
                legend=alt.Legend(title="Input value (rank)", format=".0%", orient="bottom", gradientLength=180),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("shap:Q", title="Effect (t/ha)", format="+.3f"),
                alt.Tooltip("value_rank:Q", title="Input percentile", format=".0%"),
            ],
        ),
        height,
    )


def month_chart(frame: pd.DataFrame, height: int):
    """Mean yield per harvest month (raw Kaggle CSV only)."""
    stats = frame.groupby("month")["yield"].agg(["mean", "count"]).reset_index()
    return style(
        alt.Chart(stats).mark_bar(cornerRadius=3).encode(
            x=alt.X("month:O", title="Harvest month", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("mean:Q", title="Mean yield (t/ha)", scale=alt.Scale(zero=False)),
            color=solid("gold_chart"),
            tooltip=[alt.Tooltip("month:O", title="Month"), alt.Tooltip("mean:Q", title="Mean t/ha", format=".2f"),
                     alt.Tooltip("count:Q", title="Fields")],
        ),
        height,
    )
