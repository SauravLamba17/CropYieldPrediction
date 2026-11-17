"""Predict: set a field's conditions, get the predicted yield, its 90% range and why.

Simple mode shows the 4 inputs that matter most plus the scenarios; the other 10 sit in a closed
"More field conditions" expander with their values kept. Show details keeps every input in view.
"""

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from ui import charts, data, labels, plain
from ui.components import caption, card_title, chart_headline, headline_card, show_table
from ui.plain import TIPS
from ui.report import report_csv, report_frame, report_pdf
from ui.theme import GROUP_SERIES, tip

TOP_CONTRIBUTIONS = 8  # the explanation shows the 8 largest; the rest are summed
MAIN_INPUTS = ["fertilizer_amount", "pesticide_usage", "total_rainfall", "crop_type"]  # Simple mode
WHY, WHAT_IF = "Why this prediction", "What if I change…"
EYEBROW = {"soil": "soil", "sky": "sky", "green": "green", "gold_chart": "gold"}

simple = data.is_simple()
names = data.model_names()
summary = data.data_summary()
metadata = data.app_metadata()
inputs = data.input_tuple()
predicted = data.prediction(inputs)
band = data.interval()
lo, hi = (predicted - band["quantile"], predicted + band["quantile"]) if band else (None, None)
q25, q50, q75 = summary["quartiles"]
rating = plain.rating(predicted, q25, q50, q75)
share_below = np.searchsorted(summary["sorted_yields"], predicted) / summary["n_train"]

if simple:
    headline_card(plain.result_sentence(predicted, lo, hi))
else:
    headline_card(plain.predict_headline(predicted, rating, share_below, summary["n_train"]))


def input_widget(col: str) -> None:
    """One slider (numeric) or selectbox (categorical); the value lives in session state."""
    key, help_text = f"in_{col}", plain.INPUT_TIPS.get(col)
    if col in metadata["numeric"]:
        m = metadata["numeric"][col]
        low, high = (int(m["min"]), int(m["max"])) if m["is_integer"] else (m["min"], m["max"])
        st.slider(labels.label(col, simple), low, high, key=key, help=help_text,
                  on_change=data.clear_preset)
    else:
        st.selectbox(labels.label(col, simple), metadata["categorical"][col], key=key,
                     on_change=data.clear_preset)


with st.container(key="card-inputs"):
    head, scenario, button = st.columns([30, 50, 17], vertical_alignment="center")
    with head:
        card_title("Field conditions", "Move any input or pick a scenario.", inline=True)
    with scenario:
        st.pills("Scenario", data.PRESETS, key="preset", on_change=data.apply_preset,
                 label_visibility="collapsed", width="stretch")
    with button:
        st.button("Reset to defaults", type="primary", icon=":material/restart_alt:",
                  on_click=data.reset_inputs, key="reset", width="stretch")
    if simple:  # the 4 that matter most in view; the rest closed, values kept
        for column, feature in zip(st.columns(4, gap="large"), MAIN_INPUTS):
            with column:
                input_widget(feature)
        with st.expander("More field conditions", icon=":material/tune:", expanded=False):
            others = [f for f in config.FEATURES if f not in MAIN_INPUTS]
            columns = st.columns(4, gap="large")
            for i, feature in enumerate(others):
                with columns[i % 4]:
                    input_widget(feature)
    else:
        # Soil (5 inputs) spans two sub-columns (3 + 2), so every group is at most 3 rows tall.
        columns = st.columns([2, 1, 1.15, 0.85], gap="large")
        for column, (group, features) in zip(columns, config.FEATURE_GROUPS.items()):
            with column:
                css = EYEBROW[GROUP_SERIES[group]]
                st.html(f'<p class="eyebrow {css}"><span class="dot"></span>{labels.group(group, simple)}</p>')
                if len(features) > 3:
                    for sub, part in zip(st.columns(2, gap="large"), (features[:3], features[3:])):
                        with sub:
                            for feature in part:
                                input_widget(feature)
                else:
                    for feature in features:
                        input_widget(feature)

with st.spinner("Explaining this prediction (the first time takes a few seconds)…", show_time=True):
    base_value, contributions = data.explanation(inputs)
series = pd.Series(contributions)
ranked = series.reindex(series.abs().sort_values(ascending=False).index)
top, rest = ranked.iloc[:TOP_CONTRIBUTIONS], ranked.iloc[TOP_CONTRIBUTIONS:]
range_html = (
    f'<p class="range-line">90% range <b>{lo:.2f}–{hi:.2f} t/ha</b>'
    f"{tip(TIPS['interval'] if simple else TIPS['interval_method'], left=True)}</p>"
    if band else ""
)

hero_col, chart_col = st.columns([35, 65], gap="medium")
if simple:
    with hero_col, st.container(key="hero-gauge", height="stretch"):
        st.html(f'<p class="gauge-top">Predicted yield{tip(TIPS["gauge"])}</p>')
        chart_headline(plain.gauge_headline(rating, share_below))
        bands = [("Low", summary["min"], q25), ("Typical", q25, q50), ("Good", q50, q75),
                 ("Excellent", q75, summary["max"])]
        st.altair_chart(charts.gauge_chart(predicted, summary["min"], summary["max"], bands,
                                           height=110, interval=(lo, hi) if band else None))
        st.html(
            f'<p class="gauge-value"><span class="num">{predicted:.2f}</span> '
            f'<span class="unit">t/ha</span></p>{range_html}'
            f'<p class="gauge-rating"><b>{rating}</b>&nbsp; average prediction '
            f"{base_value:.2f} t/ha ({plain.signed(predicted - base_value)})"
            f"{tip(TIPS['average'], left=True)}</p>"
        )
    with chart_col, st.container(key="card-explain"):
        # The toggle gets its own row: beside a title it was clipped at 1024 px.
        view = st.segmented_control("Explain", [WHY, WHAT_IF], required=True,
                                    key="explain_view", label_visibility="collapsed")
        if view == WHAT_IF:
            factor = st.selectbox("Factor", config.NUMERIC_FEATURES,
                                  format_func=lambda f: labels.name(f, True),
                                  key="whatif_feature", label_visibility="collapsed")
            curve = data.whatif_curve(factor, inputs)
            chart_headline(plain.curve_headline(labels.name(factor, True), labels.unit(factor),
                                                curve["x"], curve["y"]))
            st.altair_chart(charts.curve_chart(curve, st.session_state[f"in_{factor}"],
                                               labels.label(factor, True), 170,
                                               band=band["quantile"] if band else None))
            st.html(f'<p class="plain-text">{plain.curve_caption(labels.name(factor, True), labels.unit(factor), curve["x"], curve["y"])}'
                    f"{tip(TIPS['whatif'], left=True)}</p>")
        else:
            friendly = {labels.name(f, True): float(v) for f, v in contributions.items()}
            card_title("What moved it (t/ha)", help_text=plain.tip_text("contributions", True))
            chart_headline(plain.contribution_headline(friendly))
            rows = [{"feature": labels.name(f, True), "value": float(v)} for f, v in top.items()]
            rows.append({"feature": f"Other factors ({len(rest)})", "value": float(rest.sum())})
            chart_data = pd.DataFrame(rows).assign(order=range(len(rows)))
            st.altair_chart(charts.contribution_chart(chart_data, height=240))
            st.html(f'<p class="plain-text">{plain.explanation_sentence(friendly)}</p>')
else:
    boost, drag = series.idxmax(), series.idxmin()
    boost_text = f"{labels.name(boost)} ({plain.signed(series[boost])})" if series[boost] > 0 else "none"
    drag_text = f"{labels.name(drag)} ({plain.signed(series[drag])})" if series[drag] < 0 else "none"
    with hero_col, st.container(key="hero-box", height="stretch"):
        interval_line = (
            f'<p class="line">90% interval: <b>{lo:.2f}–{hi:.2f} t/ha</b> (± {band["quantile"]:.3f})'
            f"{tip(TIPS['interval_method'], left=True)}</p>" if band else ""
        )
        st.html(
            '<div class="hero">'
            f'<p class="label">Predicted yield · {labels.model(names["best_overall"])}</p>'
            f'<p class="value">{predicted:.2f} <span>t/ha</span></p>'
            f'<p class="vs">vs benchmark average {base_value:.2f} t/ha '
            f"({plain.signed(predicted - base_value)})</p>{interval_line}<hr>"
            f'<p class="line">Biggest boost: <b>{boost_text}</b></p>'
            f'<p class="line">Biggest drag: <b>{drag_text}</b></p></div>'
        )
    with chart_col, st.container(key="card-explain"):
        card_title("SHAP waterfall: from the average prediction to this one (t/ha)",
                   "Hover a bar for the input value and its exact push", help_text=TIPS["contributions"])
        shown = {f: (f"{v:.2f} {labels.unit(f)}".strip() if isinstance(v, float) else str(v))
                 for f, v in zip(config.FEATURES, inputs)}
        steps = pd.DataFrame(
            [{"feature": labels.name(f), "value": float(v), "shown": shown[f]} for f, v in top.items()]
            + [{"feature": f"Other features ({len(rest)})", "value": float(rest.sum()),
                "shown": ", ".join(labels.name(f) for f in rest.index)}]
        )
        st.altair_chart(charts.waterfall_chart(base_value, steps, predicted, height=300))

coverage = data.load_table("conformal_groups")
if not simple and band and coverage is not None:
    with st.container(key="card-coverage"):
        card_title(
            "Is the 90% range equally good for every group?",
            f"Share of the {band['n_test']} test fields whose real yield fell inside the range",
            help_text=TIPS["coverage_groups"],
        )
        shown = coverage.assign(
            group_by=coverage["group_by"].map(labels.name),
            coverage=coverage["coverage"].map(lambda c: f"{c:.1%}"),
            vs_target=(coverage["coverage"] - band["level"]).map(lambda d: f"{d * 100:+.1f} pts".replace("-", "−")),
        ).rename(columns={"group_by": "Grouped by", "group": "Group", "n": "Test fields",
                          "coverage": "Coverage", "vs_target": "vs 90%"})
        left, right = st.columns(2, gap="medium")
        for column, name in zip((left, right), ("Crop type", "Region")):
            with column:
                show_table(shown[shown["Grouped by"] == name].drop(columns="Grouped by"))
        low = coverage.loc[coverage["coverage"].idxmin()]
        caption(
            f"Overall coverage {band['test_coverage']:.1%}; lowest group {labels.name(low['group_by']).lower()} "
            f"{low['group']} at {low['coverage']:.1%}. One width (± {band['quantile']:.2f} t/ha) is used for every "
            "field, so groups whose yields are harder to predict are covered less than 90% and easier ones more."
        )

with st.container(key="card-report"):
    left, csv_col, pdf_col = st.columns([6, 2, 2], vertical_alignment="center")
    with left:
        card_title("Prediction report",
                   "Inputs, prediction, 90% range, model and the top 5 factors, with a timestamp.",
                   inline=True)
    frame = report_frame(
        dict(zip(config.FEATURES, inputs)), predicted, lo, hi, names["best_overall"],
        contributions, data.now_utc(), data.APP_VERSION, data.app_commit(),
    )
    with csv_col:
        st.download_button("Download CSV", report_csv(frame), file_name="crop_yield_report.csv",
                           mime="text/csv", icon=":material/download:", key="dl_csv",
                           width="stretch", on_click="ignore")
    with pdf_col:
        st.download_button("Download PDF", lambda: report_pdf(frame), file_name="crop_yield_report.pdf",
                           mime="application/pdf", icon=":material/picture_as_pdf:", key="dl_pdf",
                           width="stretch", on_click="ignore")
