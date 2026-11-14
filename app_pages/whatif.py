"""What-if: sweep one factor for the field set on Predict, and see what drives yield overall."""

import numpy as np
import pandas as pd
import streamlit as st
from src import config
from ui import charts, data, labels, plain
from ui.components import card_title, headline_card, setup_card
from ui.plain import TIPS
from ui.theme import GROUP_SERIES

simple = data.is_simple()
inputs = data.input_tuple()
band = data.interval()

agreement = data.load_table("importance_agreement")
if agreement is None:
    setup_card(
        "Setup needed: the feature-importance table is missing",
        "<code>importance_agreement.csv</code> is written by notebook 03. Rebuild everything with:",
        command="python run.py build",
    )
    st.stop()
shap_by_feature = agreement.set_index("feature")["mean_abs_shap"].sort_values(ascending=False)
group_totals = pd.Series(
    {g: shap_by_feature[cols].sum() for g, cols in config.FEATURE_GROUPS.items()}
).sort_values(ascending=False)
shares = group_totals / group_totals.sum()
signal = data.load_table("feature_signal")
rho = signal.set_index("feature")["spearman_rho"] if signal is not None else pd.Series(dtype=float)
top3 = list(shap_by_feature.index[:3])
negative = {labels.name(f, True).lower() for f in top3 if rho.get(f, 0) < 0}
headline_card(
    plain.drivers_headline(
        shares.index[0], shares.iloc[0], [labels.name(f, True).lower() for f in top3], negative
    )
)

with st.container(key="card-whatif"):
    head, pick = st.columns([3, 2], vertical_alignment="center")
    with head:
        card_title(
            "What if I change one factor?",
            "The prediction for the field set on the Predict page as one factor sweeps its "
            "range; everything else stays as set. Shaded: the 90% range.",
            help_text=TIPS["whatif"],
        )
    with pick:
        factor = st.selectbox(
            "Factor", config.NUMERIC_FEATURES, format_func=lambda f: labels.name(f, simple),
            key="whatif_feature", label_visibility="collapsed",
        )
    curve = data.whatif_curve(factor, inputs)
    st.altair_chart(
        charts.curve_chart(curve, st.session_state[f"in_{factor}"], labels.label(factor, simple),
                           250, band=band["quantile"] if band else None)
    )
    st.html(
        '<p class="plain-text">'
        f"{plain.curve_caption(labels.name(factor, simple), labels.unit(factor), curve['x'], curve['y'])}</p>"
    )
    st.page_link("app_pages/predict.py", label="Change the field on the Predict page",
                 icon=":material/arrow_back:")

with st.container(key="card-heatmap"):
    head, px, py = st.columns([4, 2, 2], vertical_alignment="center")
    with head:
        card_title(
            "Two factors at once",
            "Predicted yield over a 25 × 25 grid of two inputs, everything else as set on Predict",
            help_text=TIPS["heatmap"],
        )
    with px:
        fx = st.selectbox("Across", config.NUMERIC_FEATURES, format_func=lambda f: labels.name(f, simple),
                          key="heat_x")
    with py:
        fy_options = [f for f in config.NUMERIC_FEATURES if f != fx]
        if st.session_state.get("heat_y") == fx:  # never the same input on both axes
            st.session_state["heat_y"] = fy_options[0]
        fy = st.selectbox("Up", fy_options, format_func=lambda f: labels.name(f, simple), key="heat_y")
    with st.spinner("Predicting 625 field variations…", show_time=True):
        grid = data.heatmap(fx, fy, inputs)
    current_inputs = dict(zip(config.FEATURES, inputs))
    current = (current_inputs[fx], current_inputs[fy], data.prediction(inputs))
    top = grid.loc[grid["pred"].idxmax()]
    st.altair_chart(
        charts.heatmap_chart(grid, labels.label(fx, simple), labels.label(fy, simple), current,
                             (top["x"], top["y"], top["pred"]), 380)
    )
    st.html(
        '<p class="plain-text">'
        f"Best cell in this grid: <b>{labels.name(fx, simple)} {top['x']:.1f} {labels.unit(fx)}</b> with "
        f"<b>{labels.name(fy, simple).lower()} {top['y']:.1f} {labels.unit(fy)}</b> → <b>{top['pred']:.2f} t/ha</b>, "
        f"against <b>{current[2]:.2f} t/ha</b> now (◆). The grid stays inside the values seen in the "
        "training data: the model is only reliable inside the observed data range.</p>"
    )

group_side, feature_side = st.columns([2, 3], gap="medium")
if simple:
    with group_side, st.container(key="card-groups", height="stretch"):
        card_title("Which kinds of factors matter", "Share of what the model relies on",
                   help_text=TIPS["shap"])
        donut = pd.DataFrame({
            "label": [labels.group(g, True) for g in shares.index],
            "share": shares.to_numpy(),
            "series": [GROUP_SERIES[g] for g in shares.index],
        })
        st.altair_chart(charts.group_donut(donut, height=300))
    with feature_side, st.container(key="card-response", height="stretch"):
        head, pick = st.columns([3, 2], vertical_alignment="center")
        with head:
            card_title("How yield responds to each factor", "Average prediction over 300 real fields",
                       help_text=TIPS["response"])
        with pick:
            st.session_state.setdefault("response_feature", shap_by_feature.index[0])
            factor = st.selectbox(
                "Factor", config.NUMERIC_FEATURES,
                format_func=lambda f: labels.name(f, True), key="response_feature",
                label_visibility="collapsed",
            )
        curve = data.response_curve(factor)
        st.altair_chart(charts.curve_chart(curve, None, labels.label(factor, True), 260))
        st.html(f'<p class="plain-text">{plain.curve_caption(labels.name(factor, True), labels.unit(factor), curve["x"], curve["y"])}</p>')
else:
    with group_side, st.container(key="card-groups", height="stretch"):
        card_title("Importance by factor group", "Share of mean |SHAP| (XGBoost, held-out fields)",
                   help_text=TIPS["shap"])
        group_data = pd.DataFrame({"group": [labels.group(g) for g in shares.index],
                                   "share": shares.to_numpy()})
        st.altair_chart(charts.ranked_bar_chart(
            group_data, "group", "share", "Share of importance", ".0%", height=320,
            highlight={labels.group(shares.index[0])}, axis_format="%"))
    with feature_side, st.container(key="card-features", height="stretch"):
        head, toggle = st.columns([3, 2], vertical_alignment="center")
        with toggle:
            view = st.segmented_control("View", ["Ranking", "Every field"],
                                        key="drivers_view", label_visibility="collapsed")
        if view == "Every field":
            with head:
                card_title("SHAP beeswarm", "Each dot is one held-out field (XGBoost)",
                           help_text=TIPS["beeswarm"])
            with st.spinner("Computing SHAP for 960 test fields…", show_time=True):
                frame = data.tree_shap_frame()
            frame = frame.assign(
                feature=frame["feature"].map(labels.name),
                jitter=np.random.default_rng(config.RANDOM_STATE).uniform(-0.8, 0.8, len(frame)),
            )
            order = [labels.name(f) for f in shap_by_feature.index]
            st.altair_chart(charts.beeswarm_chart(frame, order, height=420))
        else:
            with head:
                card_title("Importance by feature", "Mean |SHAP| in t/ha", help_text=TIPS["shap"])
            feature_data = pd.DataFrame({"feature": [labels.name(f) for f in shap_by_feature.index],
                                         "shap": shap_by_feature.to_numpy()})
            st.altair_chart(charts.ranked_bar_chart(
                feature_data, "feature", "shap", "Mean |SHAP| (t/ha)", ".3f", height=360,
                highlight={feature_data["feature"].iloc[0]}))
