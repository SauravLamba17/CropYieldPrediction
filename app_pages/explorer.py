"""Data explorer: the key EDA from notebook 01 as interactive charts (training split only)."""

import numpy as np
import streamlit as st

from src import config
from ui import charts, data, labels
from ui.components import Kpi, caption, card_title, headline_card, kpi_row, setup_card
from ui.theme import GROUP_SERIES

simple = data.is_simple()
train = data.split()[0]  # committed parquet: works on a fresh clone without the raw CSV
y = train[config.TARGET]
group_of = {f: g for g, cols in config.FEATURE_GROUPS.items() for f in cols}

headline_card(
    f"These are the <b>{len(train):,}</b> training fields the models learned from (the "
    f"{data.data_summary()['n_test']} test fields stay locked away, exactly as in notebook 01). "
    f"Yield averages <b>{y.mean():.2f} t/ha</b> and is almost perfectly symmetric, so it is "
    "modelled as it is, without a log transform."
)
kpi_row([
    Kpi("Training fields", f"{len(train):,}", "", "80% of the de-duplicated data"),
    Kpi("Inputs", f"{len(config.FEATURES)}", "",
        f"{len(config.NUMERIC_FEATURES)} numeric, {len(config.CATEGORICAL_FEATURES)} categories"),
    Kpi("Mean yield", f"{y.mean():.2f}", "t/ha", f"sd {y.std():.2f} · range {y.min():.2f}–{y.max():.2f}"),
    Kpi("Skewness", f"{y.skew():+.2f}".replace("-", "−"), "", "0 = symmetric; |skew| > 1 would suggest a log",
        help_left=True),
])

left, right = st.columns(2, gap="medium")
with left, st.container(key="card-target", height="stretch"):
    card_title("Yield distribution", "Training fields; dashed line = mean")
    st.altair_chart(charts.histogram(y, labels.label(config.TARGET), "green", 260))
with right, st.container(key="card-feature", height="stretch"):
    head, pick = st.columns([3, 2], vertical_alignment="center")
    with pick:
        feature = st.selectbox("Feature", config.NUMERIC_FEATURES, format_func=lambda f: labels.name(f, simple),
                               key="explore_feature", label_visibility="collapsed")
    with head:
        card_title("Input distribution", f"{labels.group(group_of[feature], simple)} · dashed line = mean")
    st.altair_chart(charts.histogram(train[feature], labels.label(feature, simple),
                                     GROUP_SERIES[group_of[feature]], 260))

with st.container(key="card-corr"):
    head, pick = st.columns([3, 1], vertical_alignment="center")
    with pick:
        method = st.segmented_control("Correlation", ["Pearson", "Spearman"], required=True,
                                      key="corr_method", label_visibility="collapsed")
    with head:
        card_title("How the numeric inputs move together",
                   "Correlation between every pair (−1 to 1), including yield; hover a cell")
    cols = config.NUMERIC_FEATURES + [config.TARGET]
    corr = train[cols].corr(method=method.lower())
    corr.index = corr.columns = [labels.name(c, simple) for c in cols]
    st.altair_chart(charts.heatmap(corr, 470))
    target = labels.name(config.TARGET, simple)
    top = corr[target].drop(target).abs().idxmax()
    inputs = corr.drop(index=target, columns=target).abs()
    inputs = inputs.where(~np.eye(len(inputs), dtype=bool)).max().max()
    caption(f"Strongest link with yield: <b>{top}</b> (r = {corr.loc[top, target]:+.2f}). The largest "
            f"correlation between two inputs is only |r| = {inputs:.2f}, one sign that the data is synthetic.")

box_cols = st.columns(len(config.CATEGORICAL_FEATURES), gap="medium")
for column, col in zip(box_cols, config.CATEGORICAL_FEATURES):
    with column, st.container(key=f"card-box-{col}", height="stretch"):
        card_title(f"Yield by {labels.name(col, simple).lower()}")
        st.altair_chart(charts.box_chart(train[[col, config.TARGET]], col, config.TARGET,
                                         labels.name(col, simple), "gold_chart", 240))

with st.container(key="card-months"):
    card_title("Yield by harvest month",
               "Why the harvest date was dropped as an input: yield is flat across months")
    months = data.raw_harvest_months()
    if months is None:
        setup_card(
            "This chart needs the raw Kaggle file",
            "The harvest date is only in the raw competition CSV, which is not stored in git. "
            "Accept the competition rules on Kaggle, add your API token, then download it with:",
            command="python run.py",
        )
    else:
        st.altair_chart(charts.month_chart(months, 220))
