"""What matters: global importance, do the methods agree, coefficients, PDP/ICE, soil pH.

Simple mode: what the model relies on and the soil pH surprise, each chart with a one-line
takeaway. Show details adds the method comparison, the pH scatter, coefficients and PDP/ICE.
"""

import streamlit as st

from src import config
from ui import analysis, charts, data, labels, plain
from ui.components import (
    Kpi,
    caption,
    card_title,
    chart_headline,
    headline_card,
    kpi_row,
    page_title,
    setup_card,
)
from ui.plain import TIPS

METHODS = {"permutation": "Permutation", "shap": "Mean |SHAP|", "coefficient": "|Coefficient|"}
simple = data.is_simple()
if simple:
    page_title("What matters", "Which factors the model leans on most, and one surprise")
else:
    page_title(
        "Why the model decides",
        "Global importance, whether the methods agree, and how each input moves the prediction",
    )
names = data.model_names()
perm, shap_table = data.load_table("explain_permutation"), data.load_table("explain_shap")
coefs, signal = data.load_table("explain_coefficients"), data.load_table("feature_signal")
if perm is None or shap_table is None or coefs is None:
    setup_card(
        "Setup needed: the explainability tables are missing",
        "Permutation importance, mean |SHAP| and the bootstrap coefficients are computed once by "
        "the pipeline (5 to 17 minutes measured on a laptop) and saved to <code>results/tables/</code>. Build them with:",
        command="python -m src.explain_tables",
    )
    st.stop()

options = {"best_overall": f"{labels.model(names['best_overall'], simple)} (deployed)",
           "best_tree": f"{labels.model(names['best_tree'], simple)} (best tree)"}
key = st.session_state.get("explain_model", "best_overall")
model = names[key]
scores = analysis.method_scores(perm, shap_table, coefs, model, "multiple_lr")
ranks = analysis.method_ranks(scores)
agree = analysis.method_agreement(scores)
first = [f for f in config.FEATURES if (ranks.loc[f] == 1).all()]
spread = (ranks.max(axis=1) - ranks.min(axis=1)).sort_values(ascending=False)
mean_rho = agree["rho"].mean()
top_perm = scores["permutation"].idxmax()
if simple:
    top_name = labels.name(first[0], True) if first else None
    summary = plain.methods_headline_simple(top_name, mean_rho, labels.model(model, True))
elif first:
    summary = (f"All three methods rank <b>{labels.lower(labels.name(first[0], simple))}</b> first for the "
               f"{labels.model(model, simple)} model")
else:
    summary = (f"The methods disagree on the top factor: permutation ranks "
               f"<b>{labels.lower(labels.name(top_perm, simple))}</b> first")
if not simple:
    summary += (f", and their rankings agree with an average Spearman ρ of <b>{mean_rho:.2f}</b> "
                f"(1 = identical order). They disagree most on <b>{labels.lower(labels.name(spread.index[0], simple))}</b> "
                f"(ranks {ranks.loc[spread.index[0]].min()}–{ranks.loc[spread.index[0]].max()}).")
headline_card(summary)

with st.container(key="card-importance"):
    card_title("What the model relies on",
               "How much the typical error grows when one factor is scrambled" if simple
               else "Increase in test error when one input is shuffled (30 repeats, ± 1 sd)",
               help_text=plain.tip_text("permutation", simple))
    st.segmented_control("Model", list(options), format_func=options.get, required=True,
                         key="explain_model", label_visibility="collapsed")
    table = perm.query("model == @model").assign(label=lambda d: d["feature"].map(lambda f: labels.name(f, simple)))
    if simple:
        chart_headline(plain.importance_headline(list(zip(table["label"], table["importance_mean"]))))
    st.altair_chart(charts.ranked_bar_chart(
        table, "label", "importance_mean",
        "How much the typical error grows (t/ha)" if simple else "Increase in test RMSE (t/ha)", ".3f", height=380,
        highlight={table["label"].iloc[0]}, error="importance_std"))

if not simple:
    with st.container(key="card-agreement"):
        card_title("Do the methods agree?",
                   "Rank of every input under three methods (1 = most important); a straight column = agreement",
                   help_text=TIPS["agreement"])
        long = ranks.reset_index(names="feature").melt(id_vars="feature", var_name="method", value_name="rank")
        long = long.assign(feature=long["feature"].map(lambda f: labels.name(f, simple)),
                           method=long["method"].map(METHODS))
        order = [labels.name(f, simple) for f in scores["permutation"].sort_values(ascending=False).index]
        st.altair_chart(charts.rank_dot_chart(long, order, list(METHODS.values()), height=400))
        kpi_row([Kpi(f"{METHODS[r.a]} vs {METHODS[r.b]}", f"{r.rho:.2f}", "ρ",
                     "Spearman rank correlation", help_text=TIPS["spearman"]) for r in agree.itertuples()])
        caption(f"Permutation and mean |SHAP| are for the {labels.model(model, simple)} model; |coefficient| is the "
                "largest standardised coefficient of each input in the multiple linear model (a category's strongest level).")

st.html('<a id="soil-ph-paradox"></a>')
with st.container(key="card-ph"):
    if simple:
        card_title("The soil pH surprise", "A straight-line check misses it; the model can follow a curve",
                   help_text=TIPS["ph"])
    else:
        card_title("The soil pH paradox: why correlation misses it",
                   "Correlation looks for straight lines; the model can follow a curve", help_text=TIPS["ph"])
    train = data.split()[0]
    r, rho = analysis.correlations(train["soil_ph"], train[config.TARGET])
    curve = data.response_curve("soil_ph")
    opt = analysis.optimum_region(curve)
    lasso_ph = coefs.query("model == 'lasso' and feature == 'soil_ph'")["coef"]
    kept = (not lasso_ph.empty) and abs(float(lasso_ph.iloc[0])) > 0
    if simple:  # the model's curve only; the scatter and the statistics are with the details
        chart_headline(plain.curve_headline("Soil pH", "pH", curve["x"], curve["y"]))
        st.altair_chart(charts.curve_chart(curve, None, labels.label("soil_ph"), 260,
                                           region=(opt["low"], opt["high"])))
        st.html(f'<p class="plain-text">{plain.ph_sentence_simple(r, opt)}</p>')
        st.stop()
    left, right = st.columns(2, gap="medium")
    with left:
        st.altair_chart(charts.scatter_xy(train, "soil_ph", config.TARGET, labels.label("soil_ph"),
                                          labels.label(config.TARGET), 260))
        caption(f"Pearson r = <b>{r:+.3f}</b> · Spearman ρ = <b>{rho:+.3f}</b> (3,840 training fields)")
    with right:
        st.altair_chart(charts.curve_chart(curve, None, labels.label("soil_ph"), 260,
                                           region=(opt["low"], opt["high"])))
        caption(f"Model response: peak at pH <b>{opt['peak_x']:.1f}</b>; within 10% of the peak from "
                f"pH <b>{opt['low']:.1f}</b> to <b>{opt['high']:.1f}</b> (shaded).")
    st.html(f'<p class="plain-text">{plain.ph_sentence(r, rho, opt, kept)}</p>')
    if not simple and signal is not None:
        p_quad = float(signal.set_index("feature").loc["soil_ph", "p_quadratic"])
        caption(f"Quadratic-term test on the training data (notebook 01): p = {p_quad:.1e}. "
                f"Lasso coefficient for soil pH: {float(lasso_ph.iloc[0]):+.4f} t/ha per sd "
                f"({'kept' if kept else 'set to zero'}).")

if not simple:
    with st.container(key="card-coefficients"):
        head, pick = st.columns([3, 2], vertical_alignment="center")
        with head:
            card_title("Standardised coefficients with 95% bootstrap intervals",
                       "t/ha per 1 sd of a numeric input; a category level vs its baseline. "
                       "Grey = the interval crosses 0: not distinguishable from zero.",
                       help_text=TIPS["bootstrap"])
        with pick:
            linear = st.segmented_control("Linear model", ["multiple_lr", "ridge", "lasso"],
                                          format_func=labels.model, required=True, key="coef_model",
                                          label_visibility="collapsed")
        table = coefs.query("model == @linear").copy()
        table["label"] = [labels.term(t, f) for t, f in zip(table["term"], table["feature"])]
        table = table.reindex(table["coef"].abs().sort_values(ascending=False).index)
        st.altair_chart(charts.coefficient_chart(table, height=max(300, 28 * len(table))))
        n_zero = int(table["crosses_zero"].sum())
        caption(f"{n_zero} of {len(table)} terms are not distinguishable from zero "
                f"({int(table['n_boot'].iloc[0])} bootstrap refits of the training set, fixed seed).")

    with st.container(key="card-pdp"):
        head, pick = st.columns([3, 2], vertical_alignment="center")
        with head:
            card_title("Partial dependence with ICE lines",
                       f"Thick line: average prediction over 300 training fields; thin lines: 50 single fields "
                       f"({labels.model(names['best_overall'])}, deployed)", help_text=TIPS["ice"])
        with pick:
            feature = st.selectbox("Feature", config.NUMERIC_FEATURES, format_func=labels.name,
                                   key="pdp_feature", label_visibility="collapsed")
        with st.spinner("Sweeping 50 fields across the range…", show_time=True):
            ice_lines = data.ice(feature)
            pdp = data.response_curve(feature)
        st.altair_chart(charts.pdp_ice_chart(pdp, ice_lines, labels.label(feature), 320))

    with st.container(key="card-notes"):
        card_title("How the three methods differ")
        st.html(
            '<div class="glossary">'
            "<p><b>Permutation importance</b>How much the test error grows when one input is shuffled. "
            "Model-agnostic and measured on unseen fields; inputs that move together share the credit.</p>"
            "<p><b>Mean |SHAP|</b>The average size of each input's push on individual predictions. Captures "
            "curves and interactions; Kernel SHAP for the deployed model, TreeExplainer for XGBoost.</p>"
            "<p><b>Standardised coefficients</b>The straight-line slope per 1 sd in a linear model. A curved "
            "effect such as soil pH can have a slope near 0 even when it matters.</p>"
            "</div>"
        )
