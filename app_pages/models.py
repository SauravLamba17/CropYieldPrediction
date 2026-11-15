"""Model comparison: ranked by cross-validation; the test set is shown alongside, used once."""

import streamlit as st

from src import config
from src.evaluate import leaderboard
from ui import charts, data, labels, plain
from ui.components import (
    Kpi,
    caption,
    card_title,
    headline_card,
    kpi_row,
    setup_card,
    show_table,
    tie_card,
)
from ui.plain import TIPS
from ui.theme import tip

SELECTION_NOTE = "Selected by cross-validation; test set used once for final evaluation."
simple = data.is_simple()
names = data.model_names()
summary = data.data_summary()
results, cv = data.load_table("test_results"), data.load_table("cv_all_models")
if results is None or cv is None:
    setup_card("Setup needed: the results tables are missing",
               "<code>test_results.csv</code> and <code>cv_all_models.csv</code> are written by the pipeline:",
               command="python run.py build")
    st.stop()

ranked = leaderboard(cv, results)
models_only = ranked[ranked["model"] != "dummy_mean"]
baseline = ranked.loc[ranked["model"] == "dummy_mean", "rmse"].iloc[0]
best = ranked.set_index("model").loc[names["best_overall"]]
headline_card(plain.model_headline(best["rmse"], summary["test_mean"], best["r2"], baseline))

selected = labels.model(names["best_overall"], simple)
if simple:
    kpi_row([
        Kpi("Typical error", f"{best['rmse']:.2f}", "t/ha", f"{selected}, our best model",
            accent=True, help_text=TIPS["rmse"]),
        Kpi("Explains", plain.pct(best["r2"]), "", "of the differences between fields", help_text=TIPS["r2"]),
        Kpi("Guessing the average", f"{baseline:.2f}", "t/ha", "Typical error of one guess for all",
            help_text=TIPS["baseline"]),
        Kpi("Improvement", plain.pct(1 - best["rmse"] / baseline), "", "Smaller error than guessing",
            help_text=TIPS["improvement"], help_left=True),
    ])
else:
    kpi_row([
        Kpi("Selected model: CV RMSE", f"{best['rmse_mean']:.3f}", "t/ha",
            f"{selected} · ± {best['rmse_std']:.3f} over {config.CV_FOLDS} folds", accent=True,
            help_text=TIPS["cv"]),
        Kpi("Test RMSE (used once)", f"{best['rmse']:.3f}", "t/ha",
            f"R² {best['r2']:.2f} on {summary['n_test']} held-out fields", help_text=TIPS["test"]),
        Kpi("Baseline test RMSE", f"{baseline:.3f}", "t/ha", "Always predicting the average",
            help_text=TIPS["baseline"]),
        Kpi("Improvement over baseline", f"{1 - best['rmse'] / baseline:.0%}", "",
            "Lower test RMSE than the mean baseline", help_text=TIPS["improvement"], help_left=True),
    ])

board_side, chart_side = st.columns([10, 11], gap="medium")
with board_side, st.container(key="card-board", height="stretch"):
    if simple:
        card_title("Leaderboard",
                   f"Ranked by cross-validation · typical error on {summary['n_test']} fields the models never saw",
                   help_text=TIPS["cv"])
        medals = ["🥇", "🥈", "🥉"]
        rows = []
        for i, row in enumerate(ranked.itertuples()):
            css = "first" if i == 0 else "muted" if row.model == "dummy_mean" else ""
            mark = f'<td class="medal">{medals[i]}</td>' if i < len(medals) else f'<td class="rank">{i + 1}</td>'
            rows.append(f'<tr class="{css}">{mark}<td>{labels.model(row.model, True)}'
                        f"{tip('Technical name: ' + labels.model(row.model))}</td>"
                        f'<td class="err">{row.rmse:.2f} t/ha</td></tr>')
        note = plain.tie_note(list(models_only["rmse"]))
        st.html(f'<table class="board">{"".join(rows)}</table>'
                + (f'<p class="card-caption">{note}</p>' if note else "")
                + f'<p class="card-caption">{SELECTION_NOTE}{tip(TIPS["test"], left=True)}</p>')
    else:
        card_title("Cross-validation error by model",
                   f"Mean RMSE over {config.CV_FOLDS} folds (t/ha) with ± 1 sd error bars; the lowest is selected",
                   help_text=TIPS["cv"])
        chart_data = ranked.assign(label=ranked["model"].map(labels.model))
        st.altair_chart(charts.ranked_bar_chart(
            chart_data, "label", "rmse_mean", "CV RMSE (t/ha)", ".3f", height=320,
            highlight={selected}, muted={labels.model("dummy_mean")}, error="rmse_std"))

with chart_side, st.container(key="card-fit", height="stretch"):
    card_title("Predicted vs real yield",
               f"{summary['n_test']} held-out fields · on the dashed line = a perfect prediction")
    options = {"best_overall": f"{labels.model(names['best_overall'], simple)} (deployed)",
               "best_tree": f"{labels.model(names['best_tree'], simple)} (best tree)"}
    key = st.segmented_control("Model", list(options), format_func=options.get,
                               required=True, key="board_model", label_visibility="collapsed")
    fit = data.held_out_predictions(key)
    st.altair_chart(charts.scatter_chart(fit, 270))

band = data.interval()
with st.container(key="card-residuals"):
    card_title("Residuals: how far each prediction missed",
               "Real − predicted for the held-out fields; dashed lines: the 90% range of the deployed model",
               help_text=TIPS["interval_method"])
    st.altair_chart(charts.residual_chart(fit, 240, band=band["quantile"] if band else None))
    if band:
        inside = (fit["residual"].abs() <= band["quantile"]).mean()
        caption(f"{inside:.1%} of these {len(fit)} residuals fall inside ± {band['quantile']:.2f} t/ha "
                f"({options[key]}).")

if not simple:
    with st.container(key="card-model-table"):
        card_title("Results by model",
                   f"Ranked by CV RMSE · test = {summary['n_test']} held-out fields · best value per column highlighted",
                   help_text=TIPS["metrics"])
        columns = [c for c in ["model", "rmse_mean", "rmse_std", "train_rmse", "rmse", "mae", "r2"] if c in ranked]
        show_table(ranked[columns].rename(columns={"rmse": "test_rmse", "mae": "test_mae", "r2": "test_r2"}))
        caption(SELECTION_NOTE)
    with st.container(key="card-model-tie"):
        tie = data.load_table("tie_check")
        tie = None if tie is None else tie[(tie["dataset"] == "kaggle") & (tie["model_a"] == names["best_overall"])]
        tie_card(tie)
        if tie is not None:
            caption(plain.kaggle_tie_note(
                {labels.model(m): p for m, p in zip(tie["model_b"], tie["t_test_p"])},
                {labels.model(m): bool(d) for m, d in zip(tie["model_b"], tie["distinguishable"])}))
