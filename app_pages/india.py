"""India: every model re-checked on real Indian state crop records (1997–2020)."""

import json

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.india_features import LAG_COLUMNS
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

LOOKUP, PERSISTENCE = "crop x state lookup (no ML)", "last year's yield (no ML)"
TABLES = {  # pill label -> (saved table, one-sentence caption, ⓘ text)
    "Leaderboard": (
        "india_leaderboard",
        "Every model and the no-ML references, ranked by time-ordered CV RMSE; test years used once.",
        TIPS["time_cv"],
    ),
    "Feature ablation": (
        "india_feature_ablation",
        "The best model family with each feature group added on its own, all of them, and the hybrid.",
        TIPS["feature_ablation"],
    ),
    "Test results": (
        "india_test",
        (
            "The original suite trained on 1997–2015 and evaluated once on 2016–2020, ranked by "
            "time-ordered CV RMSE (the selection criterion)."
        ),
        TIPS["metrics"],
    ),
    "Look-up baseline": (
        "india_reference_baselines",
        "Simple predictors without machine learning, compared with the selected model.",
        TIPS["lookup"],
    ),
    "Errors by crop and state": (
        "india_error_by_group",
        "Test RMSE of the selected model, the look-up and last year's yield for every crop and state.",
        TIPS["relative"],
    ),
    "Error segments": (
        "india_error_segments",
        "Test RMSE on rows with and without history, and on crops or pairs never seen in training.",
        TIPS["segments"],
    ),
    "Year-index ablation": (
        "india_ablation",
        (
            "Test error of the original suite with vs without the per-hectare fertilizer and "
            "pesticide inputs (Δ > 0 means the inputs helped)."
        ),
        TIPS["ablation"],
    ),
    "Time-ordered CV": (
        "india_cv",
        (
            "Cross-validation of the original suite on the training years, where each validation "
            "block is later than its training years."
        ),
        TIPS["time_cv"],
    ),
    "Per-crop errors": (
        "india_per_crop_rmse",
        "Test error of the selected model for the 10 most common crops.",
        TIPS["relative"],
    ),
}
SHOWN_COLUMNS = {  # wide Phase 4 tables: the columns worth a table view
    "india_leaderboard": ["model", "feature_set", "rmse_mean", "rmse_std", "test_rmse", "test_rmse_strict", "test_r2"],
    "india_feature_ablation": ["model", "feature_set", "rmse_mean", "rmse_std", "test_rmse", "test_rmse_strict",
                               "test_r2"],
}
PHASE4_TABLES = ["india_leaderboard", "india_feature_ablation", "india_error_by_group",
                 "india_error_segments", "india_explain_permutation", "india_explain_shap"]

simple = data.is_simple()
india_test = data.load_table("india_test")
lookups = data.load_table("india_reference_baselines")
shap_shares = data.load_table("india_shap_shares")
per_crop = data.load_table("india_per_crop_rmse")
names_file = config.MODELS_DIR / "india_model_names.json"
if any(t is None for t in (india_test, lookups, shap_shares, per_crop)) or not names_file.exists():
    setup_card(
        "Real-world validation: not built on this computer yet",
        "This page checks every model on real Indian state crop records (1997–2020). To build it: "
        '1. download <code>crop_yield.csv</code> from <a href="https://data.mendeley.com/datasets/ncw2vbcgnk" '
        'target="_blank">Mendeley Data</a>; 2. save it as <code>data/raw/india/crop_yield.csv</code>; '
        "3. run the command below (about 105 minutes). The other pages work without it.",
        command="python -m src.india",
    )
    st.stop()

india_names = json.loads(names_file.read_text())
selected = india_names["best_overall"]
lookup = lookups[lookups["predictor"].str.contains("state")].iloc[0]
crop = shap_shares.set_index("feature").loc[config.INDIA_CROP, "share"]
history = shap_shares.set_index("feature")["share"].reindex(LAG_COLUMNS).sum()  # 0 before Phase 4
unit_tip = tip(TIPS["india_units"])
tables = {name: data.load_table(name) for name in PHASE4_TABLES}
selection = data.load_json("india_selection")
tie = data.load_table("tie_check")
extended = selection is not None and all(t is not None for t in tables.values())

if not extended:  # tables from before Phase 4: the original page
    by_model = india_test.set_index("model")["rmse"]
    best_rmse = by_model.get(selected, by_model.min())  # names may already point at a Phase 4 model
    headline_card(plain.india_headline(lookup["rmse"], best_rmse, crop))
    setup_card(
        "The boosted models and the feature ablation are not built yet",
        "The time-safe features (target encoding, lags, trend), LightGBM, HistGradientBoosting, the "
        "hybrid and the ablation are computed by the India build:",
        command="python -m src.india",
    )
else:
    selected = selection["selected"]
    board = tables["india_leaderboard"].set_index("model")
    ablation = tables["india_feature_ablation"]
    best_rmse = board.loc[selected, "test_rmse"]
    singles = ablation.set_index("feature_set")["rmse_mean"]
    gains = {s: singles["base"] - singles[s] for s in ("target_encoding", "lags", "trend") if s in singles}
    top_group = max(gains, key=gains.get)
    verdict_inputs = {
        "model": labels.model(selected), "model_cv": board.loc[selected, "rmse_mean"],
        "lookup_cv": board.loc[LOOKUP, "rmse_mean"], "model_test": best_rmse,
        "lookup_test": board.loc[LOOKUP, "test_rmse"], "persistence_test": board.loc[PERSISTENCE, "test_rmse"],
        "strict_test": board.loc[selected, "test_rmse_strict"], "beats_cv": selection["beats_lookup_cv"],
        "group": labels.lower(labels.feature_set_name(top_group, simple).removeprefix("+ ")),
        "gain": gains[top_group],
    }
    if tie is not None:  # paired CV tie check: (p, distinguishable) against each reference
        pairs = tie[(tie["dataset"] == "india") & (tie["model_a"] == selected)].set_index("model_b")
        for key, ref in (("tie_lookup", LOOKUP), ("tie_persistence", PERSISTENCE)):
            if ref in pairs.index:
                verdict_inputs[key] = (pairs.loc[ref, "t_test_p"], bool(pairs.loc[ref, "distinguishable"]))
    title, verdict = plain.india_verdict(verdict_inputs, simple)
    headline_card(plain.india_phase4_headline(verdict_inputs, simple))

if simple:
    lessons = [
        plain.lesson_crop(crop, history),
        plain.lesson_lookup(lookup["rmse"], best_rmse, lookup["rmse"] < best_rmse),
        plain.lesson_seasons(int((per_crop["r2"] < 0).sum()), len(per_crop)),
    ]
    st.html(
        '<div class="lessons">'
        + "".join(f'<div class="lesson"><p class="l-num"><span>{i}</span>{big}</p><p class="l-text">{text}</p></div>'
                  for i, (big, text) in enumerate(lessons, start=1))
        + "</div>"
    )
    compare_side, crop_side = st.columns(2, gap="medium")
    with compare_side, st.container(key="card-india-compare", height="stretch"):
        card_title("Typical error on the 2016–2020 test years",
                   f"{plain.INDIA_UNIT} (varies by crop){unit_tip} · lower is better", help_text=TIPS["rmse"])
        guess = india_test.set_index("model").loc["dummy_mean", "rmse"]
        best_label = "Chosen model"  # short: the long model names were cut off at every width
        rows = {"Look-up: crop's usual yield in the state": lookup["rmse"], best_label: best_rmse,
                "Guessing the average": guess}
        if extended:
            rows["Repeat last year's yield"] = board.loc[PERSISTENCE, "test_rmse"]
        compare = pd.DataFrame({"label": list(rows), "rmse": list(rows.values())}).sort_values("rmse")
        st.altair_chart(charts.ranked_bar_chart(
            compare, "label", "rmse", f"Typical error ({plain.INDIA_UNIT})", ".2f", height=240,
            highlight={compare["label"].iloc[0]}, muted={"Guessing the average"}))
    with crop_side, st.container(key="card-india-crops", height="stretch"):
        card_title(f"How well it works for the {len(per_crop)} most common crops",
                   "Typical error as a share of each crop's average yield", help_text=TIPS["relative"])
        crops = per_crop.sort_values("relative_rmse")
        verdicts = pd.DataFrame({
            "crop": crops["crop"].map(labels.level),
            "relative": crops["relative_rmse"],
            "verdict": np.where(crops["relative_rmse"] < plain.WORKS_WELL, "works well", "struggles"),
        })
        st.altair_chart(charts.verdict_bar_chart(verdicts, height=260))
    if extended:
        with st.container(key="card-india-verdict"):
            card_title(title, help_text=TIPS["lookup"])
            st.html(f'<p class="plain-text">{verdict}</p>')
    st.stop()

# --- Technical mode ---
if not extended:
    best_r2_model = india_test.loc[india_test["r2"].idxmax(), "model"]
    kpi_row([
        Kpi("Best model error", f"{best_rmse:.2f}", "RMSE",
            f"{labels.model(selected)}, test years 2016–2020", help_text=TIPS["rmse"]),
        Kpi("Crop × state look-up error", f"{lookup['rmse']:.2f}", "RMSE", "No ML", accent=True,
            help_text=TIPS["lookup"]),
        Kpi("Crop type share of importance", f"{crop:.0%}", "", "Mean |SHAP| of the best model",
            help_text=TIPS["shap"]),
        Kpi("Best R²", f"{india_test['r2'].max():.2f}", "", labels.model(best_r2_model), help_text=TIPS["r2"],
            help_left=True),
    ])
else:
    kpi_row([
        Kpi("Selected model, CV", f"{board.loc[selected, 'rmse_mean']:.2f}", "RMSE",
            f"± {board.loc[selected, 'rmse_std']:.2f} · {labels.model(selected)}", help_text=TIPS["time_cv"]),
        Kpi("Look-up, CV", f"{board.loc[LOOKUP, 'rmse_mean']:.2f}", "RMSE",
            f"± {board.loc[LOOKUP, 'rmse_std']:.2f} · same 5 year blocks", accent=True, help_text=TIPS["lookup"]),
        Kpi("Selected model, test", f"{best_rmse:.2f}", "RMSE", f"R² {board.loc[selected, 'test_r2']:.2f} · 2016–2020",
            help_text=TIPS["rmse"]),
        Kpi("Look-up, test", f"{board.loc[LOOKUP, 'test_rmse']:.2f}", "RMSE",
            f"last year's yield: {board.loc[PERSISTENCE, 'test_rmse']:.2f}", help_text=TIPS["persistence"],
            help_left=True),
    ])
    with st.container(key="card-india-verdict"):
        card_title(title, help_text=TIPS["lookup"])
        st.html(f'<p class="plain-text">{verdict}</p>')

    left, right = st.columns(2, gap="medium")
    with left, st.container(key="card-india-ablation", height="stretch"):
        family = labels.model(ablation.dropna(subset=["family"]).query("family != ''")["family"].iloc[0])
        card_title("Which feature group closes the gap?",
                   f"{family}: time-ordered CV RMSE per feature set; dashed line = the look-up",
                   help_text=TIPS["feature_ablation"])
        bars = ablation[ablation["kind"] == "model"]
        frame = pd.DataFrame({"label": bars["feature_set"].map(lambda s: labels.feature_set_name(s, short=True)),
                              "value": bars["rmse_mean"], "sd": bars["rmse_std"]})
        st.altair_chart(charts.ablation_chart(frame, board.loc[LOOKUP, "rmse_mean"], "Look-up",
                                              f"CV RMSE ({plain.INDIA_UNIT})", 300))
        caption(f"Largest single gain: <b>{labels.feature_set_name(top_group)}</b> "
                f"(CV RMSE − {gains[top_group]:.2f} vs base inputs). sd across the 5 year blocks is about "
                f"{bars['rmse_std'].median():.1f}: a few extreme records in some years dominate.")
    with right, st.container(key="card-india-errors", height="stretch"):
        by_crop = tables["india_error_by_group"].query("group_by == @config.INDIA_CROP").head(10)
        card_title("Where the errors are: the 10 worst crops",
                   f"Test RMSE per crop ({plain.INDIA_UNIT}), selected model vs the look-up",
                   help_text=TIPS["relative"])
        pairs = {"rmse_model": ("Selected model", "green"), "rmse_lookup": ("Look-up", "gold_chart")}
        st.altair_chart(charts.paired_bar_chart(by_crop.assign(group=by_crop["group"].map(labels.level)),
                                                "group", pairs, f"Test RMSE ({plain.INDIA_UNIT})", 420))
        groups = tables["india_error_by_group"]
        wins = groups.groupby("group_by")["model_beats_lookup"].agg(["sum", "count"])
        caption(f"The model beats the look-up for <b>{int(wins.loc[config.INDIA_CROP, 'sum'])} of "
                f"{int(wins.loc[config.INDIA_CROP, 'count'])}</b> crops and <b>{int(wins.loc[config.INDIA_STATE, 'sum'])} "
                f"of {int(wins.loc[config.INDIA_STATE, 'count'])}</b> states.")

    with st.container(key="card-india-leaderboard"):
        card_title("Leaderboard: ranked by time-ordered CV",
                   "Test RMSE assumes last year's yields are known; the strict column knows only 1997–2015",
                   help_text=TIPS["time_cv"])
        shown = tables["india_leaderboard"][["model", "rmse_mean", "rmse_std", "test_rmse", "test_rmse_strict",
                                             "test_r2", "selected"]].copy()
        shown["model"] = [f"{labels.model(m)}{' ✓ selected' if s else ''}" for m, s in
                          zip(shown["model"], shown["selected"])]
        show_table(shown.drop(columns="selected"))

    with st.container(key="card-india-tie"):
        tie_card(None if tie is None else tie[tie["dataset"] == "india"])

    perm, shap_table = tables["india_explain_permutation"], tables["india_explain_shap"]
    with st.container(key="card-india-explain"):
        card_title(f"What the selected model relies on: {labels.model(selected)}",
                   "Permutation: increase in test RMSE when one input is shuffled (30 repeats, ± 1 sd); "
                   "mean |SHAP| share on the test years", help_text=TIPS["permutation"])
        left, right = st.columns(2, gap="medium")
        with left:
            frame = perm.assign(label=perm["feature"].map(labels.name))
            st.altair_chart(charts.ranked_bar_chart(frame, "label", "importance_mean",
                                                    "Rise in test RMSE", ".2f", height=340,
                                                    highlight={frame["label"].iloc[0]}, error="importance_std"))
        with right:
            frame = shap_table.assign(label=shap_table["feature"].map(labels.name))
            st.altair_chart(charts.ranked_bar_chart(frame, "label", "share", "Share of mean |SHAP|", ".0%",
                                                    height=340, highlight={frame["label"].iloc[0]}, axis_format="%"))
        rho = selection["permutation_vs_shap_spearman"]
        same_top = perm["feature"].iloc[0] == shap_table["feature"].iloc[0]
        caption(
            f"The two methods {'agree on' if same_top else 'disagree on'} the top input "
            f"(permutation: <b>{labels.lower(labels.name(perm['feature'].iloc[0]))}</b>, SHAP: "
            f"<b>{labels.lower(labels.name(shap_table['feature'].iloc[0]))}</b>), with a Spearman rank "
            f"correlation of <b>{rho:.2f}</b>. Shuffling crop also shuffles its target encodings; SHAP credits "
            "the encodings to crop and state"
            + (" and explains the residual model only (the look-up offset is not a tree)."
               if "hybrid" in selected else ".")
        )

with st.container(key="card-india-tables"):
    options = [k for k, (name, _, _) in TABLES.items() if extended or name not in PHASE4_TABLES]
    choice = st.segmented_control("Table", options, key="india_table", label_visibility="collapsed")
    choice = choice if choice in options else options[0]
    table_name, caption_text, help_text = TABLES[choice]
    frame = data.load_table(table_name)
    st.html(f'<p class="card-caption">{caption_text} RMSE is in {plain.INDIA_UNIT}{unit_tip}.'
            f"{tip(help_text)}</p>")
    if frame is None:
        setup_card(f"{table_name}.csv is missing", "It is written by the India validation:",
                   command="python -m src.india")
    else:
        frame = frame[SHOWN_COLUMNS[table_name]] if table_name in SHOWN_COLUMNS else frame
        if "model" in frame:
            frame = frame.assign(model=frame["model"].map(labels.model))
        if "feature_set" in frame:
            frame = frame.assign(feature_set=frame["feature_set"].map(
                lambda s: labels.feature_set_name(s) if s in labels.FEATURE_SETS else s))
        show_table(frame)
