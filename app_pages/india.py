"""India: every model re-checked on real Indian state crop records (1997–2020)."""

import json

import numpy as np
import pandas as pd
import streamlit as st

from src import config
from ui import charts, data, labels, plain
from ui.components import Kpi, card_title, headline_card, kpi_row, setup_card, show_table
from ui.plain import TIPS
from ui.theme import tip

TABLES = {  # pill label -> (saved table, one-sentence caption, ⓘ text)
    "Test results": (
        "india_test",
        (
            "Every model trained on 1997–2015 and evaluated once on 2016–2020, ranked by "
            "time-ordered CV RMSE (the selection criterion)."
        ),
        TIPS["metrics"],
    ),
    "Look-up baseline": (
        "india_reference_baselines",
        "Simple historical-average look-ups (no machine learning) compared with the best model.",
        TIPS["lookup"],
    ),
    "Ablation": (
        "india_ablation",
        (
            "Test error with vs without the per-hectare fertilizer and pesticide inputs "
            "(Δ > 0 means the inputs helped)."
        ),
        TIPS["ablation"],
    ),
    "Time-ordered CV": (
        "india_cv",
        (
            "Cross-validation on the training years, where each validation block is later "
            "than its training years."
        ),
        TIPS["time_cv"],
    ),
    "Per-crop errors": (
        "india_per_crop_rmse",
        "Test error of the best model for the 10 most common crops.",
        TIPS["relative"],
    ),
}

simple = data.is_simple()
india_test = data.load_table("india_test")
lookups = data.load_table("india_reference_baselines")
shap_shares = data.load_table("india_shap_shares")
per_crop = data.load_table("india_per_crop_rmse")
if india_test is None or lookups is None or shap_shares is None or per_crop is None:
    setup_card(
        "Real-world validation: not built on this computer yet",
        "This page checks every model on real Indian state crop records (1997–2020). To build it: "
        '1. download <code>crop_yield.csv</code> from <a href="https://data.mendeley.com/datasets/ncw2vbcgnk" '
        'target="_blank">Mendeley Data</a>; 2. save it as <code>data/raw/india/crop_yield.csv</code>; '
        "3. run the command below (about 23 minutes). The other pages work without it.",
        command="python -m src.india",
    )
    st.stop()

india_names = json.loads((config.MODELS_DIR / "india_model_names.json").read_text())
india_best = india_test.set_index("model").loc[india_names["best_overall"]]
lookup = lookups[lookups["predictor"].str.contains("state")].iloc[0]
crop = shap_shares.set_index("feature").loc[config.INDIA_CROP, "share"]
models_india = india_test[india_test["model"] != "dummy_mean"]
headline_card(plain.india_headline(lookup["rmse"], india_best["rmse"], crop))
unit_tip = tip(TIPS["india_units"])

if simple:
    lessons = [
        plain.lesson_crop(crop),
        plain.lesson_lookup(lookup["rmse"], india_best["rmse"], lookup["rmse"] < models_india["rmse"].min()),
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
        best_label = f"Best model ({labels.model(india_names['best_overall'], True)})"
        compare = pd.DataFrame({
            "label": ["Look-up: crop's usual yield in the state", best_label, "Guessing the average"],
            "rmse": [lookup["rmse"], india_best["rmse"], guess],
        }).sort_values("rmse")
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
else:
    best_r2_model = india_test.loc[india_test["r2"].idxmax(), "model"]
    kpi_row([
        Kpi("Best model error", f"{india_best['rmse']:.2f}", "RMSE",
            f"{labels.model(india_names['best_overall'])}, test years 2016–2020", help_text=TIPS["rmse"]),
        Kpi("Crop × state look-up error", f"{lookup['rmse']:.2f}", "RMSE", "No ML: beats every model",
            accent=True, help_text=TIPS["lookup"]),
        Kpi("Crop type share of importance", f"{crop:.0%}", "", "Mean |SHAP| of the best model",
            help_text=TIPS["shap"]),
        Kpi("Best R²", f"{india_test['r2'].max():.2f}", "", labels.model(best_r2_model), help_text=TIPS["r2"],
            help_left=True),
    ])
    with st.container(key="card-india-tables"):
        choice = st.segmented_control("Table", list(TABLES), key="india_table",
                                      label_visibility="collapsed") or "Test results"
        table_name, caption_text, help_text = TABLES[choice]
        frame = data.load_table(table_name)
        st.html(f'<p class="card-caption">{caption_text} RMSE is in {plain.INDIA_UNIT}{unit_tip}.'
                f"{tip(help_text)}</p>")
        if frame is None:
            setup_card(f"{table_name}.csv is missing", "It is written by the India validation:",
                       command="python -m src.india")
        else:
            show_table(frame)
