"""Overview: the problem, the data, the three headline numbers and how the pipeline works."""

import streamlit as st

from src import config
from src.evaluate import leaderboard
from ui import data, labels, plain
from ui.components import Kpi, card_title, headline_card, kpi_row, setup_card
from ui.plain import GLOSSARY, TIPS

simple = data.is_simple()
names = data.model_names()
results, cv = data.load_table("test_results"), data.load_table("cv_all_models")
if results is None or cv is None:
    setup_card(
        "Setup needed: the results tables are missing",
        "The headline numbers are read from <code>results/tables/</code>. Rebuild them with:",
        command="python run.py build",
    )
    st.stop()

board = leaderboard(cv, results).set_index("model")
best = board.loc[names["best_overall"]]
summary = data.data_summary()
band = data.interval()
model_name = labels.model(names["best_overall"], simple)

headline_card(
    "Can we predict how much a field will yield from its soil, weather and farming? This "
    f"dashboard trains <b>{len(results)}</b> models on <b>{summary['n_train']:,}</b> fields, "
    f"picks the best by cross-validation (<b>{model_name}</b>), tests it once on "
    f"<b>{summary['n_test']}</b> fields it never saw, and re-checks every finding on real "
    "Indian crop records.",
    eyebrow="The question",
)

decimals = 2 if simple else 3
kpis = [
    Kpi(
        "Typical error when choosing (CV)" if simple else "Best CV RMSE",
        f"{best['rmse_mean']:.{decimals}f}",
        "t/ha",
        f"{model_name} · ± {best['rmse_std']:.{decimals}f} over {config.CV_FOLDS} folds",
        accent=True,
        help_text=TIPS["cv"],
    ),
    Kpi(
        "Typical error on unseen fields" if simple else "Test RMSE (used once)",
        f"{best['rmse']:.{decimals}f}",
        "t/ha",
        f"{summary['n_test']} held-out fields",
        help_text=TIPS["rmse"],
    ),
    Kpi(
        "Share of differences explained" if simple else "Test R²",
        plain.pct(best["r2"]) if simple else f"{best['r2']:.3f}",
        "",
        "1 = perfect, 0 = guessing the average",
        help_text=TIPS["r2"],
    ),
]
if band is not None:
    kpis.append(
        Kpi(
            "90% range" if simple else "90% interval half-width",
            f"± {band['quantile']:.2f}",
            "t/ha",
            f"held {band['test_coverage']:.1%} of the time on the test fields",
            help_text=TIPS["interval"] if simple else TIPS["interval_method"],
            help_left=True,
        )
    )
kpi_row(kpis)

steps = [
    ("Data", f"{summary['n_train'] + summary['n_test']:,} fields", f"{len(config.FEATURES)} inputs: soil, weather, farming, crop/region/season."),
    ("Preprocess", "inside each fold", "Scaling and one-hot encoding are fitted on training folds only (no leakage)."),
    (f"{len(results)} models", "baseline → XGBoost", "From guessing the average to boosted trees; trees tuned with nested CV."),
    ("Cross-validate", f"{config.CV_FOLDS} folds", f"Every model scored on the same folds; the lowest CV error ({model_name}) is chosen."),
    ("Test once", f"{summary['n_test']} fields", f"The kept-aside {config.TEST_SIZE:.0%} is used once, after every choice is made."),
    ("Explain", "SHAP", "Each prediction is split into the push from each factor."),
]
with st.container(key="card-pipeline"):
    card_title("How it works", "From raw records to an explained prediction")
    st.html(
        '<div class="flow">'
        + "".join(
            f'<div class="step"><span class="s-no">{i}</span><p class="s-title">{t}</p>'
            f'<p class="s-text">{text}</p><p class="s-num">{num}</p></div>'
            for i, (t, num, text) in enumerate(steps, start=1)
        )
        + "</div>"
    )

sources, glossary = st.columns([2, 3], gap="medium")
with sources, st.container(key="card-sources", height="stretch"):
    card_title("The data")
    st.html(
        '<div class="sources">'
        '<p><b>Benchmark:</b> <a href="https://www.kaggle.com/competitions/crop-yield-prediction-challenge" '
        'target="_blank">Kaggle Crop Yield Prediction Challenge</a>: '
        f"{summary['n_train'] + summary['n_test']:,} labelled fields after removing duplicates. "
        "Its patterns look synthetic, so it is treated as a controlled benchmark.</p>"
        '<p><b>Real-world check:</b> <a href="https://data.mendeley.com/datasets/ncw2vbcgnk" '
        'target="_blank">Crop Yield in Indian States, 1997–2020</a> (Mendeley Data, CC BY 4.0), '
        "built from Indian government crop statistics. Trained on 1997–2015, tested on 2016–2020.</p>"
        "</div>"
    )
with glossary, st.container(key="card-glossary", height="stretch"):
    card_title("Glossary", "The technical words used in Technical mode, in plain words")
    st.html(
        '<div class="glossary">'
        + "".join(f"<p><b>{term}</b>{text}</p>" for term, text in GLOSSARY.items())
        + "</div>"
    )
st.html(
    '<p class="honest">The main benchmark is a controlled dataset; every finding is re-checked '
    "on real Indian crop records (India page).</p>"
)
